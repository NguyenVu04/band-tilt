"""Draw and write per-interval UE positions from a density field."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.simulation import scene
from src.simulation.density import DensityField
from src.simulation.grid import GridSpec, Raster
from src.simulation.scene import SceneBounds
from src.simulation.traffic import Schedule

# Redraw the cell and position after rejection; keep the assigned component.
_MAX_REDRAW_ROUNDS = 500

CSV_COLUMNS = ("t_index", "t_s", "x", "y", "z", "cell_col", "cell_row", "component")


@dataclass(frozen=True)
class UeSpec:
    """How many UEs to draw per interval, and at what height.

    Attributes:
        count_range: Inclusive range the per-interval count is drawn uniformly
            from. A range rather than a fixed count so load varies snapshot to
            snapshot as it does in a real network.
        height_m: The plane the UEs sit on. Must match the height any radio map
            weighted by them is solved at.
    """

    count_range: tuple[int, int]
    height_m: float

    def __post_init__(self) -> None:
        """Reject a population nothing could be drawn for.

        Raises:
            ValueError: When the range is not positive or is inverted.
        """
        low, high = self.count_range
        if low <= 0:
            raise ValueError(f"simulation.ue.count_range must be positive, got {self.count_range}")
        if high < low:
            raise ValueError(f"simulation.ue.count_range is inverted: {self.count_range}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> UeSpec:
        """Read ``simulation.ue``."""
        low, high = (int(value) for value in cfg.simulation.ue.count_range)
        return cls(count_range=(low, high), height_m=float(cfg.simulation.ue.height_m))


def sample_positions(
    mi_scene: Any,
    bounds: SceneBounds,
    roi: SceneBounds,
    raster: Raster,
    field: DensityField,
    schedule: Schedule,
    grid_spec: GridSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw every interval's UEs, rejecting any inside a building or the margin.

    Assigns each UE a mixture component once, from the mass its own interval
    carries, then repeatedly draws a cell from that component and a uniform
    position inside it, keeping the positions that land on open ground inside
    the region of interest. Returns ``(interval, x, y, component)``, where
    component is ``-1`` for the uniform background and the hotspot index
    otherwise.

    The rejection is not merely a filter: it is what weights a cell by its open
    area, so the resulting density is the density function times the cell's
    true open area rather than times a sub-sampled estimate of it. The region
    test rides along for the same reason — a cell whose centre is inside the
    boundary can still reach past it, and rejecting the overhang is what makes
    the region an exact edge rather than a half-cell approximation.

    Raises:
        RuntimeError: When draws keep landing on buildings, which means the
            open-ground raster no longer describes this scene.
    """
    rng = np.random.default_rng(seed)

    interval = np.repeat(schedule.t_index, schedule.count)
    n_ue = interval.size
    component = np.empty(n_ue, dtype=np.int64)
    start = 0
    for index in range(schedule.n_intervals):
        drawn = int(schedule.count[index])
        component[start : start + drawn] = rng.choice(
            field.n_components, size=drawn, p=schedule.component_mass[index]
        )
        start += drawn

    x = np.empty(n_ue, dtype=np.float64)
    y = np.empty(n_ue, dtype=np.float64)
    pending = np.arange(n_ue)

    for _ in range(_MAX_REDRAW_ROUNDS):
        cell = np.empty(pending.size, dtype=np.int64)
        for index in range(field.n_components):
            drawn = component[pending] == index
            n_drawn = int(np.count_nonzero(drawn))
            if n_drawn:
                cell[drawn] = rng.choice(
                    field.cell_weights.shape[1], size=n_drawn, p=field.cell_weights[index]
                )

        rows, cols = np.divmod(cell, raster.n_cols)
        candidate_x = raster.origin_x + (cols + rng.random(pending.size)) * raster.cell_size_m
        candidate_y = raster.origin_y + (rows + rng.random(pending.size)) * raster.cell_size_m
        height = scene.surface_height(mi_scene, candidate_x, candidate_y, bounds.launch_z)
        accepted = (
            np.isfinite(height)
            & (height <= grid_spec.free_height_tol_m)
            & (candidate_x >= roi.min_x)
            & (candidate_x <= roi.max_x)
            & (candidate_y >= roi.min_y)
            & (candidate_y <= roi.max_y)
        )

        x[pending[accepted]] = candidate_x[accepted]
        y[pending[accepted]] = candidate_y[accepted]
        pending = pending[~accepted]
        if pending.size == 0:
            break
    else:
        raise RuntimeError(
            f"{pending.size} UE positions still land on a building or outside the region "
            f"of interest after {_MAX_REDRAW_ROUNDS} redraws. The open-ground raster and "
            "the scene disagree."
        )

    return interval, x, y, component - 1


def densest_decile_share(
    cell_col: np.ndarray,
    cell_row: np.ndarray,
    raster: Raster,
    eligible: np.ndarray,
) -> float:
    """Share of UEs falling in the densest tenth of the eligible cells.

    One scalar describing how concentrated the population is, for the run log.
    A uniform density gives roughly a tenth; anything well above that is the
    hotspots doing work. ``eligible`` is the mask the field was built over
    (:func:`src.simulation.density.eligible_cells`), so the denominator counts
    the cells a UE could actually have landed in.
    """
    counts = np.zeros(raster.n_rows * raster.n_cols, dtype=np.int64)
    np.add.at(counts, cell_row * raster.n_cols + cell_col, 1)

    n_eligible = int(np.count_nonzero(eligible))
    if n_eligible == 0 or counts.sum() == 0:
        return float("nan")
    top = max(1, round(0.1 * n_eligible))
    return float(np.sort(counts)[::-1][:top].sum() / counts.sum())


def write_csv(
    path: Path,
    interval: np.ndarray,
    schedule: Schedule,
    x: np.ndarray,
    y: np.ndarray,
    component: np.ndarray,
    raster: Raster,
    ue: UeSpec,
) -> Path:
    """Write the UE population, one row per UE per interval. Returns ``path``.

    There is no UE identifier. Each interval is an independent draw, so a row
    in one snapshot has no counterpart in the next, and an id would invite
    downstream code to join on something that does not mean what it looks like.

    Coordinates are written at millimetre precision so a rerun at the same seed
    is byte-identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_col, cell_row = raster.cell_indices(x, y)
    t_s = schedule.t_s[interval]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for index in range(x.size):
            writer.writerow(
                (
                    int(interval[index]),
                    f"{t_s[index]:.3f}",
                    f"{x[index]:.3f}",
                    f"{y[index]:.3f}",
                    f"{ue.height_m:.3f}",
                    int(cell_col[index]),
                    int(cell_row[index]),
                    int(component[index]),
                )
            )
    return path
