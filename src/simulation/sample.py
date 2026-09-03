"""Draw the UE population from the density field and write it out.

A library for :mod:`src.simulation.scenario`, which orchestrates the stage.

UEs are weights on a radio map, not ray-tracing targets — no propagation is
solved here, and none is solved per UE later either. The output carries the
grid cell each UE falls in so the per-cell counts the UE-weighted KPIs need are
a group-by rather than a re-derivation against the grid.
"""

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

# Rejection is what puts a cell's open area into the sampled density (see
# src.simulation.density.field), so a rejected draw redraws the CELL as well as
# the position. Redrawing only the position would sample from the cell's
# estimated open area instead of its true one, and that estimate overstates a
# cell holding a sliver of open ground by more than an order of magnitude.
#
# The mixture component is held fixed across redraws. Redrawing it too would
# let components sitting over dense ground reject more often and so land below
# their configured share, quietly breaking density.hotspot_mass_fraction.
#
# Every component has support on cells with open ground, so acceptance is
# bounded away from zero and this cap is only reached by a scene the raster no
# longer describes.
_MAX_REDRAW_ROUNDS = 500

CSV_COLUMNS = ("ue_id", "x", "y", "z", "cell_col", "cell_row", "component")


@dataclass(frozen=True)
class UeSpec:
    """How many UEs to draw, and at what height.

    Attributes:
        count: Number of UEs.
        height_m: The plane the UEs sit on. Must match the height any radio map
            weighted by them is solved at.
    """

    count: int
    height_m: float

    def __post_init__(self) -> None:
        """Reject a population nothing could be drawn for.

        Raises:
            ValueError: When ``count`` is not positive.
        """
        if self.count <= 0:
            raise ValueError(f"simulation.ue.count must be positive, got {self.count}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> UeSpec:
        """Read ``simulation.ue``."""
        return cls(
            count=int(cfg.simulation.ue.count),
            height_m=float(cfg.simulation.ue.height_m),
        )


def sample_positions(
    mi_scene: Any,
    bounds: SceneBounds,
    raster: Raster,
    field: DensityField,
    ue: UeSpec,
    grid_spec: GridSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw UE positions from the density field, rejecting any inside a building.

    Assigns each UE a mixture component once, then repeatedly draws a cell from
    that component and a uniform position inside it, keeping the positions that
    land on open ground. Returns ``(x, y, component)``, where component is
    ``-1`` for the uniform background and the hotspot index otherwise.

    The rejection is not merely a filter: it is what weights a cell by its open
    area, so the resulting density is the density function times the cell's
    true open ground rather than times a sub-sampled estimate of it.

    Raises:
        RuntimeError: When draws keep landing on buildings, which means the
            open-ground raster no longer describes this scene.
    """
    rng = np.random.default_rng(seed)

    component = rng.choice(field.component_mass.size, size=ue.count, p=field.component_mass)
    x = np.empty(ue.count, dtype=np.float64)
    y = np.empty(ue.count, dtype=np.float64)
    pending = np.arange(ue.count)

    for _ in range(_MAX_REDRAW_ROUNDS):
        cell = np.empty(pending.size, dtype=np.int64)
        for index in range(field.component_mass.size):
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
        free = np.isfinite(height) & (height <= grid_spec.free_height_tol_m)

        x[pending[free]] = candidate_x[free]
        y[pending[free]] = candidate_y[free]
        pending = pending[~free]
        if pending.size == 0:
            break
    else:
        raise RuntimeError(
            f"{pending.size} UE positions still land on a building after "
            f"{_MAX_REDRAW_ROUNDS} redraws. The open-ground raster and the scene disagree."
        )

    return x, y, component - 1


def densest_decile_share(cell_col: np.ndarray, cell_row: np.ndarray, raster: Raster) -> float:
    """Share of UEs falling in the densest tenth of the cells holding open ground.

    One scalar describing how concentrated the population is, for the run log.
    A uniform density gives roughly a tenth; anything well above that is the
    hotspots doing work.
    """
    counts = np.zeros(raster.n_rows * raster.n_cols, dtype=np.int64)
    np.add.at(counts, cell_row * raster.n_cols + cell_col, 1)

    eligible = int(np.count_nonzero(raster.free_fraction > 0.0))
    if eligible == 0:
        return float("nan")
    top = max(1, round(0.1 * eligible))
    return float(np.sort(counts)[::-1][:top].sum() / counts.sum())


def write_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    component: np.ndarray,
    raster: Raster,
    ue: UeSpec,
) -> Path:
    """Write the UE population, one row per UE. Returns ``path``.

    Coordinates are written at millimetre precision so a rerun at the same seed
    is byte-identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_col, cell_row = raster.cell_indices(x, y)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for index in range(x.size):
            writer.writerow(
                (
                    f"ue{index}",
                    f"{x[index]:.3f}",
                    f"{y[index]:.3f}",
                    f"{ue.height_m:.3f}",
                    int(cell_col[index]),
                    int(cell_row[index]),
                    int(component[index]),
                )
            )
    return path
