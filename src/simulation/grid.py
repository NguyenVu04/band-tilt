"""Measurement-grid construction and scene rasters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.simulation.scene import SceneBounds, surface_height


@dataclass(frozen=True)
class GridSpec:
    """How finely to divide the scene, and how to classify what is in a cell.

    Attributes:
        cell_size_m: Side of a square cell.
        subsamples_per_cell: Side of the sub-grid cast per cell; the cost is
            this squared in rays.
        free_height_tol_m: Surface height at or below which a point counts as
            open ground.
    """

    cell_size_m: float
    subsamples_per_cell: int
    free_height_tol_m: float

    def __post_init__(self) -> None:
        """Reject a grid nothing could be sampled over.

        Raises:
            ValueError: When the cell size or the sub-grid is not positive.
        """
        if self.cell_size_m <= 0:
            raise ValueError(
                f"simulation.grid.cell_size_m must be positive, got {self.cell_size_m}"
            )
        if self.subsamples_per_cell <= 0:
            raise ValueError(
                "simulation.grid.subsamples_per_cell must be positive, "
                f"got {self.subsamples_per_cell}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> GridSpec:
        """Read ``simulation.grid``."""
        grid = cfg.simulation.grid
        return cls(
            cell_size_m=float(grid.cell_size_m),
            subsamples_per_cell=int(grid.subsamples_per_cell),
            free_height_tol_m=float(grid.free_height_tol_m),
        )


@dataclass(frozen=True)
class Raster:
    """What the ray casts found, per cell.

    Attributes:
        origin_x: The x of the grid's lower corner.
        origin_y: The y of the grid's lower corner.
        cell_size_m: Side of a square cell.
        free_fraction: Share of each cell that is open ground, in ``[0, 1]``,
            shaped ``[n_rows, n_cols]``.
        mean_built_height: Mean height of the building surface within each
            cell, zero where the cell holds none, shaped ``[n_rows, n_cols]``.
    """

    origin_x: float
    origin_y: float
    cell_size_m: float
    free_fraction: np.ndarray
    mean_built_height: np.ndarray

    @property
    def n_rows(self) -> int:
        """Number of cells along y."""
        return int(self.free_fraction.shape[0])

    @property
    def n_cols(self) -> int:
        """Number of cells along x."""
        return int(self.free_fraction.shape[1])

    @property
    def cell_area_m2(self) -> float:
        """Area of one whole cell."""
        return self.cell_size_m * self.cell_size_m

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Centre coordinates of every cell, each shaped ``[n_rows, n_cols]``."""
        xs = self.origin_x + (np.arange(self.n_cols) + 0.5) * self.cell_size_m
        ys = self.origin_y + (np.arange(self.n_rows) + 0.5) * self.cell_size_m
        return np.meshgrid(xs, ys, indexing="xy")

    def cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Column and row index of each ``(x, y)``, clipped to the grid."""
        col = np.floor((np.asarray(x) - self.origin_x) / self.cell_size_m).astype(np.int64)
        row = np.floor((np.asarray(y) - self.origin_y) / self.cell_size_m).astype(np.int64)
        return (
            np.clip(col, 0, self.n_cols - 1),
            np.clip(row, 0, self.n_rows - 1),
        )


def build(mi_scene: Any, bounds: SceneBounds, spec: GridSpec, seed: int) -> Raster:
    """Raster the scene into cells by casting a jittered sub-grid over it.

    One :func:`~src.simulation.scene.surface_height` call covers the whole
    scene. Returns the open-ground and building rasters.
    """
    n_cols = int(np.ceil(bounds.width_m / spec.cell_size_m))
    n_rows = int(np.ceil(bounds.depth_m / spec.cell_size_m))
    sub = spec.subsamples_per_cell
    step = spec.cell_size_m / sub

    rng = np.random.default_rng(seed)
    shape = (n_rows, n_cols, sub, sub)
    # One random point is sampled in each sub-cell.
    offset_x = (np.arange(sub).reshape(1, 1, 1, sub) + rng.random(shape)) * step
    offset_y = (np.arange(sub).reshape(1, 1, sub, 1) + rng.random(shape)) * step

    x = bounds.min_x + np.arange(n_cols).reshape(1, n_cols, 1, 1) * spec.cell_size_m + offset_x
    y = bounds.min_y + np.arange(n_rows).reshape(n_rows, 1, 1, 1) * spec.cell_size_m + offset_y

    height = surface_height(mi_scene, x, y, bounds.launch_z)

    # A ray miss is not open ground.
    hit = np.isfinite(height)
    free = hit & (height <= spec.free_height_tol_m)
    built = hit & (height > spec.free_height_tol_m)

    built_count = built.sum(axis=(2, 3))
    built_sum = np.where(built, np.nan_to_num(height), 0.0).sum(axis=(2, 3))
    mean_built_height = np.divide(
        built_sum,
        built_count,
        out=np.zeros_like(built_sum, dtype=np.float64),
        where=built_count > 0,
    )

    return Raster(
        origin_x=bounds.min_x,
        origin_y=bounds.min_y,
        cell_size_m=spec.cell_size_m,
        free_fraction=free.mean(axis=(2, 3)).astype(np.float64),
        mean_built_height=mean_built_height,
    )


def roi_mask(raster: Raster, roi: SceneBounds) -> np.ndarray:
    """Cells whose centre lies inside ``roi``, shaped ``[n_rows, n_cols]``.

    ``roi`` is an already-inset extent (:meth:`SceneBounds.inset`), so the
    margin is applied in one place and every stage masks against the same
    region. Cell membership is decided on the centre alone: a cell straddling
    the boundary is either in or out, which keeps the mask a property of the
    grid rather than of how finely the cell was sub-sampled.
    """
    centre_x, centre_y = raster.cell_centres()
    return (
        (centre_x >= roi.min_x)
        & (centre_x <= roi.max_x)
        & (centre_y >= roi.min_y)
        & (centre_y <= roi.max_y)
    )
