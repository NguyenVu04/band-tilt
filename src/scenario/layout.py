"""Lay the nodes and their cells out on the scene's open ground.

The layout is drawn once per scenario and then held fixed: only tilt moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import Cell, Tilt, cells_to_frame
from src.scenario.grid import GridSpec, Raster, disc_offsets
from src.simulation import scene as scene_module
from src.simulation.scene import SceneBounds

# One node on the scene centre and three around it at 120 degrees, each
# node_spacing_m from the centre: a centre site and three of its first-tier
# neighbours on a hexagonal grid of that inter-site distance.
_CORNER_COUNT = 3


@dataclass(frozen=True)
class LayoutSpec:
    """Where the nodes go, and how their masts are mounted.

    Attributes:
        node_spacing_m: Inter-site distance from the centre node to each
            corner node. The corners are ``sqrt(3)`` times this apart, as
            alternate first-tier neighbours on a hexagonal grid are.
        cells_per_node: Cells per node, at evenly spaced azimuths.
        azimuth_offset_deg: Rotation applied to every node's cell fan.
        mast_height_m: Height of the mast above the ground it stands on.
        min_free_fraction: Share of a tile that must be open ground before a
            mast may stand on it. Below one rather than at it because the
            raster estimates the share from a sub-sampled cast, so an
            unobstructed tile beside a wall can read slightly short.
        clearance_radius_m: Every tile within this radius must also be free, so
            a mast is not wedged against a facade it would immediately shadow.
        snap_radius_m: How far to look for a free tile when the ideal position
            has none. Finding none within it fails the node rather than
            falling back to the ideal position.
    """

    node_spacing_m: float
    cells_per_node: int
    azimuth_offset_deg: float
    mast_height_m: float
    min_free_fraction: float
    clearance_radius_m: float
    snap_radius_m: float

    def __post_init__(self) -> None:
        """Reject a layout no node could be placed under.

        Raises:
            ValueError: When the node spacing, mast height or cells per node
                is not positive, a radius is negative, or the free fraction is
                outside ``[0, 1]``.
        """
        if self.node_spacing_m <= 0:
            raise ValueError(
                f"scenario.layout.node_spacing_m must be positive, got {self.node_spacing_m}"
            )
        if self.cells_per_node < 1:
            raise ValueError(
                f"scenario.layout.cells_per_node must be at least 1, got {self.cells_per_node}"
            )
        for name in ("clearance_radius_m", "snap_radius_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"scenario.layout.{name} must not be negative")
        if self.mast_height_m <= 0:
            raise ValueError(
                f"scenario.layout.mast_height_m must be positive, got {self.mast_height_m}"
            )
        if not 0.0 <= self.min_free_fraction <= 1.0:
            raise ValueError(
                f"scenario.layout.min_free_fraction must be in [0, 1], got {self.min_free_fraction}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> LayoutSpec:
        """Read ``scenario.layout``."""
        layout = cfg.scenario.layout
        return cls(
            node_spacing_m=float(layout.node_spacing_m),
            cells_per_node=int(layout.cells_per_node),
            azimuth_offset_deg=float(layout.azimuth_offset_deg),
            mast_height_m=float(layout.mast_height_m),
            min_free_fraction=float(layout.min_free_fraction),
            clearance_radius_m=float(layout.clearance_radius_m),
            snap_radius_m=float(layout.snap_radius_m),
        )


def default_tilts(cfg: DictConfig) -> dict[str, Tilt]:
    """Read ``scenario.layout.default_tilt``, keyed by band name.

    The starting tilt a freshly generated layout is stamped with. Higher bands
    conventionally start tilted harder, to contain a smaller footprint; that is
    a default, not a rule the cell table has to keep obeying.
    """
    entries = cfg.scenario.layout.default_tilt
    return {str(band): Tilt.from_config(value) for band, value in entries.items()}


def default_max_prb(cfg: DictConfig) -> dict[str, int]:
    """Read ``scenario.layout.default_max_prb``, keyed by band name."""
    entries = cfg.scenario.layout.default_max_prb
    return {str(band): int(value) for band, value in entries.items()}


def node_positions(bounds: SceneBounds, spacing_m: float) -> list[tuple[float, float]]:
    """Return each node's ideal ``(x, y)``: a triangle's three corners, then its centroid.

    The centroid is the scene centre and every corner is ``spacing_m`` from it,
    so ``spacing_m`` is the inter-site distance of a hexagonal layout and the
    corners are ``sqrt(3) * spacing_m`` apart. One corner points along ``+y``.

    Raises:
        ValueError: When the triangle does not fit inside the scene.
    """
    # Spans sqrt(3) * spacing_m along x, and from -spacing_m / 2 to +spacing_m
    # about the centre along y.
    max_spacing = min(bounds.width_m / math.sqrt(3.0), bounds.depth_m / 2.0)
    if spacing_m > max_spacing:
        raise ValueError(
            "the node triangle at "
            f"scenario.layout.node_spacing_m={spacing_m} does not fit the "
            f"{bounds.width_m:.1f} x {bounds.depth_m:.1f} m scene. Lower the spacing to at "
            f"most {max_spacing:.1f} m."
        )

    centre_x = 0.5 * (bounds.min_x + bounds.max_x)
    centre_y = 0.5 * (bounds.min_y + bounds.max_y)
    radius = spacing_m
    angles = [math.radians(90.0 + index * 360.0 / _CORNER_COUNT) for index in range(_CORNER_COUNT)]
    corners = [
        (centre_x + radius * math.cos(angle), centre_y + radius * math.sin(angle))
        for angle in angles
    ]
    return [*corners, (centre_x, centre_y)]


def generate_layout(
    mi_scene: Any,
    bounds: SceneBounds,
    raster: Raster,
    spec: LayoutSpec,
    grid_spec: GridSpec,
    default_tilt: dict[str, Tilt],
    max_prb: dict[str, int],
) -> pd.DataFrame:
    """Lay the nodes out on a triangle's corners and centroid, each on open ground.

    Returns the cell table (:data:`src.core.cell.CELL_COLUMNS`):
    ``spec.cells_per_node`` cells for every node. Nodes whose ideal
    position is built over are snapped to the nearest open-ground tile within
    ``spec.snap_radius_m``.

    ``grid_spec`` must be the one ``raster`` was built from, so a mast obeys
    exactly the open-ground definition the UEs were drawn against.

    Every cell is stamped with ``default_tilt`` — the same starting tilt per
    band — and with ``max_prb``. That is a starting point, not a constraint:
    the written table is the authority afterwards, and its entries are meant
    to diverge, since one tilt per cell-band pair is what is being optimized.

    Raises:
        ValueError: When the triangle does not fit inside the scene, or when a
            node finds no open ground within ``spec.snap_radius_m``. Both are
            config changes rather than something to snap away, and neither may
            be answered by placing a mast on a building.
    """
    cells: list[Cell] = []
    for node_index, (ideal_x, ideal_y) in enumerate(node_positions(bounds, spec.node_spacing_m)):
        x, y, z = _mount(
            mi_scene,
            bounds,
            raster,
            ideal_x,
            ideal_y,
            spec,
            grid_spec.free_height_tol_m,
            f"n{node_index}",
        )
        for cell_index in range(spec.cells_per_node):
            azimuth = spec.azimuth_offset_deg + cell_index * 360.0 / spec.cells_per_node
            cells.append(
                Cell(
                    name=f"n{node_index}c{cell_index}",
                    node=f"n{node_index}",
                    x=x,
                    y=y,
                    z=z,
                    azimuth_deg=azimuth % 360.0,
                    tilt=dict(default_tilt),
                    max_prb=dict(max_prb),
                )
            )
    return cells_to_frame(cells)


def _mount(
    mi_scene: Any,
    bounds: SceneBounds,
    raster: Raster,
    x: float,
    y: float,
    spec: LayoutSpec,
    free_height_tol_m: float,
    node_name: str,
) -> tuple[float, float, float]:
    """Find open ground at or near ``(x, y)`` and return the mast position on it.

    Returns ``(x, y, z)``, with ``z`` the measured ground height plus the mast.

    The raster narrows the search to tiles that are open and clear, but it
    records the open *share* of a tile, so a point inside an accepted tile can
    still be on a building. Each surviving candidate is therefore cast
    individually and kept only when the ray comes back at or below
    ``free_height_tol_m`` — the same open-ground test the UEs were drawn
    against. That cast also supplies the ground height to stand the mast on,
    and misses for a candidate beyond the scene edge, which rejects it.

    Raises:
        ValueError: When no candidate within ``spec.snap_radius_m`` is open
            ground. Placing the mast anyway would put it on a building, so the
            layout fails instead.
    """
    # Concentric rings outward from the ideal position, so the first acceptable
    # tile found is also the nearest.
    for radius in np.arange(0.0, spec.snap_radius_m + 1e-9, 5.0):
        if radius == 0.0:
            candidates_x, candidates_y = np.array([x]), np.array([y])
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, max(8, int(radius)), endpoint=False)
            candidates_x = x + radius * np.cos(angles)
            candidates_y = y + radius * np.sin(angles)

        usable = _tiles_are_clear(raster, candidates_x, candidates_y, spec)
        index = np.flatnonzero(usable)
        if index.size == 0:
            continue

        ground = scene_module.surface_height(
            mi_scene, candidates_x[index], candidates_y[index], bounds.launch_z
        )
        # A missed ray is a point beyond the ground, not a point at height zero.
        on_ground = np.isfinite(ground) & (ground <= free_height_tol_m)
        if on_ground.any():
            best = int(np.argmax(on_ground))
            return (
                float(candidates_x[index[best]]),
                float(candidates_y[index[best]]),
                float(ground[best]) + spec.mast_height_m,
            )

    raise ValueError(
        f"node {node_name} at ({x:.1f}, {y:.1f}) found no open ground within "
        f"scenario.layout.snap_radius_m={spec.snap_radius_m}. Raise it, or "
        f"lower min_free_fraction={spec.min_free_fraction} or "
        f"clearance_radius_m={spec.clearance_radius_m}; a mast may not stand on a building."
    )


def _tiles_are_clear(
    raster: Raster,
    x: np.ndarray,
    y: np.ndarray,
    spec: LayoutSpec,
) -> np.ndarray:
    """Which points stand on a free tile whose surround is also free.

    Direct accumulation over the disc of tile offsets, as in
    :func:`src.scenario.density.neighbourhood_volume`; the radius spans a
    handful of tiles, so nothing cleverer pays for itself. Offsets are clipped
    to the grid, which can only re-test an in-bounds tile, and a node beyond
    the scene is rejected by its ground cast anyway.
    """
    radius_tiles = int(math.ceil(spec.clearance_radius_m / raster.tile_size_m))
    col, row = raster.tile_indices(x, y)

    clear = np.ones(np.shape(x), dtype=bool)
    for d_row, d_col in disc_offsets(radius_tiles):
        neighbour_row = np.clip(row + d_row, 0, raster.n_rows - 1)
        neighbour_col = np.clip(col + d_col, 0, raster.n_cols - 1)
        clear &= raster.free_fraction[neighbour_row, neighbour_col] >= spec.min_free_fraction
    return clear
