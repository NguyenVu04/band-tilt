"""Generate node layouts and build Sionna-RT transmitters.

Layouts use open ground in the delivered scene and remain fixed per scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from src.core.cell import Cell, Tilt
from src.simulation import grid as grid_module
from src.simulation import scene as scene_module
from src.simulation import seeds
from src.simulation.grid import GridSpec, Raster
from src.simulation.scene import SceneBounds, SceneSpec

# Nodes per side of the square lattice.
_LATTICE_SIDE = 2


@dataclass(frozen=True)
class LayoutSpec:
    """Where the nodes go, and how their masts are mounted.

    Attributes:
        node_spacing_m: Distance between neighbouring nodes on the lattice.
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
            ValueError: When the mast height is not positive, or the free
                fraction is outside ``[0, 1]``.
        """
        if self.mast_height_m <= 0:
            raise ValueError(
                "simulation.transmitters.layout.mast_height_m must be positive, "
                f"got {self.mast_height_m}"
            )
        if not 0.0 <= self.min_free_fraction <= 1.0:
            raise ValueError(
                "simulation.transmitters.layout.min_free_fraction must be in [0, 1], "
                f"got {self.min_free_fraction}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> LayoutSpec:
        """Read ``simulation.transmitters.layout``."""
        layout = cfg.simulation.transmitters.layout
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
    """Read ``simulation.transmitters.layout.default_tilt``, keyed by band name.

    The starting tilt a freshly generated layout is stamped with. Higher bands
    conventionally start tilted harder, to contain a smaller footprint; that is
    a default, not a rule the cell table has to keep obeying.
    """
    entries = cfg.simulation.transmitters.layout.default_tilt
    return {str(band): Tilt.from_config(value) for band, value in entries.items()}


def default_max_prb(cfg: DictConfig) -> dict[str, int]:
    """Read ``simulation.transmitters.layout.default_max_prb``, keyed by band name."""
    entries = cfg.simulation.transmitters.layout.default_max_prb
    return {str(band): int(value) for band, value in entries.items()}


def load(cfg: DictConfig) -> tuple[Cell, ...]:
    """Read the cell table from ``simulation.transmitters.cells``.

    Raises:
        ValueError: When the table is empty, which means the layout generator
            has not been run yet.
    """
    entries = cfg.simulation.transmitters.cells
    if not entries:
        raise ValueError(
            "simulation.transmitters.cells is empty. Generate the layout with "
            "`task simulation:layout` and paste the printed table into "
            "configs/simulation.yaml."
        )
    return tuple(Cell.from_config(entry) for entry in entries)


def generate(
    mi_scene: Any,
    bounds: SceneBounds,
    raster: Raster,
    spec: LayoutSpec,
    grid_spec: GridSpec,
    default_tilt: dict[str, Tilt],
    max_prb: dict[str, int],
) -> tuple[Cell, ...]:
    """Lay nodes out on a square lattice and stand each on open ground.

    Returns ``spec.cells_per_node`` cells for every node. Nodes whose ideal
    position is built over are snapped to the nearest open-ground tile within
    ``spec.snap_radius_m``.

    ``grid_spec`` must be the one ``raster`` was built from, so a mast obeys
    exactly the open-ground definition the UEs were drawn against.

    Every cell is stamped with ``default_tilt`` — the same starting tilt per
    band — and with ``max_prb``. That is a starting point, not a constraint:
    the emitted table is the authority afterwards, and its entries are meant
    to diverge, since one tilt per cell-band pair is what is being optimized.

    Raises:
        ValueError: When the lattice does not fit inside the scene, or when a
            node finds no open ground within ``spec.snap_radius_m``. Both are
            config changes rather than something to snap away, and neither may
            be answered by placing a mast on a building.
    """
    span = (_LATTICE_SIDE - 1) * spec.node_spacing_m
    if span > min(bounds.width_m, bounds.depth_m):
        raise ValueError(
            f"a {_LATTICE_SIDE}x{_LATTICE_SIDE} lattice at "
            f"simulation.transmitters.layout.node_spacing_m={spec.node_spacing_m} spans "
            f"{span:.1f} m, which does not fit the {bounds.width_m:.1f} x "
            f"{bounds.depth_m:.1f} m scene. Lower the spacing to at most "
            f"{min(bounds.width_m, bounds.depth_m) / (_LATTICE_SIDE - 1):.1f} m."
        )

    centre_x = 0.5 * (bounds.min_x + bounds.max_x)
    centre_y = 0.5 * (bounds.min_y + bounds.max_y)
    offsets = [
        (index - 0.5 * (_LATTICE_SIDE - 1)) * spec.node_spacing_m for index in range(_LATTICE_SIDE)
    ]

    cells: list[Cell] = []
    for node_index, (offset_x, offset_y) in enumerate((dx, dy) for dx in offsets for dy in offsets):
        x, y, z = _mount(
            mi_scene,
            bounds,
            raster,
            centre_x + offset_x,
            centre_y + offset_y,
            spec,
            grid_spec.free_height_tol_m,
            f"n{node_index}",
        )
        for cell_index in range(spec.cells_per_node):
            azimuth = spec.azimuth_offset_deg + cell_index * 360.0 / spec.cells_per_node
            cells.append(
                Cell(
                    name=f"n{node_index}c{cell_index}",
                    x=x,
                    y=y,
                    z=z,
                    azimuth_deg=azimuth % 360.0,
                    tilt=dict(default_tilt),
                    max_prb=dict(max_prb),
                )
            )
    return tuple(cells)


def build(scene: Any, cells: tuple[Cell, ...], band_name: str, power_dbm: float) -> None:
    """Add one transmitter per cell to the scene, at that cell's band tilt.

    Only the current carrier's transmitters are added: a scene carries one
    frequency, so the bands are solved in turn rather than together. Each
    cell contributes its own tilt for ``band_name``, so two bands of the same
    cell can point differently.

    Tilt is the pitch component of the orientation. A rotation about the y axis
    carries the boresight from ``+x`` toward ``-z``, so a *positive* pitch is a
    downtilt.
    """
    from sionna.rt import Transmitter

    for cell in cells:
        scene.add(
            Transmitter(
                name=cell.name,
                position=[cell.x, cell.y, cell.z],
                orientation=[
                    math.radians(cell.azimuth_deg),
                    math.radians(cell.tilt_for(band_name).baseline_deg),
                    0.0,
                ],
                power_dbm=power_dbm,
            )
        )


def validate(
    mi_scene: Any,
    bounds: SceneBounds,
    cells: tuple[Cell, ...],
    free_height_tol_m: float,
) -> tuple[str, ...]:
    """Report masts the current geometry no longer supports. Empty tuple is all clear.

    Holds the layout to the rule :func:`generate` places it under — a mast
    stands on open ground, at a measured height above it — against whatever the
    scene holds now. A changed scene can swallow a mast, leave it standing on a
    roof, or leave it on nothing. All three are reported for the run log; none is
    silently corrected, because moving a mast would defeat the point of holding
    the layout fixed.

    ``free_height_tol_m`` is ``simulation.grid.free_height_tol_m``, so ground and
    building mean the same thing here as everywhere else in the pipeline.
    """
    x = np.array([cell.x for cell in cells])
    y = np.array([cell.y for cell in cells])
    height = scene_module.surface_height(mi_scene, x, y, bounds.launch_z)

    problems = []
    for cell, surface in zip(cells, height, strict=True):
        if not np.isfinite(surface):
            problems.append(f"{cell.name}: mast at {cell.z:.1f} m stands over no surface at all")
        elif surface > cell.z:
            problems.append(
                f"{cell.name}: mast at {cell.z:.1f} m is inside geometry reaching {surface:.1f} m"
            )
        elif surface > free_height_tol_m:
            problems.append(
                f"{cell.name}: mast at {cell.z:.1f} m stands on a building "
                f"reaching {surface:.1f} m, not on open ground"
            )
    return tuple(problems)


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
        f"simulation.transmitters.layout.snap_radius_m={spec.snap_radius_m}. Raise it, or "
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
    :func:`src.simulation.density.neighbourhood_volume`; the radius spans a
    handful of tiles, so nothing cleverer pays for itself. Offsets are clipped
    to the grid, which can only re-test an in-bounds tile, and a node beyond
    the scene is rejected by its ground cast anyway.
    """
    radius_tiles = int(math.ceil(spec.clearance_radius_m / raster.tile_size_m))
    col, row = raster.tile_indices(x, y)

    clear = np.ones(np.shape(x), dtype=bool)
    for d_row in range(-radius_tiles, radius_tiles + 1):
        for d_col in range(-radius_tiles, radius_tiles + 1):
            if d_row * d_row + d_col * d_col > radius_tiles * radius_tiles:
                continue
            neighbour_row = np.clip(row + d_row, 0, raster.n_rows - 1)
            neighbour_col = np.clip(col + d_col, 0, raster.n_cols - 1)
            clear &= raster.free_fraction[neighbour_row, neighbour_col] >= spec.min_free_fraction
    return clear


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the cell table and print it. Entry point for ``task simulation:layout``.

    Mounts against the delivered scene: the layout is surveyed once and then
    held fixed across every scenario.
    """
    spec = LayoutSpec.from_config(cfg)
    grid_spec = GridSpec.from_config(cfg)
    default_tilt = default_tilts(cfg)
    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))

    # The raster the UEs are drawn against is built the same way, from the same
    # stream, so "free tile" means one thing across the whole pipeline.
    raster = grid_module.build(scene.mi_scene, bounds, grid_spec, seeds.stream(cfg, "scene"))
    cells = generate(
        scene.mi_scene, bounds, raster, spec, grid_spec, default_tilt, default_max_prb(cfg)
    )

    print(f"# {len(cells)} cells over {len(cells) // spec.cells_per_node} nodes")
    print(f"# {len(cells) * len(default_tilt)} cell-band tilts, all at the layout default")
    print(
        f"# masts at {spec.mast_height_m} m on tiles at least "
        f"{spec.min_free_fraction:.0%} open, cleared to {spec.clearance_radius_m} m"
    )
    print("  cells:")
    for cell in cells:
        print(
            f"    - name: {cell.name}\n"
            f"      x: {cell.x:.2f}\n"
            f"      y: {cell.y:.2f}\n"
            f"      z: {cell.z:.2f}\n"
            f"      azimuth_deg: {cell.azimuth_deg:.1f}\n"
            "      tilt:"
        )
        for band, tilt in cell.tilt.items():
            print(
                f"        {band}: {{baseline_deg: {tilt.baseline_deg}, "
                f"bounds_deg: [{tilt.bounds_deg[0]}, {tilt.bounds_deg[1]}]}}"
            )
        limits = ", ".join(f"{band}: {value}" for band, value in cell.max_prb.items())
        print(f"      max_prb: {{{limits}}}")


if __name__ == "__main__":
    main()
