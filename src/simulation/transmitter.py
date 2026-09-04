"""The site layout, and the sionna-rt transmitters built from it.

Sites sit on a square lattice about the scene centre, each carrying evenly
spaced sectors, and every band is co-sited: one position, one azimuth, one mast,
several carriers. Multi-band tilt *coordination* is only a problem at all when
the bands share a footprint.

Deliberately few sites. Spread too densely over a small scene, every location
is strongly served by something, hole rate pins near zero and overlap near one
whatever the tilt, and the KPIs stop responding to the decision variable.
Coverage has to be contested for tilt to be worth optimising.

Masts stand on open ground, not on roofs: a site is placed on a grid tile
with no building under it and a clear surround, the ground there is confirmed
by a downward ray, and the mast is raised a fixed height above it. Tying the
height to the tower rather than to whatever roof happened to be nearby keeps
the sites comparable to each other and stable across scenes. A site with no
such tile in reach fails the generator rather than being placed anyway.

Masts are mounted against the **unperturbed** scene and then held fixed.
Perturbations model our uncertainty about the city, not changes an operator
reacts to: a real mast stays where it was surveyed even when the survey turns
out to have been wrong. :func:`validate` reports masts that a perturbation has
since buried, left standing on a roof, or left unsupported, rather than quietly
re-seating them.

``python -m src.simulation.transmitter`` generates the layout and prints it as
YAML to paste into ``configs/simulation.yaml``. It is a one-off, not part of the
per-run chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from src.simulation import grid as grid_module
from src.simulation import scene as scene_module
from src.simulation import seeds
from src.simulation.grid import GridSpec, Raster
from src.simulation.scene import SceneBounds, SceneSpec

# Sites per row and column of the lattice. Two gives four sites, which over a
# scene of this size is already sparser than 3GPP's urban-macro reference.
_LATTICE_SIDE = 2


@dataclass(frozen=True)
class LayoutSpec:
    """Where the sites go, and how their masts are mounted.

    Attributes:
        site_spacing_m: Distance between neighbouring sites on the lattice.
        sectors_per_site: Sectors per site, at evenly spaced azimuths.
        azimuth_offset_deg: Rotation applied to every site's sector fan.
        mast_height_m: Height of the mast above the ground it stands on.
        min_free_fraction: Share of a tile that must be open ground before a
            mast may stand on it. Below one rather than at it because the
            raster estimates the share from a sub-sampled cast, so an
            unobstructed tile beside a wall can read slightly short.
        clearance_radius_m: Every tile within this radius must also be free, so
            a mast is not wedged against a facade it would immediately shadow.
        snap_radius_m: How far to look for a free tile when the ideal position
            has none. Finding none within it fails the site rather than
            falling back to the ideal position.
    """

    site_spacing_m: float
    sectors_per_site: int
    azimuth_offset_deg: float
    mast_height_m: float
    min_free_fraction: float
    clearance_radius_m: float
    snap_radius_m: float

    def __post_init__(self) -> None:
        """Reject a layout no site could be placed under.

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
            site_spacing_m=float(layout.site_spacing_m),
            sectors_per_site=int(layout.sectors_per_site),
            azimuth_offset_deg=float(layout.azimuth_offset_deg),
            mast_height_m=float(layout.mast_height_m),
            min_free_fraction=float(layout.min_free_fraction),
            clearance_radius_m=float(layout.clearance_radius_m),
            snap_radius_m=float(layout.snap_radius_m),
        )


@dataclass(frozen=True)
class Tilt:
    """The downtilt of one cell-band pair, and the range it may move in.

    Attributes:
        baseline_deg: The downtilt this cell-band starts at.
        bounds_deg: Inclusive range an optimizer may move it within.
    """

    baseline_deg: float
    bounds_deg: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject a tilt outside the range it is allowed to move in.

        Raises:
            ValueError: When the bounds are inverted or exclude the baseline.
                A baseline outside its own bounds means the run starts from an
                infeasible configuration, which PROJECT.md forbids throughout.
        """
        low, high = self.bounds_deg
        if low > high:
            raise ValueError(f"tilt bounds_deg {self.bounds_deg} is inverted")
        if not low <= self.baseline_deg <= high:
            raise ValueError(
                f"tilt baseline_deg {self.baseline_deg} lies outside its bounds "
                f"{self.bounds_deg}, so the run would start infeasible"
            )

    @classmethod
    def from_config(cls, entry: DictConfig) -> Tilt:
        """Read one ``baseline_deg``/``bounds_deg`` pair."""
        low, high = (float(value) for value in entry.bounds_deg)
        return cls(baseline_deg=float(entry.baseline_deg), bounds_deg=(low, high))


@dataclass(frozen=True)
class Sector:
    """One cell: a mast, an azimuth, and a tilt for each band it carries.

    Tilt is held per band rather than per sector because the decision variable
    is one absolute tilt per *cell-band* pair (PROJECT.md 3.1). A single tilt
    shared across a sector's bands would remove the very thing the project
    optimizes: the freedom to point frequency layers differently.

    Attributes:
        name: Unique across the layout; becomes the transmitter name and the
            stem of the MDT column names.
        x: Position east, in scene metres.
        y: Position north, in scene metres.
        z: Mast height above the scene's ground plane.
        azimuth_deg: Boresight bearing, counter-clockwise from the x axis.
        tilt: One :class:`Tilt` per band name.
    """

    name: str
    x: float
    y: float
    z: float
    azimuth_deg: float
    tilt: dict[str, Tilt]

    def tilt_for(self, band_name: str) -> Tilt:
        """The tilt this sector carries on one band.

        Raises:
            KeyError: When the sector has no entry for that band, which means
                the sector table and the band table disagree.
        """
        if band_name not in self.tilt:
            raise KeyError(
                f"sector {self.name!r} has no tilt for band {band_name!r}. Every sector "
                f"needs one per band; this one has {sorted(self.tilt)}."
            )
        return self.tilt[band_name]

    @classmethod
    def from_config(cls, entry: DictConfig) -> Sector:
        """Read one entry of ``simulation.transmitters.sectors``."""
        return cls(
            name=str(entry.name),
            x=float(entry.x),
            y=float(entry.y),
            z=float(entry.z),
            azimuth_deg=float(entry.azimuth_deg),
            tilt={str(band): Tilt.from_config(value) for band, value in entry.tilt.items()},
        )


def default_tilts(cfg: DictConfig) -> dict[str, Tilt]:
    """Read ``simulation.transmitters.layout.default_tilt``, keyed by band name.

    The starting tilt a freshly generated layout is stamped with. Higher bands
    conventionally start tilted harder, to contain a smaller footprint; that is
    a default, not a rule the sector table has to keep obeying.
    """
    entries = cfg.simulation.transmitters.layout.default_tilt
    return {str(band): Tilt.from_config(value) for band, value in entries.items()}


def load(cfg: DictConfig) -> tuple[Sector, ...]:
    """Read the sector table from ``simulation.transmitters.sectors``.

    Raises:
        ValueError: When the table is empty, which means the layout generator
            has not been run yet.
    """
    entries = cfg.simulation.transmitters.sectors
    if not entries:
        raise ValueError(
            "simulation.transmitters.sectors is empty. Generate the layout with "
            "`task simulation:layout` and paste the printed table into "
            "configs/simulation.yaml."
        )
    return tuple(Sector.from_config(entry) for entry in entries)


def generate(
    mi_scene: Any,
    bounds: SceneBounds,
    roi: SceneBounds,
    raster: Raster,
    spec: LayoutSpec,
    grid_spec: GridSpec,
    default_tilt: dict[str, Tilt],
) -> tuple[Sector, ...]:
    """Lay sites out on a square lattice and stand each on open ground.

    Returns one :class:`Sector` per site per sector. Sites whose ideal position
    is built over are snapped to the nearest open-ground tile within
    ``spec.snap_radius_m``.

    ``grid_spec`` must be the one ``raster`` was built from, so a mast obeys
    exactly the open-ground definition the UEs were drawn against.

    Every sector is stamped with ``default_tilt`` — the same starting tilt per
    band. That is a starting point, not a constraint: the emitted table is the
    authority afterwards, and its entries are meant to diverge, since one tilt
    per cell-band pair is what is being optimized.

    Raises:
        ValueError: When the lattice does not fit inside the region of
            interest, or when a site finds no open ground within
            ``spec.snap_radius_m``. Both are config changes rather than
            something to snap away, and neither may be answered by placing a
            mast on a building.
    """
    span = (_LATTICE_SIDE - 1) * spec.site_spacing_m
    if span > min(roi.width_m, roi.depth_m):
        raise ValueError(
            f"a {_LATTICE_SIDE}x{_LATTICE_SIDE} lattice at "
            f"simulation.transmitters.layout.site_spacing_m={spec.site_spacing_m} spans "
            f"{span:.1f} m, which does not fit the {roi.width_m:.1f} x {roi.depth_m:.1f} m "
            "region of interest. Lower the spacing to at most "
            f"{min(roi.width_m, roi.depth_m) / (_LATTICE_SIDE - 1):.1f} m, or lower "
            "simulation.area.margin_m."
        )

    centre_x = 0.5 * (bounds.min_x + bounds.max_x)
    centre_y = 0.5 * (bounds.min_y + bounds.max_y)
    offsets = [
        (index - 0.5 * (_LATTICE_SIDE - 1)) * spec.site_spacing_m for index in range(_LATTICE_SIDE)
    ]

    sectors: list[Sector] = []
    for site_index, (offset_x, offset_y) in enumerate((dx, dy) for dx in offsets for dy in offsets):
        x, y, z = _mount(
            mi_scene,
            bounds,
            roi,
            raster,
            centre_x + offset_x,
            centre_y + offset_y,
            spec,
            grid_spec.free_height_tol_m,
            f"s{site_index}",
        )
        for sector_index in range(spec.sectors_per_site):
            azimuth = spec.azimuth_offset_deg + sector_index * 360.0 / spec.sectors_per_site
            sectors.append(
                Sector(
                    name=f"s{site_index}c{sector_index}",
                    x=x,
                    y=y,
                    z=z,
                    azimuth_deg=azimuth % 360.0,
                    tilt=dict(default_tilt),
                )
            )
    return tuple(sectors)


def build(scene: Any, sectors: tuple[Sector, ...], band_name: str, power_dbm: float) -> None:
    """Add one transmitter per sector to the scene, at that sector's band tilt.

    Only the current carrier's transmitters are added: a scene carries one
    frequency, so the bands are solved in turn rather than together. Each
    sector contributes its own tilt for ``band_name``, so two bands of the same
    sector can point differently.

    Tilt is the pitch component of the orientation. A rotation about the y axis
    carries the boresight from ``+x`` toward ``-z``, so a *positive* pitch is a
    downtilt.
    """
    from sionna.rt import Transmitter

    for sector in sectors:
        scene.add(
            Transmitter(
                name=sector.name,
                position=[sector.x, sector.y, sector.z],
                orientation=[
                    math.radians(sector.azimuth_deg),
                    math.radians(sector.tilt_for(band_name).baseline_deg),
                    0.0,
                ],
                power_dbm=power_dbm,
            )
        )


def validate(
    mi_scene: Any,
    bounds: SceneBounds,
    sectors: tuple[Sector, ...],
    free_height_tol_m: float,
) -> tuple[str, ...]:
    """Report masts the current geometry no longer supports. Empty tuple is all clear.

    Holds the layout to the rule :func:`generate` places it under — a mast
    stands on open ground, at a measured height above it — against whatever the
    scene holds now. A perturbation that raises a building can swallow a mast
    mounted against the delivered scene or leave it standing on a roof, and one
    that removes a building leaves its mast on nothing. All three are reported
    for the run log; none is silently corrected, because moving a mast to suit a
    perturbation would defeat the point of holding the layout fixed.

    ``free_height_tol_m`` is ``simulation.grid.free_height_tol_m``, so ground and
    building mean the same thing here as everywhere else in the pipeline.
    """
    x = np.array([sector.x for sector in sectors])
    y = np.array([sector.y for sector in sectors])
    height = scene_module.surface_height(mi_scene, x, y, bounds.launch_z)

    problems = []
    for sector, surface in zip(sectors, height, strict=True):
        if not np.isfinite(surface):
            problems.append(
                f"{sector.name}: mast at {sector.z:.1f} m stands over no surface at all"
            )
        elif surface > sector.z:
            problems.append(
                f"{sector.name}: mast at {sector.z:.1f} m is inside geometry "
                f"reaching {surface:.1f} m"
            )
        elif surface > free_height_tol_m:
            problems.append(
                f"{sector.name}: mast at {sector.z:.1f} m stands on a building "
                f"reaching {surface:.1f} m, not on open ground"
            )
    return tuple(problems)


def _mount(
    mi_scene: Any,
    bounds: SceneBounds,
    roi: SceneBounds,
    raster: Raster,
    x: float,
    y: float,
    spec: LayoutSpec,
    free_height_tol_m: float,
    site_name: str,
) -> tuple[float, float, float]:
    """Find open ground at or near ``(x, y)`` and return the mast position on it.

    Returns ``(x, y, z)``, with ``z`` the measured ground height plus the mast.

    The raster narrows the search to tiles that are open and clear, but it
    records the open *share* of a cell, so a point inside an accepted cell can
    still be on a building. Each surviving candidate is therefore cast
    individually and kept only when the ray comes back at or below
    ``free_height_tol_m`` — the same open-ground test the UEs were drawn
    against. That cast also supplies the ground height to stand the mast on.

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
        usable &= _inside(roi, candidates_x, candidates_y)
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
        f"site {site_name} at ({x:.1f}, {y:.1f}) found no open ground within "
        f"simulation.transmitters.layout.snap_radius_m={spec.snap_radius_m}. Raise it, or "
        f"lower min_free_fraction={spec.min_free_fraction} or "
        f"clearance_radius_m={spec.clearance_radius_m}; a mast may not stand on a building."
    )


def _inside(roi: SceneBounds, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which of these points lie in the region of interest."""
    return (x >= roi.min_x) & (x <= roi.max_x) & (y >= roi.min_y) & (y <= roi.max_y)


def _tiles_are_clear(
    raster: Raster,
    x: np.ndarray,
    y: np.ndarray,
    spec: LayoutSpec,
) -> np.ndarray:
    """Which points stand on a free tile whose surround is also free.

    Direct accumulation over the disc of cell offsets, as in
    :func:`src.simulation.density.neighbourhood_volume`; the radius spans a
    handful of cells, so nothing cleverer pays for itself. Offsets are clipped
    to the grid, which can only re-test an in-bounds tile, and a site outside
    the region is rejected separately anyway.
    """
    radius_cells = int(math.ceil(spec.clearance_radius_m / raster.cell_size_m))
    col, row = raster.cell_indices(x, y)

    clear = np.ones(np.shape(x), dtype=bool)
    for d_row in range(-radius_cells, radius_cells + 1):
        for d_col in range(-radius_cells, radius_cells + 1):
            if d_row * d_row + d_col * d_col > radius_cells * radius_cells:
                continue
            neighbour_row = np.clip(row + d_row, 0, raster.n_rows - 1)
            neighbour_col = np.clip(col + d_col, 0, raster.n_cols - 1)
            clear &= raster.free_fraction[neighbour_row, neighbour_col] >= spec.min_free_fraction
    return clear


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the sector table and print it. Entry point for ``task simulation:layout``.

    Mounts against the delivered scene with no perturbation applied, which is
    the point: the layout is surveyed once and then held fixed across every
    scenario.
    """
    spec = LayoutSpec.from_config(cfg)
    grid_spec = GridSpec.from_config(cfg)
    default_tilt = default_tilts(cfg)
    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))

    # The raster the UEs are drawn against is built the same way, from the same
    # stream, so "free tile" means one thing across the whole pipeline. It is
    # built here from the unperturbed scene, which is what the layout is
    # surveyed against.
    raster = grid_module.build(scene.mi_scene, bounds, grid_spec, seeds.stream(cfg, "scene"))
    roi = bounds.inset(float(cfg.simulation.area.margin_m))
    sectors = generate(scene.mi_scene, bounds, roi, raster, spec, grid_spec, default_tilt)

    print(f"# {len(sectors)} sectors over {len(sectors) // spec.sectors_per_site} sites")
    print(f"# {len(sectors) * len(default_tilt)} cell-band tilts, all at the layout default")
    print(
        f"# masts at {spec.mast_height_m} m on tiles at least "
        f"{spec.min_free_fraction:.0%} open, cleared to {spec.clearance_radius_m} m"
    )
    print("  sectors:")
    for sector in sectors:
        print(
            f"    - name: {sector.name}\n"
            f"      x: {sector.x:.2f}\n"
            f"      y: {sector.y:.2f}\n"
            f"      z: {sector.z:.2f}\n"
            f"      azimuth_deg: {sector.azimuth_deg:.1f}\n"
            "      tilt:"
        )
        for band, tilt in sector.tilt.items():
            print(
                f"        {band}: {{baseline_deg: {tilt.baseline_deg}, "
                f"bounds_deg: [{tilt.bounds_deg[0]}, {tilt.bounds_deg[1]}]}}"
            )


if __name__ == "__main__":
    main()
