"""The site layout, and the sionna-rt transmitters built from it.

Sites sit on a square lattice about the scene centre, each carrying evenly
spaced sectors, and every band is co-sited: one position, one azimuth, one mast,
several carriers. Multi-band tilt *coordination* is only a problem at all when
the bands share a footprint.

Deliberately few sites. Spread too densely over a small scene, every location
is strongly served by something, hole rate pins near zero and overlap near one
whatever the tilt, and the KPIs stop responding to the decision variable.
Coverage has to be contested for tilt to be worth optimising.

Masts are mounted against the **unperturbed** scene and then held fixed.
Perturbations model our uncertainty about the city, not changes an operator
reacts to: a real mast stays where it was surveyed even when the survey turns
out to have been wrong. :func:`validate` reports masts that a perturbation has
since buried or left unsupported, rather than quietly re-seating them.

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

from src.simulation import scene as scene_module
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
        rooftop_height_range_m: Surface heights that count as a mountable roof.
        mast_clearance_m: Height of the mast above the roof it stands on.
        snap_radius_m: How far to look for a mountable roof when the ideal
            position has none.
    """

    site_spacing_m: float
    sectors_per_site: int
    azimuth_offset_deg: float
    rooftop_height_range_m: tuple[float, float]
    mast_clearance_m: float
    snap_radius_m: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> LayoutSpec:
        """Read ``simulation.transmitters.layout``."""
        layout = cfg.simulation.transmitters.layout
        low, high = (float(value) for value in layout.rooftop_height_range_m)
        return cls(
            site_spacing_m=float(layout.site_spacing_m),
            sectors_per_site=int(layout.sectors_per_site),
            azimuth_offset_deg=float(layout.azimuth_offset_deg),
            rooftop_height_range_m=(low, high),
            mast_clearance_m=float(layout.mast_clearance_m),
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
    spec: LayoutSpec,
    default_tilt: dict[str, Tilt],
) -> tuple[Sector, ...]:
    """Lay sites out on a square lattice and mount each on a roof.

    Returns one :class:`Sector` per site per sector. Sites whose ideal position
    has no mountable roof are snapped to the nearest one within
    ``spec.snap_radius_m``; a site with none at all is mounted at street level
    on the clearance alone, which is reported by the entry point rather than
    hidden.

    Every sector is stamped with ``default_tilt`` — the same starting tilt per
    band. That is a starting point, not a constraint: the emitted table is the
    authority afterwards, and its entries are meant to diverge, since one tilt
    per cell-band pair is what is being optimized.
    """
    centre_x = 0.5 * (bounds.min_x + bounds.max_x)
    centre_y = 0.5 * (bounds.min_y + bounds.max_y)
    offsets = [
        (index - 0.5 * (_LATTICE_SIDE - 1)) * spec.site_spacing_m for index in range(_LATTICE_SIDE)
    ]

    sectors: list[Sector] = []
    for site_index, (offset_x, offset_y) in enumerate((dx, dy) for dx in offsets for dy in offsets):
        x, y, z = _mount(mi_scene, bounds, centre_x + offset_x, centre_y + offset_y, spec)
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


def validate(mi_scene: Any, bounds: SceneBounds, sectors: tuple[Sector, ...]) -> tuple[str, ...]:
    """Report masts the current geometry has buried. Empty tuple means all clear.

    A perturbation that raises a building can swallow a mast that was mounted
    against the delivered scene, and one that removes a building leaves its
    mast standing on nothing. Both are reported for the run log; neither is
    silently corrected, because moving a mast to suit a perturbation would
    defeat the point of holding the layout fixed.
    """
    x = np.array([sector.x for sector in sectors])
    y = np.array([sector.y for sector in sectors])
    height = scene_module.surface_height(mi_scene, x, y, bounds.launch_z)

    problems = []
    for sector, surface in zip(sectors, height, strict=True):
        if np.isfinite(surface) and surface > sector.z:
            problems.append(
                f"{sector.name}: mast at {sector.z:.1f} m is inside geometry "
                f"reaching {surface:.1f} m"
            )
    return tuple(problems)


def _mount(
    mi_scene: Any,
    bounds: SceneBounds,
    x: float,
    y: float,
    spec: LayoutSpec,
) -> tuple[float, float, float]:
    """Find a roof at or near ``(x, y)`` and return the mast position on it."""
    low, high = spec.rooftop_height_range_m

    # Concentric rings outward from the ideal position, so the first acceptable
    # roof found is also the nearest.
    radii = np.arange(0.0, spec.snap_radius_m + 1e-9, 5.0)
    for radius in radii:
        if radius == 0.0:
            candidates_x, candidates_y = np.array([x]), np.array([y])
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, max(8, int(radius)), endpoint=False)
            candidates_x = x + radius * np.cos(angles)
            candidates_y = y + radius * np.sin(angles)

        height = scene_module.surface_height(mi_scene, candidates_x, candidates_y, bounds.launch_z)
        usable = np.isfinite(height) & (height >= low) & (height <= high)
        if usable.any():
            index = int(np.argmax(usable))
            return (
                float(candidates_x[index]),
                float(candidates_y[index]),
                float(height[index]) + spec.mast_clearance_m,
            )

    return x, y, spec.mast_clearance_m


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the sector table and print it. Entry point for ``task simulation:layout``.

    Mounts against the delivered scene with no perturbation applied, which is
    the point: the layout is surveyed once and then held fixed across every
    scenario.
    """
    spec = LayoutSpec.from_config(cfg)
    default_tilt = default_tilts(cfg)
    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))
    sectors = generate(scene.mi_scene, bounds, spec, default_tilt)

    street_level = [sector.name for sector in sectors if sector.z <= spec.mast_clearance_m + 1e-6]
    print(f"# {len(sectors)} sectors over {len(sectors) // spec.sectors_per_site} sites")
    print(f"# {len(sectors) * len(default_tilt)} cell-band tilts, all at the layout default")
    if street_level:
        print(f"# no roof within snap_radius_m for: {', '.join(sorted(set(street_level)))}")
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
