"""The UE spatial density: a uniform background plus built-volume hotspots.

The density is deliberately non-uniform. The UE-weighted Band Priority Score
asks whether high-frequency layers serve high-demand areas; under a flat
density no such area exists, so the KPI cannot discriminate between
configurations.

Hotspot centres are drawn per scenario, weighted by the building volume
surrounding a cell. That places demand in dense street canyons, where
propagation is worst, so the optimizer faces a real tension between where the
users are and where coverage is cheap — which is the coverage-and-capacity
problem. Hotspots dropped into open squares would remove it.

The field is expressed as a mixture rather than a single array of per-cell
probabilities, because sampling a component before a cell is what lets each UE
record which component produced it.

Pure NumPy; no ray casting happens here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from src.simulation.grid import Raster


@dataclass(frozen=True)
class DensitySpec:
    """The shape of the density field, and how concentrated it is.

    Attributes:
        hotspot_mass_fraction: Share of UEs drawn from hotspots; the remainder
            is spread uniformly over open ground.
        n_hotspots: How many hotspots to draw.
        sigma_major_m: Range the major axis is drawn from.
        sigma_minor_m: Range the minor axis is drawn from, clamped to at most
            the major axis.
        built_volume_radius_m: Radius the surrounding building volume is
            summed over when weighting where hotspots land.
    """

    hotspot_mass_fraction: float
    n_hotspots: int
    sigma_major_m: tuple[float, float]
    sigma_minor_m: tuple[float, float]
    built_volume_radius_m: float

    def __post_init__(self) -> None:
        """Reject a mixture that cannot be normalised.

        Raises:
            ValueError: When the mass fraction is outside ``[0, 1]``, when
                hotspots carry mass but none are drawn, or when the radius is
                not positive.
        """
        if not 0.0 <= self.hotspot_mass_fraction <= 1.0:
            raise ValueError(
                "simulation.density.hotspot_mass_fraction must be in [0, 1], "
                f"got {self.hotspot_mass_fraction}"
            )
        if self.n_hotspots < 0:
            raise ValueError(
                f"simulation.density.n_hotspots must not be negative, got {self.n_hotspots}"
            )
        if self.n_hotspots == 0 and self.hotspot_mass_fraction > 0.0:
            raise ValueError(
                "simulation.density.hotspot_mass_fraction is "
                f"{self.hotspot_mass_fraction} but n_hotspots is 0, so that mass "
                "has nowhere to go. Set the fraction to 0 for a uniform density."
            )
        if self.built_volume_radius_m <= 0:
            raise ValueError(
                "simulation.density.built_volume_radius_m must be positive, "
                f"got {self.built_volume_radius_m}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> DensitySpec:
        """Read ``simulation.density``."""
        density = cfg.simulation.density
        major = tuple(float(value) for value in density.sigma_major_m)
        minor = tuple(float(value) for value in density.sigma_minor_m)
        return cls(
            hotspot_mass_fraction=float(density.hotspot_mass_fraction),
            n_hotspots=int(density.n_hotspots),
            sigma_major_m=(major[0], major[1]),
            sigma_minor_m=(minor[0], minor[1]),
            built_volume_radius_m=float(density.built_volume_radius_m),
        )


@dataclass(frozen=True)
class Hotspot:
    """One elliptical Gaussian centre of demand.

    Attributes:
        x: Centre x, in scene metres.
        y: Centre y, in scene metres.
        sigma_major_m: Standard deviation along the major axis.
        sigma_minor_m: Standard deviation along the minor axis.
        rotation_rad: Angle of the major axis, counter-clockwise from x.
    """

    x: float
    y: float
    sigma_major_m: float
    sigma_minor_m: float
    rotation_rad: float


@dataclass(frozen=True)
class DensityField:
    """The mixture the UE positions are drawn from.

    Attributes:
        component_mass: Share of UEs per component, summing to 1. Index 0 is
            the uniform background; the rest are the hotspots in order.
        cell_weights: Per-component distribution over flattened cells, shaped
            ``[n_components, n_rows * n_cols]``, each row summing to 1.
        hotspots: The drawn hotspots, in the order they appear above.
    """

    component_mass: np.ndarray
    cell_weights: np.ndarray
    hotspots: tuple[Hotspot, ...]


def built_volume(raster: Raster) -> np.ndarray:
    """Building volume standing in each cell.

    The blocked share of the cell's area times the mean height of the building
    surface over it — a proxy for how much occupied floorspace surrounds a
    location, and so for how many people are plausibly near it.
    """
    blocked_area = (1.0 - raster.free_fraction) * raster.cell_area_m2
    return blocked_area * raster.mean_built_height


def neighbourhood_volume(volume: np.ndarray, raster: Raster, radius_m: float) -> np.ndarray:
    """Sum ``volume`` over a disc of ``radius_m`` around each cell.

    Direct accumulation over the disc's cell offsets. The kernel spans only a
    handful of cells on a grid this size, so an FFT would cost more than it
    saves.
    """
    radius_cells = int(math.ceil(radius_m / raster.cell_size_m))
    padded = np.pad(volume, radius_cells)
    n_rows, n_cols = volume.shape

    total = np.zeros_like(volume, dtype=np.float64)
    for d_row in range(-radius_cells, radius_cells + 1):
        for d_col in range(-radius_cells, radius_cells + 1):
            if d_row * d_row + d_col * d_col > radius_cells * radius_cells:
                continue
            row0 = radius_cells + d_row
            col0 = radius_cells + d_col
            total += padded[row0 : row0 + n_rows, col0 : col0 + n_cols]
    return total


def draw_hotspots(
    raster: Raster,
    spec: DensitySpec,
    rng: np.random.Generator,
) -> tuple[Hotspot, ...]:
    """Draw hotspot centres, weighted by surrounding building volume.

    Centres are restricted to cells holding open ground. A centre inside a
    building would put its mass on whatever ring of open cells happens to
    surround the block, which is neither the intended shape nor a reproducible
    one.

    Raises:
        ValueError: When the grid holds fewer eligible cells than
            ``spec.n_hotspots``.
    """
    if spec.n_hotspots == 0:
        return ()

    weight = neighbourhood_volume(built_volume(raster), raster, spec.built_volume_radius_m)
    weight = np.where(raster.free_fraction > 0.0, weight, 0.0).ravel()

    eligible = int(np.count_nonzero(weight))
    if eligible < spec.n_hotspots:
        # A scene with no buildings beside its open ground: fall back to
        # placing hotspots uniformly rather than failing outright.
        weight = (raster.free_fraction > 0.0).astype(np.float64).ravel()
        eligible = int(np.count_nonzero(weight))
    if eligible < spec.n_hotspots:
        raise ValueError(
            f"simulation.density.n_hotspots is {spec.n_hotspots} but only {eligible} "
            "grid cells hold open ground. Lower n_hotspots or the cell size."
        )

    chosen = rng.choice(weight.size, size=spec.n_hotspots, replace=False, p=weight / weight.sum())
    rows, cols = np.divmod(chosen, raster.n_cols)

    hotspots = []
    for row, col in zip(rows, cols, strict=True):
        major = rng.uniform(*spec.sigma_major_m)
        minor = min(rng.uniform(*spec.sigma_minor_m), major)
        hotspots.append(
            Hotspot(
                x=raster.origin_x + (col + rng.random()) * raster.cell_size_m,
                y=raster.origin_y + (row + rng.random()) * raster.cell_size_m,
                sigma_major_m=major,
                sigma_minor_m=minor,
                rotation_rad=rng.uniform(0.0, math.pi),
            )
        )
    return tuple(hotspots)


def field(raster: Raster, spec: DensitySpec, seed: int) -> DensityField:
    """Build the mixture the UE positions are drawn from.

    Weights carry the density function alone. They are deliberately NOT scaled
    by a cell's open area: ``free_fraction`` is a sub-sampled estimate, and on
    a cell holding only a sliver of open ground it overstates the truth by more
    than an order of magnitude. Baking it in here would hand those cells that
    same factor in UE weight.

    Instead the open area enters through rejection in
    :func:`src.simulation.sample.sample_positions`, which draws uniformly
    inside a chosen cell and keeps only what lands on open ground. That makes
    the sampled density proportional to the density function times the cell's
    *true* open area, with the estimate used for nothing but eligibility.

    Cells with no open ground at all are excluded, since no position in them
    could ever be accepted.

    Hotspots share ``spec.hotspot_mass_fraction`` equally; concentration is
    controlled by that one number, not by per-hotspot amplitudes, so it stays
    interpretable under a sweep.
    """
    rng = np.random.default_rng(seed)
    hotspots = draw_hotspots(raster, spec, rng)

    eligible = (raster.free_fraction > 0.0).ravel().astype(np.float64)
    centre_x, centre_y = raster.cell_centres()
    centre_x = centre_x.ravel()
    centre_y = centre_y.ravel()

    weights = [_normalise(eligible)]
    for hotspot in hotspots:
        weights.append(_normalise(_gaussian(centre_x, centre_y, hotspot) * eligible))

    background_mass = 1.0 - spec.hotspot_mass_fraction
    per_hotspot = spec.hotspot_mass_fraction / len(hotspots) if hotspots else 0.0
    component_mass = np.array([background_mass] + [per_hotspot] * len(hotspots), dtype=np.float64)

    return DensityField(
        component_mass=component_mass,
        cell_weights=np.vstack(weights),
        hotspots=hotspots,
    )


def _gaussian(x: np.ndarray, y: np.ndarray, hotspot: Hotspot) -> np.ndarray:
    """Evaluate one rotated elliptical Gaussian, unnormalised, at each point."""
    cos = math.cos(hotspot.rotation_rad)
    sin = math.sin(hotspot.rotation_rad)
    d_x = x - hotspot.x
    d_y = y - hotspot.y
    major = cos * d_x + sin * d_y
    minor = -sin * d_x + cos * d_y
    return np.exp(
        -0.5 * ((major / hotspot.sigma_major_m) ** 2 + (minor / hotspot.sigma_minor_m) ** 2)
    )


def _normalise(weight: np.ndarray) -> np.ndarray:
    """Scale a non-negative weight vector to sum to 1.

    Raises:
        ValueError: When the weights sum to zero and cannot form a
            distribution.
    """
    total = weight.sum()
    if total <= 0.0:
        raise ValueError("Density component has zero total weight over the grid.")
    return weight / total
