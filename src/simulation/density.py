"""UE density: a uniform background plus built-volume-weighted hotspots."""

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
        min_built_volume_fraction: Share of the densest neighbourhood's
            building volume a tile must carry before it may hold a hotspot
            centre. Zero admits every eligible tile.
    """

    hotspot_mass_fraction: float
    n_hotspots: int
    sigma_major_m: tuple[float, float]
    sigma_minor_m: tuple[float, float]
    built_volume_radius_m: float
    min_built_volume_fraction: float

    def __post_init__(self) -> None:
        """Reject a mixture that cannot be normalised.

        Raises:
            ValueError: When the mass fraction is outside ``[0, 1]``, when
                hotspots carry mass but none are drawn, when the radius is not
                positive, or when the built-volume fraction is outside
                ``[0, 1)``.
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
        if not 0.0 <= self.min_built_volume_fraction < 1.0:
            raise ValueError(
                "simulation.density.min_built_volume_fraction must be in [0, 1), "
                f"got {self.min_built_volume_fraction}"
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
            min_built_volume_fraction=float(density.min_built_volume_fraction),
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
    """Where each mixture component puts its UEs.

    The field carries no component masses. How much of the population a
    component holds varies per interval and belongs to
    :class:`src.simulation.traffic.Schedule`; where a component puts what it
    holds is a property of the scene and is drawn once. Splitting them is what
    lets a many-interval scenario reuse a single set of per-tile weights.

    Attributes:
        tile_weights: Per-component distribution over flattened tiles, shaped
            ``[n_components, n_rows * n_cols]``, each row summing to 1. Row 0
            is the uniform background; the rest are the hotspots in order.
        hotspots: The drawn hotspots, in the order their rows appear above.
    """

    tile_weights: np.ndarray
    hotspots: tuple[Hotspot, ...]

    @property
    def n_components(self) -> int:
        """Number of mixture components, background included."""
        return int(self.tile_weights.shape[0])


def built_volume(raster: Raster) -> np.ndarray:
    """Building volume standing in each tile.

    The blocked share of the tile's area times the mean height of the building
    surface over it — a proxy for how much occupied floorspace surrounds a
    location, and so for how many people are plausibly near it.
    """
    blocked_area = (1.0 - raster.free_fraction) * raster.tile_area_m2
    return blocked_area * raster.mean_built_height


def neighbourhood_volume(volume: np.ndarray, raster: Raster, radius_m: float) -> np.ndarray:
    """Sum ``volume`` over a disc of ``radius_m`` around each tile.

    Direct accumulation over the disc's tile offsets. The kernel spans only a
    handful of tiles on a grid this size, so an FFT would cost more than it
    saves.
    """
    radius_tiles = int(math.ceil(radius_m / raster.tile_size_m))
    padded = np.pad(volume, radius_tiles)
    n_rows, n_cols = volume.shape

    total = np.zeros_like(volume, dtype=np.float64)
    for d_row in range(-radius_tiles, radius_tiles + 1):
        for d_col in range(-radius_tiles, radius_tiles + 1):
            if d_row * d_row + d_col * d_col > radius_tiles * radius_tiles:
                continue
            row0 = radius_tiles + d_row
            col0 = radius_tiles + d_col
            total += padded[row0 : row0 + n_rows, col0 : col0 + n_cols]
    return total


def eligible_tiles(raster: Raster) -> np.ndarray:
    """Tiles that may hold a UE: any holding open ground.

    The single definition of eligibility, shared by the hotspot draw and the
    per-tile weights so the two cannot drift apart.
    """
    return raster.free_fraction > 0.0


def draw_hotspots(
    raster: Raster,
    spec: DensitySpec,
    rng: np.random.Generator,
) -> tuple[Hotspot, ...]:
    """Draw hotspot centres, weighted by surrounding building volume.

    Centres are restricted to eligible tiles. A centre inside a building would
    put its mass on whatever ring of open tiles happens to surround the block,
    which is neither the intended shape nor a reproducible one.

    The weight alone does not keep centres off near-empty ground: it is linear
    in building volume, and the outskirts hold enough tiles that their small
    individual weights still sum to a real share of the draw.
    ``spec.min_built_volume_fraction`` cuts that tail off, so a tile with too
    little surrounding volume cannot be chosen at any weight.

    Raises:
        ValueError: When the grid holds fewer eligible tiles than
            ``spec.n_hotspots``, or when the built-volume threshold leaves
            fewer candidates than that.
    """
    if spec.n_hotspots == 0:
        return ()

    allowed = eligible_tiles(raster)
    candidate_weights = neighbourhood_volume(
        built_volume(raster), raster, spec.built_volume_radius_m
    )
    candidate_weights = np.where(allowed, candidate_weights, 0.0).ravel()

    if int(np.count_nonzero(candidate_weights)) < spec.n_hotspots:
        # Use uniform eligible-tile weights when nearby building volume is zero.
        candidate_weights = allowed.astype(np.float64).ravel()
        n_eligible = int(np.count_nonzero(candidate_weights))
        if n_eligible < spec.n_hotspots:
            raise ValueError(
                f"simulation.density.n_hotspots is {spec.n_hotspots} but only {n_eligible} "
                "grid tiles hold open ground. Lower n_hotspots or the tile size."
            )
    else:
        candidate_weights = _above_threshold(candidate_weights, spec)

    chosen = rng.choice(
        candidate_weights.size,
        size=spec.n_hotspots,
        replace=False,
        p=candidate_weights / candidate_weights.sum(),
    )
    rows, cols = np.divmod(chosen, raster.n_cols)

    hotspots = []
    for row, col in zip(rows, cols, strict=True):
        major = rng.uniform(*spec.sigma_major_m)
        minor = min(rng.uniform(*spec.sigma_minor_m), major)
        hotspots.append(
            Hotspot(
                x=raster.origin_x + (col + rng.random()) * raster.tile_size_m,
                y=raster.origin_y + (row + rng.random()) * raster.tile_size_m,
                sigma_major_m=major,
                sigma_minor_m=minor,
                rotation_rad=rng.uniform(0.0, math.pi),
            )
        )
    return tuple(hotspots)


def field(raster: Raster, spec: DensitySpec, seed: int) -> DensityField:
    """Build the mixture the UE positions are drawn from.

    Weights carry the density function alone. They are deliberately NOT scaled
    by a tile's open area: ``free_fraction`` is a sub-sampled estimate, and on
    a tile holding only a sliver of open ground it overstates the truth by more
    than an order of magnitude. Baking it in here would hand those tiles that
    same factor in UE weight.

    Instead the open area enters through rejection in
    :func:`src.simulation.sample.sample_positions`, which draws uniformly
    inside a chosen tile and keeps only what lands on open ground. That makes
    the sampled density proportional to the density function times the tile's
    *true* open area, with the estimate used for nothing but eligibility.

    Tiles with no open ground at all are excluded, since no position in one
    could ever be accepted.
    """
    rng = np.random.default_rng(seed)
    hotspots = draw_hotspots(raster, spec, rng)

    eligible = eligible_tiles(raster).ravel().astype(np.float64)
    centre_x, centre_y = raster.tile_centres()
    centre_x = centre_x.ravel()
    centre_y = centre_y.ravel()

    weights = [_normalise(eligible)]
    for hotspot in hotspots:
        weights.append(_normalise(_gaussian(centre_x, centre_y, hotspot) * eligible))

    return DensityField(tile_weights=np.vstack(weights), hotspots=hotspots)


def _above_threshold(candidate_weights: np.ndarray, spec: DensitySpec) -> np.ndarray:
    """Zero the candidate tiles carrying too little surrounding building volume.

    The cutoff is a share of the densest candidate rather than an absolute
    volume, so one setting carries across scenes of differing build density and
    across a change to ``spec.built_volume_radius_m``.

    Raises:
        ValueError: When the cutoff leaves fewer candidates than
            ``spec.n_hotspots``.
    """
    cutoff = spec.min_built_volume_fraction * candidate_weights.max()
    kept = np.where(candidate_weights >= cutoff, candidate_weights, 0.0)

    n_kept = int(np.count_nonzero(kept))
    if n_kept < spec.n_hotspots:
        raise ValueError(
            "simulation.density.min_built_volume_fraction of "
            f"{spec.min_built_volume_fraction} leaves only {n_kept} candidate tiles for "
            f"{spec.n_hotspots} hotspots. Lower it, or raise "
            "simulation.density.built_volume_radius_m."
        )
    return kept


def _gaussian(x: np.ndarray, y: np.ndarray, hotspot: Hotspot) -> np.ndarray:
    """Evaluate one rotated elliptical Gaussian, unnormalised, at each point."""
    cos_rotation = math.cos(hotspot.rotation_rad)
    sin_rotation = math.sin(hotspot.rotation_rad)
    d_x = x - hotspot.x
    d_y = y - hotspot.y
    major = cos_rotation * d_x + sin_rotation * d_y
    minor = -sin_rotation * d_x + cos_rotation * d_y
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
