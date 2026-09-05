"""Build time-varying UE mixture masses for independent snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

# Diurnal-profile period.
_DAY_S = 86400.0


@dataclass(frozen=True)
class TrafficSpec:
    """How long the scenario runs, and how demand varies over it.

    Attributes:
        interval_s: Seconds between snapshots.
        horizon_s: Total simulated span. A trailing part-interval is dropped.
        diurnal_amplitude: Depth of the day/night swing, in ``[0, 1)``. Zero
            leaves the profile flat; approaching one drives a hotspot's
            intensity to zero at its trough.
        ar1_rho: Correlation between one interval's log intensity and the next,
            in ``[0, 1)``. Zero makes intervals independent given the profile.
        ar1_sigma: Standard deviation of the AR(1) innovation, in log
            intensity. Zero leaves the bare diurnal profile.
    """

    interval_s: float
    horizon_s: float
    diurnal_amplitude: float
    ar1_rho: float
    ar1_sigma: float

    def __post_init__(self) -> None:
        """Reject a schedule that cannot be built.

        Raises:
            ValueError: When the interval or horizon is not positive, when the
                horizon is shorter than one interval, or when the amplitude or
                correlation is outside the range that keeps intensities
                positive and the process stationary.
        """
        if self.interval_s <= 0:
            raise ValueError(f"simulation.time.interval_s must be positive, got {self.interval_s}")
        if self.horizon_s < self.interval_s:
            raise ValueError(
                f"simulation.time.horizon_s {self.horizon_s} is shorter than one "
                f"interval of {self.interval_s} s, so no snapshot would be taken"
            )
        if not 0.0 <= self.diurnal_amplitude < 1.0:
            raise ValueError(
                "simulation.time.diurnal_amplitude must be in [0, 1) to keep every "
                f"intensity positive, got {self.diurnal_amplitude}"
            )
        if not 0.0 <= self.ar1_rho < 1.0:
            raise ValueError(
                "simulation.time.ar1_rho must be in [0, 1) for the process to be "
                f"stationary, got {self.ar1_rho}"
            )
        if self.ar1_sigma < 0:
            raise ValueError(
                f"simulation.time.ar1_sigma must not be negative, got {self.ar1_sigma}"
            )

    @property
    def n_intervals(self) -> int:
        """How many whole intervals fit in the horizon."""
        return int(math.floor(self.horizon_s / self.interval_s + 1e-9))

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TrafficSpec:
        """Read ``simulation.time``."""
        time = cfg.simulation.time
        return cls(
            interval_s=float(time.interval_s),
            horizon_s=float(time.horizon_s),
            diurnal_amplitude=float(time.diurnal_amplitude),
            ar1_rho=float(time.ar1_rho),
            ar1_sigma=float(time.ar1_sigma),
        )


@dataclass(frozen=True)
class Schedule:
    """The snapshots, and the mixture each one draws from.

    Attributes:
        interval_s: Seconds between snapshots.
        t_s: Start time of each interval, shaped ``[n_intervals]``.
        count: UEs to draw in each interval, shaped ``[n_intervals]``.
        component_mass: Share of the interval's UEs per mixture component,
            shaped ``[n_intervals, n_components]``, each row summing to 1.
            Index 0 is the uniform background; the rest are the hotspots, in
            the order :class:`src.simulation.density.DensityField` holds them.
        phase_rad: The hour each hotspot peaks at, as an angle, shaped
            ``[n_hotspots]``. Recorded so the manifest can describe the
            schedule rather than only its outcome.
    """

    interval_s: float
    t_s: np.ndarray
    count: np.ndarray
    component_mass: np.ndarray
    phase_rad: np.ndarray

    @property
    def n_intervals(self) -> int:
        """Number of snapshots."""
        return int(self.t_s.size)

    @property
    def t_index(self) -> np.ndarray:
        """Index of each interval, ``0 .. n_intervals - 1``."""
        return np.arange(self.n_intervals, dtype=np.int64)

    @property
    def n_ue(self) -> int:
        """Total UEs across every interval."""
        return int(self.count.sum())


def build(
    spec: TrafficSpec,
    count_range: tuple[int, int],
    n_hotspots: int,
    hotspot_mass_fraction: float,
    seed: int,
) -> Schedule:
    """Build the snapshot schedule and the per-interval mixture masses.

    ``hotspot_mass_fraction`` is the share of the population the hotspots hold
    *on average*. It is honoured in expectation rather than per interval: both
    the diurnal profile and the AR(1) term are scaled to unit mean, and the
    background is given the intensity that reproduces the requested share when
    every hotspot sits at that mean. An individual interval is then free to be
    more or less concentrated than the configured value, which is the point of
    having a time axis at all.

    Raises:
        ValueError: When ``count_range`` is inverted or not positive.
    """
    low, high = count_range
    if low <= 0 or high < low:
        raise ValueError(
            f"simulation.ue.count_range must be a positive non-decreasing pair, got {count_range}"
        )

    rng = np.random.default_rng(seed)
    n_t = spec.n_intervals
    t_s = np.arange(n_t, dtype=np.float64) * spec.interval_s
    count = rng.integers(low, high + 1, size=n_t, dtype=np.int64)

    if n_hotspots == 0 or hotspot_mass_fraction <= 0.0:
        return Schedule(
            interval_s=spec.interval_s,
            t_s=t_s,
            count=count,
            component_mass=_all_background(n_t, n_hotspots),
            phase_rad=np.zeros(n_hotspots, dtype=np.float64),
        )

    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_hotspots)
    profile = 1.0 + spec.diurnal_amplitude * np.sin(
        2.0 * math.pi * t_s[:, None] / _DAY_S - phase[None, :]
    )
    intensity = profile * _ar1_unit_mean(n_t, n_hotspots, spec, rng)

    # Set background intensity to the requested mean hotspot share.
    background = n_hotspots * (1.0 - hotspot_mass_fraction) / hotspot_mass_fraction
    unnormalised = np.concatenate([np.full((n_t, 1), background), intensity], axis=1)

    return Schedule(
        interval_s=spec.interval_s,
        t_s=t_s,
        count=count,
        component_mass=unnormalised / unnormalised.sum(axis=1, keepdims=True),
        phase_rad=phase,
    )


def _all_background(n_t: int, n_hotspots: int) -> np.ndarray:
    """Mixture masses for a schedule whose hotspots carry nothing."""
    mass = np.zeros((n_t, 1 + n_hotspots), dtype=np.float64)
    mass[:, 0] = 1.0
    return mass


def _ar1_unit_mean(
    n_t: int,
    n_hotspots: int,
    spec: TrafficSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """A stationary AR(1) in log intensity, exponentiated to have mean one.

    Started from the stationary distribution rather than from zero, so the
    first intervals are not systematically quieter than the rest. The
    half-variance term is the lognormal mean correction: without it a larger
    ``ar1_sigma`` would silently raise the average intensity, and with it the
    hotspots' configured share.
    """
    if spec.ar1_sigma == 0.0:
        return np.ones((n_t, n_hotspots), dtype=np.float64)

    stationary_var = spec.ar1_sigma**2 / (1.0 - spec.ar1_rho**2)
    log_intensity = np.empty((n_t, n_hotspots), dtype=np.float64)
    log_intensity[0] = rng.normal(0.0, math.sqrt(stationary_var), size=n_hotspots)
    for index in range(1, n_t):
        log_intensity[index] = spec.ar1_rho * log_intensity[index - 1] + rng.normal(
            0.0, spec.ar1_sigma, size=n_hotspots
        )
    return np.exp(log_intensity - 0.5 * stationary_var)
