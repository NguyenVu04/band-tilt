"""The time axis: interval count, mixture masses, and their reproducibility.

Expected values are hand-computed from the model, never read back from the
implementation. Where the mass is exactly determined -- a flat profile with no
AR(1) noise -- it is asserted as a literal.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.traffic import TrafficSpec, build

# A flat, noiseless schedule: every interval reduces to the nominal mixture, so
# its masses are exactly `hotspot_mass_fraction` split evenly.
FLAT = TrafficSpec(
    interval_s=900.0,
    horizon_s=3600.0,
    diurnal_amplitude=0.0,
    ar1_rho=0.0,
    ar1_sigma=0.0,
)


def test_interval_count_is_horizon_over_interval() -> None:
    """A 24 h horizon in 15 min intervals is 96 snapshots."""
    assert TrafficSpec(900.0, 86400.0, 0.6, 0.85, 0.25).n_intervals == 96


def test_trailing_part_interval_is_dropped() -> None:
    """3500 s holds three whole 900 s intervals; the leftover 800 s is not one."""
    assert TrafficSpec(900.0, 3500.0, 0.0, 0.0, 0.0).n_intervals == 3


def test_flat_schedule_splits_the_configured_mass_evenly() -> None:
    """With no swing and no noise every interval is the nominal mixture.

    Four hotspots sharing 0.8 take 0.2 each, and the background keeps the
    remaining 0.2. This is the only configuration where the mass is exact
    per interval rather than in expectation.
    """
    schedule = build(FLAT, (100, 200), n_hotspots=4, hotspot_mass_fraction=0.8, seed=1)

    assert schedule.component_mass.shape == (4, 5)
    np.testing.assert_allclose(schedule.component_mass[:, 0], 0.2)
    np.testing.assert_allclose(schedule.component_mass[:, 1:], 0.8 / 4)


def test_masses_sum_to_one() -> None:
    """Each row is a distribution, so `rng.choice` can draw a component from it."""
    schedule = build(
        TrafficSpec(900.0, 86400.0, 0.6, 0.85, 0.25),
        (100, 200),
        n_hotspots=6,
        hotspot_mass_fraction=0.7,
        seed=5,
    )
    np.testing.assert_allclose(schedule.component_mass.sum(axis=1), 1.0)


def test_counts_lie_in_the_configured_range() -> None:
    """The per-interval count is drawn from the inclusive range and nowhere else."""
    schedule = build(FLAT, (100, 200), n_hotspots=2, hotspot_mass_fraction=0.5, seed=3)

    assert schedule.count.min() >= 100
    assert schedule.count.max() <= 200
    assert schedule.n_ue == int(schedule.count.sum())


def test_demand_moves_between_intervals_once_it_is_noisy() -> None:
    """A noisy schedule is not the configured fraction repeated 96 times."""
    schedule = build(
        TrafficSpec(900.0, 86400.0, 0.6, 0.85, 0.25),
        (100, 200),
        n_hotspots=6,
        hotspot_mass_fraction=0.7,
        seed=7,
    )
    hotspot_mass = schedule.component_mass[:, 1:].sum(axis=1)

    assert hotspot_mass.std() > 0.01


def test_same_seed_gives_the_same_schedule() -> None:
    """The schedule is a pure function of its seed, so a rerun reproduces it."""
    spec = TrafficSpec(900.0, 86400.0, 0.6, 0.85, 0.25)
    first = build(spec, (100, 200), 6, 0.7, seed=11)
    second = build(spec, (100, 200), 6, 0.7, seed=11)

    np.testing.assert_array_equal(first.count, second.count)
    np.testing.assert_array_equal(first.component_mass, second.component_mass)


def test_no_hotspots_puts_everything_in_the_background() -> None:
    """Without hotspots the mixture is one component holding all the mass."""
    schedule = build(FLAT, (100, 200), n_hotspots=0, hotspot_mass_fraction=0.0, seed=1)

    assert schedule.component_mass.shape == (4, 1)
    np.testing.assert_allclose(schedule.component_mass, 1.0)


@pytest.mark.parametrize(
    "spec",
    [
        (0.0, 3600.0, 0.0, 0.0, 0.0),  # interval not positive
        (900.0, 400.0, 0.0, 0.0, 0.0),  # horizon shorter than one interval
        (900.0, 3600.0, 1.0, 0.0, 0.0),  # amplitude would zero an intensity
        (900.0, 3600.0, 0.0, 1.0, 0.0),  # rho makes the process non-stationary
        (900.0, 3600.0, 0.0, 0.0, -1.0),  # negative innovation std
    ],
)
def test_unbuildable_specs_are_rejected(spec: tuple[float, ...]) -> None:
    """A spec that would break stationarity or positivity fails at construction."""
    with pytest.raises(ValueError):
        TrafficSpec(*spec)


def test_inverted_count_range_is_rejected() -> None:
    """A range that cannot be drawn from is rejected rather than silently swapped."""
    with pytest.raises(ValueError):
        build(FLAT, (200, 100), 2, 0.5, seed=1)
