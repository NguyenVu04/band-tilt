"""Tests for the shared search space.

:mod:`src.optim.space` is what makes the comparison in PROJECT.md section 17
valid: BO and MARL must search the same set, or the experiment measures two
search spaces rather than two methods. These tests pin down the properties both
optimizers rely on.

The round trip is the one that matters
--------------------------------------
Optimizers work in the unit cube; the simulator and the report work in degrees.
An asymmetry between :meth:`to_unit` and :meth:`from_unit` means the
configuration that gets evaluated is not the configuration the optimizer
proposed — and the reported result belongs to neither.

Every test is skipped until ``src/optim/space.py`` is implemented.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from src.optim.space import TiltSpace
from src.radio import sampling

pytestmark = pytest.mark.skip(reason="implement src/optim/space.py first")


def test_dimension_is_cells_times_bands(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """One decision variable per cell-band pair — PROJECT.md section 3."""
    assert TiltSpace(cell_bands, cfg).n_dims == len(cell_bands)


def test_unit_round_trip_is_exact(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """``from_unit(to_unit(tilt)) == tilt``.

    The single most consequential property in this module.
    """
    space = TiltSpace(cell_bands, cfg)
    tilt = np.array([1.0, 4.5, 7.0, 9.25, 12.0, 14.0])
    assert space.from_unit(space.to_unit(tilt)) == pytest.approx(tilt)


def test_unit_bounds_map_to_the_corners(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """Zero maps to the lower bound and one to the upper bound, per dimension."""
    space = TiltSpace(cell_bands, cfg)
    assert space.from_unit(np.zeros(space.n_dims)) == pytest.approx(
        cell_bands["tilt_min"].to_numpy()
    )
    assert space.from_unit(np.ones(space.n_dims)) == pytest.approx(
        cell_bands["tilt_max"].to_numpy()
    )


def test_normalisation_is_per_dimension(cfg: DictConfig) -> None:
    """Each cell-band is scaled by its own bounds, not by a global range.

    Bands with different tilt ranges are the expected case, and one global
    scaling distorts the space — the GP then fits length scales against an axis
    geometry that does not match the problem.
    """
    table = pd.DataFrame(
        {
            "gcell_id": ["cell_a", "cell_a"],
            "band": ["high", "low"],
            "current_tilt": [5.0, 5.0],
            "tilt_min": [0.0, 0.0],
            "tilt_max": [10.0, 20.0],
        }
    )
    space = TiltSpace(table, cfg)
    assert space.to_unit(np.array([5.0, 5.0])) == pytest.approx([0.5, 0.25])


def test_unit_cube_maps_only_to_feasible_configurations(
    cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """Anything an optimizer proposes inside ``[0, 1]`` is deployable.

    This is the property that lets both optimizers search unconstrained: the
    parameterisation, not a downstream check, is what enforces the bounds.
    """
    space = TiltSpace(cell_bands, cfg)
    rng = np.random.default_rng(0)
    for tilt in space.from_unit(rng.random((32, space.n_dims))):
        sampling.assert_within_bounds(tilt, cell_bands)


def test_baseline_is_the_current_deployed_configuration(
    cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """The reference every reported improvement is measured against."""
    space = TiltSpace(cell_bands, cfg)
    assert space.baseline() == pytest.approx(cell_bands["current_tilt"].to_numpy())


def test_baseline_is_itself_feasible(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """The deployed network must lie inside the search space.

    If it does not, the bounds in ``configs/radio.yaml`` contradict the cell
    configuration, and every improvement is being measured against a
    configuration the optimizer was never allowed to consider.
    """
    space = TiltSpace(cell_bands, cfg)
    sampling.assert_within_bounds(space.baseline(), cell_bands)


def test_inverted_bounds_are_rejected(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """``tilt_min >= tilt_max`` is a config error and must fail at construction.

    Left alone it produces an empty or inverted interval, and normalisation then
    divides by zero or silently flips the direction of the axis.
    """
    broken = cell_bands.copy()
    broken.loc[0, "tilt_min"] = 20.0
    with pytest.raises(ValueError):
        TiltSpace(broken, cfg)


def test_sampled_configurations_are_all_feasible(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """Every configuration in the surrogate design must be deployable.

    An infeasible sample is not a hard case for the surrogate to learn — it is a
    network that cannot exist, and it spends ray-tracing budget teaching the
    surrogate about states no optimizer may propose.
    """
    for tilt in sampling.sample_configurations(cell_bands, cfg):
        sampling.assert_within_bounds(tilt, cell_bands)


def test_sampling_includes_the_baseline(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """The current network must be in the surrogate dataset.

    It is the reference point for every claim, and a surrogate that is
    inaccurate exactly there undermines every comparison built on it.
    """
    tilts = sampling.sample_configurations(cell_bands, cfg)
    baseline = cell_bands["current_tilt"].to_numpy()
    assert np.isclose(tilts, baseline).all(axis=1).any()


def test_sampling_is_deterministic(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """The same config and seed reproduce the same design.

    Each row costs a ray-tracing solve, so a non-reproducible design cannot be
    extended or resumed — the new samples would come from a different draw.
    """
    first = sampling.sample_configurations(cell_bands, cfg)
    second = sampling.sample_configurations(cell_bands, cfg)
    assert first == pytest.approx(second)
