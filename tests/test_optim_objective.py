"""The KPI vector, the sign convention, and the lexicographic pick."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    KpiVector,
    as_maximised,
    ax_objective,
    lexicographic_best,
    pareto_mask,
    tolerances,
)

# Deliberately round and unequal, so a test cannot pass by comparing the wrong
# KPI against the right tolerance.
_TOLERANCE = {
    "hole_rate": 0.01,
    "overlap_rate": 0.01,
    "mean_overlap_neighbors": 0.10,
    "band_priority_score": 0.01,
    "weak_rate": 0.01,
}


@pytest.fixture
def cfg():
    """A config carrying only what the objective reads."""
    return OmegaConf.create({"kpi": {"tolerance": dict(_TOLERANCE)}})


def _kpi(**overrides: float) -> KpiVector:
    """A middling KPI vector, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "mean_overlap_neighbors": 0.70,
        "band_priority_score": 0.20,
        "weak_rate": 0.10,
    }
    return KpiVector(**{**values, **overrides})


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which configuration wins."""
    assert KPI_NAMES == (
        "hole_rate",
        "overlap_rate",
        "mean_overlap_neighbors",
        "band_priority_score",
        "weak_rate",
    )


def test_band_priority_score_is_the_only_maximised_kpi() -> None:
    """The usual place a sign error hides."""
    assert MAXIMISED == {"band_priority_score"}


def test_ax_objective_signs_every_kpi() -> None:
    """Minus on the four minimised, bare on the one maximised."""
    assert ax_objective() == (
        "-hole_rate, -overlap_rate, -mean_overlap_neighbors, band_priority_score, -weak_rate"
    )


def test_as_maximised_flips_only_the_minimised_kpis() -> None:
    """An orientation, not a normalisation: magnitudes are untouched."""
    values = as_maximised([_kpi()])
    assert np.array_equal(values[0], [-0.10, -0.30, -0.70, 0.20, -0.10])


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape handed to Ax and read back out of it."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_kpi() -> None:
    """An incomplete measurement must not become a silent zero."""
    values = _kpi().as_dict()
    del values["weak_rate"]
    with pytest.raises(KeyError, match="weak_rate"):
        KpiVector.from_mapping(values)


def test_a_decisive_gain_on_the_top_kpi_wins(cfg) -> None:
    """Hole rate outranks everything below it."""
    incumbent = _kpi()
    candidate = _kpi(hole_rate=0.05, overlap_rate=0.99, band_priority_score=0.0, weak_rate=0.99)
    assert lexicographic_best([incumbent, candidate], cfg) == 1


def test_a_decisive_loss_on_the_top_kpi_loses(cfg) -> None:
    """No amount of lower-priority gain buys a worse hole rate."""
    incumbent = _kpi()
    candidate = _kpi(hole_rate=0.20, overlap_rate=0.0, band_priority_score=1.0, weak_rate=0.0)
    assert lexicographic_best([incumbent, candidate], cfg) == 0


def test_a_tie_within_tolerance_falls_through_to_the_next_kpi(cfg) -> None:
    """The behaviour the tolerances exist to produce.

    Without them, hole rate is continuous, never ties, and the priority order
    collapses to optimizing it alone.
    """
    incumbent = _kpi()
    candidate = _kpi(hole_rate=0.105, overlap_rate=0.20)
    assert lexicographic_best([incumbent, candidate], cfg) == 1


def test_differences_within_every_tolerance_keep_the_incumbent(cfg) -> None:
    """Solver noise must not be able to unseat a deployed configuration."""
    incumbent = _kpi()
    noise = _kpi(hole_rate=0.105, overlap_rate=0.305, mean_overlap_neighbors=0.75)
    assert lexicographic_best([incumbent, noise], cfg) == 0


def test_the_pick_is_order_dependent_because_ties_are_not_transitive(cfg) -> None:
    """Three configurations where the winner depends on the order they arrive.

    ``a`` ties ``b`` on hole rate and loses to it on overlap; ``b`` ties ``c``
    and loses to it on overlap; but ``a`` beats ``c`` on hole rate outright,
    because the two 0.008 steps that were each a tie sum to 0.016, which is not.

    Walking forwards the chain carries the winner down to ``c``; walking
    backwards, ``c`` survives ``b`` and then loses to ``a``. Both are correct
    single passes. This is why the docstring forbids sorting: a sort may
    compare any pair it likes, and there is no ordering to find.
    """
    a = _kpi(hole_rate=0.100, overlap_rate=0.30)
    b = _kpi(hole_rate=0.108, overlap_rate=0.20)
    c = _kpi(hole_rate=0.116, overlap_rate=0.10)

    forwards = [a, b, c]
    backwards = [c, b, a]
    assert forwards[lexicographic_best(forwards, cfg)] == c
    assert backwards[lexicographic_best(backwards, cfg)] == a


def test_pareto_mask_drops_only_dominated_points() -> None:
    """Better on one KPI and no worse on the rest keeps a point on the front."""
    best_hole = _kpi(hole_rate=0.01)
    best_bps = _kpi(band_priority_score=0.90)
    dominated = _kpi(hole_rate=0.99, overlap_rate=0.99, band_priority_score=0.0)
    mask = pareto_mask([best_hole, best_bps, dominated])
    assert mask.tolist() == [True, True, False]


def test_duplicate_points_both_stay_on_the_front() -> None:
    """Equality is not domination; dropping either would be arbitrary."""
    assert pareto_mask([_kpi(), _kpi()]).tolist() == [True, True]


def test_tolerances_are_read_in_priority_order(cfg) -> None:
    """Misalignment here would compare each KPI against another's threshold."""
    assert np.array_equal(tolerances(cfg), [0.01, 0.01, 0.10, 0.01, 0.01])


def test_a_missing_tolerance_block_raises_rather_than_defaulting() -> None:
    """An exact comparison never ties, which silently voids the priority."""
    with pytest.raises(ValueError, match="tolerance"):
        tolerances(OmegaConf.create({"kpi": {}}))


def test_an_incomplete_tolerance_block_names_the_gap() -> None:
    """Half a tolerance block is more dangerous than none."""
    partial = {name: 0.01 for name in KPI_NAMES if name != "weak_rate"}
    with pytest.raises(ValueError, match="weak_rate"):
        tolerances(OmegaConf.create({"kpi": {"tolerance": partial}}))


def test_choosing_from_nothing_raises(cfg) -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        lexicographic_best([], cfg)


def test_hypervolume_credits_only_improvement_over_the_reference() -> None:
    """Anchored on the incumbent, so a worse configuration is worth nothing."""
    from src.optim.objective import hypervolume

    incumbent = _kpi()
    worse = _kpi(hole_rate=0.50, overlap_rate=0.90, band_priority_score=0.0)
    better = _kpi(
        hole_rate=0.05,
        overlap_rate=0.20,
        mean_overlap_neighbors=0.60,
        band_priority_score=0.40,
        weak_rate=0.05,
    )

    assert hypervolume([incumbent], incumbent) == 0.0
    assert hypervolume([worse], incumbent) == 0.0
    assert hypervolume([better], incumbent) > 0.0
    # A dominated point adds nothing to a front that already covers it.
    assert hypervolume([better, worse], incumbent) == hypervolume([better], incumbent)


def test_hypervolume_trace_never_decreases() -> None:
    """Each prefix is a subset of the next, so progress cannot go backwards."""
    from src.optim.objective import hypervolume_trace

    incumbent = _kpi()
    trace = hypervolume_trace(
        [incumbent, _kpi(hole_rate=0.5), _kpi(hole_rate=0.05, band_priority_score=0.4)],
        incumbent,
    )
    assert len(trace) == 3
    assert np.all(np.diff(trace) >= 0.0)
