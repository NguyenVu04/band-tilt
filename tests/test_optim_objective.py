"""The KPI vector, the sign convention, and the weighted score that picks a winner."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    KpiVector,
    as_maximised,
    best_by_score,
    scores,
    tolerances,
    weights,
)

# Deliberately round and unequal, so a test cannot pass by comparing the wrong
# KPI against the right tolerance. Listed out of priority order on purpose.
_TOLERANCE = {
    "hole_rate": 0.01,
    "overlap_rate": 0.02,
    "weak_rate": 0.05,
    "served_ratio": 0.04,
}

# Unequal and out of order for the same reason.
_WEIGHTS = {
    "hole_rate": 4.0,
    "overlap_rate": 3.0,
    "weak_rate": 1.0,
    "served_ratio": 2.0,
}


@pytest.fixture
def cfg():
    """A config carrying only what the objective reads."""
    return OmegaConf.create({"kpi": {"tolerance": dict(_TOLERANCE), "weights": dict(_WEIGHTS)}})


def _kpi(**overrides: float) -> KpiVector:
    """A middling KPI vector, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "served_ratio": 0.20,
        "weak_rate": 0.10,
    }
    return KpiVector(**{**values, **overrides})


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which configuration wins."""
    assert KPI_NAMES == ("hole_rate", "overlap_rate", "served_ratio", "weak_rate")


def test_only_the_ue_weighted_kpi_is_maximised() -> None:
    """The usual place a sign error hides."""
    assert MAXIMISED == {"served_ratio"}


def test_as_maximised_flips_only_the_minimised_kpis() -> None:
    """An orientation, not a normalisation: magnitudes are untouched."""
    values = as_maximised([_kpi()])
    assert np.array_equal(values[0], [-0.10, -0.30, 0.20, -0.10])


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape run.json records and the evaluation stage reads back."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_kpi() -> None:
    """An incomplete measurement must not become a silent zero."""
    values = _kpi().as_dict()
    del values["weak_rate"]
    with pytest.raises(KeyError, match="weak_rate"):
        KpiVector.from_mapping(values)


def test_weights_are_read_in_kpi_order(cfg) -> None:
    """The fixture lists weak before served ratio; the score must not."""
    assert np.array_equal(weights(cfg), [4.0, 3.0, 2.0, 1.0])


def test_the_score_is_the_signed_raw_weighted_sum(cfg) -> None:
    """Minus on the three minimised KPIs, plus on served ratio, no normalisation."""
    assert scores([_kpi()], cfg)[0] == pytest.approx(-4 * 0.10 - 3 * 0.30 + 2 * 0.20 - 1 * 0.10)


def test_the_highest_score_wins(cfg) -> None:
    """A served-ratio gain with nothing lost elsewhere raises the score."""
    assert best_by_score([_kpi(), _kpi(served_ratio=0.30)], cfg) == 1


def test_a_heavier_weight_outvotes_a_lighter_gain(cfg) -> None:
    """+0.02 served ratio (x2) does not pay for +0.02 hole rate (x4)."""
    candidate = _kpi(hole_rate=0.12, served_ratio=0.22)
    assert best_by_score([_kpi(), candidate], cfg) == 0


def test_an_equal_score_keeps_the_earlier_configuration(cfg) -> None:
    """The incumbent holds unless a candidate actually scores higher."""
    assert best_by_score([_kpi(), _kpi()], cfg) == 0


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (None, "weights"),
        ({name: 1.0 for name in KPI_NAMES if name != "weak_rate"}, "weak_rate"),
        ({**dict.fromkeys(KPI_NAMES, 1.0), "overlap_rate": -1.0}, "non-negative"),
        (dict.fromkeys(KPI_NAMES, 0.0), "all zero"),
    ],
)
def test_unusable_weights_raise(block, match) -> None:
    """A missing, partial, negative or all-zero block cannot rank anything."""
    kpi = {} if block is None else {"weights": block}
    with pytest.raises(ValueError, match=match):
        weights(OmegaConf.create({"kpi": kpi}))


def test_tolerances_are_read_in_priority_order(cfg) -> None:
    """Misalignment here would compare each KPI against another's threshold."""
    # The fixture lists weak before served ratio; KPI_NAMES does not. Reading
    # the dict's own order instead is exactly the misalignment guarded against.
    assert np.array_equal(tolerances(cfg), [0.01, 0.02, 0.04, 0.05])


def test_a_missing_tolerance_block_raises_rather_than_defaulting() -> None:
    """An exact comparison reports solver noise as a real change."""
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
        best_by_score([], cfg)
