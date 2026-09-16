"""The KPI vector, the sign convention, and the score that picks a winner."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    REPORT_NAMES,
    TARGET_NAMES,
    KpiVector,
    best_by_score,
    desirability,
    quality_index,
    weights,
)

# Unequal and out of priority order on purpose, so a test cannot pass by
# reading the block's own order instead of TARGET_NAMES.
_WEIGHTS = {
    "hole_desirability": 4.0,
    "served_desirability": 2.0,
    "overlap_desirability": 3.0,
}


@pytest.fixture
def cfg():
    """The only config the objective reads."""
    return OmegaConf.create({"kpi": {"weights": dict(_WEIGHTS)}})


def _kpi(**overrides: float) -> KpiVector:
    """A middling KPI vector, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "served_ratio": 0.20,
        "weak_rate": 0.10,
        "edge_rsrp_dbm": -105.0,
        "hole_desirability": 0.50,
        "overlap_desirability": 0.50,
        "served_desirability": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == (
        "hole_rate",
        "overlap_rate",
        "served_ratio",
        "weak_rate",
        "edge_rsrp_dbm",
        "hole_desirability",
        "overlap_desirability",
        "served_desirability",
    )
    assert KPI_NAMES == REPORT_NAMES + TARGET_NAMES


def test_the_two_families_do_not_overlap() -> None:
    """The separation of ADR 0005: no KPI is both reported and searched for."""
    assert not set(REPORT_NAMES) & set(TARGET_NAMES)


def test_only_the_ue_and_targeted_kpis_are_maximised() -> None:
    """The usual place a sign error hides."""
    assert MAXIMISED == {
        "served_ratio",
        "edge_rsrp_dbm",
        "hole_desirability",
        "overlap_desirability",
        "served_desirability",
    }


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


def test_weights_are_read_in_target_order_and_normalised(cfg) -> None:
    """The fixture lists served before overlap; the score must not."""
    assert weights(cfg) == pytest.approx(np.array([4.0, 3.0, 2.0]) / 9.0)
    assert weights(cfg).sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (None, "weights"),
        ({name: 1.0 for name in TARGET_NAMES if name != "served_desirability"}, "served"),
        ({**dict.fromkeys(TARGET_NAMES, 1.0), "overlap_desirability": -1.0}, "non-negative"),
        (dict.fromkeys(TARGET_NAMES, 0.0), "all zero"),
    ],
)
def test_unusable_weights_raise(block, match) -> None:
    """A missing, partial, negative or all-zero block cannot rank anything."""
    kpi = {} if block is None else {"weights": block}
    with pytest.raises(ValueError, match=match):
        weights(OmegaConf.create({"kpi": kpi}))


def test_a_reported_rate_carries_no_weight(cfg) -> None:
    """The reported family is not in `kpi.weights`, so it cannot be weighted in."""
    assert not set(REPORT_NAMES) & set(cfg.kpi.weights)


def test_desirability_reads_the_premeasured_columns(cfg) -> None:
    """The objective applies no threshold: every column was softened in src.kpi."""
    kpi = _kpi(hole_desirability=0.11, overlap_desirability=0.22, served_desirability=0.33)
    assert desirability([kpi])[0] == pytest.approx([0.11, 0.22, 0.33])


def test_the_score_ignores_every_reported_kpi(cfg) -> None:
    """Moving a rate must not move the objective; they are separate metrics."""
    moved = _kpi(hole_rate=0.99, overlap_rate=0.99, served_ratio=0.01, edge_rsrp_dbm=-60.0)
    assert desirability([moved])[0] == pytest.approx(desirability([_kpi()])[0])
    assert quality_index([moved], cfg)[0] == pytest.approx(quality_index([_kpi()], cfg)[0])


def test_the_quality_index_is_the_weighted_geometric_mean(cfg) -> None:
    """Equal desirabilities give that value back, whatever the weights."""
    assert quality_index([_kpi()], cfg)[0] == pytest.approx(0.5)


def test_the_quality_index_cannot_be_bought_back(cfg) -> None:
    """Non-compensatory: perfect on two KPIs does not rescue a collapsed third.

    The contrast with the arithmetic mean of the same desirabilities is the
    point of the geometric form, so both are asserted here.
    """
    collapsed = _kpi(hole_desirability=1e-6, overlap_desirability=1.0, served_desirability=1.0)
    index = quality_index([collapsed], cfg)[0]
    compensated = float(desirability([collapsed])[0] @ weights(cfg))

    assert index < 0.01
    assert index < quality_index([_kpi()], cfg)[0]
    assert compensated > 0.5


def test_the_highest_score_wins(cfg) -> None:
    """A desirability gain with nothing lost elsewhere raises the score."""
    assert best_by_score([_kpi(), _kpi(served_desirability=0.60)], cfg) == 1


def test_a_heavier_weight_outvotes_a_lighter_gain(cfg) -> None:
    """Coverage is weighted 4 to the served ratio's 2, so an equal swap loses."""
    candidate = _kpi(hole_desirability=0.40, served_desirability=0.60)
    assert best_by_score([_kpi(), candidate], cfg) == 0


def test_an_equal_score_keeps_the_earlier_configuration(cfg) -> None:
    """The incumbent holds unless a candidate actually scores higher."""
    assert best_by_score([_kpi(), _kpi()], cfg) == 0


def test_choosing_from_nothing_raises(cfg) -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        best_by_score([], cfg)
