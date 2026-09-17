"""The KPI vector, the sign convention, and the objective that picks a winner."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    MEASURE_NAMES,
    KpiVector,
    ObjectiveSpec,
    best_by_objective,
    objective,
)

_OBJECTIVE = {"tau_r_db": 10.0, "beta": 1.0}


@pytest.fixture
def cfg():
    """The objective's parameters and the KPI thresholds it shares."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "overlap_margin_db": 6.0,
                "objective": dict(_OBJECTIVE),
            },
        }
    )


def _kpi(**overrides: float) -> KpiVector:
    """A middling measurement, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "served_ratio": 0.20,
        "weak_rate": 0.10,
        "edge_rsrp_dbm": -105.0,
        "objective": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == ("hole_rate", "overlap_rate", "served_ratio", "weak_rate", "edge_rsrp_dbm")
    assert MEASURE_NAMES == (*KPI_NAMES, "objective")


def test_only_the_ue_edge_and_objective_measures_are_maximised() -> None:
    """The usual place a sign error hides."""
    assert MAXIMISED == {"served_ratio", "edge_rsrp_dbm", "objective"}


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape run.json records and the evaluation stage reads back."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_measure() -> None:
    """An incomplete measurement must not become a silent zero."""
    values = _kpi().as_dict()
    del values["objective"]
    with pytest.raises(KeyError, match="objective"):
        KpiVector.from_mapping(values)


@pytest.mark.parametrize(
    ("key", "value"),
    [("tau_r_db", 0.0), ("beta", -0.1)],
)
def test_unusable_objective_parameters_raise(cfg, key, value) -> None:
    """Each bound keeps the objective finite and inside [0, 1]."""
    cfg.kpi.objective[key] = value
    with pytest.raises(ValueError, match=key):
        ObjectiveSpec.from_config(cfg)


def test_objective_is_half_on_the_coverage_threshold(cfg) -> None:
    """At R_s = T_cov the sigmoid is a half, and a server not above T_cov has no neighbours."""
    assert objective(_map([[-120.0, -121.0]]), cfg) == pytest.approx(0.5)


def test_objective_discounts_each_co_band_neighbour_by_q_ov(cfg) -> None:
    """A neighbour 4 dB down is inside the 6 dB margin; one 7 dB down is not."""
    inside = objective(_map([[-100.0, -104.0]]), cfg)
    outside = objective(_map([[-100.0, -107.0]]), cfg)
    covered = 1.0 / (1.0 + math.exp(-2.0))
    assert inside == pytest.approx(covered * math.exp(-1.0))
    assert outside == pytest.approx(covered)


def test_objective_ignores_a_close_layer_on_another_band(cfg) -> None:
    """Overlap is co-band: the other band's cell 1 dB down is not a neighbour."""
    assert objective(_map([[-100.0], [-101.0]]), cfg) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_objective_counts_crowding_on_a_band_that_does_not_serve(cfg) -> None:
    """'hi' serves alone at -90 dBm; on 'lo' tx1 is 2 dB below tx0, so m = 1.

    The 'lo' neighbour is judged against the 'lo' server, not against R_s, which
    it is 12 dB below.
    """
    rsrp = _map([[-90.0, -130.0], [-100.0, -102.0]])
    assert objective(rsrp, cfg) == pytest.approx(1.0 / (1.0 + math.exp(-3.0)) * math.exp(-1.0))


def test_objective_scores_a_no_path_tile_zero(cfg) -> None:
    """No path is R_s = -inf: no coverage utility at all."""
    assert objective(_map([[np.nan, np.nan]]), cfg) == pytest.approx(0.0)


def test_the_pick_ignores_every_reported_kpi() -> None:
    """A better rate with a lower objective does not win."""
    moved = _kpi(hole_rate=0.0, overlap_rate=0.0, served_ratio=1.0, objective=0.49)
    assert best_by_objective([_kpi(), moved]) == 0


def test_the_highest_objective_wins() -> None:
    """A higher objective is picked."""
    assert best_by_objective([_kpi(), _kpi(objective=0.60)]) == 1


def test_an_equal_objective_keeps_the_earlier_configuration() -> None:
    """The incumbent holds unless a candidate actually scores higher."""
    assert best_by_objective([_kpi(), _kpi()]) == 0


def test_choosing_from_nothing_raises() -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        best_by_objective([])
