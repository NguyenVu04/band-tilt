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
    best_by_objective,
    objective,
)


@pytest.fixture
def cfg():
    """The three thresholds the objective reads. It reads nothing else."""
    return OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0, "overlap_margin_db": 6.0}}
    )


def _kpi(**overrides: float) -> KpiVector:
    """A middling measurement, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "overlap_neighbor_mean": 0.45,
        "weak_rate": 0.10,
        "rsrp_p05_dbm": -105.0,
        "rsrp_p50_dbm": -95.0,
        "sinr_p05_db": -3.0,
        "sinr_p50_db": 8.0,
        "served_rate": 0.20,
        "load_imbalance": 0.44,
        "objective": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def _u(multiplicity: float) -> float:
    """The utility of a tile served by that many cells: ``lambda e^(1 - lambda)``."""
    return multiplicity * math.exp(1.0 - multiplicity)


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def _score(rsrp: np.ndarray, cfg) -> float:
    """Score a map built by :func:`_map`."""
    return objective(rsrp, cfg)


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == (
        "hole_rate",
        "overlap_rate",
        "overlap_neighbor_mean",
        "weak_rate",
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "served_rate",
        "load_imbalance",
    )
    assert MEASURE_NAMES == (*KPI_NAMES, "objective")


def test_only_the_quality_service_and_objective_measures_are_maximised() -> None:
    """The usual place a sign error hides: the load measures are minimised."""
    assert MAXIMISED == {
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "served_rate",
        "objective",
    }


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


# --- lambda, the count the objective scores ---------------------------------
#
# Every RSRP below is at or above kpi.weak_dbm, so the strength factor is 1 and
# these fixtures isolate the multiplicity. Strength has its own tests further on.


def test_one_dominant_cell_scores_the_maximum(cfg) -> None:
    """The whole point of the objective: exactly one strong server is worth 1.0."""
    assert _score(_map([[-90.0, -130.0]]), cfg) == pytest.approx(1.0)


def test_a_second_cell_inside_the_margin_costs_a_quarter(cfg) -> None:
    """A neighbour 4 dB down is inside the 6 dB margin; one 7 dB down is not."""
    assert _score(_map([[-80.0, -84.0]]), cfg) == pytest.approx(_u(2.0))
    assert _score(_map([[-80.0, -87.0]]), cfg) == pytest.approx(1.0)


def test_a_third_cell_costs_more_than_the_second(cfg) -> None:
    """The utility keeps falling, so the search never trades one crowd for a worse one."""
    two = _score(_map([[-80.0, -84.0, -130.0]]), cfg)
    three = _score(_map([[-80.0, -84.0, -85.0]]), cfg)
    assert three == pytest.approx(_u(3.0))
    assert three < two < 1.0


def test_the_margin_is_inclusive(cfg) -> None:
    """A neighbour exactly 6 dB down overlaps; 6.01 dB down does not."""
    assert _score(_map([[-80.0, -86.0]]), cfg) == pytest.approx(_u(2.0))
    assert _score(_map([[-80.0, -86.01]]), cfg) == pytest.approx(1.0)


def test_a_neighbour_below_the_hole_threshold_does_not_count(cfg) -> None:
    """Within the margin but unusable: -122 dBm serves nobody, so it crowds nobody."""
    marginal = _map([[-118.0, -122.0]])
    assert _score(marginal, cfg) == pytest.approx(_u(1.0) * (120.0 - 118.0) / 30.0)


def test_the_best_band_is_scored_not_the_preferred_one(cfg) -> None:
    """Strength and cleanliness pick the layer, not kpi.capacity.band_preference.

    'hi' is barely covered and alone; 'lo' is 49 dB stronger and crowded. ADR 0009
    scored 'hi' because it is preferred and covered, which is the asymmetry that
    let a tilt pay for killing a layer. ADR 0010 takes whichever scores higher.
    """
    rsrp = _map([[-119.0, -130.0], [-70.0, -71.0]])
    assert _score(rsrp, cfg) == pytest.approx(_u(2.0))


def test_a_band_that_goes_dark_leaves_the_other_untouched(cfg) -> None:
    """With 'hi' below the threshold the tile still has 'lo', crowding and all."""
    assert _score(_map([[-130.0, -130.0], [-70.0, -71.0]]), cfg) == pytest.approx(_u(2.0))


def test_a_tile_no_band_covers_scores_zero(cfg) -> None:
    """A hole is worth nothing, however close to the threshold it comes."""
    assert _score(_map([[-130.0], [-121.0]]), cfg) == pytest.approx(0.0)


def test_a_no_path_tile_scores_zero(cfg) -> None:
    """No path is -inf, which is a hole on its own."""
    assert _score(_map([[np.nan, np.nan]]), cfg) == pytest.approx(0.0)


def test_the_objective_is_bounded_by_one(cfg) -> None:
    """Every tile served strongly by exactly one cell is the best a map can do."""
    rsrp = np.array([[[[-90.0, -85.0], [-88.0, -80.0]]]], dtype=float)
    rsrp = np.concatenate([rsrp, np.full_like(rsrp, -130.0)], axis=1)
    assert _score(rsrp, cfg) == pytest.approx(1.0)


# --- what ADR 0010 changed --------------------------------------------------


def test_losing_a_band_never_raises_the_objective(cfg) -> None:
    """A tilt must not buy J by destroying coverage.

    Under the ADR 0009 objective this map scored on the preferred band alone, so
    stripping a crowded 'hi' layer moved the tile onto a clean 'lo' and paid for
    it. Scoring the best band instead makes the loss of a layer weakly negative.
    """
    crowded = _map([[-80.0, -82.0, -84.0], [-118.0, -130.0, -130.0]])
    stripped = crowded.copy()
    stripped[0] = -130.0
    assert _score(stripped, cfg) <= _score(crowded, cfg)


def test_a_marginal_server_scores_far_below_a_strong_one(cfg) -> None:
    """Strength between the hole and weak thresholds counts, which it did not before."""
    assert _score(_map([[-119.7]]), cfg) == pytest.approx(0.01)
    assert _score(_map([[-105.0]]), cfg) == pytest.approx(0.5)
    assert _score(_map([[-90.0]]), cfg) == pytest.approx(1.0)


# --- selection -------------------------------------------------------------


def test_the_pick_ignores_every_reported_kpi() -> None:
    """A better rate with a lower objective does not win."""
    moved = _kpi(hole_rate=0.0, overlap_rate=0.0, served_rate=1.0, objective=0.49)
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
