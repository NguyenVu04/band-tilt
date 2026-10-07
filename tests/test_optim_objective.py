"""The KPI vector, the sign convention, and the objective that picks a winner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.history import LocalRunWriter
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
    """The two thresholds the objective reads. It reads nothing else."""
    return OmegaConf.create({"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0}})


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
        "ue_service_failure_rate": 0.20,
        "estimated_throughput_p05_mbps": 1.0,
        "estimated_throughput_p50_mbps": 5.0,
        "estimated_throughput_mean_mbps": 6.0,
        "objective": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def _share(*relative_db: float) -> float:
    """The strongest sector's power share, rivals given in dB relative to it."""
    return 1.0 / (1.0 + sum(10.0 ** (db / 10.0) for db in relative_db))


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def _score(rsrp: np.ndarray, cfg) -> float:
    """Score a map built by :func:`_map`."""
    return objective(rsrp, cfg)


def test_reporting_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == (
        "hole_rate",
        "weak_rate",
        "overlap_rate",
        "overlap_neighbor_mean",
        "rsrp_p50_dbm",
        "rsrp_p05_dbm",
        "sinr_p50_db",
        "sinr_p05_db",
        "ue_service_failure_rate",
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
    )
    assert MEASURE_NAMES == (*KPI_NAMES, "objective")


def test_only_the_signal_quality_throughput_and_objective_measures_are_maximised() -> None:
    """The usual place a sign error hides: the rates are minimised."""
    assert MAXIMISED == {
        "rsrp_p50_dbm",
        "rsrp_p05_dbm",
        "sinr_p50_db",
        "sinr_p05_db",
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
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


# --- the power share the objective scores -------------------------------------
#
# Every RSRP below is at or above kpi.weak_dbm, so the strength factor is 1 and
# these fixtures isolate the share. Strength has its own tests further on.


def test_one_dominant_sector_scores_the_maximum(cfg) -> None:
    """The whole point of the objective: exactly one strong server is worth 1.0."""
    assert _score(_map([[-90.0, -130.0]]), cfg) == pytest.approx(1.0)


def test_an_equal_rival_halves_the_band(cfg) -> None:
    """Two sectors at the same power each hold half of it."""
    assert _score(_map([[-80.0, -80.0]]), cfg) == pytest.approx(0.5)


def test_a_rival_costs_in_proportion_to_its_power(cfg) -> None:
    """No margin: a rival 20 dB down still costs its 1 %, and a nearer one costs more."""
    near = _score(_map([[-80.0, -84.0]]), cfg)
    far = _score(_map([[-80.0, -100.0]]), cfg)
    assert near == pytest.approx(_share(-4.0), abs=5e-7)
    assert far == pytest.approx(_share(-20.0), abs=5e-7)
    assert near < far < 1.0


def test_a_third_sector_costs_more_than_the_second(cfg) -> None:
    """The utility keeps falling, so the search never trades one crowd for a worse one."""
    two = _score(_map([[-80.0, -84.0, -130.0]]), cfg)
    three = _score(_map([[-80.0, -84.0, -85.0]]), cfg)
    assert three == pytest.approx(_share(-4.0, -5.0), abs=5e-7)
    assert three < two < 1.0


def test_a_neighbour_below_the_hole_threshold_does_not_count(cfg) -> None:
    """-122 dBm serves nobody, so it crowds nobody."""
    marginal = _map([[-118.0, -122.0]])
    assert _score(marginal, cfg) == pytest.approx((120.0 - 118.0) / 30.0, abs=5e-7)


def test_bands_are_weighted_by_utility_not_preference(cfg) -> None:
    """Strength and cleanliness weight the layers, not a band order.

    'hi' is barely covered and alone; 'lo' is 49 dB stronger and crowded. The
    contraharmonic mean leans on 'lo', and the marginal 'hi' still pulls it down.
    """
    rsrp = _map([[-119.0, -130.0], [-70.0, -71.0]])
    hi, lo = 1.0 / 30.0, _share(-1.0)
    assert _score(rsrp, cfg) == pytest.approx((hi**2 + lo**2) / (hi + lo), abs=5e-7)


def test_a_band_that_goes_dark_leaves_the_other_untouched(cfg) -> None:
    """With 'hi' below the threshold the tile still has 'lo', crowding and all."""
    assert _score(_map([[-130.0, -130.0], [-70.0, -71.0]]), cfg) == pytest.approx(
        _share(-1.0), abs=5e-7
    )


def test_a_tile_no_band_covers_scores_zero(cfg) -> None:
    """A hole is worth nothing, however close to the threshold it comes."""
    assert _score(_map([[-130.0], [-121.0]]), cfg) == pytest.approx(0.0)


def test_a_no_path_tile_scores_zero(cfg) -> None:
    """No path is -inf, which is a hole on its own."""
    assert _score(_map([[np.nan, np.nan]]), cfg) == pytest.approx(0.0)


def test_the_objective_is_bounded_by_one(cfg) -> None:
    """Every tile served strongly by exactly one sector is the best a map can do."""
    rsrp = np.array([[[[-90.0, -85.0], [-88.0, -80.0]]]], dtype=float)
    rsrp = np.concatenate([rsrp, np.full_like(rsrp, -130.0)], axis=1)
    assert _score(rsrp, cfg) == pytest.approx(1.0)


# --- losing a layer ----------------------------------------------------------


def test_losing_a_band_never_raises_the_objective(cfg) -> None:
    """Losing a band that scores at or above the tile's score never raises it.

    Scoring the preferred band alone would have moved this tile onto a clean 'lo'
    and paid for stripping the crowded 'hi'. The general property does not hold
    under the contraharmonic mean: shedding a band that scores below the tile's
    score raises it.
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


def test_solver_round_off_does_not_reach_the_objective(cfg) -> None:
    """A sub-nano-dB change in RSRP leaves J bit-identical."""
    assert _score(_map([[-105.0 + 1e-9]]), cfg) == _score(_map([[-105.0]]), cfg)


# --- selection -------------------------------------------------------------


def test_the_pick_ignores_every_reported_kpi() -> None:
    """A better rate with a lower objective does not win."""
    moved = _kpi(hole_rate=0.0, overlap_rate=0.0, ue_service_failure_rate=0.0, objective=0.49)
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


def test_a_nan_objective_is_refused_rather_than_picked() -> None:
    """``np.argmax`` would rank a NaN first; the pick must not."""
    with pytest.raises(ValueError, match="non-finite"):
        best_by_objective([_kpi(), _kpi(objective=float("nan"))])


def test_a_weak_threshold_at_the_hole_threshold_is_refused(cfg) -> None:
    """The strength factor divides by their gap."""
    cfg.kpi.weak_dbm = cfg.kpi.hole_dbm
    with pytest.raises(ValueError, match="weak_dbm"):
        _score(_map([[-90.0]]), cfg)


def test_map_kpis_shares_reductions_without_changing_any_kpi() -> None:
    """The shared-reduction path equals each KPI measured on its own, NaN tiles included."""
    from src.kpi import (
        hole_rate,
        overlap_neighbor_mean,
        overlap_rate,
        rsrp_percentile_dbm,
        sinr_percentile_db,
        weak_rate,
    )
    from src.kpi.quality import LOW_PERCENTILE, MEDIAN_PERCENTILE
    from src.optim.objective import map_kpis

    rng = np.random.default_rng(0)
    rsrp = rng.uniform(-150.0, -60.0, (3, 4, 9, 11))
    rsrp[rng.random(rsrp.shape) < 0.2] = np.nan
    sinr = np.where(np.isfinite(rsrp), rng.normal(5.0, 10.0, rsrp.shape), np.nan)
    cfg = OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0, "overlap_margin_db": 6.0}}
    )
    assert map_kpis(rsrp, sinr, cfg) == {
        "hole_rate": hole_rate(rsrp, cfg),
        "weak_rate": weak_rate(rsrp, cfg),
        "overlap_rate": overlap_rate(rsrp, cfg),
        "overlap_neighbor_mean": overlap_neighbor_mean(rsrp, cfg),
        "rsrp_p50_dbm": rsrp_percentile_dbm(rsrp, cfg, MEDIAN_PERCENTILE),
        "rsrp_p05_dbm": rsrp_percentile_dbm(rsrp, cfg, LOW_PERCENTILE),
        "sinr_p50_db": sinr_percentile_db(rsrp, sinr, cfg, MEDIAN_PERCENTILE),
        "sinr_p05_db": sinr_percentile_db(rsrp, sinr, cfg, LOW_PERCENTILE),
    }


def test_run_json_is_strict_and_non_finite_kpis_round_trip(tmp_path) -> None:
    """``-inf`` and NaN KPIs (nothing covered) survive a parser that rejects bare constants."""
    kpi = _kpi(rsrp_p05_dbm=-np.inf, overlap_neighbor_mean=np.nan)
    path = LocalRunWriter(tmp_path).write_json("run", {"best_kpi": kpi.as_dict()})

    def refuse(token: str) -> None:
        raise ValueError(f"not strict JSON: {token}")

    loaded = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse)
    again = KpiVector.from_mapping(loaded["best_kpi"])
    assert again.rsrp_p05_dbm == -np.inf
    assert np.isnan(again.overlap_neighbor_mean)
    assert again.hole_rate == kpi.hole_rate
