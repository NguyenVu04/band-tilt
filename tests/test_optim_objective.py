"""The KPI vector, the sign convention, the objectives and the hypervolume pick."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.optim.history import LocalRunWriter
from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    MEASURE_NAMES,
    OBJECTIVE_NAMES,
    KpiVector,
    best_by_hvc,
    coverage_objective,
    hypervolume,
    hypervolume_contributions,
    pareto_mask,
    separation_objective,
    throughput_objective,
)


@pytest.fixture
def cfg():
    """The three settings the map objectives read. They read nothing else."""
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
        "estimated_throughput_p05_mbps": 0.0,
        "estimated_throughput_p50_mbps": 5.0,
        "estimated_throughput_mean_mbps": 6.0,
        "coverage_objective": 0.9,
        "separation_objective": 0.5,
        "throughput_objective": 100.0,
    }
    return KpiVector(**{**values, **overrides})


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


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
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
    )
    assert OBJECTIVE_NAMES == ("coverage_objective", "separation_objective")
    assert MEASURE_NAMES == (*KPI_NAMES, *OBJECTIVE_NAMES, "throughput_objective")


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
        *OBJECTIVE_NAMES,
        "throughput_objective",
    }


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape run.json records and the evaluation stage reads back."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_measure() -> None:
    """An incomplete measurement, or a record from the single-objective era, is refused."""
    values = _kpi().as_dict()
    del values["separation_objective"]
    with pytest.raises(KeyError, match="separation_objective"):
        KpiVector.from_mapping(values)


# --- coverage ---------------------------------------------------------------


def test_a_band_at_the_weak_threshold_covers_half(cfg) -> None:
    """The logistic is centred on kpi.weak_dbm, one decade of odds per 10 dB."""
    assert coverage_objective(_map([[-90.0]]), cfg) == pytest.approx(0.5)
    assert coverage_objective(_map([[-80.0]]), cfg) == pytest.approx(10.0 / 11.0, abs=5e-7)


def test_coverage_is_the_chance_that_any_band_covers(cfg) -> None:
    """Two bands at the threshold: 1 - (1/2)(1/2)."""
    assert coverage_objective(_map([[-90.0], [-90.0]]), cfg) == pytest.approx(0.75)


def test_coverage_reads_only_the_strongest_sector_of_each_band(cfg) -> None:
    """A rival neither adds nor removes coverage; crowding is separation's business."""
    crowded = coverage_objective(_map([[-110.0, -110.0]]), cfg)
    assert crowded == coverage_objective(_map([[-110.0, np.nan]]), cfg)


def test_a_no_path_tile_is_not_covered(cfg) -> None:
    """No path is -inf, which no logistic lifts off zero."""
    assert coverage_objective(_map([[np.nan, np.nan], [np.nan, np.nan]]), cfg) == 0.0


# --- separation -------------------------------------------------------------


def test_a_lone_server_is_perfectly_separated(cfg) -> None:
    """No co-band rival above the hole threshold, nothing to price."""
    assert separation_objective(_map([[-90.0, -125.0]]), cfg) == pytest.approx(1.0)


def test_a_rival_one_margin_down_halves_the_band(cfg) -> None:
    """10^((6 - 6) / 10) = 1, so the factor is 1 / 2."""
    assert separation_objective(_map([[-90.0, -96.0]]), cfg) == pytest.approx(0.5)


def test_an_equal_rival_costs_more_than_half(cfg) -> None:
    """At 0 dB apart the term is 10^0.6."""
    expected = 1.0 / (1.0 + 10.0**0.6)
    assert separation_objective(_map([[-90.0, -90.0]]), cfg) == pytest.approx(expected, abs=5e-7)


def test_separation_multiplies_over_bands_and_compares_within_each(cfg) -> None:
    """Each band halved by its own rival; a strong other-band layer is no rival."""
    rsrp = _map([[-90.0, -96.0], [-70.0, -76.0]])
    assert separation_objective(rsrp, cfg) == pytest.approx(0.25)


def test_a_band_with_no_server_above_the_hole_threshold_counts_one(cfg) -> None:
    """Holes are coverage's business; an uncovered band has no rival to price."""
    half = separation_objective(_map([[-125.0, -125.0], [-90.0, -96.0]]), cfg)
    assert half == pytest.approx(0.5)
    assert separation_objective(_map([[np.nan, np.nan]]), cfg) == pytest.approx(1.0)


def test_a_rival_at_or_below_the_hole_threshold_is_not_priced(cfg) -> None:
    """-120 dBm serves nobody, so it crowds nobody."""
    assert separation_objective(_map([[-118.0, -120.0]]), cfg) == pytest.approx(1.0)


def test_solver_round_off_does_not_reach_the_objectives(cfg) -> None:
    """A sub-nano-dB change in RSRP leaves both map objectives bit-identical."""
    for objective in (coverage_objective, separation_objective):
        moved = objective(_map([[-105.0 + 1e-9, -108.0]]), cfg)
        assert moved == objective(_map([[-105.0, -108.0]]), cfg)


# --- throughput -------------------------------------------------------------


def test_throughput_is_the_mean_log_with_an_unserved_ue_at_zero() -> None:
    """ln(1 + 9) twice and ln(1 + 0) for the UE on a hole, over three reports."""
    served = pd.DataFrame({"band": [0, 0, -1], "estimated_throughput_mbps": [9.0, 9.0, 0.0]})
    assert throughput_objective(served) == pytest.approx(2.0 * math.log(10.0) / 3.0, abs=5e-6)
    assert throughput_objective(served.iloc[:0]) == 0.0


# --- hypervolume ------------------------------------------------------------


def test_one_point_spans_its_box_to_the_origin() -> None:
    """The reference point is the origin."""
    assert hypervolume(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(6.0)


def test_overlapping_boxes_are_counted_once() -> None:
    """2x1x1 and 1x2x1 share a 1x1x1 cube."""
    assert hypervolume(np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])) == pytest.approx(3.0)
    assert hypervolume(np.array([[2.0, 1.0], [1.0, 2.0]])) == pytest.approx(3.0)


def test_a_point_not_above_the_origin_adds_nothing() -> None:
    """A zero objective spans no volume."""
    assert hypervolume(np.array([[0.0, 5.0, 5.0], [1.0, 1.0, 1.0]])) == pytest.approx(1.0)
    assert hypervolume(np.empty((0, 3))) == 0.0


def test_hypervolume_matches_a_brute_force_count() -> None:
    """Integer points against a count of the unit cells they dominate."""
    rng = np.random.default_rng(0)
    points = rng.integers(1, 6, size=(7, 3)).astype(float)
    centres = np.stack(np.meshgrid(*[np.arange(5) + 0.5] * 3, indexing="ij"), axis=-1)
    centres = centres.reshape(-1, 3)
    dominated = (centres[:, None, :] < points[None, :, :]).all(axis=2).any(axis=1)
    assert hypervolume(points) == pytest.approx(float(dominated.sum()))


def test_a_dominated_point_contributes_nothing() -> None:
    """Removing it loses no volume."""
    points = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    assert hypervolume_contributions(points).tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_of_identical_points_the_first_carries_the_contribution() -> None:
    """A repeated evaluation neither doubles a point's weight nor cancels it."""
    points = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 1.0, 1.0]])
    assert hypervolume_contributions(points).tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_identical_rows_do_not_dominate_each_other() -> None:
    """Domination needs a strict gain somewhere."""
    mask = pareto_mask(np.array([[1.0, 1.0], [1.0, 1.0], [0.5, 1.0]]))
    assert mask.tolist() == [True, True, False]


# --- selection --------------------------------------------------------------


def test_the_pick_is_the_largest_hypervolume_contribution() -> None:
    """The far-reaching separation point holds the largest exclusive slab here."""
    kpis = [
        _kpi(),
        _kpi(coverage_objective=0.95, separation_objective=0.51),
        _kpi(separation_objective=1.0),
    ]
    assert best_by_hvc(kpis) == 2


def test_the_pick_ignores_every_reported_kpi() -> None:
    """Better rates, or throughput, on a dominated objective vector do not win."""
    moved = _kpi(
        hole_rate=0.0, overlap_rate=0.0, separation_objective=0.49, throughput_objective=200.0
    )
    assert best_by_hvc([_kpi(), moved]) == 0


def test_an_equal_contribution_keeps_the_earlier_configuration() -> None:
    """The incumbent holds unless a candidate actually contributes more."""
    assert best_by_hvc([_kpi(), _kpi()]) == 0


def test_choosing_from_nothing_raises() -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        best_by_hvc([])


def test_a_nan_objective_is_refused_rather_than_picked() -> None:
    """A NaN would poison every comparison it enters."""
    with pytest.raises(ValueError, match="non-finite"):
        best_by_hvc([_kpi(), _kpi(coverage_objective=float("nan"))])


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
