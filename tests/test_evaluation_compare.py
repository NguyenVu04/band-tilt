"""Comparison tables, especially the verdict that keeps them honest."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import compare
from src.evaluation.runs import Run
from src.optim.objective import MEASURE_NAMES, KpiVector


def _served() -> pd.DataFrame:
    """Serve_intervals-shaped rows: two intervals, one blocked report."""
    return pd.DataFrame(
        {
            "t_index": [0, 0, 1, 1],
            "band": [0, 0, 1, -1],
            "tx": [1, 1, 0, -1],
            "prb_per_ue": [2.0, 3.0, 4.0, 9.0],
            "sinr_db": [10.0, 20.0, 5.0, np.nan],
        }
    )


def test_cell_band_load_sums_admitted_prbs_per_interval() -> None:
    """Band 0 / tx 1 carries 5 PRBs in interval 0 only; the blocked report loads nothing."""
    load = compare.cell_band_load(
        _served(), ["hi", "lo"], ["c0", "c1"], np.array([[10.0, 10.0], [10.0, 10.0]])
    ).set_index(["cell", "band"])
    assert load.loc[("c1", "hi"), "served_reports"] == 2
    assert load.loc[("c1", "hi"), "peak_prb"] == pytest.approx(5.0)
    assert load.loc[("c1", "hi"), "mean_prb"] == pytest.approx(2.5)
    assert load.loc[("c1", "hi"), "peak_utilisation"] == pytest.approx(0.5)
    assert load.loc[("c1", "hi"), "median_sinr_db"] == pytest.approx(15.0)
    assert load.loc[("c0", "lo"), "peak_prb"] == pytest.approx(4.0)
    assert load["peak_prb"].sum() == pytest.approx(9.0)


def test_service_summary_counts_blocked_reports_as_not_served() -> None:
    """One report of four is blocked; shares are of all reports."""
    summary = compare.service_summary(_served(), ["hi", "lo"])
    assert summary["not_served_share"] == pytest.approx(0.25)
    assert summary["share_hi"] == pytest.approx(0.5)
    assert summary["share_lo"] == pytest.approx(0.25)
    assert summary["sinr_median_db"] == pytest.approx(10.0)


@pytest.fixture
def cfg() -> DictConfig:
    """The thresholds, without composing the whole config."""
    return OmegaConf.create({"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0}})


@pytest.fixture
def incumbent() -> KpiVector:
    """A plausible starting point, near the committed configuration's scores."""
    return KpiVector(
        hole_rate=0.10,
        overlap_rate=0.28,
        overlap_neighbor_mean=0.45,
        weak_rate=0.12,
        rsrp_p05_dbm=-108.0,
        rsrp_p50_dbm=-95.0,
        sinr_p05_db=-3.0,
        sinr_p50_db=8.0,
        served_rate=0.009,
        load_imbalance=0.44,
        objective=0.40,
    )


def test_delta_table_is_in_priority_order(incumbent: KpiVector) -> None:
    """Reading top to bottom is reading the order the winner was decided in."""
    table = compare.delta_table(incumbent, incumbent)
    assert table["kpi"].tolist() == list(MEASURE_NAMES)


def test_identical_configurations_are_unchanged_everywhere(incumbent: KpiVector) -> None:
    """A zero delta is the same measurement twice, not an improvement of zero."""
    table = compare.delta_table(incumbent, incumbent)
    assert (table["verdict"] == compare.UNCHANGED).all()


def test_any_nonzero_change_reads_by_direction(incumbent: KpiVector) -> None:
    """No noise floor: a delta is reported at face value, however small."""
    after = dataclasses.replace(incumbent, hole_rate=incumbent.hole_rate - 1e-9)
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["hole_rate", "verdict"] == compare.BETTER


def test_a_change_reads_by_direction(incumbent: KpiVector) -> None:
    """Both are minimised, so down is better and up is worse."""
    after = dataclasses.replace(
        incumbent,
        hole_rate=incumbent.hole_rate - 0.05,
        weak_rate=incumbent.weak_rate + 0.05,
    )
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["hole_rate", "verdict"] == compare.BETTER
    assert table.loc["weak_rate", "verdict"] == compare.WORSE


def test_the_maximised_kpi_reads_the_other_way(incumbent: KpiVector) -> None:
    """The KPI where up is better, and the usual place a sign error hides."""
    after = dataclasses.replace(incumbent, served_rate=incumbent.served_rate + 0.05)
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["served_rate", "verdict"] == compare.BETTER
    assert table.loc["served_rate", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """The maximised set is the three that read upward; any other is a sign error."""
    maximised = [name for name in MEASURE_NAMES if compare.direction(name) == "maximise"]
    assert maximised == [
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "served_rate",
        "objective",
    ]


def test_coverage_comparison_puts_labels_side_by_side() -> None:
    """One column pair per configuration, so the tile/demand gap is readable."""
    table = pd.DataFrame(
        {
            "coverage": ["hole", "weak", "good"],
            "tiles": [1, 2, 3],
            "tile_share": [0.1, 0.2, 0.7],
            "reports": [1, 2, 7],
            "demand_share": [0.1, 0.2, 0.7],
        }
    )
    merged = compare.coverage_comparison({"incumbent": table, "turbo": table})
    assert list(merged.columns) == [
        "coverage",
        "incumbent_tile",
        "incumbent_demand",
        "turbo_tile",
        "turbo_demand",
    ]
    assert len(merged) == 3


def test_coverage_comparison_of_nothing_is_empty() -> None:
    """No runs is a legitimate state, not an error."""
    assert compare.coverage_comparison({}).empty


def _run(method: str, seed: int, coverage: list[float], phases: list[str] | None = None) -> Run:
    """A run whose ``objective`` per evaluation is ``coverage``, row 0 the incumbent."""
    n = len(coverage)
    history = pd.DataFrame(
        {
            "iteration": range(n),
            "phase": phases or ["incumbent"] + ["init"] * (n - 1),
            "hole_rate": [0.1] * n,
            "overlap_rate": [0.3] * n,
            "overlap_neighbor_mean": [0.45] * n,
            "weak_rate": [0.1] * n,
            "rsrp_p05_dbm": [-108.0] * n,
            "rsrp_p50_dbm": [-95.0] * n,
            "sinr_p05_db": [-3.0] * n,
            "sinr_p50_db": [8.0] * n,
            "served_rate": [0.0] * n,
            "load_imbalance": [0.44] * n,
            "objective": coverage,
        }
    )
    best = int(np.argmax(coverage))
    kpi = {name: float(history.loc[best, name]) for name in MEASURE_NAMES}
    incumbent = {name: float(history.loc[0, name]) for name in MEASURE_NAMES}
    meta = {
        "best_iteration": best,
        "best_kpi": kpi,
        "incumbent_kpi": incumbent,
        "config": {"optim": {"seed": seed}},
    }
    return Run(method, f"run{seed}", Path("."), history, pd.DataFrame(), meta)


def test_seed_summary_interval_brackets_the_mean() -> None:
    """Two seeds give a finite interval centred on the mean winner."""
    runs = [_run("turbo", 0, [0.5, 0.8]), _run("turbo", 1, [0.5, 0.6])]
    table = compare.seed_summary(runs).set_index("kpi")
    assert table.loc["objective", "mean"] == pytest.approx(0.7)
    low, high = table.loc["objective", ["ci95_low", "ci95_high"]]
    assert low < 0.7 < high
    assert table.loc["objective", "direction"] == "maximise"
    assert table.loc["objective", "verdict"] == compare.BETTER


def test_winner_vs_candidates_separates_winner_from_typical() -> None:
    """The incumbent is excluded from the candidate median."""
    row = compare.winner_vs_candidates([_run("random", 0, [0.1, 0.9, 0.8, 0.7])]).iloc[0]
    assert row["incumbent"] == pytest.approx(0.1)
    assert row["candidate_median"] == pytest.approx(0.8)
    assert row["winner"] == pytest.approx(0.9)


def test_paired_method_gain_pairs_by_seed() -> None:
    """A seed only one method ran is left out of the pairs."""
    runs = [
        _run("turbo", 0, [0.5, 0.9]),
        _run("random", 0, [0.5, 0.7]),
        _run("turbo", 1, [0.5, 0.8]),
        _run("random", 1, [0.5, 0.7]),
        _run("turbo", 2, [0.5, 0.9]),
    ]
    row = compare.paired_method_gain(runs).iloc[0]
    assert row["n_pairs"] == 2
    assert row["mean_gain"] == pytest.approx(0.15)
    assert row["method_better"] == 2


def test_pareto_front_reads_each_column_in_its_direction() -> None:
    """Hole rate is minimised and the served rate maximised; the dominated row drops out."""
    frame = pd.DataFrame({"hole_rate": [0.1, 0.2, 0.1, 0.05], "served_rate": [0.9, 0.9, 0.9, 0.5]})
    mask = compare.pareto_front(frame, ["hole_rate", "served_rate"])
    assert mask.tolist() == [True, False, True, True]


def test_relative_improvement_is_positive_when_better() -> None:
    """A falling hole rate and a rising served rate both read as gains."""
    summary = pd.DataFrame(
        {
            "method": ["turbo", "turbo"],
            "kpi": ["hole_rate", "served_rate"],
            "direction": ["minimise", "maximise"],
            "incumbent": [0.2, 0.5],
            "mean": [0.1, 0.6],
        }
    )
    row = compare.relative_improvement(summary).iloc[0]
    assert row["hole_rate"] == pytest.approx(50.0)
    assert row["served_rate"] == pytest.approx(20.0)


def test_sample_efficiency_is_nan_past_a_runs_length() -> None:
    """A three-evaluation run has no value at a budget of four."""
    runs = [_run("turbo", 0, [0.1, 0.5, 0.3, 0.9]), _run("rule", 0, [0.1, 0.4, 0.2])]
    table = compare.sample_efficiency(
        compare.convergence(runs), kpis=["objective"], budgets=[2]
    ).set_index("budget")
    assert table.loc[2, "turbo"] == pytest.approx(0.5)
    assert table.loc[4, "turbo"] == pytest.approx(0.9)
    assert np.isnan(table.loc[4, "rule"])


def test_overlap_neighbour_summary_counts_covered_tiles_only() -> None:
    """One band, three cells: tile 0 has two neighbours in margin, tile 1 is a hole."""
    rsrp = np.array([[[[-80.0, -130.0]], [[-82.0, -130.0]], [[-85.0, -130.0]]]])
    cfg = OmegaConf.create({"kpi": {"hole_dbm": -120.0, "overlap_margin_db": 6.0}})
    config = compare.Configuration(
        rsrp=rsrp,
        sinr=rsrp,
        served=pd.DataFrame(),
        demand=np.zeros(1),
        t_values=np.zeros(0, dtype=int),
        prb=np.zeros((0, 1, 3)),
    )
    row = compare.overlap_neighbour_summary({"incumbent": config}, cfg).iloc[0]
    assert row["mean_neighbours_covered"] == pytest.approx(2.0)
    assert row["mean_neighbours_all"] == pytest.approx(1.0)
    assert row["share_2_neighbours"] == pytest.approx(1.0)


def _capacity_cfg() -> DictConfig:
    """The thresholds plus the capacity model ``band_kpis`` needs for its spec."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "capacity": {
                    "band_preference": ["hi", "lo"],
                    "rsrp_threshold_dbm": -120.0,
                    "max_admission_utilisation": 0.8,
                    "throughput_per_ue_bps": 1.0,
                },
            },
            "simulation": {
                "radio_map": {"bands": [{"name": n, "scs_hz": 15000} for n in ("hi", "lo")]},
                "transmitters": {
                    "cells": [
                        {
                            "name": "c0",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 30.0,
                            "azimuth_deg": 0.0,
                            "tilt": {},
                            "max_prb": {"hi": 10, "lo": 10},
                        }
                    ]
                },
            },
        }
    )


def test_band_kpis_reads_each_layer_through_the_same_definitions() -> None:
    """Tile 1 is a hole on 'hi' and covered on 'lo', so the map has no hole at all.

    The ``all`` row is the whole map, which is what a run's own KPI vector
    measures; a band row is the same function given one band's layers.
    """
    rsrp = np.array([[[[-80.0, -130.0]]], [[[-85.0, -95.0]]]])
    served = pd.DataFrame(
        {
            "t_index": [0, 0, 0, 0],
            "band": [0, 1, 1, -1],
            "tx": [0, 0, 0, -1],
            "prb_per_ue": [2.0, 1.0, 1.0, np.nan],
        }
    )
    config = compare.Configuration(
        rsrp=rsrp,
        sinr=np.full(rsrp.shape, 10.0),
        served=served,
        demand=np.zeros((1, 2)),
        t_values=np.array([0]),
        prb=np.array([[[2.0], [2.0]]]),
    )
    table = compare.band_kpis({"incumbent": config}, ["hi", "lo"], _capacity_cfg())
    table = table.set_index("band")

    assert table.index.tolist() == [compare.ALL_BANDS, "hi", "lo"]
    assert table.loc[compare.ALL_BANDS, "hole_rate"] == pytest.approx(0.0)
    assert table.loc["hi", "hole_rate"] == pytest.approx(0.5)
    assert table.loc["lo", "hole_rate"] == pytest.approx(0.0)
    # Every band's share is of all four reports, so the band rows sum to the map's.
    assert table.loc["hi", "served_rate"] == pytest.approx(0.25)
    assert table.loc["lo", "served_rate"] == pytest.approx(0.5)
    assert table.loc[compare.ALL_BANDS, "served_rate"] == pytest.approx(0.75)
    # 2 PRBs of a 10-PRB limit on each band, so both are at 20% and balanced.
    assert table.loc[compare.ALL_BANDS, "prb_utilisation_max"] == pytest.approx(0.2)
    assert table.loc[compare.ALL_BANDS, "load_imbalance"] == pytest.approx(0.0)


def test_prb_usage_by_time_labels_every_cell_band_and_interval() -> None:
    """The series the load KPIs reduce, so a hot cell can be read interval by interval."""
    config = compare.Configuration(
        rsrp=np.full((2, 1, 1, 1), -80.0),
        sinr=np.full((2, 1, 1, 1), 10.0),
        served=pd.DataFrame(),
        demand=np.zeros((1, 1)),
        t_values=np.array([0, 1]),
        prb=np.array([[[4.0], [0.0]], [[0.0], [5.0]]]),
    )
    usage = compare.prb_usage_by_time(
        {"incumbent": config}, ["hi", "lo"], ["c0"], np.array([[10.0], [10.0]])
    )
    assert len(usage) == 4
    hot = usage.set_index(["t_index", "band"])["utilisation"]
    assert hot.loc[(0, "hi")] == pytest.approx(0.4)
    assert hot.loc[(1, "lo")] == pytest.approx(0.5)
    assert hot.loc[(1, "hi")] == pytest.approx(0.0)


def test_improvement_table_is_positive_when_better(incumbent: KpiVector) -> None:
    """A halved hole rate is +50 %; a served rate up by a tenth of itself is +10 %."""
    after = dataclasses.replace(
        incumbent, hole_rate=incumbent.hole_rate / 2, served_rate=incumbent.served_rate * 1.1
    )
    table = compare.improvement_table(incumbent, after).set_index("kpi")
    assert table.loc["hole_rate", "improvement_pct"] == pytest.approx(50.0)
    assert table.loc["served_rate", "improvement_pct"] == pytest.approx(10.0)
    assert table.loc["weak_rate", "improvement_pct"] == 0.0


def test_coverage_by_area_and_demand_weighs_every_map_by_the_incumbents_demand(
    cfg: DictConfig,
) -> None:
    """Tile 0 holds all the demand; it is a hole before and good after."""
    before = np.array([[[[-130.0, -80.0]]]])
    after = np.array([[[[-80.0, -130.0]]]])

    def config(rsrp: np.ndarray, demand: np.ndarray) -> compare.Configuration:
        return compare.Configuration(
            rsrp=rsrp,
            sinr=rsrp,
            served=pd.DataFrame(),
            demand=demand,
            t_values=np.zeros(0, dtype=int),
            prb=np.zeros((0, 1, 1)),
        )

    table = compare.coverage_by_area_and_demand(
        {
            "incumbent": config(before, np.array([[1.0, 0.0]])),
            "turbo": config(after, np.array([[0.0, 1.0]])),
        },
        cfg,
    ).set_index("coverage")
    assert list(table.columns) == [
        "Current configuration: Share of area",
        "Current configuration: Share of demand",
        "TuRBO: Share of area",
        "TuRBO: Share of demand",
    ]
    assert table.loc["hole", "Current configuration: Share of demand"] == 1.0
    assert table.loc["good", "TuRBO: Share of demand"] == 1.0
