"""Comparison tables, especially the verdict that keeps them honest."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import compare, maps
from src.evaluation.runs import Run
from src.optim.objective import MEASURE_NAMES, KpiVector


def _served() -> pd.DataFrame:
    """Serve_intervals-shaped rows: two intervals, one report with no cell."""
    return pd.DataFrame(
        {
            "t_index": [0, 0, 1, 1],
            "band": [0, 0, 1, -1],
            "tx": [1, 1, 0, -1],
            "sinr_db": [10.0, 20.0, 5.0, np.nan],
            "estimated_throughput_mbps": [4.0, 6.0, 9.0, np.nan],
        }
    )


def test_cell_band_load_counts_ues_and_their_throughput_per_cell_band() -> None:
    """Band 0 / tx 1 serves two UEs in interval 0; the unserved report loads nothing."""
    load = compare.cell_band_load(_served(), ["hi", "lo"], ["c0", "c1"]).set_index(["cell", "band"])
    assert load.loc[("c1", "hi"), "served_reports"] == 2
    assert load.loc[("c1", "hi"), "peak_ues"] == 2
    assert load.loc[("c1", "hi"), "median_throughput_mbps"] == pytest.approx(5.0)
    assert load.loc[("c1", "hi"), "median_sinr_db"] == pytest.approx(15.0)
    assert load.loc[("c0", "lo"), "peak_ues"] == 1
    assert load.loc[("c0", "hi"), "served_reports"] == 0
    assert np.isnan(load.loc[("c0", "hi"), "median_throughput_mbps"])


def test_service_summary_counts_unserved_reports_as_not_served() -> None:
    """One report of four has no cell; shares are of all reports."""
    summary = compare.service_summary(_served(), ["hi", "lo"])
    assert summary["not_served_share"] == pytest.approx(0.25)
    assert summary["share_hi"] == pytest.approx(0.5)
    assert summary["share_lo"] == pytest.approx(0.25)
    assert summary["sinr_median_db"] == pytest.approx(10.0)


def test_tile_median_is_per_tile_and_blank_where_nobody_was_served() -> None:
    """Tile (0, 0): 4 and 8 Mbit/s. Tile (0, 1): one unserved report. Tile (0, 2): no report."""
    served = pd.DataFrame(
        {
            "tile_row": [0, 0, 0],
            "tile_col": [0, 0, 1],
            "estimated_throughput_mbps": [4.0, 8.0, np.nan],
        }
    )
    median = maps.tile_median(served, "estimated_throughput_mbps", (1, 3))
    assert median[0, 0] == pytest.approx(6.0)
    assert np.isnan(median[0, 1:]).all()


def test_cell_table_reads_the_configured_cells() -> None:
    """The node is the mast part of an ``n<node>c<cell>`` name."""
    cfg = OmegaConf.create(
        {
            "simulation": {
                "transmitters": {
                    "cells": [
                        {
                            "name": "n3c1",
                            "x": 1.0,
                            "y": 2.0,
                            "z": 25.0,
                            "azimuth_deg": 165.0,
                            "tilt": {},
                        }
                    ]
                }
            }
        }
    )
    row = compare.cell_table(cfg).iloc[0]
    assert (row["cell"], row["node"], row["x"], row["azimuth_deg"]) == ("n3c1", "n3", 1.0, 165.0)


@pytest.fixture
def cfg() -> DictConfig:
    """The thresholds, without composing the whole config."""
    return OmegaConf.create({"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0}})


@pytest.fixture
def incumbent() -> KpiVector:
    """A plausible starting point, near the committed configuration's scores."""
    return KpiVector(
        hole_rate=0.10,
        weak_rate=0.12,
        overlap_rate=0.28,
        overlap_neighbor_mean=0.45,
        rsrp_p50_dbm=-95.0,
        rsrp_p05_dbm=-108.0,
        sinr_p50_db=8.0,
        sinr_p05_db=-3.0,
        ue_service_failure_rate=0.009,
        estimated_throughput_p05_mbps=1.0,
        estimated_throughput_p50_mbps=5.0,
        estimated_throughput_mean_mbps=6.0,
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


def test_a_change_between_two_empty_percentiles_is_undefined(incumbent: KpiVector) -> None:
    """``-inf - -inf`` is NaN, which is neither better nor worse."""
    before = dataclasses.replace(incumbent, sinr_p05_db=-np.inf)
    table = compare.delta_table(before, before).set_index("kpi")
    assert table.loc["sinr_p05_db", "verdict"] == compare.UNDEFINED


def test_the_maximised_kpi_reads_the_other_way(incumbent: KpiVector) -> None:
    """The KPI where up is better, and the usual place a sign error hides."""
    after = dataclasses.replace(incumbent, sinr_p05_db=incumbent.sinr_p05_db + 0.5)
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["sinr_p05_db", "verdict"] == compare.BETTER
    assert table.loc["sinr_p05_db", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """The maximised set is signal quality, throughput and J; any other is a sign error."""
    maximised = [name for name in MEASURE_NAMES if compare.direction(name) == "maximise"]
    assert maximised == [
        "rsrp_p50_dbm",
        "rsrp_p05_dbm",
        "sinr_p50_db",
        "sinr_p05_db",
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
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
            "weak_rate": [0.1] * n,
            "overlap_rate": [0.3] * n,
            "overlap_neighbor_mean": [0.45] * n,
            "rsrp_p50_dbm": [-95.0] * n,
            "rsrp_p05_dbm": [-108.0] * n,
            "sinr_p50_db": [8.0] * n,
            "sinr_p05_db": [-3.0] * n,
            "ue_service_failure_rate": [0.0] * n,
            "estimated_throughput_p05_mbps": [1.0] * n,
            "estimated_throughput_p50_mbps": [5.0] * n,
            "estimated_throughput_mean_mbps": [6.0] * n,
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
    """Hole rate is minimised and median SINR maximised; the dominated row drops out."""
    frame = pd.DataFrame({"hole_rate": [0.1, 0.2, 0.1, 0.05], "sinr_p50_db": [9.0, 9.0, 9.0, 5.0]})
    mask = compare.pareto_front(frame, ["hole_rate", "sinr_p50_db"])
    assert mask.tolist() == [True, False, True, True]


def test_relative_improvement_is_positive_when_better() -> None:
    """A falling hole rate and a rising median SINR both read as gains; J is not a KPI."""
    summary = pd.DataFrame(
        {
            "method": ["turbo", "turbo", "turbo"],
            "kpi": ["hole_rate", "sinr_p50_db", "objective"],
            "direction": ["minimise", "maximise", "maximise"],
            "incumbent": [0.2, 0.5, 0.6],
            "mean": [0.1, 0.6, 0.7],
        }
    )
    row = compare.relative_improvement(summary).iloc[0]
    assert row["hole_rate"] == pytest.approx(50.0)
    assert row["sinr_p50_db"] == pytest.approx(20.0)
    assert "objective" not in row


def test_sample_efficiency_is_nan_past_a_runs_length() -> None:
    """A three-evaluation run has no value at a budget of four."""
    runs = [_run("turbo", 0, [0.1, 0.5, 0.3, 0.9]), _run("random", 0, [0.1, 0.4, 0.2])]
    table = compare.sample_efficiency(
        compare.convergence(runs), kpis=["objective"], budgets=[2]
    ).set_index("budget")
    assert table.loc[2, "turbo"] == pytest.approx(0.5)
    assert table.loc[4, "turbo"] == pytest.approx(0.9)
    assert np.isnan(table.loc[4, "random"])


def test_overlap_neighbour_summary_counts_covered_tiles_only() -> None:
    """One band, three cells: tile 0 has two neighbours in margin, tile 1 is a hole."""
    rsrp = np.array([[[[-80.0, -130.0]], [[-82.0, -130.0]], [[-85.0, -130.0]]]])
    cfg = OmegaConf.create({"kpi": {"hole_dbm": -120.0, "overlap_margin_db": 6.0}})
    config = compare.Configuration(
        rsrp=rsrp,
        sinr=rsrp,
        served=pd.DataFrame(),
        demand=np.zeros(1),
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
                "capacity": {"max_admission_utilisation": 0.8},
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
            "estimated_throughput_mbps": [2.0, 4.0, 6.0, np.nan],
        }
    )
    config = compare.Configuration(
        rsrp=rsrp,
        sinr=np.full(rsrp.shape, 10.0),
        served=served,
        demand=np.zeros((1, 2)),
    )
    table = compare.band_kpis({"incumbent": config}, ["hi", "lo"], _capacity_cfg())
    table = table.set_index("band")

    assert table.index.tolist() == [compare.ALL_BANDS, "hi", "lo"]
    assert table.loc[compare.ALL_BANDS, "hole_rate"] == pytest.approx(0.0)
    assert table.loc["hi", "hole_rate"] == pytest.approx(0.5)
    assert table.loc["lo", "hole_rate"] == pytest.approx(0.0)
    # One report of four failed; a failure belongs to no single band.
    assert table.loc[compare.ALL_BANDS, "ue_service_failure_rate"] == pytest.approx(0.25)
    assert table.loc[["hi", "lo"], "ue_service_failure_rate"].isna().all()
    # Throughput is over the three served reports only, and network-wide too.
    assert table.loc[compare.ALL_BANDS, "estimated_throughput_p50_mbps"] == pytest.approx(4.0)
    assert table.loc[compare.ALL_BANDS, "estimated_throughput_mean_mbps"] == pytest.approx(4.0)
    assert table.loc[["hi", "lo"], "estimated_throughput_mean_mbps"].isna().all()
    # SINR 10 dB everywhere, so every covered tile reads 10 dB.
    assert table.loc[compare.ALL_BANDS, "sinr_p50_db"] == pytest.approx(10.0)


def test_improvement_table_is_positive_when_better(incumbent: KpiVector) -> None:
    """A halved hole rate is +50 %; a median SINR up by a tenth of itself is +10 %."""
    after = dataclasses.replace(
        incumbent, hole_rate=incumbent.hole_rate / 2, sinr_p50_db=incumbent.sinr_p50_db * 1.1
    )
    table = compare.improvement_table(incumbent, after).set_index("kpi")
    assert table.loc["hole_rate", "improvement_pct"] == pytest.approx(50.0)
    assert table.loc["sinr_p50_db", "improvement_pct"] == pytest.approx(10.0)
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


def test_improvement_table_is_nan_where_the_incumbent_is_zero(incumbent: KpiVector) -> None:
    """A hole rate going from 0 to anything has no relative change to report."""
    before = dataclasses.replace(incumbent, hole_rate=0.0)
    after = dataclasses.replace(incumbent, hole_rate=0.1)
    table = compare.improvement_table(before, after).set_index("kpi")
    assert np.isnan(table.loc["hole_rate", "improvement_pct"])
