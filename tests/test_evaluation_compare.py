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
    """Serve_intervals-shaped rows: two intervals, one report with no sector."""
    return pd.DataFrame(
        {
            "t_index": [0, 0, 1, 1],
            "band": [0, 0, 1, -1],
            "tx": [1, 1, 0, -1],
            "sinr_db": [10.0, 20.0, 5.0, np.nan],
            "estimated_throughput_mbps": [4.0, 6.0, 9.0, 0.0],
        }
    )


def test_sector_band_load_counts_ues_and_their_throughput_per_sector_band() -> None:
    """Band 0 / tx 1 serves two UEs in interval 0; the unserved report loads nothing."""
    load = compare.sector_band_load(_served(), ["hi", "lo"], ["c0", "c1"]).set_index(
        ["sector", "band"]
    )
    assert load.loc[("c1", "hi"), "served_reports"] == 2
    assert load.loc[("c1", "hi"), "peak_ues"] == 2
    assert load.loc[("c1", "hi"), "median_throughput_mbps"] == pytest.approx(5.0)
    assert load.loc[("c1", "hi"), "median_sinr_db"] == pytest.approx(15.0)
    assert load.loc[("c0", "lo"), "peak_ues"] == 1
    assert load.loc[("c0", "hi"), "served_reports"] == 0
    assert np.isnan(load.loc[("c0", "hi"), "median_throughput_mbps"])


def test_service_summary_shares_are_of_all_reports() -> None:
    """One report of four has no sector, so the band shares sum to three quarters."""
    summary = compare.service_summary(_served(), ["hi", "lo"])
    assert set(summary) == {"reports", "share_hi", "share_lo"}
    assert summary["share_hi"] == pytest.approx(0.5)
    assert summary["share_lo"] == pytest.approx(0.25)
    assert "sinr_median_db" not in summary


def test_tile_median_is_zero_on_a_hole_and_blank_where_nobody_stands() -> None:
    """Tile (0, 0): 4 and 8 Mbit/s. Tile (0, 1): one unserved report. Tile (0, 2): no report."""
    served = pd.DataFrame(
        {
            "tile_row": [0, 0, 0],
            "tile_col": [0, 0, 1],
            "estimated_throughput_mbps": [4.0, 8.0, 0.0],
        }
    )
    median = maps.tile_median(served, "estimated_throughput_mbps", (1, 3))
    assert median[0, :2].tolist() == pytest.approx([6.0, 0.0])
    assert np.isnan(median[0, 2])


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
        estimated_throughput_p05_mbps=1.0,
        estimated_throughput_p50_mbps=5.0,
        estimated_throughput_mean_mbps=6.0,
        coverage_objective=0.9,
        separation_objective=0.4,
        throughput_objective=3.0,
    )


def test_delta_table_reports_the_network_kpis_in_order(incumbent: KpiVector) -> None:
    """All-band best-server RSRP and SINR are not reported, nor are the objectives."""
    table = compare.delta_table(incumbent, incumbent)
    assert table["kpi"].tolist() == list(compare.NETWORK_KPIS)


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


def test_a_change_between_two_undefined_values_is_undefined(incumbent: KpiVector) -> None:
    """``-inf - -inf`` is NaN, which is neither better nor worse."""
    before = dataclasses.replace(incumbent, estimated_throughput_p05_mbps=-np.inf)
    table = compare.delta_table(before, before).set_index("kpi")
    assert table.loc["estimated_throughput_p05_mbps", "verdict"] == compare.UNDEFINED


def test_the_maximised_kpi_reads_the_other_way(incumbent: KpiVector) -> None:
    """The KPI where up is better, and the usual place a sign error hides."""
    after = dataclasses.replace(incumbent, estimated_throughput_p05_mbps=1.5)
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["estimated_throughput_p05_mbps", "verdict"] == compare.BETTER
    assert table.loc["estimated_throughput_p05_mbps", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """The maximised set is signal quality, throughput and the objectives; else a sign error."""
    maximised = [name for name in MEASURE_NAMES if compare.direction(name) == "maximise"]
    assert maximised == [
        "rsrp_p50_dbm",
        "rsrp_p05_dbm",
        "sinr_p50_db",
        "sinr_p05_db",
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
        "coverage_objective",
        "separation_objective",
        "throughput_objective",
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
    merged = compare.coverage_comparison({"incumbent": table, "morbo": table})
    assert list(merged.columns) == [
        "coverage",
        "incumbent_tile",
        "incumbent_demand",
        "morbo_tile",
        "morbo_demand",
    ]
    assert len(merged) == 3


def test_coverage_comparison_of_nothing_is_empty() -> None:
    """No runs is a legitimate state, not an error."""
    assert compare.coverage_comparison({}).empty


def _run(
    method: str,
    seed: int,
    coverage: list[float],
    phases: list[str] | None = None,
    throughput: list[float] | None = None,
) -> Run:
    """A run whose coverage objective per evaluation is ``coverage``, row 0 the incumbent.

    The other objectives are 1 throughout, so the hypervolume is the best coverage.
    """
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
            "estimated_throughput_p05_mbps": [1.0] * n,
            "estimated_throughput_p50_mbps": throughput or [5.0] * n,
            "estimated_throughput_mean_mbps": [6.0] * n,
            "coverage_objective": coverage,
            "separation_objective": [1.0] * n,
            "throughput_objective": [1.0] * n,
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
    runs = [
        _run("morbo", 0, [0.5, 0.8], throughput=[5.0, 8.0]),
        _run("morbo", 1, [0.5, 0.6], throughput=[5.0, 6.0]),
    ]
    table = compare.seed_summary(runs).set_index("kpi")
    assert table.index.tolist() == list(compare.SEARCH_MEASURES)
    name = "estimated_throughput_p50_mbps"
    assert table.loc[name, "mean"] == pytest.approx(7.0)
    low, high = table.loc[name, ["ci95_low", "ci95_high"]]
    assert low < 7.0 < high
    assert table.loc[name, "direction"] == "maximise"
    assert table.loc[name, "verdict"] == compare.BETTER


def test_hypervolume_table_separates_the_design_from_the_search() -> None:
    """The design ends at the first ``search`` row; the search adds 0.9 - 0.5."""
    phases = ["incumbent", "init", "search", "search"]
    row = compare.hypervolume_table([_run("morbo", 0, [0.1, 0.5, 0.9, 0.7], phases)]).iloc[0]
    assert row["incumbent_hv"] == pytest.approx(0.1)
    assert row["initial_design_hv"] == pytest.approx(0.5)
    assert row["final_hv"] == pytest.approx(0.9)
    assert row["pareto_points"] == 1
    assert row["best_iteration"] == 2


def test_paired_method_gain_pairs_by_seed() -> None:
    """A seed only one method ran is left out of the pairs; the gain is in hypervolume."""
    runs = [
        _run("morbo", 0, [0.5, 0.9]),
        _run("random", 0, [0.5, 0.7]),
        _run("morbo", 1, [0.5, 0.8]),
        _run("random", 1, [0.5, 0.7]),
        _run("morbo", 2, [0.5, 0.9]),
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


def test_sample_efficiency_is_nan_past_a_runs_length() -> None:
    """A three-evaluation run has no value at a budget of four."""
    runs = [_run("morbo", 0, [0.1, 0.5, 0.3, 0.9]), _run("random", 0, [0.1, 0.4, 0.2])]
    table = compare.sample_efficiency(
        compare.convergence(runs), kpis=[compare.HYPERVOLUME], budgets=[2]
    ).set_index("budget")
    assert table.loc[2, "morbo"] == pytest.approx(0.5)
    assert table.loc[4, "morbo"] == pytest.approx(0.9)
    assert np.isnan(table.loc[4, "random"])


def test_overlap_neighbour_summary_counts_covered_tiles_only() -> None:
    """One band, three sectors: tile 0 has two neighbours in margin, tile 1 is a hole."""
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
    """The thresholds the KPIs ``band_kpis`` reports read."""
    return OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0, "overlap_margin_db": 6.0}}
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
            "estimated_throughput_mbps": [2.0, 4.0, 6.0, 0.0],
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
    # Throughput is over every report, the unserved one at 0, and network-wide only.
    assert table.loc[compare.ALL_BANDS, "estimated_throughput_p50_mbps"] == pytest.approx(3.0)
    assert table.loc[compare.ALL_BANDS, "estimated_throughput_mean_mbps"] == pytest.approx(3.0)
    assert table.loc[["hi", "lo"], "estimated_throughput_mean_mbps"].isna().all()
    # SINR 10 dB everywhere, so every covered tile of a band reads 10 dB. The
    # strongest layer across bands is not reported.
    assert table.loc["lo", "sinr_p50_db"] == pytest.approx(10.0)
    assert np.isnan(table.loc[compare.ALL_BANDS, "sinr_p50_db"])
    assert np.isnan(table.loc[compare.ALL_BANDS, "rsrp_p05_dbm"])
    # One sector per band has no co-band neighbour; the count is network-wide only.
    assert table.loc[compare.ALL_BANDS, "overlap_neighbor_mean"] == pytest.approx(0.0)
    assert table.loc[["hi", "lo"], "overlap_neighbor_mean"].isna().all()


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
            "morbo": config(after, np.array([[0.0, 1.0]])),
        },
        cfg,
    ).set_index("coverage")
    assert list(table.columns) == [
        "Current configuration: Share of area",
        "Current configuration: Share of demand",
        "MORBO: Share of area",
        "MORBO: Share of demand",
    ]
    assert table.loc["hole", "Current configuration: Share of demand"] == 1.0
    assert table.loc["good", "MORBO: Share of demand"] == 1.0
