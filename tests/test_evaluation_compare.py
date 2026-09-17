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
        served_ratio=0.009,
        weak_rate=0.12,
        edge_rsrp_dbm=-108.0,
        j_radio=0.40,
        j_load=0.40,
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
    after = dataclasses.replace(incumbent, served_ratio=incumbent.served_ratio + 0.05)
    table = compare.delta_table(incumbent, after).set_index("kpi")
    assert table.loc["served_ratio", "verdict"] == compare.BETTER
    assert table.loc["served_ratio", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """The maximised set is the four that read upward; any other is a sign error."""
    maximised = [name for name in MEASURE_NAMES if compare.direction(name) == "maximise"]
    assert maximised == ["served_ratio", "edge_rsrp_dbm", "j_radio", "j_load"]


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


@pytest.fixture
def scored_cfg() -> DictConfig:
    """``gamma = 1`` puts the whole objective on ``j_radio``.

    The score then reduces to that one column, so these tests can assert exact
    arithmetic.
    """
    objective = {"tau_r_db": 10.0, "beta": 1.0, "rho_0": 0.8, "alpha": 0.9, "gamma": 1.0}
    return OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "overlap_margin_db": 6.0, "objective": objective}}
    )


def _run(method: str, seed: int, coverage: list[float], phases: list[str] | None = None) -> Run:
    """A run whose score is ``j_radio`` per evaluation, row 0 the incumbent."""
    n = len(coverage)
    history = pd.DataFrame(
        {
            "iteration": range(n),
            "phase": phases or ["incumbent"] + ["init"] * (n - 1),
            "hole_rate": [0.1] * n,
            "overlap_rate": [0.3] * n,
            "served_ratio": [0.0] * n,
            "weak_rate": [0.1] * n,
            "edge_rsrp_dbm": [-108.0] * n,
            "j_radio": coverage,
            "j_load": [0.4] * n,
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


def test_seed_summary_interval_brackets_the_mean(scored_cfg: DictConfig) -> None:
    """Two seeds give a finite interval centred on the mean winner."""
    runs = [_run("turbo", 0, [0.5, 0.8]), _run("turbo", 1, [0.5, 0.6])]
    table = compare.seed_summary(runs, scored_cfg).set_index("kpi")
    assert table.loc["j_radio", "mean"] == pytest.approx(0.7)
    low, high = table.loc["j_radio", ["ci95_low", "ci95_high"]]
    assert low < 0.7 < high
    assert table.loc["j_radio", "verdict"] == compare.BETTER
    assert table.loc["score", "direction"] == "maximise"
    assert table.loc["score", "verdict"] == compare.BETTER


def test_winner_vs_candidates_separates_winner_from_typical(scored_cfg: DictConfig) -> None:
    """The incumbent is excluded from the candidate median."""
    row = compare.winner_vs_candidates([_run("random", 0, [0.1, 0.9, 0.8, 0.7])], scored_cfg).iloc[
        0
    ]
    assert row["incumbent"] == pytest.approx(0.1)
    assert row["candidate_median"] == pytest.approx(0.8)
    assert row["winner"] == pytest.approx(0.9)


def test_paired_method_gain_pairs_by_seed(scored_cfg: DictConfig) -> None:
    """A seed only one method ran is left out of the pairs."""
    runs = [
        _run("turbo", 0, [0.5, 0.9]),
        _run("random", 0, [0.5, 0.7]),
        _run("turbo", 1, [0.5, 0.8]),
        _run("random", 1, [0.5, 0.7]),
        _run("turbo", 2, [0.5, 0.9]),
    ]
    row = compare.paired_method_gain(runs, scored_cfg).iloc[0]
    assert row["n_pairs"] == 2
    assert row["mean_gain"] == pytest.approx(0.15)
    assert row["method_better"] == 2
