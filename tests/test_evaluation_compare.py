"""Comparison tables, especially the tolerance verdict that keeps them honest."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import compare
from src.evaluation.runs import Run
from src.optim.objective import KPI_NAMES, KpiVector

TOLERANCE = {
    "hole_rate": 0.002,
    "overlap_rate": 0.001,
    "served_ratio": 0.001,
    "weak_rate": 0.004,
}


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
    """Thresholds and tolerances, without composing the whole config."""
    return OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0, "tolerance": dict(TOLERANCE)}}
    )


@pytest.fixture
def incumbent() -> KpiVector:
    """A plausible starting point, near the committed configuration's scores."""
    return KpiVector(
        hole_rate=0.10,
        overlap_rate=0.28,
        served_ratio=0.009,
        weak_rate=0.12,
    )


def test_delta_table_is_in_priority_order(cfg: DictConfig, incumbent: KpiVector) -> None:
    """Reading top to bottom is reading the order the winner was decided in."""
    table = compare.delta_table(incumbent, incumbent, cfg)
    assert table["kpi"].tolist() == list(KPI_NAMES)


def test_identical_configurations_tie_everywhere(cfg: DictConfig, incumbent: KpiVector) -> None:
    """A zero delta is a tie, not an improvement of zero."""
    table = compare.delta_table(incumbent, incumbent, cfg)
    assert (table["verdict"] == compare.TIE).all()


def test_a_change_inside_the_tolerance_is_a_tie(cfg: DictConfig, incumbent: KpiVector) -> None:
    """Half a tolerance is not distinguishable from solver noise."""
    after = dataclasses.replace(incumbent, hole_rate=incumbent.hole_rate - 0.001)
    table = compare.delta_table(incumbent, after, cfg).set_index("kpi")
    assert table.loc["hole_rate", "verdict"] == compare.TIE


def test_a_change_past_the_tolerance_reads_by_direction(
    cfg: DictConfig, incumbent: KpiVector
) -> None:
    """Both are minimised, so down is better and up is worse."""
    after = dataclasses.replace(
        incumbent,
        hole_rate=incumbent.hole_rate - 0.05,
        weak_rate=incumbent.weak_rate + 0.05,
    )
    table = compare.delta_table(incumbent, after, cfg).set_index("kpi")
    assert table.loc["hole_rate", "verdict"] == compare.BETTER
    assert table.loc["weak_rate", "verdict"] == compare.WORSE


def test_the_maximised_kpi_reads_the_other_way(cfg: DictConfig, incumbent: KpiVector) -> None:
    """The KPI where up is better, and the usual place a sign error hides."""
    after = dataclasses.replace(incumbent, served_ratio=incumbent.served_ratio + 0.05)
    table = compare.delta_table(incumbent, after, cfg).set_index("kpi")
    assert table.loc["served_ratio", "verdict"] == compare.BETTER
    assert table.loc["served_ratio", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """One KPI is maximised; a second, or none, would be a sign error."""
    maximised = [name for name in KPI_NAMES if compare.direction(name) == "maximise"]
    assert maximised == ["served_ratio"]


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


WEIGHTS = {"hole_rate": 1.0, "overlap_rate": 0.0, "served_ratio": 1.0, "weak_rate": 0.0}


@pytest.fixture
def scored_cfg() -> DictConfig:
    """Tolerances plus weights that score ``served_ratio - hole_rate``."""
    return OmegaConf.create({"kpi": {"tolerance": dict(TOLERANCE), "weights": dict(WEIGHTS)}})


def _run(method: str, seed: int, holes: list[float], phases: list[str] | None = None) -> Run:
    """A run whose score is ``-hole_rate`` per evaluation, row 0 the incumbent."""
    n = len(holes)
    history = pd.DataFrame(
        {
            "iteration": range(n),
            "phase": phases or ["incumbent"] + ["init"] * (n - 1),
            "hole_rate": holes,
            "overlap_rate": [0.3] * n,
            "served_ratio": [0.0] * n,
            "weak_rate": [0.1] * n,
        }
    )
    best = int(np.argmin(holes))
    kpi = {name: float(history.loc[best, name]) for name in KPI_NAMES}
    incumbent = {name: float(history.loc[0, name]) for name in KPI_NAMES}
    meta = {
        "best_iteration": best,
        "best_kpi": kpi,
        "incumbent_kpi": incumbent,
        "config": {"optim": {"seed": seed}},
    }
    return Run(method, f"run{seed}", Path("."), history, pd.DataFrame(), meta)


def test_seed_summary_interval_brackets_the_mean(scored_cfg: DictConfig) -> None:
    """Two seeds give a finite interval centred on the mean winner."""
    runs = [_run("turbo", 0, [0.5, 0.2]), _run("turbo", 1, [0.5, 0.4])]
    table = compare.seed_summary(runs, scored_cfg).set_index("kpi")
    assert table.loc["hole_rate", "mean"] == pytest.approx(0.3)
    assert table.loc["hole_rate", "ci95_low"] < 0.3 < table.loc["hole_rate", "ci95_high"]
    assert table.loc["hole_rate", "verdict"] == compare.BETTER
    assert table.loc["score", "direction"] == "maximise"
    assert table.loc["score", "verdict"] == ""


def test_winner_vs_candidates_separates_winner_from_typical(scored_cfg: DictConfig) -> None:
    """The incumbent is excluded from the candidate median."""
    row = compare.winner_vs_candidates([_run("random", 0, [0.9, 0.1, 0.2, 0.3])], scored_cfg).iloc[
        0
    ]
    assert row["incumbent"] == pytest.approx(-0.9)
    assert row["candidate_median"] == pytest.approx(-0.2)
    assert row["winner"] == pytest.approx(-0.1)


def test_weight_sensitivity_detects_a_changed_winner(scored_cfg: DictConfig) -> None:
    """Weighting only served ratio (constant here) ties everything, so row 0 wins instead."""
    table = compare.weight_sensitivity(
        [_run("turbo", 0, [0.5, 0.2])], scored_cfg, {"served_only": (0.0, 0.0, 1.0, 0.0)}
    ).set_index("scheme")
    assert table.loc["configured", "same_winner_share"] == pytest.approx(1.0)
    assert table.loc["served_only", "same_winner_share"] == pytest.approx(0.0)


def test_paired_method_gain_pairs_by_seed(scored_cfg: DictConfig) -> None:
    """A seed only one method ran is left out of the pairs."""
    runs = [
        _run("turbo", 0, [0.5, 0.1]),
        _run("random", 0, [0.5, 0.3]),
        _run("turbo", 1, [0.5, 0.2]),
        _run("random", 1, [0.5, 0.3]),
        _run("turbo", 2, [0.5, 0.1]),
    ]
    row = compare.paired_method_gain(runs, scored_cfg).iloc[0]
    assert row["n_pairs"] == 2
    assert row["mean_gain"] == pytest.approx(0.15)
    assert row["method_better"] == 2


def test_retraced_gain_and_solver_noise_read_the_retrace(scored_cfg: DictConfig) -> None:
    """Gains pair by solver seed; the pooled spread ignores the gap between configurations."""
    frame = pd.DataFrame(
        {
            "configuration": ["incumbent", "turbo", "incumbent", "turbo"],
            "solver_seed": [0, 0, 1, 1],
            "hole_rate": [0.50, 0.30, 0.52, 0.32],
            "overlap_rate": [0.3] * 4,
            "served_ratio": [0.0] * 4,
            "weak_rate": [0.1] * 4,
        }
    )
    gain = compare.retraced_gain(frame, scored_cfg).iloc[0]
    assert gain["mean_gain"] == pytest.approx(0.2)
    assert gain["positive_share"] == pytest.approx(1.0)
    noise = compare.solver_noise(frame, scored_cfg).set_index("kpi")
    assert noise.loc["hole_rate", "pooled_std"] == pytest.approx(np.sqrt(0.0002))
