"""Comparison tables, especially the tolerance verdict that keeps them honest."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import compare
from src.optim.objective import KPI_NAMES, KpiVector

TOLERANCE = {
    "hole_rate": 0.002,
    "overlap_rate": 0.001,
    "band_priority_score": 0.001,
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
        band_priority_score=0.009,
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
    after = dataclasses.replace(incumbent, band_priority_score=incumbent.band_priority_score + 0.05)
    table = compare.delta_table(incumbent, after, cfg).set_index("kpi")
    assert table.loc["band_priority_score", "verdict"] == compare.BETTER
    assert table.loc["band_priority_score", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """One KPI is maximised; a second, or none, would be a sign error."""
    maximised = [name for name in KPI_NAMES if compare.direction(name) == "maximise"]
    assert maximised == ["band_priority_score"]


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
    merged = compare.coverage_comparison({"incumbent": table, "mobo": table})
    assert list(merged.columns) == [
        "coverage",
        "incumbent_tile",
        "incumbent_demand",
        "mobo_tile",
        "mobo_demand",
    ]
    assert len(merged) == 3


def test_coverage_comparison_of_nothing_is_empty() -> None:
    """No runs is a legitimate state, not an error."""
    assert compare.coverage_comparison({}).empty
