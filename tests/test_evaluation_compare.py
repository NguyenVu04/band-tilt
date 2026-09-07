"""Comparison tables, especially the tolerance verdict that keeps them honest."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import compare
from src.optim.objective import KPI_NAMES, KpiVector

TOLERANCE = {
    "hole_rate": 0.002,
    "overlap_rate": 0.001,
    "mean_overlap_neighbors": 0.012,
    "band_priority_score": 0.001,
    "weak_rate": 0.004,
}


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
        mean_overlap_neighbors=0.68,
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
    """band_priority_score is the one KPI where up is better."""
    after = dataclasses.replace(incumbent, band_priority_score=incumbent.band_priority_score + 0.05)
    table = compare.delta_table(incumbent, after, cfg).set_index("kpi")
    assert table.loc["band_priority_score", "verdict"] == compare.BETTER
    assert table.loc["band_priority_score", "direction"] == "maximise"


def test_direction_names_every_kpi() -> None:
    """Exactly one KPI is maximised; a second would be a sign error."""
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
