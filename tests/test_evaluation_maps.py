"""Spatial reductions, on hand-built maps where the answer is known by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import maps
from src.kpi import hole_rate, weak_rate


@pytest.fixture
def cfg() -> DictConfig:
    """The KPI thresholds, without composing the whole config."""
    return OmegaConf.create(
        {"kpi": {"hole_dbm": -120.0, "weak_dbm": -90.0, "overlap_margin_db": 6.0}}
    )


def rsrp_from(best: np.ndarray) -> np.ndarray:
    """A one-band, one-transmitter map whose best server is ``best``."""
    return best[None, None, :, :].astype(float)


def test_coverage_class_cuts_at_the_configured_thresholds(cfg: DictConfig) -> None:
    """Both thresholds are inclusive upper bounds, matching src.kpi."""
    best = np.array([[-130.0, -120.0, -119.0, -90.0, -89.0]])
    classes = maps.coverage_class(rsrp_from(best), cfg)
    assert classes.tolist() == [[maps.HOLE, maps.HOLE, maps.WEAK, maps.WEAK, maps.GOOD]]


def test_no_path_counts_as_a_hole(cfg: DictConfig) -> None:
    """A tile no ray reached is uncovered, not missing data."""
    best = np.array([[np.nan, -50.0]])
    assert maps.coverage_class(rsrp_from(best), cfg)[0, 0] == maps.HOLE


def test_coverage_table_tile_share_is_the_kpi(cfg: DictConfig) -> None:
    """The same quantity by two routes; drift here means the reduction is wrong."""
    rng = np.random.default_rng(0)
    rsrp = rng.uniform(-140.0, -60.0, size=(2, 3, 8, 9))
    counts = rng.integers(0, 5, size=(8, 9))

    table = maps.coverage_table(rsrp, counts, cfg).set_index("coverage")
    assert table.loc["hole", "tile_share"] == pytest.approx(hole_rate(rsrp, cfg))
    assert table.loc["weak", "tile_share"] == pytest.approx(weak_rate(rsrp, cfg))
    assert table["tile_share"].sum() == pytest.approx(1.0)
    assert table["demand_share"].sum() == pytest.approx(1.0)


def test_demand_weighting_can_disagree_with_area(cfg: DictConfig) -> None:
    """The finding the demand view exists for: holes where nobody stands."""
    best = np.full((1, 10), -50.0)
    best[0, :5] = -130.0  # half the area is a hole
    counts = np.zeros((1, 10), dtype=int)
    counts[0, 5:] = 10  # every UE stands on the covered half

    table = maps.coverage_table(rsrp_from(best), counts, cfg).set_index("coverage")
    assert table.loc["hole", "tile_share"] == pytest.approx(0.5)
    assert table.loc["hole", "demand_share"] == pytest.approx(0.0)


def test_change_mask_reports_crossings_not_gradients(cfg: DictConfig) -> None:
    """Column 2 gains 5 dB but stays a hole, so no KPI can see it."""
    before = np.array([[-130.0, -100.0, -130.0, -125.0]])
    after = np.array([[-100.0, -130.0, -125.0, -100.0]])
    assert maps.change_mask(before, after, cfg).tolist() == [[1, -1, 0, 1]]


def test_underserved_needs_both_demand_and_poor_coverage(cfg: DictConfig) -> None:
    """Dark and empty is not a problem; dark and busy is."""
    best = np.array([[-130.0, -130.0, -50.0, -50.0]])
    counts = np.array([[0, 100, 0, 100]])
    flagged = maps.underserved(rsrp_from(best), counts, cfg, quantile=0.5)
    assert flagged.tolist() == [[False, True, False, False]]


def test_underserved_survives_an_empty_demand_raster(cfg: DictConfig) -> None:
    """A quantile over nothing must not raise."""
    best = np.full((2, 2), -130.0)
    flagged = maps.underserved(rsrp_from(best), np.zeros((2, 2), dtype=int), cfg)
    assert not flagged.any()


def test_coverage_cdf_ends_at_one_and_tracks_demand() -> None:
    """All demand sits on the strongest tile, so it arrives only at the last step."""
    best = np.array([[-130.0, -110.0, -70.0]])
    counts = np.array([[0, 0, 7]])
    levels, tile_share, demand_share = maps.coverage_cdf(best, counts)

    assert levels.tolist() == [-130.0, -110.0, -70.0]
    assert tile_share[-1] == pytest.approx(1.0)
    assert demand_share[-1] == pytest.approx(1.0)
    assert demand_share[0] == pytest.approx(0.0)


def test_coverage_cdf_places_unreached_tiles_at_the_floor() -> None:
    """An infinite level would make the axis unplottable."""
    best = np.array([[-np.inf, -100.0]])
    levels, _, _ = maps.coverage_cdf(best, np.ones((1, 2), dtype=int))
    assert np.isfinite(levels).all()
    assert levels[0] == pytest.approx(-100.0)


def test_demand_is_ue_count_times_prbs_per_ue(cfg: DictConfig) -> None:
    """Two UEs on one tile in one interval need twice one UE's PRBs."""
    from src.kpi import capacity

    cfg.kpi.band_priority = {"b": 1.0}
    cfg.kpi.capacity = {
        "rsrp_threshold_dbm": -110.0,
        "throughput_per_ue_bps": 1e6,
        "noise_figure_db": 9.0,
        "bands": {"b": {"scs_hz": 15000, "n_prb": 1000}},
    }
    rsrp = np.full((1, 1, 3, 4), -90.0)
    mdt = pd.DataFrame({"t_index": [0, 0, 0], "tile_row": [0, 0, 2], "tile_col": [1, 1, 3]})
    counts = maps.demand(rsrp, ["b"], mdt, cfg)

    noise = capacity.noise_per_re_dbm(15000.0, 9.0)
    per_ue = capacity.prb_per_ue(1e6, capacity.prb_rate_bps(-90.0 - noise, 180_000.0))
    assert counts.shape == (3, 4)
    assert counts[0, 1] == pytest.approx(capacity.prb_required(2, per_ue))
    assert counts[2, 3] == pytest.approx(per_ue)
    assert counts.sum() == pytest.approx(3 * per_ue)


def test_extent_spans_whole_tiles() -> None:
    """The grid rounds up, so the extent overhangs the scene rather than clipping."""
    radio = {"origin_x": -100.0, "origin_y": -50.0, "tile_size_m": 20.0, "n_cols": 5, "n_rows": 3}
    assert maps.extent_of(radio) == [-100.0, 0.0, -50.0, 10.0]
    assert maps.grid_shape(radio) == (3, 5)
