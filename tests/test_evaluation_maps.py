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


def test_serving_band_prefers_a_band_above_threshold_else_the_strongest() -> None:
    """Tile 0: preferred band 0 clears -100 dBm. Tile 1: neither does, band 1 is stronger.

    Tile 2 has no path on any layer.
    """
    rsrp = np.array(
        [
            [[[-95.0, -110.0, np.nan]], [[-99.0, -120.0, np.nan]]],
            [[[-60.0, -105.0, np.nan]], [[-70.0, -130.0, np.nan]]],
        ]
    )
    band = maps.serving_band(rsrp, np.array([0, 1]), -100.0, -120.0)
    assert band.tolist() == [[0, 1, -1]]


def test_serving_band_never_serves_on_a_layer_at_the_hole_threshold() -> None:
    """A layer exactly at min_rsrp_dbm is no candidate, as in the serving rule."""
    rsrp = np.array([[[[-120.0]]], [[[-125.0]]]])
    assert maps.serving_band(rsrp, np.array([0, 1]), -120.0, -120.0).tolist() == [[-1]]


def test_demand_is_ue_count_times_prbs_per_ue(cfg: DictConfig) -> None:
    """Two UEs on one tile in one interval need twice one UE's PRBs."""
    from src.kpi import capacity

    cfg.kpi.capacity = {
        "band_preference": ["b"],
        "rsrp_threshold_dbm": -110.0,
        "max_admission_utilisation": 1.0,
        "throughput_per_ue_bps": 1e6,
        "bands": {"b": {"scs_hz": 15000}},
    }
    cfg.simulation = {
        "seed": 0,
        "transmitters": {
            "cells": [
                {
                    "name": "c0",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 30.0,
                    "azimuth_deg": 0.0,
                    "tilt": {},
                    "max_prb": {"b": 1000},
                }
            ]
        },
    }
    rsrp = np.full((1, 1, 3, 4), -90.0)
    sinr = np.full(rsrp.shape, 20.0)
    ue = pd.DataFrame(
        {"t_index": [0, 0, 0], "t_s": [0.0] * 3, "tile_row": [0, 0, 2], "tile_col": [1, 1, 3]}
    )
    counts = capacity.demand_prb(rsrp, sinr, ["b"], ue, cfg)

    per_ue = capacity._prb_per_ue(1e6, capacity._prb_rate_bps(20.0, 180_000.0))
    assert counts.shape == (3, 4)
    assert counts[0, 1] == pytest.approx(2 * per_ue)
    assert counts[2, 3] == pytest.approx(per_ue)
    assert counts.sum() == pytest.approx(3 * per_ue)


def test_extent_spans_whole_tiles() -> None:
    """The grid rounds up, so the extent overhangs the scene rather than clipping."""
    radio = {"origin_x": -100.0, "origin_y": -50.0, "tile_size_m": 20.0, "n_cols": 5, "n_rows": 3}
    assert maps.extent_of(radio) == [-100.0, 0.0, -50.0, 10.0]
    assert maps.grid_shape(radio) == (3, 5)
