"""The KPI definitions, and the reductions they share."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy.special import expit

from src.kpi import (
    edge_rsrp_dbm,
    hole_desirability,
    hole_rate,
    overlap_desirability,
    overlap_rate,
    served_desirability,
    served_ratio,
    weak_rate,
)
from src.kpi.capacity import _tile_index
from src.kpi.overlap import overlap_neighbors


@pytest.fixture
def cfg():
    """The thresholds the KPIs read, without composing the whole config."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "quality": {"edge_percentile": 5.0},
                "soft": {
                    "hole_rate": {"target": -120.0, "temperature": 5.0},
                    "overlap_rate": {"target": 0.5, "temperature": 0.5},
                    "served_ratio": {"target": 0.5, "temperature": 0.1},
                },
                "capacity": {
                    "band_preference": ["hi", "lo"],
                    "rsrp_threshold_dbm": -100.0,
                    # Low enough that no fixture UE is blocked unless it asks to be.
                    "throughput_per_ue_bps": 1.0,
                    "bands": {"hi": {"scs_hz": 15000}, "lo": {"scs_hz": 15000}},
                },
            },
            "simulation": {
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
                            "max_prb": {"hi": 100, "lo": 100},
                        }
                    ]
                },
            },
        }
    )


def _map(values: list[list[list[float]]]) -> np.ndarray:
    """A radio map from nested ``[band][tx]`` lists of per-tile values.

    Each innermost list becomes the single row of a ``1 x n`` grid, so every
    fixture below reads as a table of cell-band layers against locations.
    """
    return np.array(values, dtype=float)[:, :, None, :]


def _sinr(rsrp: np.ndarray) -> np.ndarray:
    """A SINR map of 0 dB wherever ``rsrp`` has a path: one PRB carries 180 kbit/s."""
    return np.where(np.isfinite(rsrp), 0.0, np.nan)


def _mdt(rows: list[dict[str, float]]) -> pd.DataFrame:
    """An MDT frame carrying only the columns the KPIs read."""
    return pd.DataFrame(rows)


# --- thresholds ------------------------------------------------------------


def test_hole_and_weak_thresholds_are_inclusive_upper_bounds(cfg) -> None:
    """A tile exactly on a threshold falls in the class below it."""
    rsrp = _map([[[-120.0, -119.9, -90.0, -89.9]]])
    assert hole_rate(rsrp, cfg) == pytest.approx(0.25)
    # -119.9 and -90.0 are weak; -89.9 is strong and -120.0 is a hole.
    assert weak_rate(rsrp, cfg) == pytest.approx(0.5)


def test_a_tile_no_transmitter_reaches_is_a_hole(cfg) -> None:
    """NaN is the ray tracer's no-path marker, not a missing value to skip."""
    rsrp = _map([[[np.nan, -80.0]]])
    assert hole_rate(rsrp, cfg) == pytest.approx(0.5)
    assert weak_rate(rsrp, cfg) == pytest.approx(0.0)


def test_a_map_reaching_nothing_is_entirely_holes(cfg) -> None:
    """The soft hole term rests on this: no path must never read as coverage."""
    assert hole_rate(_map([[[np.nan, np.nan]]]), cfg) == pytest.approx(1.0)


# --- edge RSRP -------------------------------------------------------------


def test_edge_rsrp_ignores_the_locations_with_no_coverage(cfg) -> None:
    """Conditional on coverage by design: a hole has no serving RSRP to report.

    Taken at the 0th percentile so the assertion is the weakest covered tile
    itself, with no interpolation between order statistics to read past.
    """
    cfg.kpi.quality.edge_percentile = 0.0
    # No path, exactly on the hole threshold, then three covered tiles. The
    # first two are excluded, so the weakest reported is -100 and not -inf.
    rsrp = _map([[[np.nan, -120.0, -100.0, -90.0, -80.0]]])
    assert edge_rsrp_dbm(rsrp, cfg) == pytest.approx(-100.0)


def test_edge_rsrp_of_a_dead_map_is_minus_infinity(cfg) -> None:
    """Total outage has to order below every configuration that covers something."""
    assert edge_rsrp_dbm(_map([[[np.nan, np.nan]]]), cfg) == -np.inf


# --- overlap ---------------------------------------------------------------


def test_overlap_counts_within_each_band_and_sums_across_them(cfg) -> None:
    """The co-band rule: a strong other-band layer is not an overlapping neighbour.

    Tile 0 has two transmitters within the margin on each band, so each band
    contributes one neighbour beyond its own serving cell. Tile 1 has one
    reachable transmitter per band and the two bands sit 20 dB apart - close
    enough to overlap if bands were compared against each other, which the
    co-band rule does not do.
    """
    rsrp = _map(
        [
            [[-80.0, -80.0], [-84.0, -130.0]],
            [[-90.0, -100.0], [-94.0, -130.0]],
        ]
    )
    assert overlap_neighbors(rsrp, cfg).tolist() == [[2, 0]]


def test_an_uncovered_band_contributes_no_neighbours(cfg) -> None:
    """Subtracting the serving cell must not take an uncovered band below zero."""
    rsrp = _map([[[-130.0], [-130.0]], [[-80.0], [-140.0]]])
    assert overlap_neighbors(rsrp, cfg).tolist() == [[0]]


# --- tiles -----------------------------------------------------------------


def test_tile_index_rejects_a_ue_off_the_map() -> None:
    """A UE outside the grid means the MDT and the map are different scenarios."""
    with pytest.raises(ValueError, match="different grids"):
        _tile_index(_mdt([{"tile_row": 0, "tile_col": 5}]), (1, 4))


# --- served ratio ----------------------------------------------------------


def test_every_covered_ue_with_room_is_served(cfg) -> None:
    """Three UEs on covered tiles, PRBs to spare: all served."""
    rsrp = _map([[[-95.0, -105.0]], [[-70.0, -85.0]]])
    mdt = _mdt(
        [{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 2
        + [{"t_index": 0, "tile_row": 0, "tile_col": 1}]
    )
    assert served_ratio(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) == pytest.approx(1.0)


def test_a_ue_on_a_hole_counts_as_not_served(cfg) -> None:
    """Tile 1 is heard only at or below -120 dBm, so no layer may serve it."""
    rsrp = _map([[[-80.0, -120.0]], [[-90.0, -140.0]]])
    mdt = _mdt(
        [{"t_index": 0, "tile_row": 0, "tile_col": 0}, {"t_index": 0, "tile_row": 0, "tile_col": 1}]
    )
    assert served_ratio(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) == pytest.approx(0.5)


def test_a_blocked_ue_counts_as_not_served(cfg) -> None:
    """Each band holds one UE's PRBs: two of three UEs fit."""
    # At 0 dB one PRB carries 180 kbit/s, so each UE needs 0.6 PRB.
    cfg.kpi.capacity.throughput_per_ue_bps = 0.6 * 180_000.0
    cfg.simulation.transmitters.cells[0].max_prb = {"hi": 1, "lo": 1}
    rsrp = _map([[[-80.0]], [[-80.0]]])
    mdt = _mdt([{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 3)
    assert served_ratio(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) == pytest.approx(2.0 / 3.0)


def test_served_ratio_rejects_an_empty_mdt(cfg) -> None:
    """No UE, no denominator."""
    rsrp = _map([[[-80.0]], [[-80.0]]])
    empty = pd.DataFrame(columns=["t_index", "tile_row", "tile_col"])
    with pytest.raises(ValueError, match="no UE"):
        served_ratio(rsrp, _sinr(rsrp), ["hi", "lo"], empty, cfg)


def test_served_desirability_averages_per_tile_not_per_report(cfg) -> None:
    """The point of softening before averaging: a busy tile must not outvote a quiet one.

    Tile 0 takes three reports and serves them all; tile 1 takes one and serves
    none. The global share is 3/4, but every tile gets one vote here, so the
    result is the mean of the two tiles' softened ratios - and lower.
    """
    cfg.kpi.capacity.throughput_per_ue_bps = 0.6 * 180_000.0
    cfg.simulation.transmitters.cells[0].max_prb = {"hi": 100, "lo": 100}
    # Tile 1 is a hole, so its reports cannot be served at all.
    rsrp = _map([[[-80.0, -130.0]], [[-80.0, -130.0]]])
    mdt = _mdt(
        [{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 3
        + [{"t_index": 0, "tile_row": 0, "tile_col": 1}]
    )

    assert served_ratio(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) == pytest.approx(0.75)
    # Per tile: served ratios of 1.0 and 0.0, softened about a target of 0.5.
    expected = 0.5 * (expit((1.0 - 0.5) / 0.1) + expit((0.0 - 0.5) / 0.1))
    assert served_desirability(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) == pytest.approx(expected)
    assert served_desirability(rsrp, _sinr(rsrp), ["hi", "lo"], mdt, cfg) < 0.75


def test_served_desirability_ignores_tiles_carrying_no_report(cfg) -> None:
    """Empty ground has no served ratio; counting it either way would be a bias."""
    cfg.kpi.capacity.throughput_per_ue_bps = 0.6 * 180_000.0
    narrow = _map([[[-80.0]], [[-80.0]]])
    wide = _map([[[-80.0, -130.0, -130.0]], [[-80.0, -130.0, -130.0]]])
    mdt = _mdt([{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 2)

    assert served_desirability(narrow, _sinr(narrow), ["hi", "lo"], mdt, cfg) == pytest.approx(
        served_desirability(wide, _sinr(wide), ["hi", "lo"], mdt, cfg)
    )


def test_served_desirability_rejects_an_empty_mdt(cfg) -> None:
    """No UE, no tiles to average over."""
    rsrp = _map([[[-80.0]], [[-80.0]]])
    empty = pd.DataFrame(columns=["t_index", "tile_row", "tile_col"])
    with pytest.raises(ValueError, match="no UE"):
        served_desirability(rsrp, _sinr(rsrp), ["hi", "lo"], empty, cfg)


def test_hole_desirability_registers_depth_the_hard_rate_cannot_see(cfg) -> None:
    """The whole reason it exists: a step threshold is blind below itself.

    Both maps put every tile under the hole threshold, so ``hole_rate`` calls
    them identical. One is 1 dB short of coverage and the other 40 dB short,
    which is the difference a tilt change actually moves.
    """
    shallow = _map([[[-121.0, -121.0]]])
    deep = _map([[[-160.0, -160.0]]])

    assert hole_rate(shallow, cfg) == hole_rate(deep, cfg) == pytest.approx(1.0)
    assert hole_desirability(shallow, cfg) > hole_desirability(deep, cfg)


def test_hole_desirability_is_half_on_the_target(cfg) -> None:
    """The target is where the curve responds most, so it is worth pinning."""
    assert hole_desirability(_map([[[-120.0]]]), cfg) == pytest.approx(0.5)


def test_hole_desirability_scores_a_no_path_tile_zero(cfg) -> None:
    """``-inf`` must read as the worst possible tile, not drop out of the mean."""
    assert hole_desirability(_map([[[np.nan]]]), cfg) == pytest.approx(0.0)


def test_overlap_desirability_separates_counts_the_rate_collapses(cfg) -> None:
    """``overlap_rate`` asks only whether a neighbour exists; the count is the signal.

    Tile 0 is crowded by one neighbour within the margin, tile 1 by two. Both
    are simply "overlapping" to the rate.
    """
    one = _map([[[-80.0], [-82.0], [-140.0]]])
    two = _map([[[-80.0], [-82.0], [-83.0]]])

    assert overlap_rate(one, cfg) == overlap_rate(two, cfg) == pytest.approx(1.0)
    assert overlap_desirability(one, cfg) > overlap_desirability(two, cfg)


def test_overlap_desirability_is_best_where_nothing_crowds(cfg) -> None:
    """A tile with one dominant server has no neighbours, so it scores near one."""
    alone = _map([[[-80.0], [-140.0], [-140.0]]])
    assert overlap_desirability(alone, cfg) == pytest.approx(0.7311, abs=1e-4)
