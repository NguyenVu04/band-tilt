"""The four KPI definitions, and the reductions they share."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import band_priority_score, capacity, hole_rate, weak_rate
from src.kpi.serving import overlap_neighbors
from src.kpi.tiles import tile_index


@pytest.fixture
def cfg():
    """The thresholds the KPIs read, without composing the whole config."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "band_priority": {"hi": 3.0, "lo": 1.0},
                "capacity": {
                    "rsrp_threshold_dbm": -100.0,
                    # Low enough that no fixture UE is blocked unless it asks to be.
                    "throughput_per_ue_bps": 1.0,
                    "bands": {"hi": {"scs_hz": 15000}, "lo": {"scs_hz": 15000}},
                },
            },
            "simulation": {
                "radio_map": {
                    "temperature": 290.0,
                    "bands": [
                        {"name": "hi", "bandwidth": 20e6},
                        {"name": "lo", "bandwidth": 20e6},
                    ],
                },
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
        tile_index(_mdt([{"tile_row": 0, "tile_col": 5}]), (1, 4))


# --- band priority score ---------------------------------------------------


def test_band_priority_score_counts_the_serving_band_not_the_strongest(cfg) -> None:
    """Tile 0: 'hi' clears -100 dBm and serves though 'lo' is stronger.

    Tile 1: 'hi' is below the threshold, so 'lo' serves. Three of four UEs
    stand on tile 0.
    """
    rsrp = _map([[[-95.0, -105.0]], [[-70.0, -85.0]]])
    mdt = _mdt(
        [{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 3
        + [{"t_index": 0, "tile_row": 0, "tile_col": 1}]
    )
    assert band_priority_score(rsrp, ["hi", "lo"], mdt, cfg) == pytest.approx(0.75)


def test_a_blocked_ue_counts_at_weight_zero(cfg) -> None:
    """Each band holds one UE's PRBs: the second UE takes 'lo', the third is blocked."""
    # Alone on its band, SINR is RSRP over noise; each UE then needs 0.6 PRB.
    noise = capacity.thermal_noise_dbm(290.0, 20e6)
    cfg.kpi.capacity.throughput_per_ue_bps = 0.6 * float(
        capacity.prb_rate_bps(-80.0 - noise, 180_000.0)
    )
    cfg.simulation.transmitters.cells[0].max_prb = {"hi": 1, "lo": 1}
    rsrp = _map([[[-80.0]], [[-80.0]]])
    mdt = _mdt([{"t_index": 0, "tile_row": 0, "tile_col": 0}] * 3)
    assert band_priority_score(rsrp, ["hi", "lo"], mdt, cfg) == pytest.approx(1.0 / 3.0)


def test_a_ue_standing_on_a_hole_is_excluded_from_both_sums(cfg) -> None:
    """No band serves it, so it can neither raise nor lower the score."""
    rsrp = _map([[[-80.0, -130.0]], [[-90.0, -140.0]]])
    served_only = _mdt([{"t_index": 0, "tile_row": 0, "tile_col": 0}])
    with_hole = _mdt(
        [{"t_index": 0, "tile_row": 0, "tile_col": 0}, {"t_index": 0, "tile_row": 0, "tile_col": 1}]
    )
    assert band_priority_score(rsrp, ["hi", "lo"], with_hole, cfg) == pytest.approx(
        band_priority_score(rsrp, ["hi", "lo"], served_only, cfg)
    )
