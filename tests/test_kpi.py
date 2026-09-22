"""The KPI definitions, and the reductions they share."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import (
    hole_rate,
    overlap_neighbor_mean,
    rsrp_percentile_dbm,
    served_rate,
    sinr_percentile_db,
    weak_rate,
)
from src.kpi.capacity import _tile_index, serve_intervals
from src.kpi.overlap import effective_coverage, overlap_neighbors, overlap_neighbors_per_band


@pytest.fixture
def cfg():
    """The thresholds the KPIs read, without composing the whole config."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "capacity": {
                    "band_preference": ["hi", "lo"],
                    "rsrp_threshold_dbm": -100.0,
                    "max_admission_utilisation": 1.0,
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


def _ue(rows: list[dict[str, float]]) -> pd.DataFrame:
    """A UE frame carrying only the columns the KPIs read."""
    return pd.DataFrame(rows)


def _served(rsrp: np.ndarray, ue: pd.DataFrame, cfg) -> pd.DataFrame:
    """The serving assignment the served rate reduces."""
    return serve_intervals(rsrp, _sinr(rsrp), ["hi", "lo"], ue, cfg)


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


# --- RSRP and SINR percentiles ---------------------------------------------


def test_rsrp_percentile_ignores_the_locations_with_no_coverage(cfg) -> None:
    """Conditional on coverage by design: a hole has no serving RSRP to report.

    Taken at the 0th percentile so the assertion is the weakest covered tile
    itself, with no interpolation between order statistics to read past.
    """
    # No path, exactly on the hole threshold, then three covered tiles. The
    # first two are excluded, so the weakest reported is -100 and not -inf.
    rsrp = _map([[[np.nan, -120.0, -100.0, -90.0, -80.0]]])
    assert rsrp_percentile_dbm(rsrp, cfg, 0.0) == pytest.approx(-100.0)
    assert rsrp_percentile_dbm(rsrp, cfg, 50.0) == pytest.approx(-90.0)


def test_rsrp_percentile_of_a_dead_map_is_minus_infinity(cfg) -> None:
    """Total outage has to order below every configuration that covers something."""
    assert rsrp_percentile_dbm(_map([[[np.nan, np.nan]]]), cfg, 5.0) == -np.inf


def test_sinr_percentile_reads_the_layer_the_rsrp_percentile_reads(cfg) -> None:
    """The best server on tile 0 is band 'lo', so its SINR is the one reported."""
    rsrp = _map([[[-100.0, -95.0]], [[-80.0, -130.0]]])
    sinr = _map([[[3.0, 9.0]], [[12.0, -5.0]]])
    # Tile 0: 'lo' at -80 serves, SINR 12. Tile 1: 'hi' at -95 serves, SINR 9.
    assert sinr_percentile_db(rsrp, sinr, cfg, 0.0) == pytest.approx(9.0)
    assert sinr_percentile_db(rsrp, sinr, cfg, 100.0) == pytest.approx(12.0)


def test_sinr_percentile_of_a_dead_map_is_minus_infinity(cfg) -> None:
    """Nothing covered, nothing to take a percentile of."""
    rsrp = _map([[[np.nan, np.nan]]])
    assert sinr_percentile_db(rsrp, _sinr(rsrp), cfg, 50.0) == -np.inf


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


def test_overlap_neighbor_mean_averages_over_covered_tiles_only(cfg) -> None:
    """Tile 0 carries two neighbours, tile 1 is a hole: the mean is 2, not 1."""
    rsrp = _map(
        [
            [[-80.0, -130.0], [-84.0, -130.0]],
            [[-90.0, -130.0], [-94.0, -130.0]],
        ]
    )
    assert overlap_neighbors(rsrp, cfg).tolist() == [[2, 0]]
    assert overlap_neighbor_mean(rsrp, cfg) == pytest.approx(2.0)


def test_overlap_neighbor_mean_of_a_dead_map_is_nan(cfg) -> None:
    """No coverage is not the same statement as no crowding."""
    assert np.isnan(overlap_neighbor_mean(_map([[[np.nan, np.nan]]]), cfg))


def test_the_per_band_counts_are_what_the_total_sums(cfg) -> None:
    """The objective reads one band's column of the same array the KPI sums."""
    rsrp = _map(
        [
            [[-80.0, -80.0], [-84.0, -130.0]],
            [[-90.0, -100.0], [-94.0, -130.0]],
        ]
    )
    per_band = overlap_neighbors_per_band(rsrp, cfg)
    assert per_band.tolist() == [[[1, 0]], [[1, 0]]]
    assert per_band.sum(axis=0).tolist() == overlap_neighbors(rsrp, cfg).tolist()


# --- effective coverage, the quantity the objective scores --------------------


def _u(multiplicity: float) -> float:
    """``lambda e^(1 - lambda)``, before the strength factor scales it."""
    return multiplicity * math.exp(1.0 - multiplicity)


def test_effective_coverage_is_the_contraharmonic_mean_over_bands(cfg) -> None:
    """Tile 0: 'hi' crowded, 'lo' clean. Tile 1: 'hi' clean, 'lo' at two-thirds strength.

    Each band is weighted by its own utility, so neither tile reaches its best band.
    """
    rsrp = _map(
        [
            [[-80.0, -80.0], [-84.0, -130.0]],
            [[-90.0, -100.0], [-110.0, -130.0]],
        ]
    )
    crowded, clean, weaker = _u(2.0), _u(1.0), _u(1.0) * 2.0 / 3.0
    tile_0 = (crowded**2 + clean**2) / (crowded + clean)
    tile_1 = (clean**2 + weaker**2) / (clean + weaker)
    assert effective_coverage(rsrp, cfg).ravel().tolist() == pytest.approx([tile_0, tile_1])


def test_effective_coverage_counts_neighbours_within_the_winning_band(cfg) -> None:
    """Every band crowded means no clean layer to escape to."""
    rsrp = _map([[[-80.0], [-84.0]], [[-90.0], [-94.0]]])
    assert effective_coverage(rsrp, cfg).ravel().tolist() == pytest.approx([_u(2.0)])


def test_effective_coverage_is_zero_where_no_band_is_covered(cfg) -> None:
    """A hole has no serving cell to count, which is what scores it zero."""
    rsrp = _map([[[-130.0], [-130.0]], [[np.nan], [-140.0]]])
    assert effective_coverage(rsrp, cfg).tolist() == [[0.0]]


def test_effective_coverage_scales_with_strength_between_the_thresholds(cfg) -> None:
    """A lone server just above hole_dbm keeps almost none of its utility."""
    at_weak = effective_coverage(_map([[[-90.0]]]), cfg)
    halfway = effective_coverage(_map([[[-105.0]]]), cfg)
    marginal = effective_coverage(_map([[[-119.7]]]), cfg)
    assert at_weak.ravel().tolist() == pytest.approx([1.0])
    assert halfway.ravel().tolist() == pytest.approx([0.5])
    assert marginal.ravel().tolist() == pytest.approx([0.01])


def test_effective_coverage_does_not_reward_strength_above_the_weak_threshold(cfg) -> None:
    """The factor is clipped at 1, so power beyond weak_dbm buys nothing."""
    assert effective_coverage(_map([[[-40.0]]]), cfg).ravel().tolist() == pytest.approx([1.0])


def test_losing_a_layer_never_raises_effective_coverage(cfg) -> None:
    """Losing a band that scores at or above the tile's score never raises it.

    Stripping the crowded preferred band would once have moved the tile onto a
    clean lower band and scored it higher. Under ADR 0003, shedding a band that
    scores below the tile's score does raise it.
    """
    crowded = _map([[[-80.0], [-82.0]], [[-100.0], [-130.0]]])
    stripped = crowded.copy()
    stripped[0] = -130.0
    assert effective_coverage(stripped, cfg) <= effective_coverage(crowded, cfg)


# --- tiles -----------------------------------------------------------------


def test_tile_index_rejects_a_ue_off_the_map() -> None:
    """A UE outside the grid means the UE table and the map are different scenarios."""
    with pytest.raises(ValueError, match="different grids"):
        _tile_index(_ue([{"tile_row": 0, "tile_col": 5}]), (1, 4))


# --- served rate -----------------------------------------------------------


def test_every_covered_ue_with_room_is_served(cfg) -> None:
    """Three UEs on covered tiles, PRBs to spare: all served."""
    rsrp = _map([[[-95.0, -105.0]], [[-70.0, -85.0]]])
    ue = _ue(
        [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0}] * 2
        + [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 1}]
    )
    assert served_rate(_served(rsrp, ue, cfg)) == pytest.approx(1.0)


def test_a_ue_on_a_hole_counts_as_not_served(cfg) -> None:
    """Tile 1 is heard only at or below -120 dBm, so no layer may serve it."""
    rsrp = _map([[[-80.0, -120.0]], [[-90.0, -140.0]]])
    ue = _ue(
        [
            {"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0},
            {"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 1},
        ]
    )
    assert served_rate(_served(rsrp, ue, cfg)) == pytest.approx(0.5)


def test_a_blocked_ue_counts_as_not_served(cfg) -> None:
    """Each band holds one UE's PRBs: two of three UEs fit."""
    # At 0 dB one PRB carries 180 kbit/s, so each UE needs 0.6 PRB.
    cfg.kpi.capacity.throughput_per_ue_bps = 0.6 * 180_000.0
    cfg.simulation.transmitters.cells[0].max_prb = {"hi": 1, "lo": 1}
    rsrp = _map([[[-80.0]], [[-80.0]]])
    ue = _ue([{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0}] * 3)
    served = _served(rsrp, ue, cfg)
    assert served_rate(served) == pytest.approx(2.0 / 3.0)
    # One UE on each band, so each band's own rate is a third of all reports.
    assert served_rate(served, band=0) == pytest.approx(1.0 / 3.0)
    assert served_rate(served, band=1) == pytest.approx(1.0 / 3.0)


def test_served_rate_rejects_an_empty_ue_table(cfg) -> None:
    """No UE, no denominator."""
    with pytest.raises(ValueError, match="no UE"):
        served_rate(pd.DataFrame(columns=["band"]))
