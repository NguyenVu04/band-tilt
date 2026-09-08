"""The five KPI definitions, and the reductions they share."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import band_priority_score, expected_rsrp_improvement, hole_rate, weak_rate
from src.kpi.bps import ue_counts
from src.kpi.improvement import measured_serving_rsrp
from src.kpi.serving import overlap_neighbors
from src.kpi.tiles import tile_index

_TAU_DB = 3.0


@pytest.fixture
def cfg():
    """The thresholds the KPIs read, without composing the whole config."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "rsrp_improvement_tau_db": _TAU_DB,
                "band_priority": {"hi": 3.0, "lo": 1.0},
            }
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


def test_ue_counts_bins_row_major() -> None:
    """The weight raster the band priority score multiplies through."""
    mdt = _mdt([{"tile_row": 1, "tile_col": 0}] * 3 + [{"tile_row": 0, "tile_col": 1}])
    assert ue_counts(mdt, (2, 2)).tolist() == [[0, 1], [3, 0]]


# --- expected RSRP improvement ---------------------------------------------


def test_measured_serving_rsrp_is_the_row_max_over_reported_cells() -> None:
    """An unreported cell is NaN and cannot be the one serving the UE."""
    mdt = _mdt(
        [
            {"rsrp_a_hi": -95.0, "rsrp_a_lo": -88.0},
            {"rsrp_a_hi": np.nan, "rsrp_a_lo": -101.0},
        ]
    )
    assert measured_serving_rsrp(mdt).tolist() == [-88.0, -101.0]


def test_a_frame_with_no_measurements_is_named_rather_than_reduced() -> None:
    """Silently reducing the position columns would give a plausible wrong number."""
    with pytest.raises(ValueError, match="columns in the MDT"):
        measured_serving_rsrp(_mdt([{"tile_row": 0, "tile_col": 0}]))


def test_a_row_reporting_nothing_raises() -> None:
    """src.data.schema rejects this, so reaching it means the gate was skipped."""
    with pytest.raises(ValueError, match="report no measurement"):
        measured_serving_rsrp(_mdt([{"rsrp_a_hi": np.nan, "rsrp_a_lo": np.nan}]))


def test_no_change_scores_exactly_one_half(cfg) -> None:
    """The sigmoid's midpoint, and the reading every delta is relative to."""
    rsrp = _map([[[-95.0, -100.0]]])
    mdt = _mdt(
        [
            {"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0},
            {"tile_row": 0, "tile_col": 1, "rsrp_a_hi": -100.0},
        ]
    )
    assert expected_rsrp_improvement(rsrp, mdt, cfg) == pytest.approx(0.5)


def test_a_gain_of_tau_scores_the_sigmoid_of_one(cfg) -> None:
    """Pins tau to the units of the RSRP difference rather than to the score."""
    rsrp = _map([[[-95.0 + _TAU_DB]]])
    mdt = _mdt([{"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0}])
    assert expected_rsrp_improvement(rsrp, mdt, cfg) == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_a_loss_scores_below_one_half(cfg) -> None:
    """The KPI is maximised, so a weaker candidate must not read as a gain."""
    rsrp = _map([[[-105.0]]])
    mdt = _mdt([{"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0}])
    assert expected_rsrp_improvement(rsrp, mdt, cfg) < 0.5


def test_a_tile_turned_into_a_hole_scores_zero_without_warning(cfg) -> None:
    """The -inf path. np.exp would overflow and warn here; np.tanh does not."""
    rsrp = _map([[[np.nan]]])
    mdt = _mdt([{"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert expected_rsrp_improvement(rsrp, mdt, cfg) == pytest.approx(0.0)


def test_a_non_positive_tau_raises(cfg) -> None:
    """At zero the sigmoid argument divides by zero; below it the score inverts."""
    cfg.kpi.rsrp_improvement_tau_db = 0.0
    mdt = _mdt([{"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0}])
    with pytest.raises(ValueError, match="must be positive"):
        expected_rsrp_improvement(_map([[[-95.0]]]), mdt, cfg)


def test_the_score_is_weighted_by_report_count(cfg) -> None:
    """One row per UE per interval, so a busy tile pulls the mean towards itself."""
    rsrp = _map([[[-95.0 + _TAU_DB, -95.0]]])
    gain = {"tile_row": 0, "tile_col": 0, "rsrp_a_hi": -95.0}
    flat = {"tile_row": 0, "tile_col": 1, "rsrp_a_hi": -95.0}
    balanced = expected_rsrp_improvement(rsrp, _mdt([gain, flat]), cfg)
    weighted = expected_rsrp_improvement(rsrp, _mdt([gain, gain, gain, flat]), cfg)
    assert weighted > balanced


# --- band priority score ---------------------------------------------------


def test_band_priority_score_weights_tiles_by_ue_count(cfg) -> None:
    """Band 'hi' dominates tile 0 and 'lo' tile 1; tile 0 carries three of four UEs."""
    rsrp = _map([[[-80.0, -100.0]], [[-90.0, -85.0]]])
    mdt = _mdt([{"tile_row": 0, "tile_col": 0}] * 3 + [{"tile_row": 0, "tile_col": 1}])
    assert band_priority_score(rsrp, ["hi", "lo"], mdt, cfg) == pytest.approx(0.75)


def test_a_ue_standing_on_a_hole_is_excluded_from_both_sums(cfg) -> None:
    """No band serves it, so it can neither raise nor lower the score."""
    rsrp = _map([[[-80.0, -130.0]], [[-90.0, -140.0]]])
    served_only = _mdt([{"tile_row": 0, "tile_col": 0}])
    with_hole = _mdt([{"tile_row": 0, "tile_col": 0}, {"tile_row": 0, "tile_col": 1}])
    assert band_priority_score(rsrp, ["hi", "lo"], with_hole, cfg) == pytest.approx(
        band_priority_score(rsrp, ["hi", "lo"], served_only, cfg)
    )
