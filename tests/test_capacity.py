"""The serving rule and the PRB formulas, on fixtures small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import capacity

# 12 subcarriers of 15 kHz: 180 kHz per PRB.
_B_PRB = 180_000.0


def _cells(*max_prb: dict[str, int]) -> list[dict]:
    """One co-located cell per ``max_prb`` entry, in tx-axis order."""
    return [
        {"name": f"c{i}", "x": 0.0, "y": 0.0, "z": 30.0, "azimuth_deg": 0.0, "tilt": {}}
        | {"max_prb": limits}
        for i, limits in enumerate(max_prb)
    ]


@pytest.fixture
def cfg():
    """Two bands, 'hi' preferred, one cell of 10 PRBs per band."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "capacity": {
                    "band_preference": ["hi", "lo"],
                    "rsrp_threshold_dbm": -100.0,
                    "throughput_per_ue_bps": _B_PRB,
                    "bands": {"hi": {"scs_hz": 15000}, "lo": {"scs_hz": 15000}},
                },
            },
            "simulation": {"seed": 0, "transmitters": {"cells": _cells({"hi": 10, "lo": 10})}},
        }
    )


def test_the_spec_rejects_a_cell_table_that_does_not_match_the_map(cfg) -> None:
    """The cells are the map's tx axis, so their count must agree."""
    with pytest.raises(ValueError, match="1 cells for a radio map with 2"):
        capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 2)


def test_rate_and_prbs_follow_the_shannon_formula() -> None:
    """At 0 dB SINR the spectral efficiency is exactly 1 bit/s/Hz."""
    assert capacity._spectral_efficiency(0.0) == pytest.approx(1.0)
    rate = capacity._prb_rate_bps(0.0, capacity._prb_bandwidth_hz(15000.0))
    assert rate == pytest.approx(_B_PRB)
    assert capacity._prb_per_ue(5 * _B_PRB, rate) == pytest.approx(5.0)
    assert capacity._prb_per_ue(1.0, 0.0) == np.inf


def test_the_preferred_band_serves_when_it_clears_the_threshold() -> None:
    """'hi' tx1 is above -100 dBm, so it wins over a much stronger 'lo'."""
    rsrp = np.array([[-105.0, -99.0], [-60.0, -70.0]])
    order = capacity._candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [1, 2, 3, 0]


def test_a_band_below_the_threshold_passes_to_the_next() -> None:
    """No 'hi' cell clears the threshold, so 'lo' comes first, strongest cell first."""
    rsrp = np.array([[-105.0, -110.0], [-95.0, -80.0]])
    order = capacity._candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [3, 2, 0, 1]


def test_below_every_threshold_the_strongest_serves_and_no_path_is_dropped() -> None:
    """The fallback is plain RSRP order; an unreachable layer is never a candidate."""
    rsrp = np.array([[-105.0, np.nan], [-120.0, -101.0]])
    order = capacity._candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [3, 0, 2]


def test_a_layer_at_or_below_the_hole_threshold_is_never_a_candidate() -> None:
    """-120 dBm is a hole, so only the -119 dBm layer may serve."""
    rsrp = np.array([[-120.0, -119.0], [-130.0, np.nan]])
    order = capacity._candidate_order(rsrp, np.array([0, 1]), -100.0, -120.0)
    assert order.tolist() == [1]


def test_admission_order_does_not_follow_row_order(cfg) -> None:
    """One PRB-limited cell, many identical UEs: the admitted ones are not the first rows."""
    cfg.simulation.transmitters.cells = _cells({"hi": 60, "lo": 1})
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    n_ue = 40
    rsrp = np.array([[[-90.0], [np.nan]]] * n_ue)
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    band, _tx, _per_ue = capacity.serve_rows(rsrp, sinr, np.zeros(n_ue, dtype=int), spec)
    admitted = np.flatnonzero(band >= 0)
    assert admitted.size == 10
    assert admitted.tolist() != list(range(10))
    again, _, _ = capacity.serve_rows(rsrp, sinr, np.zeros(n_ue, dtype=int), spec)
    assert np.array_equal(band, again)


def test_a_full_cell_band_passes_the_ue_to_the_next_candidate(cfg) -> None:
    """Each UE needs 6 PRBs of a 10-PRB limit, so the second UE moves on."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-90.0], [-95.0]]] * 2)  # [ue, band, tx]
    # log2(1 + SINR) = 1/6 at the first choice, 1/6 at the second.
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity._select_serving(rsrp, sinr, spec)
    assert serving.band.tolist() == [0, 1]
    assert serving.prb_per_ue.tolist() == pytest.approx([6.0, 6.0])
    assert serving.load[:, 0].tolist() == pytest.approx([6.0, 6.0])


def test_a_ue_with_no_room_anywhere_is_blocked_but_keeps_its_demand(cfg) -> None:
    """Demand is a requirement: the blocked UE is counted at its first choice."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-90.0], [np.nan]]] * 2)
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity._select_serving(rsrp, sinr, spec)
    assert serving.band.tolist() == [0, -1]
    assert serving.prb_per_ue.tolist() == pytest.approx([6.0, 6.0])


def test_each_cell_band_has_its_own_limit_and_intervals_do_not_share_prbs(cfg) -> None:
    """tx0 holds 5 PRBs, tx1 10; each UE needs 6.

    The first UE skips the stronger tx0 for tx1, the second finds tx1 full and
    is blocked, and a UE in the next interval finds tx1 empty again.
    """
    cfg.simulation.transmitters.cells = _cells({"hi": 5}, {"hi": 10})
    spec = capacity.CapacitySpec.from_config(cfg, ["hi"], 2)
    rsrp = np.array([[[-90.0, -95.0]]] * 3)  # [ue, band, tx]
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    band, tx, per_ue = capacity.serve_rows(rsrp, sinr, np.array([0, 0, 1]), spec)
    # The two interval-0 UEs are identical, so which one is blocked is the shuffle's call.
    assert sorted(band[:2].tolist()) == [-1, 0]
    assert sorted(tx[:2].tolist()) == [-1, 1]
    assert (band[2], tx[2]) == (0, 1)
    assert per_ue.tolist() == pytest.approx([6.0, 6.0, 6.0])


def test_prb_by_interval_sums_each_tile_within_each_interval() -> None:
    """Two UEs share a tile in interval 3; a UE with no path adds nothing."""
    t_values, prb = capacity.prb_by_interval(
        np.array([3, 3, 5]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([1.0, 2.0, np.nan]),
        (2, 2),
    )
    assert t_values.tolist() == [3, 5]
    assert prb.shape == (2, 2, 2)
    assert prb[0, 0, 1] == pytest.approx(3.0)
    assert prb.sum() == pytest.approx(3.0)


def test_prb_by_interval_accepts_the_processed_int16_tiles() -> None:
    """The processed MDT stores tiles as int16, and row * n_cols overflows it on a large grid."""
    shape = (300, 300)
    _, prb = capacity.prb_by_interval(
        np.array([0]),
        np.array([299], dtype=np.int16),
        np.array([299], dtype=np.int16),
        np.array([2.0]),
        shape,
    )
    assert prb[0, 299, 299] == pytest.approx(2.0)


def test_demand_keeps_the_busiest_interval_per_tile(cfg) -> None:
    """One UE on the tile in interval 0, three in interval 1: the raster holds three."""
    cfg.simulation.transmitters.cells = _cells({"hi": 1000, "lo": 1000})
    rsrp = np.array([[[[-90.0, np.nan]]], [[[np.nan, np.nan]]]])  # [band, tx, row, col]
    # 0 dB wherever a path exists: 1 bit/s/Hz, so each UE needs exactly one PRB.
    sinr = np.where(np.isfinite(rsrp), 0.0, np.nan)
    mdt = pd.DataFrame({"t_index": [0, 1, 1, 1], "tile_row": [0] * 4, "tile_col": [0] * 4})
    peak = capacity.demand_prb(rsrp, sinr, ["hi", "lo"], mdt, cfg)
    assert peak.shape == (1, 2)
    assert peak[0, 0] == pytest.approx(3.0)
    assert peak[0, 1] == 0.0


def test_serve_intervals_reports_the_stored_sinr_at_the_serving_layer(cfg) -> None:
    """SINR is read from the map passed in, not derived from RSRP."""
    rsrp = np.array([[[[-90.0]]], [[[-80.0]]]])
    sinr = np.array([[[[7.0]]], [[[3.0]]]])
    mdt = pd.DataFrame({"t_index": [0], "tile_row": [0], "tile_col": [0]})
    served = capacity.serve_intervals(rsrp, sinr, ["hi", "lo"], mdt, cfg)
    assert served["band"].tolist() == [0]
    assert served["sinr_db"].tolist() == [7.0]
