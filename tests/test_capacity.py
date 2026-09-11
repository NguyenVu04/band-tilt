"""The serving rule and the PRB formulas, on fixtures small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import capacity

# 12 subcarriers of 15 kHz: 180 kHz per PRB.
_B_PRB = 180_000.0

_BANDWIDTH_HZ = 20_000_000.0


def _cells(*max_prb: dict[str, int]) -> list[dict]:
    """One co-located cell per ``max_prb`` entry, in tx-axis order."""
    return [
        {"name": f"c{i}", "x": 0.0, "y": 0.0, "z": 30.0, "azimuth_deg": 0.0, "tilt": {}}
        | {"max_prb": limits}
        for i, limits in enumerate(max_prb)
    ]


@pytest.fixture
def cfg():
    """Two bands, 'hi' preferred, one cell of 10 PRBs per band.

    Zero kelvin makes the noise exactly zero, so SINR is signal over co-band
    interference.
    """
    return OmegaConf.create(
        {
            "kpi": {
                "band_priority": {"hi": 3.0, "lo": 1.0},
                "capacity": {
                    "rsrp_threshold_dbm": -100.0,
                    "throughput_per_ue_bps": _B_PRB,
                    "bands": {"hi": {"scs_hz": 15000}, "lo": {"scs_hz": 15000}},
                },
            },
            "simulation": {
                "radio_map": {
                    "temperature": 0.0,
                    "bands": [
                        {"name": "hi", "bandwidth": _BANDWIDTH_HZ},
                        {"name": "lo", "bandwidth": _BANDWIDTH_HZ},
                    ],
                },
                "transmitters": {"cells": _cells({"hi": 10, "lo": 10})},
            },
        }
    )


def test_thermal_noise_is_kt_times_bandwidth() -> None:
    """Noise at 290 K over 1 Hz is -173.975 dBm, and scales as 10 log10(B)."""
    assert capacity.thermal_noise_dbm(290.0, 1.0) == pytest.approx(-173.975, abs=1e-3)
    assert capacity.thermal_noise_dbm(290.0, 20e6) == pytest.approx(-173.975 + 73.010, abs=1e-3)


def test_the_spec_rejects_a_cell_table_that_does_not_match_the_map(cfg) -> None:
    """The cells are the map's tx axis, so their count must agree."""
    with pytest.raises(ValueError, match="1 cells for a radio map with 2"):
        capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 2)


def test_sinr_counts_only_co_band_interference() -> None:
    """Two transmitters 10 dB apart on one band; the other band is not interference."""
    rsrp = np.array([[-80.0, -90.0], [-50.0, np.nan]])[:, :, None, None]
    sinr = capacity.sinr_db(rsrp, -np.inf)[:, :, 0, 0]
    assert sinr[0].tolist() == pytest.approx([10.0, -10.0])
    assert sinr[1, 0] == np.inf
    assert sinr[1, 1] == -np.inf


def test_rate_and_prbs_follow_the_shannon_formula() -> None:
    """At 0 dB SINR the spectral efficiency is exactly 1 bit/s/Hz."""
    assert capacity.spectral_efficiency(0.0) == pytest.approx(1.0)
    rate = capacity.prb_rate_bps(0.0, capacity.prb_bandwidth_hz(15000.0))
    assert rate == pytest.approx(_B_PRB)
    per_ue = capacity.prb_per_ue(5 * _B_PRB, rate)
    assert per_ue == pytest.approx(5.0)
    assert capacity.prb_required(3, per_ue) == pytest.approx(15.0)
    assert capacity.required_throughput_bps(3, 1e6) == pytest.approx(3e6)
    assert capacity.prb_per_ue(1.0, 0.0) == np.inf


def test_the_preferred_band_serves_when_it_clears_the_threshold() -> None:
    """'hi' tx1 is above -100 dBm, so it wins over a much stronger 'lo'."""
    rsrp = np.array([[-105.0, -99.0], [-60.0, -70.0]])
    order = capacity.candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [1, 2, 3, 0]


def test_a_band_below_the_threshold_passes_to_the_next() -> None:
    """No 'hi' cell clears the threshold, so 'lo' comes first, strongest cell first."""
    rsrp = np.array([[-105.0, -110.0], [-95.0, -80.0]])
    order = capacity.candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [3, 2, 0, 1]


def test_below_every_threshold_the_strongest_serves_and_no_path_is_dropped() -> None:
    """The fallback is plain RSRP order; an unreachable layer is never a candidate."""
    rsrp = np.array([[-105.0, np.nan], [-120.0, -101.0]])
    order = capacity.candidate_order(rsrp, np.array([0, 1]), -100.0)
    assert order.tolist() == [3, 0, 2]


def test_a_full_cell_band_passes_the_ue_to_the_next_candidate(cfg) -> None:
    """Each UE needs 6 PRBs of a 10-PRB limit, so the second UE moves on."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-90.0], [-95.0]]] * 2)  # [ue, band, tx]
    # log2(1 + SINR) = 1/6 at the first choice, 1/6 at the second.
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity.select_serving(rsrp, sinr, spec)
    assert serving.band.tolist() == [0, 1]
    assert serving.prb_per_ue.tolist() == pytest.approx([6.0, 6.0])
    assert serving.load[:, 0].tolist() == pytest.approx([6.0, 6.0])


def test_a_ue_with_no_room_anywhere_is_blocked_but_keeps_its_demand(cfg) -> None:
    """Demand is a requirement: the blocked UE is counted at its first choice."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-90.0], [np.nan]]] * 2)
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity.select_serving(rsrp, sinr, spec)
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
    assert band.tolist() == [0, -1, 0]
    assert tx.tolist() == [1, -1, 1]
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


def test_demand_keeps_the_busiest_interval_per_tile(cfg) -> None:
    """One UE on the tile in interval 0, three in interval 1: the raster holds three."""
    cfg.simulation.transmitters.cells = _cells({"hi": 1000, "lo": 1000})
    rsrp = np.array([[[[-90.0, np.nan]]], [[[np.nan, np.nan]]]])  # [band, tx, row, col]
    mdt = pd.DataFrame({"t_index": [0, 1, 1, 1], "tile_row": [0] * 4, "tile_col": [0] * 4})
    peak = capacity.demand_prb(rsrp, ["hi", "lo"], mdt, cfg)
    # Alone on its band with no noise, SINR is infinite; use a finite noise instead.
    assert peak.shape == (1, 2)
    assert peak[0, 1] == 0.0

    cfg.simulation.radio_map.temperature = 290.0
    noise = capacity.thermal_noise_dbm(290.0, _BANDWIDTH_HZ)
    rate = capacity.prb_rate_bps(-90.0 - noise, _B_PRB)
    peak = capacity.demand_prb(rsrp, ["hi", "lo"], mdt, cfg)
    assert peak[0, 0] == pytest.approx(3 * _B_PRB / rate)
