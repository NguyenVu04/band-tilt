"""The serving rule and the PRB formulas, on fixtures small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import capacity

# 12 subcarriers of 15 kHz: 180 kHz per PRB.
_B_PRB = 180_000.0


@pytest.fixture
def cfg():
    """Two bands, 'hi' preferred, no noise so SINR is signal over co-band interference."""
    return OmegaConf.create(
        {
            "kpi": {
                "band_priority": {"hi": 3.0, "lo": 1.0},
                "capacity": {
                    "rsrp_threshold_dbm": -100.0,
                    "throughput_per_ue_bps": _B_PRB,
                    # -inf noise figure: noise power is exactly zero.
                    "noise_figure_db": float("-inf"),
                    "bands": {
                        "hi": {"scs_hz": 15000, "n_prb": 10},
                        "lo": {"scs_hz": 15000, "n_prb": 10},
                    },
                },
            }
        }
    )


def test_noise_per_re_is_kt_over_one_subcarrier_plus_nf() -> None:
    """-174 dBm/Hz + 10 log10(15 kHz) + 9 dB."""
    assert capacity.noise_per_re_dbm(15000.0, 9.0) == pytest.approx(
        -174.0 + 41.7609 + 9.0, abs=1e-3
    )


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
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"])
    rsrp = np.array([[[-90.0], [-95.0]]] * 2)  # [ue, band, tx]
    # log2(1 + SINR) = 1/6 at the first choice, 1/6 at the second.
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity.select_serving(rsrp, sinr, spec)
    assert serving.band.tolist() == [0, 1]
    assert serving.prb_per_ue.tolist() == pytest.approx([6.0, 6.0])
    assert serving.load[:, 0].tolist() == pytest.approx([6.0, 6.0])


def test_a_ue_with_no_room_anywhere_is_blocked_but_keeps_its_demand(cfg) -> None:
    """Demand is a requirement: the blocked UE is counted at its first choice."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"])
    rsrp = np.array([[[-90.0], [np.nan]]] * 2)
    sinr = np.full(rsrp.shape, 10.0 * np.log10(2.0 ** (1.0 / 6.0) - 1.0))
    serving = capacity.select_serving(rsrp, sinr, spec)
    assert serving.band.tolist() == [0, -1]
    assert serving.prb_per_ue.tolist() == pytest.approx([6.0, 6.0])


def test_demand_keeps_the_busiest_interval_per_tile(cfg) -> None:
    """One UE on the tile in interval 0, three in interval 1: the raster holds three."""
    cfg.kpi.capacity.bands.hi.n_prb = 1000
    rsrp = np.array([[[[-90.0, np.nan]]], [[[np.nan, np.nan]]]])  # [band, tx, row, col]
    mdt = pd.DataFrame({"t_index": [0, 1, 1, 1], "tile_row": [0] * 4, "tile_col": [0] * 4})
    peak = capacity.demand_prb(rsrp, ["hi", "lo"], mdt, cfg)
    # Alone on its band with no noise, SINR is infinite; use a finite noise instead.
    assert peak.shape == (1, 2)
    assert peak[0, 1] == 0.0

    cfg.kpi.capacity.noise_figure_db = 0.0
    noise = capacity.noise_per_re_dbm(15000.0, 0.0)
    rate = capacity.prb_rate_bps(-90.0 - noise, _B_PRB)
    peak = capacity.demand_prb(rsrp, ["hi", "lo"], mdt, cfg)
    assert peak[0, 0] == pytest.approx(3 * _B_PRB / rate)
