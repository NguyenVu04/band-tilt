"""The serving rule and the throughput formulas, on fixtures small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.evaluation import compare
from src.kpi import capacity
from tests.conftest import write_cells

# 12 subcarriers of 15 kHz: 180 kHz per PRB.
_B_PRB = 180_000.0
# The SINR at which log2(1 + SINR) = 0.6: a 'lo' PRB carries 0.6 of a 0 dB one.
_SINR_SE_06 = 10.0 * np.log10(2.0**0.6 - 1.0)


def _cells(*max_prb: dict[str, int]) -> list[dict]:
    """One co-located cell per ``max_prb`` entry, in tx-axis order."""
    return [
        {"name": f"c{i}", "x": 0.0, "y": 0.0, "z": 30.0, "azimuth_deg": 0.0, "tilt": {}}
        | {"max_prb": limits}
        for i, limits in enumerate(max_prb)
    ]


@pytest.fixture
def cfg(tmp_path):
    """Two bands at 15 kHz SCS, one cell of 10 PRBs per band, the whole pool usable."""
    return OmegaConf.create(
        {
            "kpi": {"hole_dbm": -120.0, "capacity": {"max_admission_utilisation": 1.0}},
            "simulation": {
                "radio_map": {"bands": [{"name": n, "scs_hz": 15000} for n in ("hi", "lo")]},
            },
            "data": {"output": {"cells_file": write_cells(tmp_path, _cells({"hi": 10, "lo": 10}))}},
        }
    )


def _two_layers(n_ue: int, rsrp_dbm: float = -90.0) -> tuple[np.ndarray, np.ndarray]:
    """``n_ue`` UEs hearing both bands; a 'hi' PRB carries 180 kbit/s, a 'lo' one 108."""
    rsrp = np.full((n_ue, 2, 1), rsrp_dbm)
    sinr = np.stack([np.zeros((n_ue, 1)), np.full((n_ue, 1), _SINR_SE_06)], axis=1)
    return rsrp, sinr


def test_the_spec_rejects_a_cell_table_that_does_not_match_the_map(cfg) -> None:
    """The cells are the map's tx axis, so their count must agree."""
    with pytest.raises(ValueError, match="1 cells for a radio map with 2"):
        capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 2)


def test_the_spec_reads_scs_from_the_radio_map_bands(cfg) -> None:
    """SCS is the solver's noise bandwidth, so capacity reads the same entry."""
    cfg.simulation.radio_map.bands[1].scs_hz = 30000
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    assert spec.prb_bandwidth_hz.tolist() == [_B_PRB, 2 * _B_PRB]
    with pytest.raises(ValueError, match="No simulation.radio_map.bands entry for mid"):
        capacity.CapacitySpec.from_config(cfg, ["hi", "mid"], 1)


def test_the_usable_share_is_checked_against_config(cfg) -> None:
    """Zero would leave no PRB to share; above one would share PRBs that do not exist."""
    for bad in (0.0, 1.5):
        cfg.kpi.capacity.max_admission_utilisation = bad
        with pytest.raises(ValueError, match="max_admission_utilisation"):
            capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)


def test_the_prb_rate_follows_the_shannon_formula() -> None:
    """At 0 dB SINR the spectral efficiency is exactly 1 bit/s/Hz."""
    assert capacity.spectral_efficiency(0.0) == pytest.approx(1.0)
    assert capacity._prb_rate_bps(0.0, capacity._prb_bandwidth_hz(15000.0)) == pytest.approx(_B_PRB)


def test_each_ue_takes_the_cell_band_with_the_largest_equal_share(cfg) -> None:
    """'hi' is worth 1.8 Mbit/s whole, 'lo' 1.08.

    The first UE takes 'hi'. The second finds half of 'hi' (0.9) below all of
    'lo' and takes 'lo'. The third finds a third of 'hi' (0.6) above half of
    'lo' (0.54) and takes 'hi'. Each UE ends at its cell-band's final share.
    """
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp, sinr = _two_layers(3)
    layer, throughput = capacity._select_serving(rsrp, sinr, np.arange(3.0), spec)
    assert layer.tolist() == [0, 1, 0]
    assert (throughput / 1e6).tolist() == pytest.approx([0.9, 1.08, 0.9])


def test_the_usable_share_scales_the_pool(cfg) -> None:
    """Half of 10 PRBs at 180 kbit/s each: 0.9 Mbit/s for a lone UE."""
    cfg.kpi.capacity.max_admission_utilisation = 0.5
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-90.0], [np.nan]]])
    _, throughput = capacity._select_serving(rsrp, np.zeros(rsrp.shape), np.zeros(1), spec)
    assert throughput[0] / 1e6 == pytest.approx(0.9)


def test_the_earlier_ue_connects_first_and_simultaneous_ones_strongest_first(cfg) -> None:
    """Whoever connects first takes 'hi' and pushes the other onto 'lo'."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp, sinr = _two_layers(2)
    rsrp[1] = -80.0
    # Simultaneous: the -80 dBm UE goes first, whatever the row order.
    layer, _ = capacity._select_serving(rsrp, sinr, np.zeros(2), spec)
    assert layer.tolist() == [1, 0]
    flipped, _ = capacity._select_serving(rsrp[::-1], sinr, np.zeros(2), spec)
    assert flipped.tolist() == [0, 1]
    # Reporting first beats reporting stronger.
    layer, _ = capacity._select_serving(rsrp, sinr, np.array([10.0, 20.0]), spec)
    assert layer.tolist() == [0, 1]


def test_a_ue_with_no_layer_above_the_hole_threshold_is_not_served(cfg) -> None:
    """-120 dBm is a hole and no path is no candidate."""
    spec = capacity.CapacitySpec.from_config(cfg, ["hi", "lo"], 1)
    rsrp = np.array([[[-120.0], [np.nan]], [[np.nan], [-121.0]], [[-119.0], [np.nan]]])
    layer, throughput = capacity._select_serving(rsrp, np.zeros(rsrp.shape), np.zeros(3), spec)
    assert layer.tolist() == [-1, -1, 0]
    assert np.isnan(throughput[:2]).all()
    assert throughput[2] / 1e6 == pytest.approx(1.8)


def test_intervals_do_not_share_prbs(cfg, tmp_path) -> None:
    """Two UEs split 'hi' in interval 0; the lone UE of interval 1 has it whole."""
    cfg.data.output.cells_file = write_cells(tmp_path, _cells({"hi": 10}), "hi_only.csv")
    spec = capacity.CapacitySpec.from_config(cfg, ["hi"], 1)
    rsrp = np.full((3, 1, 1), -90.0)
    band, tx, throughput = capacity.serve_rows(
        rsrp, np.zeros(rsrp.shape), np.array([0, 0, 1]), np.zeros(3), spec
    )
    assert band.tolist() == [0, 0, 0]
    assert tx.tolist() == [0, 0, 0]
    assert (throughput / 1e6).tolist() == pytest.approx([0.9, 0.9, 1.8])


def test_serve_intervals_reports_the_stored_sinr_and_throughput_at_the_serving_layer(
    cfg,
) -> None:
    """SINR is read from the map passed in, not derived from RSRP; 'hi' wins on it."""
    rsrp = np.array([[[[-90.0]]], [[[-80.0]]]])
    sinr = np.array([[[[7.0]]], [[[3.0]]]])
    ue = pd.DataFrame(
        {"t_index": [0, 1], "t_s": [0.0, 0.0], "tile_row": [0, 0], "tile_col": [0, 0]}
    )
    served = capacity.serve_intervals(rsrp, sinr, ["hi", "lo"], ue, cfg)
    assert served["band"].tolist() == [0, 0]
    assert served["sinr_db"].tolist() == [7.0, 7.0]
    expected = 10 * _B_PRB * np.log2(1.0 + 10**0.7) / 1e6
    assert served["estimated_throughput_mbps"].tolist() == pytest.approx([expected] * 2)


def test_demand_counts_every_ue_report_per_tile_served_or_not(cfg) -> None:
    """Four reports on (0, 0), none elsewhere; the processed table's int16 tiles do not overflow."""
    n = 300
    rsrp = np.full((2, 1, n, n), np.nan)
    rsrp[0, 0, 0, 0] = -90.0
    sinr = np.zeros(rsrp.shape)
    ue = pd.DataFrame(
        {
            "t_index": [0, 1, 1, 1, 2],
            "t_s": [0.0] * 5,
            "tile_row": np.array([0, 0, 0, 0, n - 1], dtype=np.int16),
            "tile_col": np.array([0, 0, 0, 0, n - 1], dtype=np.int16),
        }
    )
    archive = {"rsrp_dbm": rsrp, "sinr_db": sinr, "band_label": np.array(["hi", "lo"])}
    demand = compare.configuration(archive, ue, cfg).demand
    assert demand[0, 0] == 4
    assert demand[n - 1, n - 1] == 1
    assert demand.sum() == 5
