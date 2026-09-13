"""The MDT stage's SINR, computed from the reported RSRP."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation import mdt


def test_thermal_noise_is_kt_times_bandwidth() -> None:
    """Noise at 290 K over 1 Hz is -173.975 dBm, and scales as 10 log10(B)."""
    assert mdt._thermal_noise_dbm(290.0, 1.0) == pytest.approx(-173.975, abs=1e-3)
    assert mdt._thermal_noise_dbm(290.0, 20e6) == pytest.approx(-173.975 + 73.010, abs=1e-3)


def test_sinr_counts_only_co_band_interference() -> None:
    """Two transmitters 10 dB apart on one band; the other band is not interference."""
    rsrp = np.array([[-80.0, -90.0], [-50.0, np.nan]])[:, :, None, None]
    sinr = mdt._sinr_db(rsrp, -np.inf)[:, :, 0, 0]
    assert sinr[0].tolist() == pytest.approx([10.0, -10.0])
    assert sinr[1, 0] == np.inf
    assert np.isnan(sinr[1, 1])


def test_sinr_per_ue_matches_the_map() -> None:
    """Sampling then computing SINR equals computing then sampling.

    ``src.simulation.mdt.build`` flattens the sampled map to ``[ue, cell-band]``
    to write the CSV and transposes it for SINR and the serving rule. A band/tx
    swap anywhere in that round trip still yields plausible numbers, so it is
    pinned against the map layout directly.
    """
    n_band, n_tx, n_rows, n_cols = 2, 3, 4, 5
    rng = np.random.default_rng(0)
    rsrp = rng.uniform(-110.0, -60.0, size=(n_band, n_tx, n_rows, n_cols))
    # A no-path layer must stay NaN through the transposes.
    rsrp[1, 2, 0, 0] = np.nan
    noise_dbm = np.array([-95.0, -100.0])
    row = np.array([0, 2, 3, 0])
    col = np.array([0, 4, 1, 0])

    # As mdt.build does it: sample -> [ue, tx, band] -> flatten -> [ue, tx, band].
    flat = rsrp[:, :, row, col].transpose(2, 1, 0).reshape(len(row), -1)
    rsrp_ue = flat.reshape(-1, n_tx, n_band)
    actual = mdt._sinr_db(rsrp_ue.transpose(2, 1, 0), noise_dbm).transpose(2, 1, 0)

    expected = mdt._sinr_db(rsrp, noise_dbm)[:, :, row, col].transpose(2, 1, 0)
    assert actual.shape == (len(row), n_tx, n_band)
    np.testing.assert_allclose(actual, expected)
    assert np.isnan(actual[0, 2, 1])


def test_spec_rejects_negative_sigma() -> None:
    """The one noise level is validated."""
    with pytest.raises(ValueError, match="rsrp_noise_sigma_db"):
        mdt.MdtSpec(rsrp_noise_sigma_db=-1.0)
