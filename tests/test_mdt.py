"""The MDT stage's SINR recomputation, which no longer reads a stored SINR map."""

from __future__ import annotations

import numpy as np

from src.kpi import capacity
from src.simulation import mdt


def test_sinr_recomputed_per_ue_matches_the_map() -> None:
    """Sampling then recomputing SINR equals recomputing then sampling.

    ``src.simulation.mdt.build`` flattens the sampled map to ``[ue, cell-band]``
    to write the CSV, then unflattens it to ``[ue, band, tx]`` for the serving
    rule. A band/tx swap anywhere in that round trip still yields finite,
    plausible PRBs, so it has to be pinned against the map layout directly.
    """
    n_band, n_tx, n_rows, n_cols = 2, 3, 4, 5
    rng = np.random.default_rng(0)
    rsrp = rng.uniform(-110.0, -60.0, size=(n_band, n_tx, n_rows, n_cols))
    # A no-path layer: NaN must survive the transposes as -inf SINR, not 0 dBm.
    rsrp[1, 2, 0, 0] = np.nan
    noise_dbm = np.array([-95.0, -100.0])
    row = np.array([0, 2, 3, 0])
    col = np.array([0, 4, 1, 0])

    # As mdt.build does it: sample -> [ue, tx, band] -> flatten -> [ue, band, tx].
    flat = rsrp[:, :, row, col].transpose(2, 1, 0).reshape(len(row), -1)
    rsrp_ue = flat.reshape(-1, n_tx, n_band).transpose(0, 2, 1)
    actual = capacity.sinr_db(rsrp_ue.transpose(1, 2, 0), noise_dbm).transpose(2, 0, 1)

    expected = capacity.sinr_db(rsrp, noise_dbm)[:, :, row, col].transpose(2, 0, 1)
    assert actual.shape == (len(row), n_band, n_tx)
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 1, 2] == -np.inf


def test_spec_rejects_negative_sigma() -> None:
    """The one remaining noise level is still validated."""
    try:
        mdt.MdtSpec(rsrp_noise_sigma_db=-1.0)
    except ValueError as error:
        assert "rsrp_noise_sigma_db" in str(error)
    else:
        raise AssertionError("a negative sigma was accepted")
