"""The MDT stage's settings."""

from __future__ import annotations

import pytest

from src.simulation import mdt


def test_spec_rejects_negative_sigma() -> None:
    """The one noise level is validated."""
    with pytest.raises(ValueError, match="rsrp_noise_sigma_db"):
        mdt.MdtSpec(rsrp_noise_sigma_db=-1.0)
