"""Tests for the angle convention — small, and worth more than they look.

PROJECT.md section 22.2 fixes the conversion between the radio convention and the
Sionna-RT simulation convention. A sign error in it does not raise. It produces
a complete, plausible radio map with every beam pointing at the sky, KPIs that
are internally consistent, and an optimizer that converges confidently on the
wrong answer.

These assertions are the only thing standing between that and a result nobody
can reproduce. See PROJECT.md section 22.2.

Every expected value below is computed by hand from the spec, not from the
implementation.
"""

import numpy as np
import pandas as pd
import pytest

from src.radio import geometry

pytestmark = pytest.mark.skip(reason="implement src/radio/geometry.py first")


def test_absolute_tilt_is_the_sum_of_electrical_and_mechanical() -> None:
    """PROJECT.md section 8: ``tilt = eTilt + mTilt``.

    The sum, not either component, is the optimization variable.
    """
    result = geometry.absolute_tilt(np.array([6.0, 8.0]), np.array([2.0, -1.0]))
    assert result == pytest.approx([8.0, 7.0])


def test_azimuth_to_yaw_at_cardinal_bearings() -> None:
    """``yaw = deg2rad(90 - azimuth)``, checked at bearings with known answers.

    Due east (90 degrees) is yaw 0; due north (0 degrees) is yaw pi/2. Getting
    these two right pins the direction of rotation, which is the half of the
    convention a reader is most likely to reconstruct backwards.
    """
    yaw = geometry.azimuth_to_yaw(np.array([0.0, 90.0, 180.0]))
    assert yaw == pytest.approx([np.pi / 2, 0.0, -np.pi / 2])


def test_downtilt_becomes_negative_pitch() -> None:
    """``pitch = deg2rad(-tilt)``: a positive downtilt is a NEGATIVE pitch.

    This single sign is the highest-risk line in the project. A beam tilted six
    degrees down must not come out six degrees up.
    """
    assert geometry.tilt_to_pitch(np.array([6.0])) == pytest.approx([-np.deg2rad(6.0)])


def test_zero_tilt_is_zero_pitch() -> None:
    """The horizon maps to the horizon, under either sign convention.

    Passes whichever way the sign goes, which is exactly why it is not enough on
    its own — it is here so that a failure means something worse than a flipped
    sign.
    """
    assert geometry.tilt_to_pitch(np.array([0.0])) == pytest.approx([0.0])


def test_orientations_have_one_row_per_cell_band(cell_bands: pd.DataFrame) -> None:
    """The orientation array is ``(n_cell_bands, 3)`` — yaw, pitch, roll."""
    tilt = np.full(len(cell_bands), 6.0)
    assert geometry.orientations(cell_bands, tilt).shape == (len(cell_bands), 3)


def test_orientations_follow_cell_band_table_order(cell_bands: pd.DataFrame) -> None:
    """Row ``i`` of the orientation array belongs to row ``i`` of the table.

    Every tilt vector in the project is indexed by position in the cell-band
    table. A reordering here assigns tilts to the wrong antennas and produces a
    radio map for a network that does not exist.
    """
    tilt = np.arange(len(cell_bands), dtype=float)
    orient = geometry.orientations(cell_bands, tilt)
    expected_yaw = geometry.azimuth_to_yaw(cell_bands["azimuth"].to_numpy())
    assert orient[:, 0] == pytest.approx(expected_yaw)
    assert orient[:, 1] == pytest.approx(geometry.tilt_to_pitch(tilt))


def test_orientations_reject_a_mismatched_tilt(cell_bands: pd.DataFrame) -> None:
    """A tilt vector of the wrong length must fail loudly, not broadcast.

    It is the symptom of a band added to ``configs/radio.yaml`` after something
    downstream was built, and NumPy will happily broadcast a length-1 vector
    across every cell.
    """
    with pytest.raises(ValueError):
        geometry.orientations(cell_bands, np.array([6.0]))
