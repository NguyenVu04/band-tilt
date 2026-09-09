"""The pure scene channels: distance field, facade bearing, depression angle.

Nothing here touches Sionna-RT or a GPU. :func:`src.surrogate.features.build`
and :func:`~src.surrogate.features.los_fraction` cast rays and are therefore
covered by the smoke check in the module docstring rather than by a unit test.
"""

from __future__ import annotations

import numpy as np

from src.core.cell import Cell, Tilt
from src.surrogate.features import elevation_deg, facade_azimuth, pool, signed_distance

_STEP_M = 5.0
_TOL_M = 0.5


def _block_dsm() -> np.ndarray:
    """A 9x9 surface model holding one 3x3 building, ``nan`` in a far corner."""
    dsm = np.zeros((9, 9))
    dsm[3:6, 3:6] = 10.0
    dsm[0, 0] = np.nan
    return dsm


def test_signed_distance_is_negative_inside_and_positive_outside() -> None:
    """The field changes sign at the wall, and a ray miss is not a building."""
    sdf = signed_distance(_block_dsm(), _TOL_M, _STEP_M)

    assert sdf[4, 4] < 0
    assert (sdf[3:6, 3:6] < 0).all()
    assert (sdf[0, :3] > 0).all(), "nan is open ground, not a structure"


def test_signed_distance_measures_in_metres() -> None:
    """Distance is scaled by the cell size, not counted in cells.

    The cell two columns clear of the wall at column 3 is two cells out, and
    the transform measures to the nearest occupied cell centre.
    """
    sdf = signed_distance(_block_dsm(), _TOL_M, _STEP_M)

    assert sdf[4, 1] == 2.0 * _STEP_M
    assert sdf[4, 4] == -2.0 * _STEP_M


def test_facade_azimuth_points_away_from_the_wall() -> None:
    """A wall filling the low columns faces east, the +x direction at 0 degrees."""
    dsm = np.zeros((16, 16))
    dsm[:, :8] = 10.0

    azimuth = facade_azimuth(signed_distance(dsm, _TOL_M, _STEP_M), _STEP_M)

    # Away from the array edges, where the smoothing kernel runs off the raster.
    assert np.allclose(azimuth[4:12, 9:12], 0.0, atol=1.0)


def test_facade_azimuth_is_nan_where_the_field_is_flat() -> None:
    """A constant field has no downhill direction, so it has no bearing."""
    azimuth = facade_azimuth(np.zeros((8, 8)), _STEP_M)

    assert np.isnan(azimuth).all()


def test_elevation_deg_is_the_depression_angle_and_is_shared_by_colocated_cells() -> None:
    """Two cells on one mast see one geometry, however they are pointed."""
    tilt = {"b1": Tilt(baseline_deg=8.0, bounds_deg=(0.0, 16.0))}
    cells = (
        Cell(name="n0c0", x=0.0, y=0.0, z=30.0, azimuth_deg=0.0, tilt=tilt),
        Cell(name="n0c1", x=0.0, y=0.0, z=30.0, azimuth_deg=120.0, tilt=tilt),
    )
    centre_x = np.array([[30.0]])
    centre_y = np.array([[0.0]])

    angles = elevation_deg(cells, centre_x, centre_y, ue_height_m=1.5)

    assert angles.shape == (2, 1, 1)
    assert angles[0, 0, 0] == np.degrees(np.arctan2(28.5, 30.0))
    assert angles[0, 0, 0] == angles[1, 0, 0]


def test_pool_reduces_sub_tiles_onto_the_tile_grid() -> None:
    """Each 2x2 block becomes one tile, under whichever reduction is asked for."""
    fine = np.array([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])

    assert pool(fine, 2).tolist() == [[2.5, 0.0]]
    assert pool(fine, 2, np.max).tolist() == [[4.0, 0.0]]
