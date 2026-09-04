"""The region of interest: the inset extent and the cells inside it.

Both are pure geometry, so the expected values are counted by hand off a grid
small enough to hold in the head.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.grid import Raster, roi_mask
from src.simulation.scene import SceneBounds

SCENE = SceneBounds(min_x=0.0, max_x=100.0, min_y=0.0, max_y=80.0, min_z=0.0, max_z=30.0)


def test_inset_pulls_in_all_four_horizontal_sides() -> None:
    """A 10 m margin on a 100x80 m scene leaves 80x60 m from (10, 10) to (90, 70)."""
    roi = SCENE.inset(10.0)

    assert (roi.min_x, roi.max_x, roi.min_y, roi.max_y) == (10.0, 90.0, 10.0, 70.0)
    assert roi.width_m == 80.0
    assert roi.depth_m == 60.0


def test_inset_leaves_the_z_bounds_alone() -> None:
    """The margin is horizontal, so the launch height is unchanged.

    A ray cast inside the region still has to clear the tallest geometry in
    the whole scene, including the part standing in the margin.
    """
    roi = SCENE.inset(10.0)

    # launch_z must still clear the tallest geometry in the whole scene, not
    # only the part inside the region.
    assert (roi.min_z, roi.max_z) == (0.0, 30.0)
    assert roi.launch_z == SCENE.launch_z


def test_a_zero_margin_is_the_scene_itself() -> None:
    """No margin is the identity, so the region can be switched off."""
    assert SCENE.inset(0.0) == SCENE


@pytest.mark.parametrize("margin_m", [-1.0, 40.0, 60.0])
def test_a_margin_leaving_no_interior_is_rejected(margin_m: float) -> None:
    """40 m off both sides of an 80 m depth leaves exactly nothing."""
    with pytest.raises(ValueError):
        SCENE.inset(margin_m)


def _raster() -> Raster:
    """A 5x5 grid of 10 m cells at the origin, so centres sit at 5, 15, 25, 35, 45."""
    return Raster(
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=10.0,
        free_fraction=np.ones((5, 5)),
        mean_built_height=np.zeros((5, 5)),
    )


def test_roi_mask_keeps_cells_whose_centre_is_inside() -> None:
    """Centres at 15, 25 and 35 fall inside 10..40, giving a 3x3 block of the 5x5."""
    roi = SceneBounds(min_x=10.0, max_x=40.0, min_y=10.0, max_y=40.0, min_z=0.0, max_z=1.0)

    mask = roi_mask(_raster(), roi)

    assert mask.shape == (5, 5)
    assert int(mask.sum()) == 9
    np.testing.assert_array_equal(np.argwhere(mask.any(axis=0)).ravel(), [1, 2, 3])
    np.testing.assert_array_equal(np.argwhere(mask.any(axis=1)).ravel(), [1, 2, 3])


def test_roi_mask_covering_the_grid_keeps_everything() -> None:
    """A region wider than the grid masks nothing out."""
    roi = SceneBounds(min_x=0.0, max_x=50.0, min_y=0.0, max_y=50.0, min_z=0.0, max_z=1.0)

    assert roi_mask(_raster(), roi).all()
