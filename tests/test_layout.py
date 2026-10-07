"""Tests for the node layout geometry."""

import itertools
import math

import pytest

from src.scenario.layout import LayoutSpec, node_positions
from src.simulation.scene import SceneBounds

BOUNDS = SceneBounds(min_x=0.0, max_x=1000.0, min_y=-200.0, max_y=800.0, min_z=0.0, max_z=50.0)


def test_every_corner_is_one_spacing_from_the_centre_node():
    """The spacing is the inter-site distance; the corners are sqrt(3) of it apart."""
    *corners, centre = node_positions(BOUNDS, 400.0)

    assert len(corners) == 3
    assert centre == pytest.approx((500.0, 300.0))
    for corner in corners:
        assert math.dist(corner, centre) == pytest.approx(400.0)
    for (ax, ay), (bx, by) in itertools.combinations(corners, 2):
        assert math.dist((ax, ay), (bx, by)) == pytest.approx(400.0 * math.sqrt(3.0))
    assert sum(x for x, _ in corners) / 3 == pytest.approx(500.0)
    assert sum(y for _, y in corners) / 3 == pytest.approx(300.0)


def test_largest_fitting_triangle_stays_inside_the_scene():
    """The spacing the fit check allows keeps every corner within the bounds."""
    spacing = min(BOUNDS.width_m / math.sqrt(3.0), BOUNDS.depth_m / 2.0)
    for x, y in node_positions(BOUNDS, spacing):
        assert BOUNDS.min_x - 1e-9 <= x <= BOUNDS.max_x + 1e-9
        assert BOUNDS.min_y - 1e-9 <= y <= BOUNDS.max_y + 1e-9


def test_triangle_wider_than_the_scene_is_rejected():
    """A spacing whose top corner leaves the scene fails and names the config key."""
    with pytest.raises(ValueError, match="node_spacing_m"):
        node_positions(BOUNDS, 501.0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sectors_per_node", 0, "sectors_per_node"),
        ("node_spacing_m", 0.0, "node_spacing_m"),
        ("snap_radius_m", -1.0, "snap_radius_m"),
        ("clearance_radius_m", -1.0, "clearance_radius_m"),
    ],
)
def test_a_layout_no_node_could_be_placed_under_is_rejected(field, value, match):
    """A zero-sector or negative-radius layout fails at the config, not as an empty table."""
    good = dict(
        node_spacing_m=400.0,
        sectors_per_node=3,
        azimuth_offset_deg=0.0,
        mast_height_m=25.0,
        min_free_fraction=0.9,
        clearance_radius_m=20.0,
        snap_radius_m=100.0,
    )
    LayoutSpec(**good)
    with pytest.raises(ValueError, match=match):
        LayoutSpec(**{**good, field: value})
