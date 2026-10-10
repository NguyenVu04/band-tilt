"""Tests for the node layout geometry."""

import itertools
import math

import pytest

from src.scenario.layout import LayoutSpec, max_spacing, node_positions
from src.simulation.scene import SceneBounds

BOUNDS = SceneBounds(min_x=0.0, max_x=1000.0, min_y=-200.0, max_y=800.0, min_z=0.0, max_z=50.0)


def test_four_corners_of_a_square_are_one_spacing_from_the_centre_node():
    """The spacing is centre-to-corner; adjacent corners are sqrt(2) of it apart."""
    *corners, centre = node_positions(BOUNDS, 300.0)

    assert len(corners) == 4
    assert centre == pytest.approx((500.0, 300.0))
    for corner in corners:
        assert math.dist(corner, centre) == pytest.approx(300.0)
    sides = sorted(math.dist(a, b) for a, b in itertools.combinations(corners, 2))
    assert sides[:4] == pytest.approx([300.0 * math.sqrt(2.0)] * 4)
    assert sum(x for x, _ in corners) / 4 == pytest.approx(500.0)
    assert sum(y for _, y in corners) / 4 == pytest.approx(300.0)


def test_largest_fitting_spacing_keeps_every_node_inside_the_scene():
    """At the spacing the fit check allows every node is in bounds, and just past it fails."""
    spacing = max_spacing(BOUNDS)
    for x, y in node_positions(BOUNDS, spacing):
        assert BOUNDS.min_x - 1e-9 <= x <= BOUNDS.max_x + 1e-9
        assert BOUNDS.min_y - 1e-9 <= y <= BOUNDS.max_y + 1e-9
    with pytest.raises(ValueError, match="node_spacing_m"):
        node_positions(BOUNDS, spacing + 1.0)


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
