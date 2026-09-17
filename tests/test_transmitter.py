"""Tests for the node layout geometry."""

import itertools
import math

import pytest

from src.simulation.scene import SceneBounds
from src.simulation.transmitter import node_positions

BOUNDS = SceneBounds(min_x=0.0, max_x=1000.0, min_y=-200.0, max_y=800.0, min_z=0.0, max_z=50.0)


def test_nodes_form_an_equilateral_triangle_around_a_centre_node():
    """Three corners, one spacing apart, then a fourth node on the scene centre."""
    *corners, centre = node_positions(BOUNDS, 600.0)

    assert len(corners) == 3
    assert centre == pytest.approx((500.0, 300.0))
    for (ax, ay), (bx, by) in itertools.combinations(corners, 2):
        assert math.dist((ax, ay), (bx, by)) == pytest.approx(600.0)
    assert sum(x for x, _ in corners) / 3 == pytest.approx(500.0)
    assert sum(y for _, y in corners) / 3 == pytest.approx(300.0)
    for corner in corners:
        assert math.dist(corner, (500.0, 300.0)) == pytest.approx(600.0 / math.sqrt(3.0))


def test_largest_fitting_triangle_stays_inside_the_scene():
    """The spacing the fit check allows keeps every corner within the bounds."""
    spacing = BOUNDS.depth_m * math.sqrt(3.0) / 2.0
    for x, y in node_positions(BOUNDS, spacing):
        assert BOUNDS.min_x - 1e-9 <= x <= BOUNDS.max_x + 1e-9
        assert BOUNDS.min_y - 1e-9 <= y <= BOUNDS.max_y + 1e-9


def test_triangle_wider_than_the_scene_is_rejected():
    """A spacing past the scene width fails and names the config key."""
    with pytest.raises(ValueError, match="node_spacing_m"):
        node_positions(BOUNDS, 1001.0)
