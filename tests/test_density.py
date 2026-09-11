"""The hotspot-centre draw, and the built-volume floor it is gated on.

No scene is loaded. The raster is written by hand: a block of four buildings in
one corner, and one building of its own standing out on the far corner. The two
carry comparable volume per tile, so what separates them is the neighbourhood
the floor is measured over, not the size of any one building.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.density import DensitySpec, built_volume, draw_hotspots, neighbourhood_volume
from src.simulation.grid import Raster

BLOCK = (slice(0, 2), slice(0, 2))
LONE = (7, 7)


def _raster() -> Raster:
    free = np.ones((8, 8))
    height = np.zeros((8, 8))
    free[BLOCK] = 0.1
    height[BLOCK] = 40.0
    free[LONE] = 0.5
    height[LONE] = 30.0
    return Raster(
        origin_x=0.0,
        origin_y=0.0,
        tile_size_m=10.0,
        free_fraction=free,
        mean_built_height=height,
    )


def _spec(fraction: float, n_hotspots: int = 2) -> DensitySpec:
    return DensitySpec(
        hotspot_mass_fraction=0.9,
        n_hotspots=n_hotspots,
        sigma_major_m=(20.0, 20.0),
        sigma_minor_m=(10.0, 10.0),
        built_volume_radius_m=15.0,
        min_built_volume_fraction=fraction,
    )


def _centre_tiles(spec: DensitySpec, draws: int = 200) -> set[tuple[int, int]]:
    """Every tile the draw ever puts a centre in, over ``draws`` seeds."""
    raster = _raster()
    seen = set()
    for seed in range(draws):
        for hotspot in draw_hotspots(raster, spec, np.random.default_rng(seed)):
            seen.add((int(hotspot.y // 10.0), int(hotspot.x // 10.0)))
    return seen


def test_the_lone_building_is_reachable_without_a_floor() -> None:
    """An ungated draw does land on the sparse corner.

    This is the behaviour the floor exists to remove; asserting it here keeps
    the next test from passing for the wrong reason.
    """
    assert LONE in _centre_tiles(_spec(0.0))


def test_the_floor_keeps_centres_off_the_sparse_corner() -> None:
    """A quarter of the peak admits the block and rejects the lone building."""
    reachable = _centre_tiles(_spec(0.25))

    assert LONE not in reachable
    assert (0, 0) in reachable


def test_a_zero_floor_only_ever_admits_more() -> None:
    """Zero is the identity, so the floor can be switched off and only widens."""
    assert _centre_tiles(_spec(0.25)) < _centre_tiles(_spec(0.0))


def test_a_floor_that_starves_the_draw_is_an_error() -> None:
    """Too high a floor names itself, rather than silently reverting to uniform."""
    raster = _raster()
    volume = neighbourhood_volume(built_volume(raster), raster, 15.0)
    n_at_peak = int(np.count_nonzero(volume >= 0.99 * volume.max()))

    with pytest.raises(ValueError, match="min_built_volume_fraction"):
        draw_hotspots(raster, _spec(0.99, n_at_peak + 1), np.random.default_rng(0))


def test_a_scene_with_no_buildings_still_falls_back_to_uniform() -> None:
    """The floor is a share of the peak, so it must not bite when there is no peak."""
    bare = Raster(
        origin_x=0.0,
        origin_y=0.0,
        tile_size_m=10.0,
        free_fraction=np.ones((8, 8)),
        mean_built_height=np.zeros((8, 8)),
    )

    assert len(draw_hotspots(bare, _spec(0.25), np.random.default_rng(0))) == 2


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_a_fraction_outside_the_unit_interval_is_rejected(fraction: float) -> None:
    """One is excluded too: it would admit only the tiles tied with the peak."""
    with pytest.raises(ValueError, match="min_built_volume_fraction"):
        _spec(fraction)
