"""Drawing the population across intervals.

No scene is loaded. ``surface_height`` is the one call into Mitsuba, and it is
replaced with a flat ground so the rejection loop accepts on the first round and
the interval bookkeeping is what is under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation import sample
from src.simulation.density import DensityField
from src.simulation.grid import GridSpec, Raster
from src.simulation.sample import UeSpec, sample_positions, write_csv
from src.simulation.scene import SceneBounds
from src.simulation.traffic import Schedule

BOUNDS = SceneBounds(min_x=0.0, max_x=40.0, min_y=0.0, max_y=40.0, min_z=0.0, max_z=10.0)
GRID = GridSpec(cell_size_m=10.0, subsamples_per_cell=2, free_height_tol_m=0.5)
COUNTS = (3, 5, 2)


@pytest.fixture(autouse=True)
def flat_ground(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every position lands on open ground at height zero."""
    monkeypatch.setattr(
        sample.scene, "surface_height", lambda mi_scene, x, y, launch_z: np.zeros_like(x)
    )


def _raster() -> Raster:
    return Raster(
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=10.0,
        free_fraction=np.ones((4, 4)),
        mean_built_height=np.zeros((4, 4)),
    )


def _field() -> DensityField:
    """Background only, spread evenly over the sixteen cells."""
    return DensityField(cell_weights=np.full((1, 16), 1.0 / 16.0), hotspots=())


def _schedule() -> Schedule:
    return Schedule(
        interval_s=900.0,
        t_s=np.array([0.0, 900.0, 1800.0]),
        count=np.array(COUNTS, dtype=np.int64),
        component_mass=np.ones((3, 1)),
        phase_rad=np.zeros(0),
    )


def _draw(roi: SceneBounds = BOUNDS) -> tuple[np.ndarray, ...]:
    return sample_positions(
        mi_scene=None,
        bounds=BOUNDS,
        roi=roi,
        raster=_raster(),
        field=_field(),
        schedule=_schedule(),
        grid_spec=GRID,
        seed=17,
    )


def test_every_interval_gets_the_count_it_asked_for() -> None:
    """Counts of 3, 5 and 2 give ten rows labelled 0, 0, 0, 1, 1, 1, 1, 1, 2, 2."""
    interval, x, y, component = _draw()

    assert x.size == y.size == component.size == sum(COUNTS)
    np.testing.assert_array_equal(interval, np.repeat([0, 1, 2], COUNTS))


def test_positions_stay_inside_the_region_of_interest() -> None:
    """The overhang past the region is rejected, not merely the cells outside it.

    A cell whose centre is inside can still reach past the boundary, so the
    region has to be an exact edge rather than a half-cell approximation.
    """
    roi = SceneBounds(min_x=10.0, max_x=30.0, min_y=10.0, max_y=30.0, min_z=0.0, max_z=10.0)

    _, x, y, _ = _draw(roi)

    assert (x >= roi.min_x).all() and (x <= roi.max_x).all()
    assert (y >= roi.min_y).all() and (y <= roi.max_y).all()


def test_background_draws_are_reported_as_component_minus_one() -> None:
    """Component 0 is the background and is written out as -1."""
    _, _, _, component = _draw()

    np.testing.assert_array_equal(component, -1)


def test_same_seed_gives_the_same_population() -> None:
    """The draw is a pure function of its seed, so a rerun reproduces it."""
    _, first_x, _, _ = _draw()
    _, second_x, _, _ = _draw()

    np.testing.assert_array_equal(first_x, second_x)


def test_write_csv_has_no_ue_id_and_one_row_per_ue(tmp_path) -> None:
    """Ten UEs give ten rows under a header carrying the interval, not an id."""
    interval, x, y, component = _draw()

    path = write_csv(
        tmp_path / "ue.csv",
        interval,
        _schedule(),
        x,
        y,
        component,
        _raster(),
        UeSpec(count_range=(3, 5), height_m=1.5),
    )
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "t_index,t_s,x,y,z,cell_col,cell_row,component"
    assert "ue_id" not in lines[0]
    assert len(lines) == 1 + sum(COUNTS)
    # The third interval starts at 1800 s, and its two rows are written last.
    assert lines[-1].split(",")[1] == "1800.000"


def test_densest_decile_share_is_a_fraction() -> None:
    """The concentration statistic is a share of the population."""
    raster = _raster()
    _, x, y, _ = _draw()
    col, row = raster.cell_indices(x, y)

    share = sample.densest_decile_share(col, row, raster, np.ones((4, 4), dtype=bool))

    assert 0.0 < share <= 1.0


@pytest.mark.parametrize("count_range", [(0, 100), (200, 100)])
def test_unusable_count_ranges_are_rejected(count_range: tuple[int, int]) -> None:
    """A range that is empty or inverted fails at construction."""
    with pytest.raises(ValueError):
        UeSpec(count_range=count_range, height_m=1.5)
