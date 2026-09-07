"""The tilt space: dimension order, bounds, and the vector-to-cell mapping."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.space import TiltSpace

# Two cells, two bands, with deliberately different boxes per band so a test
# that silently transposed the dimension order could not still pass.
_CONFIG = {
    "simulation": {
        "radio_map": {"bands": [{"name": "high"}, {"name": "low"}]},
        "transmitters": {
            "cells": [
                {
                    "name": "c0",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 30.0,
                    "azimuth_deg": 0.0,
                    "tilt": {
                        "high": {"baseline_deg": 8.0, "bounds_deg": [0.0, 16.0]},
                        "low": {"baseline_deg": 4.0, "bounds_deg": [2.0, 12.0]},
                    },
                },
                {
                    "name": "c1",
                    "x": 10.0,
                    "y": 0.0,
                    "z": 30.0,
                    "azimuth_deg": 120.0,
                    "tilt": {
                        "high": {"baseline_deg": 9.0, "bounds_deg": [0.0, 16.0]},
                        "low": {"baseline_deg": 5.0, "bounds_deg": [2.0, 12.0]},
                    },
                },
            ]
        },
    }
}


@pytest.fixture
def space() -> TiltSpace:
    """A two-cell, two-band space."""
    return TiltSpace.from_config(OmegaConf.create(_CONFIG))


def test_dimension_order_is_cell_major_band_minor(space: TiltSpace) -> None:
    """The order every downstream join relies on."""
    assert space.pairs == (("c0", "high"), ("c0", "low"), ("c1", "high"), ("c1", "low"))
    assert space.parameter_names == (
        "tilt_c0_high",
        "tilt_c0_low",
        "tilt_c1_high",
        "tilt_c1_low",
    )
    assert space.n_dim == 4


def test_bounds_and_baseline_come_from_the_cell_table(space: TiltSpace) -> None:
    """Per-band boxes differ, so the space is a box and not a cube."""
    assert np.array_equal(space.lower, [0.0, 2.0, 0.0, 2.0])
    assert np.array_equal(space.upper, [16.0, 12.0, 16.0, 12.0])
    assert np.array_equal(space.baseline, [8.0, 4.0, 9.0, 5.0])


def test_to_cells_round_trips_the_baseline(space: TiltSpace) -> None:
    """Rebuilding at the baseline reproduces the committed layout."""
    cells = space.to_cells(space.baseline)
    assert [cell.name for cell in cells] == ["c0", "c1"]
    assert cells[0].tilt_for("high").baseline_deg == 8.0
    assert cells[1].tilt_for("low").baseline_deg == 5.0


def test_to_cells_keeps_position_and_azimuth(space: TiltSpace) -> None:
    """Only tilt is a decision variable; the geometry is fixed."""
    cells = space.to_cells(np.array([1.0, 3.0, 2.0, 4.0]))
    assert (cells[1].x, cells[1].y, cells[1].z) == (10.0, 0.0, 30.0)
    assert cells[1].azimuth_deg == 120.0
    assert cells[1].tilt_for("high").bounds_deg == (0.0, 16.0)


def test_to_cells_rejects_a_value_outside_its_own_band_box(space: TiltSpace) -> None:
    """13 degrees is legal on ``high`` and illegal on ``low``.

    The one case a shared cube would wave through, which is why the bounds are
    per dimension rather than per space.
    """
    with pytest.raises(ValueError, match=r"c0/low"):
        space.to_cells(np.array([13.0, 13.0, 8.0, 8.0]))


def test_to_cells_rejects_a_wrong_length_or_non_finite_vector(space: TiltSpace) -> None:
    """Both would otherwise reach the ray tracer as a silent misalignment."""
    with pytest.raises(ValueError, match="expected 4 tilts"):
        space.to_cells(np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        space.to_cells(np.array([np.nan, 4.0, 9.0, 5.0]))


def test_clip_moves_a_proposal_inside_the_box(space: TiltSpace) -> None:
    """What the search applies before handing a candidate to the evaluator."""
    clipped = space.clip(np.array([-5.0, 99.0, 8.0, 5.0]))
    assert np.array_equal(clipped, [0.0, 12.0, 8.0, 5.0])
    space.to_cells(clipped)


def test_as_frame_is_one_row_per_dimension(space: TiltSpace) -> None:
    """The shape the deliverable table is built from."""
    frame = space.as_frame(space.baseline)
    assert len(frame) == space.n_dim
    assert list(frame.columns) == ["cell", "band", "tilt_deg", "tilt_min_deg", "tilt_max_deg"]
    assert frame["cell"].tolist() == ["c0", "c0", "c1", "c1"]


def test_from_config_names_a_cell_missing_a_band() -> None:
    """A cell-band pair with no tilt is a dimension with no bounds."""
    config = OmegaConf.create(_CONFIG)
    del config.simulation.transmitters.cells[1].tilt["low"]
    with pytest.raises(ValueError, match=r"c1/low"):
        TiltSpace.from_config(config)
