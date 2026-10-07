"""The tilt space: dimension order, bounds, and the vector-to-sector mapping."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.space import TiltSpace
from tests.conftest import write_sectors

# Two sectors, two bands, with deliberately different boxes per band so a test
# that silently transposed the dimension order could not still pass.
_BANDS = {
    "simulation": {"radio_map": {"bands": [{"name": "high"}, {"name": "low"}]}},
    "optim": {"tilt_resolution_deg": 0.1},
}
_SECTORS = [
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


def _config(directory, sectors=_SECTORS):
    """The bands and the sector table written under ``directory``."""
    return OmegaConf.merge(
        _BANDS, {"simulation": {"input": {"sectors_file": write_sectors(directory, sectors)}}}
    )


@pytest.fixture
def space(tmp_path) -> TiltSpace:
    """A two-sector, two-band space."""
    return TiltSpace.from_config(_config(tmp_path))


def test_dimension_order_is_sector_major_band_minor(space: TiltSpace) -> None:
    """The order every downstream join relies on."""
    assert space.pairs == (("c0", "high"), ("c0", "low"), ("c1", "high"), ("c1", "low"))
    assert space.parameter_names == (
        "tilt_c0_high",
        "tilt_c0_low",
        "tilt_c1_high",
        "tilt_c1_low",
    )
    assert space.n_dim == 4


def test_bounds_and_baseline_come_from_the_sector_table(space: TiltSpace) -> None:
    """Per-band boxes differ, so the space is a box and not a cube."""
    assert np.array_equal(space.lower, [0.0, 2.0, 0.0, 2.0])
    assert np.array_equal(space.upper, [16.0, 12.0, 16.0, 12.0])
    assert np.array_equal(space.baseline, [8.0, 4.0, 9.0, 5.0])


def test_to_sectors_round_trips_the_baseline(space: TiltSpace) -> None:
    """Rebuilding at the baseline reproduces the committed layout."""
    sectors = space.to_sectors(space.baseline)
    assert [sector.name for sector in sectors] == ["c0", "c1"]
    assert sectors[0].tilt_for("high").baseline_deg == 8.0
    assert sectors[1].tilt_for("low").baseline_deg == 5.0


def test_to_sectors_keeps_position_and_azimuth(space: TiltSpace) -> None:
    """Only tilt is a decision variable; the geometry is fixed."""
    sectors = space.to_sectors(np.array([1.0, 3.0, 2.0, 4.0]))
    assert (sectors[1].x, sectors[1].y, sectors[1].z) == (10.0, 0.0, 30.0)
    assert sectors[1].azimuth_deg == 120.0
    assert sectors[1].tilt_for("high").bounds_deg == (0.0, 16.0)


def test_to_sectors_rejects_a_value_outside_its_own_band_box(space: TiltSpace) -> None:
    """13 degrees is legal on ``high`` and illegal on ``low``.

    The one case a shared cube would wave through, which is why the bounds are
    per dimension rather than per space.
    """
    with pytest.raises(ValueError, match=r"c0/low"):
        space.to_sectors(np.array([13.0, 13.0, 8.0, 8.0]))


def test_to_sectors_rejects_a_wrong_length_or_non_finite_vector(space: TiltSpace) -> None:
    """Both would otherwise reach the ray tracer as a silent misalignment."""
    with pytest.raises(ValueError, match="expected 4 tilts"):
        space.to_sectors(np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        space.to_sectors(np.array([np.nan, 4.0, 9.0, 5.0]))


def test_clip_moves_a_proposal_inside_the_box(space: TiltSpace) -> None:
    """What the search applies before handing a candidate to the evaluator."""
    clipped = space.clip(np.array([-5.0, 99.0, 8.0, 5.0]))
    assert np.array_equal(clipped, [0.0, 12.0, 8.0, 5.0])
    space.to_sectors(clipped)


def test_as_frame_is_one_row_per_dimension(space: TiltSpace) -> None:
    """The shape the deliverable table is built from."""
    frame = space.as_frame(space.baseline)
    assert len(frame) == space.n_dim
    assert list(frame.columns) == ["sector", "band", "tilt_deg", "tilt_min_deg", "tilt_max_deg"]
    assert frame["sector"].tolist() == ["c0", "c0", "c1", "c1"]


def test_from_config_names_a_sector_missing_a_band(tmp_path) -> None:
    """A sector-band pair with no tilt is a dimension with no bounds."""
    sectors = [_SECTORS[0], {**_SECTORS[1], "tilt": {"high": _SECTORS[1]["tilt"]["high"]}}]
    with pytest.raises(ValueError, match=r"c1/low"):
        TiltSpace.from_config(_config(tmp_path, sectors))


def test_unit_cube_mapping_round_trips_and_pins_a_zero_width_dimension() -> None:
    """A dimension whose bounds coincide maps every unit value back to its one tilt."""
    space = TiltSpace(
        sectors=(),
        band_names=(),
        lower=np.array([0.0, 5.0]),
        upper=np.array([10.0, 5.0]),
        baseline=np.array([2.0, 5.0]),
        resolution_deg=0.1,
    )
    assert space.from_unit(np.array([0.25, 0.9])).tolist() == [2.5, 5.0]
    assert space.to_unit(space.baseline).tolist() == [0.2, 0.0]
    assert space.from_unit(space.to_unit(space.baseline)).tolist() == space.baseline.tolist()
    assert space.from_unit(np.array([2.0, 0.0])).tolist() == [10.0, 5.0]


def test_from_unit_snaps_to_the_resolution_lattice_from_the_lower_bound() -> None:
    """0.123 of a 10-degree span sets as 1.2; a 2-degree floor shifts the lattice."""
    space = TiltSpace(
        sectors=(),
        band_names=(),
        lower=np.array([0.0, 2.0]),
        upper=np.array([10.0, 12.0]),
        baseline=np.array([1.0, 3.0]),
        resolution_deg=0.1,
    )
    assert space.from_unit(np.array([0.123, 0.987])).tolist() == pytest.approx([1.2, 11.9])
