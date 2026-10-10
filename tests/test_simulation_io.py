"""The seed streams and the radio-map schema, pinned so a refactor cannot move them."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.sector import Sector
from src.simulation import radio, seeds
from src.simulation.radio import Band, SolverSpec, baseline_tilts, radio_map
from tests.conftest import sector_from_mapping


def test_the_seed_streams_do_not_move() -> None:
    """Every simulation output is keyed to these; a changed value redraws the scenario."""
    assert {name: seeds.stream(42, name) for name in ("scene", "sample", "solver")} == {
        "scene": 4220626289,
        "sample": 2334169895,
        "solver": 3733018943,
    }


def _sector(name: str, tilts: dict[str, float], max_prb: int = 50) -> Sector:
    return sector_from_mapping(
        {
            "name": name,
            "x": 0.0,
            "y": 0.0,
            "z": 25.0,
            "azimuth_deg": 0.0,
            "tilt": {
                band: {"baseline_deg": tilt, "bounds_deg": [0.0, 15.0]}
                for band, tilt in tilts.items()
            },
            "max_prb": {band: max_prb for band in tilts},
        }
    )


def test_max_prb_must_be_n_rb_for_the_band() -> None:
    """A max_prb left behind by a bandwidth change is refused before any solve."""
    sectors = (_sector("a", {"b700": 8.0}, max_prb=52),)
    radio._check_tilt_table(sectors, (Band("b700", 7e8, 1e7, 15e3),))
    with pytest.raises(ValueError, match="a/b700 has 52, needs 106"):
        radio._check_tilt_table(sectors, (Band("b700", 7e8, 2e7, 15e3),))
    with pytest.raises(ValueError, match="No N_RB"):
        radio._check_tilt_table(sectors, (Band("b700", 7e8, 1.2e7, 15e3),))


def test_a_radio_map_carries_its_tilts_and_scenario() -> None:
    """The map records the tilt it was solved at, [band, tx], and its scenario."""
    sectors = (_sector("a", {"hi": 3.0, "lo": 5.0}), _sector("b", {"hi": 4.0, "lo": 6.0}))
    bands = (Band("hi", 2.6e9, 4e7, 15e3), Band("lo", 7e8, 1e7, 15e3))
    solver = SolverSpec(10, 2, True, True, False, True, False, False, True, -1, 0.95, 298.15)
    rsrp = np.full((2, 2, 3, 4), -90.0)
    grid = {"origin_x": 0.0, "origin_y": 0.0, "tile_size_m": 20.0, "n_cols": 4, "n_rows": 3}

    arrays = radio_map(
        rsrp=rsrp,
        sinr=rsrp - 100.0,
        bands=bands,
        sectors=sectors,
        grid_meta=grid,
        solver_spec=solver,
        solver_seed=7,
        height_m=1.5,
        power_dbm=4.85,
        scenario_id="scn",
        centres=np.zeros((3, 4, 3)),
    )
    assert arrays["rsrp_dbm"].shape == rsrp.shape
    np.testing.assert_array_equal(arrays["tilt_deg"], [[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(arrays["tilt_deg"], baseline_tilts(sectors, ["hi", "lo"]))
    assert str(arrays["scenario_id"]) == "scn"
