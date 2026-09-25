"""The seed streams and the radio-map archive, pinned so a refactor cannot move them."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.core.cell import Cell
from src.simulation import seeds
from src.simulation.grid import disc_offsets
from src.simulation.radio import Band, SolverSpec, baseline_tilts, write_radio_map


def test_the_seed_streams_do_not_move() -> None:
    """Every simulation output is keyed to these; a changed value redraws the scenario."""
    cfg = OmegaConf.create({"simulation": {"seed": 42}})
    assert {name: seeds.stream(cfg, name) for name in ("scene", "sample", "solver")} == {
        "scene": 4220626289,
        "sample": 2334169895,
        "solver": 3733018943,
    }


def test_disc_offsets_are_the_lattice_points_of_the_disc() -> None:
    """Radius 1 is the centre and its four edge neighbours, in row-major order."""
    assert disc_offsets(1) == [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    assert disc_offsets(0) == [(0, 0)]


def _cell(name: str, tilts: dict[str, float]) -> Cell:
    return Cell.from_config(
        OmegaConf.create(
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
                "max_prb": {band: 50 for band in tilts},
            }
        )
    )


def test_a_written_radio_map_reads_back_without_pickle(tmp_path) -> None:
    """Every array is plain, so the loaders' ``allow_pickle=False`` accepts it."""
    cells = (_cell("a", {"hi": 3.0, "lo": 5.0}), _cell("b", {"hi": 4.0, "lo": 6.0}))
    bands = (Band("hi", 2.6e9, 4e7, 15e3), Band("lo", 7e8, 1e7, 15e3))
    solver = SolverSpec(10, 2, True, True, False, True, False, False, True, -1, 0.95, 298.15)
    rsrp = np.full((2, 2, 3, 4), -90.0)
    grid = {"origin_x": 0.0, "origin_y": 0.0, "tile_size_m": 20.0, "n_cols": 4, "n_rows": 3}

    path = write_radio_map(
        tmp_path / "map.npz",
        rsrp=rsrp,
        sinr=rsrp - 100.0,
        bands=bands,
        cells=cells,
        grid_meta=grid,
        solver_spec=solver,
        solver_seed=7,
        height_m=1.5,
        power_dbm=4.85,
        scenario_id="scn",
        centres=np.zeros((3, 4, 3)),
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["rsrp_dbm"].shape == rsrp.shape
        np.testing.assert_array_equal(archive["tilt_deg"], [[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(archive["tilt_deg"], baseline_tilts(cells, ["hi", "lo"]))
        assert str(archive["scenario_id"]) == "scn"
