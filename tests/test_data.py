"""The preprocessing contract: every check holds on consistent artifacts, and each lie fails it."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.core.cell import cells_from_frame, cells_to_frame, read_cells, site_frame
from src.data import schema
from src.data.build import build_ue
from src.data.load import Artifacts

_CELLS = pd.DataFrame(
    {
        "node": ["n0"],
        "node_x": [0.0],
        "node_y": [0.0],
        "node_z": [25.0],
        "cell": ["c0"],
        "azimuth_deg": [0.0],
        "band": ["b1"],
        "tilt_deg": [8.0],
        "tilt_min_deg": [0.0],
        "tilt_max_deg": [20.0],
        "max_prb": [52],
    }
)


@pytest.fixture
def cfg(tmp_path):
    """One cell on one band over a 2 x 2 grid of 10 m tiles."""
    cells_file = tmp_path / "cells.csv"
    _CELLS.to_csv(cells_file, index=False)
    return OmegaConf.create(
        {
            "simulation": {
                "ue": {"height_m": 1.5},
                "input": {"cells_file": str(cells_file)},
                "antenna": {"power_rs": 5.0},
                "radio_map": {"bands": [{"name": "b1", "frequency": 700000000}]},
            },
        }
    )


@pytest.fixture
def artifacts() -> Artifacts:
    """Three UEs over two intervals, written in an order ``build_ue`` must not keep."""
    grid = {"origin_x": 0.0, "origin_y": 0.0, "tile_size_m": 10.0, "n_cols": 2, "n_rows": 2}
    ue = pd.DataFrame(
        {
            "t_index": [1, 0, 0],
            "t_s": [950.0, 10.0, 20.0],
            "x": [15.0, 5.0, 12.0],
            "y": [15.0, 5.0, 3.0],
            "z": [1.5, 1.5, 1.5],
            "tile_col": [1, 0, 1],
            "tile_row": [1, 0, 0],
            "component": [-1, 0, 0],
        }
    )
    radio = {
        "tx_name": np.array(["c0"]),
        "band_label": np.array(["b1"]),
        "band_hz": np.array([7e8]),
        "scenario_id": np.array("scn_x"),
        "ue_height_m": np.array(1.5),
        "rsrp_dbm": np.full((1, 1, 2, 2), -80.0, dtype=np.float32),
        "sinr_db": np.full((1, 1, 2, 2), 10.0, dtype=np.float32),
        "tilt_deg": np.array([[8.0]]),
        **{key: np.array(value) for key, value in grid.items()},
    }
    manifest = {
        "scenario_id": "scn_x",
        "grid": grid,
        "time": {"t_s": [0.0, 900.0], "interval_s": 900.0},
    }
    return Artifacts(ue=ue, radio=radio, manifest=manifest)


def test_consistent_artifacts_pass_every_check(artifacts, cfg) -> None:
    """Nothing to report on a scenario whose files agree with the config."""
    checks = schema.verify(artifacts, cfg)
    assert checks["holds"].all(), checks[~checks["holds"]]
    schema.require(checks)


def test_the_synthetic_component_column_is_optional(artifacts, cfg) -> None:
    """Real data carries no mixture component, and still meets the contract."""
    ue = artifacts.ue.drop(columns="component")
    assert schema.verify(dataclasses.replace(artifacts, ue=ue), cfg)["holds"].all()


def _ue(artifacts: Artifacts, **columns) -> Artifacts:
    ue = artifacts.ue.copy()
    for name, values in columns.items():
        ue[name] = values
    return dataclasses.replace(artifacts, ue=ue)


def _radio(artifacts: Artifacts, **arrays) -> Artifacts:
    return dataclasses.replace(artifacts, radio={**artifacts.radio, **arrays})


def _manifest_without(artifacts: Artifacts, *path: str) -> Artifacts:
    manifest = copy.deepcopy(artifacts.manifest)
    node = manifest
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return dataclasses.replace(artifacts, manifest=manifest)


def _sinr_nan_where_rsrp_is_not(artifacts: Artifacts) -> Artifacts:
    sinr = artifacts.radio["sinr_db"].copy()
    sinr[0, 0, 0, 0] = np.nan
    return _radio(artifacts, sinr_db=sinr)


# Each wrong artifact, and the one check it must fail.
_LIES = {
    "NaN x": (lambda a: _ue(a, x=[np.nan, 5.0, 12.0]), "no missing ue value"),
    "NaN t_s": (lambda a: _ue(a, t_s=[np.nan, 10.0, 20.0]), "no missing ue value"),
    "NaN tile_row": (
        lambda a: _ue(a, tile_row=[np.nan, 0, 0]),
        "integer ue columns hold integers",
    ),
    "fractional t_index": (
        lambda a: _ue(a, t_index=[0.5, 0.0, 0.0]),
        "integer ue columns hold integers",
    ),
    "fractional component": (
        lambda a: _ue(a, component=[0.5, 0.0, 0.0]),
        "integer ue columns hold integers",
    ),
    "string x": (lambda a: _ue(a, x=["15", "5", "12"]), "real ue columns are numeric"),
    "tile disagrees with x": (
        lambda a: _ue(a, tile_col=[0, 0, 1]),
        "tile_col, tile_row are the tile holding x, y",
    ),
    "missing column": (
        lambda a: dataclasses.replace(a, ue=a.ue.drop(columns="z")),
        "ue columns are the contract's",
    ),
    "unknown column": (lambda a: _ue(a, rsrp=[1.0, 2.0, 3.0]), "ue columns are the contract's"),
    "empty UE table": (
        lambda a: dataclasses.replace(a, ue=a.ue.iloc[:0]),
        "ue table holds at least one row",
    ),
    "retuned carrier": (
        lambda a: _radio(a, band_hz=np.array([1.8e9])),
        "npz band_hz matches the configured frequencies",
    ),
    "SINR without RSRP NaN": (
        _sinr_nan_where_rsrp_is_not,
        "npz sinr_db is NaN exactly where rsrp_dbm is",
    ),
    "no scenario_id": (
        lambda a: _manifest_without(a, "scenario_id"),
        "manifest carries scenario_id, grid and time",
    ),
    "no grid": (
        lambda a: _manifest_without(a, "grid"),
        "manifest carries scenario_id, grid and time",
    ),
    "no interval_s": (
        lambda a: _manifest_without(a, "time", "interval_s"),
        "manifest carries scenario_id, grid and time",
    ),
    "t_s after its interval": (
        lambda a: _ue(a, t_s=[5000.0, 10.0, 20.0]),
        "t_s lies inside the interval its t_index names",
    ),
}


@pytest.mark.parametrize("lie", list(_LIES))
def test_each_lie_fails_its_check_and_require_raises(artifacts, cfg, lie) -> None:
    """A wrong artifact is named by the check it breaks, never crashes ``verify``."""
    corrupt, check = _LIES[lie]
    checks = schema.verify(corrupt(artifacts), cfg)
    assert check in checks.loc[~checks["holds"], "check"].tolist()
    with pytest.raises(schema.SchemaError, match="schema checks failed"):
        schema.require(checks)


def test_a_missing_cell_table_fails_the_contract(artifacts, cfg) -> None:
    """The radio map is checked against the cell table, so no table is a failure, not a crash."""
    cfg.simulation.input.cells_file = "no/such/cells.csv"
    checks = schema.verify(artifacts, cfg)
    assert checks.loc[~checks["holds"], "check"].tolist() == ["cell table reads"]


def test_a_map_solved_at_other_tilts_fails_the_contract(artifacts, cfg) -> None:
    """The map must be the one the cell table's baseline tilts produce."""
    checks = schema.verify(_radio(artifacts, tilt_deg=np.array([[9.0]])), cfg)
    failed = checks.loc[~checks["holds"], "check"].tolist()
    assert failed == ["npz tilt_deg equals the cell table's baseline tilts"]


def test_build_ue_types_and_sorts_without_dropping_a_row(artifacts) -> None:
    """Every UE survives preprocessing, in producer-independent order."""
    ue = build_ue(artifacts)
    assert len(ue) == len(artifacts.ue)
    assert ue["t_index"].tolist() == [0, 0, 1]
    assert ue["tile_col"].tolist() == [0, 1, 1]
    assert ue["tile_row"].dtype == np.int16
    assert ue["component"].dtype == np.int16
    assert ue["t_s"].dtype == np.float64
    assert ue["scenario_id"].astype(str).unique().tolist() == ["scn_x"]


def test_build_ue_keeps_a_table_without_component(artifacts) -> None:
    """The optional column is typed when present and not invented when absent."""
    ue = build_ue(dataclasses.replace(artifacts, ue=artifacts.ue.drop(columns="component")))
    assert "component" not in ue.columns


def test_a_cell_table_round_trips_through_its_frame() -> None:
    """Writing cells and reading them back keeps every tilt, bound and PRB limit."""
    cells = cells_from_frame(_CELLS)
    again = cells_from_frame(cells_to_frame(cells))
    assert again == cells
    assert again[0].tilt["b1"].bounds_deg == (0.0, 20.0)
    assert again[0].max_prb_for("b1") == 52


def test_a_tilt_outside_its_bounds_is_refused() -> None:
    """The run would start infeasible, so the cell table does not load."""
    with pytest.raises(ValueError, match="outside its bounds"):
        cells_from_frame(_CELLS.assign(tilt_deg=25.0))


def test_a_repeated_cell_band_row_is_refused() -> None:
    """Two rows for one pair would otherwise let the later one silently win."""
    with pytest.raises(ValueError, match="more than one row for c0/b1"):
        cells_from_frame(pd.concat([_CELLS, _CELLS.assign(tilt_deg=9.0)]))


def test_the_site_frame_is_one_row_per_cell_with_its_node(tmp_path) -> None:
    """Two band rows of one cell collapse to one site, its mast named by ``node``."""
    path = tmp_path / "cells.csv"
    pd.concat([_CELLS, _CELLS.assign(band="b2", tilt_deg=6.0)]).to_csv(path, index=False)
    sites = site_frame(read_cells(path))
    assert sites.to_dict("records") == [
        {"cell": "c0", "node": "n0", "x": 0.0, "y": 0.0, "z": 25.0, "azimuth_deg": 0.0}
    ]
