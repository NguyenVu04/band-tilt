"""The preprocessing contract: every check holds on consistent artifacts, and one lie fails it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data import schema
from src.data.build import build_ue
from src.data.load import Artifacts


@pytest.fixture
def cfg():
    """One cell on one band over a 2 x 2 grid of 10 m tiles."""
    return OmegaConf.create(
        {
            "simulation": {
                "ue": {"height_m": 1.5},
                "antenna": {"power_rs": 5.0},
                "radio_map": {"bands": [{"name": "b1"}]},
                "transmitters": {
                    "cells": [
                        {
                            "name": "c0",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 25.0,
                            "azimuth_deg": 0.0,
                            "tilt": {"b1": {"baseline_deg": 8.0, "bounds_deg": [0.0, 15.0]}},
                        }
                    ]
                },
            }
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
        "time": {"t_s": [0.0, 900.0], "spec": {"interval_s": 900.0}},
    }
    return Artifacts(ue=ue, radio=radio, manifest=manifest)


def test_consistent_artifacts_pass_every_check(artifacts, cfg) -> None:
    """Nothing to report on a scenario whose three files agree with the config."""
    checks = schema.verify(artifacts, cfg)
    assert checks["holds"].all(), checks[~checks["holds"]]
    schema.require(checks)


def test_a_ue_outside_its_interval_fails_the_contract(artifacts, cfg) -> None:
    """A report timed after its interval ends is a writer fault, and names itself."""
    ue = artifacts.ue.copy()
    ue.loc[0, "t_s"] = 5000.0
    checks = schema.verify(dataclasses.replace(artifacts, ue=ue), cfg)
    failed = checks.loc[~checks["holds"], "check"].tolist()
    assert failed == ["t_s lies inside the interval its t_index names"]
    with pytest.raises(schema.SchemaError, match="t_s lies inside"):
        schema.require(checks)


def test_build_ue_types_and_sorts_without_dropping_a_row(artifacts) -> None:
    """Every drawn UE survives preprocessing, in simulator-independent order."""
    ue = build_ue(artifacts)
    assert len(ue) == len(artifacts.ue)
    assert ue["t_index"].tolist() == [0, 0, 1]
    assert ue["tile_col"].tolist() == [0, 1, 1]
    assert ue["tile_row"].dtype == np.int16
    assert ue["t_s"].dtype == np.float64
    assert ue["scenario_id"].astype(str).unique().tolist() == ["scn_x"]
