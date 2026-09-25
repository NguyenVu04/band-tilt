"""The MDT selection, on a map small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.simulation import mdt
from src.simulation.sample import CSV_COLUMNS


def test_mdt_keeps_served_ues_with_their_serving_rsrp() -> None:
    """Tile (0, 0) is covered on 'lo' only; tile (0, 1) has no path, so its UE is dropped."""
    cfg = OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "capacity": {
                    "band_preference": ["hi", "lo"],
                    "rsrp_threshold_dbm": -100.0,
                    "max_admission_utilisation": 1.0,
                    "throughput_per_ue_bps": 180_000.0,
                },
            },
            "simulation": {
                "radio_map": {"bands": [{"name": n, "scs_hz": 15000} for n in ("hi", "lo")]},
                "seed": 0,
                "transmitters": {
                    "cells": [
                        {"name": "c0", "x": 0.0, "y": 0.0, "z": 30.0, "azimuth_deg": 0.0}
                        | {"tilt": {}, "max_prb": {"hi": 100, "lo": 100}}
                    ]
                },
            },
        }
    )
    rsrp = np.array([[[[np.nan, np.nan]]], [[[-95.0, np.nan]]]])  # [band, tx, row, col]
    sinr = np.where(np.isfinite(rsrp), 0.0, np.nan)
    ue = pd.DataFrame(
        [[0, 0.0, 10.0, 10.0, 1.5, 0, 0, -1], [0, 0.0, 30.0, 10.0, 1.5, 1, 0, -1]],
        columns=list(CSV_COLUMNS),
    )

    table = mdt.select(rsrp, sinr, ["hi", "lo"], ue, cfg)

    # The demand map counts rows, so the MDT carries no PRB column to count.
    assert list(table.columns) == [*CSV_COLUMNS, "rsrp_dbm"] == list(mdt.MDT_COLUMNS)
    assert table["x"].tolist() == [10.0]
    assert table["rsrp_dbm"].tolist() == [-95.0]
