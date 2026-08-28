"""Shared fixtures.

The fixtures here are deliberately tiny and synthetic. Tests must not depend on
the real dataset: it is DVC-tracked, possibly large, and possibly confidential,
and a test suite that only runs after ``dvc pull`` stops being run.

They also must not depend on Sionna-RT. Every KPI in this project is computed
from a plain RSRP array, which is exactly what lets the objective be tested
against hand-built inputs with hand-computed answers — no simulator, no GPU, no
minutes-long scene load. :func:`rsrp_grid` is that input.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def cell_bands() -> pd.DataFrame:
    """A tiny cell-band table: three cells, two bands each.

    Returns:
        Six rows in the canonical ``(gcell_id, band)`` order — note that
        ``"high"`` sorts before ``"low"`` — carrying the geometry, current tilt
        and bounds that :func:`src.radio.cell_band.build_table` produces.

    Notes:
        Three cells, not two. Mean overlap neighbours can never exceed one with
        two cells, so a two-cell fixture cannot tell that KPI apart from overlap
        rate — which is precisely the distinction PROJECT.md section 4.6 adds it
        to make.

        Two bands per cell is likewise the minimum that separates the three
        aggregations that look alike: over bands within a cell, over cells, and
        over every cell-band pair.
    """
    return pd.DataFrame(
        {
            "gcell_id": ["cell_a", "cell_a", "cell_b", "cell_b", "cell_c", "cell_c"],
            "gnodeb_id": ["site_1", "site_1", "site_1", "site_1", "site_2", "site_2"],
            "band": ["high", "low", "high", "low", "high", "low"],
            "sim_x": [0.0, 0.0, 100.0, 100.0, 50.0, 50.0],
            "sim_y": [0.0, 0.0, 0.0, 0.0, 50.0, 50.0],
            "antenna_height": [30.0, 30.0, 30.0, 30.0, 25.0, 25.0],
            "azimuth": [0.0, 0.0, 180.0, 180.0, 90.0, 90.0],
            "current_tilt": [8.0, 6.0, 8.0, 6.0, 8.0, 6.0],
            "tilt_min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "tilt_max": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            "priority_weight": [3.0, 1.0, 3.0, 1.0, 3.0, 1.0],
        }
    )


@pytest.fixture
def rsrp_grid() -> np.ndarray:
    """A hand-built RSRP array with one grid cell per KPI case.

    Returns:
        Shape ``(6, 5)`` in dBm — six cell-bands in the row order of
        :func:`cell_bands`, and five grid cells, one per case:

        =====  ==========================================================
        index  case
        =====  ==========================================================
        0      hole: every cell-band at ``-inf``
        1      weak: strongest is ``-100``, inside ``(-120, -90]``
        2      covered, no neighbour within the margin
        3      covered, exactly one overlapping neighbour
        4      covered, two overlapping neighbours
        =====  ==========================================================

    Notes:
        With ``hole_dbm = -120``, ``weak_dbm = -90`` and
        ``overlap_margin_db = 6``, the KPIs of this array are exactly::

            hole_rate              = 20.0
            weak_rate              = 20.0
            overlap_rate           = 40.0
            mean_overlap_neighbors = 1.5
            band_priority_score    = 2.4   (with ue_density_vector)

        Those values are derived by hand from this table, and the KPI tests
        assert against them literally. Do not regenerate them by running the
        code under test — a test that asserts the implementation agrees with
        itself proves nothing.

        Grid cell 0 is the one that catches sign and masking bugs: every
        cell-band is ``-inf``, so the serving-cell difference is
        ``-inf - -inf = nan``. The overlap computation must gate on the coverage
        condition rather than relying on the comparison, or that ``nan`` decides
        the result.
    """
    ninf = -np.inf
    return np.array(
        [
            [ninf, -105.0, -70.0, -70.0, -78.0],  # cell_a / high
            [ninf, -100.0, -80.0, -82.0, -70.0],  # cell_a / low
            [ninf, -118.0, -85.0, -76.0, -73.0],  # cell_b / high
            [ninf, -115.0, -95.0, -74.0, -80.0],  # cell_b / low
            [ninf, -118.0, -92.0, -90.0, -72.0],  # cell_c / high
            [ninf, -125.0, -90.0, -98.0, -84.0],  # cell_c / low
        ]
    )


@pytest.fixture
def ue_density_vector() -> np.ndarray:
    """UE observation counts for the five grid cells of :func:`rsrp_grid`.

    Returns:
        Shape ``(5,)``, summing to ``N_UE = 200``.

    Notes:
        The zero entry is deliberate: an area with no measurements must
        contribute nothing to the Band Priority Score without being dropped from
        the grid. It is also the grid cell that is a coverage hole, so a
        Band Priority Score that changes when the hole is filled has a masking
        bug.

        The counts are uneven on purpose. With uniform density the score
        degenerates into an unweighted average over dominant bands, and a
        missing ``rho`` factor would pass unnoticed.
    """
    return np.array([0, 10, 100, 40, 50], dtype=float)


@pytest.fixture
def mdt_df() -> pd.DataFrame:
    """A small MDT frame spanning the cases the cleaning rules must catch.

    Returns:
        Twelve records over four UEs, carrying exactly one instance of each
        violation the cleaning stage is responsible for:

        =====  ==========================================================
        index  case
        =====  ==========================================================
        4      ``rsrp = 0.0`` — outside the 3GPP reporting range
        8      position far outside the scene bounding box
        10     an exact duplicate of record 9, every column included
        11     measured against a cell absent from the configuration
        =====  ==========================================================

    Notes:
        One instance each, so a cleaning test can assert an exact row count
        rather than an inequality. An assertion like ``len(cleaned) < len(df)``
        passes for a filter that drops everything.

        Record 10 duplicates record 9 in every column, ``date`` included.
        Deduplication is on the full record, so a fixture whose rows differ by a
        timestamp or a noise column has no duplicates to find and the test
        passes vacuously.
    """
    times = pd.to_datetime(
        [
            "2026-03-12T00:00Z",
            "2026-03-12T01:00Z",
            "2026-03-12T02:00Z",
            "2026-03-12T03:00Z",
            "2026-03-12T04:00Z",
            "2026-03-12T05:00Z",
            "2026-03-12T06:00Z",
            "2026-03-12T07:00Z",
            "2026-03-12T08:00Z",
            "2026-03-12T09:00Z",
            "2026-03-12T09:00Z",  # duplicate of the previous record
            "2026-03-12T11:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "ue_id": np.repeat(["ue_1", "ue_2", "ue_3", "ue_4"], 3),
            "sim_x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 9e6, 90.0, 90.0, 95.0],
            "sim_y": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 0.0, 90.0, 90.0, 95.0],
            "date": times,
            "rsrp": [
                -82.0,
                -85.0,
                -90.0,
                -95.0,
                0.0,
                -70.0,
                -75.0,
                -88.0,
                -91.0,
                -80.0,
                -80.0,
                -84.0,
            ],
            "gcell_id": [
                "cell_a",
                "cell_a",
                "cell_a",
                "cell_a",
                "cell_a",
                "cell_a",
                "cell_b",
                "cell_b",
                "cell_b",
                "cell_b",
                "cell_b",
                "cell_zzz",
            ],
            "ue_height": np.full(12, 0.5),
        }
    )


@pytest.fixture
def cfg() -> DictConfig:
    """A minimal config matching the fixtures above.

    Returns:
        The subset of ``configs/`` that the modules under test read.

    Notes:
        Keep this in step with the real config files. If a key is renamed in
        both places at once, no test notices — so at least one test composes the
        real ``configs/config.yaml`` and asserts the keys agree.

        The KPI thresholds here are the real ones from ``configs/kpi.yaml``, not
        invented values: the expected KPIs documented on :func:`rsrp_grid` are
        derived from them.
    """
    return OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "mdt_path": "data/raw/measurement_data.csv",
                "cell_config_path": "data/raw/gcell_conf.csv",
                "train_path": "data/processed/mdt_train.parquet",
                "test_path": "data/processed/mdt_test.parquet",
                "clean": {
                    "drop_duplicates": True,
                    "drop_rsrp_out_of_range": True,
                    "drop_unknown_cells": True,
                    "drop_outside_scene": True,
                },
                "split": {
                    "method": "spatial_block",
                    "group_col": "ue_id",
                    "block_size_m": 25.0,
                    "cutoff_date": None,
                    "seed": 42,
                    "test_size": 0.25,
                    "val_size": 0.25,
                },
                "schema": {
                    "mdt": {
                        "columns": {
                            "ue_id": {"dtype": "string", "nullable": False},
                            "sim_x": {"dtype": "float64", "nullable": False},
                            "sim_y": {"dtype": "float64", "nullable": False},
                            "rsrp": {
                                "dtype": "float64",
                                "min": -156.0,
                                "max": -31.0,
                                "nullable": False,
                            },
                            "gcell_id": {"dtype": "string", "nullable": False},
                            "ue_height": {"dtype": "float64", "nullable": False},
                        }
                    },
                    "cell_config": {
                        "columns": {
                            "gcell_id": {"dtype": "string", "nullable": False},
                            "azimuth": {
                                "dtype": "float64",
                                "min": 0.0,
                                "max": 360.0,
                                "nullable": False,
                            },
                        }
                    },
                },
            },
            "radio": {
                "bands": {
                    "low": {
                        "carrier_hz": 7.0e8,
                        "tx_power_dbm": 46.0,
                        "priority_weight": 1.0,
                        "tilt": {"min": 0.0, "max": 15.0},
                    },
                    "high": {
                        "carrier_hz": 3.5e9,
                        "tx_power_dbm": 46.0,
                        "priority_weight": 3.0,
                        "tilt": {"min": 0.0, "max": 15.0},
                    },
                },
                "cells": {"from_cell_config": False, "default": ["low", "high"], "overrides": {}},
                "grid": {"cell_size_m": 50.0, "height_m": 0.5, "bounds": [0.0, 0.0, 250.0, 50.0]},
                "sampling": {"strategy": "sobol", "n_samples": 8, "step_deg": None, "seed": 42},
            },
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "mode": "lexicographic",
                "order": [
                    "hole_rate",
                    "overlap_rate",
                    "weak_rate",
                    "mean_overlap_neighbors",
                    "band_priority_score",
                ],
                "direction": {
                    "hole_rate": "minimize",
                    "overlap_rate": "minimize",
                    "weak_rate": "minimize",
                    "mean_overlap_neighbors": "minimize",
                    "band_priority_score": "maximize",
                },
                "tolerance": {
                    "hole_rate": 0.1,
                    "overlap_rate": 0.1,
                    "weak_rate": 0.1,
                    "mean_overlap_neighbors": 0.05,
                },
                "normalization": "reference",
                "reference": None,
                "weights": {
                    "hole_rate": 5.0,
                    "overlap_rate": 3.0,
                    "weak_rate": 1.0,
                    "mean_overlap_neighbors": 0.5,
                    "band_priority_score": 0.5,
                },
            },
        }
    )
