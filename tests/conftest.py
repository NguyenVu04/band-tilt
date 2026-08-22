"""Shared fixtures.

The fixtures here are deliberately tiny and synthetic. Tests must not depend on
the real dataset: it is DVC-tracked, possibly large, and possibly confidential,
and a test suite that only runs after ``dvc pull`` stops being run.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def grouped_df() -> pd.DataFrame:
    """A small frame with repeated group values.

    Returns:
        Twelve records spread over four groups, three records each — enough to
        catch a splitter that ignores grouping, small enough to read in a
        failure message.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat([1, 2, 3, 4], 3)
    return pd.DataFrame(
        {
            "record_id": range(12),
            "group_col": groups,
            "feature_a": rng.normal(size=12),
            "feature_b": rng.integers(0, 3, size=12),
            "target": rng.normal(size=12),
        }
    )


@pytest.fixture
def cfg() -> DictConfig:
    """A minimal config matching :func:`grouped_df`.

    Returns:
        The subset of ``configs/data.yaml`` that the data modules read. Keep it
        in step with the real file — if a key is renamed there and here at the
        same time, no test will notice, so also assert against the real config
        in at least one test.
    """
    return OmegaConf.create(
        {
            "seed": 42,
            "target": "target",
            "drop_columns": ["record_id"],
            "split": {
                "method": "group_shuffle",
                "group_col": "group_col",
                "stratify_col": None,
                "seed": 42,
                "test_size": 0.25,
                "val_size": 0.25,
            },
            "schema": {
                "columns": {
                    "feature_a": {"dtype": "float64", "min": -5, "max": 5, "nullable": False},
                    "feature_b": {"dtype": "int64", "nullable": False},
                    "group_col": {"dtype": "int64", "nullable": False},
                    "target": {"dtype": "float64", "nullable": False},
                }
            },
        }
    )
