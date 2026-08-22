"""Tests for schema enforcement and cleaning.

These tests protect the notebook 01 boundary: cleaning may drop records that
violate externally known bounds, and may not apply anything learned from the
data.
"""

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data import schema

pytestmark = pytest.mark.skip(reason="implement src/data/schema.py first")


def test_valid_frame_passes(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """A conforming frame is returned unchanged in shape."""
    validated = schema.validate(grouped_df, cfg)
    assert len(validated) == len(grouped_df)


def test_missing_column_raises(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """A declared column that is absent is an error, not a warning.

    Silently tolerating it means the model trains on a different feature set
    than the one the config describes.
    """
    with pytest.raises(schema.SchemaError):
        schema.validate(grouped_df.drop(columns=["feature_a"]), cfg)


def test_null_in_non_nullable_column_raises(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Nulls where the contract forbids them must be caught before splitting."""
    broken = grouped_df.copy()
    broken.loc[0, "feature_a"] = None
    with pytest.raises(schema.SchemaError):
        schema.validate(broken, cfg)


def test_out_of_bounds_value_is_reported(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Values outside a declared hard bound are found by ``find_violations``.

    The bound comes from the config, whose ``source`` field records where the
    limit is defined — never from the data itself.
    """
    broken = grouped_df.copy()
    broken.loc[0, "feature_a"] = 999.0
    violations = schema.find_violations(broken, cfg)
    assert len(violations) == 1


def test_dtype_is_enforced(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Declared dtypes are applied, so a reload cannot change types silently."""
    validated = schema.validate(grouped_df, cfg)
    assert str(validated["feature_a"].dtype) == "float64"
