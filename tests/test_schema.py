"""Tests for schema enforcement and cleaning.

These protect the notebook 01 boundary: cleaning may drop records that violate
externally known bounds, and may not apply anything learned from the data.

The RSRP bound is the worked example. ``[-156, -31]`` dBm is the 3GPP TS 38.133
reporting range, so dropping the ``rsrp = 0.0`` records in the raw export is a
contract decision with a citable source — not an outlier judgement made after
looking at the distribution.
"""

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data import clean, schema

pytestmark = pytest.mark.skip(reason="implement src/data/schema.py and clean.py first")


def test_valid_frame_passes(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """A conforming frame is returned unchanged in shape."""
    conforming = mdt_df[mdt_df["rsrp"] < 0]
    assert len(schema.validate(conforming, cfg, contract="mdt")) == len(conforming)


def test_missing_column_raises(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """A declared column that is absent is an error, not a warning.

    Tolerating it silently means the pipeline runs against a different set of
    fields than the config describes.
    """
    with pytest.raises(schema.SchemaError):
        schema.validate(mdt_df.drop(columns=["rsrp"]), cfg, contract="mdt")


def test_null_in_non_nullable_column_raises(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Nulls where the contract forbids them must be caught before splitting."""
    broken = mdt_df.copy()
    broken.loc[0, "rsrp"] = None
    with pytest.raises(schema.SchemaError):
        schema.validate(broken, cfg, contract="mdt")


def test_unreportable_rsrp_is_a_violation(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """``rsrp = 0.0`` is outside the 3GPP reporting range.

    The raw export contains it, and it is best read as a sentinel for "no
    report" rather than a measurement.
    """
    violations = schema.find_violations(mdt_df, cfg, contract="mdt")
    assert (violations["column"] == "rsrp").any()


def test_placeholder_columns_are_skipped(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """A declared column whose name is still ``<placeholder>`` is not required.

    ``configs/data.yaml`` documents the multi-band columns before they exist.
    Treating them as required would fail every load until the export arrives.
    """
    pending = cfg.copy()
    pending.data.schema.mdt.columns["<band-column>"] = {"dtype": "string", "nullable": False}
    conforming = mdt_df[mdt_df["rsrp"] < 0]
    assert len(schema.validate(conforming, pending, contract="mdt")) == len(conforming)


def test_dtype_is_enforced(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Declared dtypes are applied, so a reload cannot change types silently."""
    conforming = mdt_df[mdt_df["rsrp"] < 0]
    validated = schema.validate(conforming, cfg, contract="mdt")
    assert str(validated["rsrp"].dtype) == "float64"


def test_cleaning_drops_unreportable_rsrp(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The out-of-range record is removed and the rest survive."""
    cleaned = clean.drop_invalid_rsrp(mdt_df, cfg)
    assert (cleaned["rsrp"] != 0.0).all()
    assert len(cleaned) == len(mdt_df) - 1


def test_cleaning_drops_measurements_against_unknown_cells(
    mdt_df: pd.DataFrame, cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """A measurement against an unconfigured cell cannot be placed in the scene.

    Without a position, azimuth and tilt for that cell there is nothing to
    compare a simulated radio map against.
    """
    cells = cell_bands[["gcell_id"]].drop_duplicates()
    cleaned = clean.drop_unknown_cells(mdt_df, cells, cfg)
    assert set(cleaned["gcell_id"]) <= set(cells["gcell_id"])


def test_cleaning_removes_exact_duplicates(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Deduplication is on the full record, not on ``ue_id``.

    One UE legitimately contributes many measurements — that is what an MDT
    export is — so deduplicating by identifier would delete most of the dataset.
    """
    cleaned = clean.drop_duplicates(mdt_df, cfg)
    assert len(cleaned) == len(mdt_df) - 1
    assert cleaned["ue_id"].nunique() == mdt_df["ue_id"].nunique()


def test_cleaning_never_uses_a_data_derived_threshold(
    mdt_df: pd.DataFrame, cfg: DictConfig
) -> None:
    """Cleaning a subset must drop exactly the records it would in the full set.

    This is the leakage guard. A filter fitted on the data — a quantile, an IQR
    fence — gives different answers on different subsets, and it has then seen
    the test records while deciding what counts as valid.
    """
    subset = mdt_df.iloc[:6]
    from_full = clean.drop_invalid_rsrp(mdt_df, cfg).index.intersection(subset.index)
    assert set(clean.drop_invalid_rsrp(subset, cfg).index) == set(from_full)
