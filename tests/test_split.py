"""Tests for the shared splitter.

Splitting bugs do not raise — they produce results that look better than they
are, and nothing downstream notices. MDT makes this worse than usual: records
are correlated both spatially and temporally, so a split that looks random and
balanced can still put a measurement's near-twin on the other side of the
boundary. See PROJECT.md section 12.3.

Every test is skipped until ``src/data/split.py`` is implemented; the skip list
is the implementation checklist.
"""

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data import split

pytestmark = pytest.mark.skip(reason="implement src/data/split.py first")


def test_split_is_exhaustive_and_disjoint(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Every record lands in exactly one split — none lost, none duplicated."""
    train_df, test_df = split.train_test_split(mdt_df, cfg)
    assert len(train_df) + len(test_df) == len(mdt_df)
    assert train_df.index.intersection(test_df.index).empty


def test_split_is_deterministic(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The same config and seed must reproduce the same partition.

    The test split is frozen from notebook 01 until notebook 06. Without
    determinism a re-run silently evaluates against a different test set and
    results stop being comparable across commits.
    """
    first = split.train_test_split(mdt_df, cfg)
    second = split.train_test_split(mdt_df, cfg)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_spatial_split_shares_no_block(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Under ``spatial_block``, no spatial block straddles the boundary.

    Consecutive MDT points on one trajectory are metres apart and see almost the
    same propagation environment. Splitting between them scores the model on
    something it effectively trained on.
    """
    train_df, test_df = split.train_test_split(mdt_df, cfg)
    size = cfg.data.split.block_size_m

    def blocks(df: pd.DataFrame) -> set:
        return set(zip(df["sim_x"] // size, df["sim_y"] // size, strict=True))

    assert blocks(train_df) & blocks(test_df) == set()


def test_temporal_split_has_no_overlapping_window(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Under ``temporal``, the test window starts after the train window ends.

    One UE contributes a burst of measurements over one session, during which
    the network configuration is constant — so an interleaved split leaks the
    configuration as much as the position.
    """
    temporal = cfg.copy()
    temporal.data.split.method = "temporal"
    train_df, test_df = split.train_test_split(mdt_df, temporal)
    assert train_df["date"].max() <= test_df["date"].min()


def test_group_split_shares_no_group(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Under ``group_shuffle``, no ``group_col`` value appears on both sides."""
    grouped = cfg.copy()
    grouped.data.split.method = "group_shuffle"
    train_df, test_df = split.train_test_split(mdt_df, grouped)
    col = cfg.data.split.group_col
    assert set(train_df[col]) & set(test_df[col]) == set()


def test_test_size_is_approximately_respected(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The test split is close to the configured fraction.

    Leakage-safe splitting cannot hit the fraction exactly — whole blocks or
    groups move together — so allow roughly one unit of tolerance.
    """
    _, test_df = split.train_test_split(mdt_df, cfg)
    expected = len(mdt_df) * cfg.data.split.test_size
    assert abs(len(test_df) - expected) <= 3


def test_val_split_uses_config_val_size(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """``val_size`` comes from the data config, not from a model config.

    If the surrogate and each optimizer set their own validation fraction, they
    would tune against different data and the comparison in notebook 06 would be
    measuring the folds as much as the methods.
    """
    train_df, val_df = split.train_val_split(mdt_df, cfg)
    expected = len(mdt_df) * cfg.data.split.val_size
    assert abs(len(val_df) - expected) <= 3
    assert train_df.index.intersection(val_df.index).empty


def test_validation_uses_the_same_method_as_the_outer_split(
    mdt_df: pd.DataFrame, cfg: DictConfig
) -> None:
    """The inner split must be leakage-safe in the same way as the outer one.

    Validating on a random fold of a spatially split training set measures a
    different kind of generalisation from the one the test set measures, so the
    two numbers cannot be read together.
    """
    train_df, val_df = split.train_val_split(mdt_df, cfg)
    size = cfg.data.split.block_size_m

    def blocks(df: pd.DataFrame) -> set:
        return set(zip(df["sim_x"] // size, df["sim_y"] // size, strict=True))

    assert blocks(train_df) & blocks(val_df) == set()


def test_leakage_assertion_raises_on_overlap(mdt_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The guard must actually fire when the two sides overlap.

    A guard that never fails is worse than none: it is read as evidence.
    """
    with pytest.raises(AssertionError):
        split.assert_no_leakage(mdt_df, mdt_df, cfg)
