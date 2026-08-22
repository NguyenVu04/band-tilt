"""Tests for the shared splitter.

These are the highest-value tests in the project. Splitting bugs do not raise —
they produce results that look better than they are, and nothing downstream
notices.

Every test is skipped until ``src/data/split.py`` is implemented; the skip list
is the implementation checklist.
"""

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data import split

pytestmark = pytest.mark.skip(reason="implement src/data/split.py first")


def test_no_group_appears_in_both_splits(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """No group may straddle the train/test boundary.

    Correlated records on both sides mean the model is scored on data it
    effectively trained on, and the test estimate comes out optimistic.
    """
    train_df, test_df = split.train_test_split(grouped_df, cfg)
    assert set(train_df["group_col"]) & set(test_df["group_col"]) == set()


def test_split_is_exhaustive_and_disjoint(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """Every record lands in exactly one split — none lost, none duplicated."""
    train_df, test_df = split.train_test_split(grouped_df, cfg)
    assert len(train_df) + len(test_df) == len(grouped_df)
    assert set(train_df["record_id"]).isdisjoint(test_df["record_id"])


def test_split_is_deterministic(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The same config and seed must reproduce the same partition.

    Without this, a re-run silently evaluates against a different test set and
    results stop being comparable across commits.
    """
    first = split.train_test_split(grouped_df, cfg)
    second = split.train_test_split(grouped_df, cfg)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_test_size_is_respected(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The test split is close to the configured fraction.

    Group-aware splitting cannot hit the fraction exactly — whole groups move
    together — so allow a tolerance of roughly one group.
    """
    _, test_df = split.train_test_split(grouped_df, cfg)
    expected = len(grouped_df) * cfg.split.test_size
    assert abs(len(test_df) - expected) <= 3


def test_val_split_uses_config_val_size(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """``val_size`` comes from the data config, not from a model config.

    If a model could set its own validation fraction, models would tune against
    different data and notebook 03's comparison would be meaningless.
    """
    train_df, val_df = split.train_val_split(grouped_df, cfg)
    expected = len(grouped_df) * cfg.split.val_size
    assert abs(len(val_df) - expected) <= 3
    assert set(train_df["group_col"]) & set(val_df["group_col"]) == set()


def test_leakage_assertion_raises_on_overlap(grouped_df: pd.DataFrame, cfg: DictConfig) -> None:
    """The guard must actually fire when groups overlap."""
    with pytest.raises(AssertionError):
        split.assert_no_group_leakage(grouped_df, grouped_df, cfg)
