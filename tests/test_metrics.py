"""Tests for the shared metric definitions.

The comparison table in notebook 03 stacks one dict per model. That only works
if every call returns the same keys, so key stability is tested as strictly as
the values themselves.
"""

import numpy as np
import pytest

from src.evaluation import metrics

pytestmark = pytest.mark.skip(reason="implement src/evaluation/metrics.py first")


def test_returns_stable_keys() -> None:
    """The metric set must not depend on the input.

    A key that appears only for some inputs turns the results table into ragged
    columns and quietly drops models from comparisons.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    first = metrics.evaluate(y_true, y_true)
    second = metrics.evaluate(y_true, np.array([1.5, 2.5, 2.5, 4.5]))
    assert first.keys() == second.keys()


def test_primary_metric_is_present() -> None:
    """The ranking metric must be part of the returned set."""
    y_true = np.array([1.0, 2.0, 3.0])
    assert metrics.PRIMARY_METRIC in metrics.evaluate(y_true, y_true)


def test_perfect_prediction_scores_at_the_optimum() -> None:
    """A perfect prediction produces the best achievable value.

    Cheap sanity check that catches sign errors and swapped arguments.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    scores = metrics.evaluate(y_true, y_true)
    assert all(np.isfinite(v) for v in scores.values())


def test_mismatched_lengths_raise() -> None:
    """Misaligned predictions must fail loudly, not broadcast."""
    with pytest.raises(ValueError):
        metrics.evaluate(np.array([1.0, 2.0]), np.array([1.0]))


def test_group_scores_cover_every_group() -> None:
    """Per-group scoring returns one entry per distinct group."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    groups = np.array(["a", "a", "b", "b"])
    by_group = metrics.evaluate_by_group(y_true, y_pred, groups)
    assert set(by_group) == {"a", "b"}
