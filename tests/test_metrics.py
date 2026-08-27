"""Tests for surrogate prediction error.

Not tests of the KPIs themselves — those are in ``test_kpi.py``. These check how
faithfully the surrogate reproduces them, which is what decides whether an
optimization result computed against the surrogate is defensible at all.

The reports stack one dict per surrogate variant into a comparison table, so key
stability is tested as strictly as the values.
"""

import numpy as np
import pytest

from src.evaluation import metrics
from src.kpi.vector import KPI_NAMES

pytestmark = pytest.mark.skip(reason="implement src/evaluation/metrics.py first")


def test_reports_every_kpi_separately() -> None:
    """Error is reported per KPI, never averaged across them.

    A surrogate that is excellent on weak rate and useless on hole rate is
    useless, because hole rate is the highest-priority objective. A mean over
    the five hides exactly that, in units that mean nothing.
    """
    y = np.tile(np.arange(5.0), (8, 1))
    assert set(metrics.prediction_error(y, y, KPI_NAMES)) == set(KPI_NAMES)


def test_returns_stable_keys() -> None:
    """The metric set must not depend on the input.

    A key that appears only for some inputs turns the comparison table into
    ragged columns and quietly drops variants from comparisons.
    """
    y_true = np.tile(np.arange(5.0), (8, 1))
    y_pred = y_true + 0.5
    first = metrics.prediction_error(y_true, y_true, KPI_NAMES)
    second = metrics.prediction_error(y_true, y_pred, KPI_NAMES)
    assert first[KPI_NAMES[0]].keys() == second[KPI_NAMES[0]].keys()


def test_primary_metric_names_a_real_kpi() -> None:
    """The headline error must refer to a KPI that exists.

    The template shipped a ``<primary-metric>`` placeholder here; a metric name
    that matches nothing produces a comparison table sorted by a missing column.
    """
    assert any(metrics.PRIMARY_METRIC.startswith(name) for name in KPI_NAMES)


def test_perfect_prediction_has_zero_error() -> None:
    """Cheap sanity check that catches sign errors and swapped arguments."""
    y = np.tile(np.arange(5.0), (8, 1))
    report = metrics.prediction_error(y, y, KPI_NAMES)
    assert all(report[name]["mae"] == pytest.approx(0.0) for name in KPI_NAMES)


def test_bias_is_signed_and_separate_from_mae() -> None:
    """A consistent offset is a different problem from noise of the same size.

    A surrogate that is optimistic by two percentage points everywhere still
    ranks configurations correctly and can be corrected. One that is noisy by
    two cannot. Collapsing both into MAE makes them indistinguishable.
    """
    y_true = np.tile(np.arange(5.0), (8, 1))
    report = metrics.prediction_error(y_true, y_true + 2.0, KPI_NAMES)
    assert report[KPI_NAMES[0]]["bias"] == pytest.approx(2.0)
    assert report[KPI_NAMES[0]]["mae"] == pytest.approx(2.0)


def test_constant_bias_preserves_rank_correlation() -> None:
    """Ranking survives a constant offset, and ranking is what the optimizer uses.

    A biased surrogate with the right ordering optimises perfectly; an unbiased
    one that shuffles the ordering does not. Reporting only MAE would rate them
    the same way.
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(32, 5))
    report = metrics.prediction_error(y_true, y_true + 3.0, KPI_NAMES)
    assert report[KPI_NAMES[0]]["rank_correlation"] == pytest.approx(1.0)


def test_mismatched_shapes_raise() -> None:
    """Misaligned predictions must fail loudly, not broadcast."""
    with pytest.raises(ValueError):
        metrics.prediction_error(np.zeros((8, 5)), np.zeros((8, 4)), KPI_NAMES)


def test_near_optimum_error_ranks_by_true_objective() -> None:
    """The slice is chosen by the true objective, not the predicted one.

    Ranking by prediction selects the configurations the surrogate is most
    optimistic about, which biases the error downward in precisely the region
    being examined — the region the optimizer actually searches.
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(64, 5))
    y_pred = y_true + rng.normal(scale=0.1, size=(64, 5))
    objective = y_true[:, 0]
    report = metrics.error_near_optimum(y_true, y_pred, objective, KPI_NAMES, quantile=0.25)
    assert set(report) == set(KPI_NAMES)
