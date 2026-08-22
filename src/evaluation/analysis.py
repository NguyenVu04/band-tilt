"""Error analysis, calibration, feature importance and comparison plots.

:mod:`src.evaluation.metrics` answers "how good is it"; this module answers
"where and why is it wrong". Used by ``02x`` notebooks on the validation split
and by notebook 03 on the test split.

Every function returns a figure or a frame rather than drawing to a global
figure, so the same call works in a notebook and in a report script.

Plotting requires the ``viz`` extra: ``uv sync --extra viz``.
"""

from typing import Any

import numpy as np
import pandas as pd


def error_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: pd.DataFrame | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Return the worst-predicted records for inspection.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predictions.
        features: Optional feature frame, joined so failures can be read in
            context.
        top_n: How many records to return.

    Returns:
        The ``top_n`` records with the largest error, worst first.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Read the actual rows. Systematic failure modes — a sensor range, a
        rare category, a time window — show up here long before they show up
        in an aggregate metric.
    """
    # TODO(1): compute the per-record error appropriate to the task
    # TODO(2): sort, take top_n, join the features if provided
    raise NotImplementedError("src.evaluation.analysis.error_table")


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> Any:
    """Plot residuals against predicted values.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predictions.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Structure in the residuals — a fan shape, a curve, a plateau at the
        range limits — means the model is missing something the features could
        express. Regression tasks; use :func:`calibration_plot` for classifiers.
    """
    # TODO(1): scatter residuals vs predictions with a zero reference line
    raise NotImplementedError("src.evaluation.analysis.residual_plot")


def calibration_plot(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Any:
    """Plot predicted probability against observed frequency.

    Args:
        y_true: Binary ground truth.
        y_proba: Predicted positive-class probability.
        n_bins: Number of probability bins.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        A model can rank well (good ROC-AUC) and still be badly calibrated. If
        downstream decisions use the probability as a probability, calibration
        matters more than ranking.
    """
    # TODO(1): bin by predicted probability, compute observed frequency per bin
    # TODO(2): plot against the diagonal
    raise NotImplementedError("src.evaluation.analysis.calibration_plot")


def feature_importance(model: Any, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
    """Rank features by their contribution to the model.

    Args:
        model: A fitted model from ``src.models``.
        X: Features to compute importance over.
        y: Labels, required for permutation importance.

    Returns:
        Feature name and importance, most important first.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Prefer permutation importance over built-in attributes: it is defined
        the same way for every model type, which keeps this comparable across
        models the way the shared metrics are. Names must come from the fitted
        preprocessor, not from the raw frame, or encoded columns will be
        mislabelled.
    """
    # TODO(1): compute permutation importance on the given split
    # TODO(2): map back to post-preprocessing feature names
    raise NotImplementedError("src.evaluation.analysis.feature_importance")


def comparison_plot(results: pd.DataFrame, metric: str) -> Any:
    """Plot one metric across all models.

    Args:
        results: Models as rows, metrics as columns — the table built in
            notebook 03.
        metric: Which column to plot.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Show the spread, not just the point estimate: a bar chart of single
        numbers invites over-reading differences that are inside the noise.
    """
    # TODO(1): sort by the metric, plot with error bars where available
    raise NotImplementedError("src.evaluation.analysis.comparison_plot")
