"""The single definition of every metric in this project.

Used identically by ``02x`` validation analysis, by ``src.models.train``, and
by notebook 03's final comparison. That identity is the whole point: it is what
makes the numbers in the results table comparable.

What must NOT go here
---------------------
Model-specific scoring. If a model needs an internal objective of its own, that
is part of its training loop, not part of the shared metric set.

Adding or changing a metric changes every historical comparison, so treat this
file as an interface: extend it, and re-run rather than silently redefine.
"""

import numpy as np

#: The metric used to rank models and to drive early stopping. Named once here
#: so the choice is explicit rather than implied by column order.
PRIMARY_METRIC = "<primary-metric>"


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Score predictions with the project's full metric set.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels or values, aligned with ``y_true``.
        sample_weight: Optional per-record weights.

    Returns:
        Metric name to value. Keys are stable across calls and across models,
        so results can be stacked straight into a comparison frame.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, on length mismatch or empty input.

    Notes:
        Return every metric the project reports, always the same keys, even
        when a value is ``nan`` — a missing key breaks the results table, while
        a ``nan`` is visible and explainable. Include :data:`PRIMARY_METRIC`.

    Example:
        >>> results = {name: evaluate(y_test, preds[name]) for name in models}
        >>> pd.DataFrame(results).T.sort_values(PRIMARY_METRIC)
    """
    # TODO(1): validate shapes and dtypes
    # TODO(2): compute the task's metrics (regression: MAE/RMSE/R2;
    #          classification: accuracy/precision/recall/F1/ROC-AUC)
    # TODO(3): return a flat dict of python floats, PRIMARY_METRIC included
    raise NotImplementedError("src.evaluation.metrics.evaluate")


def evaluate_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Score predictions separately within each group or segment.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predictions.
        groups: Segment label per record — a device, a site, a class band.

    Returns:
        Group label to the metric dict from :func:`evaluate`.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        An aggregate score hides which slice a model fails on. Two models with
        the same headline number can behave very differently per segment, and
        that difference is usually what decides which one ships.
    """
    # TODO(1): group the indices, call evaluate() per group
    # TODO(2): flag groups with too few records to score reliably
    raise NotImplementedError("src.evaluation.metrics.evaluate_by_group")
