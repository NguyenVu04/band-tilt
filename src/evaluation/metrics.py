"""Surrogate prediction error — how well f_sur approximates Sionna-RT.

Not to be confused with :mod:`src.kpi`. That package defines the five KPIs, the
objective the network is optimised against. This module measures how accurately
the surrogate predicts them, which is a question about the model rather than
about the network.

Per KPI, never averaged
-----------------------
A surrogate that is excellent on weak rate and useless on hole rate is useless,
because hole rate is the highest-priority objective (PROJECT.md section 5). A
mean over the five KPIs hides exactly the failure that matters, and it is
reported in units that mean nothing.

In the units of each KPI
------------------------
Percentage points for the three rates, neighbours for overlap severity, score
units for the Band Priority Score. The acceptance test in PROJECT.md
section 19 Step 6 is a comparison against the improvement a result claims, and a
dimensionless normalised error cannot be compared against anything.

Ranking matters more than absolute accuracy
-------------------------------------------
Both optimizers use the surrogate to choose between configurations, so a model
with a constant bias but the right ordering optimises perfectly, and an unbiased
one that shuffles the ranking does not. Report rank correlation alongside the
errors, and weight it more heavily when deciding whether the surrogate is good
enough.
"""

import numpy as np

#: The error reported first when a single number is needed. Hole rate is the
#: highest-priority objective, so it is the one whose surrogate error most
#: directly limits what the optimization can claim.
PRIMARY_METRIC = "hole_rate_mae"


def prediction_error(y_true: np.ndarray, y_pred: np.ndarray, kpi_names: tuple) -> dict:
    """Per-KPI surrogate prediction error.

    Args:
        y_true: Ground-truth KPI vectors from Sionna-RT, shape ``(n, 5)``.
        y_pred: Surrogate predictions, same shape.
        kpi_names: KPI names in column order, normally
            :data:`src.kpi.vector.KPI_NAMES`.

    Returns:
        A mapping from KPI name to its MAE, RMSE, bias and rank correlation.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, on a shape mismatch.

    Notes:
        Report bias separately from MAE. A surrogate that is consistently
        optimistic by two percentage points is a different problem from one that
        is noisy by two: the first still ranks configurations correctly and can
        be corrected, the second cannot.

    Example:
        >>> prediction_error(y_true, y_pred, KPI_NAMES)["hole_rate"]["mae"]
    """
    # TODO(1): raise ValueError on a shape mismatch, naming both shapes
    # TODO(2): MAE, RMSE and signed bias per column
    # TODO(3): Spearman rank correlation per column
    raise NotImplementedError("src.evaluation.metrics.prediction_error")


def error_near_optimum(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    objective: np.ndarray,
    kpi_names: tuple,
    quantile: float = 0.1,
) -> dict:
    """Prediction error restricted to the best-performing configurations.

    Args:
        y_true: Ground-truth KPI vectors, shape ``(n, 5)``.
        y_pred: Surrogate predictions, same shape.
        objective: Scalar objective per configuration, used to rank them.
        kpi_names: KPI names in column order.
        quantile: Fraction of best configurations to keep.

    Returns:
        The same structure as :func:`prediction_error`, over the top slice only.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is the number that matters. An optimizer spends its time in the
        best region of the space, so error averaged over the whole dataset is
        dominated by configurations it would never propose — and a surrogate can
        look accurate globally while being useless exactly where it is used.

        Rank by the TRUE objective, not the predicted one. Ranking by prediction
        selects the configurations the surrogate is most optimistic about, which
        biases the error downward in precisely the region being examined.

    Example:
        >>> error_near_optimum(y_true, y_pred, objective, KPI_NAMES)
    """
    # TODO(1): rank by the true objective and keep the best `quantile`
    # TODO(2): delegate to prediction_error on that slice
    raise NotImplementedError("src.evaluation.metrics.error_near_optimum")
