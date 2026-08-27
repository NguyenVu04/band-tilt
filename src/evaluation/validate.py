"""Re-evaluate an optimized configuration with Sionna-RT — PROJECT.md section 26.

The validation flow::

    theta*  ->  Sionna-RT  ->  ground-truth radio map  ->  five KPIs
                                                       ->  compare with surrogate

This is where a result becomes a claim
--------------------------------------
Everything before this point ran against the surrogate, which is an
approximation trained on a few hundred configurations and used to search a space
of far more. An optimizer given an approximate objective will find the places
where that approximation is most optimistic — that is what optimization does.
Validation is what separates a real improvement from an exploited surrogate
error.

So: the numbers in any report come from this module. A KPI improvement that
exists only in the surrogate is not a result, and the gap between prediction and
ground truth is reported rather than quietly dropped.

Validate against the test split
-------------------------------
UE density used during optimization came from the training MDT split. Recompute
it from the test split here: the Band Priority Score weights locations by where
users were measured, and scoring the optimized network on the same measurements
that shaped the objective is the same leakage the split exists to prevent.
"""

from typing import Any

import numpy as np
from omegaconf import DictConfig


def validate(theta: np.ndarray, cfg: DictConfig, predicted: dict | None = None) -> dict:
    """Evaluate a configuration with Sionna-RT and compare against the prediction.

    Args:
        theta: The configuration to validate, in degrees, in cell-band table
            order.
        cfg: Composed config.
        predicted: The surrogate KPI prediction for this configuration, when
            one exists.

    Returns:
        The ground-truth KPIs, the prediction, and the per-KPI gap between them.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Assert the configuration is within bounds before spending a solve on it.
        A theta that drifted out of the feasible box during optimization
        produces a perfectly good radio map for a network that cannot be
        deployed.

        Report the gap per KPI, in each KPI own units. A gap larger than the
        improvement being claimed means the result is not supported, regardless
        of how good the ground-truth number looks on its own.

    Example:
        >>> result = validate(theta_star, cfg, predicted=bo_prediction)
        >>> result["ground_truth"]["hole_rate"]
    """
    # TODO(1): assert_within_bounds against the cell-band table
    # TODO(2): rho from the TEST split, not the train split used during optimization
    # TODO(3): radiomap.evaluate then kpi.vector.kpi_vector
    # TODO(4): gap = ground truth minus prediction, per KPI
    raise NotImplementedError("src.evaluation.validate.validate")


def validate_many(thetas: dict, cfg: DictConfig) -> Any:
    """Validate several configurations under one scene load.

    Args:
        thetas: A mapping from method name (``"baseline"``, ``"bo"``,
            ``"marl"``) to its configuration.
        cfg: Composed config.

    Returns:
        One row per method, holding its ground-truth KPIs.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Always include the baseline. Every reported improvement is stated
        relative to the current network (PROJECT.md section 27.2), and a
        baseline evaluated in a different run, against a different scene build
        or a different grid, is not a valid reference.

        One scene load for all of them. Reloading per method costs minutes each
        and, worse, risks the configurations being scored against
        subtly different geometry.

    Example:
        >>> results = validate_many({"baseline": t0, "bo": t_bo, "marl": t_marl}, cfg)
    """
    # TODO(1): load the scene and attach transmitters once
    # TODO(2): raise when "baseline" is absent from thetas
    # TODO(3): evaluate each configuration against the identical grid and density
    raise NotImplementedError("src.evaluation.validate.validate_many")
