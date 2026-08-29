"""The MARL reward.

The scalar reward is::

    r = -lambda_H*K_H - lambda_O*K_O - lambda_W*K_W - lambda_ON*K_ON
        + lambda_BPS*K_BPS

over NORMALISED KPIs, with the weights constrained by::

    lambda_H > lambda_O > lambda_W

The failure this module exists to prevent
-----------------------------------------
The KPIs are on wildly different scales: hole rate is a percentage, mean overlap
neighbours is a small count, the Band Priority Score is on whatever scale the
band weights use. Applying weights to raw values means the effective priority is
set by the scales rather than by the weights — so a config that says
``lambda_H > lambda_O`` can produce an agent that optimises overlap first. Every
individual KPI still looks correct, and the reward still looks like the formula
in the spec.

So normalisation is not optional here, and the weight ordering is asserted
rather than assumed.

A weighted sum is a lossy encoding of the priority
--------------------------------------------------
``cfg.kpi.order`` states a strict order; a weighted sum will always trade
some hole rate for enough of everything else. A hierarchical or lexicographic
reward is allowed instead. If the trained policy
turns out to accept coverage holes in exchange for band-priority gains, that is
this trade-off appearing, not a bug — reach for
:func:`src.kpi.vector.lexicographic_better` rather than re-tuning the weights.

Shaping
-------
Absolute-level rewards are dominated by the constant part of the network
quality, which carries no gradient. Rewarding the improvement over the previous
step concentrates the signal on what the action changed. Use a potential-based
form so the optimal policy is unchanged.
"""

import numpy as np
from omegaconf import DictConfig


def assert_weight_ordering(cfg: DictConfig) -> None:
    """Assert the reward weights respect the coverage priority.

    Args:
        cfg: Composed config; uses the active reward weights.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when the ordering
            ``lambda_H > lambda_O > lambda_W`` is violated.

    Notes:
        Call this at construction, not per step. It is a config error, and
        discovering it after a training run has burned hours is the expensive
        way to find out.

        By default the weights come from ``configs/kpi.yaml`` so the MARL reward
        and the BO scalarization cannot drift apart. If they are overridden in
        ``configs/optim/marl.yaml`` for a shaping experiment, this check still
        applies and the override must be stated when reporting the result.

    Example:
        >>> assert_weight_ordering(cfg)
    """
    # TODO(1): read the active weights, following the interpolation to cfg.kpi
    # TODO(2): raise ValueError naming the offending pair, with both values
    raise NotImplementedError("src.optim.marl.reward.assert_weight_ordering")


def compute(kpis: dict, cfg: DictConfig, previous: dict | None = None) -> float:
    """Compute the scalar reward for one transition.

    Args:
        kpis: The KPI mapping for the configuration just applied.
        cfg: Composed config; uses the reward weights, normalisation and
            shaping settings.
        previous: The KPI mapping from the previous step, required when
            potential-based shaping is enabled.

    Returns:
        The scalar reward. Larger is better.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when shaping is enabled and ``previous``
            is missing.

    Notes:
        Normalise through :func:`src.kpi.vector.normalize` and nowhere else.
        That function already orients every KPI so larger is better, which is
        what stops the Band Priority Score needing a sign flip here — and a
        second, local sign flip is how the one maximised KPI ends up minimised.

    Example:
        >>> r = compute(kpis, cfg, previous=prev_kpis)
    """
    # TODO(1): normalize(kpis, cfg) — uniformly oriented, larger is better
    # TODO(2): weighted sum over the normalised values
    # TODO(3): when shaping, subtract the same quantity computed from previous
    # TODO(4): raise ValueError when shaping is on and previous is None
    raise NotImplementedError("src.optim.marl.reward.compute")


def per_agent(kpis: dict, agent_masks: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Split the global reward into per-agent signals.

    Args:
        kpis: The KPI mapping for the current configuration.
        agent_masks: Boolean mask per agent over the evaluation grid, marking
            the area that agent serves.
        cfg: Composed config.

    Returns:
        One reward per agent.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Optional. With a centralised critic the shared global reward is already
        learnable, and a per-agent split is an approximation that can mislead:
        an agent that improves its own area by pushing interference into a
        neighbour looks rewarded under a local signal and is not.

        If used, keep the global reward as the reported objective and treat the
        split purely as a variance-reduction device.

    Example:
        >>> rewards = per_agent(kpis, masks, cfg)
    """
    # TODO(1): recompute the KPIs restricted to each agent mask
    # TODO(2): apply compute() to each masked KPI mapping
    # TODO(3): document that the global reward stays the reported objective
    raise NotImplementedError("src.optim.marl.reward.per_agent")
