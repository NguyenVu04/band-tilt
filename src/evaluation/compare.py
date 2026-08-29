"""Compare baseline, BO and MARL on identical terms.

Four dimensions, and a comparison that reports only the first is incomplete:

*Solution quality* (25.1) — the five KPIs, from Sionna-RT.
*Computational efficiency* (25.2) — Sionna-RT evaluations, wall-clock, and the
MARL training cost, which BO does not pay.
*Stability* (25.3) — mean, standard deviation, best and worst across seeds.
*Scalability* (25.4) — how each behaves as cells, bands and dimensions grow.

What "identical terms" requires
-------------------------------
Same input data, same scene, same grid, same tilt bounds, same KPI definitions,
same surrogate, same validation procedure. Most of that is already guaranteed
structurally — both optimizers read :mod:`src.optim.space` and score through
:mod:`src.kpi` — but the comparison table is where a violation would surface, so
this module checks rather than assumes.

Report the cost honestly
------------------------
MARL trains once and then proposes configurations cheaply; BO pays per problem
instance. Whether the training cost amortises depends entirely on whether the
policy transfers to a network it was not trained on — which is a claim that has
to be demonstrated, not assumed. Report the training cost inside the method
cost, and state which case was shown.

Seeds
-----
A single-seed result for a stochastic method is an anecdote. MARL varies more
across seeds than BO does, and reporting one run of each flatters whichever
happened to land well.
"""

import pandas as pd
from omegaconf import DictConfig


def kpi_table(results: dict, cfg: DictConfig) -> pd.DataFrame:
    """Build the baseline vs. BO vs. MARL KPI table.

    Args:
        results: Validation output per method, from
            :func:`src.evaluation.validate.validate_many`.
        cfg: Composed config; uses ``cfg.kpi.direction``.

    Returns:
        One row per KPI, one column per method, with the improvement over
        baseline.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when the baseline is missing.

    Notes:
        Show the KPIs in the priority order from ``cfg.kpi.order``, not
        alphabetically. The order is the objective, and a table that hides it
        invites the reader to weigh weak rate equally with hole rate.

        Mark the improvement direction per row. Four KPIs improve by decreasing
        and one by increasing, so an unlabelled delta column is read wrongly
        exactly once per reader.

    Example:
        >>> kpi_table(results, cfg)
    """
    # TODO(1): raise ValueError when results has no baseline
    # TODO(2): one row per KPI in cfg.kpi.order, one column per method
    # TODO(3): delta vs. baseline, signed by cfg.kpi.direction so positive is better
    raise NotImplementedError("src.evaluation.compare.kpi_table")


def cost_table(results: dict) -> pd.DataFrame:
    """Build the computational cost comparison.

    Args:
        results: Per-method run records carrying evaluation counts and timings.

    Returns:
        One row per method: Sionna-RT evaluations, surrogate evaluations,
        wall-clock, and training cost where it applies.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Separate Sionna-RT evaluations from surrogate evaluations. They differ
        by orders of magnitude in cost, and a single combined count makes the
        cheaper method look expensive.

        MARL training cost belongs in this table with an explicit note on
        whether the policy was shown to transfer. Excluding it because it is
        one-off is an argument, not a measurement.

    Example:
        >>> cost_table(results)
    """
    # TODO(1): one row per method
    # TODO(2): columns for sionna evals, surrogate evals, wall-clock, training cost
    # TODO(3): annotate whether policy transfer was demonstrated
    raise NotImplementedError("src.evaluation.compare.cost_table")


def stability_table(runs: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Summarise across seeds.

    Args:
        runs: One row per (method, seed) with its validated KPIs.
        cfg: Composed config; uses ``cfg.kpi.order``.

    Returns:
        Mean, standard deviation, best and worst per method and KPI.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Report best AND worst, not just mean and standard deviation. A method
        whose worst seed opens a coverage hole is not usable regardless of its
        average, because a deployment gets one run, not the mean of many.

        State the number of seeds in the output. Summary statistics over three
        runs and over thirty read identically and mean very different things.

    Example:
        >>> stability_table(runs, cfg)
    """
    # TODO(1): group by method and KPI
    # TODO(2): mean, std, min, max, and the seed count
    # TODO(3): orient best/worst by cfg.kpi.direction, not by numeric magnitude
    raise NotImplementedError("src.evaluation.compare.stability_table")
