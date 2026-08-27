"""Fit the surrogate and report its error — PROJECT.md section 19 Step 6.

Also runs as a script: ``python -m src.surrogate.train`` (or
``task surrogate:train``), so the DVC stage and notebook 04 cannot diverge.

What the error report has to answer
-----------------------------------
Not "is the surrogate good", but "is it good enough to optimise against". Those
are different questions and only the second one matters here.

*Per KPI, never averaged.* A surrogate that is excellent on weak rate and
useless on hole rate is useless, because hole rate is the highest-priority
objective (PROJECT.md section 17). An average over the five hides exactly the
failure that matters.

*Near the optimum, not just globally.* The optimizer spends its time in the
best-performing region of the tilt space. Error averaged over the whole dataset
is dominated by configurations no optimizer would ever propose, so
``cfg.surrogate.acceptance.report_near_optimum`` asks for the error restricted
to the best decile as well.

*Against the KPI differences being claimed.* If predicted hole rate is off by
more than the improvement a result claims, the result is noise. That comparison
is the actual acceptance test, and it is why the thresholds live in
``cfg.surrogate.acceptance`` rather than in a rule of thumb.

Ranking matters more than absolute accuracy
-------------------------------------------
Both optimizers use the surrogate to choose between configurations. A model with
a constant bias but the right ordering optimises perfectly; an unbiased one that
shuffles the ranking does not. Report a rank correlation alongside the errors.
"""

import hydra
from omegaconf import DictConfig


def evaluate_surrogate(model: object, x: object, y: object, cfg: DictConfig) -> dict:
    """Report per-KPI prediction error for a fitted surrogate.

    Args:
        model: A fitted surrogate.
        x: Feature tensor for the partition being scored.
        y: True KPI vectors for that partition.
        cfg: Composed config; uses ``cfg.surrogate.acceptance``.

    Returns:
        A mapping from KPI name to its error metrics, plus the near-optimum
        breakdown and a rank correlation.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Report the errors in the units of each KPI — percentage points for the
        three rates, neighbours for overlap severity, score units for the Band
        Priority Score. A dimensionless normalised error cannot be compared
        against the improvement a result claims, which is the whole purpose of
        the report.

    Example:
        >>> report = evaluate_surrogate(model, x_test, y_test, cfg)
        >>> report["hole_rate"]["mae"]
    """
    # TODO(1): MAE and RMSE per KPI, in that KPI units
    # TODO(2): restrict to the best decile by true objective and repeat
    # TODO(3): rank correlation between predicted and true ordering
    # TODO(4): flag every KPI exceeding cfg.surrogate.acceptance.max_mae
    raise NotImplementedError("src.surrogate.train.evaluate_surrogate")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Train the surrogate as a script, so the DVC stage matches notebook 04.

    Args:
        cfg: Composed by Hydra from ``configs/``.

    Returns:
        The primary validation error, for Hydra sweepers to optimise.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Fail the run when an acceptance threshold is exceeded rather than
        warning. A surrogate that quietly ships below tolerance produces
        optimization results nobody can defend, and the failure surfaces much
        later and much more expensively.

    Example:
        >>> # task surrogate:train
    """
    # TODO(1): set_seed(cfg.seed)
    # TODO(2): dataset.load + dataset.split
    # TODO(3): features.build_state, features.fit on train only, features.transform
    # TODO(4): instantiate(cfg.surrogate) and fit with the validation partition
    # TODO(5): evaluate_surrogate on test; log to MLflow via src.utils.tracking
    # TODO(6): raise when acceptance thresholds are exceeded
    # TODO(7): save to cfg.surrogate.artifact_path
    raise NotImplementedError("src.surrogate.train.main")


if __name__ == "__main__":
    main()
