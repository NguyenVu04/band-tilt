"""Fit the surrogate and report its error — PROJECT.md section 16 Phase 5.

Also runs as a script: ``python -m src.surrogate.train`` (or
``task surrogate:train``), so the DVC stage and notebook 04 cannot diverge.

What the error report has to answer
-----------------------------------
Not "is the surrogate good", but "is it good enough to optimise against". Those
are different questions and only the second one matters here.

*At two levels.* The model predicts a radio map, so the training loss is
map error in dB (PROJECT.md section 11.3) — but acceptance is decided on the
KPIs that map produces. Report both. A small average map error can still move
hole rate a long way, because hole rate is a hard threshold at -120 dBm and
error concentrated at the coverage edge costs far more than the same error in
the cell centre.

*Per KPI, never averaged.* A surrogate that is excellent on weak rate and
useless on hole rate is useless, because hole rate is the highest-priority
objective (PROJECT.md section 5). An average over the five hides exactly the
failure that matters.

*On held-out scenarios, not just held-out configurations.* Generalising to a new
tilt vector in a scene the model has seen is a much weaker claim than
generalising to a perturbed environment (PROJECT.md section 12), and only the
second supports the sim-to-reality conclusion.

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
shuffles the ranking does not. Report a rank correlation alongside the errors —
computed on the derived KPIs, since that is what the optimizers actually compare.

Freeze on acceptance
--------------------
Once the acceptance thresholds in ``cfg.surrogate.acceptance`` are met the model
is frozen and used unchanged for the whole optimization phase (PROJECT.md
section 11.4). It is not retrained on the candidates TuRBO or MARL propose;
doing so would put a Sionna-RT solve back inside the loop, which is the expense
the surrogate exists to avoid.
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
