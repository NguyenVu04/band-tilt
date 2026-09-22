"""Sobol random search over the tilt space: the control the methods are read against.

Matched to :mod:`src.optim.methods.turbo` on evaluations and on what is recorded
per evaluation, and drawn from the same seeded Sobol sequence as TuRBO's initial
design, so a difference between the two runs is a difference in where they
looked; ADR 0002.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import ATTACHED, INCUMBENT, INIT, SEARCH, SOBOL, sobol


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Evaluate ``n_init + n_iter`` scrambled Sobol points after the incumbent.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.budget`` and
            ``cfg.optim.seed``.

    Returns:
        The history, whose first row is always the committed incumbent. The
        first ``n_init`` Sobol rows are labelled ``init``, the rest ``search``,
        matching TuRBO's phase boundary.
    """
    space = evaluator.space
    budget = cfg.optim.method.budget
    n_init = int(budget.n_init)
    n_total = n_init + int(budget.n_iter)

    history = History(space)
    history.append(evaluator.evaluate(space.baseline), phase=INCUMBENT, generation_node=ATTACHED)
    span = space.upper - space.lower
    for index, point in enumerate(sobol(space.n_dim, n_total, int(cfg.optim.seed))):
        result = evaluator.evaluate(space.clip(space.lower + point * span))
        history.append(result, phase=INIT if index < n_init else SEARCH, generation_node=SOBOL)
    return history
