"""The interface every search method implements, and the vocabulary they share.

A method is a callable, not a class: it has no state that outlives one run, and
:class:`~src.optim.history.History` already owns everything a run accumulates.

The phase names live here rather than in each method because they are the schema
of ``History.frame()["phase"]`` that ``src/evaluation`` and every plot read. The
methods are otherwise independent of one another; this is the one thing they may
not disagree about.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History

# Recorded per evaluation, so a plot can separate the exploration budget from
# the model-driven one.
INCUMBENT = "incumbent"
INIT = "init"
SEARCH = "search"
SWEEP = "sweep"

# Generator names recorded per evaluation.
ATTACHED = "attached"
SOBOL = "Sobol"


def sobol(dim: int, n: int, seed: int) -> np.ndarray:
    """``n`` scrambled Sobol points in the unit cube, ``[n, dim]``.

    Deterministic in ``seed``, and a shorter draw is a prefix of a longer one,
    so random search and TuRBO's initial design share their first points.
    """
    import torch
    from torch.quasirandom import SobolEngine

    if n <= 0:
        return np.empty((0, dim))
    return SobolEngine(dim, scramble=True, seed=seed).draw(n, dtype=torch.float64).numpy()


class SearchMethod(Protocol):
    """One way of searching the tilt space.

    Implementations take the :class:`~src.optim.evaluator.ObjectiveEvaluator`
    protocol rather than the concrete evaluator, which is what lets a method be
    tested against a stub without a GPU.
    """

    def __call__(self, evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
        """Search, and return the log. Row zero is always the committed incumbent.

        Args:
            evaluator: Scores a tilt vector.
            cfg: Composed config. A method reads ``cfg.optim.method`` — its own
                block, selected by the ``optim/method`` config group — and
                ``cfg.optim.seed``.
        """
        ...
