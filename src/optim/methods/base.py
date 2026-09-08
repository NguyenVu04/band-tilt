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

from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History

# Recorded per evaluation, so a plot can separate the exploration budget from
# the model-driven one.
INCUMBENT = "incumbent"
INIT = "init"
SEARCH = "search"
SWEEP = "sweep"

# How Ax's own generation nodes map onto those phases. Anything not listed is
# model-driven: Sobol is the only generator that is not.
PHASE_OF_NODE = {"attached": INCUMBENT, "Sobol": INIT}


class SearchMethod(Protocol):
    """One way of searching the tilt space.

    Implementations take the :class:`~src.optim.evaluator.ObjectiveEvaluator`
    protocol rather than the concrete evaluator, which is what lets a method be
    tested without a GPU and lets a future surrogate stand in for the ray tracer
    without the method noticing.
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
