"""The operator heuristic: one shared tilt per band, swept by coordinate descent."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import INCUMBENT, SWEEP
from src.optim.objective import lexicographic_best


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Sweep one shared tilt per band by coordinate descent.

    Every cell on a band points alike, which collapses the space from one
    dimension per cell-band pair to one per band. Passes over the bands in turn,
    trying ``n_steps`` values on each and keeping the best before moving on, so
    the cost is ``n_band * n_steps * n_rounds`` rather than the full grid's
    ``n_steps ** n_band``.

    Candidates are compared with the same lexicographic rule that picks the
    final winner, so this baseline and the Bayesian runs agree on what "better"
    means and differ only in where they look.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.n_steps`` and
            ``cfg.optim.method.n_rounds``.

    Returns:
        The history, whose first row is always the committed incumbent.
    """
    space = evaluator.space
    n_steps = int(cfg.optim.method.n_steps)
    n_rounds = int(cfg.optim.method.n_rounds)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=INCUMBENT)

    current = space.baseline.copy()
    best = incumbent
    for _ in range(n_rounds):
        for band in space.band_names:
            axis = [index for index, (_cell, name) in enumerate(space.pairs) if name == band]
            # The band's usable range is the intersection over its cells, so a
            # shared value stays feasible for every one of them.
            low = float(space.lower[axis].max())
            high = float(space.upper[axis].min())

            candidates = []
            for value in np.linspace(low, high, n_steps):
                proposal = current.copy()
                proposal[axis] = value
                if np.allclose(proposal, current):
                    continue
                candidates.append((proposal, evaluator.evaluate(proposal)))
                history.append(candidates[-1][1], phase=SWEEP)

            if not candidates:
                continue
            # Index 0 is the incumbent for this axis, so a sweep that improves
            # on nothing leaves the band where it was.
            choice = lexicographic_best([best.kpi] + [result.kpi for _p, result in candidates], cfg)
            if choice > 0:
                current, best = candidates[choice - 1]

    return history
