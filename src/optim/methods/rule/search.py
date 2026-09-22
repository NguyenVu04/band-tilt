"""The operator heuristic: one shared tilt per band, swept by coordinate descent."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import ATTACHED, COORDINATE, INCUMBENT, SWEEP
from src.optim.objective import best_by_objective
from src.optim.space import TiltSpace


def _band_axes(space: TiltSpace) -> dict[str, tuple[np.ndarray, float, float]]:
    """Each band's dimensions and the range every one of its cells accepts.

    Raises:
        ValueError: When a band's cells share no tilt, so no shared value is feasible.
    """
    axes = {}
    for band in space.band_names:
        axis = np.flatnonzero([name == band for _cell, name in space.pairs])
        # The band's usable range is the intersection over its cells, so a
        # shared value stays feasible for every one of them.
        low = float(space.lower[axis].max())
        high = float(space.upper[axis].min())
        if low > high:
            raise ValueError(
                f"band {band}: its cells' tilt bounds do not overlap ({low} > {high}), "
                "so no shared tilt is feasible."
            )
        axes[band] = (axis, low, high)
    return axes


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Sweep one shared tilt per band by coordinate descent.

    Every cell on a band points alike, which collapses the space from one
    dimension per cell-band pair to one per band. Passes over the bands in turn,
    trying ``n_steps`` values on each and keeping the best before moving on, so
    the cost is ``n_band * n_steps * n_rounds`` rather than the full grid's
    ``n_steps ** n_band``.

    Candidates are compared by the objective that picks the final winner,
    so this baseline and TuRBO agree on what "better" means and differ only in
    where they look.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.n_steps`` and
            ``cfg.optim.method.n_rounds``.

    Returns:
        The history, whose first row is always the committed incumbent.

    Raises:
        ValueError: When ``n_steps`` or ``n_rounds`` is below 1, or as
            :func:`_band_axes`.
    """
    space = evaluator.space
    n_steps = int(cfg.optim.method.n_steps)
    n_rounds = int(cfg.optim.method.n_rounds)
    if n_steps < 1 or n_rounds < 1:
        raise ValueError(f"n_steps ({n_steps}) and n_rounds ({n_rounds}) must be at least 1.")
    axes = _band_axes(space)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=INCUMBENT, generation_node=ATTACHED)

    current = space.baseline.copy()
    best = incumbent
    for _ in range(n_rounds):
        for axis, low, high in axes.values():
            candidates = []
            for value in np.linspace(low, high, n_steps):
                proposal = current.copy()
                proposal[axis] = value
                if np.allclose(proposal, current):
                    continue
                candidates.append((proposal, evaluator.evaluate(proposal)))
                history.append(candidates[-1][1], phase=SWEEP, generation_node=COORDINATE)

            if not candidates:
                continue
            # Index 0 is the incumbent for this axis, so a sweep that improves
            # on nothing leaves the band where it was.
            choice = best_by_objective([best.kpi] + [result.kpi for _p, result in candidates])
            if choice > 0:
                current, best = candidates[choice - 1]

    return history
