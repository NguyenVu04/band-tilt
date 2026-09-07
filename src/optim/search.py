"""The three ways of searching the tilt space.

``mobo`` and ``random`` are the same loop and differ only in which generation
strategy Ax is given, which is what makes random search a control rather than a
separate experiment: every other line the two execute is identical. ``rule``
is the operator heuristic — one tilt per band, swept by coordinate descent —
and shares only the evaluator and the winner rule.

Nothing here imports :class:`~src.optim.evaluator.Evaluator`. The loop is
written against the :class:`~src.optim.evaluator.ObjectiveEvaluator` protocol,
so it runs against the ray tracer, a stub, or a future surrogate without
changing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.objective import KPI_NAMES, MAXIMISED, ax_objective, lexicographic_best

# Phases, recorded per evaluation so a plot can separate the exploration budget
# from the model-driven one.
_INCUMBENT = "incumbent"
_INIT = "init"
_SEARCH = "search"
_SWEEP = "sweep"

# How Ax's own generation nodes map onto those phases. Anything not listed
# is model-driven: Sobol is the only generator that is not.
_PHASE_OF_NODE = {"attached": _INCUMBENT, "Sobol": _INIT}


def ax_search(
    evaluator: ObjectiveEvaluator,
    cfg: DictConfig,
    *,
    model_driven: bool,
) -> History:
    """Ask-tell loop over Ax, multi-objective on all five KPIs.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.bo.budget`` and ``cfg.bo.seed``.
        model_driven: True for Bayesian optimization, False for Sobol random
            search. The only difference between the two runs.

    Returns:
        The history, whose first row is always the committed incumbent.
    """
    from ax import Client, RangeParameterConfig
    from ax.core.objective import Objective
    from ax.core.optimization_config import MultiObjectiveOptimizationConfig
    from ax.core.outcome_constraint import OutcomeConstraint

    space = evaluator.space
    budget = cfg.bo.budget
    n_init = int(budget.n_init)
    n_total = n_init + int(budget.n_iter)
    batch_size = max(1, int(budget.batch_size))
    seed = int(cfg.bo.seed)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=_INCUMBENT)

    client = Client(random_seed=seed)
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name=name, bounds=(float(low), float(high)), parameter_type="float"
            )
            for name, low, high in zip(space.parameter_names, space.lower, space.upper, strict=True)
        ],
        name=f"band-tilt-{'mobo' if model_driven else 'random'}",
    )
    client.configure_optimization(objective=ax_objective())
    client.configure_generation_strategy(
        method="fast" if model_driven else "random_search",
        initialization_budget=n_init,
        initialization_random_seed=seed,
        # The centre of the box is not this project's incumbent, and spending
        # the first trial there would evaluate a configuration nobody chose.
        initialize_with_center=False,
    )

    # Explicit thresholds, anchored on the incumbent: hypervolume is then
    # credited only for beating what is deployed today. Left unset, Ax infers
    # them from whatever has been observed and warns that it would rather not.
    # Ax's expression parser needs a metric name to resolve to a signature. The
    # KPIs are plain scalars reported under their own names, so that map is the
    # identity.
    signatures = {name: name for name in KPI_NAMES}
    client.set_optimization_config(
        MultiObjectiveOptimizationConfig(
            objective=Objective(expression=ax_objective(), metric_name_to_signature=signatures),
            objective_thresholds=[
                OutcomeConstraint(
                    expression=f"{name} {'>=' if name in MAXIMISED else '<='} {value!r}",
                    relative=False,
                    metric_name_to_signature=signatures,
                )
                for name, value in incumbent.kpi.as_dict().items()
            ],
        )
    )

    baseline_index = client.attach_trial(_parameterization(space.parameter_names, space.baseline))
    client.complete_trial(trial_index=baseline_index, raw_data=incumbent.kpi.as_dict())
    trial_indices = [baseline_index]

    evaluated = 0
    while evaluated < n_total:
        trials = client.get_next_trials(max_trials=min(batch_size, n_total - evaluated))
        if not trials:
            # Ax can legitimately return fewer trials than asked for, but never
            # none: an empty batch means the strategy cannot continue, and
            # looping on it would spin forever.
            print(f"WARNING Ax generated no trial at {evaluated}/{n_total}; stopping early")
            break
        for trial_index, parameters in trials.items():
            tilt = space.clip(np.array([float(parameters[name]) for name in space.parameter_names]))
            result = evaluator.evaluate(tilt)
            client.complete_trial(trial_index=trial_index, raw_data=result.kpi.as_dict())
            history.append(result, phase=_INIT if evaluated < n_init else _SEARCH)
            trial_indices.append(trial_index)
            evaluated += 1

    # Ax counts the attached incumbent toward its initialization budget, so it
    # can leave Sobol a trial earlier than the loop's own counter expects. Its
    # record is the authoritative one, so the phase is relabelled from it and
    # the two columns cannot tell different stories.
    nodes = _generation_nodes(client, trial_indices)
    history.set_generation_nodes(nodes)
    history.set_phases([_PHASE_OF_NODE.get(node, _SEARCH) for node in nodes])
    return history


def rule_based(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Sweep one shared tilt per band by coordinate descent.

    The operator heuristic: every cell on a band points alike, which collapses
    the space from one dimension per cell-band pair to one per band. Passes over
    the bands in turn, trying ``n_steps`` values on each and keeping the best
    before moving on, so the cost is ``n_band * n_steps * n_rounds`` rather than
    the full grid's ``n_steps ** n_band``.

    Candidates are compared with the same lexicographic rule that picks the
    final winner, so this baseline and the Bayesian runs agree on what "better"
    means and differ only in where they look.
    """
    space = evaluator.space
    n_steps = int(cfg.bo.rule.n_steps)
    n_rounds = int(cfg.bo.rule.n_rounds)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=_INCUMBENT)

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
                history.append(candidates[-1][1], phase=_SWEEP)

            if not candidates:
                continue
            # Index 0 is the incumbent for this axis, so a sweep that improves
            # on nothing leaves the band where it was.
            choice = lexicographic_best([best.kpi] + [result.kpi for _p, result in candidates], cfg)
            if choice > 0:
                current, best = candidates[choice - 1]

    return history


SEARCHES: Mapping[str, Callable[[ObjectiveEvaluator, DictConfig], History]] = {
    "mobo": partial(ax_search, model_driven=True),
    "random": partial(ax_search, model_driven=False),
    "rule": rule_based,
}


def run_search(evaluator: ObjectiveEvaluator, cfg: DictConfig, method: str) -> History:
    """Dispatch to one named method.

    Raises:
        KeyError: When the method is not registered, naming those that are.
    """
    if method not in SEARCHES:
        raise KeyError(f"unknown bo.method {method!r}; registered methods are {sorted(SEARCHES)}")
    return SEARCHES[method](evaluator, cfg)


def _parameterization(names: tuple[str, ...], tilt_deg: np.ndarray) -> dict[str, float]:
    """One tilt vector as the name-keyed mapping Ax passes around."""
    return {name: float(value) for name, value in zip(names, tilt_deg, strict=True)}


def _generation_nodes(client: object, trial_indices: list[int]) -> list[str]:
    """Which generator produced each recorded trial.

    Read from Ax's own summary rather than tracked during the loop: the
    strategy decides when to move off Sobol, so its record of that is the
    honest one. Returns empty strings when the summary cannot supply them,
    since this is provenance for a plot and not something to fail a run over.
    """
    try:
        summary = client.summarize()
        node_of = dict(zip(summary["trial_index"], summary["generation_node"], strict=False))
    except Exception:  # pragma: no cover - provenance is optional
        return ["" for _ in trial_indices]

    nodes = []
    for index in trial_indices:
        node = node_of.get(index)
        # The incumbent was attached rather than generated, so it has no node.
        nodes.append("attached" if node is None or pd.isna(node) else str(node))
    return nodes
