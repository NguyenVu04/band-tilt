"""Sobol random search over the tilt space: the control the methods are read against.

Matched to :mod:`src.optim.methods.mobo` on evaluations and on everything the
loop does with them, so a difference between the two runs is a difference in
where they looked and in nothing else. The generation strategy is the only line
that differs, and ADR 0002 is the reason it must stay the only one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import INCUMBENT, INIT, PHASE_OF_NODE, SEARCH
from src.optim.objective import KPI_NAMES, MAXIMISED, ax_objective


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Ask-tell loop over Ax, drawing every trial from Sobol.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.budget`` and
            ``cfg.optim.seed``.

    Returns:
        The history, whose first row is always the committed incumbent.
    """
    from ax import Client, RangeParameterConfig
    from ax.core.objective import Objective
    from ax.core.optimization_config import MultiObjectiveOptimizationConfig
    from ax.core.outcome_constraint import OutcomeConstraint

    space = evaluator.space
    budget = cfg.optim.method.budget
    n_init = int(budget.n_init)
    n_total = n_init + int(budget.n_iter)
    batch_size = max(1, int(budget.batch_size))
    seed = int(cfg.optim.seed)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=INCUMBENT)

    client = Client(random_seed=seed)
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name=name, bounds=(float(low), float(high)), parameter_type="float"
            )
            for name, low, high in zip(space.parameter_names, space.lower, space.upper, strict=True)
        ],
        name="band-tilt-random",
    )
    client.configure_optimization(objective=ax_objective())
    client.configure_generation_strategy(
        method="random_search",
        initialization_budget=n_init,
        initialization_random_seed=seed,
        # The centre of the box is not this project's incumbent, and spending
        # the first trial there would evaluate a configuration nobody chose.
        initialize_with_center=False,
    )

    # Thresholds anchored on the incumbent, set for the same reason as in mobo:
    # the two runs must be scored against one reference to be comparable, even
    # though nothing here consumes hypervolume.
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
            history.append(result, phase=INIT if evaluated < n_init else SEARCH)
            trial_indices.append(trial_index)
            evaluated += 1

    # Relabelled from Ax's own record for the same reason as in mobo: its count
    # of the initialization budget includes the attached incumbent, and its
    # record outranks the loop's counter.
    nodes = _generation_nodes(client, trial_indices)
    history.set_generation_nodes(nodes)
    history.set_phases([PHASE_OF_NODE.get(node, SEARCH) for node in nodes])
    return history


def _parameterization(names: tuple[str, ...], tilt_deg: np.ndarray) -> dict[str, float]:
    """One tilt vector as the name-keyed mapping Ax passes around."""
    return {name: float(value) for name, value in zip(names, tilt_deg, strict=True)}


def _generation_nodes(client: object, trial_indices: list[int]) -> list[str]:
    """Which generator produced each recorded trial.

    Read from Ax's own summary rather than tracked during the loop. Returns
    empty strings when the summary cannot supply them, since this is provenance
    for a plot and not something to fail a run over.
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
