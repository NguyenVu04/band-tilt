"""Turn a finished search into the choice an operator makes.

:mod:`src.optim.run` measures every candidate with Sionna-RT and then calls
these to publish. Nothing here re-solves anything: the run already holds the
measurements, so this selects from them, shapes the two tables an operator
reads, and prints the result.

The front, not a winner. Four objectives do not have a best, so the run
recommends the priority-order pick but publishes the whole non-dominated set
beside it and lets the trade-off be chosen rather than assumed.

This lives in ``src/optim/`` and not ``src/evaluation/`` on purpose:
:mod:`src.evaluation` states that it re-solves nothing and imports neither
Sionna-RT nor :mod:`src.optim.evaluator`, and that boundary is what lets a
comparison run on a machine with no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.history import (
    History,
    LocalRunWriter,
    write_pareto_options,
    write_run,
    write_tilt_change,
)
from src.optim.objective import KPI_NAMES, KpiVector, as_maximised, pareto_mask


def crowding_distance(values: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance over ``[n, n_objective]``, all maximised.

    Ranking a front by the KPI priority order would return solutions from one
    corner of it, which is not a choice. Crowding distance (Deb et al., 2002)
    scores a solution by how much objective space separates its neighbours, so
    taking the top of it gives the extremes -- which score infinite -- and a
    spread across the middle.

    A front of one or two is entirely extremes, and every entry scores infinite.
    """
    count = len(values)
    if count <= 2:
        return np.full(count, np.inf)

    distance = np.zeros(count)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective])
        column = values[order, objective]
        distance[order[0]] = distance[order[-1]] = np.inf
        span = column[-1] - column[0]
        if span > 0:
            distance[order[1:-1]] += (column[2:] - column[:-2]) / span
    return distance


def choose(kpis: list[KpiVector], n_solutions: int, keep: Sequence[int] = (0,)) -> list[int]:
    """Which rows to publish, best first, as indices into ``kpis``.

    ``keep`` rows come first whatever their crowding distance, then the rest of
    the front by crowding distance up to the budget, so what is offered is a
    spread of trade-offs rather than several versions of one.

    Args:
        kpis: Every evaluation's measured KPI vector.
        n_solutions: How many to offer. Raised to fit ``keep``, which is a
            floor and not a preference.
        keep: Rows that must be in the shortlist. The caller passes the
            incumbent, which every published delta is measured against, and the
            winner, so the run cannot recommend a solution it did not offer.
    """
    front = np.flatnonzero(pareto_mask(kpis))
    ranked = front[np.argsort(-crowding_distance(as_maximised([kpis[i] for i in front])))]

    # dict.fromkeys keeps this order while dropping the repeats it can make: the
    # incumbent is often on the front, and the winner always is.
    required = dict.fromkeys(int(index) for index in keep)
    picks = dict.fromkeys([*required, *(int(index) for index in ranked)])
    return list(picks)[: max(n_solutions, len(required))]


def solutions(frame: pd.DataFrame, picks: list[int], best_index: int) -> pd.DataFrame:
    """One row per published solution, in the order :func:`choose` returned.

    A view of the run's own measurements rather than a new table: the searched
    rows and the published ones are the same rows, so this reindexes the
    history frame and labels it.

    ``on_pareto`` therefore carries the front over the whole run, not over the
    handful offered -- which is the honest reading of whether a solution is
    dominated.
    """
    table = frame.loc[picks].reset_index(drop=True)
    table.insert(0, "solution", range(len(picks)))
    table.insert(1, "is_incumbent", [pick == 0 for pick in picks])
    table.insert(2, "recommended", [pick == best_index for pick in picks])
    return table


def choice_table(published: pd.DataFrame, incumbent: KpiVector) -> pd.DataFrame:
    """The front an operator reads: each solution's KPIs and what it moves.

    Only the non-dominated rows, plus the incumbent it is all measured against
    and the row the run recommends. A dominated solution is worse on every count
    than one already in the table, so offering it is offering a mistake.

    The recommended row is kept even when it is dominated. It can be: the
    priority order compares KPIs within ``kpi.tolerance``, so it will pick a
    solution that ties on every KPI it reaches while losing by a hair on one it
    never gets to. Dropping it would publish a front with nothing marked
    ``recommended``.
    """
    keep = published["on_pareto"] | published["is_incumbent"] | published["recommended"]
    table = published.loc[keep, ["solution", "is_incumbent", "recommended", *KPI_NAMES]].copy()
    for name in KPI_NAMES:
        table[f"delta_{name}"] = table[name] - getattr(incumbent, name)
    return table.reset_index(drop=True)


def tilt_options(published: pd.DataFrame, history: History, picks: list[int]) -> pd.DataFrame:
    """Long form: what each offered solution becomes on the antennas.

    The same rows :func:`choice_table` keeps, so the two deliverables cannot
    disagree about what is on offer.
    """
    frames = []
    for position, (_, row) in enumerate(published.iterrows()):
        if not (row["on_pareto"] or row["is_incumbent"] or row["recommended"]):
            continue
        table = history.tilt_table(history.results[picks[position]].tilt_deg)
        table.insert(0, "recommended", bool(row["recommended"]))
        table.insert(0, "solution", int(row["solution"]))
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def publish(
    history: History,
    cfg: DictConfig,
    directory: Path,
    method: str,
    extra: dict[str, object] | None = None,
) -> dict[str, str]:
    """Select from a finished search, write every artifact, and print the front.

    Everything a run leaves behind once the measuring is done, so the entry
    point and a notebook holding the same history publish identically rather
    than each assembling the sequence themselves.

    Returns the locators :func:`src.optim.history.write_run` wrote, keyed by
    name.
    """
    best_index = history.best_index(cfg)
    best = history.results[best_index]
    picks = choose(history.kpis, int(cfg.optim.n_solutions), keep=(0, best_index))
    published = solutions(history.frame(), picks, best_index)

    writer = LocalRunWriter(directory)
    writer.write_frame("pareto_verified", published)
    tilt_change = write_tilt_change(history.tilt_table(best.tilt_deg), cfg, method)
    scores, options = write_pareto_options(
        choice_table(published, history.results[0].kpi),
        tilt_options(published, history, picks),
        cfg,
        method,
    )

    written = write_run(
        history,
        writer,
        cfg,
        method=method,
        best_index=best_index,
        extra={
            "n_solutions_offered": len(picks),
            "tilt_change": str(tilt_change),
            "pareto_scores": str(scores),
            "tilt_options": str(options),
            **(extra or {}),
        },
    )
    print_summary(published, method, directory, written, scores, options)
    return written


def print_summary(
    published: pd.DataFrame,
    method: str,
    directory: Path,
    written: dict[str, str],
    scores: Path,
    options: Path,
) -> None:
    """Print the measured front and where the run wrote it."""
    offered = published["on_pareto"].sum()

    print(f"\n{method}: {len(published)} solutions published, {offered} non-dominated\n")

    columns = ["solution", "recommended", "on_pareto", *KPI_NAMES]
    print(published[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nwrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")
    print(f"\nchoose from: {scores}")
    print(f"tilt tables: {options}")
