"""Turn a finished search into the choice an operator makes.

:mod:`src.optim.run` measures every candidate with Sionna-RT and then calls
these to publish. Nothing here re-solves anything: the run already holds the
measurements, so this selects from them, shapes the two tables an operator
reads, and prints the result.

The shortlist is the whole Pareto front, ranked by hypervolume contribution,
so the recommended row is published with the trade-offs it was chosen from
rather than alone. The incumbent is listed only when it is on the front; every
delta is still measured against it. Every measure here is the search's own,
over every UE.

This lives in ``src/optim/`` and not ``src/evaluation/`` on purpose:
:mod:`src.evaluation` states that it re-solves nothing and imports neither
Sionna-RT nor :mod:`src.optim.evaluator`, and that boundary is what lets a
comparison run on a machine with no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.history import History, LocalRunWriter, write_run, write_solution_options
from src.optim.objective import (
    MEASURE_NAMES,
    KpiVector,
    hypervolume_contributions,
    objective_matrix,
    pareto_mask,
)


def choose(kpis: list[KpiVector]) -> list[int]:
    """Which rows to publish, as indices into ``kpis``: the Pareto front.

    Largest hypervolume contribution first, so the recommendation
    (:func:`src.optim.objective.best_by_hvc`) leads; a tie keeps the earlier
    row, as that function does. A dominated row is never offered.
    """
    points = objective_matrix(kpis)
    front = np.flatnonzero(pareto_mask(points))
    ranked = front[np.argsort(-hypervolume_contributions(points)[front], kind="stable")]
    return [int(index) for index in ranked]


def solutions(frame: pd.DataFrame, picks: list[int], best_index: int) -> pd.DataFrame:
    """One row per published solution, in the order :func:`choose` returned.

    A view of the run's own measurements rather than a new table: the searched
    rows and the published ones are the same rows, so this reindexes the
    history frame and labels it.
    """
    table = frame.loc[picks].reset_index(drop=True)
    table.insert(0, "solution", range(len(picks)))
    table.insert(1, "is_incumbent", [pick == 0 for pick in picks])
    table.insert(2, "recommended", [pick == best_index for pick in picks])
    return table


def choice_table(published: pd.DataFrame, incumbent: KpiVector) -> pd.DataFrame:
    """The shortlist an operator reads: each solution's KPIs, objectives and what it moves.

    The objectives ranked the shortlist and picked the recommendation. The KPIs
    sit beside them, so the result can be read against the thresholds a
    deployment would apply without the shortlist being ordered by them.
    """
    columns = ["solution", "is_incumbent", "recommended", *MEASURE_NAMES]
    table = published[columns].copy()
    for name in MEASURE_NAMES:
        reference = getattr(incumbent, name)
        values = table[name]
        # The percentile measures are -inf when nothing is covered
        # (src/kpi/quality.py), so a total-outage pair would subtract to NaN and
        # print as a blank cell. Two configurations that both cover nothing have
        # not moved the measure.
        delta = np.where(
            (values == reference) & np.isinf(values), 0.0, values.to_numpy() - reference
        )
        table[f"delta_{name}"] = delta
    return table.reset_index(drop=True)


def tilt_options(published: pd.DataFrame, history: History, picks: list[int]) -> pd.DataFrame:
    """Long form: what each offered solution becomes on the antennas."""
    frames = []
    for pick, (_, row) in zip(picks, published.iterrows(), strict=True):
        table = history.tilt_table(history.results[pick].tilt_deg)
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
    """Select from a finished search, write every artifact, and print the shortlist.

    Everything a run leaves behind once the measuring is done, so the entry
    point and a notebook holding the same history publish identically rather
    than each assembling the sequence themselves.

    Returns the locators :func:`src.optim.history.write_run` wrote, keyed by
    name.
    """
    best_index = history.best_index()
    picks = choose(history.kpis)
    published = solutions(history.frame(), picks, best_index)

    writer = LocalRunWriter(directory)
    writer.write_frame("solutions", published)
    shortlist, options = write_solution_options(
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
            "solutions": str(shortlist),
            "tilt_options": str(options),
            **(extra or {}),
        },
    )
    print_summary(published, method, directory, written, shortlist, options)
    return written


def print_summary(
    published: pd.DataFrame,
    method: str,
    directory: Path,
    written: dict[str, str],
    shortlist: Path,
    options: Path,
) -> None:
    """Print the measured shortlist and where the run wrote it."""
    print(f"\n{method}: {len(published)} solutions published\n")

    columns = ["solution", "recommended", *MEASURE_NAMES]
    print(published[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nwrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")
    print(f"\nchoose from: {shortlist}")
    print(f"tilt tables: {options}")
