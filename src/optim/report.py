"""Turn a finished search into the choice an operator makes.

:mod:`src.optim.run` measures every candidate with Sionna-RT and then calls
these to publish. Nothing here re-solves anything: the run already holds the
measurements, so this selects from them, shapes the two tables an operator
reads, and prints the result.

The shortlist is the highest weighted scores (``kpi.weights``, ADR 0003) beside
the incumbent, so the recommended row is published with the runners-up it beat
rather than alone.

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
    write_run,
    write_solution_options,
    write_tilt_change,
)
from src.optim.objective import KPI_NAMES, KpiVector, scores


def choose(
    kpis: list[KpiVector], cfg: DictConfig, n_solutions: int, keep: Sequence[int] = (0,)
) -> list[int]:
    """Which rows to publish, best first, as indices into ``kpis``.

    ``keep`` rows come first, then the rest by weighted score, highest first, up
    to the budget. A tie keeps the earlier row, as
    :func:`src.optim.objective.best_by_score` does.

    Args:
        kpis: Every evaluation's measured KPI vector.
        cfg: Composed config; reads ``kpi.weights``.
        n_solutions: How many to offer. Raised to fit ``keep``, which is a
            floor and not a preference.
        keep: Rows that must be in the shortlist. The caller passes the
            incumbent, which every published delta is measured against, and the
            winner, so the run cannot recommend a solution it did not offer.

    Raises:
        ValueError: When ``kpi.weights`` is unusable; see
            :func:`src.optim.objective.weights`.
    """
    ranked = np.argsort(-scores(kpis, cfg), kind="stable")

    # dict.fromkeys keeps this order while dropping the repeats it can make: the
    # winner always ranks first, and the incumbent can rank anywhere.
    required = dict.fromkeys(int(index) for index in keep)
    picks = dict.fromkeys([*required, *(int(index) for index in ranked)])
    return list(picks)[: max(n_solutions, len(required))]


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
    """The shortlist an operator reads: each solution's score, KPIs and what it moves."""
    table = published[["solution", "is_incumbent", "recommended", "score", *KPI_NAMES]].copy()
    for name in KPI_NAMES:
        table[f"delta_{name}"] = table[name] - getattr(incumbent, name)
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
    best_index = history.best_index(cfg)
    best = history.results[best_index]
    picks = choose(history.kpis, cfg, int(cfg.optim.n_solutions), keep=(0, best_index))
    published = solutions(history.frame(cfg), picks, best_index)

    writer = LocalRunWriter(directory)
    writer.write_frame("solutions", published)
    tilt_change = write_tilt_change(history.tilt_table(best.tilt_deg), cfg, method)
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
            "tilt_change": str(tilt_change),
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

    columns = ["solution", "recommended", "score", *KPI_NAMES]
    print(published[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nwrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")
    print(f"\nchoose from: {shortlist}")
    print(f"tilt tables: {options}")
