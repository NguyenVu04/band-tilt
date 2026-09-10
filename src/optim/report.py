"""Phase two: re-solve the searched front with Sionna-RT and publish the choice.

:mod:`src.optim.run` searches with the surrogate and stops. This re-solves the
solutions it proposed, re-derives the Pareto front from what the ray tracer
measured, and republishes the front as the two tables an operator chooses from.

The front, not a winner. Five objectives do not have a best, and with a
surrogate in the loop reporting only the single lexicographic pick would trust
the model twice — once to *find* the front and again to *rank within* it.
Re-deriving the front from measured KPIs asks it for the first alone.

This lives in ``src/optim/`` and not ``src/evaluation/`` on purpose:
:mod:`src.evaluation` states that it re-solves nothing and imports neither
Sionna-RT nor :mod:`src.optim.evaluator`, and that boundary is what lets a
comparison run on a machine with no GPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.evaluator import EvaluationResult, Evaluator
from src.optim.history import (
    History,
    LocalRunWriter,
    write_pareto_options,
    write_run,
    write_tilt_change,
)
from src.optim.methods.base import VERIFY
from src.optim.objective import (
    KPI_NAMES,
    SURROGATE,
    KpiVector,
    as_maximised,
    lexicographic_best,
    pareto_mask,
)
from src.optim.space import TiltSpace


def latest_run(cfg: DictConfig, method: str) -> Path:
    """The newest run directory for ``method``.

    Run ids are UTC timestamps, so lexical order is chronological.

    Raises:
        FileNotFoundError: When the method has no runs to report on.
    """
    root = Path(cfg.optim.output.dir) / method
    found = sorted(path.parent for path in root.glob("*/run.json"))
    if not found:
        raise FileNotFoundError(f"No runs under {root}. Run `task bo` first.")
    return found[-1]


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


def choose(kpis: list[KpiVector], cfg: DictConfig, n_solutions: int) -> list[int]:
    """Which rows to re-solve, best first, as indices into ``kpis``.

    Three things have to be in the shortlist, for three different reasons.

    The **incumbent** (row zero) always, because every reported delta is
    measured against it: a ray-traced winner compared to a predicted incumbent
    would put two measurement systems on either side of one subtraction.

    The **priority-order pick**, so the solution the run ends up recommending is
    certainly among those measured.

    Then the **front by crowding distance**, up to the budget, so what is
    published is a spread of trade-offs rather than several versions of one.
    """
    front = np.flatnonzero(pareto_mask(kpis))
    ranked = front[np.argsort(-crowding_distance(as_maximised([kpis[i] for i in front])))]

    priority = int(front[lexicographic_best([kpis[i] for i in front], cfg)])
    # dict.fromkeys keeps this order while dropping the repeats it can make: the
    # incumbent is often on the front, and the priority pick always is.
    picks = dict.fromkeys([0, priority, *(int(index) for index in ranked)])
    # Never below two, or the budget could drop the pick this run recommends.
    return list(picks)[: max(n_solutions, 2)]


def report(cfg: DictConfig, directory: Path | None = None) -> Path:
    """Re-solve one run's searched front with Sionna-RT and complete its directory.

    Returns the directory reported on.

    Raises:
        FileNotFoundError: When the run has no ``history.parquet`` to read.
    """
    method = str(cfg.optim.method.name)
    configured = str(cfg.optim.report.run_dir)
    directory = directory or (Path(configured) if configured else latest_run(cfg, method))

    history_path = directory / "history.parquet"
    if not history_path.is_file():
        raise FileNotFoundError(f"No {history_path}. Run `task bo` before reporting.")

    space = TiltSpace.from_config(cfg)
    searched = pd.read_parquet(history_path)
    predicted = [
        KpiVector.from_mapping({name: float(row[name]) for name in KPI_NAMES})
        for _, row in searched.iterrows()
    ]
    picks = choose(predicted, cfg, int(cfg.optim.report.n_solutions))
    tilts = searched[list(space.parameter_names)].to_numpy(dtype=float)

    started = time.time()
    # The searched rows are replayed into a History rather than left in the
    # frame, so one history.parquet carries both phases and `source` says which
    # is which. Writing only the verified rows would throw the search away.
    history = _replay(searched, predicted, tilts, space)
    verified: list[int] = []
    # keep_rsrp so the recommendation's map is already in hand: re-solving it
    # afterwards purely to archive it would be a second full-fidelity solve of a
    # vector this loop has already measured.
    with Evaluator(cfg, keep_rsrp=True) as evaluator:
        scenario_id = evaluator.scenario_id
        print(f"re-solving {len(picks)} solutions with Sionna-RT")
        for rank, index in enumerate(picks):
            result = evaluator.evaluate(tilts[index])
            history.append(result, phase=VERIFY, generation_node=f"searched{index}")
            verified.append(len(history) - 1)
            print(f"  {rank + 1}/{len(picks)}  searched row {index}  {result.seconds:.1f}s")

        best_row = verified[lexicographic_best([history.results[i].kpi for i in verified], cfg)]
        best = history.results[best_row]
        radio_map = None
        if bool(cfg.optim.output.save_radio_map):
            radio_map = str(evaluator.write_radio_map(directory / "best_radio_map.npz", best))

    front = pareto_mask([history.results[i].kpi for i in verified])
    solutions = _solutions(history, searched, picks, verified, front, best_row, space)

    writer = LocalRunWriter(directory)
    writer.write_frame("pareto_verified", solutions)
    tilt_change = write_tilt_change(history.tilt_table(best.tilt_deg), cfg, method)
    scores, options = write_pareto_options(
        _choice_table(solutions, history.results[verified[0]].kpi),
        _tilt_options(solutions, history, verified, space),
        cfg,
        method,
    )

    written = write_run(
        history,
        writer,
        cfg,
        method=method,
        best_index=best_row,
        # Row zero of the verified history is the re-solved incumbent, because
        # choose() always puts it first.
        incumbent_index=verified[0],
        extra={
            "scenario_id": scenario_id,
            "searched_evaluations": int(len(searched)),
            "n_solutions_verified": len(verified),
            "n_pareto_verified": int(front.sum()),
            "report_seconds": time.time() - started,
            "best_radio_map": radio_map,
            "tilt_change": str(tilt_change),
            "pareto_scores": str(scores),
            "tilt_options": str(options),
        },
    )
    _print(solutions, method, directory, written, scores, options)
    return directory


def _replay(
    searched: pd.DataFrame,
    predicted: list[KpiVector],
    tilts: np.ndarray,
    space: TiltSpace,
) -> History:
    """Rebuild the search phase's history from the table it wrote.

    The two phases are chained through the run directory rather than through
    memory, so the search can run where there is no GPU and the report only
    where there is one. That means phase two reads its predecessor back rather
    than being handed it.
    """
    history = History(space=space)
    for position, (_, row) in enumerate(searched.iterrows()):
        history.append(
            EvaluationResult(
                tilt_deg=tilts[position],
                kpi=predicted[position],
                seconds=float(row["seconds"]),
                source=str(row.get("source", SURROGATE)),
            ),
            phase=str(row["phase"]),
            generation_node=str(row["generation_node"]),
        )
    return history


def _solutions(
    history: History,
    searched: pd.DataFrame,
    picks: list[int],
    verified: list[int],
    front: np.ndarray,
    best_row: int,
    space: TiltSpace,
) -> pd.DataFrame:
    """One row per re-solved solution: what was predicted, what was measured, the gap.

    The surrogate's error on exactly the solutions it recommended, recorded on
    every run rather than assumed from the training report -- which was measured
    on a different set of tilts.
    """
    rows = []
    for solution, (searched_row, verified_row) in enumerate(zip(picks, verified, strict=True)):
        measured = history.results[verified_row].kpi
        row = {
            "solution": solution,
            "searched_iteration": int(searched_row),
            "is_incumbent": searched_row == 0,
            "recommended": verified_row == best_row,
            "on_pareto": bool(front[solution]),
            "seconds": history.results[verified_row].seconds,
        }
        for name in KPI_NAMES:
            actual = getattr(measured, name)
            guess = float(searched.iloc[searched_row][name])
            row[name] = actual
            row[f"predicted_{name}"] = guess
            row[f"error_{name}"] = guess - actual
        for column, value in zip(
            space.parameter_names, history.results[verified_row].tilt_deg, strict=True
        ):
            row[column] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _choice_table(solutions: pd.DataFrame, incumbent: KpiVector) -> pd.DataFrame:
    """The front an operator reads: each solution's KPIs and what it moves.

    Only the non-dominated rows, plus the incumbent it is all measured against.
    A dominated solution is worse on every count than one already in the table,
    so offering it is offering a mistake.
    """
    keep = solutions["on_pareto"] | solutions["is_incumbent"]
    table = solutions.loc[keep, ["solution", "is_incumbent", "recommended", *KPI_NAMES]].copy()
    for name in KPI_NAMES:
        table[f"delta_{name}"] = table[name] - getattr(incumbent, name)
    return table.reset_index(drop=True)


def _tilt_options(
    solutions: pd.DataFrame, history: History, verified: list[int], space: TiltSpace
) -> pd.DataFrame:
    """Long form: what each offered solution becomes on the antennas."""
    frames = []
    for position, (_, row) in enumerate(solutions.iterrows()):
        if not (row["on_pareto"] or row["is_incumbent"]):
            continue
        table = history.tilt_table(history.results[verified[position]].tilt_deg)
        table.insert(0, "recommended", bool(row["recommended"]))
        table.insert(0, "solution", int(row["solution"]))
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _print(
    solutions: pd.DataFrame,
    method: str,
    directory: Path,
    written: dict[str, str],
    scores: Path,
    options: Path,
) -> None:
    """Print the measured front, and how far the surrogate was off on it."""
    offered = solutions["on_pareto"].sum()
    error = solutions[[f"error_{name}" for name in KPI_NAMES]].abs().to_numpy().max()

    print(
        f"\n{method}: {len(solutions)} solutions re-solved, "
        f"{offered} still non-dominated once measured"
    )
    print(f"largest surrogate KPI error over them: {error:.4f}\n")

    columns = ["solution", "recommended", "on_pareto", *KPI_NAMES]
    print(solutions[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nwrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")
    print(f"\nchoose from: {scores}")
    print(f"tilt tables: {options}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report on one run. Entry point for ``task optim:report``.

    Example:
        $ task optim:report -- optim/method=random optim.report.n_solutions=4
    """
    report(cfg)


if __name__ == "__main__":
    main()
