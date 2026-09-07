"""Run one optimization method and write its artifacts.

Entry point for ``task bo`` and ``task baseline``. The notebooks call
:func:`run` directly with a composed config, so the two paths execute the same
code and produce the same artifacts.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.optim.evaluator import Evaluator
from src.optim.history import History, LocalRunWriter, write_run
from src.optim.objective import KPI_NAMES
from src.optim.search import run_search


def output_directory(cfg: DictConfig, method: str) -> Path:
    """One directory per run: ``<bo.output.dir>/<method>/<timestamp>``.

    Timestamped rather than overwritten, because comparing methods means
    comparing runs and a rerun must not destroy the one it is compared against.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(cfg.bo.output.dir) / method / stamp


def run(cfg: DictConfig) -> tuple[History, Path]:
    """Search the tilt space with ``cfg.bo.method`` and write the results.

    Returns the history and the directory written to.
    """
    method = str(cfg.bo.method)
    directory = output_directory(cfg, method)
    started = time.time()

    with Evaluator(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        history = run_search(evaluator, cfg, method)
        best_index = history.best_index(cfg)
        best = history.results[best_index]

        radio_map = None
        if bool(cfg.bo.output.save_radio_map):
            # Re-solved rather than retained: holding every candidate's map
            # costs more memory than the run needs, and the winner is not known
            # until the run is over. The solver seed is fixed, so this repeats
            # the evaluation that produced the recorded KPIs.
            evaluator.keep_rsrp = True
            archived = evaluator.evaluate(best.tilt_deg)
            radio_map = str(evaluator.write_radio_map(directory / "best_radio_map.npz", archived))

    writer = LocalRunWriter(directory)
    written = write_run(
        history,
        writer,
        cfg,
        method=method,
        best_index=best_index,
        extra={
            "scenario_id": scenario_id,
            "wall_clock_seconds": time.time() - started,
            "best_radio_map": radio_map,
        },
    )

    _report(history, best_index, method, directory, written)
    return history, directory


def _report(
    history: History,
    best_index: int,
    method: str,
    directory: Path,
    written: dict[str, str],
) -> None:
    """Print what the run found and where it went."""
    incumbent = history.results[0].kpi
    best = history.results[best_index].kpi
    frame = history.frame()

    print(f"\n{method}: {len(history)} evaluations, {int(frame['on_pareto'].sum())} non-dominated")
    print(f"winner is evaluation {best_index}\n")
    print(f"{'KPI':<26}{'incumbent':>12}{'best':>12}{'delta':>12}")
    for name in KPI_NAMES:
        before, after = getattr(incumbent, name), getattr(best, name)
        print(f"{name:<26}{before:>12.4f}{after:>12.4f}{after - before:>+12.4f}")
    print(f"\nray tracing {frame['seconds'].sum():.1f}s over {len(history)} evaluations")
    print(f"wrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")


def _quiet_ax_logging() -> None:
    """Stop Ax logging every generated trial, which here is 36 tilts a line.

    Done in the entry point rather than in the library, so importing
    :mod:`src.optim` never reconfigures a caller's logging.

    Ax sets an explicit level on each of its child loggers, so lowering the
    parent alone does not reach them — they have to be walked. Ax is imported
    first to make sure they exist by the time we do. Warnings are left alone: a
    short batch from the generation strategy is worth seeing.
    """
    import ax  # noqa: F401

    for name, logger in logging.root.manager.loggerDict.items():
        if (name == "ax" or name.startswith("ax.")) and isinstance(logger, logging.Logger):
            logger.setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one method. Entry point for ``task bo``.

    Example:
        $ task bo -- bo.method=random bo.budget.n_iter=0
    """
    _quiet_ax_logging()
    run(cfg)


if __name__ == "__main__":
    main()
