"""Phase one: search the tilt space with the surrogate.

Entry point for ``task bo`` and ``task baseline``. Every candidate is scored by
:class:`~src.surrogate.evaluator.SurrogateEvaluator`, so a few hundred of them
cost minutes and need no GPU, and **nothing here is ground truth**. The run this
writes is deliberately incomplete: it has a predicted Pareto front and no
winner, and ``run.json`` says ``verified: false`` so nothing downstream can
mistake a prediction for a measurement.

:mod:`src.optim.report` is phase two. It re-solves the front with Sionna-RT,
picks from what it measured, and completes the run directory.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.optim.history import History, LocalRunWriter, write_run
from src.optim.methods import run_search


def output_directory(cfg: DictConfig, method: str) -> Path:
    """One directory per run: ``<optim.output.dir>/<method>/<timestamp>``.

    Timestamped rather than overwritten, because comparing methods means
    comparing runs and a rerun must not destroy the one it is compared against.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(cfg.optim.output.dir) / method / stamp


def run(cfg: DictConfig) -> tuple[History, Path]:
    """Search the tilt space with the selected method and write the search log.

    Returns the history and the directory written to. No deliverable is
    published here: which point on a five-objective front to deploy is a
    judgement made against measured KPIs, and this phase has none.
    """
    method = str(cfg.optim.method.name)
    directory = output_directory(cfg, method)
    started = time.time()

    # Imported here rather than at module scope so the report phase, and any
    # environment without torch, can import this module for output_directory.
    from src.surrogate.evaluator import from_config

    with from_config(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        history = run_search(evaluator, cfg)

    writer = LocalRunWriter(directory)
    written = write_run(
        history,
        writer,
        cfg,
        method=method,
        extra={
            "scenario_id": scenario_id,
            "wall_clock_seconds": time.time() - started,
            "surrogate_model": str(cfg.optim.search.model_file),
        },
    )

    _report(history, method, directory, written)
    return history, directory


def _report(history: History, method: str, directory: Path, written: dict[str, str]) -> None:
    """Print what the search proposed and what has to happen next."""
    frame = history.frame()
    front = int(frame["on_pareto"].sum())

    print(f"\n{method}: {len(history)} candidates scored, {front} on the predicted front")
    print(f"surrogate time {frame['seconds'].sum():.1f}s\n")
    print(f"wrote {directory}")
    for name, locator in written.items():
        print(f"  {name}: {locator}")
    print(
        "\nThese KPIs are predictions, not measurements, and this run is not reportable yet.\n"
        "Run `task optim:report` to re-solve the front with Sionna-RT and choose from it."
    )


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
    """Search with one method. Entry point for ``task bo``.

    Example:
        $ task bo -- optim/method=random optim.method.budget.n_iter=0
    """
    _quiet_ax_logging()
    run(cfg)


if __name__ == "__main__":
    main()
