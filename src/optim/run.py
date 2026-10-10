"""Search the tilt space with Sionna-RT and publish what it found.

Entry point for ``task bo``, ``task baseline`` and each method ``task optim``
loops over. Every candidate is ray
traced at the configured fidelity, with no surrogate, so every KPI this writes
is a measurement and the run it produces is complete: the history, its Pareto
shortlist, and the two tables an operator chooses from.
The UE KPIs count every UE (``data.output.ue_file``).

Needs a CUDA GPU and the ``rt`` extra.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.optim.evaluator import Evaluator
from src.optim.history import History
from src.optim.methods import run_search
from src.optim.report import publish
from src.tracking import log_stage


def output_directory(cfg: DictConfig, method: str) -> Path:
    """One directory per run: ``<optim.output.dir>/<method>/<timestamp>``.

    Timestamped rather than overwritten, because comparing methods means
    comparing runs and a rerun must not destroy the one it is compared against.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(cfg.optim.output.dir) / method
    # A second run in the same second would otherwise write into the first's directory.
    directory, suffix = base / stamp, 0
    while directory.exists():
        suffix += 1
        directory = base / f"{stamp}-{suffix}"
    return directory


def run(cfg: DictConfig) -> tuple[History, Path]:
    """Search the tilt space with the selected method, then publish the result.

    Returns the history and the directory written to.
    """
    method = str(cfg.optim.method.name)
    directory = output_directory(cfg, method)
    started = time.time()

    with Evaluator(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        # The sector table lives outside the config snapshot, so the PRB limits the
        # throughput was measured under are recorded with the run.
        max_prb = {sector.name: sector.max_prb for sector in evaluator.space.sectors}
        history = run_search(evaluator, cfg)

    publish(
        history,
        cfg,
        directory,
        method,
        extra={
            "scenario_id": scenario_id,
            "max_prb": max_prb,
            "wall_clock_seconds": time.time() - started,
        },
    )
    return history, directory


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize with one method.

    Entry point for ``task bo``, ``task baseline`` and each method ``task optim``
    loops over.
    """
    history, directory = run(cfg)
    best = history.results[history.best_index()].kpi
    log_stage(
        cfg,
        "optimization",
        groups=["optim", "kpi"],
        metrics={
            **{f"best_{name}": value for name, value in best.as_dict().items()},
            "n_candidates": len(history),
        },
        artifacts=sorted(directory.glob("*.parquet")) + [directory / "run.json"],
        tags={"method": cfg.optim.method.name, "run_dir": directory},
    )


if __name__ == "__main__":
    main()
