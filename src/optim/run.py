"""Search the tilt space with Sionna-RT and publish what it found.

Entry point for ``task bo`` and ``task baseline``. Every candidate is ray
traced at the configured fidelity, so every KPI this writes is a measurement
and the run it produces is complete: a named winner and its shortlist, the
winner's radio map, and the two tables an operator chooses from.

One phase: every candidate is ray-traced, with no surrogate and no re-solve;
``outputs/fidelity_bench/`` holds the timing that made that affordable.

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
    return Path(cfg.optim.output.dir) / method / stamp


def run(cfg: DictConfig) -> tuple[History, Path]:
    """Search the tilt space with the selected method, then publish the result.

    Returns the history and the directory written to.
    """
    method = str(cfg.optim.method.name)
    directory = output_directory(cfg, method)
    started = time.time()

    with Evaluator(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        history = run_search(evaluator, cfg)
        best_index = history.best_index(cfg)
        best = history.results[best_index]

        radio_map = None
        if bool(cfg.optim.output.save_radio_map):
            # One extra solve, on the evaluator already holding the scene: the
            # search runs keep_rsrp=False because every map of a long run does
            # not fit in memory, and building a second Evaluator to archive one
            # map would pay the scene load again for nothing. The solver seed is
            # fixed for this evaluator's life, so this is the map that produced
            # the KPIs above. It is deliberately not appended to the history,
            # which would duplicate a measured point.
            evaluator.keep_rsrp = True
            radio_map = str(
                evaluator.write_radio_map(
                    directory / "best_radio_map.npz", evaluator.evaluate(best.tilt_deg)
                )
            )

    publish(
        history,
        cfg,
        directory,
        method,
        extra={
            "scenario_id": scenario_id,
            "wall_clock_seconds": time.time() - started,
            "best_radio_map": radio_map,
        },
    )
    return history, directory


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize with one method. Entry point for ``task bo``.

    Example:
        $ task bo -- optim/method=random optim.method.budget.n_iter=0
    """
    history, directory = run(cfg)
    frame = history.frame()
    best = history.results[history.best_index(cfg)].kpi
    log_stage(
        cfg,
        "optimization",
        groups=["optim", "kpi"],
        metrics={
            **{f"best_{name}": value for name, value in best.as_dict().items()},
            "n_candidates": len(frame),
        },
        artifacts=sorted(directory.glob("*.parquet")) + [directory / "run.json"],
        tags={"method": cfg.optim.method.name, "run_dir": directory},
    )


if __name__ == "__main__":
    main()
