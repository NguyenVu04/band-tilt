"""Search the tilt space with Sionna-RT and publish what it found.

Entry point for ``task bo`` and ``task baseline``. Every candidate is ray
traced at the configured fidelity, with no surrogate, so every KPI this writes
is a measurement and the run it produces is complete: a named winner and its
shortlist, the winner's radio map, and the two tables an operator chooses from.
The served ratio counts every UE (``data.output.ue_file``).

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

    Besides :func:`src.optim.report.publish`'s artifacts, writes the winner's
    radio map when ``optim.output.save_radio_map`` is set.

    Returns the history and the directory written to.
    """
    method = str(cfg.optim.method.name)
    directory = output_directory(cfg, method)
    started = time.time()

    with Evaluator(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        history = run_search(evaluator, cfg)
        radio_map = None
        if bool(cfg.optim.output.save_radio_map):
            # Re-solved, not kept during the search: a long run's maps do not fit
            # in memory. Same solver seed as the search, but GPU ray tracing is not
            # bit-reproducible, so a tile can differ. Not appended to the history.
            evaluator.keep_rsrp = True
            result = evaluator.evaluate(history.results[history.best_index()].tilt_deg)
            radio_map = str(evaluator.write_radio_map(directory / "best_radio_map.npz", result))

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
