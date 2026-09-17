"""Search the tilt space with Sionna-RT and publish what it found.

Entry point for ``task bo`` and ``task baseline``. Every candidate is ray
traced at the configured fidelity, so every KPI this writes is a measurement
and the run it produces is complete: a named winner and its shortlist, the
winner's radio map, and the two tables an operator chooses from.

Every candidate is ray-traced, with no surrogate. The search scores the
UE-counted measures on the MDT. The published shortlist is then re-traced and scored on
every UE, and that table, not the search history, is what evaluation compares.

Needs a CUDA GPU and the ``rt`` extra.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.optim.evaluator import Evaluator
from src.optim.history import History, LocalRunWriter
from src.optim.methods import run_search
from src.optim.objective import MEASURE_NAMES, score_frame
from src.optim.report import choose, publish
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

    Besides :func:`src.optim.report.publish`'s artifacts, writes
    ``evaluation.parquet``: the published solutions re-solved and scored on
    every UE, in shortlist order.

    Returns the history and the directory written to.
    """
    method = str(cfg.optim.method.name)
    directory = output_directory(cfg, method)
    started = time.time()

    with Evaluator(cfg) as evaluator:
        scenario_id = evaluator.scenario_id
        history = run_search(evaluator, cfg)
        best_index = history.best_index(cfg)
        # The same deterministic call publish makes, so solution numbers agree.
        picks = choose(history.kpis, cfg, int(cfg.optim.n_solutions), keep=(0, best_index))

        # One re-solve per published solution, on the evaluator already holding
        # the scene: the search keeps no maps, since a long run's do not fit in
        # memory. The solver seed is fixed for this evaluator's life, so these
        # are the maps the search measured. Not appended to the history, which
        # would duplicate measured points.
        evaluator.keep_rsrp = True
        rows, radio_map = [], None
        for solution, pick in enumerate(picks):
            result = evaluator.evaluate(history.results[pick].tilt_deg, all_ues=True)
            rows.append(
                {
                    "iteration": pick,
                    "solution": solution,
                    "is_incumbent": pick == 0,
                    "recommended": pick == best_index,
                    **result.kpi.as_dict(),
                }
            )
            if pick == best_index and bool(cfg.optim.output.save_radio_map):
                radio_map = str(evaluator.write_radio_map(directory / "best_radio_map.npz", result))

    evaluation = pd.DataFrame(rows)
    evaluation["score"] = score_frame(evaluation, cfg)
    LocalRunWriter(directory).write_frame("evaluation", evaluation)
    by_iteration = evaluation.set_index("iteration")

    publish(
        history,
        cfg,
        directory,
        method,
        extra={
            "scenario_id": scenario_id,
            "wall_clock_seconds": time.time() - started,
            "best_radio_map": radio_map,
            "incumbent_kpi_all_ues": _measures(by_iteration.loc[0]),
            "best_kpi_all_ues": _measures(by_iteration.loc[best_index]),
        },
    )
    print(f"\n{method}: published solutions scored on all UEs\n")
    print(evaluation.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return history, directory


def _measures(row: pd.Series) -> dict[str, float]:
    """One evaluation row's measures, as ``run.json`` records a KPI vector."""
    return {name: float(row[name]) for name in MEASURE_NAMES}


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize with one method. Entry point for ``task bo``.

    Example:
        $ task bo -- optim/method=random optim.method.budget.n_iter=0
    """
    history, directory = run(cfg)
    best = history.results[history.best_index(cfg)].kpi
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
