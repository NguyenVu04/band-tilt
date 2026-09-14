"""The evaluation log, and the artifacts a run leaves behind.

Every method writes the same tables, so a comparison between them is a
comparison of search strategies rather than of bookkeeping. :class:`History`
only builds frames; :class:`LocalRunWriter` puts them on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.optim.evaluator import EvaluationResult
from src.optim.objective import (
    KPI_NAMES,
    KpiVector,
    best_by_score,
    scores,
)
from src.optim.space import TiltSpace


@dataclass(frozen=True)
class LocalRunWriter:
    """Writes a run's artifacts into one directory.

    Parquet rather than CSV for the tables, matching :func:`src.data.load.save`:
    it round-trips dtypes, so the boolean flags stay boolean and the KPI columns
    keep their float type instead of being re-guessed on read.
    """

    directory: Path

    def __post_init__(self) -> None:
        """Create the directory, including parents."""
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_frame(self, name: str, frame: pd.DataFrame) -> str:
        """Write one table as ``<name>.parquet``."""
        path = self.directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return str(path)

    def write_json(self, name: str, payload: dict[str, Any]) -> str:
        """Write one document as ``<name>.json``."""
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)


def write_tilt_change(table: pd.DataFrame, cfg: DictConfig, method: str) -> Path:
    """Republish the tilt table as the current deliverable for ``method``.

    One file per method under ``cfg.optim.output.deliverable_dir``, overwritten
    every run: the run directory keeps the history, and this answers what the
    current answer is without globbing timestamps. CSV rather than parquet
    because the reader is an operator, not this codebase.

    Returns:
        The path written.
    """
    directory = Path(cfg.optim.output.deliverable_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tilt_change_{method}.csv"
    # Index dropped for the same reason as src.data.load.save: every column
    # this table needs is already a column.
    table.to_csv(path, index=False)
    return path


def write_solution_options(
    shortlist: pd.DataFrame, tilts: pd.DataFrame, cfg: DictConfig, method: str
) -> tuple[Path, Path]:
    """Republish the shortlist as the two tables an operator chooses from.

    The weighted score (``kpi.weights``, ADR 0003) marks one row
    ``recommended``; the runners-up are published beside it rather than
    discarded.

    Two tables because they answer two questions. ``solutions_<method>.csv`` is
    one row per solution and says what each one costs and buys.
    ``tilt_options_<method>.csv`` is one row per solution and cell-band, and is
    what a chosen row turns into on the antennas.

    Returns:
        The two paths written, shortlist first.
    """
    directory = Path(cfg.optim.output.deliverable_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shortlist_path = directory / f"solutions_{method}.csv"
    tilt_path = directory / f"tilt_options_{method}.csv"
    shortlist.to_csv(shortlist_path, index=False)
    tilts.to_csv(tilt_path, index=False)
    return shortlist_path, tilt_path


@dataclass
class History:
    """Every evaluation of one run, in the order they were made.

    Attributes:
        space: The space the vectors belong to, supplying the column names.
        results: The evaluations themselves, appended as they complete.
    """

    space: TiltSpace
    results: list[EvaluationResult] = field(default_factory=list)
    _phases: list[str] = field(default_factory=list, repr=False)
    _nodes: list[str] = field(default_factory=list, repr=False)

    def append(
        self,
        result: EvaluationResult,
        *,
        phase: str,
        generation_node: str = "",
    ) -> None:
        """Record one evaluation.

        Args:
            result: What the evaluator returned.
            phase: Which part of the run produced it — ``incumbent``, ``init``,
                ``search`` or ``sweep``. What separates the exploration budget
                from the model-driven one in a plot.
            generation_node: The generator's name, e.g. ``Sobol`` or
                ``TuRBO``: the record of whether a point came from the
                design or from the model.
        """
        self.results.append(result)
        self._phases.append(phase)
        self._nodes.append(generation_node)

    def __len__(self) -> int:
        """How many evaluations have been recorded."""
        return len(self.results)

    @property
    def kpis(self) -> list[KpiVector]:
        """The KPI vector of every evaluation, in order."""
        return [result.kpi for result in self.results]

    def frame(self, cfg: DictConfig | None = None) -> pd.DataFrame:
        """One row per evaluation: provenance, the four KPIs, every tilt.

        With ``cfg``, also ``score``, the weighted score that selects the winner
        (reads ``kpi.weights``).

        Raises:
            ValueError: When nothing has been recorded.
        """
        if not self.results:
            raise ValueError("no evaluations to tabulate")

        tilts = np.array([result.tilt_deg for result in self.results], dtype=float)
        frame = pd.DataFrame(
            {
                "iteration": np.arange(len(self.results)),
                "phase": self._phases,
                "generation_node": self._nodes,
                "seconds": [result.seconds for result in self.results],
            }
        )
        for name in KPI_NAMES:
            frame[name] = [getattr(result.kpi, name) for result in self.results]
        if cfg is not None:
            frame["score"] = scores(self.kpis, cfg)
        for index, column in enumerate(self.space.parameter_names):
            frame[column] = tilts[:, index]
        return frame

    def best_index(self, cfg: DictConfig) -> int:
        """Index of the highest weighted score over every evaluation.

        A tie keeps the earlier row, so the incumbent holds unless beaten.
        """
        return best_by_score(self.kpis, cfg)

    def tilt_table(self, tilt_deg: np.ndarray) -> pd.DataFrame:
        """The deliverable: current, optimized and delta tilt per cell-band.

        ``delta_tilt_deg`` is reported, never optimized. A penalty on antenna
        movement is an explicit non-goal, so nothing in the objective has seen
        this column.
        """
        table = self.space.as_frame(tilt_deg).rename(columns={"tilt_deg": "optimized_tilt_deg"})
        table.insert(2, "current_tilt_deg", self.space.baseline)
        table.insert(4, "delta_tilt_deg", table["optimized_tilt_deg"] - table["current_tilt_deg"])
        return table


def write_run(
    history: History,
    writer: LocalRunWriter,
    cfg: DictConfig,
    *,
    method: str,
    best_index: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write every artifact of one run. Returns the locators, keyed by name.

    The run document carries the whole composed config and the scenario it was
    solved against, so a result can be traced back to the inputs that produced
    it without consulting anything outside its own directory.

    Args:
        history: The evaluation log.
        writer: Where the artifacts go.
        cfg: The composed config, recorded whole.
        method: The search that produced the history.
        best_index: The row to report as the winner.
        extra: Merged into the run document, for whatever the caller knows and
            this function does not.
    """
    frame = history.frame(cfg)
    best = history.results[best_index]
    # Row zero is the committed incumbent every delta is measured against; the
    # SearchMethod contract puts it there.
    incumbent = history.results[0].kpi

    written = {
        "history": writer.write_frame("history", frame),
        "best_tilt": writer.write_frame("best_tilt", history.tilt_table(best.tilt_deg)),
    }

    written["run"] = writer.write_json(
        "run",
        {
            "method": method,
            "n_evaluations": len(history),
            "best_iteration": int(best_index),
            "best_kpi": best.kpi.as_dict(),
            "incumbent_kpi": incumbent.as_dict(),
            "ray_tracing_seconds": float(frame["seconds"].sum()),
            "config": OmegaConf.to_container(cfg, resolve=True),
            **(extra or {}),
        },
    )
    return written
