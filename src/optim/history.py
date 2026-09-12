"""The evaluation log, and the artifacts a run leaves behind.

Every method writes the same tables, so a comparison between them is a
comparison of search strategies rather than of bookkeeping. :class:`History`
only builds frames; where those frames land is the writer's business, which is
what lets a deployment send a run somewhere other than a local directory
without the search knowing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.optim.evaluator import EvaluationResult
from src.optim.objective import (
    KPI_NAMES,
    KpiVector,
    lexicographic_best,
    pareto_mask,
)
from src.optim.space import TiltSpace


class RunWriter(Protocol):
    """Where a run's artifacts go."""

    def write_frame(self, name: str, frame: pd.DataFrame) -> str:
        """Persist a table under ``name``. Returns a locator for the log."""
        ...

    def write_json(self, name: str, payload: dict[str, Any]) -> str:
        """Persist a document under ``name``. Returns a locator for the log."""
        ...


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


def write_pareto_options(
    scores: pd.DataFrame, tilts: pd.DataFrame, cfg: DictConfig, method: str
) -> tuple[Path, Path]:
    """Republish the verified Pareto front as the two tables an operator chooses from.

    Four objectives do not have a best; they have a front, and which point on
    it to deploy is a judgement about what this network needs, not something
    the priority order in ADR 0001 can settle. That rule still runs and marks
    one row ``recommended``, but the rest of the front is published beside it
    rather than discarded.

    Two tables because they answer two questions. ``pareto_<method>.csv`` is
    one row per solution and says what each one costs and buys.
    ``tilt_options_<method>.csv`` is one row per solution and cell-band, and is
    what a chosen row turns into on the antennas.

    Returns:
        The two paths written, scores first.
    """
    directory = Path(cfg.optim.output.deliverable_dir)
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / f"pareto_{method}.csv"
    tilt_path = directory / f"tilt_options_{method}.csv"
    scores.to_csv(score_path, index=False)
    tilts.to_csv(tilt_path, index=False)
    return score_path, tilt_path


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
            generation_node: The generator's own name, when it has one. Ax
                reports this per trial, and it is the honest record of whether
                a point came from Sobol or from the model.
        """
        self.results.append(result)
        self._phases.append(phase)
        self._nodes.append(generation_node)

    def __len__(self) -> int:
        """How many evaluations have been recorded."""
        return len(self.results)

    def set_generation_nodes(self, nodes: list[str]) -> None:
        """Attach the generator name for each recorded evaluation.

        Separate from :meth:`append` because the authority on which generator
        produced a trial is the optimizer's own record, and that is only
        readable once the run is over.

        Raises:
            ValueError: When the count does not match the recorded evaluations,
                which would silently misattribute every row after the gap.
        """
        if len(nodes) != len(self.results):
            raise ValueError(f"{len(nodes)} generation nodes for {len(self.results)} evaluations")
        self._nodes = list(nodes)

    def set_phases(self, phases: list[str]) -> None:
        """Relabel which part of the run produced each evaluation.

        For an optimizer that decides its own phase boundary, the label is
        only knowable once its record can be read, and its record outranks
        any count the loop kept.

        Raises:
            ValueError: When the count does not match the recorded
                evaluations.
        """
        if len(phases) != len(self.results):
            raise ValueError(f"{len(phases)} phases for {len(self.results)} evaluations")
        self._phases = list(phases)

    @property
    def kpis(self) -> list[KpiVector]:
        """The KPI vector of every evaluation, in order."""
        return [result.kpi for result in self.results]

    def frame(self) -> pd.DataFrame:
        """One row per evaluation: provenance, the four KPIs, all 36 tilts.

        ``on_pareto`` is computed here rather than stored, because it is a
        property of the set and every append can change it.

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
        for index, column in enumerate(self.space.parameter_names):
            frame[column] = tilts[:, index]

        frame["on_pareto"] = pareto_mask(self.kpis)
        return frame

    def best_index(self, cfg: DictConfig) -> int:
        """Index of the configuration the priority order selects.

        The single pass runs over every evaluation rather than over the Pareto
        front alone: the winner is non-dominated either way, and restricting the
        candidate set first would make the result depend on the front's order.
        """
        return lexicographic_best(self.kpis, cfg)

    def pareto_frame(self) -> pd.DataFrame:
        """The non-dominated rows of :meth:`frame`."""
        frame = self.frame()
        return frame[frame["on_pareto"]].reset_index(drop=True)

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
    writer: RunWriter,
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
    frame = history.frame()
    best = history.results[best_index]
    # Row zero is the committed incumbent every delta is measured against; the
    # SearchMethod contract puts it there.
    incumbent = history.results[0].kpi

    written = {
        "history": writer.write_frame("history", frame),
        "pareto": writer.write_frame("pareto", history.pareto_frame()),
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
            "n_pareto": int(frame["on_pareto"].sum()),
            "config": OmegaConf.to_container(cfg, resolve=True),
            **(extra or {}),
        },
    )
    return written
