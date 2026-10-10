"""Find optimization runs on disk, load them, and check they are comparable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.optim.objective import KpiVector

# Written beside every run by src.optim.history.write_run.
_TABLES = ("history",)

# The config blocks that define what a measurement IS rather than what it cost:
# the scene and solver, the antennas and the UE plane. Two runs that differ on
# any of them are not two results, they are two experiments.
_SIMULATION_KEYS = ("input", "ue", "scene", "radio_map", "antenna")


class RunError(Exception):
    """A run directory is unusable, or two runs cannot be compared."""


@dataclass(frozen=True)
class Run:
    """One optimization run, read back from its directory.

    Attributes:
        method: Which search produced it.
        run_id: The timestamped directory name, unique within a method.
        directory: Where it lives.
        history: One row per search evaluation; see
            :meth:`src.optim.history.History.frame`.
        meta: The parsed ``run.json``.
    """

    method: str
    run_id: str
    directory: Path
    history: pd.DataFrame
    meta: dict[str, Any]

    @property
    def label(self) -> str:
        """Method and run id, for a legend or an index."""
        return f"{self.method}/{self.run_id}"

    @property
    def n_evaluations(self) -> int:
        """How many configurations were evaluated, including the incumbent."""
        return len(self.history)

    @property
    def incumbent_kpi(self) -> KpiVector:
        """The committed tilts' measures."""
        return KpiVector.from_mapping(self.meta["incumbent_kpi"])

    @property
    def seed(self) -> int:
        """The search seed, ``optim.seed``, from the run's own config snapshot."""
        return int(self.meta["config"]["optim"]["seed"])

    @property
    def scenario_id(self) -> str:
        """The scenario this run optimized against."""
        return str(self.meta["scenario_id"])

    @property
    def wall_clock_seconds(self) -> float:
        """Total run time."""
        return float(self.meta["wall_clock_seconds"])

    @property
    def ray_tracing_seconds(self) -> float:
        """Simulator time, summed over every evaluation."""
        return float(self.history["seconds"].sum())


def load(directory: str | Path) -> Run:
    """Read one run directory.

    Raises:
        RunError: When ``run.json`` or any table is missing, which means the
            directory is not a run or the run did not finish.
    """
    directory = Path(directory)
    meta_path = directory / "run.json"
    if not meta_path.is_file():
        raise RunError(f"No run.json in {directory}; this is not a run directory.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tables = {}
    for name in _TABLES:
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise RunError(f"No {path}. The run did not finish writing.")
        tables[name] = pd.read_parquet(path)

    return Run(
        method=str(meta["method"]),
        run_id=directory.name,
        directory=directory,
        history=tables["history"],
        meta=meta,
    )


def discover(root: str | Path) -> list[Run]:
    """Every run under ``<root>/<method>/<run_id>/``, oldest first.

    Directories that are not runs are skipped rather than raising: the output
    root is shared scratch, and one stray folder should not stop a comparison.
    """
    runs = []
    for meta_path in sorted(Path(root).glob("*/*/run.json")):
        try:
            runs.append(load(meta_path.parent))
        except (RunError, KeyError, ValueError) as exc:
            print(f"WARNING skipping {meta_path.parent}: {exc}")
    return runs


def latest_per_method(runs: list[Run]) -> list[Run]:
    """The newest run of each method, ordered by method.

    Newest by run id, which is a UTC timestamp, so lexical order is
    chronological order.
    """
    latest = {run.method: run for run in sorted(runs, key=lambda run: run.run_id)}
    return [latest[method] for method in sorted(latest)]


def verify(runs: list[Run], cfg: DictConfig, scenario_id: str) -> pd.DataFrame:
    """Check every run may be compared with the others and with the current config.

    Returns one row per check, with ``holds`` and the offenders. Mirrors
    :func:`src.data.schema.verify` so the two read alike.

    Each run's ``run.json`` snapshot is checked against ``cfg``: the scenario,
    the simulation blocks and the KPI definition each change what a stored
    measurement means, and the evaluation re-traces under ``cfg``. A difference
    on any of them would not be attributable to tilt, which is the only thing
    this project varies.
    """
    checks: list[dict[str, Any]] = []

    def record(check: str, offenders: list[str]) -> None:
        checks.append({"check": check, "holds": not offenders, "offenders": ", ".join(offenders)})

    record(
        "every run optimized the current scenario",
        [run.label for run in runs if run.scenario_id != scenario_id],
    )
    current = OmegaConf.to_container(cfg.simulation, resolve=True)
    for key in _SIMULATION_KEYS:
        record(
            f"simulation.{key} matches the current config",
            [run.label for run in runs if run.meta["config"]["simulation"][key] != current[key]],
        )
    # The sector table lives outside the config snapshot, so its PRB limits,
    # which set every throughput, are compared as recorded.
    reference = runs[0].meta["max_prb"] if runs else None
    record(
        "sector PRB limits agree across runs",
        [run.label for run in runs if run.meta["max_prb"] != reference],
    )
    kpi = OmegaConf.to_container(cfg.kpi, resolve=True)
    record(
        "kpi matches the current config",
        [run.label for run in runs if run.meta["config"]["kpi"] != kpi],
    )
    return pd.DataFrame(checks, columns=["check", "holds", "offenders"])


def require(checks: pd.DataFrame) -> None:
    """Raise unless every check holds.

    Raises:
        RunError: Naming each failed check and who failed it.
    """
    failed = checks[~checks["holds"]]
    if failed.empty:
        return
    lines = [f"  {row.check}: {row.offenders}" for row in failed.itertuples()]
    raise RunError(
        f"{len(failed)} of {len(checks)} comparability checks failed:\n"
        + "\n".join(lines)
        + "\nThese runs did not measure the same thing. Re-run them against one "
        "scenario at one fidelity, or compare them separately."
    )
