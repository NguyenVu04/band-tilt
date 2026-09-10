"""Find optimization runs on disk, load them, and check they are comparable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.objective import RAY_TRACED, KpiVector

# Written beside every run by src.optim.history.write_run.
_TABLES = ("history", "pareto", "best_tilt")
_RADIO_MAP = "best_radio_map.npz"

# Settings that define what a radio map IS rather than what it cost. Two maps
# that differ on any of them are not two results, they are two experiments.
_SOLVER_KEYS = (
    "samples_per_tx",
    "max_depth",
    "los",
    "specular_reflection",
    "diffuse_reflection",
    "refraction",
    "diffraction",
    "edge_diffraction",
    "diffraction_lit_region",
    "rr_depth",
    "power_dbm",
)
_GRID_KEYS = ("n_rows", "n_cols", "tile_size_m", "origin_x", "origin_y", "ue_height_m")


class RunError(Exception):
    """A run directory is unusable, or two runs cannot be compared."""


@dataclass(frozen=True)
class Run:
    """One optimization run, read back from its directory.

    Attributes:
        method: Which search produced it.
        run_id: The timestamped directory name, unique within a method.
        directory: Where it lives.
        history: One row per evaluation; see
            :meth:`src.optim.history.History.frame`.
        best_tilt: The deliverable table, one row per cell-band.
        meta: The parsed ``run.json``.
    """

    method: str
    run_id: str
    directory: Path
    history: pd.DataFrame
    best_tilt: pd.DataFrame
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
    def best_index(self) -> int:
        """Row of ``history`` the priority order selected."""
        return int(self.meta["best_iteration"])

    @property
    def incumbent_kpi(self) -> KpiVector:
        """The committed tilts' score, evaluated at the start of this run."""
        return KpiVector.from_mapping(self.meta["incumbent_kpi"])

    @property
    def best_kpi(self) -> KpiVector:
        """The winner's score."""
        return KpiVector.from_mapping(self.meta["best_kpi"])

    @property
    def scenario_id(self) -> str:
        """The scenario this run optimized against."""
        return str(self.meta["scenario_id"])

    @property
    def wall_clock_seconds(self) -> float | None:
        """Total run time, when the writer recorded it. None from a notebook."""
        value = self.meta.get("wall_clock_seconds")
        return None if value is None else float(value)

    @property
    def ray_tracing_seconds(self) -> float:
        """Simulator time, summed over the evaluations that were ray traced.

        A run holds both phases: the surrogate scored a few hundred candidates
        in milliseconds and Sionna-RT measured a handful in tens of seconds.
        Summing the column whole would report the search as simulator time and
        make the number meaningless. Runs written before the ``source`` column
        existed were ray traced throughout, so the sum is the same for them.
        """
        frame = self.history
        if "source" in frame:
            frame = frame[frame["source"] == RAY_TRACED]
        return float(frame["seconds"].sum())

    @cached_property
    def radio_map(self) -> dict[str, np.ndarray]:
        """The winner's radio map, in the schema the simulation stage writes.

        Read from the run's own directory rather than from
        ``meta["best_radio_map"]``: that field is whatever string the writing
        platform produced, and a Windows path does not resolve elsewhere.

        Raises:
            RunError: When the archive is missing.
        """
        path = self.directory / _RADIO_MAP
        if not path.is_file():
            raise RunError(
                f"No {path}. The run was written with optim.output.save_radio_map=false."
            )
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}


def load(directory: str | Path) -> Run:
    """Read one run directory.

    Raises:
        RunError: When ``run.json`` or any table is missing, which means the
            directory is not a run or the run did not finish; or when the run
            has only been searched and not yet reported on, whose KPIs are
            surrogate predictions and must not reach a comparison as though
            they were measurements.
    """
    directory = Path(directory)
    meta_path = directory / "run.json"
    if not meta_path.is_file():
        raise RunError(f"No run.json in {directory}; this is not a run directory.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Absent in runs written before the search and report phases were split.
    # Those ray traced every candidate, so they were verified throughout.
    if meta.get("verified", True) is False:
        raise RunError(
            f"{directory} has been searched but not reported on, so its KPIs are surrogate "
            "predictions rather than measurements. Run `task optim:report` first."
        )

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
        best_tilt=tables["best_tilt"],
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


def latest_per_method(runs: list[Run]) -> dict[str, Run]:
    """The newest run of each method, keyed by method.

    Newest by run id, which is a UTC timestamp, so lexical order is
    chronological order.
    """
    latest: dict[str, Run] = {}
    for run in sorted(runs, key=lambda run: run.run_id):
        latest[run.method] = run
    return latest


def baseline_map(cfg: DictConfig) -> dict[str, np.ndarray]:
    """The committed configuration's radio map, as the before of before-and-after.

    No separate artifact is needed: row 0 of every run's history is the
    committed tilt, and evaluating it reproduces this file.

    Raises:
        RunError: When the simulation stage has not been run.
    """
    path = Path(cfg.simulation.output.radio_map_file)
    if not path.is_file():
        raise RunError(f"No {path}. Run `task simulation:radio` first.")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify(runs: list[Run], baseline: dict[str, np.ndarray]) -> pd.DataFrame:
    """Check every run may be compared against the baseline and each other.

    Returns one row per check, with ``holds`` and the offenders. Mirrors
    :func:`src.data.schema.verify` so the two read alike.

    ``README.md`` names the ray-tracing settings, the grid resolution and the
    KPI thresholds as the three things that invalidate stored results rather
    than adding to them. Plotting two maps that disagree on any of them on one
    axis would produce a difference that is not attributable to tilt, which is
    the only thing this project varies.
    """
    checks: list[dict[str, Any]] = []

    def record(check: str, offenders: list[str]) -> None:
        checks.append({"check": check, "holds": not offenders, "offenders": ", ".join(offenders)})

    expected_scenario = str(baseline["scenario_id"])
    record(
        "every run optimized the baseline's scenario",
        [run.label for run in runs if run.scenario_id != expected_scenario],
    )

    for key in _GRID_KEYS:
        record(
            f"grid {key} matches the baseline",
            [
                run.label
                for run in runs
                if not np.array_equal(np.asarray(run.radio_map[key]), np.asarray(baseline[key]))
            ],
        )

    for key in _SOLVER_KEYS:
        record(
            f"solver {key} matches the baseline",
            [
                run.label
                for run in runs
                if not np.array_equal(np.asarray(run.radio_map[key]), np.asarray(baseline[key]))
            ],
        )

    record(
        "bands match the baseline, in order",
        [
            run.label
            for run in runs
            if [str(label) for label in run.radio_map["band_label"]]
            != [str(label) for label in baseline["band_label"]]
        ],
    )
    record(
        "every run recorded the KPI thresholds it scored with",
        [run.label for run in runs if "kpi" not in run.meta.get("config", {})],
    )

    thresholds = [
        (run.label, _thresholds(run)) for run in runs if "kpi" in run.meta.get("config", {})
    ]
    reference = thresholds[0][1] if thresholds else None
    record(
        "KPI thresholds agree across runs",
        [label for label, values in thresholds if values != reference],
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


def _thresholds(run: Run) -> dict[str, Any]:
    """The KPI thresholds a run scored with, from its own config snapshot."""
    kpi = run.meta["config"]["kpi"]
    return {key: kpi.get(key) for key in ("hole_dbm", "weak_dbm", "overlap_margin_db")}
