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

from src.optim.objective import KpiVector

# Written beside every run by src.optim.history.write_run.
_TABLES = ("history", "best_tilt")
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
    "rr_prob",
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
        history: One row per search evaluation; see
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
        """Row of ``history`` with the highest objective."""
        return int(self.meta["best_iteration"])

    @property
    def incumbent_kpi(self) -> KpiVector:
        """The committed tilts' measures."""
        return KpiVector.from_mapping(self.meta["incumbent_kpi"])

    @property
    def best_kpi(self) -> KpiVector:
        """The winner's measures."""
        return KpiVector.from_mapping(self.meta["best_kpi"])

    @property
    def seed(self) -> int:
        """The search seed, ``optim.seed``, from the run's own config snapshot."""
        return int(self.meta["config"]["optim"]["seed"])

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
        """Simulator time, summed over every evaluation. All of them are solved."""
        return float(self.history["seconds"].sum())

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


def latest_per_method_and_seed(runs: list[Run]) -> list[Run]:
    """The newest run of each (method, seed), ordered by method then seed.

    Newest by run id, which is a UTC timestamp, so lexical order is
    chronological order.

    Raises:
        KeyError: When a run's config snapshot records no ``optim.seed``.
    """
    latest: dict[tuple[str, int], Run] = {}
    for run in sorted(runs, key=lambda run: run.run_id):
        latest[(run.method, run.seed)] = run
    return [latest[key] for key in sorted(latest)]


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


def verify(runs: list[Run], baseline: dict[str, np.ndarray], cfg: DictConfig) -> pd.DataFrame:
    """Check every run may be compared against the baseline and each other.

    Returns one row per check, with ``holds`` and the offenders. Mirrors
    :func:`src.data.schema.verify` so the two read alike.

    The ray-tracing settings, the grid and the KPI definition each change what
    a stored result measures. Plotting two maps that disagree on any of them on
    one axis would produce a difference that is not attributable to tilt, which
    is the only thing this project varies.

    ``cfg`` supplies the objective version the running code implements. Checking
    the runs against each other is not enough on its own: a set that is uniformly
    stale agrees with itself and would be reported under the current code's
    labels.
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
    # A band keeps its name when its carrier is retuned, so the labels agreeing
    # does not make two maps the same network.
    record(
        "band carrier frequencies match the baseline, in order",
        [
            run.label
            for run in runs
            if not np.array_equal(
                np.asarray(run.radio_map["band_hz"]), np.asarray(baseline["band_hz"])
            )
        ],
    )
    record(
        "every run recorded the KPI definition it scored with",
        [run.label for run in runs if "kpi" not in run.meta.get("config", {})],
    )

    definitions = [
        (run.label, _kpi_definition(run)) for run in runs if "kpi" in run.meta.get("config", {})
    ]
    reference = definitions[0][1] if definitions else None
    record(
        "KPI definition agrees across runs",
        [label for label, values in definitions if values != reference],
    )

    expected_version = cfg.kpi.get("objective_version")
    record(
        "every run was scored by the objective this code implements",
        [
            run.label
            for run in runs
            if run.meta.get("config", {}).get("kpi", {}).get("objective_version")
            != expected_version
        ],
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


def _kpi_definition(run: Run) -> dict[str, Any]:
    """Everything a reported measure reads, from the run's own config snapshot.

    The thresholds, the capacity model and each cell's PRB limit: the KPIs and
    the objective depend on all of them, so two runs that differ on any one did
    not measure the same thing. The RSRP and SINR percentiles are not here
    because they are constants in :mod:`src.kpi.quality` and no run can differ
    on them. ``capacity`` is here for the serving rule behind the served rate and
    both load measures; the objective does not read it (ADR 0003).

    ``scs_hz`` and ``temperature`` are here because kT over the subcarrier
    spacing is the per-RE noise floor behind every SINR, and SINR and ``scs_hz``
    set the PRBs a UE needs and therefore the served rate. ``bandwidth`` fixes
    ``max_prb``. None is stored in the archive, so the config snapshot is the
    only place they can be checked.

    ``objective_version`` is the guard for the objective's functional form, which
    lives in code and not in config: the config alone cannot tell two utilities
    apart, so the version is bumped by hand whenever the form changes and that
    difference is what refuses to pool two scores on different scales. An earlier
    objective relied instead on its parameter block disappearing, which only worked by
    accident. ``objective`` is still read so runs predating the version key,
    which carry the old ``tau_r_db`` / ``beta`` / ``alpha`` block, are refused too.
    """
    config = run.meta["config"]
    kpi = config["kpi"]
    simulation = config.get("simulation", {})
    cells = simulation.get("transmitters", {}).get("cells", [])
    radio_map = simulation.get("radio_map", {})
    return {
        **{key: kpi.get(key) for key in ("hole_dbm", "weak_dbm", "overlap_margin_db")},
        "capacity": kpi.get("capacity"),
        "objective": kpi.get("objective"),
        "objective_version": kpi.get("objective_version"),
        "max_prb": {cell.get("name"): cell.get("max_prb") for cell in cells},
        "bandwidth": {
            band.get("name"): band.get("bandwidth") for band in radio_map.get("bands", [])
        },
        "scs_hz": {band.get("name"): band.get("scs_hz") for band in radio_map.get("bands", [])},
        "temperature": radio_map.get("temperature"),
    }
