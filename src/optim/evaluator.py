"""Turn a tilt vector into a KPI vector by ray tracing.

The expensive half of every optimization run, and the reason a run is feasible
at all: the scene, its perturbation and the antenna arrays do not depend on
tilt, so they are built once at construction and reused for every candidate.
Only the transmitters are rebuilt per evaluation, which is what
:func:`src.simulation.radio.solve_band` already does.

:class:`ObjectiveEvaluator` is the seam the search depends on. Anything mapping
a tilt vector to an :class:`EvaluationResult` satisfies it — the ray tracer
here, a stub in a test, or the surrogate the roadmap plans — so the search code
never learns which one it holds.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.objective import KpiVector, evaluate_kpis
from src.optim.space import TiltSpace
from src.simulation import perturb, radio, seeds, transmitter
from src.simulation import scenario as scenario_module
from src.simulation import scene as scene_module
from src.simulation.grid import GridSpec
from src.simulation.materials import MaterialSpec
from src.simulation.perturb import PerturbSpec
from src.simulation.scene import SceneSpec


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate, scored.

    Attributes:
        tilt_deg: The vector evaluated, in :class:`~src.optim.space.TiltSpace`
            dimension order.
        kpi: Its score on all five KPIs.
        seconds: Wall clock for the ray tracing, summed over bands, measured
            around the point the maps are actually materialised. Excludes
            scoring and anything the caller does, so a run can report
            simulator time apart from model time.
        rsrp: The radio map, ``[n_band, n_tx, n_rows, n_cols]`` in dBm, or None
            when the evaluator was asked not to retain it. Every map of a long
            run does not fit in memory and the run does not need them.
    """

    tilt_deg: np.ndarray
    kpi: KpiVector
    seconds: float
    rsrp: np.ndarray | None = None


class ObjectiveEvaluator(Protocol):
    """Anything that can score a tilt vector.

    The search depends on this rather than on a concrete evaluator, so the loop
    can be exercised without a GPU and a faster evaluator can replace the ray
    tracer without the search changing.
    """

    space: TiltSpace

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Score one tilt vector."""
        ...


@dataclass
class Evaluator:
    """Ray-trace a tilt vector and score the resulting radio map.

    Construction loads the scene, applies this scenario's building
    perturbation, attaches the antenna arrays and checks the masts still stand
    on open ground — everything :func:`src.simulation.radio.solve` does per call
    that does not depend on tilt. It is a large fixed fraction of what one
    evaluation costs, so paying it per candidate would add roughly half again
    to every point in the run.

    Use it as a context manager. It holds GPU and scene state, and releasing
    that when the run ends rather than whenever the collector notices is what
    lets a long process or a service scope it predictably.

    Attributes:
        cfg: The composed config, read for everything below.
        keep_rsrp: Whether each result carries its radio map. False by default
            because a run keeps every result and the maps do not fit.
    """

    cfg: DictConfig
    keep_rsrp: bool = False

    space: TiltSpace = field(init=False)
    n_calls: int = field(init=False, default=0)
    total_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Build the scene and everything else that does not depend on tilt."""
        cfg = self.cfg
        self.space = TiltSpace.from_config(cfg)
        self._manifest = _read_manifest(cfg)
        self._grid_meta = self._manifest["grid"]
        self._bands = tuple(
            radio.Band.from_config(entry) for entry in cfg.simulation.radio_map.bands
        )
        self._solver = radio.SolverSpec.from_config(cfg)
        self._materials = MaterialSpec.from_config(cfg)
        self._material_seed = seeds.stream(cfg, "materials")
        # One solver seed for every candidate, deliberately. Sharing the
        # Monte-Carlo stream makes its noise common to all of them, so the KPI
        # differences the optimizer compares are much less noisy than the KPIs
        # themselves.
        self._solver_seed = seeds.stream(cfg, "solver")
        self._height_m = float(cfg.simulation.ue.height_m)
        self._power_dbm = float(cfg.simulation.antenna.power_rs)
        self._mdt = pd.read_parquet(cfg.data.output.mdt_file)
        self._centres: np.ndarray | None = None

        scene, delivered = scene_module.load(SceneSpec.from_config(cfg))
        perturb.apply(scene, PerturbSpec.from_config(cfg), seeds.stream(cfg, "scene"))
        bounds = dataclasses.replace(
            delivered, max_z=max(delivered.max_z, scene_module.bounds_of(scene).max_z)
        )
        for problem in transmitter.validate(
            scene.mi_scene, bounds, self.space.cells, GridSpec.from_config(cfg).free_height_tol_m
        ):
            print(f"WARNING transmitter {problem}")
        radio.configure_arrays(scene, cfg)
        self._scene: Any | None = scene

    def __enter__(self) -> Evaluator:
        """Return the evaluator, ready to score."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Drop the scene, so its GPU memory is released here and not later."""
        self.close()

    def close(self) -> None:
        """Release the scene. Evaluating afterwards raises."""
        self._scene = None

    @property
    def band_labels(self) -> tuple[str, ...]:
        """Band names in the radio map's band-axis order."""
        return tuple(band.name for band in self._bands)

    @property
    def tile_centres(self) -> np.ndarray | None:
        """The solver's tile centres from the last evaluation, for the archive."""
        return self._centres

    @property
    def scenario_id(self) -> str:
        """The scenario every evaluation here belongs to."""
        return str(self._manifest["scenario_id"])

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Ray-trace this tilt vector and score it.

        Raises:
            RuntimeError: When the evaluator has been closed.
            ValueError: When the vector leaves the box; see
                :meth:`src.optim.space.TiltSpace.to_cells`.
        """
        if self._scene is None:
            raise RuntimeError("this Evaluator is closed; build a new one to evaluate again")

        tilt_deg = np.asarray(tilt_deg, dtype=float).reshape(-1)
        cells = self.space.to_cells(tilt_deg)

        # Timed here rather than taken from solve_band's returned elapsed. Dr.Jit
        # is lazy, so the solver call returns before the map is computed and the
        # work lands on the first read of `radio_map.rss` — which happens after
        # that timer has already stopped. Trusting it under-reports the cost of
        # an evaluation by roughly an order of magnitude, and a budget planned
        # against it would be wrong by the same factor.
        started = time.perf_counter()
        maps: list[np.ndarray] = []
        for band in self._bands:
            rsrp, _elapsed, centres, _radio_map = radio.solve_band(
                self._scene,
                cells,
                band,
                self._solver,
                self._materials,
                self._material_seed,
                self._solver_seed,
                self._grid_meta,
                self._height_m,
                self._power_dbm,
            )
            maps.append(rsrp)
            self._centres = centres
        seconds = time.perf_counter() - started

        stacked = np.stack(maps)
        kpi = evaluate_kpis(stacked, self.band_labels, self._mdt, self.cfg)

        self.n_calls += 1
        self.total_seconds += seconds
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=kpi,
            seconds=seconds,
            rsrp=stacked if self.keep_rsrp else None,
        )

    def write_radio_map(self, path: str | Path, result: EvaluationResult) -> Path:
        """Archive one result's radio map in the schema ``radio.solve`` writes.

        Same keys, same axis order, same provenance fields, so the KPIs, the
        EDA plots and :mod:`src.data` read an optimized map exactly as they read
        the baseline one.

        Raises:
            ValueError: When the result carries no map, which means the
                evaluator was built with ``keep_rsrp=False``.
        """
        if result.rsrp is None:
            raise ValueError(
                "this result carries no radio map. Build the Evaluator with keep_rsrp=True, "
                "or re-evaluate the winning tilt with one that does."
            )

        cells = self.space.to_cells(result.tilt_deg)
        solver = self._solver
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rsrp_dbm=result.rsrp.astype(np.float32),
            band_hz=np.array([band.frequency_hz for band in self._bands]),
            band_label=np.array(self.band_labels),
            tilt_deg=np.array(
                [[cell.tilt_for(band.name).baseline_deg for cell in cells] for band in self._bands]
            ),
            tx_name=np.array([cell.name for cell in cells]),
            origin_x=self._grid_meta["origin_x"],
            origin_y=self._grid_meta["origin_y"],
            tile_size_m=self._grid_meta["tile_size_m"],
            n_cols=self._grid_meta["n_cols"],
            n_rows=self._grid_meta["n_rows"],
            ue_height_m=self._height_m,
            scenario_id=self.scenario_id,
            samples_per_tx=solver.samples_per_tx,
            max_depth=solver.max_depth,
            los=solver.los,
            specular_reflection=solver.specular_reflection,
            diffuse_reflection=solver.diffuse_reflection,
            refraction=solver.refraction,
            diffraction=solver.diffraction,
            edge_diffraction=solver.edge_diffraction,
            diffraction_lit_region=solver.diffraction_lit_region,
            rr_depth=solver.rr_depth,
            rr_prob=solver.rr_prob,
            solver_seed=self._solver_seed,
            temperature_k=solver.temperature_k,
            bandwidth_hz=np.array([band.bandwidth_hz for band in self._bands]),
            power_dbm=self._power_dbm,
            tile_centre=self._centres,
        )
        return path


def _read_manifest(cfg: DictConfig) -> dict[str, Any]:
    """Read the scenario manifest and check it describes this config.

    Raises:
        FileNotFoundError: When the scenario stage has not been run.
        ValueError: When the manifest is for a different scenario, which means
            the config changed after the UEs were drawn. Every KPI would then
            score a map against a population that does not belong to it.
    """
    path = Path(cfg.simulation.output.manifest_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"No scenario manifest at {path}. Run `task simulation:scenario` first."
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = scenario_module.scenario_id(cfg)
    if manifest["scenario_id"] != expected:
        raise ValueError(
            f"{path} describes scenario {manifest['scenario_id']}, but this config is "
            f"{expected}. Re-run `task simulation` before optimizing."
        )
    return manifest
