"""Turn a tilt vector into a KPI vector by ray tracing.

The expensive half of every optimization run, and the reason a run is feasible
at all: the scene and the antenna arrays do not depend on tilt, so they are
built once at construction and reused for every candidate.
Only the transmitters are rebuilt per evaluation, which is what
:func:`src.simulation.radio.solve_band` already does.

:class:`ObjectiveEvaluator` is the seam the search depends on. Anything mapping
a tilt vector to an :class:`EvaluationResult` satisfies it — the ray tracer here
or a stub in a test — so a search can be exercised without a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.optim.objective import KpiVector, evaluate_kpis
from src.optim.space import TiltSpace
from src.simulation import radio, seeds, transmitter
from src.simulation import scene as scene_module
from src.simulation.grid import GridSpec
from src.simulation.scene import SceneSpec


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate, scored.

    Attributes:
        tilt_deg: The vector evaluated, in :class:`~src.optim.space.TiltSpace`
            dimension order.
        kpi: Its score on all four KPIs.
        seconds: Wall clock for the ray tracing, summed over bands, measured
            around the point the maps are actually materialised. Excludes
            scoring and anything the caller does, so a run can report
            simulator time apart from model time.
        rsrp: The radio map, ``[n_band, n_tx, n_rows, n_cols]`` in dBm, or None
            when the evaluator was asked not to retain it. Every map of a long
            run does not fit in memory and the run does not need them.
        sinr: The solver's SINR in dB, same shape, retained alongside ``rsrp``.
    """

    tilt_deg: np.ndarray
    kpi: KpiVector
    seconds: float
    rsrp: np.ndarray | None = None
    sinr: np.ndarray | None = None


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

    Construction loads the scene, attaches the antenna arrays and checks the
    masts still stand on open ground — everything :func:`src.simulation.radio.solve` does per call
    that does not depend on tilt. It is a large fixed fraction of what one
    evaluation costs, so paying it per candidate would add roughly half again
    to every point in the run.

    Use it as a context manager. It holds GPU and scene state, and releasing
    that when the run ends rather than whenever the collector notices is what
    lets a long process or a service scope it predictably.

    Attributes:
        cfg: The composed config, read for everything below.
        keep_rsrp: Whether each result carries its radio maps. False by default
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
        self._manifest = radio.read_manifest(cfg)
        self._grid_meta = self._manifest["grid"]
        self._bands = tuple(
            radio.Band.from_config(entry) for entry in cfg.simulation.radio_map.bands
        )
        self._solver = radio.SolverSpec.from_config(cfg)
        # Shared seed: common Monte-Carlo noise cancels, so KPI *differences* are much cleaner.
        self._solver_seed = seeds.stream(cfg, "solver")
        self._height_m = float(cfg.simulation.ue.height_m)
        self._power_dbm = float(cfg.simulation.antenna.power_rs)
        self._mdt = pd.read_parquet(cfg.data.output.mdt_file)
        self._centres: np.ndarray | None = None

        scene, bounds = scene_module.load(SceneSpec.from_config(cfg))
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

        # Time here, not from solve_band's elapsed: Dr.Jit is lazy, so work
        # lands on the first `.rss` read — after solve_band's timer has stopped.
        started = time.perf_counter()
        rsrp_maps, sinr_maps = [], []
        for band in self._bands:
            rsrp, sinr, _elapsed, self._centres, _radio_map = radio.solve_band(
                self._scene,
                cells,
                band,
                self._solver,
                self._solver_seed,
                self._grid_meta,
                self._height_m,
                self._power_dbm,
            )
            rsrp_maps.append(rsrp)
            sinr_maps.append(sinr)
        seconds = time.perf_counter() - started

        rsrp, sinr = np.stack(rsrp_maps), np.stack(sinr_maps)
        kpi = evaluate_kpis(rsrp, sinr, self.band_labels, self._mdt, self.cfg)

        self.n_calls += 1
        self.total_seconds += seconds
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=kpi,
            seconds=seconds,
            rsrp=rsrp if self.keep_rsrp else None,
            sinr=sinr if self.keep_rsrp else None,
        )

    def write_radio_map(self, path: str | Path, result: EvaluationResult) -> Path:
        """Archive one result's radio map with :func:`src.simulation.radio.write_radio_map`.

        Same schema as the baseline map, so the KPIs, the plots and
        :mod:`src.data` read an optimized map exactly as they read that one.

        Raises:
            ValueError: When the result carries no map, which means the
                evaluator was built with ``keep_rsrp=False``.
        """
        if result.rsrp is None or result.sinr is None:
            raise ValueError(
                "this result carries no radio map. Build the Evaluator with keep_rsrp=True, "
                "or re-evaluate the winning tilt with one that does."
            )

        return radio.write_radio_map(
            path,
            rsrp=result.rsrp,
            sinr=result.sinr,
            bands=self._bands,
            cells=self.space.to_cells(result.tilt_deg),
            grid_meta=self._grid_meta,
            solver_spec=self._solver,
            solver_seed=self._solver_seed,
            height_m=self._height_m,
            power_dbm=self._power_dbm,
            scenario_id=self.scenario_id,
            centres=self._centres,
        )
