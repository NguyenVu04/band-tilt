"""Turn a tilt vector into a KPI vector by ray tracing.

The expensive half of every optimization run, and the reason a run is feasible
at all: the scene and the antenna arrays do not depend on tilt, so they are
built once at construction and reused for every candidate.
Only the transmitters are rebuilt per evaluation, which is what
:func:`src.simulation.radio.solve_bands` already does.

The served rate counts every UE in ``data.output.ue_file``.

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

from src.kpi.capacity import CapacitySpec
from src.optim.objective import KpiVector, evaluate_kpis
from src.optim.space import TiltSpace
from src.simulation import radio, seeds


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate, scored.

    Attributes:
        tilt_deg: The vector evaluated, in :class:`~src.optim.space.TiltSpace`
            dimension order.
        kpi: Its measures.
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
    masts still stand on open ground — the setup
    :func:`src.simulation.radio.solve_band` needs but that no tilt changes.
    Hoisting it out of the loop is why a search pays for it once rather than
    once per candidate.

    Use it as a context manager. It holds GPU and scene state, and releasing
    that when the run ends rather than whenever the collector notices is what
    lets a long process or a service scope it predictably.

    Attributes:
        cfg: The composed config, read for everything below.
        keep_rsrp: Whether each result carries its radio maps. False by default
            because a run keeps every result and the maps do not fit.
        solver_seed: The ray tracer's Monte-Carlo seed; the ``solver`` stream of
            ``simulation.seed`` when None. Reassign it to re-measure a
            configuration under other solver noise.
    """

    cfg: DictConfig
    keep_rsrp: bool = False
    solver_seed: int | None = None

    space: TiltSpace = field(init=False)
    _setup: radio.RadioSetup = field(init=False, repr=False)
    _capacity: CapacitySpec = field(init=False, repr=False)
    _ue: pd.DataFrame = field(init=False, repr=False)
    _centres: np.ndarray | None = field(init=False, default=None, repr=False)
    _scene: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Build the scene and everything else that does not depend on tilt.

        Raises:
            FileNotFoundError: When the scenario manifest or ``data.output.ue_file``
                is missing.
        """
        cfg = self.cfg
        self.space = TiltSpace.from_config(cfg)
        self._setup = radio.RadioSetup.from_config(cfg)
        # Shared seed: common Monte-Carlo noise cancels, so KPI *differences* are much cleaner.
        if self.solver_seed is None:
            self.solver_seed = seeds.stream(cfg, "solver")
        ue_file = Path(cfg.data.output.ue_file)
        if not ue_file.is_file():
            raise FileNotFoundError(f"No UE table at {ue_file}. Run `task preprocess` first.")
        self._ue = pd.read_parquet(ue_file)
        self._capacity = CapacitySpec.from_config(cfg, self.band_labels, len(self.space.cells))
        self._scene = radio.load_scene(cfg, self.space.cells)

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
        return tuple(band.name for band in self._setup.bands)

    @property
    def scenario_id(self) -> str:
        """The scenario every evaluation here belongs to."""
        return self._setup.scenario_id

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Ray-trace this tilt vector and score it.

        Raises:
            RuntimeError: When the evaluator has been closed.
            ValueError: When the vector leaves the box; see
                :meth:`src.optim.space.TiltSpace.to_cells`.
        """
        if self._scene is None:
            raise RuntimeError("this Evaluator is closed; build a new one to evaluate again")

        # A copy, so the result never aliases an array the caller goes on to mutate.
        tilt_deg = np.array(tilt_deg, dtype=float).reshape(-1)
        cells = self.space.to_cells(tilt_deg)

        # Timed around the whole loop, so per-band scene setup counts as simulator time.
        started = time.perf_counter()
        rsrp, sinr, self._centres, _elapsed = radio.solve_bands(
            self._scene, cells, self._setup, int(self.solver_seed)
        )
        seconds = time.perf_counter() - started

        kpi = evaluate_kpis(rsrp, sinr, self.band_labels, self._ue, self.cfg, spec=self._capacity)

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
        if result.rsrp is None or result.sinr is None or self._centres is None:
            raise ValueError(
                "this result carries no radio map. Build the Evaluator with keep_rsrp=True, "
                "or re-evaluate the winning tilt with one that does."
            )

        setup = self._setup
        return radio.write_radio_map(
            path,
            rsrp=result.rsrp,
            sinr=result.sinr,
            bands=setup.bands,
            cells=self.space.to_cells(result.tilt_deg),
            grid_meta=setup.grid_meta,
            solver_spec=setup.solver,
            solver_seed=int(self.solver_seed),
            height_m=setup.height_m,
            power_dbm=setup.power_dbm,
            scenario_id=setup.scenario_id,
            centres=self._centres,
        )
