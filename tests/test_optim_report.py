"""Phase two: which solutions get re-solved, and what the report writes.

No Sionna-RT and no GPU. The report's own evaluator is a stub, which is what
the :class:`~src.optim.evaluator.ObjectiveEvaluator` protocol seam is for, and
the search phase is replayed from a hand-built frame rather than run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import runs as run_store
from src.optim.evaluator import EvaluationResult
from src.optim.history import History, LocalRunWriter, write_run
from src.optim.objective import KPI_NAMES, RAY_TRACED, SURROGATE, KpiVector
from src.optim.report import choose, crowding_distance, report
from src.optim.space import TiltSpace

_CELLS = [
    {
        "name": f"c{index}",
        "x": float(index),
        "y": 0.0,
        "z": 30.0,
        "azimuth_deg": 120.0 * index,
        "tilt": {
            "high": {"baseline_deg": 8.0, "bounds_deg": [0.0, 16.0]},
            "low": {"baseline_deg": 4.0, "bounds_deg": [0.0, 12.0]},
        },
    }
    for index in range(3)
]

_CONFIG = {
    "simulation": {
        "radio_map": {"bands": [{"name": "high"}, {"name": "low"}]},
        "transmitters": {"cells": _CELLS},
    },
    "kpi": {"tolerance": dict.fromkeys(KPI_NAMES, 0.001)},
    "optim": {
        "method": {"name": "mobo"},
        "output": {"dir": "", "deliverable_dir": "", "save_radio_map": False},
        "report": {"n_solutions": 4, "run_dir": ""},
        "seed": 0,
    },
}


@dataclass
class StubEvaluator:
    """Stands in for the ray tracer, and scores differently from the search.

    Deliberately not the surrogate's function: the point of phase two is that
    the measurement can disagree with the prediction, and a stub that agreed
    would let a report that never re-solved anything still pass.
    """

    space: TiltSpace
    seen: list[np.ndarray] = field(default_factory=list)
    keep_rsrp: bool = False
    scenario_id: str = "scn_test"

    def __enter__(self) -> StubEvaluator:
        """A context manager, because the report phase uses one."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """No scene and no GPU memory to release."""
        return None

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Score one vector, shifted from the prediction by a fixed offset."""
        tilt_deg = np.asarray(tilt_deg, dtype=float)
        self.seen.append(tilt_deg.copy())
        unit = (tilt_deg - self.space.lower) / (self.space.upper - self.space.lower)
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=KpiVector(
                hole_rate=float(np.mean((unit - 0.35) ** 2)),
                overlap_rate=float(np.mean(unit) * 0.5),
                band_priority_score=float(1.0 - np.mean((unit - 0.75) ** 2)),
                weak_rate=float(np.mean((unit - 0.25) ** 2)),
            ),
            seconds=1.0,
        )


@pytest.fixture
def cfg(tmp_path) -> DictConfig:
    """A config whose output directories are the test's own."""
    config = OmegaConf.create(_CONFIG)
    config.optim.output.dir = str(tmp_path / "optim")
    config.optim.output.deliverable_dir = str(tmp_path / "deliverable")
    return config


@pytest.fixture
def space(cfg) -> TiltSpace:
    """The three-cell, two-band space the stub scores over."""
    return TiltSpace.from_config(cfg)


def _predicted(space: TiltSpace, count: int) -> list[KpiVector]:
    """A spread of KPI vectors, so the front has an interior to thin."""
    rng = np.random.default_rng(0)
    kpis = [KpiVector(0.5, 0.5, 0.5, 0.5)]
    for _ in range(count - 1):
        unit = rng.random(4)
        kpis.append(KpiVector(*(float(value) for value in unit)))
    return kpis


def _searched(space: TiltSpace, kpis: list[KpiVector]) -> pd.DataFrame:
    """The frame phase one writes, hand-built."""
    rng = np.random.default_rng(1)
    tilts = space.lower + rng.random((len(kpis), space.n_dim)) * (space.upper - space.lower)
    tilts[0] = space.baseline
    frame = pd.DataFrame(
        {
            "iteration": np.arange(len(kpis)),
            "phase": ["incumbent"] + ["search"] * (len(kpis) - 1),
            "generation_node": [""] * len(kpis),
            "source": [SURROGATE] * len(kpis),
            "seconds": np.full(len(kpis), 0.01),
        }
    )
    for name in KPI_NAMES:
        frame[name] = [getattr(kpi, name) for kpi in kpis]
    for index, column in enumerate(space.parameter_names):
        frame[column] = tilts[:, index]
    return frame


@pytest.fixture
def searched_run(cfg, space, tmp_path):
    """A phase-one run directory, written the way :mod:`src.optim.run` writes one."""
    kpis = _predicted(space, 24)
    frame = _searched(space, kpis)

    history = History(space=space)
    for position, row in frame.iterrows():
        history.append(
            EvaluationResult(
                tilt_deg=frame.loc[position, list(space.parameter_names)].to_numpy(dtype=float),
                kpi=kpis[position],
                seconds=0.01,
                source=SURROGATE,
            ),
            phase=str(row["phase"]),
        )

    directory = tmp_path / "optim" / "mobo" / "2026-09-09_00-00-00"
    write_run(history, LocalRunWriter(directory), cfg, method="mobo")
    return directory


def test_crowding_distance_scores_the_extremes_infinite() -> None:
    """The ends of every objective are what a spread has to keep."""
    values = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(values)
    assert np.isinf(distance[0]) and np.isinf(distance[2])
    assert np.isfinite(distance[1])


def test_crowding_distance_prefers_the_lonelier_solution() -> None:
    """A point with room around it carries more of the front than a crowded one.

    Rows 1 and 2 sit almost on top of each other. Row 2 is the one with the
    long gap to row 3, so it is the one worth keeping when only one of the pair
    can be afforded — which is exactly the thinning the report budget does.
    """
    values = np.array([[0.0, 1.0], [0.10, 0.90], [0.11, 0.89], [1.0, 0.0]])
    distance = crowding_distance(values)
    assert distance[2] > distance[1]
    assert np.isinf(distance[0]) and np.isinf(distance[3])


def test_a_front_of_two_is_all_extremes() -> None:
    """Nothing is interior, so nothing can be thinned."""
    assert np.isinf(crowding_distance(np.array([[0.0, 1.0], [1.0, 0.0]]))).all()


def test_choose_always_re_solves_the_incumbent_first(cfg, space) -> None:
    """Every reported delta is measured against it, so it cannot be a prediction."""
    picks = choose(_predicted(space, 24), cfg, 4)
    assert picks[0] == 0


def test_choose_respects_the_budget_and_never_repeats(cfg, space) -> None:
    """The incumbent is usually on the front too, and the priority pick always is."""
    picks = choose(_predicted(space, 24), cfg, 4)
    assert len(picks) == 4
    assert len(set(picks)) == len(picks)


def test_choose_keeps_two_even_when_asked_for_one(cfg, space) -> None:
    """Below two the budget would drop the pick the run goes on to recommend."""
    assert len(choose(_predicted(space, 24), cfg, 1)) == 2


def test_choose_only_offers_non_dominated_solutions(cfg, space) -> None:
    """A dominated solution is worse on every count than one already offered."""
    from src.optim.objective import pareto_mask

    kpis = _predicted(space, 24)
    front = set(np.flatnonzero(pareto_mask(kpis)).tolist())
    assert set(choose(kpis, cfg, 6)) <= front | {0}


def test_report_re_solves_and_marks_the_run_verified(cfg, searched_run, monkeypatch) -> None:
    """The run becomes reportable, and its KPIs come from the evaluator."""
    import src.optim.report as report_module

    space = TiltSpace.from_config(cfg)
    stub = StubEvaluator(space)
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    run = run_store.load(searched_run)

    assert run.meta["verified"] is True
    assert run.meta["n_solutions_verified"] == 4
    assert len(stub.seen) == 4
    # The verified rows sit after the searched ones, and the winner is one.
    assert run.history.loc[run.best_index, "source"] == RAY_TRACED
    assert (
        run.meta["best_kpi"]
        == stub.evaluate(
            run.history.loc[run.best_index, list(space.parameter_names)].to_numpy(dtype=float)
        ).kpi.as_dict()
    )


def test_report_keeps_the_search_trace(cfg, searched_run, monkeypatch) -> None:
    """One history holds both phases, told apart by source, not thrown away."""
    import src.optim.report as report_module

    stub = StubEvaluator(TiltSpace.from_config(cfg))
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    history = run_store.load(searched_run).history

    assert (history["source"] == SURROGATE).sum() == 24
    assert (history["source"] == RAY_TRACED).sum() == 4
    assert set(history.loc[history["source"] == RAY_TRACED, "phase"]) == {"verify"}


def test_reported_ray_tracing_time_excludes_the_search(cfg, searched_run, monkeypatch) -> None:
    """The surrogate spent milliseconds; reporting them as simulator time would lie."""
    import src.optim.report as report_module

    stub = StubEvaluator(TiltSpace.from_config(cfg))
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    run = run_store.load(searched_run)

    # Four solutions at one second each, and none of the 24 searched rows.
    assert run.ray_tracing_seconds == pytest.approx(4.0)
    assert run.meta["search_seconds"] == pytest.approx(24 * 0.01)


def test_the_incumbent_delta_compares_two_measurements(cfg, searched_run, monkeypatch) -> None:
    """Both sides of the subtraction come from the evaluator, not one of each."""
    import src.optim.report as report_module

    space = TiltSpace.from_config(cfg)
    stub = StubEvaluator(space)
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    run = run_store.load(searched_run)

    assert run.incumbent_kpi.as_dict() == stub.evaluate(space.baseline).kpi.as_dict()


def test_report_publishes_a_front_to_choose_from(cfg, searched_run, monkeypatch) -> None:
    """The deliverable offers the trade-offs, not only the priority pick."""
    import src.optim.report as report_module

    stub = StubEvaluator(TiltSpace.from_config(cfg))
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    scores = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/pareto_mobo.csv")
    options = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/tilt_options_mobo.csv")

    assert scores["recommended"].sum() == 1
    for name in KPI_NAMES:
        assert f"delta_{name}" in scores
    # One tilt table per offered solution, each covering every cell-band pair.
    assert set(options["solution"]) == set(scores["solution"])
    assert len(options) == len(scores) * 6


def test_pareto_verified_records_what_the_surrogate_got_wrong(
    cfg, searched_run, monkeypatch
) -> None:
    """The model's error on the solutions it recommended, measured every run."""
    import src.optim.report as report_module

    stub = StubEvaluator(TiltSpace.from_config(cfg))
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    solutions = pd.read_parquet(searched_run / "pareto_verified.parquet")

    assert len(solutions) == 4
    for name in KPI_NAMES:
        assert {name, f"predicted_{name}", f"error_{name}"} <= set(solutions.columns)
        assert np.allclose(
            solutions[f"error_{name}"], solutions[f"predicted_{name}"] - solutions[name]
        )


def test_a_searched_but_unreported_run_is_refused(searched_run) -> None:
    """Its KPIs are predictions, and a comparison would read them as measurements."""
    with pytest.raises(run_store.RunError, match="task optim:report"):
        run_store.load(searched_run)


def test_a_front_is_never_mixed_across_sources(cfg, searched_run, monkeypatch) -> None:
    """An optimistic prediction must not be able to dominate a measurement."""
    import src.optim.report as report_module

    stub = StubEvaluator(TiltSpace.from_config(cfg))
    monkeypatch.setattr(report_module, "Evaluator", lambda cfg, keep_rsrp=False: stub)

    report(cfg, searched_run)
    history = run_store.load(searched_run).history

    # Both sources carry a front of their own; neither is empty because a front
    # of one point is still a front.
    for source in (SURROGATE, RAY_TRACED):
        assert history.loc[history["source"] == source, "on_pareto"].sum() >= 1
