"""One run: which solutions get published, and what it writes.

No Sionna-RT and no GPU. The evaluator is a stub, which is what the
:class:`~src.optim.evaluator.ObjectiveEvaluator` protocol seam is for, so the
whole search-publish-archive path runs in a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import runs as run_store
from src.optim.evaluator import EvaluationResult
from src.optim.objective import MEASURE_NAMES, KpiVector
from src.optim.report import choose
from src.optim.run import run
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
    "kpi": {
        "hole_dbm": -120.0,
        "overlap_margin_db": 6.0,
        "capacity": {"band_preference": ["high", "low"]},
    },
    "optim": {
        # `rule` rather than `turbo`: deterministic, no model, and it still
        # exercises the whole publish path.
        "method": {"name": "rule", "n_steps": 5, "n_rounds": 2},
        "output": {"dir": "", "deliverable_dir": "", "save_radio_map": False},
        "n_solutions": 4,
        "seed": 0,
    },
}


@dataclass
class StubEvaluator:
    """Stands in for the ray tracer, scoring a tilt vector by a fixed function."""

    space: TiltSpace
    seen: list[np.ndarray] = field(default_factory=list)
    keep_rsrp: bool = False
    scenario_id: str = "scn_test"
    archived: list[np.ndarray] = field(default_factory=list)

    def __enter__(self) -> StubEvaluator:
        """A context manager, because the run uses one."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """No scene and no GPU memory to release."""
        return None

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Score one vector, and record that it was solved."""
        tilt_deg = np.asarray(tilt_deg, dtype=float)
        self.seen.append(tilt_deg.copy())
        unit = (tilt_deg - self.space.lower) / (self.space.upper - self.space.lower)
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=KpiVector(
                hole_rate=float(np.mean((unit - 0.35) ** 2)),
                overlap_rate=float(np.mean(unit) * 0.5),
                overlap_neighbor_mean=float(np.mean(unit) * 1.5),
                weak_rate=float(np.mean((unit - 0.25) ** 2)),
                rsrp_p05_dbm=float(-120.0 + 20.0 * np.mean(unit)),
                rsrp_p50_dbm=float(-100.0 + 20.0 * np.mean(unit)),
                sinr_p05_db=float(-5.0 + 10.0 * np.mean(unit)),
                sinr_p50_db=float(5.0 + 10.0 * np.mean(unit)),
                served_rate=float(1.0 - np.mean((unit - 0.75) ** 2)),
                load_imbalance=float(0.5 * np.mean(unit)),
                objective=float(1.0 - np.mean((unit - 0.35) ** 2)),
            ),
            seconds=1.0,
            rsrp=np.zeros((1, 1, 1, 1)) if self.keep_rsrp else None,
        )

    def write_radio_map(self, path: str | Path, result: EvaluationResult) -> Path:
        """Archive the winner's map, refusing a result that carries none.

        The refusal is the point: it is what proves the run flipped ``keep_rsrp``
        before re-solving, rather than archiving an empty map.
        """
        if result.rsrp is None:
            raise ValueError("this result carries no radio map")
        self.archived.append(result.tilt_deg.copy())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        return Path(path)


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


@pytest.fixture
def stub(cfg, space, monkeypatch) -> StubEvaluator:
    """The evaluator :func:`src.optim.run.run` will build."""
    import src.optim.run as run_module

    evaluator = StubEvaluator(space)
    monkeypatch.setattr(run_module, "Evaluator", lambda cfg, keep_rsrp=False: evaluator)
    return evaluator


def _kpi(objective: float, rng: np.random.Generator | None = None) -> KpiVector:
    """One KPI vector with the given objective; the rest is filler to rank around."""
    draw = (lambda: 0.5) if rng is None else (lambda: float(rng.random()))
    return KpiVector(
        hole_rate=draw(),
        overlap_rate=draw(),
        overlap_neighbor_mean=draw(),
        weak_rate=draw(),
        rsrp_p05_dbm=-110.0,
        rsrp_p50_dbm=-95.0,
        sinr_p05_db=-3.0,
        sinr_p50_db=8.0,
        served_rate=draw(),
        load_imbalance=draw(),
        objective=objective,
    )


def _kpis(count: int) -> list[KpiVector]:
    """A spread of KPI vectors to rank, the first a middling incumbent."""
    rng = np.random.default_rng(0)
    return [_kpi(0.5), *(_kpi(float(rng.random()), rng) for _ in range(count - 1))]


def test_choose_always_publishes_the_incumbent_first() -> None:
    """Every published delta is measured against it, so it has to be offered."""
    assert choose(_kpis(24), 4)[0] == 0


def test_choose_respects_the_budget_and_never_repeats() -> None:
    """A required row that also ranks high is offered once."""
    picks = choose(_kpis(24), 4, keep=(0, 3))
    assert len(picks) == 4
    assert len(set(picks)) == len(picks)


def test_choose_never_drops_a_required_row_to_fit_the_budget() -> None:
    """The budget is a preference; the incumbent and the winner are not."""
    picks = choose(_kpis(24), 1, keep=(0, 3))
    assert set(picks) == {0, 3}


def test_choose_fills_the_budget_by_objective() -> None:
    """After the incumbent, nothing left out outscores anything offered.

    Ranked by the objective that selects the winner, or the shortlist would
    disagree with the recommendation printed beside it.
    """
    kpis = _kpis(24)
    picks = choose(kpis, 6)
    values = np.array([kpi.objective for kpi in kpis])

    offered = values[picks[1:]]
    left_out = np.delete(values, picks)
    assert np.all(np.diff(offered) <= 0)
    assert offered.min() >= left_out.max()


def test_one_run_searches_publishes_and_archives(cfg, space, stub) -> None:
    """A single command leaves a directory a comparison can load."""
    history, directory = run(cfg)
    loaded = run_store.load(directory)

    # save_radio_map is off, so nothing is re-solved.
    assert len(stub.seen) == len(history)
    assert loaded.meta["best_kpi"] == history.results[loaded.best_index].kpi.as_dict()
    # The two-phase flags are gone: nothing is a prediction any more.
    assert "verified" not in loaded.meta
    assert "search_seconds" not in loaded.meta
    assert "source" not in loaded.history


def test_every_published_kpi_came_from_the_evaluator(cfg, space, stub) -> None:
    """No row is a prediction, so the stub must have solved every one it offers."""
    _history, directory = run(cfg)
    published = pd.read_parquet(directory / "solutions.parquet")

    solved = {tuple(np.round(vector, 9)) for vector in stub.seen}
    for _, row in published.iterrows():
        tilts = tuple(np.round(row[list(space.parameter_names)].to_numpy(dtype=float), 9))
        assert tilts in solved
    for name in MEASURE_NAMES:
        assert f"predicted_{name}" not in published
        assert f"error_{name}" not in published


def test_the_recommended_row_is_the_run_json_winner(cfg, stub) -> None:
    """`choose` and `best_index` must not be able to name different solutions."""
    _history, directory = run(cfg)
    published = pd.read_parquet(directory / "solutions.parquet")
    meta = run_store.load(directory).meta

    assert published["recommended"].sum() == 1
    assert published.loc[published["recommended"], "iteration"].item() == meta["best_iteration"]


def test_the_run_publishes_a_shortlist_to_choose_from(cfg, stub) -> None:
    """The deliverable offers the runners-up, and always names one solution."""
    run(cfg)
    shortlist = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/solutions_rule.csv")
    options = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/tilt_options_rule.csv")

    assert shortlist["recommended"].sum() == 1
    for name in MEASURE_NAMES:
        assert f"delta_{name}" in shortlist
    # One tilt table per offered solution, each covering every cell-band pair.
    assert set(options["solution"]) == set(shortlist["solution"])
    assert len(options) == len(shortlist) * 6


def test_the_incumbent_delta_compares_two_measurements(cfg, space, stub) -> None:
    """Both sides of the subtraction come from the evaluator, not one of each."""
    _history, directory = run(cfg)
    loaded = run_store.load(directory)

    assert (
        loaded.incumbent_kpi.as_dict()
        == StubEvaluator(space).evaluate(space.baseline).kpi.as_dict()
    )


def test_archiving_the_winners_map_costs_exactly_one_extra_solve(cfg, stub) -> None:
    """The winner is re-solved once for its map, and that solve is not an evaluation."""
    cfg.optim.output.save_radio_map = True
    history, directory = run(cfg)
    best_index = history.best_index()

    assert len(stub.seen) == len(history) + 1
    assert len(stub.archived) == 1
    assert np.allclose(stub.archived[0], history.results[best_index].tilt_deg)
    assert run_store.load(directory).meta["n_evaluations"] == len(history)
