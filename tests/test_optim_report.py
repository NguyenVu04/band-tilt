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
from src.optim.objective import KPI_NAMES, KpiVector, scores
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
    "kpi": {"tolerance": dict.fromkeys(KPI_NAMES, 0.001), "weights": dict.fromkeys(KPI_NAMES, 1.0)},
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
                served_ratio=float(1.0 - np.mean((unit - 0.75) ** 2)),
                weak_rate=float(np.mean((unit - 0.25) ** 2)),
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


def _kpis(count: int) -> list[KpiVector]:
    """A spread of KPI vectors to rank."""
    rng = np.random.default_rng(0)
    kpis = [KpiVector(0.5, 0.5, 0.5, 0.5)]
    for _ in range(count - 1):
        kpis.append(KpiVector(*(float(value) for value in rng.random(4))))
    return kpis


def test_choose_always_publishes_the_incumbent_first(cfg) -> None:
    """Every published delta is measured against it, so it has to be offered."""
    assert choose(_kpis(24), cfg, 4)[0] == 0


def test_choose_respects_the_budget_and_never_repeats(cfg) -> None:
    """A required row that also ranks high is offered once."""
    picks = choose(_kpis(24), cfg, 4, keep=(0, 3))
    assert len(picks) == 4
    assert len(set(picks)) == len(picks)


def test_choose_never_drops_a_required_row_to_fit_the_budget(cfg) -> None:
    """The budget is a preference; the incumbent and the winner are not."""
    picks = choose(_kpis(24), cfg, 1, keep=(0, 3))
    assert set(picks) == {0, 3}


def test_choose_fills_the_budget_by_score(cfg) -> None:
    """After the incumbent, nothing left out outscores anything offered."""
    kpis = _kpis(24)
    picks = choose(kpis, cfg, 6)
    values = scores(kpis, cfg)

    offered = values[picks[1:]]
    left_out = np.delete(values, picks)
    assert np.all(np.diff(offered) <= 0)
    assert offered.min() >= left_out.max()


def test_one_run_searches_publishes_and_archives(cfg, space, stub) -> None:
    """A single command leaves a directory a comparison can load."""
    history, directory = run(cfg)
    loaded = run_store.load(directory)

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
    for name in KPI_NAMES:
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
    for name in KPI_NAMES:
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


def test_the_winners_map_costs_exactly_one_extra_solve(cfg, stub) -> None:
    """Archiving re-solves the winner alone, on the evaluator already in hand."""
    cfg.optim.output.save_radio_map = True
    history, directory = run(cfg)

    assert len(stub.seen) == len(history) + 1
    # The extra solve is the winner, and it is not recorded as an evaluation.
    winner = history.results[history.best_index(cfg)].tilt_deg
    assert np.allclose(stub.seen[-1], winner)
    assert len(stub.archived) == 1
    assert np.allclose(stub.archived[0], winner)
    assert run_store.load(directory).meta["n_evaluations"] == len(history)
