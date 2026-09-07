"""The search loops and the run artifacts, against a stub evaluator.

No ray tracing and no GPU: every test here drives the real loops through the
:class:`~src.optim.evaluator.ObjectiveEvaluator` protocol with an analytic
stand-in. That is what the protocol seam is for, and it is what keeps the loop
logic, the budget accounting and the artifact schema covered in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.optim.evaluator import EvaluationResult
from src.optim.history import History, LocalRunWriter, write_run
from src.optim.objective import KPI_NAMES, KpiVector
from src.optim.search import run_search
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
        "tolerance": {
            "hole_rate": 0.001,
            "overlap_rate": 0.001,
            "mean_overlap_neighbors": 0.01,
            "band_priority_score": 0.001,
            "weak_rate": 0.001,
        }
    },
    "bo": {
        "method": "mobo",
        "budget": {"n_init": 4, "n_iter": 2, "batch_size": 2},
        "rule": {"n_steps": 3, "n_rounds": 1},
        "seed": 0,
    },
}


@dataclass
class StubEvaluator:
    """Scores a tilt vector analytically, and records what it was asked.

    Satisfies :class:`~src.optim.evaluator.ObjectiveEvaluator` structurally, so
    the search cannot tell it from the ray tracer.
    """

    space: TiltSpace
    seen: list[np.ndarray] = field(default_factory=list)

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Score one vector. Optimal tilt falls at 30% of each band's range."""
        tilt_deg = np.asarray(tilt_deg, dtype=float)
        self.seen.append(tilt_deg.copy())
        unit = (tilt_deg - self.space.lower) / (self.space.upper - self.space.lower)
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=KpiVector(
                hole_rate=float(np.mean((unit - 0.3) ** 2)),
                overlap_rate=float(np.mean(unit) * 0.4),
                mean_overlap_neighbors=float(np.mean(unit) * 2.0),
                band_priority_score=float(1.0 - np.mean((unit - 0.8) ** 2)),
                weak_rate=float(np.mean((unit - 0.2) ** 2)),
            ),
            seconds=0.0,
        )


@pytest.fixture
def cfg():
    """A composed config carrying only what the search reads."""
    return OmegaConf.create(_CONFIG)


@pytest.fixture
def evaluator(cfg) -> StubEvaluator:
    """A stub over a three-cell, two-band space."""
    return StubEvaluator(TiltSpace.from_config(cfg))


@pytest.mark.parametrize("method", ["mobo", "random", "rule"])
def test_every_method_starts_from_the_committed_incumbent(cfg, evaluator, method) -> None:
    """Row zero is always the deployed configuration.

    A comparison then always has its reference, and a run that improves on
    nothing still reports the baseline rather than an empty table.
    """
    history = run_search(evaluator, cfg, method)
    frame = history.frame()
    assert frame.loc[0, "phase"] == "incumbent"
    assert np.allclose(evaluator.seen[0], evaluator.space.baseline)
    assert np.allclose(
        frame.loc[0, list(evaluator.space.parameter_names)].to_numpy(dtype=float),
        evaluator.space.baseline,
    )


@pytest.mark.parametrize("method", ["mobo", "random", "rule"])
def test_no_method_ever_proposes_a_tilt_outside_the_box(cfg, evaluator, method) -> None:
    """Bounds are a hard constraint, never a relaxation the search may soften."""
    run_search(evaluator, cfg, method)
    proposals = np.array(evaluator.seen)
    assert (proposals >= evaluator.space.lower - 1e-9).all()
    assert (proposals <= evaluator.space.upper + 1e-9).all()


@pytest.mark.parametrize("method", ["mobo", "random"])
def test_the_ax_methods_spend_exactly_their_budget(cfg, evaluator, method) -> None:
    """``n_init + n_iter`` searched evaluations, plus the incumbent."""
    history = run_search(evaluator, cfg, method)
    assert len(history) == 1 + cfg.bo.budget.n_init + cfg.bo.budget.n_iter


def test_the_ax_methods_record_which_generator_made_each_point(cfg, evaluator) -> None:
    """Separating the Sobol phase from the model phase needs this provenance."""
    frame = run_search(evaluator, cfg, "mobo").frame()
    assert frame.loc[0, "generation_node"] == "attached"
    assert set(frame["phase"]) == {"incumbent", "init", "search"}
    assert frame.loc[frame["phase"] == "init", "generation_node"].eq("Sobol").all()


def test_random_search_never_reaches_a_model(cfg, evaluator) -> None:
    """What makes it the control: same loop, same budget, no surrogate."""
    frame = run_search(evaluator, cfg, "random").frame()
    assert set(frame["generation_node"]) <= {"attached", "Sobol"}


def test_the_rule_sweep_moves_a_whole_band_together(cfg, evaluator) -> None:
    """The operator heuristic: every cell on a band points alike."""
    run_search(evaluator, cfg, "rule")
    space = evaluator.space
    for proposal in evaluator.seen:
        for band in space.band_names:
            axis = [i for i, (_cell, name) in enumerate(space.pairs) if name == band]
            assert len(set(np.round(proposal[axis], 9))) == 1


def test_the_rule_sweep_stays_within_its_declared_budget(cfg, evaluator) -> None:
    """``n_band * n_steps * n_rounds`` is the ceiling; repeats are skipped."""
    history = run_search(evaluator, cfg, "rule")
    ceiling = len(evaluator.space.band_names) * cfg.bo.rule.n_steps * cfg.bo.rule.n_rounds
    assert len(history) <= 1 + ceiling


def test_an_unknown_method_names_the_registered_ones(cfg, evaluator) -> None:
    """Adding a method is a registry entry, not an edit to a dispatch chain."""
    with pytest.raises(KeyError, match="mobo"):
        run_search(evaluator, cfg, "annealing")


def test_history_frame_carries_provenance_kpis_and_every_tilt(cfg, evaluator) -> None:
    """The schema all three methods share."""
    frame = run_search(evaluator, cfg, "random").frame()
    expected = {"iteration", "phase", "generation_node", "seconds", "on_pareto"}
    assert expected <= set(frame.columns)
    assert set(KPI_NAMES) <= set(frame.columns)
    assert set(evaluator.space.parameter_names) <= set(frame.columns)
    assert frame[list(KPI_NAMES)].notna().all().all()


def test_the_tilt_table_reports_delta_against_the_incumbent(cfg, evaluator) -> None:
    """The deliverable. Delta is reported, never optimized."""
    history = run_search(evaluator, cfg, "random")
    best = history.results[history.best_index(cfg)]
    table = history.tilt_table(best.tilt_deg)
    assert len(table) == evaluator.space.n_dim
    assert np.allclose(table["current_tilt_deg"], evaluator.space.baseline)
    assert np.allclose(
        table["delta_tilt_deg"], table["optimized_tilt_deg"] - table["current_tilt_deg"]
    )


def test_write_run_persists_every_artifact(cfg, evaluator, tmp_path: Path) -> None:
    """A run directory must be readable without the session that produced it."""
    history = run_search(evaluator, cfg, "random")
    best_index = history.best_index(cfg)
    written = write_run(
        history,
        LocalRunWriter(tmp_path),
        cfg,
        method="random",
        best_index=best_index,
        extra={"scenario_id": "scn_test"},
    )
    assert set(written) == {"history", "pareto", "best_tilt", "run"}
    for locator in written.values():
        assert Path(locator).is_file()

    import json

    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["scenario_id"] == "scn_test"
    assert run["n_evaluations"] == len(history)
    assert run["best_kpi"] == history.results[best_index].kpi.as_dict()
    assert run["config"]["kpi"]["tolerance"]["hole_rate"] == 0.001


def test_set_generation_nodes_rejects_a_length_mismatch(evaluator) -> None:
    """A short list would misattribute every row after the gap."""
    history = History(evaluator.space)
    history.append(evaluator.evaluate(evaluator.space.baseline), phase="incumbent")
    with pytest.raises(ValueError, match="1 evaluations"):
        history.set_generation_nodes(["Sobol", "MBM"])


def test_an_empty_history_has_nothing_to_tabulate(evaluator) -> None:
    """Better than a zero-row frame that reads as a run which found nothing."""
    with pytest.raises(ValueError, match="no evaluations"):
        History(evaluator.space).frame()
