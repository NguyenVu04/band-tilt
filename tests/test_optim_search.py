"""The search loops and the run artifacts, against a stub evaluator.

No ray tracing and no GPU: every test here drives the real loops through the
:class:`~src.optim.evaluator.ObjectiveEvaluator` protocol with an analytic
stand-in. That is what the protocol seam is for, and it is what keeps the loop
logic, the budget accounting and the artifact schema covered in seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.optim.evaluator import EvaluationResult
from src.optim.history import History, LocalRunWriter, write_run
from src.optim.methods import run_search
from src.optim.methods.morbo.search import (
    RESTART,
    TrustRegion,
    _hv_scalarisation,
    _improvement,
    _local_rows,
    _recentre,
    _select,
    failure_tolerance,
    perturbation_probability,
)
from src.optim.objective import MEASURE_NAMES, KpiVector, hypervolume_contributions
from src.optim.space import TiltSpace
from tests.conftest import write_sectors

_SECTORS = [
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
    },
    "kpi": {
        "hole_dbm": -120.0,
        "overlap_margin_db": 6.0,
        "capacity": {"max_admission_utilisation": 0.8},
    },
    "optim": {
        "output": {
            "dir": "outputs/optim",
            "deliverable_dir": "reports/outputs",
        },
        "seed": 0,
        "tilt_resolution_deg": 0.1,
    },
}

# One block per method, as the ``optim/method`` config group supplies them.
_METHODS = {
    "morbo": {
        "name": "morbo",
        "budget": {"n_init": 4, "n_iter": 4, "batch_size": 2},
        "n_candidates": 64,
        "trust_region": {
            "count": 2,
            "length_init": 0.8,
            "length_min": 0.01,
            "improvement": 1e-3,
            "perturbed_dimensions": 20,
        },
    },
    "random": {"name": "random", "budget": {"n_init": 4, "n_iter": 2}},
}


@dataclass
class StubEvaluator:
    """Scores a tilt vector analytically, and records what it was asked.

    Satisfies :class:`~src.optim.evaluator.ObjectiveEvaluator` structurally, so
    the search cannot tell it from the ray tracer. Coverage peaks at 30 % of
    each range and separation at 70 %, so the two conflict.
    """

    space: TiltSpace
    seen: list[np.ndarray] = field(default_factory=list)

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Score one vector."""
        tilt_deg = np.asarray(tilt_deg, dtype=float)
        self.seen.append(tilt_deg.copy())
        unit = (tilt_deg - self.space.lower) / (self.space.upper - self.space.lower)
        return EvaluationResult(
            tilt_deg=tilt_deg,
            kpi=KpiVector(
                hole_rate=float(np.mean((unit - 0.3) ** 2)),
                overlap_rate=float(np.mean(unit) * 0.4),
                overlap_neighbor_mean=float(np.mean(unit) * 1.2),
                weak_rate=float(np.mean((unit - 0.2) ** 2)),
                rsrp_p05_dbm=float(-120.0 + 20.0 * np.mean(unit)),
                rsrp_p50_dbm=float(-100.0 + 20.0 * np.mean(unit)),
                sinr_p05_db=float(-5.0 + 10.0 * np.mean(unit)),
                sinr_p50_db=float(5.0 + 10.0 * np.mean(unit)),
                estimated_throughput_p05_mbps=float(np.mean(unit)),
                estimated_throughput_p50_mbps=float(2.0 * np.mean(unit)),
                estimated_throughput_mean_mbps=float(3.0 * np.mean(unit)),
                coverage_objective=float(1.0 - np.mean((unit - 0.3) ** 2)),
                separation_objective=float(1.0 - np.mean((unit - 0.7) ** 2)),
                throughput_objective=float(10.0 * (1.0 + np.mean(unit))),
            ),
            seconds=0.0,
        )


@pytest.fixture
def make_cfg(tmp_path):
    """Compose a config for one method, the way the ``optim/method`` group does.

    A method's parameters only exist in its own composition, so a test that
    drives two methods composes twice rather than passing a name around.
    """

    def build(method: str) -> DictConfig:
        config = OmegaConf.create(_CONFIG)
        config.optim.method = OmegaConf.create(_METHODS[method])
        config.simulation.input = {"sectors_file": write_sectors(tmp_path, _SECTORS)}
        return config

    return build


@pytest.fixture
def cfg(make_cfg):
    """A composed config carrying only what the search reads, defaulting to morbo."""
    return make_cfg("morbo")


@pytest.fixture
def evaluator(cfg) -> StubEvaluator:
    """A stub over a three-sector, two-band space."""
    return StubEvaluator(TiltSpace.from_config(cfg))


@pytest.mark.parametrize("method", ["morbo", "random"])
def test_every_method_starts_from_the_committed_incumbent(make_cfg, evaluator, method) -> None:
    """Row zero is always the deployed configuration.

    A comparison then always has its reference, and a run that improves on
    nothing still reports the baseline rather than an empty table.
    """
    history = run_search(evaluator, make_cfg(method))
    frame = history.frame()
    assert frame.loc[0, "phase"] == "incumbent"
    assert np.allclose(evaluator.seen[0], evaluator.space.baseline)
    assert np.allclose(
        frame.loc[0, list(evaluator.space.parameter_names)].to_numpy(dtype=float),
        evaluator.space.baseline,
    )


@pytest.mark.parametrize("method", ["morbo", "random"])
def test_no_method_ever_proposes_a_tilt_outside_the_box(make_cfg, evaluator, method) -> None:
    """Bounds are a hard constraint, never a relaxation the search may soften."""
    run_search(evaluator, make_cfg(method))
    proposals = np.array(evaluator.seen)
    assert (proposals >= evaluator.space.lower - 1e-9).all()
    assert (proposals <= evaluator.space.upper + 1e-9).all()


@pytest.mark.parametrize("method", ["morbo", "random"])
def test_every_method_proposes_tilts_on_the_resolution_lattice(make_cfg, evaluator, method) -> None:
    """An antenna is set in 0.1 degree steps, so every proposal must be one."""
    run_search(evaluator, make_cfg(method))
    steps = (np.array(evaluator.seen) - evaluator.space.lower) / 0.1
    assert np.allclose(steps, np.round(steps))


@pytest.mark.parametrize("method", ["morbo", "random"])
def test_the_budgeted_methods_spend_exactly_their_budget(make_cfg, evaluator, method) -> None:
    """``n_init + n_iter`` searched evaluations, plus the incumbent."""
    cfg = make_cfg(method)
    history = run_search(evaluator, cfg)
    budget = cfg.optim.method.budget
    assert len(history) == 1 + budget.n_init + budget.n_iter


def test_morbo_records_which_generator_made_each_point(make_cfg, evaluator) -> None:
    """Separating the Sobol design from the trust-region proposals needs this provenance."""
    frame = run_search(evaluator, make_cfg("morbo")).frame()
    assert frame.loc[0, "generation_node"] == "attached"
    assert set(frame["phase"]) == {"incumbent", "init", "search"}
    assert frame.loc[frame["phase"] == "init", "generation_node"].eq("Sobol").all()
    assert frame.loc[frame["phase"] == "search", "generation_node"].eq("MORBO").all()


def test_a_region_halves_after_its_failure_tolerance_and_a_success_resets_it() -> None:
    """The region only shrinks: there is no success streak that grows it."""
    region = TrustRegion(centre=0, length=0.8)
    region.record(improved=False, n_new=1, tolerance=3)
    region.record(improved=True, n_new=1, tolerance=3)
    assert (region.length, region.failures) == (0.8, 0)
    for _ in range(3):
        region.record(improved=False, n_new=1, tolerance=3)
    assert (region.length, region.failures) == (pytest.approx(0.4), 0)


def test_failures_count_evaluations_not_rounds() -> None:
    """One failed batch as large as the tolerance halves the region at once."""
    region = TrustRegion(centre=0, length=0.8)
    region.record(improved=False, n_new=3, tolerance=3)
    assert region.length == pytest.approx(0.4)


def test_the_failure_tolerance_follows_the_paper() -> None:
    """``max(10, ceil(d / 3))`` rounds."""
    assert failure_tolerance(6) == 10
    assert failure_tolerance(36) == 12


def test_the_perturbation_probability_halves_over_the_budget() -> None:
    """Appendix A's schedule: ``p0`` until the design ends, ``p0 / 2`` at the last evaluation."""
    p0 = 20.0 / 36.0
    assert perturbation_probability(36, 20.0, 8, 8, 72) == pytest.approx(p0)
    assert perturbation_probability(36, 20.0, 72, 8, 72) == pytest.approx(p0 / 2.0)
    assert perturbation_probability(6, 20.0, 40, 8, 72) < 1.0


def test_a_collapsed_region_restarts_at_a_restart_point(make_cfg) -> None:
    """A region born below ``length_min`` restarts every round and never proposes."""
    cfg = make_cfg("morbo")
    cfg.optim.method.trust_region.length_min = 0.9
    evaluator = StubEvaluator(TiltSpace.from_config(cfg))
    frame = run_search(evaluator, cfg).frame()
    assert len(frame) == 1 + 4 + 4
    assert set(frame["phase"]) == {"incumbent", "init"}
    assert frame["generation_node"].iloc[1:5].eq("Sobol").all()
    assert frame["generation_node"].iloc[5:].eq(RESTART).all()


def test_the_hv_scalarisation_keeps_the_best_point_of_each_set() -> None:
    """Per point ``min_j (y_j / w_j) ** m``, below-origin values clipped, then the set's maximum."""
    weights = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    sets = torch.tensor([[[3.0, 4.0], [0.6, -1.0]], [[0.6, 0.8], [0.3, 0.4]]], dtype=torch.float64)
    assert _hv_scalarisation(sets, weights).tolist() == pytest.approx([25.0, 1.0])


def test_a_region_moves_to_the_largest_contribution_inside_it() -> None:
    """Point 2 is out of reach; of the two inside, point 1 adds the most volume."""
    x = np.array([[0.5, 0.5], [0.6, 0.5], [0.0, 0.0]])
    y = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 1.0], [1.0, 9.0, 1.0]])
    region = TrustRegion(centre=0, length=0.4)
    _recentre(region, [], hypervolume_contributions(y), x, tabu=())
    assert region.centre == 1


def test_two_regions_take_two_front_points() -> None:
    """The second region cannot share the first one's centre while another point is free."""
    x = np.array([[0.5, 0.5], [0.6, 0.5]])
    y = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 1.0]])
    first, second = TrustRegion(centre=0, length=0.4), TrustRegion(centre=0, length=0.4)
    _recentre(first, [second], hypervolume_contributions(y), x, tabu=())
    _recentre(second, [first], hypervolume_contributions(y), x, tabu=())
    assert (first.centre, second.centre) == (1, 0)


def test_a_lone_front_point_is_shared() -> None:
    """With nothing else on the front, the taken point is still better than a dominated one."""
    x = np.array([[0.5, 0.5], [0.6, 0.5]])
    y = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    first, second = TrustRegion(centre=1, length=0.4), TrustRegion(centre=0, length=0.4)
    _recentre(second, [first], hypervolume_contributions(y), x, tabu=())
    assert second.centre == 1


def test_a_region_with_no_front_point_inside_moves_to_one_outside() -> None:
    """A dominated centre is never kept while a free front point exists anywhere."""
    x = np.array([[0.1, 0.1], [0.9, 0.9]])
    y = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    region = TrustRegion(centre=0, length=0.2)
    _recentre(region, [], hypervolume_contributions(y), x, tabu=())
    assert region.centre == 1


def test_a_tabu_point_is_skipped() -> None:
    """The centre a collapsed region left behind is not picked again."""
    x = np.array([[0.5, 0.5], [0.6, 0.5]])
    y = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 1.0]])
    region = TrustRegion(centre=1, length=0.4)
    _recentre(region, [], hypervolume_contributions(y), x, tabu=[1])
    assert region.centre == 0


def test_a_local_model_is_topped_up_with_the_nearest_points() -> None:
    """Fewer than ``min(250, 2d)`` points inside the 2L cube: the nearest ones fill in."""
    x = np.array([[0.5, 0.5], [0.52, 0.5], [0.9, 0.9], [0.0, 0.0], [0.6, 0.6]])
    rows = _local_rows(x, TrustRegion(centre=0, length=0.05))
    assert sorted(rows.tolist()) == [0, 1, 2, 4]


def test_a_sample_the_front_dominates_adds_nothing() -> None:
    """Only a sample beyond the front scores, by the volume it would add."""
    front = np.array([[2.0, 1.0, 1.0]])
    samples = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    assert _improvement(front, samples).tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_selection_prefers_improvement_and_falls_back_to_a_scalarisation() -> None:
    """An improving sample is rule 2; with none, the sample best on every weight wins, rule 1."""
    front = np.array([[2.0, 2.0, 2.0]])
    improving = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    assert _select(front, improving)[:2] == (2, 1)
    dominated = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [1.0, 1.0, 1.0]])
    assert _select(front, dominated)[:2] == (1, 1)


def test_random_search_never_reaches_a_model(make_cfg, evaluator) -> None:
    """What makes it the control: same loop, same budget, no model."""
    frame = run_search(evaluator, make_cfg("random")).frame()
    assert set(frame["generation_node"]) <= {"attached", "Sobol"}


def test_random_search_opens_with_morbos_initial_design(make_cfg) -> None:
    """Same seed, same Sobol prefix: the two methods diverge only after ``n_init``."""
    cfg = make_cfg("morbo")
    morbo_eval = StubEvaluator(TiltSpace.from_config(cfg))
    run_search(morbo_eval, cfg)
    random_eval = StubEvaluator(TiltSpace.from_config(cfg))
    run_search(random_eval, make_cfg("random"))
    n_init = cfg.optim.method.budget.n_init
    assert np.allclose(morbo_eval.seen[: 1 + n_init], random_eval.seen[: 1 + n_init])


def test_an_unknown_method_names_the_registered_ones(cfg, evaluator) -> None:
    """Adding a method is a registry entry, not an edit to a dispatch chain."""
    cfg.optim.method.name = "annealing"
    with pytest.raises(KeyError, match="morbo"):
        run_search(evaluator, cfg)


def test_history_frame_carries_provenance_kpis_and_every_tilt(make_cfg, evaluator) -> None:
    """The schema every method shares."""
    frame = run_search(evaluator, make_cfg("random")).frame()
    expected = {"iteration", "phase", "generation_node", "seconds"}
    assert expected <= set(frame.columns)
    assert set(MEASURE_NAMES) <= set(frame.columns)
    assert set(evaluator.space.parameter_names) <= set(frame.columns)
    assert frame[list(MEASURE_NAMES)].notna().all().all()


def test_the_tilt_table_reports_delta_against_the_incumbent(make_cfg, evaluator) -> None:
    """The deliverable. Delta is reported, never optimized."""
    cfg = make_cfg("random")
    history = run_search(evaluator, cfg)
    best = history.results[history.best_index()]
    table = history.tilt_table(best.tilt_deg)
    assert len(table) == evaluator.space.n_dim
    assert np.allclose(table["current_tilt_deg"], evaluator.space.baseline)
    assert np.allclose(
        table["delta_tilt_deg"], table["optimized_tilt_deg"] - table["current_tilt_deg"]
    )


def test_write_run_persists_every_artifact(make_cfg, evaluator, tmp_path: Path) -> None:
    """A run directory must be readable without the session that produced it."""
    cfg = make_cfg("random")
    history = run_search(evaluator, cfg)
    written = write_run(
        history,
        LocalRunWriter(tmp_path),
        cfg,
        method="random",
        extra={"scenario_id": "scn_test"},
    )
    assert set(written) == {"history", "run"}
    for locator in written.values():
        assert Path(locator).is_file()

    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["scenario_id"] == "scn_test"
    assert run["n_evaluations"] == len(history)
    assert run["incumbent_kpi"] == history.results[0].kpi.as_dict()
    assert run["config"]["kpi"]["capacity"]["max_admission_utilisation"] == 0.8


def test_an_empty_history_has_nothing_to_tabulate(evaluator) -> None:
    """Better than a zero-row frame that reads as a run which found nothing."""
    with pytest.raises(ValueError, match="no evaluations"):
        History(evaluator.space).frame()
