"""One run: which solutions get published, and what it writes.

No Sionna-RT and no GPU. The evaluator is a stub, which is what the
:class:`~src.optim.evaluator.ObjectiveEvaluator` protocol seam is for, so the
whole search-publish-archive path runs in a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.evaluation import runs as run_store
from src.optim.evaluator import EvaluationResult
from src.optim.objective import (
    MEASURE_NAMES,
    KpiVector,
    hypervolume_contributions,
    objective_matrix,
    pareto_mask,
)
from src.optim.report import choose
from src.optim.run import run
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
        # `random` rather than `morbo`: deterministic in the seed, no model, and
        # it still exercises the whole publish path.
        "method": {"name": "random", "budget": {"n_init": 4, "n_iter": 6}},
        "output": {"dir": "", "deliverable_dir": ""},
        "seed": 0,
        "tilt_resolution_deg": 0.1,
    },
}


@dataclass
class StubEvaluator:
    """Stands in for the ray tracer, scoring a tilt vector by a fixed function."""

    space: TiltSpace
    seen: list[np.ndarray] = field(default_factory=list)
    scenario_id: str = "scn_test"

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
                estimated_throughput_p05_mbps=float(np.mean(unit)),
                estimated_throughput_p50_mbps=float(2.0 * np.mean(unit)),
                estimated_throughput_mean_mbps=float(3.0 * np.mean(unit)),
                coverage_objective=float(1.0 - np.mean((unit - 0.35) ** 2)),
                separation_objective=float(1.0 - np.mean((unit - 0.75) ** 2)),
                throughput_objective=float(1.0 + np.mean(unit)),
            ),
            seconds=1.0,
        )


@pytest.fixture
def cfg(tmp_path) -> DictConfig:
    """A config whose output directories are the test's own."""
    config = OmegaConf.create(_CONFIG)
    config.optim.output.dir = str(tmp_path / "optim")
    config.optim.output.deliverable_dir = str(tmp_path / "deliverable")
    config.simulation.input = {"sectors_file": write_sectors(tmp_path, _SECTORS)}
    return config


@pytest.fixture
def space(cfg) -> TiltSpace:
    """The three-sector, two-band space the stub scores over."""
    return TiltSpace.from_config(cfg)


@pytest.fixture
def stub(cfg, space, monkeypatch) -> StubEvaluator:
    """The evaluator :func:`src.optim.run.run` will build."""
    import src.optim.run as run_module

    evaluator = StubEvaluator(space)
    monkeypatch.setattr(run_module, "Evaluator", lambda cfg, keep_rsrp=False: evaluator)
    return evaluator


def _kpi(objectives: np.ndarray) -> KpiVector:
    """One KPI vector with the given objectives; the KPIs are filler the pick never reads."""
    return KpiVector(
        hole_rate=0.1,
        overlap_rate=0.3,
        overlap_neighbor_mean=0.5,
        weak_rate=0.2,
        rsrp_p05_dbm=-110.0,
        rsrp_p50_dbm=-95.0,
        sinr_p05_db=-3.0,
        sinr_p50_db=8.0,
        estimated_throughput_p05_mbps=0.0,
        estimated_throughput_p50_mbps=5.0,
        estimated_throughput_mean_mbps=6.0,
        coverage_objective=float(objectives[0]),
        separation_objective=float(objectives[1]),
        throughput_objective=float(objectives[2]),
    )


def _kpis(count: int) -> list[KpiVector]:
    """A spread of objective vectors to rank, the first a middling incumbent."""
    rng = np.random.default_rng(0)
    return [_kpi(np.full(3, 0.5)), *(_kpi(rng.uniform(0.1, 1.0, 3)) for _ in range(count - 1))]


def test_choose_publishes_the_whole_front_by_contribution() -> None:
    """Every Pareto point, largest hypervolume contribution first, and nothing else.

    Ranked by the measure that selects the winner, or the shortlist would
    disagree with the recommendation printed beside it.
    """
    kpis = _kpis(24)
    points = objective_matrix(kpis)
    contribution = hypervolume_contributions(points)
    picks = choose(kpis)

    assert sorted(picks) == np.flatnonzero(pareto_mask(points)).tolist()
    assert np.all(np.diff(contribution[picks]) <= 0)


def test_choose_lists_the_incumbent_only_when_it_is_on_the_front() -> None:
    """A dominated incumbent is not a solution; every delta still reads against it."""
    dominated = [_kpi(np.full(3, 0.5)), _kpi(np.full(3, 0.9)), _kpi(np.full(3, 0.4))]
    assert choose(dominated) == [1]
    on_front = [_kpi(np.array([0.9, 0.1, 0.5])), _kpi(np.array([0.1, 0.9, 0.5]))]
    assert sorted(choose(on_front)) == [0, 1]


def test_one_run_searches_and_publishes(cfg, space, stub) -> None:
    """A single command leaves a directory a comparison can load, and solves nothing extra."""
    history, directory = run(cfg)
    loaded = run_store.load(directory)

    assert len(stub.seen) == len(history)
    assert loaded.n_evaluations == len(history)


def test_run_json_records_the_code_and_packages_it_ran_with(cfg, stub) -> None:
    """A result is traceable to a commit and the library versions behind it."""
    _history, directory = run(cfg)
    provenance = run_store.load(directory).meta["provenance"]
    assert set(provenance) == {"git_commit", "git_dirty", "packages"}
    assert provenance["packages"]["numpy"] == np.__version__


def test_every_published_kpi_came_from_the_evaluator(cfg, space, stub) -> None:
    """Every published row was solved by the evaluator."""
    _history, directory = run(cfg)
    published = pd.read_parquet(directory / "solutions.parquet")

    solved = {tuple(np.round(vector, 9)) for vector in stub.seen}
    for _, row in published.iterrows():
        tilts = tuple(np.round(row[list(space.parameter_names)].to_numpy(dtype=float), 9))
        assert tilts in solved


def test_the_recommended_row_is_the_largest_contribution(cfg, stub) -> None:
    """`choose` and `best_index` must not be able to name different solutions."""
    history, directory = run(cfg)
    published = pd.read_parquet(directory / "solutions.parquet")

    assert published["recommended"].sum() == 1
    assert published.loc[published["recommended"], "iteration"].item() == history.best_index()


def test_the_run_publishes_a_shortlist_to_choose_from(cfg, stub) -> None:
    """The deliverable offers the whole front, and always names one solution first."""
    run(cfg)
    shortlist = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/solutions_random.csv")
    options = pd.read_csv(f"{cfg.optim.output.deliverable_dir}/tilt_options_random.csv")

    assert shortlist["recommended"].sum() == 1
    assert bool(shortlist.loc[0, "recommended"])
    for name in MEASURE_NAMES:
        assert f"delta_{name}" in shortlist
    # One tilt table per offered solution, each covering every sector-band pair.
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
