"""Loading runs, and refusing to compare ones that measured different things."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.evaluation import runs as run_store
from src.optim.objective import MEASURE_NAMES

KPI = {
    "hole_rate": 0.10,
    "weak_rate": 0.12,
    "overlap_rate": 0.28,
    "overlap_neighbor_mean": 0.45,
    "rsrp_p50_dbm": -95.0,
    "rsrp_p05_dbm": -108.0,
    "sinr_p50_db": 8.0,
    "sinr_p05_db": -3.0,
    "estimated_throughput_p05_mbps": 1.0,
    "estimated_throughput_p50_mbps": 5.0,
    "estimated_throughput_mean_mbps": 6.0,
    "coverage_objective": 0.9,
    "separation_objective": 0.4,
    "throughput_objective": 3.0,
}

CONFIG = {
    "kpi": {
        "hole_dbm": -120.0,
        "weak_dbm": -90.0,
        "overlap_margin_db": 6.0,
        "capacity": {"max_admission_utilisation": 0.8},
    },
    "simulation": {
        "input": {"scene_file": "scene.xml"},
        "ue": {"height_m": 1.5},
        "scene": {"merge_shapes": True},
        "radio_map": {
            "temperature": 298.15,
            "samples_per_tx": 1000,
            "bands": [{"name": "b700", "bandwidth": 10000000, "scs_hz": 15000}],
        },
        "antenna": {"power_rs": 4.85},
    },
}


def make_run(
    root: Path,
    method: str,
    run_id: str,
    *,
    n: int = 3,
    scenario_id: str = "scn_test",
    max_prb: int = 106,
    edit=None,
) -> Path:
    """Write a run directory the way src.optim.history does; ``edit`` changes its config."""
    directory = root / method / run_id
    directory.mkdir(parents=True)

    history = pd.DataFrame(
        {
            "iteration": range(n),
            "phase": ["init"] * n,
            "seconds": [30.0] * n,
            **{name: [KPI[name]] * n for name in MEASURE_NAMES},
        }
    )
    history.to_parquet(directory / "history.parquet", index=False)
    config = copy.deepcopy(CONFIG) | {"optim": {"seed": 0}}
    if edit:
        edit(config)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "method": method,
                "n_evaluations": n,
                "incumbent_kpi": KPI,
                "scenario_id": scenario_id,
                "max_prb": {"n0s0": {"b700": max_prb}},
                "wall_clock_seconds": 120.0,
                "config": config,
            }
        ),
        encoding="utf-8",
    )
    return directory


def verify(runs):
    """:func:`run_store.verify` against :data:`CONFIG` and its scenario."""
    return run_store.verify(runs, OmegaConf.create(CONFIG), "scn_test")


def failed(checks: pd.DataFrame) -> list[str]:
    """The names of the checks that do not hold."""
    return checks.loc[~checks["holds"], "check"].tolist()


def test_load_reads_tables_and_metadata(tmp_path) -> None:
    """Every field a comparison reads comes back off disk."""
    directory = make_run(tmp_path, "morbo", "2026-01-01_00-00-00")
    run = run_store.load(directory)

    assert run.method == "morbo"
    assert run.n_evaluations == 3
    assert run.label == "morbo/2026-01-01_00-00-00"
    assert run.incumbent_kpi.hole_rate == pytest.approx(KPI["hole_rate"])
    assert run.ray_tracing_seconds == pytest.approx(90.0)
    assert run.wall_clock_seconds == pytest.approx(120.0)


def test_load_rejects_a_directory_that_is_not_a_run(tmp_path) -> None:
    """The output root is shared scratch."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(run_store.RunError, match="not a run directory"):
        run_store.load(tmp_path / "empty")


def test_load_rejects_an_unfinished_run(tmp_path) -> None:
    """A half-written run must not read as a result."""
    directory = make_run(tmp_path, "morbo", "2026-01-01_00-00-00")
    (directory / "history.parquet").unlink()
    with pytest.raises(run_store.RunError, match="did not finish"):
        run_store.load(directory)


def test_discover_skips_strays_and_orders_by_id(tmp_path) -> None:
    """One stray folder must not stop a comparison."""
    make_run(tmp_path, "morbo", "2026-01-02_00-00-00")
    make_run(tmp_path, "morbo", "2026-01-01_00-00-00")
    (tmp_path / "morbo" / "not-a-run").mkdir()

    found = run_store.discover(tmp_path)
    assert [run.run_id for run in found] == ["2026-01-01_00-00-00", "2026-01-02_00-00-00"]


def test_latest_per_method_keeps_the_newest_run_of_each(tmp_path) -> None:
    """A rerun replaces its method's earlier run."""
    make_run(tmp_path, "morbo", "2026-01-01_00-00-00")
    make_run(tmp_path, "morbo", "2026-01-09_00-00-00")
    make_run(tmp_path, "random", "2026-01-05_00-00-00")

    latest = run_store.latest_per_method(run_store.discover(tmp_path))
    assert [(run.method, run.run_id) for run in latest] == [
        ("morbo", "2026-01-09_00-00-00"),
        ("random", "2026-01-05_00-00-00"),
    ]


def test_verify_passes_when_everything_matches(tmp_path) -> None:
    """The happy path must not raise."""
    checks = verify([run_store.load(make_run(tmp_path, "morbo", "2026-01-01_00-00-00"))])
    assert checks["holds"].all()
    run_store.require(checks)  # must not raise


def test_verify_catches_a_different_scenario(tmp_path) -> None:
    """Two scenarios are two experiments, not two results."""
    run = run_store.load(make_run(tmp_path, "morbo", "2026-01-01_00-00-00", scenario_id="other"))
    assert failed(verify([run])) == ["every run optimized the current scenario"]


def test_verify_catches_a_different_fidelity(tmp_path) -> None:
    """A run solved at another sample count is a different experiment."""

    def edit(config):
        config["simulation"]["radio_map"]["samples_per_tx"] = 99

    run = run_store.load(make_run(tmp_path, "morbo", "2026-01-01_00-00-00", edit=edit))
    assert failed(verify([run])) == ["simulation.radio_map matches the current config"]


def test_verify_catches_a_different_capacity_model(tmp_path) -> None:
    """Throughput depends on kpi.capacity, so a changed PRB share is another measurement."""

    def edit(config):
        config["kpi"]["capacity"]["max_admission_utilisation"] = 1.0

    run = run_store.load(make_run(tmp_path, "random", "2026-01-01_00-00-00", edit=edit))
    checks = verify([run])
    assert failed(checks) == ["kpi matches the current config"]
    assert checks.loc[~checks["holds"], "offenders"].tolist() == ["random/2026-01-01_00-00-00"]


def test_verify_catches_different_prb_limits(tmp_path) -> None:
    """The sector table is outside the config snapshot, so its recorded PRB limits are compared."""
    runs = [
        run_store.load(make_run(tmp_path, "morbo", "2026-01-01_00-00-00")),
        run_store.load(make_run(tmp_path, "random", "2026-01-01_00-00-00", max_prb=52)),
    ]
    assert failed(verify(runs)) == ["sector PRB limits agree across runs"]


def test_require_names_the_offender(tmp_path) -> None:
    """An error that does not say who failed is not actionable."""
    run = run_store.load(make_run(tmp_path, "morbo", "2026-01-01_00-00-00", scenario_id="other"))
    with pytest.raises(run_store.RunError, match="morbo/2026-01-01_00-00-00"):
        run_store.require(verify([run]))


def test_a_single_evaluation_run_still_loads(tmp_path) -> None:
    """Degenerate budgets must not break the reader."""
    run = run_store.load(make_run(tmp_path, "random", "2026-01-01_00-00-00", n=1))
    assert run.n_evaluations == 1
    assert run.ray_tracing_seconds == pytest.approx(30.0)
