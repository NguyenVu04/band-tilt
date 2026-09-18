"""Loading runs, and refusing to compare ones that measured different things."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import runs as run_store
from src.optim.objective import MEASURE_NAMES

KPI = {
    "hole_rate": 0.10,
    "overlap_rate": 0.28,
    "served_ratio": 0.009,
    "weak_rate": 0.12,
    "edge_rsrp_dbm": -108.0,
    "objective": 0.40,
}


def radio_archive(**overrides: object) -> dict[str, np.ndarray]:
    """A radio map carrying every key `verify` compares."""
    archive = {
        "rsrp_dbm": np.full((1, 1, 2, 3), -80.0, dtype=np.float32),
        "band_label": np.array(["b700"]),
        "band_hz": np.array([700000000]),
        "scenario_id": np.array("scn_test"),
        "n_rows": np.array(2),
        "n_cols": np.array(3),
        "tile_size_m": np.array(20.0),
        "origin_x": np.array(0.0),
        "origin_y": np.array(0.0),
        "ue_height_m": np.array(1.5),
        "samples_per_tx": np.array(1000),
        "max_depth": np.array(5),
        "los": np.array(True),
        "specular_reflection": np.array(True),
        "diffuse_reflection": np.array(False),
        "refraction": np.array(True),
        "diffraction": np.array(False),
        "edge_diffraction": np.array(False),
        "diffraction_lit_region": np.array(True),
        "rr_depth": np.array(-1),
        "rr_prob": np.array(0.95),
        "power_dbm": np.array(4.85),
    }
    archive.update({key: np.array(value) for key, value in overrides.items()})
    return archive


def make_run(
    root: Path,
    method: str,
    run_id: str,
    *,
    n: int = 3,
    seed: int = 0,
    throughput_per_ue_bps: float = 1e6,
    edge_percentile: float = 5.0,
    bandwidth: int = 10000000,
    **radio: object,
) -> Path:
    """Write a run directory the way src.optim.history does."""
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
    pd.DataFrame(
        {
            "cell": ["n0c0", "n0c0"],
            "band": ["b700", "b2600"],
            "current_tilt_deg": [4.0, 8.0],
            "optimized_tilt_deg": [6.0, 8.0],
            "delta_tilt_deg": [2.0, 0.0],
        }
    ).to_parquet(directory / "best_tilt.parquet", index=False)

    np.savez_compressed(directory / "best_radio_map.npz", **radio_archive(**radio))
    (directory / "run.json").write_text(
        json.dumps(
            {
                "method": method,
                "n_evaluations": n,
                "best_iteration": 0,
                "best_kpi": KPI,
                "incumbent_kpi": KPI,
                "scenario_id": str(radio.get("scenario_id", "scn_test")),
                "wall_clock_seconds": 120.0,
                # Deliberately a Windows-style path: it must never be resolved.
                "best_radio_map": r"C:\somewhere\else\best_radio_map.npz",
                "config": {
                    "optim": {"seed": seed},
                    "kpi": {
                        "hole_dbm": -120.0,
                        "weak_dbm": -90.0,
                        "overlap_margin_db": 6.0,
                        "capacity": {"throughput_per_ue_bps": throughput_per_ue_bps},
                        "objective": {"beta": 1.0},
                        "quality": {"edge_percentile": edge_percentile},
                    },
                    "simulation": {
                        "transmitters": {"cells": [{"name": "n0c0", "max_prb": {"b700": 106}}]},
                        "radio_map": {
                            "temperature": 298.15,
                            "bands": [{"name": "b700", "bandwidth": bandwidth}],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_load_reads_tables_and_metadata(tmp_path) -> None:
    """Every field a comparison reads comes back off disk."""
    directory = make_run(tmp_path, "turbo", "2026-01-01_00-00-00")
    run = run_store.load(directory)

    assert run.method == "turbo"
    assert run.n_evaluations == 3
    assert run.label == "turbo/2026-01-01_00-00-00"
    assert run.incumbent_kpi.hole_rate == pytest.approx(KPI["hole_rate"])
    assert run.ray_tracing_seconds == pytest.approx(90.0)
    assert run.wall_clock_seconds == pytest.approx(120.0)


def test_radio_map_comes_from_the_directory_not_the_recorded_path(tmp_path) -> None:
    """run.json's path was written on whatever platform produced it."""
    run = run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00"))
    assert run.radio_map["rsrp_dbm"].shape == (1, 1, 2, 3)


def test_load_rejects_a_directory_that_is_not_a_run(tmp_path) -> None:
    """The output root is shared scratch."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(run_store.RunError, match="not a run directory"):
        run_store.load(tmp_path / "empty")


def test_load_rejects_an_unfinished_run(tmp_path) -> None:
    """A half-written run must not read as a result."""
    directory = make_run(tmp_path, "turbo", "2026-01-01_00-00-00")
    (directory / "best_tilt.parquet").unlink()
    with pytest.raises(run_store.RunError, match="did not finish"):
        run_store.load(directory)


def test_discover_skips_strays_and_orders_by_id(tmp_path) -> None:
    """One stray folder must not stop a comparison."""
    make_run(tmp_path, "turbo", "2026-01-02_00-00-00")
    make_run(tmp_path, "turbo", "2026-01-01_00-00-00")
    (tmp_path / "turbo" / "not-a-run").mkdir()

    found = run_store.discover(tmp_path)
    assert [run.run_id for run in found] == ["2026-01-01_00-00-00", "2026-01-02_00-00-00"]


def test_latest_per_method_and_seed_keeps_one_run_per_seed(tmp_path) -> None:
    """A rerun replaces its seed's earlier run; another seed is kept beside it."""
    make_run(tmp_path, "turbo", "2026-01-01_00-00-00", seed=0)
    make_run(tmp_path, "turbo", "2026-01-09_00-00-00", seed=0)
    make_run(tmp_path, "turbo", "2026-01-02_00-00-00", seed=1)
    make_run(tmp_path, "random", "2026-01-05_00-00-00", seed=0)

    latest = run_store.latest_per_method_and_seed(run_store.discover(tmp_path))
    assert [(run.method, run.seed, run.run_id) for run in latest] == [
        ("random", 0, "2026-01-05_00-00-00"),
        ("turbo", 0, "2026-01-09_00-00-00"),
        ("turbo", 1, "2026-01-02_00-00-00"),
    ]


def test_verify_passes_when_everything_matches(tmp_path) -> None:
    """The happy path must not raise."""
    runs = [run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00"))]
    checks = run_store.verify(runs, radio_archive())
    assert checks["holds"].all()
    run_store.require(checks)  # must not raise


def test_verify_catches_a_different_scenario(tmp_path) -> None:
    """Two scenarios are two experiments, not two results."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00", scenario_id="scn_other"))
    ]
    checks = run_store.verify(runs, radio_archive())
    failed = checks[~checks["holds"]]["check"].tolist()
    assert "every run optimized the baseline's scenario" in failed


def test_verify_catches_a_different_fidelity(tmp_path) -> None:
    """A map solved at another sample count is a different experiment."""
    runs = [run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00", samples_per_tx=99))]
    checks = run_store.verify(runs, radio_archive())
    assert (
        "solver samples_per_tx matches the baseline" in checks[~checks["holds"]]["check"].tolist()
    )


def test_verify_catches_a_different_capacity_model(tmp_path) -> None:
    """The served ratio depends on kpi.capacity, so a changed demand is another objective."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00")),
        run_store.load(
            make_run(tmp_path, "random", "2026-01-01_00-00-00", throughput_per_ue_bps=2e6)
        ),
    ]
    checks = run_store.verify(runs, radio_archive())
    failed = checks[~checks["holds"]]
    assert failed["check"].tolist() == ["KPI definition agrees across runs"]
    assert failed["offenders"].tolist() == ["random/2026-01-01_00-00-00"]


def test_verify_catches_a_retuned_carrier(tmp_path) -> None:
    """A band keeps its name when its carrier moves, so the label cannot carry this."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00", band_hz=[3500000000]))
    ]
    checks = run_store.verify(runs, radio_archive())
    failed = checks[~checks["holds"]]["check"].tolist()
    assert "band carrier frequencies match the baseline, in order" in failed


def test_verify_catches_a_different_edge_percentile(tmp_path) -> None:
    """edge_rsrp_dbm is a reported KPI, so the percentile behind it must agree."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00")),
        run_store.load(make_run(tmp_path, "random", "2026-01-01_00-00-00", edge_percentile=50.0)),
    ]
    checks = run_store.verify(runs, radio_archive())
    failed = checks[~checks["holds"]]
    assert failed["check"].tolist() == ["KPI definition agrees across runs"]
    assert failed["offenders"].tolist() == ["random/2026-01-01_00-00-00"]


def test_verify_catches_a_different_bandwidth(tmp_path) -> None:
    """Bandwidth sets the noise floor, so it sets SINR and the served ratio."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00")),
        run_store.load(make_run(tmp_path, "random", "2026-01-01_00-00-00", bandwidth=40000000)),
    ]
    checks = run_store.verify(runs, radio_archive())
    assert checks[~checks["holds"]]["check"].tolist() == ["KPI definition agrees across runs"]


def test_require_names_the_offender(tmp_path) -> None:
    """An error that does not say who failed is not actionable."""
    runs = [
        run_store.load(make_run(tmp_path, "turbo", "2026-01-01_00-00-00", scenario_id="scn_other"))
    ]
    with pytest.raises(run_store.RunError, match="turbo/2026-01-01_00-00-00"):
        run_store.require(run_store.verify(runs, radio_archive()))


def test_a_single_evaluation_run_still_loads(tmp_path) -> None:
    """Degenerate budgets must not break the reader."""
    run = run_store.load(make_run(tmp_path, "rule", "2026-01-01_00-00-00", n=1))
    assert run.n_evaluations == 1
    assert run.ray_tracing_seconds == pytest.approx(30.0)
