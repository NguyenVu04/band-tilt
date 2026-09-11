"""The MLflow stage logger: what it records, and that disabling it records nothing."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tracking import log_stage

mlflow = pytest.importorskip("mlflow")


def config(tmp_path, enabled: bool = True):
    """A minimal config with a SQLite store under ``tmp_path``."""
    return OmegaConf.create(
        {
            "seed": 3,
            "mlflow": {
                "enabled": enabled,
                "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                "experiment_name": "test",
            },
            "kpi": {"hole_dbm": -110.0, "band_priority": {"weights": [1, 2]}},
        }
    )


def test_logs_params_metrics_and_artifacts(tmp_path, monkeypatch) -> None:
    """Scalar config leaves become params; lists are left to config.yaml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "table.csv").write_text("a\n1\n")
    cfg = config(tmp_path)

    run_id = log_stage(
        cfg,
        "unit",
        groups=["kpi"],
        metrics={"hole_rate": 0.25},
        step_metrics=[{"loss": 2.0}, {"loss": 1.0}],
        artifacts=[tmp_path / "table.csv", tmp_path / "missing.csv"],
        outputs=["data/interim/radio_map.npz"],
        tags={"method": "mobo"},
    )

    run = mlflow.MlflowClient(str(cfg.mlflow.tracking_uri)).get_run(run_id)
    assert run.data.params == {"seed": "3", "kpi.hole_dbm": "-110.0"}
    assert run.data.metrics == {"hole_rate": 0.25, "loss": 1.0}
    assert run.data.tags["stage"] == "unit"
    assert run.data.tags["method"] == "mobo"
    assert run.data.tags["output.radio_map.npz"] == "data/interim/radio_map.npz"
    artifacts = {
        f.path for f in mlflow.MlflowClient(str(cfg.mlflow.tracking_uri)).list_artifacts(run_id)
    }
    assert artifacts == {"config.yaml", "table.csv"}


def test_disabled_writes_nothing(tmp_path) -> None:
    """``mlflow.enabled: false`` never opens the store."""
    assert log_stage(config(tmp_path, enabled=False), "unit", metrics={"x": 1.0}) is None
    assert not (tmp_path / "mlflow.db").exists()
