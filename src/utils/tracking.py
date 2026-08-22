"""Thin MLflow wrapper.

Training code calls this module rather than the MLflow API directly, so the
tracker can be swapped, extended or disabled by editing one file instead of
every notebook.

Log manually, not with ``autolog``
----------------------------------
Framework autologging records different parameters and metrics for gradient
boosting than for a PyTorch model, which is exactly the like-for-like
comparison this project is built to protect. Log the shared metrics from
:mod:`src.evaluation.metrics` explicitly, so every run is measured identically.
Autologging may be switched on as a supplement, never as the primary record.

Requires the ``tracking`` extra: ``uv sync --extra tracking``. MLflow is
imported lazily so the base environment works without it.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def start_run(cfg: DictConfig, run_name: str | None = None) -> AbstractContextManager:
    """Open an MLflow run for the current config.

    Args:
        cfg: Composed config; uses ``cfg.mlflow.tracking_uri`` and
            ``cfg.mlflow.experiment_name``.
        run_name: Optional label; defaults to something derived from the model
            name and timestamp.

    Returns:
        A context manager that closes the run on exit.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Return a no-op context manager when MLflow is not installed, so
        training code needs no conditional branches around tracking.

    Example:
        >>> with start_run(cfg):
        ...     log_config(cfg)
        ...     model.fit(X_train, y_train)
    """
    # TODO(1): import mlflow lazily; fall back to contextlib.nullcontext()
    # TODO(2): set_tracking_uri / set_experiment from cfg.mlflow
    # TODO(3): return mlflow.start_run(run_name=run_name)
    raise NotImplementedError("src.utils.tracking.start_run")


def log_config(cfg: DictConfig) -> None:
    """Log the composed config as run parameters.

    Args:
        cfg: The full composed config.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Log the *composed* config, not the file names: the whole point is to
        record the values that actually applied after overrides. Flatten nested
        keys to dotted names, and attach the YAML itself as an artifact so
        nothing is lost to MLflow's parameter length limit.
    """
    # TODO(1): OmegaConf.to_container(cfg, resolve=True) and flatten to dotted keys
    # TODO(2): mlflow.log_params(flat)
    # TODO(3): log the resolved YAML as an artifact
    raise NotImplementedError("src.utils.tracking.log_config")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics from :mod:`src.evaluation.metrics`.

    Args:
        metrics: Metric name to value, as returned by ``evaluate()``.
        step: Optional epoch or iteration index for training curves.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Prefix by split (``val/mae``, ``test/mae``) so the same metric from
        different splits stays distinguishable in the UI.
    """
    # TODO(1): no-op when no run is active
    # TODO(2): mlflow.log_metrics(metrics, step=step)
    raise NotImplementedError("src.utils.tracking.log_metrics")


def log_artifact(path: str | Path) -> None:
    """Attach a file — a figure, a results table, a model — to the active run.

    Args:
        path: File to upload.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        DVC already versions model artifacts by content against a Git commit,
        so attaching them here is optional; it is mainly useful for the MLflow
        Model Registry when models move into serving.
    """
    # TODO(1): mlflow.log_artifact(str(path)) when a run is active
    raise NotImplementedError("src.utils.tracking.log_artifact")


def log_model_summary(model: Any) -> None:
    """Log a human-readable description of the fitted model.

    Args:
        model: A fitted model from ``src.models``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Parameter counts, tree counts, layer shapes — cheap to record and the
        first thing anyone wants when comparing two runs months later.
    """
    # TODO(1): build the summary text and log it as an artifact
    raise NotImplementedError("src.utils.tracking.log_model_summary")
