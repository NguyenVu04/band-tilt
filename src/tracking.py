"""Record one pipeline stage as one MLflow run.

Called from each stage's ``@hydra.main`` entry point only, never from library
code, so importing ``src`` never touches a tracking store and notebooks stay
untracked unless they call this themselves.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# MLflow rejects param values longer than this (``MAX_PARAM_VAL_LENGTH``).
_MAX_PARAM_LENGTH = 6000


def _scalar_params(tree: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys, keeping scalar leaves only.

    Lists (the cell layout, hotspots) are dropped: they are in ``config.yaml``
    whole, and as params they would exceed MLflow's value limit.
    """
    if isinstance(tree, Mapping):
        flat: dict[str, Any] = {}
        for key, value in tree.items():
            flat.update(_scalar_params(value, f"{prefix}{key}."))
        return flat
    if isinstance(tree, str | int | float | bool) or tree is None:
        text = str(tree)
        return {prefix[:-1]: text} if len(text) <= _MAX_PARAM_LENGTH else {}
    return {}


def log_stage(
    cfg: DictConfig,
    stage: str,
    *,
    groups: Sequence[str] = (),
    metrics: Mapping[str, float] | None = None,
    step_metrics: Iterable[Mapping[str, float]] = (),
    artifacts: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    tags: Mapping[str, Any] | None = None,
) -> str | None:
    """Log one finished stage to MLflow. Returns the run id, or None when skipped.

    Reads ``cfg.mlflow.enabled``, ``cfg.mlflow.tracking_uri`` and
    ``cfg.mlflow.experiment_name``. Skips, with one printed line, when tracking
    is disabled or the ``tracking`` extra is not installed.

    Args:
        cfg: The composed config. Logged whole as the ``config.yaml`` artifact.
        stage: The run name and the ``stage`` tag.
        groups: Top-level config groups whose scalar leaves become params.
        metrics: Final scalar metrics.
        step_metrics: One mapping per step, logged with the step as its index.
        artifacts: Files or directories copied into the run; a relative
            directory keeps its path. Missing ones are skipped.
        outputs: Paths recorded as ``output.<name>`` tags and not copied —
            large data artifacts belong to DVC, not the tracking store.
        tags: Extra run tags, e.g. the method or the scenario id.
    """
    if not bool(cfg.mlflow.enabled):
        return None
    if importlib.util.find_spec("mlflow") is None:
        print("mlflow not installed (task sync --extra tracking); stage not tracked")
        return None
    import mlflow

    resolved = OmegaConf.to_container(cfg, resolve=True)
    params: dict[str, Any] = {"seed": resolved.get("seed")}
    for group in groups:
        params.update(_scalar_params(resolved.get(group, {}), f"{group}."))

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    with mlflow.start_run(run_name=stage) as run:
        mlflow.set_tags({"stage": stage, **{k: str(v) for k, v in (tags or {}).items()}})
        mlflow.set_tags({f"output.{Path(p).name}": str(p) for p in outputs})
        mlflow.log_params(params)
        mlflow.log_dict(resolved, "config.yaml")
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for step, row in enumerate(step_metrics):
            mlflow.log_metrics({k: float(v) for k, v in row.items()}, step=step)
        for path in map(Path, artifacts):
            if path.is_dir():
                # The relative path, so reports/figures/X and reports/tables/X stay apart.
                where = path.name if path.is_absolute() else path.as_posix()
                mlflow.log_artifacts(str(path), artifact_path=where)
            elif path.is_file():
                mlflow.log_artifact(str(path))
        print(f"mlflow: {stage} logged as run {run.info.run_id} in {cfg.mlflow.tracking_uri}")
        return run.info.run_id
