"""Scripted training entry point.

The ``02x`` notebooks are where a model is developed. This module is where it
runs unattended once it is settled: sweeps, DVC stages, CI, retraining.

Usage::

    task train -- models=model_a
    task train -- models=model_a seed=7 models.optim.lr=1e-3
    task sweep -- models=model_a,model_b

Both paths must produce identical results for identical configs. Keep this
script thin: it wires config to the same ``src`` functions a notebook calls,
and adds nothing of its own.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Train one model under one composed config.

    Args:
        cfg: Config composed by Hydra from ``configs/config.yaml`` plus any
            command-line overrides.

    Returns:
        The primary validation metric, so Hydra sweepers can optimise it.
        ``None`` when no validation split is configured.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The test split is never read here. Scoring on test happens once, in
        notebook 03, after all models exist.
    """
    # TODO(1): set_seed(cfg.seed)
    # TODO(2): train_df = load_processed(cfg, "train")
    # TODO(3): train_df, val_df = train_val_split(train_df, cfg)   # shared splitter
    # TODO(4): model = instantiate(cfg.models)                     # via _target_
    # TODO(5): with tracking.start_run(cfg): log_config(cfg); model.fit(...)
    # TODO(6): score val predictions with src.evaluation.metrics.evaluate
    # TODO(7): log metrics, then model.save(cfg.models.artifact_path)
    # TODO(8): return the primary metric for sweepers
    raise NotImplementedError("src.models.train.main")


if __name__ == "__main__":
    main()
