"""Shared resources injected into routes.

FastAPI dependencies exist so that routes never construct expensive objects
themselves: the model artifact is loaded once per process and handed to every
request. Tests override these to inject a stub instead of a real model.
"""

from functools import lru_cache

from omegaconf import DictConfig

from app.api.inference import ModelService


@lru_cache(maxsize=1)
def get_config() -> DictConfig:
    """Return the composed project config, cached per process.

    Returns:
        The same config object the training code uses.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Serving reads the same ``configs/`` tree as training, so the artifact
        path and feature contract cannot drift apart. Environment-specific
        values (ports, remote URIs) come from ``.env``, not from here.
    """
    # TODO(1): return load_config() from src.config
    raise NotImplementedError("app.api.dependencies.get_config")


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Return the process-wide, already-loaded model service.

    Returns:
        A :class:`~app.api.inference.ModelService` with its artifact loaded.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Cached, so the artifact is read from disk once. Which model is served
        comes from ``cfg.models.artifact_path`` — the same value ``save()``
        wrote to, so promoting a model is a config change, not a code change.
    """
    # TODO(1): cfg = get_config()
    # TODO(2): return ModelService(cfg.models.artifact_path).load()
    raise NotImplementedError("app.api.dependencies.get_model_service")
