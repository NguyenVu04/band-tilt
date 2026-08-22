"""Artifact serialisation.

One rule governs this module: **the fitted preprocessor and the model are saved
together, as a single artifact.** Notebook 03 and the serving layer load that
artifact and call ``transform`` — never ``fit``. A model saved without its
preprocessing cannot be scored or served the way it was trained, and the
mismatch usually surfaces as quietly wrong predictions rather than an error.

The canonical payload is::

    {"preprocessor": <fitted transformer>, "model": <fitted estimator>}

Artifacts are written to ``models/``, tracked by DVC, and never committed.
"""

from pathlib import Path
from typing import Any


def save_artifact(obj: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Serialise an artifact to disk.

    Args:
        obj: What to save — normally the ``{"preprocessor", "model"}`` dict.
        path: Destination, normally ``cfg.models.artifact_path``.
        metadata: Optional provenance stored alongside the object: Git commit,
            config hash, library versions, training timestamp.

    Returns:
        The path written.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Pickle-based formats (``joblib``) execute code on load — only ever load
        artifacts this project produced. Record the library versions in
        ``metadata``: an artifact that silently fails to unpickle after an
        upgrade is much easier to diagnose with them than without.

    Example:
        >>> save_artifact({"preprocessor": preproc, "model": model}, "models/model_a.pkl")
    """
    # TODO(1): mkdir parents for the destination
    # TODO(2): wrap obj with metadata (commit, config hash, versions, timestamp)
    # TODO(3): joblib.dump and return the resolved path
    raise NotImplementedError("src.utils.io.save_artifact")


def load_artifact(path: str | Path) -> Any:
    """Load an artifact written by :func:`save_artifact`.

    Args:
        path: Artifact path.

    Returns:
        The stored object, in the same shape it was saved.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the artifact is missing —
            usually because ``dvc pull`` has not been run.

    Example:
        >>> artifact = load_artifact("models/model_a.pkl")
        >>> preds = artifact["model"].predict(artifact["preprocessor"].transform(X))
    """
    # TODO(1): check the path exists, mentioning `task dvc:pull` if not
    # TODO(2): joblib.load and return the payload
    raise NotImplementedError("src.utils.io.load_artifact")


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    """Read an artifact's provenance without unpickling the model.

    Args:
        path: Artifact path.

    Returns:
        The metadata recorded at save time.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Useful for reports and for the serving layer's health endpoint, which
        should be able to say which model version it is running without
        loading it.
    """
    # TODO(1): read the metadata section only
    raise NotImplementedError("src.utils.io.artifact_metadata")
