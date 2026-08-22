"""Model loading and prediction.

This module is where train/serve skew is prevented. It loads the artifact
produced by a ``02x`` notebook — preprocessor and model together — and applies
the *same fitted* preprocessing that training used::

    artifact = load_artifact("models/model_a.pkl")


    def predict(features):
        X = artifact["preprocessor"].transform(features)  # transform, never fit
        return artifact["model"].predict(X)

What must NOT go here
---------------------
Any transformation that is not already inside the fitted preprocessor. Cleaning
a field "just for serving" means production sees different inputs than
training did, and the difference will not show up in any offline metric.
"""

from pathlib import Path
from typing import Any

import pandas as pd


class ModelService:
    """Holds the loaded artifact and serves predictions.

    One instance per process, created at startup and injected into routes by
    :mod:`app.api.dependencies`. Loading per request would re-read the artifact
    from disk every time.

    Attributes:
        artifact_path: Path the artifact was loaded from.
        artifact: The loaded ``{"preprocessor", "model"}`` payload.
        version: Version string reported by ``/health`` and in responses.
    """

    def __init__(self, artifact_path: str | Path) -> None:
        """Record where the artifact lives; load lazily via :meth:`load`.

        Args:
            artifact_path: Path to the serialised artifact.
        """
        self.artifact_path = Path(artifact_path)
        self.artifact: dict[str, Any] | None = None
        self.version: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the artifact is in memory."""
        return self.artifact is not None

    def load(self) -> "ModelService":
        """Load the artifact into memory.

        Returns:
            ``self``.

        Raises:
            NotImplementedError: Always — implement this module first.
            FileNotFoundError: Once implemented, when the artifact is missing.

        Notes:
            Called once at application startup so that a broken or missing
            artifact fails the deployment immediately, rather than the first
            request.
        """
        # TODO(1): self.artifact = load_artifact(self.artifact_path)
        # TODO(2): self.version = artifact_metadata(self.artifact_path).get("version")
        raise NotImplementedError("app.api.inference.ModelService.load")

    def predict(self, features: pd.DataFrame) -> list[float]:
        """Score a batch of records.

        Args:
            features: One row per record, columns matching the training input.

        Returns:
            One prediction per row, in input order.

        Raises:
            NotImplementedError: Always — implement this module first.
            RuntimeError: Once implemented, when called before :meth:`load`.

        Notes:
            ``transform`` only — see the module docstring. Batch the whole
            frame in a single call rather than looping per record.
        """
        # TODO(1): guard on self.is_loaded
        # TODO(2): X = self.artifact["preprocessor"].transform(features)
        # TODO(3): return self.artifact["model"].predict(X).tolist()
        raise NotImplementedError("app.api.inference.ModelService.predict")


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn validated request payloads into the frame the preprocessor expects.

    Args:
        records: Request models dumped to dicts.

    Returns:
        A frame with the training column names, in the training order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Column order and dtypes matter: a preprocessor fitted on a frame will
        happily transform a differently ordered one and produce silently wrong
        features. Reindex against the stored training columns instead of
        trusting the caller.
    """
    # TODO(1): build the frame, then reindex to the stored training columns
    raise NotImplementedError("app.api.inference.to_frame")
