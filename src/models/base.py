"""The common model interface.

Every model exposes the same four methods, so training, evaluation and serving
code is written once and works for all of them. Internals may differ
completely — a single ``.fit()`` call for gradient boosting, a full epoch loop
with early stopping for a neural network — as long as the surface holds.

This is a :class:`typing.Protocol`, i.e. structural typing: a model class does
not need to inherit from anything, it just needs the right methods. Inherit
from :class:`BaseModelMixin` if you want the shared ``save``/``load`` behaviour
for free.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class BaseModel(Protocol):
    """Structural interface every model in ``src.models`` must satisfy."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseModel":
        """Fit preprocessing and model on the training data.

        Args:
            X_train: Training features, already split by ``src.data.split``.
            y_train: Training labels.
            X_val: Optional validation features, used for early stopping and
                tuning. Never used to fit any transformation.
            y_val: Optional validation labels.

        Returns:
            ``self``, so calls can be chained.

        Notes:
            Everything that learns — imputers, scalers, encoders, statistical
            outlier thresholds — is fitted here, on ``X_train`` only, and
            stored on the instance so :meth:`save` can persist it.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for new data.

        Args:
            X: Raw features in the same shape as the training input.

        Returns:
            Predictions as a 1-D array aligned with ``X``.

        Notes:
            Applies the *fitted* preprocessing with ``transform``. Calling
            ``fit`` or ``fit_transform`` anywhere in this path is train/serve
            skew and invalidates the evaluation.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Persist the preprocessor and the model together as one artifact."""
        ...

    def load(self, path: str | Path) -> "BaseModel":
        """Restore a model previously written by :meth:`save`."""
        ...


class BaseModelMixin:
    """Shared ``save`` / ``load`` implementation for concrete models.

    Optional. Subclass it to inherit artifact handling that already follows the
    project rule — preprocessor and model serialised as a single object — so
    notebook 03 and the serving layer can only ever ``transform``.

    Attributes:
        preprocessor: The fitted preprocessing object, set during ``fit``.
        model: The fitted estimator, set during ``fit``.
    """

    preprocessor: Any = None
    model: Any = None

    def save(self, path: str | Path) -> None:
        """Write ``{"preprocessor": ..., "model": ...}`` to ``path``.

        Args:
            path: Destination artifact path, normally ``cfg.models.artifact_path``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Saving the model without its preprocessor is the single most common
            way a project ends up with results it cannot reproduce at serving
            time. Keep them in one file.
        """
        # TODO(1): delegate to src.utils.io.save_artifact with both objects
        raise NotImplementedError("src.models.base.BaseModelMixin.save")

    def load(self, path: str | Path) -> "BaseModelMixin":
        """Restore ``preprocessor`` and ``model`` from an artifact.

        Args:
            path: Artifact written by :meth:`save`.

        Returns:
            ``self``, populated.

        Raises:
            NotImplementedError: Always — implement this module first.
        """
        # TODO(1): load via src.utils.io.load_artifact and assign both attributes
        raise NotImplementedError("src.models.base.BaseModelMixin.load")
