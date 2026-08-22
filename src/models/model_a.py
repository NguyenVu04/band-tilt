"""Placeholder model — copy this file to add a real one.

To add a model:

1. ``cp src/models/model_a.py src/models/<your_model>.py`` and rename the class.
2. ``cp configs/models/_template.yaml configs/models/<your_model>.yaml`` and point
   ``_target_`` at the new class.
3. ``cp notebooks/02a_model_a.ipynb notebooks/02<letter>_<your_model>.ipynb``.
4. Declare any new library in the matching extra in ``pyproject.toml``.
5. ``task train -- models=<your_model>``.

This module owns everything specific to one model: its preprocessing, its
feature engineering, its architecture and its training loop.

What must NOT go here
---------------------
- Splitting — call :mod:`src.data.split`, never ``sklearn.model_selection``
  directly. Models validated on different folds cannot be compared.
- Metric definitions — call :mod:`src.evaluation.metrics`. A locally defined
  metric makes the results table in notebook 03 meaningless.

Everything that learns from data belongs here rather than in notebook 01:
imputation, scaling, encoding, statistical outlier removal, feature selection.
All of it is fitted on the training split only and persisted with the model.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModelMixin


class ModelA(BaseModelMixin):
    """A model.

    Hydra constructs this class from ``configs/models/model_a.yaml`` via
    ``instantiate(cfg.models)``, which passes the ``model:`` group — and only
    that group — as keyword arguments. ``optim:`` and ``train:`` are read
    inside :meth:`fit`, which is why the config keeps the three groups apart.

    Attributes:
        preprocessor: Fitted preprocessing pipeline; ``None`` until ``fit``.
        model: The fitted estimator; ``None`` until ``fit``.
        params: The architecture arguments this instance was built with.
    """

    def __init__(self, **params: Any) -> None:
        """Store architecture arguments; build nothing yet.

        Args:
            **params: The ``model:`` group from the model config.

        Notes:
            Keep the constructor cheap and side-effect free. Building the
            estimator in :meth:`fit` keeps an unfitted instance trivially
            picklable and makes hyperparameter sweeps cheap to set up.
        """
        self.params = params
        self.preprocessor: Any = None
        self.model: Any = None

    def build_preprocessor(self, X_train: pd.DataFrame) -> Any:
        """Construct this model's preprocessing pipeline.

        Args:
            X_train: Training features, used only to learn column structure.

        Returns:
            An unfitted transformer exposing ``fit`` and ``transform``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Everything data-dependent goes here: imputation, scaling, encoding,
            statistical outlier clipping. Returning a single composed object
            (e.g. a ``ColumnTransformer`` inside a ``Pipeline``) is what makes
            it possible to serialise the whole preprocessing with the model.
        """
        # TODO(1): declare numeric / categorical column groups
        # TODO(2): compose imputation, scaling and encoding into one transformer
        raise NotImplementedError("src.models.model_a.ModelA.build_preprocessor")

    def build_model(self) -> Any:
        """Construct the untrained estimator from :attr:`params`.

        Returns:
            The estimator or network, not yet fitted.

        Raises:
            NotImplementedError: Always — implement this module first.
        """
        # TODO(1): instantiate the estimator using **self.params
        raise NotImplementedError("src.models.model_a.ModelA.build_model")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> "ModelA":
        """Fit preprocessing and model on the training split.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features for early stopping and tuning.
            y_val: Optional validation labels.
            **kwargs: The ``optim:`` and ``train:`` config groups, when the
                caller passes them through.

        Returns:
            ``self``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            The validation set is *transformed*, never fitted on. Fitting the
            preprocessor on train+val inflates validation scores and hides
            overfitting, which defeats the point of having a validation split.

        Example:
            >>> model = ModelA(**cfg.models.model).fit(X_train, y_train, X_val, y_val)
        """
        # TODO(1): self.preprocessor = self.build_preprocessor(X_train)
        # TODO(2): Xt = self.preprocessor.fit_transform(X_train)   # train only
        # TODO(3): Xv = self.preprocessor.transform(X_val) if X_val is not None
        # TODO(4): self.model = self.build_model(); train with early stopping on (Xv, y_val)
        # TODO(5): log params and per-epoch metrics via src.utils.tracking
        raise NotImplementedError("src.models.model_a.ModelA.fit")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for new data using the fitted preprocessing.

        Args:
            X: Raw features, same columns as the training input.

        Returns:
            Predictions aligned with ``X``.

        Raises:
            NotImplementedError: Always — implement this module first.
            RuntimeError: Once implemented, when called before ``fit`` or ``load``.

        Notes:
            ``transform`` only. This method is called by notebook 03 and by the
            serving layer, and both must see exactly the training-time
            transformation.
        """
        # TODO(1): guard against an unfitted instance
        # TODO(2): Xt = self.preprocessor.transform(X)
        # TODO(3): return self.model.predict(Xt)
        raise NotImplementedError("src.models.model_a.ModelA.predict")
