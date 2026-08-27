"""The KPI surrogate itself — f_sur: (s, theta) -> K_hat.

PROJECT.md section 30 item 7 leaves the architecture open, so this module
defines the interface rather than a particular model. The concrete class is
instantiated from ``configs/surrogate.yaml`` through Hydra's ``_target_``, which
makes swapping architectures a config edit and an ablation a sweep.

The interface has one unusual requirement
-----------------------------------------
:meth:`Surrogate.predict` optionally returns uncertainty. Bayesian Optimization
needs a posterior, not a point estimate — an acquisition function that cannot
tell a confident prediction from a guess degenerates into greedy search over the
surrogate mean, and then reliably finds the region where the surrogate is most
wrong rather than where the network is best.

A model that cannot express uncertainty is still usable for MARL, whose reward
only needs the mean. Say so in the config rather than returning a fabricated
variance.

Predicting five outputs at once
-------------------------------
The five KPIs are computed from the same radio map and are strongly related — a
configuration that opens a coverage hole changes overlap and weak rate together.
A joint model can use that; five independent regressors cannot, and they can
produce KPI combinations that no radio map could generate.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Structural interface every surrogate in this project must satisfy."""

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "Surrogate":
        """Fit the model. Returns self so calls can be chained."""
        ...

    def predict(self, x: np.ndarray, *, return_std: bool = False) -> Any:
        """Predict the KPI vector, optionally with a standard deviation."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist the model and everything needed to reload it."""
        ...

    def load(self, path: str | Path) -> "Surrogate":
        """Restore a model saved by :meth:`save`."""
        ...


class SurrogateMixin:
    """Shared persistence for surrogate implementations.

    Saving the model without its feature transformer produces an artifact that
    silently mispredicts: the tilts arrive on a different scale from the one it
    was trained on, and nothing raises. Both objects are written together.
    """

    #: Fitted transformer from :func:`src.surrogate.features.fit`.
    transformer: Any = None
    #: The underlying estimator.
    model: Any = None

    def save(self, path: str | Path) -> None:
        """Persist the transformer, the model and the provenance together.

        Args:
            path: Destination, from ``cfg.surrogate.artifact_path``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Include the cell-band table ordering and the KPI names in the
            payload. A theta vector is meaningless without the column order it
            was built with, and a prediction vector is meaningless without
            knowing which KPI is at which position.

        Example:
            >>> surrogate.save(cfg.surrogate.artifact_path)
        """
        # TODO(1): mkdir the parent directory
        # TODO(2): save {transformer, model, cell_band_order, kpi_names, metadata}
        raise NotImplementedError("src.surrogate.model.SurrogateMixin.save")

    def load(self, path: str | Path) -> "SurrogateMixin":
        """Restore a surrogate saved by :meth:`save`.

        Args:
            path: The artifact path.

        Returns:
            Self, with the transformer and model restored.

        Raises:
            NotImplementedError: Always — implement this module first.
            FileNotFoundError: Once implemented, when the artifact is missing.

        Notes:
            Check the restored cell-band ordering against the one the current
            config produces. A band added to ``configs/radio.yaml`` after
            training makes every stored column index wrong, and the mismatch is
            otherwise invisible until the results look strange.

        Example:
            >>> surrogate = MySurrogate().load(cfg.surrogate.artifact_path)
        """
        # TODO(1): load the payload
        # TODO(2): raise when the stored cell-band order disagrees with the config
        # TODO(3): assign transformer and model, return self
        raise NotImplementedError("src.surrogate.model.SurrogateMixin.load")
