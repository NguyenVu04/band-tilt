"""The radio-map surrogate itself — f_sur: (x, tilt) -> R_hat.

PROJECT.md section 22.2 leaves the architecture open, so this module
defines the interface rather than a particular model. The concrete class is
instantiated from ``configs/surrogate.yaml`` through Hydra's ``_target_``, which
makes swapping architectures a config edit and an ablation a sweep.

It predicts a map, not five numbers
-----------------------------------
The output is RSRP in dBm over the whole evaluation grid, indexed
``(cell_band, grid_cell)`` — PROJECT.md section 10 and Decision 6. The five KPIs
are then derived from the prediction by :mod:`src.kpi`, the same code that
derives them from a ray-traced map. One evaluator, two possible maps underneath
it.

That makes this a dense spatial prediction problem rather than a tabular
regression, and it decouples the model from the KPI thresholds entirely: moving
the hole threshold changes the score of a saved prediction without invalidating
the model that produced it.

The interface has one unusual requirement
-----------------------------------------
:meth:`Surrogate.predict` optionally returns uncertainty. TuRBO fits its own
Gaussian process over the KPIs derived from these predictions, so it does not
strictly need a posterior from this model — but propagating map uncertainty
through the KPI evaluator is far more informative than discarding it, because
uncertainty concentrated near the -120 dBm threshold is exactly what makes a
predicted hole rate untrustworthy.

A model that cannot express uncertainty is still usable. Say so in the config
rather than returning a fabricated variance.
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
            payload. A tilt vector is meaningless without the column order it
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
