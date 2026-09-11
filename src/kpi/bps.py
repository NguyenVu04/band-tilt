"""KPI 4 - the UE-weighted Band Priority Score.

The three coverage rates say nothing about WHICH layer serves a location; this
one asks whether the layers the project would rather use are the ones serving
the ground the users actually stand on. Maximised.

Each MDT UE is served by :mod:`src.kpi.capacity` - band preference above an
RSRP threshold, under per-cell-band PRB limits - from the radio map at its tile,
so the score counts UEs rather than tiles and inherits whatever bias the MDT
sampling had, which is worth stating whenever it is reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import serve_intervals
from src.kpi.serving import max_rsrp


def _normalized_weights(band_labels: Sequence[str], cfg: DictConfig) -> np.ndarray:
    """Band priority weights, min-max normalised to ``[0, 1]``.

    Args:
        band_labels: Band names in the order of the radio map's band axis.
        cfg: Composed config; reads ``cfg.kpi.band_priority``.

    Returns:
        ``w_tilde_b``, shape ``[n_band]``.

    Raises:
        ValueError: When a band has no weight, or every weight is equal and the
            normalisation has no range to divide by.
    """
    priority = cfg.kpi.band_priority
    missing = [label for label in band_labels if label not in priority]
    if missing:
        raise ValueError(
            f"No kpi.band_priority weight for {', '.join(missing)}. Every band in the radio "
            "map needs one entry in configs/kpi.yaml."
        )

    weights = np.array([float(priority[label]) for label in band_labels])
    if not np.isfinite(weights).all():
        raise ValueError("kpi.band_priority weights must be finite")
    spread = weights.max() - weights.min()
    if spread == 0:
        raise ValueError(
            "kpi.band_priority gives every band the same weight, so the score cannot be "
            "normalised and would rank no configuration above another."
        )
    return (weights - weights.min()) / spread


def band_priority_score(
    rsrp: np.ndarray,
    band_labels: Sequence[str],
    mdt: pd.DataFrame,
    cfg: DictConfig,
) -> float:
    """UE-weighted share of UEs served by higher-priority bands.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        band_labels: The radio map's ``band_label``, aligned to axis 0 of
            ``rsrp``.
        mdt: Synthetic MDT; ``t_index``, ``tile_row`` and ``tile_col`` place
            each UE.
        cfg: Composed config; reads ``cfg.kpi.band_priority``,
            ``cfg.kpi.hole_dbm`` and what
            :meth:`src.kpi.capacity.CapacitySpec.from_config` reads.

    Returns:
        The score in ``[0, 1]``, larger when more UEs are served by
        higher-priority bands. **Maximised**, the only KPI that is.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, when
            the weights are unusable, when the MDT sits on a different grid, or
            when no UE stands on covered ground.

    Notes:
        A UE whose tile is a coverage hole is excluded from both sums. No band
        serves it, so it can neither raise nor lower the score. A UE blocked by
        the PRB limits stays in the denominator at weight zero, so overload
        lowers the score. Neither case is specified elsewhere; both are
        decisions recorded here.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )

    weights = _normalized_weights(band_labels, cfg)
    served = serve_intervals(rsrp, band_labels, mdt, cfg)
    row, col = served["tile_row"].to_numpy(), served["tile_col"].to_numpy()
    covered = max_rsrp(rsrp)[row, col] > float(cfg.kpi.hole_dbm)
    if not covered.any():
        raise ValueError(
            "No UE reports fall on covered ground, so the score has no denominator. The MDT "
            "and the radio map are unlikely to describe the same scenario."
        )
    band = served["band"].to_numpy()[covered]
    return float(np.where(band >= 0, weights[band], 0.0).mean())
