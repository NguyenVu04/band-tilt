"""KPI 4 - the UE-weighted Band Priority Score.

The three coverage rates say nothing about WHICH layer serves a location; this
one asks whether the layers the project would rather use are the ones serving
the ground the users actually stand on. Maximised.

The radio map decides the dominant band per grid tile; the MDT supplies the UE
weight for that tile. Weighting is the point: two tiles with the same dominant
band contribute differently when one holds a hundred UE reports and the other
holds ten. The score therefore inherits whatever bias the MDT sampling had,
which is worth stating whenever it is reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.serving import dominant_band, max_rsrp
from src.kpi.tiles import tile_index


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


def ue_counts(mdt: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """UE reports per grid tile, shape ``[n_rows, n_cols]``.

    Public, though not exported from :mod:`src.kpi`, so that anything reporting
    on where the demand is weights tiles by the same definition this score
    does. Two copies of it would be free to drift apart.

    Args:
        mdt: Synthetic MDT, one row per UE per interval, carrying ``tile_row``
            and ``tile_col``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``rho(g)``, the report count per tile. Rows accumulate across intervals,
        which is what makes this the UE spatial distribution rather than one
        snapshot of it.

    Raises:
        ValueError: When a UE falls outside the map's grid; see
            :func:`src.kpi.tiles.tile_index`.
    """
    n_rows, n_cols = shape
    row, col = tile_index(mdt, shape)
    flat = np.bincount(row * n_cols + col, minlength=n_rows * n_cols)
    return flat.reshape(n_rows, n_cols)


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
        mdt: Synthetic MDT, supplying the UE weight of each grid tile.
        cfg: Composed config; reads ``cfg.kpi.band_priority`` and
            ``cfg.kpi.hole_dbm``.

    Returns:
        The score in ``[0, 1]``, larger when more UEs are served by
        higher-priority bands. **Maximised**, the only KPI that is.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, when
            the weights are unusable, when the MDT sits on a different grid, or
            when no UE stands on covered ground.

    Notes:
        A UE whose tile is a coverage hole is excluded from both sums. No band
        serves it, so it can neither raise nor lower the score. This case is
        not specified elsewhere; it is a decision recorded here.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )

    weights = _normalized_weights(band_labels, cfg)
    counts = ue_counts(mdt, rsrp.shape[-2:])
    dominant = dominant_band(rsrp)
    served = counts * (max_rsrp(rsrp) > float(cfg.kpi.hole_dbm))

    total = served.sum()
    if total == 0:
        raise ValueError(
            "No UE reports fall on covered ground, so the score has no denominator. The MDT "
            "and the radio map are unlikely to describe the same scenario."
        )
    return float((served * weights[dominant]).sum() / total)
