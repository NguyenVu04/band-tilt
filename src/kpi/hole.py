"""Hole rate: the share of the grid with no usable signal."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import max_rsrp


def hole_rate(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Fraction of the grid receiving no usable signal from any cell-band.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.

    Returns:
        ``|{g : R_max(g) <= hole_dbm}| / |G|``, in ``[0, 1]``. Minimised. A
        tile where every layer is NaN (no path) has ``R_max = -inf`` and is
        always a hole.
    """
    return hole_rate_of(max_rsrp(rsrp), cfg)


def hole_rate_of(r_max: np.ndarray, cfg: DictConfig) -> float:
    """:func:`hole_rate` from a :func:`~src.kpi.capacity.max_rsrp` already taken."""
    return float((r_max <= float(cfg.kpi.hole_dbm)).mean())
