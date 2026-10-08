"""Hole rate: the share of the grid with no usable signal."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def hole_rate_of(r_max: np.ndarray, cfg: DictConfig) -> float:
    """Fraction of the grid receiving no usable signal from any sector-band.

    Args:
        r_max: :func:`~src.kpi.capacity.max_rsrp` of the map.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.

    Returns:
        ``|{g : R_max(g) <= hole_dbm}| / |G|``, in ``[0, 1]``. Minimised. A
        tile where every layer is NaN (no path) has ``R_max = -inf`` and is
        always a hole.
    """
    return float((r_max <= float(cfg.kpi.hole_dbm)).mean())
