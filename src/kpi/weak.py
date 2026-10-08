"""Weak rate: the share of the grid covered, but only just."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def weak_rate_of(r_max: np.ndarray, cfg: DictConfig) -> float:
    """Fraction of the grid that is covered, but only just.

    Args:
        r_max: :func:`~src.kpi.capacity.max_rsrp` of the map.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and ``cfg.kpi.weak_dbm``.

    Returns:
        ``|{g : hole_dbm < R_max(g) <= weak_dbm}| / |G|``, in ``[0, 1]``.
        Minimised. The interval is open at ``hole_dbm``, which is where the hole
        rate is closed, so hole and weak cannot both hold at one location.
    """
    return float(((r_max > float(cfg.kpi.hole_dbm)) & (r_max <= float(cfg.kpi.weak_dbm))).mean())
