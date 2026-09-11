"""KPI 5 - Weak Rate. Lowest priority in the lexicographic order."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import max_rsrp


def weak_rate(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Fraction of the grid that is covered, but only just.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and ``cfg.kpi.weak_dbm``.

    Returns:
        ``|{g : hole_dbm < R_max(g) <= weak_dbm}| / |G|``, in ``[0, 1]``.
        Minimised. The interval is half-open at both ends, so hole and weak
        cannot both hold at one location.
    """
    r_max = max_rsrp(rsrp)
    return float(((r_max > float(cfg.kpi.hole_dbm)) & (r_max <= float(cfg.kpi.weak_dbm))).mean())
