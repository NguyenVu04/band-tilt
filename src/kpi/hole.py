"""KPI 1 - Hole Rate. Highest priority in the lexicographic order."""

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
        ``|{g : R_max(g) <= hole_dbm}| / |G|``, in ``[0, 1]``. Minimised.
    """
    return float((max_rsrp(rsrp) <= float(cfg.kpi.hole_dbm)).mean())
