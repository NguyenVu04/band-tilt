"""KPI 3 - Mean Overlap Neighbors. How badly frequency layers collide."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.serving import overlap_neighbors


def mean_overlap_neighbors(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Average overlapping-neighbour count per evaluation location.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``sum_g N_ov(g) / |G|``, a neighbour count. Minimised.

    Notes:
        The denominator is every location, holes and non-overlapping ground
        included, so this is largely a rescaling of overlap rate rather than an
        independent measure of severity. That cost is recorded in
        docs/adr/0001-five-kpis-under-lexicographic-priority.md; it is
        nonetheless the definition to implement.
    """
    return float(overlap_neighbors(rsrp, cfg).mean())
