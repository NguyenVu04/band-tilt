"""KPI 5 - Edge RSRP. The strength of coverage where coverage exists.

Reported, never optimised. See :func:`edge_rsrp_dbm` for why that distinction
is load-bearing rather than incidental.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import max_rsrp


def edge_rsrp_dbm(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Serving RSRP at the cell edge: a low percentile over covered locations.

    The 5% point is the cell-edge measure of 3GPP TR 36.814 Annex A.2.1.4,
    carried here from user throughput to RSRP.

    Unlike the other KPIs this one is **conditional on coverage**: the
    percentile is taken over locations receiving something above
    ``kpi.hole_dbm``, so a configuration can raise it by covering less. It is
    therefore read beside ``hole_rate``, never alone, and never enters the
    score. Note the deliberate asymmetry with :func:`src.kpi.hole.hole_rate`,
    which counts a no-path location as a hole: that KPI measures the absence of
    coverage, this one the quality of coverage where it exists.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.quality.edge_percentile``.

    Returns:
        The percentile in dBm, or ``-inf`` when nothing is covered, which keeps
        the total outage case ordered below every configuration that covers
        something.
    """
    covered = max_rsrp(rsrp)
    covered = covered[covered > float(cfg.kpi.hole_dbm)]
    if covered.size == 0:
        return float("-inf")
    return float(np.percentile(covered, float(cfg.kpi.quality.edge_percentile)))
