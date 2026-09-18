"""KPI 5 - Signal quality percentiles. The strength of coverage where coverage exists.

Reported, never optimised. See :func:`rsrp_percentile_dbm` for why that
distinction is load-bearing rather than incidental.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import max_rsrp, serving_sinr

# The two points every reported percentile is read at. Constants and not
# config values: the measures are named after them (rsrp_p05_dbm, sinr_p50_db),
# so a config that moved them would make every column name a lie.
#
# 5%: the cell-edge point of 3GPP TR 36.814 Annex A.2.1.4, carried from user
# throughput to RSRP and SINR. 50%: the median, which needs no authority.
LOW_PERCENTILE = 5.0
MEDIAN_PERCENTILE = 50.0


def _covered(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Mask of locations receiving something above ``cfg.kpi.hole_dbm``."""
    return max_rsrp(rsrp) > float(cfg.kpi.hole_dbm)


def rsrp_percentile_dbm(rsrp: np.ndarray, cfg: DictConfig, percentile: float) -> float:
    """Serving RSRP at a percentile, over covered locations.

    Read at :data:`LOW_PERCENTILE` and :data:`MEDIAN_PERCENTILE`: the cell edge
    and the typical location.

    Unlike the rates this measure is **conditional on coverage**: the
    percentile is taken over locations receiving something above
    ``kpi.hole_dbm``, so a configuration can raise it by covering less. It is
    therefore read beside ``hole_rate``, never alone, and never enters the
    objective. Note the deliberate asymmetry with :func:`src.kpi.hole.hole_rate`,
    which counts a no-path location as a hole: that KPI measures the absence of
    coverage, this one the quality of coverage where it exists.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.
        percentile: The point to read, in ``[0, 100]``.

    Returns:
        The percentile in dBm, or ``-inf`` when nothing is covered, which keeps
        the total outage case ordered below every configuration that covers
        something.
    """
    values = max_rsrp(rsrp)[_covered(rsrp, cfg)]
    if values.size == 0:
        return float("-inf")
    return float(np.percentile(values, float(percentile)))


def sinr_percentile_db(
    rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig, percentile: float
) -> float:
    """Best-server SINR at a percentile, over covered locations.

    The SINR of the layer :func:`src.kpi.capacity.max_rsrp` reports, so this and
    :func:`rsrp_percentile_dbm` describe the same cell-band at every location.
    It is the solver's own SINR (:func:`src.simulation.radio.solve_band`), which
    assumes every co-band transmitter is fully loaded; it is not recomputed
    against the PRB load the serving rule produces.

    Conditional on coverage for the same reason as
    :func:`rsrp_percentile_dbm`, and read beside ``hole_rate`` likewise.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.
        percentile: The point to read, in ``[0, 100]``.

    Returns:
        The percentile in dB, or ``-inf`` when nothing is covered. A covered
        location whose SINR the solver left undefined is dropped, so a
        percentile is never NaN while anything is covered.
    """
    values = serving_sinr(rsrp, sinr)[_covered(rsrp, cfg)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    return float(np.percentile(values, float(percentile)))
