"""Signal-quality KPIs: RSRP and SINR of the best server.

Every measure here is taken over covered tiles only (best-server RSRP above
``kpi.hole_dbm``), so a configuration can raise it by covering less. Read it
beside ``hole_rate``, never alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import covered, covered_best, max_rsrp, serving_sinr

# The two points every reported percentile is read at. Constants and not
# config values: the measures are named after them (rsrp_p05_dbm, sinr_p50_db).
# 5%: the cell edge. 50%: the median.
LOW_PERCENTILE = 5.0
MEDIAN_PERCENTILE = 50.0


def rsrp_percentile_dbm(rsrp: np.ndarray, cfg: DictConfig, percentile: float) -> float:
    """Best-server RSRP at a percentile, over covered tiles.

    A no-path tile is a hole for ``hole_rate`` but is simply not a sample here:
    that KPI measures the absence of coverage, this one its quality.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.
        percentile: The point to read, in ``[0, 100]``.

    Returns:
        The percentile in dBm, or ``-inf`` when nothing is covered, which keeps
        a total outage ordered below every configuration that covers something.
    """
    r_max = max_rsrp(rsrp)
    return percentiles_over(r_max[covered_best(r_max, cfg)], (percentile,))[0]


def sinr_percentile_db(
    rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig, percentile: float
) -> float:
    """Best-server SINR at a percentile, over covered tiles.

    The solver's own SINR (:func:`src.simulation.radio.solve_band`), which
    assumes every co-band transmitter is fully loaded.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.
        percentile: The point to read, in ``[0, 100]``.

    Returns:
        The percentile in dB, or ``-inf`` when no covered tile has a defined SINR.
    """
    return percentiles_over(serving_sinr(rsrp, sinr)[covered(rsrp, cfg)], (percentile,))[0]


def percentiles_over(values: np.ndarray, percentiles: Sequence[float]) -> tuple[float, ...]:
    """The finite ``values`` at each of ``percentiles``, in one sort.

    Returns:
        One value per percentile; all ``-inf`` when no value is finite, which
        keeps a total outage ordered below every configuration that covers
        something.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return tuple(float("-inf") for _ in percentiles)
    return tuple(float(v) for v in np.percentile(values, [float(p) for p in percentiles]))
