"""Signal-quality KPIs: RSRP, SINR and spectral efficiency of the best server.

Every measure here is taken over covered tiles only (best-server RSRP above
``kpi.hole_dbm``), so a configuration can raise it by covering less. Read it
beside ``hole_rate``, never alone.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import covered, max_rsrp, serving_sinr, spectral_efficiency

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
    values = max_rsrp(rsrp)[covered(rsrp, cfg)]
    if values.size == 0:
        return float("-inf")
    return float(np.percentile(values, float(percentile)))


def _covered_sinr(rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Best-server SINR in dB on covered tiles, undefined values dropped."""
    values = serving_sinr(rsrp, sinr)[covered(rsrp, cfg)]
    return values[np.isfinite(values)]


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
    values = _covered_sinr(rsrp, sinr, cfg)
    if values.size == 0:
        return float("-inf")
    return float(np.percentile(values, float(percentile)))


def spectral_efficiency_percentile(
    rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig, percentile: float
) -> float:
    """Best-server Shannon spectral efficiency at a percentile, over covered tiles.

    Returns:
        ``log2(1 + SINR)`` in bit/s/Hz at the percentile, or ``-inf`` when no
        covered tile has a defined SINR. Monotone in SINR, so it equals the
        SINR percentile mapped through the same formula.
    """
    values = _covered_sinr(rsrp, sinr, cfg)
    if values.size == 0:
        return float("-inf")
    return float(np.percentile(spectral_efficiency(values), float(percentile)))


def spectral_efficiency_mean(rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig) -> float:
    """Mean best-server Shannon spectral efficiency over covered tiles.

    Returns:
        bit/s/Hz, or ``-inf`` when no covered tile has a defined SINR.
    """
    values = _covered_sinr(rsrp, sinr, cfg)
    if values.size == 0:
        return float("-inf")
    return float(spectral_efficiency(values).mean())
