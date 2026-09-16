"""KPI 1 - Coverage holes. Highest in ADR 0001's priority order.

Two readings of ``kpi.hole_dbm``:

- :func:`hole_rate` counts tiles at or below it. Reported.
- :func:`hole_desirability` softens that comparison per tile, in dB, and is what
  the search maximises. See its docstring for why the count cannot steer one.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from scipy.special import expit

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
    return float((max_rsrp(rsrp) <= float(cfg.kpi.hole_dbm)).mean())


def hole_desirability(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Coverage softened per tile in dB, then averaged over the grid.

    ``mean over tiles of sigmoid(R_max - kpi.hole_dbm)``. One is a tile
    comfortably covered, zero a deep hole, a half a tile exactly on the
    threshold.

    This exists because :func:`hole_rate` cannot steer a search. Its ``<=`` is a
    step: a tile at -119.9 dBm counts fully, one at -120.1 counts nothing, and
    one at -160 counts exactly the same as the one at -120.1. A tilt change that
    lifts a tile from -125 to -121 dBm is real physical improvement and moves the
    rate not at all, while the same change across the threshold moves it by a
    whole tile. Softening the *rate* afterwards cannot recover what the step
    threw away; the comparison has to be softened where it is made.

    A tile the ray tracer found no path to has ``R_max = -inf`` and scores
    exactly zero, so no-path ground is maximally undesirable rather than
    excluded - the same convention :func:`hole_rate` uses.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.

    Returns:
        A value in ``[0, 1]``. Maximised.
    """
    return float(expit(max_rsrp(rsrp) - float(cfg.kpi.hole_dbm)).mean())
