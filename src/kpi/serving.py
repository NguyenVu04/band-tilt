"""Coverage quantities the grid KPIs are built from.

Two reductions over the RSRP array, read by the hole, weak and overlap rates
and the Band Priority Score's hole gate. They live here so that neither is
written twice and the KPIs cannot drift apart. The band that serves a UE in the
Band Priority Score is :mod:`src.kpi.capacity`'s rule, not the strongest layer.

The overlap rule here is CO-BAND: within one band, the strongest transmitter
serves and the other transmitters on that same band are its neighbours; the
counts are then summed across bands. Two carriers of one cell are therefore
never neighbours of each other.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def _finite(rsrp: np.ndarray) -> np.ndarray:
    """RSRP with the ray tracer's no-path NaN replaced by ``-inf``.

    Substituting once, here, is what lets every threshold comparison downstream
    run without a NaN special case: an unreachable location compares as a hole
    on its own.
    """
    return np.where(np.isfinite(rsrp), rsrp, -np.inf)


def max_rsrp(rsrp: np.ndarray) -> np.ndarray:
    """Strongest signal at each location, over every cell-band layer.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.

    Returns:
        ``R_max(g)``, shape ``[n_rows, n_cols]``, ``-inf`` where nothing is
        received.
    """
    return _finite(rsrp).max(axis=(0, 1))


def overlap_neighbors(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Count overlapping neighbours at each location, summed over bands.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``N_ov(g)``, shape ``[n_rows, n_cols]``. Uncovered locations contribute
        zero: they have nothing to overlap with.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    margin_db = float(cfg.kpi.overlap_margin_db)

    finite = _finite(rsrp)
    serving = finite.max(axis=1, keepdims=True)
    covered = serving > hole_dbm
    # Stated as a lower bound on the neighbour rather than as a difference:
    # `serving - finite` is NaN where both are -inf, and warns.
    counted = (finite >= serving - margin_db) & (finite > hole_dbm) & covered
    # The serving transmitter is within the margin of itself; drop it, but only
    # where the band is covered, or an uncovered location would count -1.
    per_band = np.where(covered[:, 0], counted.sum(axis=1) - 1, 0)
    return per_band.sum(axis=0)
