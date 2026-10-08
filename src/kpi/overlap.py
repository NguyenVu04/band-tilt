"""Overlap: how often co-band sectors crowd each other, and by how many.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours. Two carriers of
one sector are therefore never neighbours of each other.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import finite


def overlap_neighbors(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Count overlapping neighbours at each location, summed over bands.

    A neighbour of band ``b``'s strongest transmitter is another transmitter on
    ``b`` that is itself above ``cfg.kpi.hole_dbm`` and within
    ``cfg.kpi.overlap_margin_db`` of it.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``N_ov(g)``, shape ``[n_rows, n_cols]``. A band contributes zero where
        it covers nothing: it has nothing to overlap with there.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    margin_db = float(cfg.kpi.overlap_margin_db)

    layers = finite(rsrp)
    serving = layers.max(axis=1, keepdims=True)
    is_covered = serving > hole_dbm
    # Stated as a lower bound on the neighbour rather than as a difference:
    # `serving - layers` is NaN where both are -inf, and warns.
    counted = (layers >= serving - margin_db) & (layers > hole_dbm) & is_covered
    # The serving transmitter is within the margin of itself; drop it, but only
    # where the band is covered, or an uncovered location would count -1.
    return np.where(is_covered[:, 0], counted.sum(axis=1) - 1, 0).sum(axis=0)


def overlap_neighbor_mean_of(n_ov: np.ndarray, covered_mask: np.ndarray) -> float:
    """Average number of overlapping co-band neighbours over covered locations.

    The severity behind :func:`overlap_rate_of`'s incidence: the rate says how
    much of the map is crowded, this says how badly. Taken over covered
    locations only, because an uncovered one has no neighbours by definition and
    would otherwise pull the average down for having no coverage at all.

    Args:
        n_ov: :func:`overlap_neighbors` of the map.
        covered_mask: :func:`~src.kpi.capacity.covered` of the map.

    Returns:
        ``mean{ N_ov(g) : R_max(g) > hole_dbm }``. Minimised. NaN when nothing
        is covered, which keeps a total outage out of the average rather than
        scoring it a perfect zero.
    """
    if not covered_mask.any():
        return float("nan")
    return float(n_ov[covered_mask].mean())


def overlap_rate_of(n_ov: np.ndarray) -> float:
    """Fraction of the grid where any neighbour crowds the serving transmitter.

    Args:
        n_ov: :func:`overlap_neighbors` of the map.

    Returns:
        ``|{g : N_ov(g) > 0}| / |G|``, in ``[0, 1]``. Minimised.
    """
    return float((n_ov > 0).mean())
