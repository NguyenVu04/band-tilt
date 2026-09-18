"""KPI 2 - Overlap. How often frequency layers collide.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours. Two carriers of
one cell are therefore never neighbours of each other.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import finite, max_rsrp


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

    layers = finite(rsrp)
    serving = layers.max(axis=1, keepdims=True)
    covered = serving > hole_dbm
    # Stated as a lower bound on the neighbour rather than as a difference:
    # `serving - layers` is NaN where both are -inf, and warns.
    counted = (layers >= serving - margin_db) & (layers > hole_dbm) & covered
    # The serving transmitter is within the margin of itself; drop it, but only
    # where the band is covered, or an uncovered location would count -1.
    per_band = np.where(covered[:, 0], counted.sum(axis=1) - 1, 0)
    return per_band.sum(axis=0)


def overlap_neighbor_mean(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Average number of overlapping co-band neighbours over covered locations.

    The severity behind :func:`overlap_rate`'s incidence: the rate says how much
    of the map is crowded, this says how badly. Taken over covered locations
    only, because an uncovered one has no neighbours by definition and would
    otherwise pull the average down for having no coverage at all.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``mean{ N_ov(g) : R_max(g) > hole_dbm }``. Minimised. NaN when nothing
        is covered, which keeps a total outage out of the average rather than
        scoring it a perfect zero.
    """
    counts = overlap_neighbors(rsrp, cfg)
    covered = max_rsrp(rsrp) > float(cfg.kpi.hole_dbm)
    if not covered.any():
        return float("nan")
    return float(counts[covered].mean())


def overlap_rate(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Fraction of the grid where any neighbour crowds the serving transmitter.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``|{g : N_ov(g) > 0}| / |G|``, in ``[0, 1]``. Minimised.
    """
    return float((overlap_neighbors(rsrp, cfg) > 0).mean())
