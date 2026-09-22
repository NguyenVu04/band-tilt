"""KPI 2 - Overlap. How often frequency layers collide, and how many cells serve.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours. Two carriers of
one cell are therefore never neighbours of each other.

:func:`effective_coverage` reads those same counts on every band and takes their
contraharmonic mean. It is what the objective scores
(docs/adr/0003-contraharmonic-objective-and-kpi-set.md).
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import covered, finite


def overlap_neighbors_per_band(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Count overlapping neighbours at each location, per band.

    A neighbour of band ``b``'s strongest transmitter is another transmitter on
    ``b`` that is itself above ``cfg.kpi.hole_dbm`` and within
    ``cfg.kpi.overlap_margin_db`` of it.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``[n_band, n_rows, n_cols]``. A band scores zero where it covers
        nothing: it has nothing to overlap with there.
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
    return np.where(covered[:, 0], counted.sum(axis=1) - 1, 0)


def overlap_neighbors(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Count overlapping neighbours at each location, summed over bands.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; as :func:`overlap_neighbors_per_band`.

    Returns:
        ``N_ov(g)``, shape ``[n_rows, n_cols]``. Uncovered locations contribute
        zero: they have nothing to overlap with.
    """
    return overlap_neighbors_per_band(rsrp, cfg).sum(axis=0)


def effective_coverage(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """How well each tile is served, over its layers, in ``[0, 1]``.

    Per band, ``lambda_b = 1 + `` :func:`overlap_neighbors_per_band` where the band
    clears ``cfg.kpi.hole_dbm``, and ``lambda_b e^(1 - lambda_b)`` peaks at exactly
    1 for a single dominant cell. That is scaled by how far the band's strongest
    cell sits between ``cfg.kpi.hole_dbm`` and ``cfg.kpi.weak_dbm``, so a server
    barely above the hole threshold scores near nothing and one at or above the
    weak threshold scores in full. The tile takes the contraharmonic mean of those
    per-band utilities, ``sum_b u_b^2 / sum_b u_b``.

    Each band is weighted by its own utility, so the result never exceeds the best
    band and an uncovered band carries no weight.
    Unlike a maximum over bands, it is not monotone in the layers present: a
    covered layer weaker than the rest lowers the tile's score, so removing it
    can raise it.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``, ``cfg.kpi.weak_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``[n_rows, n_cols]`` in ``[0, 1]``, zero where no band is covered, which
        includes every location the ray tracer found no path to.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    weak_dbm = float(cfg.kpi.weak_dbm)
    if weak_dbm <= hole_dbm:
        raise ValueError(f"kpi.weak_dbm ({weak_dbm}) must exceed kpi.hole_dbm ({hole_dbm}).")
    strongest = finite(rsrp).max(axis=1)
    multiplicity = np.where(strongest > hole_dbm, overlap_neighbors_per_band(rsrp, cfg) + 1.0, 0.0)
    strength = np.clip((strongest - hole_dbm) / (weak_dbm - hole_dbm), 0.0, 1.0)
    utility = multiplicity * np.exp(1.0 - multiplicity) * strength
    total = utility.sum(axis=0)
    # 0/0 on a tile no band covers; it scores 0.
    return np.divide((utility**2).sum(axis=0), total, out=np.zeros_like(total), where=total > 0.0)


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
    mask = covered(rsrp, cfg)
    if not mask.any():
        return float("nan")
    return float(counts[mask].mean())


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
