"""KPI 2 - Overlap. How often frequency layers collide.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours. Two carriers of
one cell are therefore never neighbours of each other.

Both KPIs read ``kpi.overlap_margin_db``, and differ in how sharply:
:func:`overlap_rate` reports the share of the grid where a neighbour is inside
the margin, while :func:`overlap_desirability` softens the margin on the RSRP
difference itself and is what the search maximises.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from scipy.special import expit

from src.kpi.capacity import finite


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


def _separation(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Per-band separation desirability at each location.

    Returns:
        Shape ``[n_band, n_rows, n_cols]``, in ``(0, 1]``. A band with no
        co-band neighbour - including an uncovered one - has nothing to overlap
        and scores exactly one.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    margin_db = float(cfg.kpi.overlap_margin_db)

    layers = finite(rsrp)
    serving = layers.max(axis=1, keepdims=True)
    covered = serving > hole_dbm
    serves = np.zeros(layers.shape, dtype=bool)
    np.put_along_axis(serves, layers.argmax(axis=1, keepdims=True), True, axis=1)
    neighbour = (layers > hole_dbm) & covered & ~serves

    # Both operands are zero-filled before the subtraction: -inf - -inf is NaN
    # and warns, the hazard `overlap_neighbors` avoids by not subtracting.
    delta = np.where(covered, serving, 0.0) - np.where(neighbour, layers, 0.0)
    scored = np.where(neighbour, expit(delta - margin_db), 0.0)
    count = neighbour.sum(axis=1)
    return np.where(count > 0, scored.sum(axis=1) / np.maximum(count, 1), 1.0)


def overlap_desirability(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """The margin softened per co-band neighbour, then averaged over the grid.

    ``sigmoid(delta - kpi.overlap_margin_db)`` on each neighbour's RSRP
    difference below the serving transmitter, averaged over the neighbours of a
    band, then over bands and tiles. A neighbour level with the server scores
    near zero, one a margin down exactly a half, and one well clear near one.

    This softens the margin itself, where the physics is: the ``>=`` of
    :func:`overlap_neighbors` makes a neighbour 5.9 dB down a whole neighbour
    and one 6.1 dB down nothing at all, so the tilt change that moved it
    registers as a step or as nothing.

    Averaging over the neighbours reads the band's typical separation rather
    than its worst offender, and is not a count: a further neighbour raises the
    average, so a tile crowded by one layer can score below a tile crowded by
    one and shadowed by another. The count is what :func:`overlap_rate`
    reports.

    An uncovered tile has no neighbour and so scores as maximally desirable,
    exactly as it does in :func:`overlap_rate`: it has nothing to overlap with.
    That is why this KPI is never read alone - a configuration that covers
    nothing scores perfectly here, and is caught by the hole term.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``kpi.hole_dbm`` and
            ``kpi.overlap_margin_db``.

    Returns:
        A value in ``(0, 1]``. Maximised.
    """
    return float(_separation(rsrp, cfg).mean())
