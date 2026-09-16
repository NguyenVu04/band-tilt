"""KPI 2 - Overlap. How often frequency layers collide.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours; the counts are
then summed across bands. Two carriers of one cell are therefore never
neighbours of each other.

:func:`overlap_neighbors` counts them. The two KPIs differ only in what they do
with that count: :func:`overlap_rate` asks whether it is nonzero, which is what
the hard score reads, while :func:`overlap_desirability` softens the count
itself and is what the objective reads.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import finite
from src.kpi.soft import soft_spec, soften


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


def overlap_desirability(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """The neighbour count softened per tile, then averaged over the grid.

    ``mean over tiles of sigmoid((target - N_ov) / T)``, with the target and
    temperature in neighbours from ``kpi.soft.overlap_rate``. One is a tile with
    the layers comfortably separated, zero a tile crowded by several.

    This softens the ``> 0`` of :func:`overlap_rate`, not the 6 dB margin.
    :func:`overlap_neighbors` already returns a count, and collapsing it to a
    yes/no throws away the difference between one crowding neighbour and four -
    the difference a tilt change actually moves. The margin comparison inside
    the count stays hard, so a neighbour 6.1 dB down still contributes nothing.

    An uncovered tile contributes zero neighbours and so scores as maximally
    desirable, exactly as it does in :func:`overlap_rate`: it has nothing to
    overlap with. That is why this KPI is never read alone - a configuration
    that covers nothing scores perfectly here, and is caught by the hole term.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``kpi.hole_dbm``, ``kpi.overlap_margin_db``
            and ``kpi.soft.overlap_rate``.

    Returns:
        A value in ``[0, 1]``. Maximised.

    Raises:
        ValueError: When ``kpi.soft.overlap_rate`` is unusable; see
            :func:`src.kpi.soft.soft_spec`.
    """
    neighbours = overlap_neighbors(rsrp, cfg)
    return float(soften(neighbours, soft_spec(cfg, "overlap_rate"), maximise=False).mean())
