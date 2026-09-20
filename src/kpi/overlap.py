"""KPI 2 - Overlap. How often frequency layers collide, and how many cells serve.

The overlap rule is CO-BAND: within one band, the strongest transmitter serves
and the other transmitters on that same band are its neighbours. Two carriers of
one cell are therefore never neighbours of each other.

:func:`serving_multiplicity` reads those same counts on the one band a tile
would be served on. It is what the objective scores
(docs/adr/0009-effective-coverage-objective.md).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omegaconf import DictConfig

from src.kpi.capacity import band_rank, finite, max_rsrp


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


def serving_multiplicity(
    rsrp: np.ndarray, cfg: DictConfig, band_labels: Sequence[str]
) -> np.ndarray:
    """Cells effectively serving each tile, on the band that would serve it.

    ``lambda(g)``: the band is the most preferred one whose strongest cell clears
    ``cfg.kpi.hole_dbm``, and the count is that band's serving transmitter plus
    its overlapping neighbours. One is the ideal: a single cell dominating the
    tile, with nothing else within the margin to contend with it.

    The band is picked by ``kpi.capacity.band_preference``, which is also how
    :func:`src.kpi.capacity.serve_intervals` admits UEs. The two agree only
    while ``kpi.capacity.rsrp_threshold_dbm`` equals ``kpi.hole_dbm``, as the
    committed config has it; raise the threshold and the serving rule starts
    skipping a preferred band that this count would still score.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``,
            ``cfg.kpi.overlap_margin_db`` and ``cfg.kpi.capacity.band_preference``.
        band_labels: Band names aligned to axis 0 of ``rsrp``; they carry the
            preference order onto the array.

    Returns:
        ``[n_rows, n_cols]``, at least one where any band is covered and zero
        where none is, which includes every location the ray tracer found no
        path to.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, or
            as :func:`src.kpi.capacity.band_rank`.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    available = finite(rsrp).max(axis=1) > float(cfg.kpi.hole_dbm)
    # An unavailable band is ranked past every real one rather than dropped, so
    # argmin returns a valid band axis position even where nothing is covered;
    # the `available.any` below is what zeroes those tiles.
    rank = np.where(available, band_rank(cfg, band_labels)[:, None, None], len(band_labels))
    served_on = rank.argmin(axis=0)
    neighbors = np.take_along_axis(overlap_neighbors_per_band(rsrp, cfg), served_on[None], axis=0)
    return np.where(available.any(axis=0), neighbors[0] + 1.0, 0.0)


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
