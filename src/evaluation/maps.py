"""Spatial reductions over a radio map: coverage, demand, and what changed.

The coverage classes here cut the map at ``cfg.kpi.hole_dbm`` and
``cfg.kpi.weak_dbm``, the same thresholds :mod:`src.kpi` scores with, so the
tile-weighted numbers this module reports are the KPIs by another route. What it
adds is the demand-weighted view of the same cut — see the package docstring on
why that is a diagnostic and not an objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import CapacitySpec, demand_prb, max_rsrp, sinr_db

# Display range for RSRP images. The lower bound is the hole threshold, so the
# darkest colour and "uncovered" mean the same thing to the eye; the upper bound
# is chosen for contrast and carries no meaning.
RSRP_LIMITS = (-120.0, -60.0)

# Display range for SINR images, chosen for contrast; it carries no meaning.
SINR_LIMITS = (-20.0, 30.0)

# Coverage classes, worst first. The order is the one a stacked bar or a legend
# should use, and the integers are what `coverage_class` returns.
COVERAGE_CLASSES = ("hole", "weak", "good")

HOLE, WEAK, GOOD = range(3)


def best_server(rsrp: np.ndarray) -> np.ndarray:
    """Strongest RSRP at each tile over every cell-band layer.

    Args:
        rsrp: ``[n_band, n_tx, n_rows, n_cols]`` in dBm, NaN where no path.

    Returns:
        ``[n_rows, n_cols]`` in dBm, ``-inf`` where nothing is received.
    """
    return max_rsrp(rsrp)


def best_sinr(rsrp: np.ndarray, band_labels: Sequence[str], cfg: DictConfig) -> np.ndarray:
    """Highest SINR at each tile over every cell-band layer, in dB.

    Recomputed from RSRP by :func:`src.kpi.capacity.sinr_db`, so a map that
    stores no SINR - an optimizer's or the surrogate's - draws the same way.

    Returns:
        ``[n_rows, n_cols]`` in dB, ``-inf`` where nothing is received.
    """
    spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    return sinr_db(rsrp, spec.noise_dbm).max(axis=(0, 1))


def coverage_class(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Each tile as hole, weak or good.

    Returns:
        ``[n_rows, n_cols]`` of :data:`HOLE`, :data:`WEAK` or :data:`GOOD`.
    """
    best = best_server(rsrp)
    hole_dbm = float(cfg.kpi.hole_dbm)
    weak_dbm = float(cfg.kpi.weak_dbm)
    return np.where(best <= hole_dbm, HOLE, np.where(best <= weak_dbm, WEAK, GOOD))


def demand(
    rsrp: np.ndarray, band_labels: Sequence[str], mdt: pd.DataFrame, cfg: DictConfig
) -> np.ndarray:
    """PRBs required per tile in its busiest interval; see :func:`src.kpi.capacity.demand_prb`.

    Depends on the map: SINR, and so PRBs per UE, move with the tilts.
    """
    return demand_prb(rsrp, band_labels, mdt, cfg)


def coverage_table(rsrp: np.ndarray, counts: np.ndarray, cfg: DictConfig) -> pd.DataFrame:
    """Coverage by area and by demand, one row per class.

    Returns:
        Columns ``tiles``, ``tile_share``, ``prb``, ``demand_share``.

        ``tile_share`` for the hole row is the hole rate KPI; ``demand_share``
        is the share of PRB demand standing on such a tile. They can differ by
        a large factor, because holes need not fall where anyone is, and that
        difference is the reason this table exists.
    """
    classes = coverage_class(rsrp, cfg)
    total = counts.sum()
    rows = []
    for index, name in enumerate(COVERAGE_CLASSES):
        mask = classes == index
        prb = float(counts[mask].sum())
        rows.append(
            {
                "coverage": name,
                "tiles": int(mask.sum()),
                "tile_share": float(mask.mean()),
                "prb": prb,
                "demand_share": float(prb / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def change_mask(before: np.ndarray, after: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Which tiles crossed the hole threshold, and in which direction.

    Args:
        before: Best-server RSRP before, ``[n_rows, n_cols]``.
        after: Best-server RSRP after, same shape.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.

    Returns:
        ``+1`` where a hole was filled, ``-1`` where one was opened, ``0``
        where the tile stayed on the same side. Deliberately not a signed RSRP
        difference: a tile gaining 3 dB while remaining a hole has not changed
        anything a KPI can see.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    was_hole = before <= hole_dbm
    is_hole = after <= hole_dbm
    return np.where(was_hole & ~is_hole, 1, np.where(~was_hole & is_hole, -1, 0))


def underserved(
    rsrp: np.ndarray,
    counts: np.ndarray,
    cfg: DictConfig,
    quantile: float = 0.75,
) -> np.ndarray:
    """Tiles carrying real demand that are not well covered.

    Args:
        rsrp: The radio map.
        counts: The demand raster from :func:`demand`.
        cfg: Composed config; reads ``cfg.kpi``.
        quantile: Demand quantile, taken over occupied tiles only, above which
            a tile counts as busy. Over all tiles it would be meaningless here,
            since about half of them hold no report at all.

    Returns:
        Boolean ``[n_rows, n_cols]``. These are the tiles worth fixing, as
        opposed to the ones that are merely dark.
    """
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return np.zeros(counts.shape, dtype=bool)
    busy = counts >= np.quantile(occupied, quantile)
    return busy & (coverage_class(rsrp, cfg) != GOOD)


def coverage_cdf(best: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Share of tiles, and of demand, at or below each RSRP level.

    Args:
        best: Best-server RSRP, ``[n_rows, n_cols]``, possibly ``-inf``.
        counts: The demand raster.

    Returns:
        ``(levels, tile_share, demand_share)``, each 1-D and aligned. Unreached
        tiles are placed at the lowest finite level so the curves start
        together; their share is exactly the uncovered fraction.
    """
    flat = np.asarray(best, dtype=float).ravel()
    weights = np.asarray(counts, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    floor = finite.min() if finite.size else 0.0
    flat = np.where(np.isfinite(flat), flat, floor)

    order = np.argsort(flat)
    levels = flat[order]
    tile_share = np.arange(1, levels.size + 1) / levels.size
    ordered_weights = weights[order]
    total = ordered_weights.sum()
    demand_share = np.cumsum(ordered_weights) / total if total else np.zeros_like(tile_share)
    return levels, tile_share, demand_share


def extent_of(radio: dict[str, Any]) -> list[float]:
    """Metric bounds of the grid, as matplotlib's ``imshow`` extent.

    Slightly larger than the scene: the grid rounds up to whole tiles, so the
    top row and right column overhang.
    """
    origin_x, origin_y = float(radio["origin_x"]), float(radio["origin_y"])
    tile = float(radio["tile_size_m"])
    return [
        origin_x,
        origin_x + int(radio["n_cols"]) * tile,
        origin_y,
        origin_y + int(radio["n_rows"]) * tile,
    ]


def grid_shape(radio: dict[str, Any]) -> tuple[int, int]:
    """The grid's ``(n_rows, n_cols)``."""
    return int(radio["n_rows"]), int(radio["n_cols"])
