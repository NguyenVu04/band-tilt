"""Spatial reductions over a radio map: coverage, demand, and what changed.

The coverage classes here cut the map at ``cfg.kpi.hole_dbm`` and
``cfg.kpi.weak_dbm``, the same thresholds :mod:`src.kpi` scores with, so the
tile-weighted numbers this module reports are the KPIs by another route. What it
adds is the demand-weighted view of the same cut — see the package docstring on
why that is a diagnostic and not an objective.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import CapacitySpec, covered_best, max_rsrp, spectral_efficiency

# Display range for RSRP images. The lower bound mirrors kpi.hole_dbm in
# configs/kpi.yaml, so the darkest colour and "uncovered" mean the same thing to
# the eye; the upper bound is chosen for contrast and carries no meaning.
RSRP_LIMITS = (-120.0, -60.0)

# Coverage classes, worst first. The order is the one a stacked bar or a legend
# should use, and the integers are what `coverage_class` returns.
COVERAGE_CLASSES = ("hole", "weak", "good")

HOLE, WEAK, GOOD = range(3)


def serving_band(rsrp: np.ndarray, sinr: np.ndarray, spec: CapacitySpec) -> np.ndarray:
    """Band a lone UE on each tile would be served on.

    The serving rule of :mod:`src.kpi.capacity` with nobody else connected: the
    candidate above ``spec.min_rsrp_dbm`` whose whole PRB pool carries the most
    throughput. With other UEs connected the choice also depends on load, so
    this is the map's static reading of the rule, not a UE's assignment.

    Args:
        rsrp: ``[n_band, n_tx, n_rows, n_cols]`` in dBm, NaN where no path.
        sinr: The solver's SINR in dB, same shape.
        spec: The capacity settings.

    Returns:
        ``[n_rows, n_cols]`` band index, ``-1`` where no layer is above
        ``spec.min_rsrp_dbm``.
    """
    n_tx = rsrp.shape[1]
    rate = spectral_efficiency(sinr) * spec.prb_bandwidth_hz[:, None, None, None]
    offer = spec.pool_prb[:, :, None, None] * rate
    candidate = (rsrp > spec.min_rsrp_dbm) & np.isfinite(offer)
    layers = np.where(candidate, offer, -np.inf).reshape(-1, *rsrp.shape[-2:])
    band = layers.argmax(axis=0) // n_tx
    return np.where(candidate.any(axis=(0, 1)), band, -1)


def coverage_class(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Each tile as hole, weak or good.

    Returns:
        ``[n_rows, n_cols]`` of :data:`HOLE`, :data:`WEAK` or :data:`GOOD`.
    """
    best = max_rsrp(rsrp)
    weak_dbm = float(cfg.kpi.weak_dbm)
    return np.where(~covered_best(best, cfg), HOLE, np.where(best <= weak_dbm, WEAK, GOOD))


def coverage_table(rsrp: np.ndarray, counts: np.ndarray, cfg: DictConfig) -> pd.DataFrame:
    """Coverage by area and by demand, one row per class.

    Returns:
        Columns ``tiles``, ``tile_share``, ``reports``, ``demand_share``.

        ``tile_share`` for the hole row is the hole rate KPI; ``demand_share``
        is the share of UE reports standing on such a tile. They can differ by
        a large factor, because holes need not fall where anyone is, and that
        difference is the reason this table exists.
    """
    classes = coverage_class(rsrp, cfg)
    total = counts.sum()
    rows = []
    for index, name in enumerate(COVERAGE_CLASSES):
        mask = classes == index
        reports = float(counts[mask].sum())
        rows.append(
            {
                "coverage": name,
                "tiles": int(mask.sum()),
                "tile_share": float(mask.mean()),
                "reports": reports,
                "demand_share": float(reports / total) if total else float("nan"),
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
    was_hole = ~covered_best(before, cfg)
    is_hole = ~covered_best(after, cfg)
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
        counts: The demand raster, :attr:`src.evaluation.compare.Configuration.demand`.
        cfg: Composed config; reads ``cfg.kpi``.
        quantile: Demand quantile, taken over occupied tiles only, above which
            a tile counts as busy. Over all tiles the empty ones would drag it
            to zero.

    Returns:
        Boolean ``[n_rows, n_cols]``. These are the tiles worth fixing, as
        opposed to the ones that are merely dark.
    """
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return np.zeros(counts.shape, dtype=bool)
    busy = counts >= np.quantile(occupied, quantile)
    return busy & (coverage_class(rsrp, cfg) != GOOD)


def tile_median(served: pd.DataFrame, column: str, shape: tuple[int, int]) -> np.ndarray:
    """Median of one column of the UE reports on each tile, over every interval.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        column: The column to reduce, e.g. ``estimated_throughput_mbps``.
        shape: The grid's ``(n_rows, n_cols)``.

    Returns:
        ``[n_rows, n_cols]``, NaN on a tile with no report or none with a value:
        "no one here" is not "everyone here got nothing".
    """
    median = served.groupby(["tile_row", "tile_col"])[column].median()
    grid = np.full(shape, np.nan)
    grid[median.index.get_level_values(0), median.index.get_level_values(1)] = median.to_numpy()
    return grid


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
