"""The demand map: where the traffic actually is, as one share per grid tile.

One step, deterministic, read from the MDT alone: count the reports that landed
on each tile over the whole horizon, and normalise the raster to sum to one.
:mod:`src.optim.objective` reads each tile's count relative to the busiest
tile's and weights the tile ``w_g = 1 + r_g``, so the busiest tile counts twice
and ground the MDT never saw still counts once. The map carries no band: a UE
standing on a tile is a UE standing on a tile whichever layer ends up serving
it.

Counting rows rather than the PRBs they required is what keeps the map
independent of tilt. The PRB a report needs is a function of its SINR, which is
a function of the tilt being optimized, so a PRB-weighted map would weight the
objective by the baseline network's own coverage.

The reported KPIs stay tile-uniform, so a rate and the objective answer
different questions on purpose: how much of the *map* is bad, and how much of
the *traffic* sits where it is bad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Arrays in the artifact, keyed as written.
REPORTS = "reports"
SHARE = "share"


def reports_per_tile(mdt: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """Count the MDT reports that landed on each tile, over the whole horizon.

    Rows are counted as they stand, with no de-duplication. This simulator
    redraws every UE position independently per interval
    (:func:`src.simulation.sample.sample_positions`) and the schema carries no UE identity,
    so each row is a distinct UE by construction. Real MDT, where one UE can
    send several reports inside one interval, must be de-duplicated by UE id
    first or a stationary chatty handset outweighs a crowd.

    Args:
        mdt: The MDT table, one row per served report, carrying ``tile_row``
            and ``tile_col``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``[n_rows, n_cols]`` counts, zero on every tile the MDT never reported.

    Raises:
        ValueError: When a report falls outside the grid, which means the MDT
            and the radio map were built on different grids.
    """
    n_rows, n_cols = shape
    raster = np.zeros(shape, dtype=float)
    if mdt.empty:
        return raster

    row = mdt["tile_row"].to_numpy()
    col = mdt["tile_col"].to_numpy()
    if row.min() < 0 or row.max() >= n_rows or col.min() < 0 or col.max() >= n_cols:
        raise ValueError(
            f"MDT tiles span rows {row.min()}..{row.max()} cols {col.min()}..{col.max()}, "
            f"outside the radio map's {n_rows} x {n_cols} grid. The two were built on "
            "different grids."
        )

    count = mdt.groupby(["tile_row", "tile_col"], observed=True).size()
    rows, cols = zip(*count.index, strict=True)
    raster[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = count.to_numpy()
    return raster


def share(reports: np.ndarray) -> np.ndarray:
    """Normalise the counts to sum to one.

    Returns:
        ``[n_rows, n_cols]``, non-negative and summing to one. Uniform when the
        MDT reported nothing anywhere, so a scenario with no reports degrades to
        the equal-tile average rather than to a division by zero.
    """
    reports = np.asarray(reports, dtype=float)
    total = float(reports.sum())
    if total <= 0.0:
        return np.full(reports.shape, 1.0 / reports.size)
    return reports / total


def build(mdt: pd.DataFrame, shape: tuple[int, int], tile_size_m: float) -> dict:
    """The demand map's two rasters and the grid they sit on.

    Returns:
        The arrays :func:`save` writes, keyed as the artifact stores them.

    Raises:
        ValueError: As :func:`reports_per_tile`.
    """
    reports = reports_per_tile(mdt, shape)
    return {
        REPORTS: reports,
        SHARE: share(reports),
        "n_rows": shape[0],
        "n_cols": shape[1],
        "tile_size_m": float(tile_size_m),
    }


def save(arrays: dict, path: str | Path) -> Path:
    """Write the demand map as an ``.npz``, creating the directory. Returns the path.

    Not Parquet, unlike the other processed tables: this is a raster on the
    radio map's grid, and it is read as one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_share(cfg: DictConfig) -> np.ndarray:
    """The tile shares from ``data.output.demand_file``.

    Side effect: reads the artifact from disk on every call. Deliberately not
    cached: a rebuilt map inside the filesystem's timestamp resolution would be
    served stale, and nothing needs the cache. A search reads the share once, at
    :class:`src.optim.evaluator.Evaluator` construction, and passes it to every
    candidate; only a notebook or a re-score reaches this by leaving ``share``
    at None, a handful of times per run.

    Raises:
        FileNotFoundError: When the demand map has not been built, naming the
            stage that builds it.
    """
    path = Path(cfg.data.output.demand_file)
    if not path.is_file():
        raise FileNotFoundError(f"No {path}. Run `task preprocess` first.")
    with np.load(path, allow_pickle=False) as archive:
        return archive[SHARE].astype(float)
