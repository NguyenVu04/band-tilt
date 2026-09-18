"""The demand map: where the traffic actually is, as one weight per grid tile.

Two steps, both deterministic and both read from the MDT alone.

1. **Median requested PRB per tile.** Each MDT report carries the PRBs the
   serving rule required for it (:func:`src.kpi.capacity.serve_intervals`). Sum
   those per tile per interval, then take the **median over the intervals in
   which that tile was reported at all**. Not over every interval in the
   horizon: a tile is reported in a handful of the horizon's intervals, so that
   median is zero for every tile and carries no information. A tile never
   reported is zero, which is a statement about the sample and not about the
   tile.
2. **Kernel density estimation.** The medians are a few thousand isolated tiles
   on a grid of a hundred thousand. An isotropic Gaussian kernel of
   ``data.demand.bandwidth_m`` spreads them into a field defined everywhere,
   which is what lets a tile with no report of its own still carry the weight of
   the traffic beside it. On a regular lattice that estimate is exactly the
   discrete Gaussian convolution of the per-tile medians, so that is how it is
   computed; mass that falls off the grid edge is dropped and the result is
   renormalised, making this a KDE restricted to the study area.

The weights sum to one and :func:`src.optim.objective.objective` averages its
per-tile utility against them, which is the only thing that reads them. The
reported KPIs stay tile-uniform, so a rate and the objective answer different
questions on purpose: how much of the *map* is bad, and how much of the
*traffic* sits where it is bad.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter

# Arrays in the artifact, keyed as written.
MEDIAN_PRB = "median_prb"
WEIGHT = "weight"


@dataclass(frozen=True)
class DemandSpec:
    """``data.demand``: how the medians become weights.

    Attributes:
        bandwidth_m: Standard deviation of the KDE's Gaussian kernel, in
            scene metres. Zero leaves the medians unsmoothed.
        uniform_share: Share of the weight spread evenly over every tile,
            blended in after the KDE. Zero lets the objective ignore a region
            the MDT never reported; raising it is how a tile far from any
            traffic keeps a floor of influence.
    """

    bandwidth_m: float
    uniform_share: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> DemandSpec:
        """Read ``data.demand``.

        Raises:
            ValueError: When ``data.demand`` is absent, ``bandwidth_m`` is
                negative, or ``uniform_share`` is outside ``[0, 1]``.
        """
        block = cfg.data.get("demand")
        if block is None:
            raise ValueError("configs/data.yaml has no `demand` block.")
        spec = cls(
            bandwidth_m=float(block.bandwidth_m),
            uniform_share=float(block.uniform_share),
        )
        if spec.bandwidth_m < 0:
            raise ValueError(
                f"data.demand.bandwidth_m must be non-negative, got {spec.bandwidth_m}"
            )
        if not 0.0 <= spec.uniform_share <= 1.0:
            raise ValueError(
                f"data.demand.uniform_share must be in [0, 1], got {spec.uniform_share}"
            )
        return spec


def median_prb_per_tile(mdt: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """Median PRBs requested on each tile, over the intervals it was reported in.

    Args:
        mdt: The MDT table, one row per served report, carrying ``tile_row``,
            ``tile_col``, ``t_index`` and ``prb_per_ue``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``[n_rows, n_cols]`` in PRBs, zero on every tile the MDT never reported.

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

    per_interval = mdt.groupby(["tile_row", "tile_col", "t_index"], observed=True)[
        "prb_per_ue"
    ].sum()
    median = per_interval.groupby(level=["tile_row", "tile_col"], observed=True).median()
    rows, cols = zip(*median.index, strict=True)
    raster[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = median.to_numpy()
    return raster


def kde_weights(median_prb: np.ndarray, tile_size_m: float, spec: DemandSpec) -> np.ndarray:
    """Turn the per-tile medians into weights that sum to one.

    The Gaussian kernel is isotropic and its bandwidth is given in metres
    rather than derived by Scott's or Silverman's rule: those scale with the
    spread of the samples, which here is the size of the study area, and would
    smooth every hotspot away. See ``docs/adr/0007``.

    Args:
        median_prb: :func:`median_prb_per_tile` output.
        tile_size_m: The grid's tile side, for converting the bandwidth to tiles.
        spec: The KDE settings.

    Returns:
        ``[n_rows, n_cols]``, non-negative and summing to one. Uniform when the
        MDT reported nothing anywhere, so a missing demand map degrades to the
        equal-tile average rather than to a division by zero.
    """
    total_demand = float(np.sum(median_prb))
    if total_demand <= 0.0:
        return np.full(np.shape(median_prb), 1.0 / np.size(median_prb))

    sigma = spec.bandwidth_m / float(tile_size_m)
    # mode="constant" is the KDE truncated at the study area: kernel mass that
    # falls off the edge is dropped, and renormalising restores the total.
    density = gaussian_filter(np.asarray(median_prb, dtype=float), sigma=sigma, mode="constant")
    density = np.clip(density, 0.0, None)
    weight = density / density.sum()
    if spec.uniform_share > 0.0:
        uniform = 1.0 / np.size(weight)
        weight = (1.0 - spec.uniform_share) * weight + spec.uniform_share * uniform
    return weight


def build(mdt: pd.DataFrame, shape: tuple[int, int], tile_size_m: float, cfg: DictConfig) -> dict:
    """The demand map's two rasters and the settings behind them.

    Returns:
        The arrays :func:`save` writes, keyed as the artifact stores them.

    Raises:
        ValueError: As :meth:`DemandSpec.from_config` and
            :func:`median_prb_per_tile`.
    """
    spec = DemandSpec.from_config(cfg)
    median_prb = median_prb_per_tile(mdt, shape)
    return {
        MEDIAN_PRB: median_prb,
        WEIGHT: kde_weights(median_prb, tile_size_m, spec),
        "n_rows": shape[0],
        "n_cols": shape[1],
        "tile_size_m": float(tile_size_m),
        "bandwidth_m": spec.bandwidth_m,
        "uniform_share": spec.uniform_share,
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


def load_weights(cfg: DictConfig) -> np.ndarray:
    """The tile weights from ``data.output.demand_file``.

    Side effect: reads the artifact from disk on every call. Deliberately not
    cached: a rebuilt map inside the filesystem's timestamp resolution would be
    served stale, and nothing needs the cache. A search reads the weights once,
    at :class:`src.optim.evaluator.Evaluator` construction, and passes them to
    every candidate; only a notebook or a re-score reaches this by leaving
    ``weights`` at None, a handful of times per run.

    Raises:
        FileNotFoundError: When the demand map has not been built, naming the
            stage that builds it.
    """
    path = Path(cfg.data.output.demand_file)
    if not path.is_file():
        raise FileNotFoundError(f"No {path}. Run `task preprocess` first.")
    with np.load(path, allow_pickle=False) as archive:
        return archive[WEIGHT].astype(float)
