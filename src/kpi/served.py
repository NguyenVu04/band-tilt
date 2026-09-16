"""KPI 3 - Served Ratio. Whether the network actually serves its traffic.

Two readings of the same serving rule, both maximised, and the only KPIs
counted over UE reports rather than grid tiles:

- :func:`served_ratio` is the share of UE reports served. It inherits the
  MDT's sampling bias, so it is dominated by the hotspot tiles that carry most
  of the traffic. Reported.
- :func:`served_desirability` takes the same ratio **per tile** and averages
  over occupied tiles, giving each one an equal vote. It is what the search
  maximises; see its docstring for why the order of those two steps matters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import serve_intervals


def served_ratio(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    mdt: pd.DataFrame,
    cfg: DictConfig,
) -> float:
    """Fraction of MDT UEs admitted to a cell-band by the serving rule.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape; sets PRBs per UE.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: ``t_index``, ``tile_row`` and ``tile_col`` place each UE.
        cfg: Composed config; see :meth:`src.kpi.capacity.CapacitySpec.from_config`.

    Returns:
        ``|served UEs| / |UEs|`` in ``[0, 1]``. A UE is served when it fits the
        PRBs of a cell-band whose RSRP is above ``kpi.hole_dbm``; a UE on a hole
        or blocked everywhere counts as not served.

    Raises:
        ValueError: When the MDT is empty, ``band_labels`` does not match axis 0
            of ``rsrp``, or as :func:`src.kpi.capacity.serve_intervals`.
    """
    if mdt.empty:
        raise ValueError("The MDT holds no UE, so the served ratio has no denominator.")
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    served = serve_intervals(rsrp, sinr, band_labels, mdt, cfg)
    return float((served["band"].to_numpy() >= 0).mean())


def served_desirability(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    mdt: pd.DataFrame,
    cfg: DictConfig,
) -> float:
    """The served ratio taken per tile, then averaged over occupied tiles.

    Averaging *per tile* rather than over UEs is the whole point. The global
    ratio is dominated by the hotspot tiles carrying most of the reports, so a
    badly served quiet tile is invisible in it, while this gives every occupied
    tile an equal vote. It is also smoother in tilt: tiles cross into service
    one at a time, where the global ratio moves in UE-sized jumps.

    A tile with no UE report has no served ratio and is excluded. Scoring it
    zero would punish a configuration for not covering ground nobody stands on,
    and scoring it one would dilute the result with empty map.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape; sets PRBs per UE.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: ``t_index``, ``tile_row`` and ``tile_col`` place each UE.
        cfg: Composed config; see :meth:`src.kpi.capacity.CapacitySpec.from_config`.

    Returns:
        A value in ``[0, 1]``. Maximised.

    Raises:
        ValueError: When the MDT is empty, ``band_labels`` does not match axis 0
            of ``rsrp``, or as :func:`src.kpi.capacity.serve_intervals`.
    """
    if mdt.empty:
        raise ValueError("The MDT holds no UE, so the served ratio has no denominator.")
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    served = serve_intervals(rsrp, sinr, band_labels, mdt, cfg)
    by_tile = served.assign(served=served["band"].to_numpy() >= 0).groupby(
        ["tile_row", "tile_col"], sort=False
    )["served"]
    return float(by_tile.mean().mean())
