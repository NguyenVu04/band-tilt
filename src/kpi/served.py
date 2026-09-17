"""KPI 3 - Served Ratio. Whether the network actually serves its traffic.

The only KPI counted over UE reports rather than grid tiles, so it inherits the
UE table's sampling bias: the hotspot tiles carrying most of the traffic dominate it.
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
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> float:
    """Fraction of UEs admitted to a cell-band by the serving rule.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape; sets PRBs per UE.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: ``t_index``, ``tile_row`` and ``tile_col`` place each UE.
        cfg: Composed config; see :meth:`src.kpi.capacity.CapacitySpec.from_config`.

    Returns:
        ``|served UEs| / |UEs|`` in ``[0, 1]``. A UE is served when it fits the
        PRBs of a cell-band whose RSRP is above ``kpi.hole_dbm``; a UE on a hole
        or blocked everywhere counts as not served.

    Raises:
        ValueError: When the UE table is empty, ``band_labels`` does not match axis 0
            of ``rsrp``, or as :func:`src.kpi.capacity.serve_intervals`.
    """
    if ue.empty:
        raise ValueError("The UE table holds no UE, so the served ratio has no denominator.")
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    return float((served["band"].to_numpy() >= 0).mean())
