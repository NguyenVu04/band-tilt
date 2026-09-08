"""Where the MDT's UE reports sit on the radio map's grid.

Two KPIs weight the map by the UE population - the Band Priority Score and the
Expected RSRP Improvement - and both need the same tile indices under the same
guarantee that the MDT and the map were built on one grid. Extracted so the
check exists once: two copies would be free to disagree about which grid a
report belongs to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tile_index(mdt: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The MDT's ``(row, col)`` tile indices, checked against the map's grid.

    Args:
        mdt: Synthetic MDT, one row per UE per interval, carrying ``tile_row``
            and ``tile_col``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``(row, col)``, each of shape ``[n_report]``, ready to index a
        ``[n_rows, n_cols]`` raster.

    Raises:
        ValueError: When a UE falls outside the map's grid, which means the MDT
            and the radio map were built on different grids.
    """
    n_rows, n_cols = shape
    row = mdt["tile_row"].to_numpy()
    col = mdt["tile_col"].to_numpy()
    if row.min() < 0 or row.max() >= n_rows or col.min() < 0 or col.max() >= n_cols:
        raise ValueError(
            f"MDT tiles span rows {row.min()}..{row.max()} cols {col.min()}..{col.max()}, "
            f"outside the radio map's {n_rows} x {n_cols} grid. The two were built on "
            "different grids."
        )
    return row, col
