"""Strongest signal, serving cell, and dominant cell-band — PROJECT.md section 8.

Three related quantities, each feeding a different KPI, and each defined over a
different axis of the RSRP array. Keeping them straight matters because they
look interchangeable and are not.

``max_rsrp``      the strongest signal from ANY cell-band at a location.
                  Feeds hole rate and weak rate (sections 10 and 11).
``serving_cell``  the strongest CELL, after aggregating its bands.
                  Feeds overlap rate (section 12), which is a cell-level
                  concept: two bands of one cell are not interfering
                  neighbours of each other.
``dominant_band`` the band of the strongest cell-band pair.
                  Feeds the Band Priority Score (section 14).

The open question, stated plainly
---------------------------------
PROJECT.md section 30 item 5 lists the cell-level aggregation rule across bands
as still to be determined. It is the difference between "a cell is as strong as
its best band" and "a cell combines its bands", and it changes the overlap KPI
directly. This module takes the rule from ``cfg.kpi`` rather than assuming one,
and the choice must be recorded before results are compared.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def max_rsrp(rsrp: np.ndarray) -> np.ndarray:
    """Strongest signal at each location, over every cell-band.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.

    Returns:
        ``R_max`` per grid cell, shape ``(|G|,)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is ``R_max(x) = max over (i, b) of R_ib(x)`` from PROJECT.md
        section 10 — the quantity the hole and weak thresholds are applied to.

        Locations with no signal at all stay ``-inf``, which compares correctly
        against the hole threshold with no special case.

    Example:
        >>> r_max = max_rsrp(rsrp)
    """
    # TODO(1): reduce over axis 0
    raise NotImplementedError("src.kpi.serving.max_rsrp")


def cell_rsrp(rsrp: np.ndarray, table: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Aggregate cell-band RSRP to one value per cell.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.
        table: The cell-band table, giving the cell each row belongs to.
        cfg: Composed config; uses the aggregation rule under ``cfg.kpi``.

    Returns:
        Per-cell RSRP, shape ``(n_cells, |G|)``, with cells in the sorted order
        of ``table.gcell_id``.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown aggregation rule.

    Notes:
        Overlap is a relationship between cells, not between carriers: a cell
        does not interfere with itself. Collapsing the band axis before the
        overlap computation is what enforces that.

        If the rule sums power rather than taking a maximum, convert out of dB
        first. Adding dBm values directly is a multiplication in linear terms
        and produces numbers that look reasonable and are meaningless.

    Example:
        >>> per_cell = cell_rsrp(rsrp, table, cfg)
    """
    # TODO(1): group rows of rsrp by table.gcell_id, preserving sorted cell order
    # TODO(2): apply the configured rule (max over bands, or linear power sum)
    # TODO(3): for a power sum, convert dBm -> mW, sum, convert back
    raise NotImplementedError("src.kpi.serving.cell_rsrp")


def serving_cell(per_cell: np.ndarray) -> np.ndarray:
    """Index of the strongest cell at each location.

    Args:
        per_cell: Per-cell RSRP from :func:`cell_rsrp`, shape
            ``(n_cells, |G|)``.

    Returns:
        Serving cell index per grid cell, shape ``(|G|,)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``s(x) = argmax over i of R_i(x)`` from PROJECT.md section 8.1.
        Accessibility plays no part in this selection — PROJECT.md section 3.4
        excludes it from the formulation entirely. See docs/adr/0002.

        Locations with no coverage still return an index, because ``argmax`` of
        an all ``-inf`` column is well defined and arbitrary. Callers must gate
        on the hole condition rather than trusting the index.

    Example:
        >>> s = serving_cell(per_cell)
    """
    # TODO(1): argmax over axis 0
    raise NotImplementedError("src.kpi.serving.serving_cell")


def dominant_band(rsrp: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Band of the strongest cell-band pair at each location.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.
        table: The cell-band table, mapping a row index to its band.

    Returns:
        Band index per grid cell, shape ``(|G|,)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        PROJECT.md section 8.2 defines the dominant band directly as the band of
        ``argmax over (i, b) of R_ib(g)`` — the single strongest pair, with no
        cell-level aggregation in between. This is deliberately a different rule
        from :func:`serving_cell`, so do not implement one in terms of the
        other.

    Example:
        >>> b_star = dominant_band(rsrp, table)
    """
    # TODO(1): argmax over axis 0 to get the winning cell-band row
    # TODO(2): map that row index to its band via table
    raise NotImplementedError("src.kpi.serving.dominant_band")
