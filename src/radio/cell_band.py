"""The cell-band table — the atomic unit of the decision variable.

The project optimises one absolute tilt per ``(cell, band)`` pair, so the
project's fundamental object is not a cell and not a band but the pair. This
module builds the table of those pairs and pins down their order.

Row order is the contract
-------------------------
Every tilt vector in this project — the BO search space, a MARL action, a
surrogate feature row, a reported result — is indexed by position in this table.
So the order must be deterministic and stable: sort explicitly, and never rely
on the order a CSV happened to arrive in or on a groupby that preserves it by
accident. Two runs that build the table differently will produce tilt vectors
that mean different things while comparing as equal.

Where the bands come from
-------------------------
Today's export has no band column, so the pairs are formed by applying the band
declaration in ``configs/radio.yaml`` to each configured cell. When the
multi-band export arrives, set ``cfg.radio.cells.from_cell_config`` and the
table is read from the data instead. Nothing downstream changes: no module
hardcodes the number of bands. See the cell configuration schema in
``configs/data.yaml``.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def build_table(cells: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Build the ordered cell-band table.

    Args:
        cells: Cell configuration from :func:`src.data.load.load_cell_config`.
        cfg: Composed config; uses ``cfg.radio.bands`` and ``cfg.radio.cells``.

    Returns:
        One row per ``(cell, band)`` pair, carrying the cell geometry, the band
        parameters, the current absolute tilt and the tilt bounds. Sorted by
        ``(gcell_id, band)`` and indexed from zero.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when a band named in ``cfg.radio.cells``
            is not declared in ``cfg.radio.bands``.

    Notes:
        The returned index is the canonical ordering of the decision vector.
        Write it down with any result that is saved, so a tilt vector can be
        interpreted later.

    Example:
        >>> table = build_table(load_cell_config(cfg), cfg)
        >>> len(table)
    """
    # TODO(1): cross cells with the band list from cfg.radio.cells (default + overrides)
    # TODO(2): raise ValueError for any band not declared in cfg.radio.bands
    # TODO(3): attach carrier_hz, tx_power_dbm, priority_weight, tilt bounds per band
    # TODO(4): sort by (gcell_id, band) and reset_index(drop=True)
    raise NotImplementedError("src.radio.cell_band.build_table")


def current_tilt(table: pd.DataFrame) -> np.ndarray:
    """Extract the network as deployed, current_tilt.

    Args:
        table: The cell-band table from :func:`build_table`.

    Returns:
        Absolute tilt per cell-band in degrees, ordered to match ``table``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This vector is the baseline every reported result is measured against,
        and the starting point for the MARL reset
        strategy ``baseline``. It is also the only input to the tilt offset in
        :func:`tilt_offset`.

    Example:
        >>> tilt_0 = current_tilt(table)
    """
    # TODO(1): read the absolute tilt column produced by build_table
    # TODO(2): return as a float array in table order
    raise NotImplementedError("src.radio.cell_band.current_tilt")


def tilt_bounds(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract the per-cell-band tilt bounds that define the feasible set.

    Args:
        table: The cell-band table from :func:`build_table`.

    Returns:
        ``(lower, upper)``, each of length ``len(table)``, in degrees.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        These two vectors are the whole of the tilt-bound constraint. They are
        consumed by :mod:`src.optim.space`, which is what
        makes BO and MARL provably search the same set.

    Example:
        >>> lower, upper = tilt_bounds(table)
    """
    # TODO(1): read the tilt min/max columns produced by build_table
    # TODO(2): assert lower < upper elementwise before returning
    raise NotImplementedError("src.radio.cell_band.tilt_bounds")


def tilt_offset(tilt: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Derive the tilt change from the current configuration, for reporting only.

    Args:
        tilt: An absolute tilt configuration, ordered to match ``table``.
        table: The cell-band table from :func:`build_table`.

    Returns:
        One row per cell-band with current tilt, optimal tilt and the offset —
        the deliverable tilt table, rendered by :mod:`src.evaluation.report`.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The offset is computed AFTER optimization and never enters it. It is not
        a decision variable, not a KPI, and carries no penalty: the research
        question is network quality, not minimal reconfiguration. The absolute
        tilt is the decision variable — see :mod:`src.optim.space`.

        If a deployment later caps how far a tilt may move in one step, that is
        an operational constraint on the feasible set — express it by narrowing
        the bounds in ``configs/radio.yaml``, not by penalising the objective.

    Example:
        >>> report = tilt_offset(optimized_tilt, table)
        >>> report.columns.tolist()
        ['gcell_id', 'band', 'current_tilt', 'optimal_tilt', 'tilt_offset']
    """
    # TODO(1): assemble gcell_id, band, current_tilt, optimal_tilt
    # TODO(2): tilt_offset = optimal_tilt - current_tilt
    raise NotImplementedError("src.radio.cell_band.tilt_offset")
