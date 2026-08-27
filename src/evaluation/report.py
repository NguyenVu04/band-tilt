"""The final deliverable — PROJECT.md section 27.

The tilt table is what a network engineer actually receives::

    | Cell | Band | Current Tilt | Optimal Tilt | Tilt Offset |

Everything else in this project exists to fill in that fourth column.

The offset is derived here and nowhere else
-------------------------------------------
``delta = theta* - theta_current`` is computed after optimization, purely to
communicate how far each antenna has to move. It is not a decision variable, not
a KPI, and carries no penalty — PROJECT.md sections 3.2 and 3.3 are explicit
that the research question is network quality, not minimal reconfiguration. See
docs/adr/0001.

If a deployment caps how far a tilt may move in one step, that is an operational
constraint on the feasible set: narrow the bounds in ``configs/radio.yaml`` and
re-run. Do not add a penalty term, and do not post-process this table to make
the offsets smaller — the result would no longer be the configuration that was
validated.

State the provenance
--------------------
Every reported KPI comes from Sionna-RT (PROJECT.md section 26). Where a
surrogate prediction is shown, label it. A table that mixes the two without
saying so is the single most misleading artifact this project can produce.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def tilt_table(theta_star: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Build the per-cell-band tilt report — PROJECT.md section 27.1.

    Args:
        theta_star: The optimized configuration in degrees, in cell-band table
            order.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.

    Returns:
        Columns ``gcell_id``, ``band``, ``current_tilt``, ``optimal_tilt``,
        ``tilt_offset``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Delegates to :func:`src.radio.cell_band.tilt_offset` rather than
        subtracting here. One definition of the offset, in the module that owns
        the cell-band table.

        Sort by absolute offset when presenting. The rows a reader needs are the
        antennas that move most, and cell identifiers are opaque hashes that
        sort into no useful order.

    Example:
        >>> tilt_table(theta_star, table)
    """
    # TODO(1): cell_band.tilt_offset(theta_star, table)
    # TODO(2): sort by absolute offset, descending, for presentation
    raise NotImplementedError("src.evaluation.report.tilt_table")


def summary(results: dict, tilts: pd.DataFrame, cfg: DictConfig) -> dict:
    """Assemble the full result record for one method.

    Args:
        results: Validation output from :mod:`src.evaluation.validate`.
        tilts: The tilt table from :func:`tilt_table`.
        cfg: Composed config.

    Returns:
        The KPIs, the tilt table, the evaluation cost, and the provenance needed
        to reproduce the run.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Include the provenance: the seed, the cell-band table ordering, the grid
        geometry, the ray-tracing settings, and the surrogate artifact used. A
        theta vector without its column ordering cannot be interpreted later,
        and a KPI without its grid resolution cannot be compared against
        anything.

    Example:
        >>> record = summary(results, tilts, cfg)
    """
    # TODO(1): assemble KPIs, tilt table, and Sionna-RT evaluation count
    # TODO(2): attach seed, cell-band ordering, grid geometry, ray-tracing settings
    # TODO(3): label every KPI with whether it came from Sionna-RT or the surrogate
    raise NotImplementedError("src.evaluation.report.summary")


def export(records: dict, path: str | Path) -> None:
    """Write the final result set to ``reports/results/``.

    Args:
        records: One summary per method, from :func:`summary`.
        path: Destination directory.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Write the tilt tables as CSV and the KPI comparison as both CSV and
        Markdown. The CSV is what gets loaded again; the Markdown is what gets
        read, and a result nobody reads has not been delivered.

    Example:
        >>> export(records, "reports/results")
    """
    # TODO(1): mkdir the destination
    # TODO(2): one tilt CSV per method, plus the combined KPI comparison
    # TODO(3): render the KPI comparison as Markdown as well
    raise NotImplementedError("src.evaluation.report.export")
