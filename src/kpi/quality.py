"""Signal-quality KPIs: RSRP and SINR of the best server.

Every measure here is taken over covered tiles only (best-server RSRP above
``kpi.hole_dbm``), so a configuration can raise it by covering less. Read it
beside the hole rate, never alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# The two points every reported percentile is read at. Constants and not
# config values: the measures are named after them (rsrp_p05_dbm, sinr_p50_db).
LOW_PERCENTILE = 5.0
MEDIAN_PERCENTILE = 50.0


def percentiles_over(values: np.ndarray, percentiles: Sequence[float]) -> tuple[float, ...]:
    """The finite ``values`` at each of ``percentiles``, in one sort.

    Returns:
        One value per percentile; all ``-inf`` when no value is finite, which
        keeps a total outage ordered below every configuration that covers
        something.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return tuple(float("-inf") for _ in percentiles)
    return tuple(float(v) for v in np.percentile(values, [float(p) for p in percentiles]))
