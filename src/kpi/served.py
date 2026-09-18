"""KPI 3 - Served Rate. Whether the network actually serves its traffic.

The only KPI counted over UE reports rather than grid tiles, so it inherits the
UE table's sampling bias: the hotspot tiles carrying most of the traffic dominate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def served_rate(served: pd.DataFrame, band: int | None = None) -> float:
    """Fraction of UE reports the serving rule admitted.

    Takes the assignment rather than the radio map, because
    :func:`src.kpi.capacity.serve_intervals` is the expensive part of every
    measurement and the load KPIs read the same output. The caller serves once
    and reduces several times.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output, one row per UE
            report.
        band: Count only reports admitted on this band index; every band when
            None. The denominator is all reports either way, so the per-band
            rates sum to the overall one.

    Returns:
        ``|admitted| / |reports|`` in ``[0, 1]``. Maximised. A UE on a hole,
        and a UE blocked everywhere, both count as not served.

    Raises:
        ValueError: When ``served`` holds no report, so the rate has no
            denominator.
    """
    if served.empty:
        raise ValueError("The UE table holds no UE, so the served rate has no denominator.")
    assigned = served["band"].to_numpy()
    admitted = assigned >= 0 if band is None else assigned == int(band)
    return float(np.mean(admitted))
