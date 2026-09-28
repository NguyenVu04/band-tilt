"""UE service failure rate: the share of UE reports the network did not serve.

The only KPI counted over UE reports rather than grid tiles, so the hotspot
tiles carrying most of the traffic dominate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ue_service_failure_rate(served: pd.DataFrame) -> float:
    """Fraction of UE reports the serving rule did not admit, ``1 - served share``.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output, one row per UE
            report.

    Returns:
        ``|not admitted| / |reports|`` in ``[0, 1]``. Minimised. A UE on a hole
        and a UE blocked everywhere by the PRB ceiling both count as failed.

    Raises:
        ValueError: When ``served`` holds no report, so the rate has no
            denominator.
    """
    if served.empty:
        raise ValueError("The UE table holds no UE, so the failure rate has no denominator.")
    return float(np.mean(served["band"].to_numpy() < 0))
