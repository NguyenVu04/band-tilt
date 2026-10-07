"""The UE KPIs: the throughput every UE report can expect to receive.

The only KPIs counted over UE reports rather than grid tiles, so the hotspot
tiles carrying most of the traffic dominate them. Every report counts: one the
serving rule leaves without a sector is credited 0 Mbit/s rather than left out,
so closing a hole under a UE raises these measures instead of hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ue_throughput_mbps(served: pd.DataFrame) -> np.ndarray:
    """Estimated throughput of every UE report, in Mbit/s, 0 where not served.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output, one row per UE
            report.
    """
    return served["estimated_throughput_mbps"].to_numpy(float)


def throughput_percentile_mbps(served: pd.DataFrame, percentile: float) -> float:
    """Percentile of the estimated throughput over every UE report, in Mbit/s.

    Returns:
        ``0.0`` when there is no report. Maximised.
    """
    values = ue_throughput_mbps(served)
    return float(np.percentile(values, float(percentile))) if values.size else 0.0


def throughput_mean_mbps(served: pd.DataFrame) -> float:
    """Mean estimated throughput over every UE report, in Mbit/s.

    Returns:
        ``0.0`` when there is no report. Maximised.
    """
    values = ue_throughput_mbps(served)
    return float(values.mean()) if values.size else 0.0
