"""The UE KPIs: who the network did not serve, and what the rest can expect to receive.

The only KPIs counted over UE reports rather than grid tiles, so the hotspot
tiles carrying most of the traffic dominate them. The failure rate and the
throughput statistics split the reports along one mask: a report is either not
served and counted as a failure, or served and inside the throughput statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ue_service_failure_rate(served: pd.DataFrame) -> float:
    """Fraction of UE reports the serving rule left without a sector, ``1 - served share``.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output, one row per UE
            report.

    Returns:
        ``|not served| / |reports|`` in ``[0, 1]``. Minimised. The rule refuses
        nobody, so a failure is a UE with no layer above ``kpi.hole_dbm``.

    Raises:
        ValueError: When ``served`` holds no report, so the rate has no
            denominator.
    """
    if served.empty:
        raise ValueError("The UE table holds no UE, so the failure rate has no denominator.")
    return float(np.mean(served["band"].to_numpy() < 0))


def _served_throughput(served: pd.DataFrame) -> np.ndarray:
    """Estimated throughput of the served reports, in Mbit/s."""
    return served.loc[served["band"].to_numpy() >= 0, "estimated_throughput_mbps"].to_numpy(float)


def throughput_percentile_mbps(served: pd.DataFrame, percentile: float) -> float:
    """Percentile of the estimated throughput over served UE reports, in Mbit/s.

    Returns:
        ``0.0`` when no report is served, which keeps a total outage ordered
        below every configuration that serves someone. Maximised.
    """
    values = _served_throughput(served)
    return float(np.percentile(values, float(percentile))) if values.size else 0.0


def throughput_mean_mbps(served: pd.DataFrame) -> float:
    """Mean estimated throughput over served UE reports, in Mbit/s.

    Returns:
        ``0.0`` when no report is served, as :func:`throughput_percentile_mbps`.
        Maximised.
    """
    values = _served_throughput(served)
    return float(values.mean()) if values.size else 0.0
