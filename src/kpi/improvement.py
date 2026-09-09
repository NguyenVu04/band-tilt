"""KPI 3 - the Expected RSRP Improvement over the MDT measurement locations.

The four coverage KPIs score a candidate map on its own terms. This one is the
only KPI that scores it against what the network is doing today: it compares the
serving RSRP each UE actually reported with the serving RSRP the candidate tilts
would give that UE's tile.

``R_real`` comes from the MDT and does not depend on the tilt, so this KPI
carries the measurement noise and the censoring of the reporting model as a
fixed reference. Because the sigmoid is nonlinear, measurement noise can change
the ranking of candidates. The baseline need not score exactly 0.5; compare
against its measured KPI rather than treating 0.5 as "no change".

Weighted by UE report, like the Band Priority Score and unlike the three
grid-uniform rates. Ground with no reports on it carries no weight at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.serving import max_rsrp
from src.kpi.tiles import tile_index

_RSRP_PREFIX = "rsrp_"


def measured_serving_rsrp(mdt: pd.DataFrame) -> np.ndarray:
    """``R_real(u)``: the strongest RSRP each UE reported, shape ``[n_report]``.

    A NaN in an ``rsrp_*`` column means *no path* or *not reported* and the file
    cannot tell the two apart (``src.data.build.build_mdt``). Either way the UE
    did not report that cell, so it cannot be the one serving it.

    Raises:
        ValueError: When the frame carries no ``rsrp_*`` columns, or when a row
            reports nothing at all. ``src.data.schema.verify`` rejects the
            latter, so reaching it means this MDT did not come through that gate.
    """
    columns = [column for column in mdt.columns if column.startswith(_RSRP_PREFIX)]
    if not columns:
        raise ValueError(
            f"No {_RSRP_PREFIX}* columns in the MDT, so no UE reported a serving cell. This "
            "frame is not the one src.data.build writes."
        )

    values = mdt[columns].to_numpy(dtype=np.float64)
    heard = np.isfinite(values)
    silent = int((~heard.any(axis=1)).sum())
    if silent:
        raise ValueError(
            f"{silent} MDT rows report no measurement at all, so they have no serving RSRP. "
            "src.data.schema.verify rejects this; regenerate with `python -m src.data.build`."
        )
    # -inf rather than NaN so the reduction needs no all-NaN special case; the
    # check above has already established every row has something finite in it.
    return np.where(heard, values, -np.inf).max(axis=1)


def expected_rsrp_improvement(rsrp: np.ndarray, mdt: pd.DataFrame, cfg: DictConfig) -> float:
    """Mean sigmoid of the serving-RSRP change over the MDT locations.

    Args:
        rsrp: Candidate RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        mdt: Synthetic MDT, supplying both the reported serving RSRP and the
            tile each report was made on.
        cfg: Composed config; reads ``cfg.kpi.rsrp_improvement_tau_db``.

    Returns:
        ``K_EI`` in ``[0, 1]``: the mean sigmoid, not the sigmoid of the mean
        RSRP change. **Maximised**, along with the Band Priority Score.

    Raises:
        ValueError: When ``tau`` is not positive, when the MDT sits on a
            different grid, or when it carries no usable measurements.
    """
    tau = float(cfg.kpi.rsrp_improvement_tau_db)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(
            f"kpi.rsrp_improvement_tau_db must be positive, got {tau}. At zero the sigmoid "
            "argument is a division by zero; below it the score is inverted."
        )

    row, col = tile_index(mdt, rsrp.shape[-2:])
    simulated = max_rsrp(rsrp)[row, col]
    delta = simulated - measured_serving_rsrp(mdt)

    # tanh form of the logistic sigmoid: 1/(1+e^-x) == (1 + tanh(x/2))/2. A tile
    # the candidate turned into a hole gives -inf here, where np.exp overflows
    # and warns; np.tanh saturates at -1 and returns the correct 0.0 quietly.
    return float((0.5 * (1.0 + np.tanh(delta / (2.0 * tau)))).mean())
