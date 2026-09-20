"""KPIs 10 and 11 - Cell load. How hard the serving rule works each cell-band.

Everything here reads one :func:`src.kpi.capacity.serve_intervals` output, so a
measurement serves the UE table once and reduces it several times. Only admitted
UEs load a cell-band: a blocked UE's PRBs are demand, and
:func:`src.kpi.capacity.demand_prb` is where they are counted.

Load is not in the objective (docs/adr/0009-effective-coverage-objective.md); these
are reported beside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prb_by_cell_interval(
    served: pd.DataFrame, n_band: int, n_tx: int
) -> tuple[np.ndarray, np.ndarray]:
    """PRBs each cell-band carries in each interval.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        n_band: Bands on the radio map, the ``band`` index range.
        n_tx: Transmitters on the radio map, the ``tx`` index range.

    Returns:
        ``(t_values, prb)``: the sorted intervals present in ``served``,
        including ones where nothing was admitted, and ``prb`` shaped
        ``[n_t, n_band, n_tx]`` aligned to them.
    """
    t_values, t_pos = np.unique(served["t_index"].to_numpy(), return_inverse=True)
    band = served["band"].to_numpy()
    admitted = band >= 0
    size = n_band * n_tx
    flat = (
        t_pos[admitted].astype(np.int64) * size
        + band[admitted].astype(np.int64) * n_tx
        + served["tx"].to_numpy()[admitted].astype(np.int64)
    )
    prb = np.bincount(
        flat,
        weights=np.nan_to_num(served["prb_per_ue"].to_numpy()[admitted]),
        minlength=len(t_values) * size,
    )
    return t_values, prb.reshape(len(t_values), n_band, n_tx)


def utilisation(prb: np.ndarray, max_prb: np.ndarray) -> np.ndarray:
    """PRB load as a share of each cell-band's limit.

    Args:
        prb: ``[n_t, n_band, n_tx]`` from :func:`prb_by_cell_interval`.
        max_prb: ``[n_band, n_tx]`` limits, as ``CapacitySpec.max_prb``.

    Returns:
        ``[n_t, n_band, n_tx]``, NaN where a cell-band's limit is zero. Such a
        cell-band carries no traffic, and a ratio over zero would sort to the
        top of every table it appears in.
    """
    limit = np.asarray(max_prb, dtype=float)[None]
    return np.divide(prb, limit, out=np.full(np.shape(prb), np.nan), where=limit > 0)


def prb_utilisation_max(prb: np.ndarray, max_prb: np.ndarray) -> float:
    """The highest load any cell-band reaches in any interval, as a share of its limit.

    The number ``kpi.capacity.max_admission_utilisation`` bounds: the serving
    rule refuses an admission that would carry a cell-band past that share, so
    this measure at or below it is the rule holding, and above it is a fault.

    Returns:
        A value in ``[0, 1]``. Minimised — headroom is what absorbs the traffic
        a tilt change moves. NaN when no cell-band has a positive limit.
    """
    values = utilisation(prb, max_prb)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def load_imbalance(prb: np.ndarray, max_prb: np.ndarray) -> float:
    """Coefficient of variation of utilisation across cell-bands.

    Each cell-band's utilisation is averaged over every interval, idle ones as
    zero, and the spread of those averages is divided by their mean. Scale-free,
    so it does not move when the whole network gets busier — only when the
    traffic sits unevenly, which is what a tilt change can fix.

    The standard deviation is the population one (``ddof=0``): the cell-bands
    are the whole network, not a sample drawn from one.

    Returns:
        ``0`` when every cell-band carries the same share of its limit, rising
        without bound as the load concentrates. Minimised. NaN when no
        cell-band has a positive limit or none carries any traffic, because a
        ratio of spread to nothing says nothing.
    """
    # The mean over intervals first, then the ratio: nanmean over a cell-band
    # that is NaN throughout warns about an empty slice.
    mean_utilisation = utilisation(np.asarray(prb, dtype=float).mean(axis=0)[None], max_prb)[0]
    values = mean_utilisation[np.isfinite(mean_utilisation)]
    if values.size == 0 or values.mean() == 0.0:
        return float("nan")
    return float(values.std(ddof=0) / values.mean())
