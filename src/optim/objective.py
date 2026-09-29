"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective:

    J = mean_g effective_coverage(g)

:func:`src.kpi.overlap.effective_coverage` scores every band and takes the
contraharmonic mean over the tile's layers: one dominant cell at or above
``kpi.weak_dbm`` is worth 1, a second cell inside the overlap margin costs a
quarter, and a server barely above ``kpi.hole_dbm`` is worth near nothing. So J
is the share of the grid served cleanly and strongly by one cell, bounded in
``[0, 1]``.

The mean is bounded by the best layer but, unlike a maximum over bands, it is not
monotone in the layers present: a weak extra layer lowers a tile's score. The
objective has no parameters of its own: it reads ``kpi.hole_dbm``,
``kpi.weak_dbm`` and ``kpi.overlap_margin_db``, each of which the reported KPIs
already define.

:data:`KPI_NAMES` are reported and no selection reads them; ``objective`` is
stored beside them, so a history is ranked without re-reading a radio map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import CapacitySpec, serve_intervals
from src.kpi.hole import hole_rate
from src.kpi.overlap import effective_coverage, overlap_neighbor_mean, overlap_rate
from src.kpi.quality import (
    LOW_PERCENTILE,
    MEDIAN_PERCENTILE,
    rsrp_percentile_dbm,
    sinr_percentile_db,
)
from src.kpi.served import ue_service_failure_rate
from src.kpi.weak import weak_rate

# The reporting order: where coverage fails, how crowded it is, how strong and
# clean the signal is, and whether the traffic got served.
KPI_NAMES = (
    "hole_rate",
    "weak_rate",
    "overlap_rate",
    "overlap_neighbor_mean",
    "rsrp_p50_dbm",
    "rsrp_p05_dbm",
    "sinr_p50_db",
    "sinr_p05_db",
    "ue_service_failure_rate",
)

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = (*KPI_NAMES, "objective")

# The measures where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset(
    {
        "rsrp_p50_dbm",
        "rsrp_p05_dbm",
        "sinr_p50_db",
        "sinr_p05_db",
        "objective",
    }
)


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement: the KPIs, then the objective.

    The signal-quality measures are over covered tiles at the best server, so
    they are read beside ``hole_rate``.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``,
            counting locations the ray tracer found no path to at all.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        overlap_neighbor_mean: Overlapping co-band neighbours per covered tile.
        rsrp_p50_dbm: Median best-server RSRP.
        rsrp_p05_dbm: Cell-edge (5th percentile) best-server RSRP.
        sinr_p50_db: Median best-server SINR.
        sinr_p05_db: Cell-edge best-server SINR.
        ue_service_failure_rate: Share of UE reports not admitted to any cell-band.
        objective: See :func:`objective`. In ``[0, 1]``.
    """

    hole_rate: float
    weak_rate: float
    overlap_rate: float
    overlap_neighbor_mean: float
    rsrp_p50_dbm: float
    rsrp_p05_dbm: float
    sinr_p50_db: float
    sinr_p05_db: float
    ue_service_failure_rate: float
    objective: float

    def as_dict(self) -> dict[str, float]:
        """The values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> KpiVector:
        """Build from a name-keyed mapping.

        Raises:
            KeyError: When a measure is absent, which means a trial was
                completed with an incomplete measurement, or the record predates
                the current measure set.
        """
        missing = [name for name in MEASURE_NAMES if name not in values]
        if missing:
            raise KeyError(f"no value for {', '.join(missing)}")
        return cls(**{name: float(values[name]) for name in MEASURE_NAMES})


def objective(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Share of the grid served cleanly and strongly by exactly one cell.

    The mean of :func:`src.kpi.overlap.effective_coverage` over every tile of the
    grid. A band scores its full 1 only when one cell reaches ``kpi.weak_dbm`` on
    it with nothing else on that band within ``kpi.overlap_margin_db``. A second
    cell inside the margin costs it a quarter and a third nearly two thirds; a
    server just above ``kpi.hole_dbm`` keeps almost none of it. The tile takes the
    contraharmonic mean over bands. A hole scores 0, and so does a tile the ray
    tracer found no path to.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; see :func:`src.kpi.overlap.effective_coverage`.

    Returns:
        A value in ``[0, 1]``, rounded to 1e-6, reaching 1 only if every covered
        band on every tile has one server at or above ``kpi.weak_dbm``. Maximised.
    """
    # GPU ray-map accumulation order varies the trailing digits, and TuRBO's GP
    # fit turns any difference in J into a different proposal.
    return round(float(effective_coverage(rsrp, cfg).mean()), 6)


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
    spec: CapacitySpec | None = None,
) -> KpiVector:
    """Measure one radio map on every KPI and the objective.

    The UE table is served once, here, for the service failure rate.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``t_s``, ``tile_row`` and ``tile_col``
            place the UEs the failure rate counts.
        cfg: Composed config; the measures read ``cfg.kpi``.
        spec: The capacity model already read from ``cfg``; built here when None.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, or
            as :func:`src.kpi.capacity.serve_intervals`.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    if spec is None:
        spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg, spec=spec)

    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        overlap_neighbor_mean=overlap_neighbor_mean(rsrp, cfg),
        rsrp_p50_dbm=rsrp_percentile_dbm(rsrp, cfg, MEDIAN_PERCENTILE),
        rsrp_p05_dbm=rsrp_percentile_dbm(rsrp, cfg, LOW_PERCENTILE),
        sinr_p50_db=sinr_percentile_db(rsrp, sinr, cfg, MEDIAN_PERCENTILE),
        sinr_p05_db=sinr_percentile_db(rsrp, sinr, cfg, LOW_PERCENTILE),
        ue_service_failure_rate=ue_service_failure_rate(served),
        objective=objective(rsrp, cfg),
    )


def best_by_objective(kpis: Sequence[KpiVector]) -> int:
    """Index of the highest ``objective``.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty, or an objective is not finite —
            ``np.argmax`` would otherwise pick a NaN as the best.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    scores = np.array([kpi.objective for kpi in kpis], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite objective at index {np.flatnonzero(~np.isfinite(scores))}")
    return int(np.argmax(scores))
