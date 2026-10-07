"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective:

    J = mean_g effective_coverage(g)

:func:`src.kpi.overlap.effective_coverage` scores every band and takes the
contraharmonic mean over the tile's layers: a band is worth its strongest sector's
share of the band's received power, scaled by that sector's strength between
``kpi.hole_dbm`` and ``kpi.weak_dbm``. One sector alone at or above
``kpi.weak_dbm`` is worth 1, an equal co-band rival halves it, and a server
barely above ``kpi.hole_dbm`` is worth near nothing. So J is the share of the
grid served cleanly and strongly by one sector, bounded in ``[0, 1]``.

The mean is bounded by the best layer but, unlike a maximum over bands, it is not
monotone in the layers present: a weak extra layer lowers a tile's score. The
objective has no parameters of its own: it reads ``kpi.hole_dbm`` and
``kpi.weak_dbm``, which the reported KPIs already define.

:data:`KPI_NAMES` are reported and no selection reads them; ``objective`` is
stored beside them, so a history is ranked without re-reading a radio map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import CapacitySpec, covered_best, max_rsrp, serve_intervals, serving_sinr
from src.kpi.hole import hole_rate_of
from src.kpi.overlap import (
    effective_coverage,
    overlap_neighbor_mean_of,
    overlap_neighbors,
    overlap_rate_of,
)
from src.kpi.quality import LOW_PERCENTILE, MEDIAN_PERCENTILE, percentiles_over
from src.kpi.served import (
    throughput_mean_mbps,
    throughput_percentile_mbps,
    ue_service_failure_rate,
)
from src.kpi.weak import weak_rate_of

# The reporting order: where coverage fails, how crowded it is, how strong and
# clean the signal is, whether the traffic got served, and at what throughput.
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
    "estimated_throughput_p05_mbps",
    "estimated_throughput_p50_mbps",
    "estimated_throughput_mean_mbps",
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
        "estimated_throughput_p05_mbps",
        "estimated_throughput_p50_mbps",
        "estimated_throughput_mean_mbps",
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
        ue_service_failure_rate: Share of UE reports with no sector-band above
            ``kpi.hole_dbm``.
        estimated_throughput_p05_mbps: Cell-edge (5th percentile) estimated
            throughput of the served UE reports.
        estimated_throughput_p50_mbps: Median estimated throughput of the
            served UE reports.
        estimated_throughput_mean_mbps: Mean estimated throughput of the
            served UE reports.
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
    estimated_throughput_p05_mbps: float
    estimated_throughput_p50_mbps: float
    estimated_throughput_mean_mbps: float
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
    """Share of the grid served cleanly and strongly by exactly one sector.

    The mean of :func:`src.kpi.overlap.effective_coverage` over every tile of the
    grid. A band scores its full 1 only when one sector reaches ``kpi.weak_dbm`` on
    it with no other sector on that band above ``kpi.hole_dbm``. Each such rival
    costs in proportion to its power relative to the strongest sector, so an equal
    one halves the band; a server just above ``kpi.hole_dbm`` keeps almost none
    of it. The tile takes the contraharmonic mean over bands. A hole scores 0, and
    so does a tile the ray tracer found no path to.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; see :func:`src.kpi.overlap.effective_coverage`.

    Returns:
        A value in ``[0, 1]``, rounded to 1e-6, reaching 1 only if every covered
        band on every tile has one server at or above ``kpi.weak_dbm`` and no
        co-band rival above ``kpi.hole_dbm``. Maximised.
    """
    # GPU ray-map accumulation order varies the trailing digits, and TuRBO's GP
    # fit turns any difference in J into a different proposal.
    return round(float(effective_coverage(rsrp, cfg).mean()), 6)


def map_kpis(rsrp: np.ndarray, sinr: np.ndarray, cfg: DictConfig) -> dict[str, float]:
    """The KPIs read off the radio map alone, keyed as :data:`KPI_NAMES` names them.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``; a band slice
            gives that band's reading.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        cfg: Composed config; the measures read ``cfg.kpi``.

    Each reduction of the full map (best server, covered mask, overlap counts,
    serving SINR) is taken once and shared, through the same definitions the
    single-KPI functions of :mod:`src.kpi` wrap.
    """
    r_max = max_rsrp(rsrp)
    is_covered = covered_best(r_max, cfg)
    n_ov = overlap_neighbors(rsrp, cfg)
    points = (MEDIAN_PERCENTILE, LOW_PERCENTILE)
    rsrp_p50, rsrp_p05 = percentiles_over(r_max[is_covered], points)
    sinr_p50, sinr_p05 = percentiles_over(serving_sinr(rsrp, sinr)[is_covered], points)
    return {
        "hole_rate": hole_rate_of(r_max, cfg),
        "weak_rate": weak_rate_of(r_max, cfg),
        "overlap_rate": overlap_rate_of(n_ov),
        "overlap_neighbor_mean": overlap_neighbor_mean_of(n_ov, is_covered),
        "rsrp_p50_dbm": rsrp_p50,
        "rsrp_p05_dbm": rsrp_p05,
        "sinr_p50_db": sinr_p50,
        "sinr_p05_db": sinr_p05,
    }


def ue_kpis(served: pd.DataFrame) -> dict[str, float]:
    """The KPIs counted over UE reports, keyed as :data:`KPI_NAMES` names them.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.

    Raises:
        ValueError: As :func:`src.kpi.served.ue_service_failure_rate`.
    """
    return {
        "ue_service_failure_rate": ue_service_failure_rate(served),
        "estimated_throughput_p05_mbps": throughput_percentile_mbps(served, LOW_PERCENTILE),
        "estimated_throughput_p50_mbps": throughput_percentile_mbps(served, MEDIAN_PERCENTILE),
        "estimated_throughput_mean_mbps": throughput_mean_mbps(served),
    }


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
    spec: CapacitySpec | None = None,
    served: pd.DataFrame | None = None,
) -> KpiVector:
    """Measure one radio map on every KPI and the objective.

    The UE table is served here for the UE KPIs, unless ``served`` already
    holds that assignment for this map.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``t_s``, ``tile_row`` and ``tile_col``
            place the UEs the UE KPIs count.
        cfg: Composed config; the measures read ``cfg.kpi``.
        spec: The capacity model already read from ``cfg``; built here when None.
        served: :func:`src.kpi.capacity.serve_intervals` of this map and ``ue``;
            computed here when None.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, or
            as :func:`src.kpi.capacity.serve_intervals`.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    if served is None:
        if spec is None:
            spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
        served = serve_intervals(rsrp, sinr, band_labels, ue, cfg, spec=spec)
    return KpiVector(**map_kpis(rsrp, sinr, cfg), **ue_kpis(served), objective=objective(rsrp, cfg))


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
