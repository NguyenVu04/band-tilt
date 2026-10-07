"""The KPI vector, its sign convention, and the objectives a search maximises.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", the
objectives, and the hypervolume that ranks objective vectors.

Notation: ``R_{i,b}(g)`` is sector ``i``'s RSRP on band ``b`` at tile ``g``,
``s`` the strongest sector on that band, and each map objective is the mean over
every tile of the grid.

- Coverage: ``1 - prod_b (1 - sigma(R_{s,b}(g) - kpi.weak_dbm))``, with
  ``sigma(x) = 1 / (1 + 10^(-x / 10))``: the soft chance that at least one band
  is above the weak threshold. A band at ``kpi.weak_dbm`` scores 1/2 alone.
- Separation: ``prod_b 1 / (1 + sum_i 10^((m - (R_{s,b} - R_{i,b})) / 10))``,
  ``m = kpi.overlap_margin_db``, over the other sectors on band ``b`` above
  ``kpi.hole_dbm``. A rival ``m`` dB down halves the band; a band with no server
  above ``kpi.hole_dbm`` has no rival to price and contributes 1.
- Throughput: ``mean_u ln(1 + R_u)`` over every UE report, ``R_u`` its estimated
  throughput in Mbit/s, 0 when no layer reaches it. Measured and recorded, but
  not in :data:`OBJECTIVE_NAMES`, so no search reads it.

Each objective is maximised. A history is ranked by hypervolume against the
origin, every objective's natural floor: :func:`best_by_hvc` picks the point with
the largest hypervolume contribution. :data:`KPI_NAMES` are reported and no
selection reads them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.special import expit

from src.kpi.capacity import (
    CapacitySpec,
    covered_best,
    finite,
    max_rsrp,
    serve_intervals,
    serving_sinr,
)
from src.kpi.hole import hole_rate_of
from src.kpi.overlap import overlap_neighbor_mean_of, overlap_neighbors, overlap_rate_of
from src.kpi.quality import LOW_PERCENTILE, MEDIAN_PERCENTILE, percentiles_over
from src.kpi.served import throughput_mean_mbps, throughput_percentile_mbps, ue_throughput_mbps
from src.kpi.weak import weak_rate_of

# The reporting order: where coverage fails, how crowded it is, how strong and
# clean the signal is, and at what throughput the UEs are served.
KPI_NAMES = (
    "hole_rate",
    "weak_rate",
    "overlap_rate",
    "overlap_neighbor_mean",
    "rsrp_p50_dbm",
    "rsrp_p05_dbm",
    "sinr_p50_db",
    "sinr_p05_db",
    "estimated_throughput_p05_mbps",
    "estimated_throughput_p50_mbps",
    "estimated_throughput_mean_mbps",
)

# What a search maximises, in the column order of every objective matrix.
OBJECTIVE_NAMES = ("coverage_objective", "separation_objective")

# Measured and recorded but not searched: out of the search pending review.
UNSEARCHED_OBJECTIVES = ("throughput_objective",)

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = (*KPI_NAMES, *OBJECTIVE_NAMES, *UNSEARCHED_OBJECTIVES)

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
        *OBJECTIVE_NAMES,
        *UNSEARCHED_OBJECTIVES,
    }
)


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement: the KPIs, then the objectives.

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
        estimated_throughput_p05_mbps: 5th percentile of the estimated
            throughput over every UE report, an unserved one counting 0.
        estimated_throughput_p50_mbps: Median of the same.
        estimated_throughput_mean_mbps: Mean of the same.
        coverage_objective: See :func:`coverage_objective`. In ``[0, 1]``.
        separation_objective: See :func:`separation_objective`. In ``(0, 1]``.
        throughput_objective: See :func:`throughput_objective`. Non-negative.
            Recorded, not searched.
    """

    hole_rate: float
    weak_rate: float
    overlap_rate: float
    overlap_neighbor_mean: float
    rsrp_p50_dbm: float
    rsrp_p05_dbm: float
    sinr_p50_db: float
    sinr_p05_db: float
    estimated_throughput_p05_mbps: float
    estimated_throughput_p50_mbps: float
    estimated_throughput_mean_mbps: float
    coverage_objective: float
    separation_objective: float
    throughput_objective: float

    def as_dict(self) -> dict[str, float]:
        """The values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    @property
    def objectives(self) -> np.ndarray:
        """The objectives in :data:`OBJECTIVE_NAMES` order."""
        return np.array([getattr(self, name) for name in OBJECTIVE_NAMES], dtype=float)

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


def _rounded(value: float) -> float:
    """``value`` to 6 significant digits.

    GPU ray-map accumulation order varies the trailing digits, and the GP fit
    turns any difference in an objective into a different proposal. Significant
    digits rather than decimals, so the guard holds whatever an objective's scale.
    """
    return float(f"{float(value):.6g}")


def coverage_objective(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Mean over tiles of the soft chance that at least one band is above the weak threshold.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.weak_dbm``.

    Returns:
        A value in ``[0, 1]``; a tile no band reaches, and every no-path tile,
        scores 0.
    """
    strongest = finite(rsrp).max(axis=1)
    # sigma(x) = 1 / (1 + 10^(-x / 10)) is the logistic of x ln(10) / 10.
    band = expit((strongest - float(cfg.kpi.weak_dbm)) * np.log(10.0) / 10.0)
    return _rounded((1.0 - np.prod(1.0 - band, axis=0)).mean())


def separation_objective(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Mean over tiles of how cleanly the strongest sector stands out on every band.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        A value in ``(0, 1]``, 1 where every covered band has one sector with no
        co-band rival above ``kpi.hole_dbm``.
    """
    hole_dbm = float(cfg.kpi.hole_dbm)
    margin_db = float(cfg.kpi.overlap_margin_db)
    layers = finite(rsrp)
    strongest = layers.max(axis=1, keepdims=True)
    counted = (layers > hole_dbm) & (strongest > hole_dbm)
    # Only counted layers are differenced: `-inf - -inf` is NaN, and warns.
    relative_db = np.subtract(layers, strongest, out=np.full_like(layers, -np.inf), where=counted)
    terms = (10.0 ** ((relative_db + margin_db) / 10.0)).sum(axis=1)
    # `s` is counted against itself at 10^(m / 10) wherever its band is covered.
    itself = np.where(counted.any(axis=1), 10.0 ** (margin_db / 10.0), 0.0)
    rivals = np.clip(terms - itself, 0.0, None)
    return _rounded(np.prod(1.0 / (1.0 + rivals), axis=0).mean())


def throughput_objective(served: pd.DataFrame) -> float:
    """``mean_u ln(1 + R_u)`` over every UE report, ``R_u`` in Mbit/s and 0 when unserved.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.

    Returns:
        A non-negative value; 0 when there is no report.
    """
    values = ue_throughput_mbps(served)
    return _rounded(np.log1p(values).mean()) if values.size else 0.0


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
    """
    return {
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
    """Measure one radio map on every KPI and every objective.

    The UE table is served here for the UE measures, unless ``served`` already
    holds that assignment for this map.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``t_s``, ``tile_row`` and ``tile_col``
            place the UEs the UE measures count.
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
    return KpiVector(
        **map_kpis(rsrp, sinr, cfg),
        **ue_kpis(served),
        coverage_objective=coverage_objective(rsrp, cfg),
        separation_objective=separation_objective(rsrp, cfg),
        throughput_objective=throughput_objective(served),
    )


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Which rows no other row dominates, every column maximised.

    A row dominates another when it is no worse on every column and strictly
    better on at least one, so identical rows do not dominate each other.
    """
    # ponytail: O(n^2) pairwise comparison; fine for a few thousand rows, sort-based if more.
    points = np.asarray(points, dtype=float)
    no_worse = (points[:, None, :] >= points[None, :, :]).all(axis=2)
    better = (points[:, None, :] > points[None, :, :]).any(axis=2)
    return ~(no_worse & better).any(axis=0)


def _dominated_volume(points: np.ndarray) -> float:
    """Volume ``points`` dominate above the origin; every coordinate positive, two or more columns.

    Slices along the last column: each slab between consecutive heights is the
    lower-dimensional volume of the points at least that high.
    """
    points = points[np.argsort(-points[:, -1], kind="stable")]
    slabs = points[:, -1] - np.append(points[1:, -1], 0.0)
    if points.shape[1] == 2:
        return float(slabs @ np.maximum.accumulate(points[:, 0]))
    return float(
        sum(slab * _dominated_volume(points[: i + 1, :-1]) for i, slab in enumerate(slabs) if slab)
    )


def hypervolume(points: np.ndarray) -> float:
    """Hypervolume of ``points`` against the origin, every column maximised.

    Args:
        points: ``[n, m]`` objective vectors, ``m >= 2``. A row not strictly
            above the origin on every column dominates nothing and adds nothing.
    """
    points = np.asarray(points, dtype=float)
    points = points[(points > 0.0).all(axis=1)]
    return _dominated_volume(points) if len(points) else 0.0


def hypervolume_contributions(points: np.ndarray) -> np.ndarray:
    """Each row's exclusive share of :func:`hypervolume`: what removing it would lose.

    A dominated row contributes 0. Of identical rows the first carries the
    contribution and the rest 0, so a repeated evaluation neither doubles a
    point's weight nor cancels it.
    """
    points = np.asarray(points, dtype=float)
    contributions = np.zeros(len(points))
    front = np.flatnonzero(pareto_mask(points))
    _, first = np.unique(points[front], axis=0, return_index=True)
    keep = front[np.sort(first)]
    total = hypervolume(points[keep])
    for k, row in enumerate(keep):
        contributions[row] = total - hypervolume(np.delete(points[keep], k, axis=0))
    return contributions


def objective_matrix(kpis: Sequence[KpiVector]) -> np.ndarray:
    """``[n, len(OBJECTIVE_NAMES)]`` objectives of each KPI vector, in that order."""
    return np.array([kpi.objectives for kpi in kpis], dtype=float).reshape(-1, len(OBJECTIVE_NAMES))


def best_by_hvc(kpis: Sequence[KpiVector]) -> int:
    """Index of the largest hypervolume contribution: the recommended configuration.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually contributes more.

    Raises:
        ValueError: When ``kpis`` is empty, or an objective is not finite.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    points = objective_matrix(kpis)
    if not np.isfinite(points).all():
        bad = np.flatnonzero(~np.isfinite(points).all(axis=1))
        raise ValueError(f"non-finite objective at index {bad}")
    return int(np.argmax(hypervolume_contributions(points)))
