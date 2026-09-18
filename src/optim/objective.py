"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective (docs/adr/0007-demand-weighted-objective.md):

    J = sum_g w_g * sigmoid((R_s - T_cov) / tau_R) * exp(-beta * m_g)

with ``sum_g w_g = 1``. The weights are the demand map's
(:mod:`src.data.demand`): a tile carries the share of the measured traffic that
stands on and around it, so the search spends its effort where the UEs are. Set
``data.demand.uniform_share`` to 1 and the weights are equal, which is the
tile-uniform objective of ADR 0006.

:data:`KPI_NAMES` are reported and no selection reads them; ``objective`` is
stored beside them, so a history is ranked without re-reading a radio map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.special import expit

from src.data import demand
from src.kpi.capacity import CapacitySpec, max_rsrp, serve_intervals
from src.kpi.hole import hole_rate
from src.kpi.load import load_imbalance, prb_by_cell_interval, prb_utilisation_max
from src.kpi.overlap import overlap_neighbor_mean, overlap_neighbors, overlap_rate
from src.kpi.quality import (
    LOW_PERCENTILE,
    MEDIAN_PERCENTILE,
    rsrp_percentile_dbm,
    sinr_percentile_db,
)
from src.kpi.served import served_rate
from src.kpi.weak import weak_rate

# What a deployment reads, in ADR 0007's reporting order: where coverage fails,
# how crowded it is, how strong it is, whether the traffic got served, and how
# the load sits.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "overlap_neighbor_mean",
    "weak_rate",
    "rsrp_p05_dbm",
    "rsrp_p50_dbm",
    "sinr_p05_db",
    "sinr_p50_db",
    "served_rate",
    "prb_utilisation_max",
    "load_imbalance",
)

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = (*KPI_NAMES, "objective")

# The measures where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset(
    {
        "served_rate",
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "objective",
    }
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """``kpi.objective``, plus the thresholds it shares with the KPIs.

    Attributes:
        t_cov_dbm: Coverage threshold ``T_cov``, ``kpi.hole_dbm``.
        delta_r_db: Overlap margin ``Delta_R``, ``kpi.overlap_margin_db``.
        tau_r_db: Width ``tau_R`` of the coverage sigmoid.
        beta: Overlap penalty; each neighbour keeps ``q_ov = exp(-beta)``.
    """

    t_cov_dbm: float
    delta_r_db: float
    tau_r_db: float
    beta: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ObjectiveSpec:
        """Read ``kpi.objective``, ``kpi.hole_dbm`` and ``kpi.overlap_margin_db``.

        Raises:
            ValueError: When ``kpi.objective`` is absent, ``tau_r_db`` is not
                positive, or ``beta`` is negative.
        """
        block = cfg.kpi.get("objective")
        if block is None:
            raise ValueError("configs/kpi.yaml has no `objective` block.")
        spec = cls(
            t_cov_dbm=float(cfg.kpi.hole_dbm),
            delta_r_db=float(cfg.kpi.overlap_margin_db),
            tau_r_db=float(block.tau_r_db),
            beta=float(block.beta),
        )
        if not spec.tau_r_db > 0:
            raise ValueError(f"kpi.objective.tau_r_db must be positive, got {spec.tau_r_db}")
        if not spec.beta >= 0:
            raise ValueError(f"kpi.objective.beta must be non-negative, got {spec.beta}")
        return spec


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement: the KPIs, then the objective.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``,
            counting locations the ray tracer found no path to at all.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        overlap_neighbor_mean: Overlapping co-band neighbours per covered tile,
            the severity behind that rate's incidence.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        rsrp_p05_dbm: Cell-edge serving RSRP over covered locations only, so it
            is read beside ``hole_rate``.
        rsrp_p50_dbm: Median serving RSRP over the same locations.
        sinr_p05_db: Cell-edge best-server SINR over the same locations.
        sinr_p50_db: Median best-server SINR over the same locations.
        served_rate: Share of UE reports admitted to a cell-band.
        prb_utilisation_max: Highest load any cell-band reaches in any interval,
            as a share of its limit.
        load_imbalance: Coefficient of variation of cell-band utilisation.
        objective: See :func:`objective`.
    """

    hole_rate: float
    overlap_rate: float
    overlap_neighbor_mean: float
    weak_rate: float
    rsrp_p05_dbm: float
    rsrp_p50_dbm: float
    sinr_p05_db: float
    sinr_p50_db: float
    served_rate: float
    prb_utilisation_max: float
    load_imbalance: float
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


def tile_weights(cfg: DictConfig, shape: tuple[int, int]) -> np.ndarray:
    """The demand map's weights, checked against the radio map's grid.

    Side effect: reads ``data.output.demand_file``; see
    :func:`src.data.demand.load_weights` for the caching.

    Raises:
        FileNotFoundError: When the demand map has not been built.
        ValueError: When it was built on another grid, which would silently
            weight the wrong tiles.
    """
    weights = demand.load_weights(cfg)
    if weights.shape != tuple(shape):
        raise ValueError(
            f"the demand map is {weights.shape[0]} x {weights.shape[1]} but the radio map is "
            f"{shape[0]} x {shape[1]}. The two were built on different grids; re-run "
            "`task preprocess`."
        )
    return weights


def objective(rsrp: np.ndarray, cfg: DictConfig, weights: np.ndarray | None = None) -> float:
    """Coverage utility discounted by co-band overlap, averaged over the demand map.

    ``sum_g w_g * sigmoid((R_s - T_cov) / tau_R) * q_ov ** m_g``. ``R_s`` is the
    strongest cell-band at the tile. ``m_g`` is
    :func:`src.kpi.overlap.overlap_neighbors`, the count ``overlap_rate`` reads:
    per band, the transmitters on that band above ``T_cov`` and within
    ``Delta_R`` of that band's strongest, summed over bands. A no-path tile has
    ``R_s = -inf`` and scores zero. ``w_g`` is the demand map's weight for the
    tile, so a hole where nobody stands costs less than one in a hotspot.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; see :meth:`ObjectiveSpec.from_config`.
        weights: Tile weights, ``[n_rows, n_cols]``. Read from
            ``data.output.demand_file`` when None, which is what every caller
            but a test does.

    Returns:
        A value in ``[0, 1]``. Maximised.

    Raises:
        FileNotFoundError: When ``weights`` is None and the demand map has not
            been built.
        ValueError: When the weights do not cover the radio map's grid, or sum
            to nothing.
    """
    spec = ObjectiveSpec.from_config(cfg)
    serving = max_rsrp(rsrp)
    overlaps = overlap_neighbors(rsrp, cfg)
    utility = expit((serving - spec.t_cov_dbm) / spec.tau_r_db) * np.exp(-spec.beta * overlaps)

    weight = tile_weights(cfg, serving.shape) if weights is None else np.asarray(weights, float)
    if weight.shape != serving.shape:
        raise ValueError(
            f"weights are {weight.shape} for a {serving.shape} grid; they must cover it exactly."
        )
    total = float(weight.sum())
    if not total > 0:
        raise ValueError("the tile weights sum to zero, so there is nothing to average over.")
    return float(np.dot(utility.ravel(), weight.ravel()) / total)


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
    weights: np.ndarray | None = None,
) -> KpiVector:
    """Measure one radio map on every KPI and the objective.

    The UE table is served once, here, and the served rate and both load
    measures read that one assignment: serving is the expensive half of a
    measurement, and running it three times would treble what a search pays per
    candidate on top of the ray tracing.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``t_s``, ``tile_row`` and ``tile_col``
            place the UEs the served rate counts.
        cfg: Composed config; the measures read ``cfg.kpi``.
        weights: Passed to :func:`objective`.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, or
            as :func:`src.kpi.capacity.serve_intervals` and :func:`objective`.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    _t_values, prb = prb_by_cell_interval(served, rsrp.shape[0], rsrp.shape[1])

    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        overlap_neighbor_mean=overlap_neighbor_mean(rsrp, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        rsrp_p05_dbm=rsrp_percentile_dbm(rsrp, cfg, LOW_PERCENTILE),
        rsrp_p50_dbm=rsrp_percentile_dbm(rsrp, cfg, MEDIAN_PERCENTILE),
        sinr_p05_db=sinr_percentile_db(rsrp, sinr, cfg, LOW_PERCENTILE),
        sinr_p50_db=sinr_percentile_db(rsrp, sinr, cfg, MEDIAN_PERCENTILE),
        served_rate=served_rate(served),
        prb_utilisation_max=prb_utilisation_max(prb, spec.max_prb),
        load_imbalance=load_imbalance(prb, spec.max_prb),
        objective=objective(rsrp, cfg, weights),
    )


def best_by_objective(kpis: Sequence[KpiVector]) -> int:
    """Index of the highest ``objective``.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax([kpi.objective for kpi in kpis]))
