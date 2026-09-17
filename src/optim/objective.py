"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective (docs/adr/0006-radio-load-cvar-objective.md):

    J = J_radio ** gamma * J_load ** (1 - gamma)

that reduces a run to one configuration a deployment can act on.

:data:`KPI_NAMES` are reported and no score reads them. :data:`OBJECTIVE_NAMES`
are the two terms of ``J``, measured from the radio map and the serving rule
alongside the KPIs. Storing the terms rather than ``J`` is what lets a stored
history be re-scored under another ``gamma``; every other objective parameter
is inside the measurement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.special import expit

from src.kpi import edge_rsrp_dbm, hole_rate, overlap_rate, served_ratio, weak_rate
from src.kpi.capacity import CapacitySpec, max_rsrp, serve_intervals
from src.kpi.overlap import overlap_neighbors

# What a deployment reads, in ADR 0001's reporting order.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
    "edge_rsrp_dbm",
)

# The two terms of the objective, each in [0, 1].
OBJECTIVE_NAMES = ("j_radio", "j_load")

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = KPI_NAMES + OBJECTIVE_NAMES

# The measures where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset({"served_ratio", "edge_rsrp_dbm", *OBJECTIVE_NAMES})


@dataclass(frozen=True)
class ObjectiveSpec:
    """``kpi.objective``, plus the thresholds it shares with the KPIs.

    Attributes:
        t_cov_dbm: Coverage threshold ``T_cov``, ``kpi.hole_dbm``.
        delta_r_db: Overlap margin ``Delta_R``, ``kpi.overlap_margin_db``.
        tau_r_db: Width ``tau_R`` of the coverage sigmoid.
        beta: Overlap penalty; each neighbour keeps ``q_ov = exp(-beta)``.
        rho_0: PRB utilisation above which load counts against ``J_load``.
        alpha: CVaR level; the tail is the worst ``1 - alpha`` share.
        gamma: Weight of ``J_radio`` against ``J_load``.
    """

    t_cov_dbm: float
    delta_r_db: float
    tau_r_db: float
    beta: float
    rho_0: float
    alpha: float
    gamma: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ObjectiveSpec:
        """Read ``kpi.objective``, ``kpi.hole_dbm`` and ``kpi.overlap_margin_db``.

        Raises:
            ValueError: When ``kpi.objective`` is absent, ``tau_r_db`` is not
                positive, ``beta`` is negative, ``rho_0`` or ``alpha`` is
                outside ``[0, 1)``, or ``gamma`` is outside ``[0, 1]``.
        """
        block = cfg.kpi.get("objective")
        if block is None:
            raise ValueError("configs/kpi.yaml has no `objective` block.")
        spec = cls(
            t_cov_dbm=float(cfg.kpi.hole_dbm),
            delta_r_db=float(cfg.kpi.overlap_margin_db),
            tau_r_db=float(block.tau_r_db),
            beta=float(block.beta),
            rho_0=float(block.rho_0),
            alpha=float(block.alpha),
            gamma=float(block.gamma),
        )
        if not spec.tau_r_db > 0:
            raise ValueError(f"kpi.objective.tau_r_db must be positive, got {spec.tau_r_db}")
        if not spec.beta >= 0:
            raise ValueError(f"kpi.objective.beta must be non-negative, got {spec.beta}")
        for name in ("rho_0", "alpha"):
            if not 0.0 <= getattr(spec, name) < 1.0:
                raise ValueError(
                    f"kpi.objective.{name} must be in [0, 1), got {getattr(spec, name)}"
                )
        if not 0.0 <= spec.gamma <= 1.0:
            raise ValueError(f"kpi.objective.gamma must be in [0, 1], got {spec.gamma}")
        return spec


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement: the KPIs, then the objective terms.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``,
            counting locations the ray tracer found no path to at all.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        served_ratio: Share of UEs admitted to a cell-band above the hole threshold.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        edge_rsrp_dbm: Cell-edge serving RSRP over covered locations only, so
            it is read beside ``hole_rate``.
        j_radio: See :func:`j_radio`.
        j_load: See :func:`j_load`.
    """

    hole_rate: float
    overlap_rate: float
    served_ratio: float
    weak_rate: float
    edge_rsrp_dbm: float
    j_radio: float
    j_load: float

    def as_dict(self) -> dict[str, float]:
        """The values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> KpiVector:
        """Build from a name-keyed mapping.

        Raises:
            KeyError: When a measure is absent, which means a trial was
                completed with an incomplete measurement, or the record predates
                the current objective.
        """
        missing = [name for name in MEASURE_NAMES if name not in values]
        if missing:
            raise KeyError(f"no value for {', '.join(missing)}")
        return cls(**{name: float(values[name]) for name in MEASURE_NAMES})


def j_radio(rsrp: np.ndarray, cfg: DictConfig) -> float:
    """Coverage utility discounted by co-band overlap, averaged over every tile.

    ``mean_g sigmoid((R_s - T_cov) / tau_R) * q_ov ** m_g``, with equal tile
    weights. ``R_s`` is the strongest cell-band at the tile. ``m_g`` is
    :func:`src.kpi.overlap.overlap_neighbors`, the count ``overlap_rate`` reads:
    per band, the transmitters on that band above ``T_cov`` and within
    ``Delta_R`` of that band's strongest, summed over bands. A no-path tile has
    ``R_s = -inf`` and scores zero.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        cfg: Composed config; see :meth:`ObjectiveSpec.from_config`.

    Returns:
        A value in ``[0, 1]``. Maximised.
    """
    spec = ObjectiveSpec.from_config(cfg)
    serving = max_rsrp(rsrp)
    overlaps = overlap_neighbors(rsrp, cfg)
    utility = expit((serving - spec.t_cov_dbm) / spec.tau_r_db) * np.exp(-spec.beta * overlaps)
    return float(utility.mean())


def cvar(values: np.ndarray, alpha: float) -> float:
    """Mean of the largest ``ceil((1 - alpha) * n)`` values: the upper-tail CVaR.

    Raises:
        ValueError: When ``values`` is empty.
    """
    flat = np.sort(np.asarray(values, dtype=float).ravel())
    if flat.size == 0:
        raise ValueError("CVaR of an empty sample is undefined.")
    tail = max(1, math.ceil((1.0 - alpha) * flat.size))
    return float(flat[-tail:].mean())


def j_load(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> float:
    """``1 - CVaR_alpha([rho - rho_0]_+) / (1 - rho_0)`` over cell-band intervals.

    ``rho`` is the PRBs the serving rule admitted to one cell-band in one
    interval over that cell-band's ``max_prb``. Every cell-band counts in every
    interval the UE table holds, idle ones at zero, so the tail share is of the whole
    network rather than of the busy cells.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape; sets PRBs per UE.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: ``t_index``, ``tile_row`` and ``tile_col`` place each UE.
        cfg: Composed config; see :meth:`ObjectiveSpec.from_config` and
            :meth:`src.kpi.capacity.CapacitySpec.from_config`.

    Returns:
        A value in ``[0, 1]``; one when no cell-band exceeds ``rho_0``. Maximised.

    Raises:
        ValueError: When the UE table is empty, or as
            :func:`src.kpi.capacity.serve_intervals`.
    """
    if ue.empty:
        raise ValueError("The UE table holds no UE, so there is no load to measure.")
    spec = ObjectiveSpec.from_config(cfg)
    capacity = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    t_values, t_pos = np.unique(served["t_index"].to_numpy(), return_inverse=True)
    band = served["band"].to_numpy()
    tx = served["tx"].to_numpy()
    on = band >= 0
    load = np.zeros((len(t_values), *capacity.max_prb.shape))
    np.add.at(load, (t_pos[on], band[on], tx[on]), served["prb_per_ue"].to_numpy()[on])
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(capacity.max_prb > 0, load / capacity.max_prb, 0.0)
    excess = np.maximum(rho - spec.rho_0, 0.0)
    return float(1.0 - cvar(excess, spec.alpha) / (1.0 - spec.rho_0))


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> KpiVector:
    """Measure one radio map on every KPI and objective term.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``tile_row`` and ``tile_col`` place the
            UEs the served ratio and the load term count. The search passes the
            MDT, evaluation every UE.
        cfg: Composed config; the measures read ``cfg.kpi``.
    """
    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        served_ratio=served_ratio(rsrp, sinr, band_labels, ue, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        edge_rsrp_dbm=edge_rsrp_dbm(rsrp, cfg),
        j_radio=j_radio(rsrp, cfg),
        j_load=j_load(rsrp, sinr, band_labels, ue, cfg),
    )


def score(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """The objective ``J = j_radio ** gamma * j_load ** (1 - gamma)``, in ``[0, 1]``.

    This is both what the search maximises and the published score, so the
    thing optimised and the thing reported cannot drift apart.

    Raises:
        ValueError: When ``kpi.objective`` is unusable; see
            :meth:`ObjectiveSpec.from_config`.
    """
    gamma = ObjectiveSpec.from_config(cfg).gamma
    radio = np.array([kpi.j_radio for kpi in kpis], dtype=float)
    load = np.array([kpi.j_load for kpi in kpis], dtype=float)
    return radio**gamma * load ** (1.0 - gamma)


def score_frame(frame: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """:func:`score` over a table whose columns are :data:`MEASURE_NAMES`.

    Raises:
        KeyError: When a measure column is missing.
        ValueError: When ``kpi.objective`` is unusable.
    """
    kpis = [KpiVector.from_mapping(row) for row in frame[list(MEASURE_NAMES)].to_dict("records")]
    return score(kpis, cfg)


def best_by_score(kpis: Sequence[KpiVector], cfg: DictConfig) -> int:
    """Index of the highest :func:`score`.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty, or ``kpi.objective`` is unusable.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax(score(kpis, cfg)))
