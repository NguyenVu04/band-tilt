"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective (docs/adr/0006-radio-coverage-objective.md):

    J = mean_g sigmoid((R_s - T_cov) / tau_R) * exp(-beta * m_g)

that reduces a run to one configuration a deployment can act on.

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

from src.kpi import edge_rsrp_dbm, hole_rate, overlap_rate, served_ratio, weak_rate
from src.kpi.capacity import max_rsrp
from src.kpi.overlap import overlap_neighbors

# What a deployment reads, in ADR 0001's reporting order.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
    "edge_rsrp_dbm",
)

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = (*KPI_NAMES, "objective")

# The measures where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset({"served_ratio", "edge_rsrp_dbm", "objective"})


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
        served_ratio: Share of UEs admitted to a cell-band above the hole threshold.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        edge_rsrp_dbm: Cell-edge serving RSRP over covered locations only, so
            it is read beside ``hole_rate``.
        objective: See :func:`objective`.
    """

    hole_rate: float
    overlap_rate: float
    served_ratio: float
    weak_rate: float
    edge_rsrp_dbm: float
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
                the current objective.
        """
        missing = [name for name in MEASURE_NAMES if name not in values]
        if missing:
            raise KeyError(f"no value for {', '.join(missing)}")
        return cls(**{name: float(values[name]) for name in MEASURE_NAMES})


def objective(rsrp: np.ndarray, cfg: DictConfig) -> float:
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


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> KpiVector:
    """Measure one radio map on every KPI and the objective.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``tile_row`` and ``tile_col`` place the
            UEs the served ratio counts.
        cfg: Composed config; the measures read ``cfg.kpi``.
    """
    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        served_ratio=served_ratio(rsrp, sinr, band_labels, ue, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        edge_rsrp_dbm=edge_rsrp_dbm(rsrp, cfg),
        objective=objective(rsrp, cfg),
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
