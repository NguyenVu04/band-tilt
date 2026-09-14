"""The KPI vector, its sign convention, and the score that picks one winner.

The four definitions live in :mod:`src.kpi` and are not restated here. What
this module adds is what an optimizer needs around them: one value object
carrying all four, the orientation that turns them into "larger is better", and
the weighted score (``kpi.weights``, ADR 0003) that TuRBO optimizes and that
reduces every run to the single configuration a deployment can act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi import band_priority_score, hole_rate, overlap_rate, weak_rate

# ADR 0001's priority order, highest first: the column order of every table and
# the key order ``kpi.weights`` and ``kpi.tolerance`` are read in.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "band_priority_score",
    "weak_rate",
)

# The KPIs where larger is better. Named once, so no call site re-decides a
# sign; the other three are minimised.
MAXIMISED = frozenset({"band_priority_score"})


@dataclass(frozen=True)
class KpiVector:
    """One configuration's score on all four KPIs, in priority order.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        band_priority_score: Mean normalised priority weight of each UE's serving band.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
    """

    hole_rate: float
    overlap_rate: float
    band_priority_score: float
    weak_rate: float

    def as_dict(self) -> dict[str, float]:
        """The four values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    def as_array(self) -> np.ndarray:
        """The four values in :data:`KPI_NAMES` order."""
        return np.array([getattr(self, name) for name in KPI_NAMES], dtype=float)

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> KpiVector:
        """Build from a name-keyed mapping.

        Raises:
            KeyError: When a KPI is absent, which means a trial was completed
                with an incomplete measurement.
        """
        missing = [name for name in KPI_NAMES if name not in values]
        if missing:
            raise KeyError(f"no value for {', '.join(missing)}")
        return cls(**{name: float(values[name]) for name in KPI_NAMES})


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    mdt: pd.DataFrame,
    cfg: DictConfig,
) -> KpiVector:
    """Score one radio map on all four KPIs.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: The UE reports; ``tile_row`` and ``tile_col`` weight the band
            priority score.
        cfg: Composed config; the KPIs read ``cfg.kpi``.
    """
    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        band_priority_score=band_priority_score(rsrp, sinr, band_labels, mdt, cfg),
        weak_rate=weak_rate(rsrp, cfg),
    )


def as_maximised(kpis: Sequence[KpiVector]) -> np.ndarray:
    """The KPI matrix reoriented so larger is better in every column.

    Returns:
        Shape ``[len(kpis), 4]`` in :data:`KPI_NAMES` order, with the three
        minimised KPIs negated. Units are untouched: this is an orientation,
        not a normalisation, because nothing downstream compares one column
        against another.
    """
    signs = np.array([1.0 if name in MAXIMISED else -1.0 for name in KPI_NAMES])
    return np.array([kpi.as_array() for kpi in kpis], dtype=float) * signs


def tolerances(cfg: DictConfig) -> np.ndarray:
    """Per-KPI tie thresholds from ``kpi.tolerance``, in :data:`KPI_NAMES` order.

    Used to report a delta as better, worse or a tie; selection does not read it.

    Raises:
        ValueError: When ``kpi.tolerance`` is absent or incomplete. There is no
            safe default: a missing tolerance is an exact comparison, which
            reports solver noise as a real change.
    """
    tolerance = cfg.kpi.get("tolerance")
    if tolerance is None:
        raise ValueError(
            "configs/kpi.yaml has no `tolerance` block. Verdicts need one entry per KPI "
            f"({', '.join(KPI_NAMES)})."
        )
    missing = [name for name in KPI_NAMES if name not in tolerance]
    if missing:
        raise ValueError(f"kpi.tolerance has no entry for {', '.join(missing)}")
    return np.array([float(tolerance[name]) for name in KPI_NAMES], dtype=float)


def weights(cfg: DictConfig) -> np.ndarray:
    """Per-KPI score weights from ``kpi.weights``, in :data:`KPI_NAMES` order.

    Raises:
        ValueError: When ``kpi.weights`` is absent or incomplete, a weight is
            negative or non-finite, or every weight is zero.
    """
    block = cfg.kpi.get("weights")
    if block is None:
        raise ValueError(
            "configs/kpi.yaml has no `weights` block. The score needs one weight per KPI "
            f"({', '.join(KPI_NAMES)})."
        )
    missing = [name for name in KPI_NAMES if name not in block]
    if missing:
        raise ValueError(f"kpi.weights has no entry for {', '.join(missing)}")
    values = np.array([float(block[name]) for name in KPI_NAMES], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("kpi.weights must be finite and non-negative")
    if not values.any():
        raise ValueError("kpi.weights are all zero, so every configuration would tie")
    return values


def scores(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """The weighted score of each KPI vector; larger is better.

    ``sum_k w_k * s_k * KPI_k`` on raw values, with ``s_k`` the sign
    :func:`as_maximised` applies. Deliberately not normalised (ADR 0003): a KPI's
    influence is its weight times its range.

    Raises:
        ValueError: When ``kpi.weights`` is unusable; see :func:`weights`.
    """
    return as_maximised(kpis) @ weights(cfg)


def best_by_score(kpis: Sequence[KpiVector], cfg: DictConfig) -> int:
    """Index of the highest weighted score.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty, or ``kpi.weights`` is unusable.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax(scores(kpis, cfg)))
