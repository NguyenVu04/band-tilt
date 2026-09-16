"""The KPI vector, its sign convention, and the score that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
them all, the orientation that turns them into "larger is better", and the two
scores that reduce a run to one configuration a deployment can act on.

There are two scores because they answer different questions (ADR 0004):

- :func:`quality_index` is the **soft** score, a weighted geometric mean of
  per-KPI desirabilities (Derringer & Suich, 1980). It is what TuRBO maximises,
  and being a unitless ``[0, 1]`` figure it doubles as the published overall
  network quality.
- :func:`scores` is the **hard** score of ADR 0003, the raw weighted sum. It is
  kept unchanged as a final-evaluation audit.

:func:`selection_scores` picks between them on ``optim.objective``; every call
site goes through it, so a method, a run and a notebook cannot disagree about
which configuration is best.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi import (
    edge_rsrp_dbm,
    hole_desirability,
    hole_rate,
    overlap_desirability,
    overlap_rate,
    served_desirability,
    served_ratio,
    weak_rate,
)

# Everything measured per candidate, in ADR 0001's priority order, highest
# first: the column order of every table and the key order ``kpi.tolerance`` is
# read in.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
    "edge_rsrp_dbm",
    "hole_desirability",
    "overlap_desirability",
    "served_desirability",
)

# What each score reads, which is not everything measured. The hard score keeps
# ADR 0003's four and the key order ``kpi.weights`` is read in; the soft score
# drops the weak rate (ADR 0004). Neither reads ``edge_rsrp_dbm``, which is
# conditional on coverage and so reported only - see :func:`src.kpi.quality`.
WEIGHTED_NAMES = ("hole_rate", "overlap_rate", "served_ratio", "weak_rate")
OBJECTIVE_NAMES = ("hole_rate", "overlap_rate", "served_ratio")

# The KPIs where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset(
    {
        "served_ratio",
        "edge_rsrp_dbm",
        "hole_desirability",
        "overlap_desirability",
        "served_desirability",
    }
)

# The field carrying each objective KPI's desirability. Every one of them is
# softened per tile in :mod:`src.kpi`, at measure time, because a per-tile
# transform cannot be recovered from the share it averages to. So this module
# applies no threshold of its own: by the time a KpiVector exists the soft
# thresholds are already in it, and all that is left to do is weight them.
DESIRABILITY_OF = {
    "hole_rate": "hole_desirability",
    "overlap_rate": "overlap_desirability",
    "served_ratio": "served_desirability",
}


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement on every KPI, in priority order.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``,
            counting locations the ray tracer found no path to at all.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        served_ratio: Share of UEs admitted to a cell-band above the hole threshold.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        edge_rsrp_dbm: Cell-edge serving RSRP over covered locations only, so
            it is read beside ``hole_rate`` and never enters a score.
        hole_desirability: Coverage softened per tile in dB, then averaged.
        overlap_desirability: The neighbour count softened per tile, averaged.
        served_desirability: The served ratio softened per tile, averaged.
            These three are what the soft score reads, in place of the rates
            above; see :data:`DESIRABILITY_OF`.
    """

    hole_rate: float
    overlap_rate: float
    served_ratio: float
    weak_rate: float
    edge_rsrp_dbm: float
    hole_desirability: float
    overlap_desirability: float
    served_desirability: float

    def as_dict(self) -> dict[str, float]:
        """The values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    def as_array(self) -> np.ndarray:
        """The values in :data:`KPI_NAMES` order."""
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
    """Measure one radio map on every KPI.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: The UE reports; ``t_index``, ``tile_row`` and ``tile_col`` place the
            UEs the served ratio counts.
        cfg: Composed config; the KPIs read ``cfg.kpi``.
    """
    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        served_ratio=served_ratio(rsrp, sinr, band_labels, mdt, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        edge_rsrp_dbm=edge_rsrp_dbm(rsrp, cfg),
        hole_desirability=hole_desirability(rsrp, cfg),
        overlap_desirability=overlap_desirability(rsrp, cfg),
        served_desirability=served_desirability(rsrp, sinr, band_labels, mdt, cfg),
    )


def _matrix(kpis: Sequence[KpiVector], names: Sequence[str]) -> np.ndarray:
    """The named KPIs of each vector, signed so larger is better.

    Returns:
        Shape ``[len(kpis), len(names)]``, with the minimised KPIs negated.
        Units are untouched: this is an orientation, not a normalisation.
    """
    signs = np.array([1.0 if name in MAXIMISED else -1.0 for name in names])
    values = np.array([[getattr(kpi, name) for name in names] for kpi in kpis], dtype=float)
    return values * signs


def as_maximised(kpis: Sequence[KpiVector]) -> np.ndarray:
    """Every measured KPI reoriented so larger is better, for reporting.

    Returns:
        Shape ``[len(kpis), len(KPI_NAMES)]`` in :data:`KPI_NAMES` order. This
        is the display orientation and spans more columns than either score
        reads; the scores use :data:`WEIGHTED_NAMES` and
        :data:`OBJECTIVE_NAMES` respectively.
    """
    return _matrix(kpis, KPI_NAMES)


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
    """Score weights from ``kpi.weights``, in :data:`WEIGHTED_NAMES` order.

    Both scores read this block: the hard score uses all four raw, the soft
    score takes the :data:`OBJECTIVE_NAMES` subset and renormalises it
    (:func:`normalised_weights`).

    Raises:
        ValueError: When ``kpi.weights`` is absent or incomplete, a weight is
            negative or non-finite, or every weight is zero.
    """
    block = cfg.kpi.get("weights")
    if block is None:
        raise ValueError(
            "configs/kpi.yaml has no `weights` block. The score needs one weight per KPI "
            f"({', '.join(WEIGHTED_NAMES)})."
        )
    missing = [name for name in WEIGHTED_NAMES if name not in block]
    if missing:
        raise ValueError(f"kpi.weights has no entry for {', '.join(missing)}")
    values = np.array([float(block[name]) for name in WEIGHTED_NAMES], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("kpi.weights must be finite and non-negative")
    if not values.any():
        raise ValueError("kpi.weights are all zero, so every configuration would tie")
    return values


def scores(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """The hard weighted score of each KPI vector; larger is better.

    ``sum_k w_k * s_k * KPI_k`` over :data:`WEIGHTED_NAMES` on raw values, with
    ``s_k`` the sign :func:`as_maximised` applies. Deliberately not normalised
    (ADR 0003): a KPI's influence is its weight times its range.

    ADR 0004 moved the search and the winner onto :func:`quality_index`; this
    is kept unchanged as the final-evaluation audit, so a result can be read
    against the thresholds a deployment would apply.

    Raises:
        ValueError: When ``kpi.weights`` is unusable; see :func:`weights`.
    """
    return _matrix(kpis, WEIGHTED_NAMES) @ weights(cfg)


def desirability(kpis: Sequence[KpiVector]) -> np.ndarray:
    """Each objective KPI's desirability in ``(0, 1)``; larger is better.

    A gather, not a transform. Every objective KPI was softened per tile in
    :mod:`src.kpi` at measure time (:data:`DESIRABILITY_OF`), so there is no
    threshold left for this module to apply.

    That split is deliberate. Softening the aggregate share instead would leave
    the hard comparison inside each KPI untouched: a tile 0.1 dB below the hole
    threshold and one 40 dB below are the same tile to ``hole_rate``, and no
    curve applied afterwards can tell them apart. It also means re-scoring an
    archived run under different weights is honest, while re-scoring it under a
    different temperature is not possible - the temperature is in the
    measurement, not in the score (ADR 0004).

    Returns:
        Shape ``[len(kpis), len(OBJECTIVE_NAMES)]``.
    """
    return _matrix(kpis, [DESIRABILITY_OF[name] for name in OBJECTIVE_NAMES])


def normalised_weights(cfg: DictConfig) -> np.ndarray:
    """``kpi.weights`` restricted to :data:`OBJECTIVE_NAMES` and summing to one.

    Normalising is what makes :func:`quality_index` a weighted geometric *mean*
    rather than a product that shrinks as KPIs are added, so the result stays in
    ``[0, 1]`` and comparable across weightings.

    Raises:
        ValueError: When ``kpi.weights`` is unusable (see :func:`weights`), or
            every objective KPI's weight is zero.
    """
    values = weights(cfg)[[WEIGHTED_NAMES.index(name) for name in OBJECTIVE_NAMES]]
    if not values.any():
        raise ValueError(
            "kpi.weights are all zero across "
            f"{', '.join(OBJECTIVE_NAMES)}, so every configuration would tie"
        )
    return values / values.sum()


def quality_index(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """Overall network quality in ``[0, 1]``; larger is better.

    The weighted geometric mean ``prod_k d_k ** w_k`` of the desirabilities
    (Derringer & Suich, 1980, *Simultaneous Optimization of Several Response
    Variables*), with ``w`` from :func:`normalised_weights`.

    Multiplicative rather than additive on purpose: it is **non-compensatory**,
    so one KPI collapsing drives the whole index to zero and cannot be bought
    back by the others. That is what a threshold means, and it is what a
    weighted sum does not do.

    This is both the objective TuRBO maximises and the published quality figure
    (ADR 0004) - one number, so the thing being optimised and the thing being
    reported cannot drift apart.

    Raises:
        ValueError: When ``kpi.weights`` is unusable; the soft thresholds were
            already applied at measure time and are not read here.
    """
    return np.prod(desirability(kpis) ** normalised_weights(cfg), axis=1)


def selection_scores(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """The configured score of each KPI vector; larger is better.

    ``optim.objective`` picks: ``soft`` is :func:`quality_index`, ``hard`` is
    :func:`scores`. Every search, every winner and every published table goes
    through here rather than naming one directly.

    Raises:
        ValueError: When ``optim.objective`` is neither ``soft`` nor ``hard``,
            or the chosen score's config is unusable.
    """
    objective = str(cfg.optim.objective)
    if objective == "soft":
        return quality_index(kpis, cfg)
    if objective == "hard":
        return scores(kpis, cfg)
    raise ValueError(f"optim.objective must be 'soft' or 'hard', not {objective!r}")


def score_frame(frame: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """:func:`selection_scores` over a table whose columns are :data:`KPI_NAMES`.

    The history CSVs carry the KPIs as columns rather than as objects; this is
    the one place that converts, so no caller rebuilds the score by hand.

    Raises:
        KeyError: When a KPI column is missing.
        ValueError: When the configured score's config is unusable.
    """
    kpis = [KpiVector.from_mapping(row) for row in frame[list(KPI_NAMES)].to_dict("records")]
    return selection_scores(kpis, cfg)


def best_by_score(kpis: Sequence[KpiVector], cfg: DictConfig) -> int:
    """Index of the highest :func:`selection_scores`.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty, or the configured score is
            unusable; see :func:`selection_scores`.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax(selection_scores(kpis, cfg)))
