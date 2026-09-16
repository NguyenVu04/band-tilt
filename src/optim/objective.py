"""The KPI vector, its sign convention, and the score that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
them all, the orientation that turns them into "larger is better", and the one
score that reduces a run to one configuration a deployment can act on.

The two families are kept apart (ADR 0005). :data:`TARGET_NAMES` are the
desirabilities the search maximises through :func:`quality_index`;
:data:`REPORT_NAMES` are the rates a deployment reads, and no score reads them.
Neither family is derived from the other and neither reads the other's config,
so re-weighting the objective cannot move a reported rate and re-reading a rate
cannot change what was searched for.
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

# What a deployment reads, in ADR 0001's priority order, highest first. No score
# reads these: the rates are steps a search cannot descend, and `edge_rsrp_dbm`
# is conditional on coverage - see :func:`src.kpi.quality.edge_rsrp_dbm`.
REPORT_NAMES = (
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
    "edge_rsrp_dbm",
)

# What the search maximises, and the key order `kpi.weights` is read in.
TARGET_NAMES = (
    "hole_desirability",
    "overlap_desirability",
    "served_desirability",
)

# Everything measured per candidate: the column order of every table.
KPI_NAMES = REPORT_NAMES + TARGET_NAMES

# The KPIs where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset({"served_ratio", "edge_rsrp_dbm", *TARGET_NAMES})


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
            it is read beside ``hole_rate``.
        hole_desirability: Coverage softened per tile in dB, then averaged.
        overlap_desirability: The overlap margin softened per co-band neighbour.
        served_desirability: The served ratio per tile, averaged over occupied
            tiles. These three are what :func:`quality_index` reads; the five
            above are reported and nothing else.
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


def weights(cfg: DictConfig) -> np.ndarray:
    """``kpi.weights`` in :data:`TARGET_NAMES` order, normalised to sum to one.

    Normalising is what makes :func:`quality_index` a weighted geometric *mean*
    rather than a product that shrinks as KPIs are added, so the result stays in
    ``[0, 1]`` and comparable across weightings.

    Raises:
        ValueError: When ``kpi.weights`` is absent or incomplete, a weight is
            negative or non-finite, or every weight is zero.
    """
    block = cfg.kpi.get("weights")
    if block is None:
        raise ValueError(
            "configs/kpi.yaml has no `weights` block. The objective needs one weight per KPI "
            f"({', '.join(TARGET_NAMES)})."
        )
    missing = [name for name in TARGET_NAMES if name not in block]
    if missing:
        raise ValueError(f"kpi.weights has no entry for {', '.join(missing)}")
    values = np.array([float(block[name]) for name in TARGET_NAMES], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("kpi.weights must be finite and non-negative")
    if not values.any():
        raise ValueError("kpi.weights are all zero, so every configuration would tie")
    return values / values.sum()


def desirability(kpis: Sequence[KpiVector]) -> np.ndarray:
    """Each targeted KPI in ``[0, 1]``; larger is better.

    A gather, not a transform. Every one of them was softened per tile in
    :mod:`src.kpi` at measure time, so there is no threshold left for this
    module to apply, and no sign to fix - :data:`TARGET_NAMES` are all
    maximised.

    That split is deliberate. Softening an aggregate rate here instead would
    leave the hard comparison inside each KPI untouched: a tile 0.1 dB below the
    hole threshold and one 40 dB below are the same tile to ``hole_rate``, and
    no curve applied afterwards can tell them apart. It also means re-scoring an
    archived run under different weights is honest, while a changed threshold is
    a new measurement rather than a re-score.

    Returns:
        Shape ``[len(kpis), len(TARGET_NAMES)]``.
    """
    return np.array([[getattr(kpi, name) for name in TARGET_NAMES] for kpi in kpis], dtype=float)


def quality_index(kpis: Sequence[KpiVector], cfg: DictConfig) -> np.ndarray:
    """Overall network quality in ``[0, 1]``; larger is better.

    The weighted geometric mean ``prod_k d_k ** w_k`` of the desirabilities
    (Derringer & Suich, 1980, *Simultaneous Optimization of Several Response
    Variables*), with ``w`` from :func:`weights`.

    Multiplicative rather than additive on purpose: it is **non-compensatory**,
    so one KPI collapsing drives the whole index to zero and cannot be bought
    back by the others. That is what a threshold means, and it is what a
    weighted sum does not do.

    This is both the objective the search maximises and the published quality
    figure - one number, so the thing being optimised and the thing being
    reported cannot drift apart.

    Raises:
        ValueError: When ``kpi.weights`` is unusable; see :func:`weights`.
    """
    return np.prod(desirability(kpis) ** weights(cfg), axis=1)


def score_frame(frame: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """:func:`quality_index` over a table whose columns are :data:`KPI_NAMES`.

    The history tables carry the KPIs as columns rather than as objects; this is
    the one place that converts, so no caller rebuilds the score by hand.

    Raises:
        KeyError: When a KPI column is missing.
        ValueError: When ``kpi.weights`` is unusable.
    """
    kpis = [KpiVector.from_mapping(row) for row in frame[list(KPI_NAMES)].to_dict("records")]
    return quality_index(kpis, cfg)


def best_by_score(kpis: Sequence[KpiVector], cfg: DictConfig) -> int:
    """Index of the highest :func:`quality_index`.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty, or ``kpi.weights`` is unusable.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax(quality_index(kpis, cfg)))
