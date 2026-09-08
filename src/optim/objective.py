"""The KPI vector, its sign convention, and the rule that picks one winner.

The five definitions live in :mod:`src.kpi` and are not restated here. What
this module adds is what an optimizer needs around them: one value object
carrying all five, the orientation that turns them into "larger is better", and
the lexicographic rule that reduces a Pareto front to the single configuration
a deployment can act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi import (
    band_priority_score,
    expected_rsrp_improvement,
    hole_rate,
    overlap_rate,
    weak_rate,
)

# Priority order, highest first. The ordering is the point: it is what the
# lexicographic pick walks, and reordering it changes which configuration wins.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "expected_rsrp_improvement",
    "band_priority_score",
    "weak_rate",
)

# The KPIs where larger is better. Named once, so no call site re-decides a
# sign; the other three are minimised.
MAXIMISED = frozenset({"expected_rsrp_improvement", "band_priority_score"})


@dataclass(frozen=True)
class KpiVector:
    """One configuration's score on all five KPIs, in priority order.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        expected_rsrp_improvement: Mean sigmoid of the serving-RSRP change over
            the MDT locations, against what the UEs actually reported.
        band_priority_score: UE-weighted share served by higher-priority bands.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
    """

    hole_rate: float
    overlap_rate: float
    expected_rsrp_improvement: float
    band_priority_score: float
    weak_rate: float

    def as_dict(self) -> dict[str, float]:
        """The five values keyed by name, shaped for Ax's ``raw_data``."""
        return {name: float(value) for name, value in asdict(self).items()}

    def as_array(self) -> np.ndarray:
        """The five values in :data:`KPI_NAMES` order."""
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
    band_labels: Sequence[str],
    mdt: pd.DataFrame,
    cfg: DictConfig,
) -> KpiVector:
    """Score one radio map on all five KPIs.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: The UE reports. ``tile_row`` and ``tile_col`` weight the band
            priority score; the ``rsrp_*`` columns are the measured serving RSRP
            the expected improvement is scored against.
        cfg: Composed config; the KPIs read ``cfg.kpi``.
    """
    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        expected_rsrp_improvement=expected_rsrp_improvement(rsrp, mdt, cfg),
        band_priority_score=band_priority_score(rsrp, band_labels, mdt, cfg),
        weak_rate=weak_rate(rsrp, cfg),
    )


def ax_objective() -> str:
    """Ax's objective expression: comma-separated, minus where minimised.

    Derived from :data:`KPI_NAMES` and :data:`MAXIMISED` rather than written
    out, so the sign convention exists in one place and a newly added KPI
    cannot reach Ax pointing the wrong way.
    """
    return ", ".join(name if name in MAXIMISED else f"-{name}" for name in KPI_NAMES)


def as_maximised(kpis: Sequence[KpiVector]) -> np.ndarray:
    """The KPI matrix reoriented so larger is better in every column.

    Returns:
        Shape ``[len(kpis), 5]`` in :data:`KPI_NAMES` order, with the three
        minimised KPIs negated. Units are untouched: this is an orientation,
        not a normalisation, because nothing downstream compares one column
        against another.
    """
    signs = np.array([1.0 if name in MAXIMISED else -1.0 for name in KPI_NAMES])
    return np.array([kpi.as_array() for kpi in kpis], dtype=float) * signs


def pareto_mask(kpis: Sequence[KpiVector]) -> np.ndarray:
    """Which entries are non-dominated, as a boolean mask.

    Domination is the standard strict rule on the maximised orientation: one
    point dominates another when it is at least equal on all five and strictly
    better on at least one. Tolerances play no part here — they belong to the
    single-winner pick, not to the front.
    """
    values = as_maximised(kpis)
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominated = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        mask[index] = not dominated.any()
    return mask


def hypervolume(kpis: Sequence[KpiVector], reference: KpiVector) -> float:
    """Volume dominated by these KPIs above a reference point.

    The one number that says whether a multi-objective run is making progress:
    it rises when the front pushes outward and is flat when it does not.

    Args:
        kpis: The evaluations to measure.
        reference: The corner the volume is measured from. Pass the incumbent,
            and the result is then the improvement over what is deployed — a
            configuration worse than the incumbent on any KPI contributes
            nothing, which is the intended reading.

    Returns:
        The dominated volume, in the product of the five KPIs' own units.

        Expect very small numbers: this is a five-way product of improvements
        that are themselves fractions, so a real gain can read as ``1e-10``.
        Only the trend carries meaning — plot it on a log scale, and do not
        compare it against a run that used a different reference point.
    """
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    values = torch.as_tensor(as_maximised(kpis), dtype=torch.double)
    point = torch.as_tensor(as_maximised([reference])[0], dtype=torch.double)
    return float(Hypervolume(ref_point=point).compute(values))


def hypervolume_trace(kpis: Sequence[KpiVector], reference: KpiVector) -> np.ndarray:
    """Hypervolume after each evaluation, for a progress plot.

    Monotonically non-decreasing by construction, since each prefix is a subset
    of the next.
    """
    return np.array(
        [hypervolume(kpis[: index + 1], reference) for index in range(len(kpis))], dtype=float
    )


def tolerances(cfg: DictConfig) -> np.ndarray:
    """Per-KPI tie thresholds from ``kpi.tolerance``, in :data:`KPI_NAMES` order.

    Raises:
        ValueError: When ``kpi.tolerance`` is absent or incomplete. There is no
            safe default: a missing tolerance is an exact comparison, which on
            a continuous KPI never ties, and the priority order then collapses
            to optimizing ``hole_rate`` alone.
    """
    tolerance = cfg.kpi.get("tolerance")
    if tolerance is None:
        raise ValueError(
            "configs/kpi.yaml has no `tolerance` block. The lexicographic pick needs one "
            f"entry per KPI ({', '.join(KPI_NAMES)}); without it the priority does not bind."
        )
    missing = [name for name in KPI_NAMES if name not in tolerance]
    if missing:
        raise ValueError(f"kpi.tolerance has no entry for {', '.join(missing)}")
    return np.array([float(tolerance[name]) for name in KPI_NAMES], dtype=float)


def lexicographic_best(kpis: Sequence[KpiVector], cfg: DictConfig) -> int:
    """Index of the winner under the priority order, with tolerances.

    Walks the KPIs highest priority first. A gap within a KPI's tolerance is a
    tie and the comparison moves down; the first KPI that separates the two
    decides, and nothing below it is consulted.

    Returns:
        The index of the best entry. A tie resolves to the earlier index, which
        is what makes the incumbent hold when no candidate actually beats it.

    Raises:
        ValueError: When ``kpis`` is empty, or ``kpi.tolerance`` is unusable.

    Notes:
        Deliberately a single pass rather than a sort. The relation is not
        transitive once ties carry slack — ``a`` can tie ``b`` and ``b`` tie
        ``c`` while ``a`` beats ``c`` — so it is not a valid sort key, and a
        full ranking would depend on the order candidates arrived in.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")

    values = as_maximised(kpis)
    tolerance = tolerances(cfg)

    best = 0
    for index in range(1, len(values)):
        gap = values[index] - values[best]
        decisive = np.flatnonzero(np.abs(gap) > tolerance)
        if decisive.size and gap[decisive[0]] > 0:
            best = index
    return best
