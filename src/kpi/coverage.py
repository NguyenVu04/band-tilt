"""Coverage KPIs 1-4 — hole, overlap, mean overlap neighbours and weak rate.

Four numbers, in the priority order ``cfg.kpi.order`` fixes:

``hole_rate``              fraction of the grid with no usable signal at all
``overlap_rate``           fraction of the grid where neighbours pile onto the
                           serving cell
``weak_rate``              fraction of the grid that is covered but poorly
``mean_overlap_neighbors`` how many neighbours are involved, where overlap
                           happens

Hole and weak partition the grid; they cannot both be true at one location.
Overlap is independent of both and is evaluated only where the serving cell is
itself covered.

All four are computed from the same RSRP array in one pass wherever possible.
The grid is large, the optimizer calls this constantly, and four separate
reductions over the same array is the easiest avoidable cost in the project.

Thresholds come from ``configs/kpi.yaml``. Never write ``-120``, ``-90`` or
``6`` into this module: the same constants appear in the surrogate acceptance
report and the MARL reward, and a literal here is how those three drift apart.
"""

import numpy as np
from omegaconf import DictConfig


def hole_rate(r_max: np.ndarray, cfg: DictConfig) -> float:
    """KPI 1 — percentage of the grid classified as a coverage hole.

    Args:
        r_max: Strongest signal per grid cell from
            :func:`src.kpi.serving.max_rsrp`, shape ``(|G|,)``.
        cfg: Composed config; uses ``cfg.kpi.hole_dbm``.

    Returns:
        Hole rate as a percentage in ``[0, 100]``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``H(x) = 1[R_max(x) <= hole_dbm]``, with ``hole_dbm`` from
        ``cfg.kpi``. The
        comparison is inclusive at the threshold, and the denominator is the
        full grid ``|G|``.

        This is the highest-priority objective in the lexicographic order.

    Example:
        >>> hole_rate(r_max, cfg)
        3.42
    """
    # TODO(1): mask r_max <= cfg.kpi.hole_dbm
    # TODO(2): return mask.mean() * 100
    raise NotImplementedError("src.kpi.coverage.hole_rate")


def weak_rate(r_max: np.ndarray, cfg: DictConfig) -> float:
    """KPI 5 — percentage of the grid classified as weak coverage.

    Args:
        r_max: Strongest signal per grid cell, shape ``(|G|,)``.
        cfg: Composed config; uses ``cfg.kpi.hole_dbm`` and ``cfg.kpi.weak_dbm``.

    Returns:
        Weak rate as a percentage in ``[0, 100]``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``W(x) = 1[hole_dbm < R_max(x) <= weak_dbm]``, both from ``cfg.kpi``.
        The band is half-open on both sides, so hole and weak are disjoint by
        construction — a location cannot be counted twice, and a bug that makes
        it possible shows up as ``hole + weak > 100``.

        Lowest priority of the three coverage KPIs.

    Example:
        >>> weak_rate(r_max, cfg)
        18.7
    """
    # TODO(1): mask (r_max > cfg.kpi.hole_dbm) & (r_max <= cfg.kpi.weak_dbm)
    # TODO(2): return mask.mean() * 100
    raise NotImplementedError("src.kpi.coverage.weak_rate")


def overlap_neighbors(per_cell: np.ndarray, serving: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Count overlapping neighbours at each location.

    Args:
        per_cell: Per-cell RSRP from :func:`src.kpi.serving.cell_rsrp`, shape
            ``(n_cells, |G|)``.
        serving: Serving cell index per grid cell, shape ``(|G|,)``.
        cfg: Composed config; uses ``cfg.kpi.hole_dbm`` and
            ``cfg.kpi.overlap_margin_db``.

    Returns:
        ``N_ov(x)``, the neighbour count per grid cell, shape ``(|G|,)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        A neighbour ``j`` counts when both conditions hold: the serving cell
        is covered, ``R_s(x) > hole_dbm``; and the
        neighbour is within the margin, ``R_s(x) - R_j(x) < overlap_margin_db``.

        The serving cell must be excluded from its own count — it trivially
        satisfies the margin condition against itself, and including it would
        add exactly one to every covered location.

        Uncovered locations contribute zero, not a masked or missing value. They
        are not overlapping; they have nothing to overlap with.

    Example:
        >>> n_ov = overlap_neighbors(per_cell, serving, cfg)
    """
    # TODO(1): gather serving RSRP per location from per_cell and serving
    # TODO(2): covered = serving_rsrp > cfg.kpi.hole_dbm
    # TODO(3): within = (serving_rsrp - per_cell) < cfg.kpi.overlap_margin_db
    # TODO(4): zero out the serving cell own row before counting
    # TODO(5): count over axis 0, zeroed where not covered
    raise NotImplementedError("src.kpi.coverage.overlap_neighbors")


def overlap_rate(n_ov: np.ndarray) -> float:
    """KPI 2 — percentage of the grid where any overlap occurs.

    Args:
        n_ov: Neighbour count per grid cell from :func:`overlap_neighbors`.

    Returns:
        Overlap rate as a percentage in ``[0, 100]``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``O(x) = 1[N_ov(x) > 0]``, over the full grid.
        Second priority, above weak rate.

    Example:
        >>> overlap_rate(n_ov)
        22.1
    """
    # TODO(1): mask n_ov > 0
    # TODO(2): return mask.mean() * 100
    raise NotImplementedError("src.kpi.coverage.overlap_rate")


def mean_overlap_neighbors(n_ov: np.ndarray) -> float:
    """KPI 3 — average number of overlapping neighbours per evaluation location.

    Args:
        n_ov: Neighbour count per grid cell from :func:`overlap_neighbors`.

    Returns:
        Mean neighbour count over ALL locations, ``sum(n_ov) / |G|``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The denominator is ``|G|``, every evaluation location, including the
        ones where nothing overlaps and the ones that are coverage holes::

            MeanOverlapNeighbors = (1 / |G|) * sum_g N_ov(g)

        This is a change of definition. It was previously the mean over
        OVERLAPPING locations only, which made it independent of overlap rate —
        one said how often overlap happens, the other how severe it is where it
        does. Averaging over ``|G|`` instead makes this largely a rescaling of
        overlap rate: the two now move together, and a configuration that
        concentrates severe overlap in a few places is no longer distinguished
        from one that spreads mild overlap widely. Recorded as a cost in
        docs/adr/0001; the spec is nonetheless what this must implement.

        No empty-denominator case exists any more — ``|G|`` is never zero — so
        an all-zero ``n_ov`` returns ``0.0`` naturally rather than by a guard.

        This KPI now ranks THIRD in the lexicographic order, ahead of Band
        Priority Score and weak rate, per ``cfg.kpi.order``.

    Example:
        >>> mean_overlap_neighbors(n_ov)
        0.9
    """
    # TODO(1): return n_ov.sum() / n_ov.size
    raise NotImplementedError("src.kpi.coverage.mean_overlap_neighbors")
