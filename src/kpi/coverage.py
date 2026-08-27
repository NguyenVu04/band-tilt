"""Coverage KPIs 1-4 — PROJECT.md sections 10 to 13.

Four numbers, in the priority order PROJECT.md section 17 fixes:

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
        ``H(x) = 1[R_max(x) <= hole_dbm]`` (PROJECT.md section 10). The
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
    """KPI 2 — percentage of the grid classified as weak coverage.

    Args:
        r_max: Strongest signal per grid cell, shape ``(|G|,)``.
        cfg: Composed config; uses ``cfg.kpi.hole_dbm`` and ``cfg.kpi.weak_dbm``.

    Returns:
        Weak rate as a percentage in ``[0, 100]``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``W(x) = 1[hole_dbm < R_max(x) <= weak_dbm]`` (PROJECT.md section 11).
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
        A neighbour ``j`` counts when both conditions of PROJECT.md section 12
        hold: the serving cell is covered, ``R_s(x) > hole_dbm``; and the
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
    """KPI 3 — percentage of the grid where any overlap occurs.

    Args:
        n_ov: Neighbour count per grid cell from :func:`overlap_neighbors`.

    Returns:
        Overlap rate as a percentage in ``[0, 100]``. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``O(x) = 1[N_ov(x) > 0]`` (PROJECT.md section 12), over the full grid.
        Second priority, above weak rate.

    Example:
        >>> overlap_rate(n_ov)
        22.1
    """
    # TODO(1): mask n_ov > 0
    # TODO(2): return mask.mean() * 100
    raise NotImplementedError("src.kpi.coverage.overlap_rate")


def mean_overlap_neighbors(n_ov: np.ndarray) -> float:
    """KPI 4 — average number of overlapping neighbours, where overlap occurs.

    Args:
        n_ov: Neighbour count per grid cell from :func:`overlap_neighbors`.

    Returns:
        Mean neighbour count over overlapping locations only. Minimised.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The denominator is the number of OVERLAPPING locations, not ``|G|``
        (PROJECT.md section 13). That is the whole point of this KPI: overlap
        rate says how often overlap happens, this says how severe it is where it
        does.

        Dividing by ``|G|`` instead would make this a rescaled overlap rate and
        it would carry no information the third KPI does not already have.

        When nothing overlaps the denominator is zero. Return ``0.0`` rather
        than ``nan``: an optimizer comparing candidates cannot order a ``nan``,
        and no overlap is unambiguously the best case for this KPI.

    Example:
        >>> mean_overlap_neighbors(n_ov)
        1.8
    """
    # TODO(1): overlapping = n_ov > 0
    # TODO(2): return 0.0 when nothing overlaps
    # TODO(3): otherwise n_ov.sum() / overlapping.sum()
    raise NotImplementedError("src.kpi.coverage.mean_overlap_neighbors")
