"""Spatial maps — PROJECT.md section 27.3.

A KPI table says hole rate fell from 8% to 5%. It does not say whether the
remaining holes moved to the edge of the area or opened up in the town centre,
and those are very different outcomes for the same number. These maps are how
that gets seen.

The recommended set, from PROJECT.md section 27.3: baseline RSRP, optimized
RSRP, hole, weak, overlap, UE density, and dominant band.

Plot on the evaluation grid, not on scattered points
----------------------------------------------------
Every map here is a reshape of a length ``|G|`` vector back onto the grid built
by :func:`src.data.ue_density.build_grid`. Keeping the flat index and the grid
shape together is what makes the maps comparable to each other and to the KPIs —
a map assembled by scattering coordinates instead can be visually convincing and
spatially wrong.

Diverging colour for differences, sequential for levels
-------------------------------------------------------
An RSRP map is a level: sequential colour, zero has no special meaning. A
baseline-minus-optimized map is a difference: diverging colour centred on zero,
so improvement and regression are distinguishable at a glance. Using a
sequential map for a difference hides the sign, which is the only thing the
reader is looking for.

Styling comes from :mod:`src.utils.plotting`, which is already implemented.
"""

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def rsrp_map(rsrp: np.ndarray, cfg: DictConfig, title: str = "RSRP") -> Any:
    """Plot the strongest received signal across the area.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.
        cfg: Composed config; uses ``cfg.radio.grid`` for the grid shape.
        title: Plot title.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Plot ``R_max`` from :func:`src.kpi.serving.max_rsrp`, and mark the hole
        and weak thresholds on the colour bar. Without them the reader cannot
        tell an acceptable region from a failing one, which is the entire
        question the map is being asked.

        Clip the display floor rather than letting ``-inf`` set the colour
        scale, or the whole map renders as one colour.

    Example:
        >>> fig = rsrp_map(rsrp, cfg, title="Baseline RSRP")
    """
    # TODO(1): r_max = serving.max_rsrp(rsrp), reshaped to the grid
    # TODO(2): sequential colormap from src.utils.plotting
    # TODO(3): mark cfg.kpi.hole_dbm and cfg.kpi.weak_dbm on the colour bar
    raise NotImplementedError("src.evaluation.analysis.rsrp_map")


def coverage_map(rsrp: np.ndarray, cfg: DictConfig) -> Any:
    """Plot hole, weak and good coverage as three categories.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.
        cfg: Composed config; uses ``cfg.kpi`` thresholds and
            ``cfg.radio.grid``.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Categorical, not continuous: the thresholds are the point. Hole and weak
        partition the covered area by construction, so a pixel that appears in
        both is a bug in :mod:`src.kpi.coverage`, and this map is where it
        becomes visible.

    Example:
        >>> fig = coverage_map(rsrp, cfg)
    """
    # TODO(1): classify each grid cell as hole, weak or good using cfg.kpi thresholds
    # TODO(2): render as three discrete colours, not a continuous ramp
    raise NotImplementedError("src.evaluation.analysis.coverage_map")


def overlap_map(n_ov: np.ndarray, cfg: DictConfig) -> Any:
    """Plot the number of overlapping neighbours per location.

    Args:
        n_ov: Neighbour count per grid cell from
            :func:`src.kpi.coverage.overlap_neighbors`.
        cfg: Composed config; uses ``cfg.radio.grid``.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Shows what the two overlap KPIs cannot separate on their own: whether
        overlap is spread thinly across the area or concentrated into a few
        severe pockets. Those call for different fixes.

    Example:
        >>> fig = overlap_map(n_ov, cfg)
    """
    # TODO(1): reshape n_ov to the grid
    # TODO(2): discrete colour steps — the values are small integers
    raise NotImplementedError("src.evaluation.analysis.overlap_map")


def ue_density_map(rho: np.ndarray, cfg: DictConfig) -> Any:
    """Plot the UE observation density used to weight the Band Priority Score.

    Args:
        rho: UE density per grid cell from
            :func:`src.data.ue_density.ue_density`.
        cfg: Composed config; uses ``cfg.radio.grid``.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Worth publishing beside every Band Priority Score. The score inherits
        whatever bias the MDT sampling had, and this map is the only place a
        reader can judge how much of the score is driven by a handful of dense
        cells.

        Use a log scale unless the density is unusually flat; a few dense cells
        otherwise flatten everything else to the background colour.

    Example:
        >>> fig = ue_density_map(rho, cfg)
    """
    # TODO(1): reshape rho to the grid
    # TODO(2): log-scale the colour normalisation, handling zeros
    raise NotImplementedError("src.evaluation.analysis.ue_density_map")


def dominant_band_map(b_star: np.ndarray, table: pd.DataFrame, cfg: DictConfig) -> Any:
    """Plot which band dominates at each location.

    Args:
        b_star: Dominant band index per grid cell from
            :func:`src.kpi.serving.dominant_band`.
        table: The cell-band table, for band labels.
        cfg: Composed config; uses ``cfg.radio.grid``.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is the visual counterpart of the Band Priority Score, and the one
        map that shows whether multi-band coordination actually did anything.
        Overlay the UE density contours: the score rewards high-priority bands
        dominating where users are, and the two maps together show whether that
        happened or whether the score improved somewhere empty.

    Example:
        >>> fig = dominant_band_map(b_star, table, cfg)
    """
    # TODO(1): reshape b_star to the grid
    # TODO(2): categorical colours, labelled with band ids from the table
    # TODO(3): overlay UE density contours
    raise NotImplementedError("src.evaluation.analysis.dominant_band_map")


def difference_map(before: np.ndarray, after: np.ndarray, cfg: DictConfig) -> Any:
    """Plot the change between two configurations.

    Args:
        before: A per-grid-cell quantity for the baseline.
        after: The same quantity for the optimized configuration.
        cfg: Composed config; uses ``cfg.radio.grid``.

    Returns:
        The matplotlib figure.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Diverging colormap centred on zero, and centred honestly — an
        automatically scaled colour range that puts zero off-centre makes a
        uniformly small regression look like an improvement.

        This is where an aggregate improvement is checked for a bad trade: a
        configuration that improves the average by degrading a dense area is
        visible here and nowhere in the KPI table.

    Example:
        >>> fig = difference_map(r_max_baseline, r_max_optimized, cfg)
    """
    # TODO(1): reshape both to the grid and subtract
    # TODO(2): diverging colormap from src.utils.plotting, symmetric about zero
    raise NotImplementedError("src.evaluation.analysis.difference_map")
