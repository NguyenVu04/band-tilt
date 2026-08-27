"""KPI 5 — the UE-weighted Band Priority Score, PROJECT.md section 14.

The four coverage KPIs say nothing about which band serves a location. This one
does: it rewards configurations where high-priority bands dominate in places
where users actually are.

    BPS = sum over g of rho(g) * w[b*(g)]  /  sum over g of rho(g)

with ``rho(g)`` the UE observation count in grid cell ``g``
(:func:`src.data.ue_density.ue_density`), ``b*(g)`` the dominant band
(:func:`src.kpi.serving.dominant_band`), and ``w_b`` the positive priority
weight declared per band in ``configs/radio.yaml``.

This is the only KPI that is MAXIMISED
--------------------------------------
Every other KPI in the project is minimised. That asymmetry is the most common
source of sign errors here — an optimizer that negates the wrong term will
quietly drive high-priority bands out of dense areas while every other number
improves. :mod:`src.kpi.vector` is the one place the directions are encoded, and
this function returns the raw score, never a negated one.

The weighting is the point
--------------------------
Two grid cells with the same dominant band contribute differently if one holds
a hundred UE observations and the other holds ten. That is intended: coverage
quality in an empty field is worth less than coverage quality where the users
are. It also means the score inherits whatever bias the MDT sampling had, which
is worth stating when reporting it.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def band_weights(table: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Priority weight per band index, ``w_b``.

    Args:
        table: The cell-band table, giving the band index ordering.
        cfg: Composed config; uses ``cfg.radio.bands[*].priority_weight``.

    Returns:
        Weights per band index, shape ``(n_bands,)``.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when a weight is missing or not strictly
            positive.

    Notes:
        PROJECT.md section 14 requires ``w_b > 0``. A zero or negative weight
        makes the score non-monotonic in band quality and the objective stops
        meaning what section 14 says it means.

        Only the ORDER of the weights is fixed by the spec
        (``w_low < w_mid < w_high``); the values are open parameters
        (section 30 item 2). Their scale changes the magnitude of the score but
        not the ranking of configurations, so it matters mainly for
        normalisation.

    Example:
        >>> w = band_weights(table, cfg)
    """
    # TODO(1): read priority_weight for each band in the table band ordering
    # TODO(2): raise ValueError on a missing, zero or negative weight
    raise NotImplementedError("src.kpi.band_priority.band_weights")


def band_priority_score(
    b_star: np.ndarray,
    rho: np.ndarray,
    weights: np.ndarray,
) -> float:
    """KPI 5 — the UE-weighted Band Priority Score.

    Args:
        b_star: Dominant band index per grid cell from
            :func:`src.kpi.serving.dominant_band`, shape ``(|G|,)``.
        rho: UE observation count per grid cell from
            :func:`src.data.ue_density.ue_density`, shape ``(|G|,)``.
        weights: Priority weight per band from :func:`band_weights`.

    Returns:
        The score, a UE-weighted average of band priority weights. **Maximised.**

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when ``b_star`` and ``rho`` disagree on
            the grid size.

    Notes:
        Assert the grid alignment before computing anything — the two arrays
        broadcast happily at mismatched offsets and the resulting score is
        plausible and wrong. Use
        :func:`src.data.ue_density.assert_grid_aligned`.

        The denominator is ``N_UE``, the total observation count. Grid cells
        with ``rho = 0`` contribute nothing to either sum, so empty areas are
        correctly ignored without being dropped from the grid.

        When ``N_UE`` is zero the score is undefined; raise rather than
        returning a default, because a zero-density map means the density was
        built from the wrong frame.

    Example:
        >>> band_priority_score(b_star, rho, w)
        2.31
    """
    # TODO(1): assert b_star and rho index the same grid
    # TODO(2): numerator = (rho * weights[b_star]).sum()
    # TODO(3): raise when rho.sum() == 0, rather than returning a default
    # TODO(4): return numerator / rho.sum()
    raise NotImplementedError("src.kpi.band_priority.band_priority_score")
