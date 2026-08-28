"""Assemble, normalise and order the KPI vector — PROJECT.md sections 4 to 5.

The complete objective, in priority order, is::

    K = [K_H, K_O, K_ON, K_BPS, K_W]

minimising all but ``K_BPS``, which is maximised. The canonical all-minimising
form negates it (PROJECT.md section 5)::

    J = [K_H, K_O, K_ON, -K_BPS, K_W]

This module is the only place that direction is written down, and the only place
two candidates are compared.

Lexicographic, with slack
-------------------------
PROJECT.md section 5 states the priority as ``Hole > Overlap > MeanOverlap-
Neighbors > BPS > Weak``. Implemented literally, that degenerates: hole rate is
continuous, exact ties essentially never happen, and the comparison never
reaches the second objective. So each objective carries a tolerance from
``configs/kpi.yaml`` — a difference smaller than the tolerance is a tie, and the
comparison moves on.

Note that weak rate is **last**. Under the previous formulation it ranked third,
above overlap severity and band coordination; a configuration may now trade weak
coverage for either of those, which it previously could not. See docs/adr/0002.

Choosing those tolerances is a real decision, not a formality. Too tight and
this is single-objective optimization on hole rate; too loose and the priority
order stops binding.

Scalarization is a fallback, not the definition
-----------------------------------------------
Some optimizers cannot express a lexicographic goal and need one number.
:func:`scalarize` provides it, but a weighted sum is a lossy encoding of the
priority — it can always be made to trade a lot of hole rate for enough of the
other four, which the lexicographic order forbids outright. Prefer
:func:`lexicographic_better` where the optimizer allows it, and say which was
used when reporting.

Normalise before weighting. Always.
-----------------------------------
Hole rate is a percentage, mean overlap neighbours is a small count, and the
Band Priority Score is on whatever scale the band weights happen to use.
Applying weights to raw values silently reorders the priority regardless of what
the weights say — the exact failure PROJECT.md section 25.3 warns about.
"""

import numpy as np
from omegaconf import DictConfig

#: Canonical KPI order — PROJECT.md section 5. Every array in this module follows
#: it, and it must stay identical to ``cfg.kpi.order`` — which
#: :func:`src.config.validate_config` is specified to check, at TODO(5).
#:
#: Reordering this invalidates any result already saved, because saved KPI
#: vectors index by position and carry no names. Nothing has been produced yet,
#: which is the only reason the reorder below was safe to make.
KPI_NAMES = (
    "hole_rate",
    "overlap_rate",
    "mean_overlap_neighbors",
    "band_priority_score",
    "weak_rate",
)


def kpi_vector(rsrp: np.ndarray, rho: np.ndarray, table: object, cfg: DictConfig) -> dict:
    """Compute all five KPIs from one radio map.

    Args:
        rsrp: RSRP in dBm, shape ``(n_cell_bands, |G|)``.
        rho: UE density per grid cell, shape ``(|G|,)``.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        cfg: Composed config; uses ``cfg.kpi`` and ``cfg.radio.bands``.

    Returns:
        A mapping from each name in :data:`KPI_NAMES` to its value.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The single entry point for scoring a configuration. Bayesian
        Optimization, MARL, the surrogate labels and the final validation all
        call this one function, which is what makes the comparison in PROJECT.md
        section 25 valid.

        Returns a mapping rather than an array so a caller cannot silently
        misread position 3 as position 4. :func:`as_array` converts when a
        numeric vector is needed.

    Example:
        >>> k = kpi_vector(rsrp, rho, table, cfg)
        >>> k["hole_rate"]
    """
    # TODO(1): r_max = serving.max_rsrp(rsrp); per_cell = serving.cell_rsrp(...)
    # TODO(2): n_ov = coverage.overlap_neighbors(per_cell, serving.serving_cell(per_cell), cfg)
    # TODO(3): hole, weak, overlap, mean overlap neighbours
    # TODO(4): band_priority_score from serving.dominant_band and band_weights
    # TODO(5): return a dict keyed by KPI_NAMES
    raise NotImplementedError("src.kpi.vector.kpi_vector")


def as_array(kpis: dict) -> np.ndarray:
    """Convert a KPI mapping to an array in :data:`KPI_NAMES` order.

    Args:
        kpis: A mapping as returned by :func:`kpi_vector`.

    Returns:
        Shape ``(5,)``, ordered by :data:`KPI_NAMES`.

    Raises:
        NotImplementedError: Always — implement this module first.
        KeyError: Once implemented, when a KPI is missing.

    Example:
        >>> as_array(kpi_vector(rsrp, rho, table, cfg)).shape
        (5,)
    """
    # TODO(1): read KPI_NAMES in order, raising KeyError naming any missing key
    raise NotImplementedError("src.kpi.vector.as_array")


def normalize(kpis: dict, cfg: DictConfig) -> dict:
    """Put every KPI on a comparable scale, oriented so that larger is better.

    Args:
        kpis: A mapping as returned by :func:`kpi_vector`.
        cfg: Composed config; uses ``cfg.kpi.normalization`` and
            ``cfg.kpi.reference``.

    Returns:
        The normalised mapping, with every entry oriented so that a larger
        value is a better network.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown normalisation method.

    Notes:
        Flipping the four minimised KPIs here is what stops the Band Priority
        Score being a special case downstream. Do it once, in this function, and
        let the MARL reward and the scalarized objective consume uniformly
        oriented values.

        With ``cfg.kpi.reference`` set to the baseline KPI vector, a normalised
        value of zero means no better than the network we started with, which
        makes a reward or an objective directly readable.

    Example:
        >>> normalize(k, cfg)["hole_rate"]
    """
    # TODO(1): dispatch on cfg.kpi.normalization (minmax, zscore, reference)
    # TODO(2): negate the four minimised KPIs so larger is better everywhere
    # TODO(3): when cfg.kpi.reference is set, express each KPI relative to it
    raise NotImplementedError("src.kpi.vector.normalize")


def scalarize(kpis: dict, cfg: DictConfig) -> float:
    """Collapse the KPI vector to one number for optimizers that need it.

    Args:
        kpis: A mapping as returned by :func:`kpi_vector`.
        cfg: Composed config; uses ``cfg.kpi.weights`` and
            ``cfg.kpi.normalization``.

    Returns:
        A scalar objective. Larger is better, so a minimising optimizer must
        negate it.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when the weights do not satisfy
            ``hole > overlap > weak``.

    Notes:
        Normalise first, then weight — see the module docstring.

        Check the weight ordering rather than trusting it. PROJECT.md
        section 23 requires ``lambda_H > lambda_O > lambda_W``, and a config
        that violates it produces an objective that silently contradicts the
        stated priority while every individual KPI still looks correct.

    Example:
        >>> scalarize(k, cfg)
        -0.42
    """
    # TODO(1): normalize(kpis, cfg)
    # TODO(2): assert cfg.kpi.weights satisfies hole > overlap > weak, else ValueError
    # TODO(3): return the weighted sum over the normalised, uniformly oriented values
    raise NotImplementedError("src.kpi.vector.scalarize")


def lexicographic_better(left: dict, right: dict, cfg: DictConfig) -> bool:
    """Decide whether ``left`` is a better configuration than ``right``.

    Args:
        left: A KPI mapping.
        right: A KPI mapping to compare against.
        cfg: Composed config; uses ``cfg.kpi.order``, ``cfg.kpi.direction`` and
            ``cfg.kpi.tolerance``.

    Returns:
        ``True`` when ``left`` wins under the lexicographic order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is the authoritative statement of PROJECT.md section 5. Walk
        ``cfg.kpi.order``; at each objective, treat a difference smaller than
        its tolerance as a tie and continue; otherwise return on the first
        objective that separates them.

        The relation is not a total order — with slack it is not transitive, so
        it cannot be handed to a general-purpose sort. Use it to select a best
        candidate in a single pass, and say so when reporting a ranking.

    Example:
        >>> lexicographic_better(k_bo, k_baseline, cfg)
        True
    """
    # TODO(1): walk cfg.kpi.order
    # TODO(2): orient each comparison by cfg.kpi.direction
    # TODO(3): treat abs(difference) < cfg.kpi.tolerance[name] as a tie and continue
    # TODO(4): return False when every objective ties
    raise NotImplementedError("src.kpi.vector.lexicographic_better")
