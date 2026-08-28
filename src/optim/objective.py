"""Turn a tilt configuration into the number an optimizer consumes.

Sits between :mod:`src.optim.space` and :mod:`src.kpi`, and is the only place
the two optimizers agree on what "better" means. It does not define any KPI —
:mod:`src.kpi` does — it decides how a candidate is evaluated and in what form
the answer comes back.

Two evaluation sources, one interface
-------------------------------------
``surrogate``  cheap, approximate. What BO and MARL use in their inner loops.
``sionna``     expensive, exact. What validates a result before it is reported.

Both return the same KPI mapping, so switching between them is a config change
rather than a code path. That is what makes the validation in PROJECT.md
section 26 a re-run of the same evaluation rather than a separate
implementation that might disagree for its own reasons.

Count the expensive evaluations
-------------------------------
PROJECT.md section 17 compares BO and MARL on the number of Sionna-RT
evaluations each consumed. That count is a headline result, so it is tracked
here rather than estimated afterwards — an estimate reconstructed from logs is
exactly the number a reader will question.
"""

from typing import Any

import numpy as np
from omegaconf import DictConfig


class Objective:
    """Evaluate tilt configurations for an optimizer.

    Holds the scene, the surrogate, the cell-band table and the UE density, so
    an optimizer can be handed one object and call it without knowing where the
    KPIs come from.
    """

    def __init__(self, space: Any, cfg: DictConfig, surrogate: Any = None) -> None:
        """Assemble everything needed to score a configuration.

        Args:
            space: A :class:`src.optim.space.TiltSpace`.
            cfg: Composed config; uses ``cfg.kpi`` and the active optimizer
                ``objective.source``.
            surrogate: A fitted surrogate. Required when the source is
                ``surrogate``.

        Raises:
            NotImplementedError: Always — implement this module first.
            ValueError: Once implemented, when the source is ``surrogate`` and
                none was supplied.

        Notes:
            Load the Sionna-RT scene lazily. Constructing it costs minutes, and
            a run configured to use the surrogate throughout should not pay for
            a simulator it never calls.
        """
        # TODO(1): store space, cfg, surrogate, and a zeroed evaluation counter
        # TODO(2): raise ValueError when source is surrogate and surrogate is None
        # TODO(3): defer scene construction until the first sionna evaluation
        raise NotImplementedError("src.optim.objective.Objective.__init__")

    def kpis(self, tilt: np.ndarray, *, source: str | None = None) -> dict:
        """Evaluate one configuration and return its five KPIs.

        Args:
            tilt: Absolute tilts in degrees, in cell-band table order.
            source: Override the configured source for this call — used to
                validate a surrogate-selected candidate against Sionna-RT.

        Returns:
            The KPI mapping from :func:`src.kpi.vector.kpi_vector`.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Validate the configuration before evaluating it. An out-of-bounds
            tilt produces a perfectly plausible KPI vector for a network that
            cannot be built, and nothing downstream will notice.

            Increment the Sionna-RT counter here, not at the call site — a
            counter maintained by callers is one that eventually misses a path.

        Example:
            >>> k = objective.kpis(tilt)
            >>> k_true = objective.kpis(tilt, source="sionna")
        """
        # TODO(1): sampling.assert_within_bounds(tilt, self.space.table)
        # TODO(2): surrogate -> features.transform then predict, mapped to KPI names
        # TODO(3): sionna    -> radiomap.evaluate then kpi.vector.kpi_vector
        # TODO(4): increment the evaluation counter for the source actually used
        raise NotImplementedError("src.optim.objective.Objective.kpis")

    def scalar(self, tilt: np.ndarray) -> float:
        """Evaluate one configuration as a single number to maximise.

        Args:
            tilt: Absolute tilts in degrees.

        Returns:
            The scalarized objective. Larger is better, so a minimising
            optimizer must negate it.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            For optimizers that cannot express the lexicographic goal. The sign
            convention is stated in the return description and nowhere else —
            check it rather than assuming, because a flipped sign here produces
            a run that converges confidently on the worst configuration it can
            find.

        Example:
            >>> objective.scalar(tilt)
        """
        # TODO(1): kpis(tilt) then kpi.vector.scalarize
        raise NotImplementedError("src.optim.objective.Objective.scalar")

    def best(self, candidates: np.ndarray) -> np.ndarray:
        """Select the best configuration from a set, under the KPI priority.

        Args:
            candidates: Shape ``(n, n_dims)``, in degrees.

        Returns:
            The winning configuration, shape ``(n_dims,)``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Use :func:`src.kpi.vector.lexicographic_better` in a single pass
            rather than sorting. With tolerance slack the relation is not
            transitive, so it is not a valid sort key — handing it to ``sorted``
            gives an order that depends on the input order.

        Example:
            >>> optimized_tilt = objective.best(candidates)
        """
        # TODO(1): evaluate every candidate
        # TODO(2): single pass, keeping the incumbent under lexicographic_better
        raise NotImplementedError("src.optim.objective.Objective.best")

    @property
    def n_sionna_evaluations(self) -> int:
        """How many Sionna-RT solves this objective has consumed.

        Returns:
            The count since construction.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            A headline result for PROJECT.md section 17, not diagnostics.
            Report it beside every optimization outcome.
        """
        # TODO(1): return the counter
        raise NotImplementedError("src.optim.objective.Objective.n_sionna_evaluations")
