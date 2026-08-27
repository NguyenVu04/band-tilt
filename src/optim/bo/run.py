"""The Bayesian Optimization loop — PROJECT.md section 20.

The loop is::

    initial samples -> model -> acquisition -> candidate
                            -> evaluate -> update -> repeat

Two nested surrogates, and they are not the same thing
------------------------------------------------------
This is the single most confusing part of the design, so it is worth stating
plainly. There are two approximations in play:

1. **The KPI surrogate** (:mod:`src.surrogate`), trained offline on Sionna-RT
   radio maps. It replaces the simulator.
2. **The BO surrogate** — the Gaussian process Ax fits online over evaluated
   configurations. It replaces the objective function within the search.

With ``objective.source: surrogate``, the GP is being fitted to the predictions
of another model. That is legitimate and it is what makes the search affordable,
but it stacks two error sources, and it is why
``cfg.optim.objective.validate_top_k`` exists: the best candidates are re-run
through Sionna-RT before anything is reported (PROJECT.md sections 20 and 26).

Dimensionality
--------------
One dimension per cell-band: 26 today, more as bands are added. Standard
GP-based BO becomes unreliable in the tens of dimensions, well before MARL does.
PROJECT.md section 25.4 asks for exactly this comparison — so record where it
degrades rather than quietly limiting the experiment to sizes where it looks
good.

Multi-objective or scalarized
-----------------------------
A single-objective acquisition (``ei``, ``logei``, ``ucb``) needs
``cfg.kpi.mode: scalarized`` and returns one configuration. ``qnehvi`` optimises
the KPI vector directly and returns a Pareto front, from which
:meth:`BayesianOptimizer.best` still has to pick under the lexicographic
priority. The two answer different questions; say which was used.
"""

from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig


class BayesianOptimizer:
    """Run Bayesian Optimization over the absolute-tilt space."""

    def __init__(self, space: Any, objective: Any, cfg: DictConfig) -> None:
        """Configure the optimizer.

        Args:
            space: A :class:`src.optim.space.TiltSpace`.
            objective: An :class:`src.optim.objective.Objective`.
            cfg: Composed config; uses ``cfg.optim`` (``backend``, ``search``,
                ``surrogate_model``, ``acquisition``, ``stopping``).

        Raises:
            NotImplementedError: Always — implement this module first.
            ValueError: Once implemented, when the acquisition function and
                ``cfg.kpi.mode`` are incompatible.

        Notes:
            Search in the unit cube via :meth:`src.optim.space.TiltSpace.to_unit`
            rather than in degrees. A GP with one length scale per dimension
            fitted on raw degrees is fitted on axes of different widths, and the
            resulting model is worse for no reason.

            Reject the incompatible combination up front: a single-objective
            acquisition with ``cfg.kpi.mode: lexicographic`` has no scalar to
            optimise and will otherwise fail deep inside the loop.
        """
        # TODO(1): store space, objective, cfg
        # TODO(2): raise ValueError when acquisition and cfg.kpi.mode disagree
        # TODO(3): build the Ax client or BoTorch model per cfg.optim.backend
        # TODO(4): declare the search space as the unit cube, n_dims wide
        raise NotImplementedError("src.optim.bo.run.BayesianOptimizer.__init__")

    def initialize(self) -> None:
        """Evaluate the quasi-random configurations that seed the model.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Include the baseline configuration among the initial points. It is
            the reference every reported improvement is measured against, and
            having it in the model from the start keeps the acquisition function
            from spending iterations rediscovering it.

            Too few initial points and the first GP is fitted on almost nothing,
            so the early candidates are effectively random but consume the
            evaluation budget. A common floor is ``2 * n_dims + 2``.

        Example:
            >>> optimizer.initialize()
        """
        # TODO(1): draw cfg.optim.search.n_init Sobol points in the unit cube
        # TODO(2): prepend space.baseline() mapped through to_unit
        # TODO(3): evaluate each and attach the result to the model
        raise NotImplementedError("src.optim.bo.run.BayesianOptimizer.initialize")

    def step(self) -> np.ndarray:
        """Propose, evaluate and record one candidate configuration.

        Returns:
            The proposed configuration in degrees.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Map out of the unit cube with
            :meth:`src.optim.space.TiltSpace.from_unit` and assert the result is
            in bounds before evaluating. An acquisition optimizer that returns
            a point marginally outside ``[0, 1]`` is normal; evaluating the
            configuration that maps to is not.

        Example:
            >>> theta = optimizer.step()
        """
        # TODO(1): ask the model for the next candidate in unit coordinates
        # TODO(2): from_unit, then assert_within_bounds
        # TODO(3): objective.kpis(theta), attach the result, return theta
        raise NotImplementedError("src.optim.bo.run.BayesianOptimizer.step")

    def run(self) -> dict:
        """Run the full loop and validate the best candidates.

        Returns:
            The best configuration, its Sionna-RT KPIs, the full trial history,
            and the number of Sionna-RT evaluations consumed.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            The returned KPIs must come from Sionna-RT, not the surrogate.
            PROJECT.md section 26 is explicit: the reported performance is the
            high-fidelity result. Return the surrogate prediction alongside it
            so the two can be compared — that comparison is itself a result
            (PROJECT.md section 19 Step 6).

            Return the whole history, not just the winner. The convergence
            trace and the evaluation count are what PROJECT.md sections 25.2 and
            25.3 compare against MARL.

        Example:
            >>> result = optimizer.run()
            >>> result["theta_star"]
        """
        # TODO(1): initialize, then step until cfg.optim.stopping is met
        # TODO(2): take the top validate_top_k candidates by predicted objective
        # TODO(3): re-evaluate each with source="sionna"
        # TODO(4): pick the winner with objective.best under lexicographic priority
        # TODO(5): return theta_star, sionna KPIs, predicted KPIs, history, eval count
        raise NotImplementedError("src.optim.bo.run.BayesianOptimizer.run")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Run Bayesian Optimization as a script, so the DVC stage matches notebook 05a.

    Args:
        cfg: Composed by Hydra from ``configs/``, with ``optim=bo``.

    Returns:
        The final validated objective, for Hydra sweepers.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Run over ``cfg.optim.seeds`` and report mean, standard deviation, best
        and worst (PROJECT.md section 25.3). Persist the trial history as well
        as the winner: the convergence trace and the Sionna-RT evaluation count
        are what the comparison against MARL is built from.

    Example:
        >>> # task bo
    """
    # TODO(1): raise unless cfg.optim is the bo config — guard against a default compose
    # TODO(2): build the space and the surrogate-backed objective
    # TODO(3): run per seed in cfg.optim.seeds
    # TODO(4): persist theta_star, the Sionna-RT KPIs and the history to reports/results/
    raise NotImplementedError("src.optim.bo.run.main")


if __name__ == "__main__":
    main()
