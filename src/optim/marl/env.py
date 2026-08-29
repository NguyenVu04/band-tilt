"""The TorchRL environment wrapping the tilt problem.

An agent proposes absolute tilts, the environment scores the resulting network,
and the reward comes back. The expensive part is the scoring, so the environment
calls the surrogate rather than Sionna-RT during training, and Sionna-RT
validates the learned policy afterwards.

The action is the tilt, not a change to it
------------------------------------------
Every agent emits absolute tilts in ``[tilt_min, tilt_max]``
(:mod:`src.optim.space`). With an offset parameterisation the feasible action
range would depend on the current configuration, so the action space would move
under the policy during training.

A consequence worth expecting: a single step can already reach any
configuration. Episodes exist to let the agent refine, not to travel, so a long
episode is not obviously better than a short one here.

Bound the action inside the network, not afterwards
---------------------------------------------------
Squash the policy output into the box with ``tanh`` or ``sigmoid``. Clipping an
unbounded output piles probability mass onto the boundary and gives the wrong
gradient there, and it is the usual reason a MARL run parks every tilt at its
maximum and stays.

Credit assignment is the hard part
----------------------------------
Every agent tilt affects every KPI, and the KPIs are global sums over the grid.
A decentralised critic cannot attribute a change in hole rate to one cell among
26. Hence centralised training with decentralised execution: the critic sees the
joint state, each policy sees only its own observation.
"""

from typing import Any

from omegaconf import DictConfig


class TiltEnv:
    """A TorchRL-compatible environment over the absolute-tilt space.

    One transition is: agents emit tilts, the KPIs are evaluated, a reward is
    returned. State is the current configuration plus whatever local context
    ``cfg.optim.observation`` enables.
    """

    def __init__(self, space: Any, objective: Any, cfg: DictConfig) -> None:
        """Build the environment.

        Args:
            space: A :class:`src.optim.space.TiltSpace`.
            objective: An :class:`src.optim.objective.Objective`, normally
                backed by the surrogate.
            cfg: Composed config; uses ``cfg.optim`` (``agents``, ``action``,
                ``observation``, ``env``, ``reward``).

        Raises:
            NotImplementedError: Always — implement this module first.
            ValueError: Once implemented, when the agent granularity does not
                partition the cell-band table exactly.

        Notes:
            The agent partition must cover every cell-band exactly once. A
            cell-band owned by two agents gets its tilt overwritten
            nondeterministically; one owned by none silently keeps its baseline
            tilt and is quietly excluded from the optimization.

            Warn when ``objective`` is backed by Sionna-RT. It will work and it
            will be correct, but training will take weeks — this is the mistake
            the frozen surrogate exists to prevent.
        """
        # TODO(1): partition the cell-band table by cfg.optim.agents.granularity
        # TODO(2): raise ValueError unless the partition is an exact cover
        # TODO(3): build per-agent action specs from the space bounds
        # TODO(4): build observation specs from cfg.optim.observation
        # TODO(5): warn when the objective source is sionna
        raise NotImplementedError("src.optim.marl.env.TiltEnv.__init__")

    def reset(self, tensordict: Any = None) -> Any:
        """Start a new episode from the configured initial tilt.

        Args:
            tensordict: Optional TorchRL input carrying a seed or an explicit
                starting configuration.

        Returns:
            The initial observation for every agent.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            ``cfg.optim.env.reset_strategy`` selects the start: ``baseline``
            always begins at the deployed network, ``random`` draws uniformly in
            the box, ``sampled`` draws from the surrogate dataset.

            ``baseline`` alone teaches the policy only the neighbourhood of one
            configuration, and it will not generalise to a network that has
            drifted. Vary the start if the policy is meant to transfer.

        Example:
            >>> td = env.reset()
        """
        # TODO(1): draw tilt per cfg.optim.env.reset_strategy
        # TODO(2): evaluate the starting KPIs so the first reward has a reference
        # TODO(3): return the per-agent observations as a TensorDict
        raise NotImplementedError("src.optim.marl.env.TiltEnv.reset")

    def step(self, tensordict: Any) -> Any:
        """Apply the joint action, score the network, and return the reward.

        Args:
            tensordict: TorchRL input carrying each agent action.

        Returns:
            The next observation, the reward, and the done flags.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Assemble the joint tilt in cell-band table order. The agent
            partition is a grouping of that table, and reassembling it in agent
            order instead assigns tilts to the wrong cells — which produces a
            complete, plausible, entirely wrong training run.

            Assert the assembled configuration is in bounds before evaluating,
            even though the squashing should guarantee it. This is where a
            squashing bug becomes visible.

        Example:
            >>> td = env.step(td)
        """
        # TODO(1): gather per-agent actions into one tilt in table order
        # TODO(2): assert_within_bounds, catching a squashing bug here
        # TODO(3): objective.kpis(tilt) -> reward.compute(...)
        # TODO(4): build the next observation and the truncation flag
        raise NotImplementedError("src.optim.marl.env.TiltEnv.step")

    def observation(self, tilt: Any, kpis: dict) -> Any:
        """Build the per-agent local observation.

        Args:
            tilt: The current joint configuration.
            kpis: The KPIs for that configuration.

        Returns:
            One observation per agent.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Keep it local. An observation containing the global KPI vector makes
            every agent observation identical, so parameter-shared policies
            cannot tell agents apart and the multi-agent structure is lost. The
            global picture is the critic job.

            Local KPIs mean KPIs restricted to the grid cells this agent serves,
            which is a different computation from the global ones and needs its
            own masking.

        Example:
            >>> obs = env.observation(tilt, kpis)
        """
        # TODO(1): per agent, assemble the components enabled in cfg.optim.observation
        # TODO(2): local_kpi -> recompute over that agent serving area only
        # TODO(3): normalise so the components are on comparable scales
        raise NotImplementedError("src.optim.marl.env.TiltEnv.observation")
