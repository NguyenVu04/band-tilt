"""The MARL training loop — PROJECT.md sections 21 and 24.

Also runs as a script: ``python -m src.optim.marl.train optim=marl`` (or
``task marl``).

Centralised training, decentralised execution
---------------------------------------------
The critic sees the joint state during training; each policy sees only its own
observation at execution. This is what makes a shared, non-decomposable reward
learnable — every agent tilt affects every KPI, so a decentralised critic cannot
attribute a change in hole rate to one cell among 26.

Parameter sharing
-----------------
One policy network conditioned on agent identity, rather than 26 separate
networks. It is what makes this tractable at the current network size, and it
assumes agents are behaviourally interchangeable given their observation — which
is reasonable for cells that differ mainly in position and azimuth, and less
reasonable if some carry bands that others do not.

Compare against BO honestly
---------------------------
PROJECT.md section 25.2 compares the two methods on computational cost, and MARL
carries a training cost that BO does not. Report the training cost as part of
the method cost, not as a fixed setup expense excluded from the comparison — a
policy that transfers across networks amortises it, and one that does not, does
not. Say which case was demonstrated.

Validate the policy, not the training curve
-------------------------------------------
A converged reward curve says the agent learned to maximise the surrogate. The
result that matters is the KPI vector of the configuration it produces, measured
with Sionna-RT (PROJECT.md section 26).
"""

from typing import Any

import hydra
from omegaconf import DictConfig


class MARLTrainer:
    """Train a multi-agent policy over the absolute-tilt space."""

    def __init__(self, env: Any, cfg: DictConfig) -> None:
        """Configure the trainer.

        Args:
            env: A :class:`src.optim.marl.env.TiltEnv`.
            cfg: Composed config; uses ``cfg.optim`` (``algorithm``, ``policy``,
                ``optim``, ``train``).

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Assert the reward weight ordering here, at construction, via
            :func:`src.optim.marl.reward.assert_weight_ordering`. It is a config
            error, and finding it after hours of training is the expensive way.
        """
        # TODO(1): reward.assert_weight_ordering(cfg)
        # TODO(2): build the shared policy and the centralised critic
        # TODO(3): build the TorchRL collector and replay buffer from cfg.optim.train
        raise NotImplementedError("src.optim.marl.train.MARLTrainer.__init__")

    def train(self) -> dict:
        """Run the training loop.

        Returns:
            The trained policy, the reward history and the wall-clock cost.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Log the KPI vector alongside the reward, not just the reward. The
            reward is a weighted scalar and can improve while hole rate — the
            highest-priority objective — gets worse. That is the specific
            failure mode of a scalarized encoding of a lexicographic priority,
            and the KPI trace is the only place it is visible.

        Example:
            >>> history = trainer.train()
        """
        # TODO(1): collect rollouts, update the policy per cfg.optim.algorithm
        # TODO(2): log reward AND the five KPIs per iteration via src.utils.tracking
        # TODO(3): track wall-clock and environment-step counts for section 25.2
        raise NotImplementedError("src.optim.marl.train.MARLTrainer.train")

    def extract_policy_configuration(self) -> Any:
        """Take the tilt configuration the trained policy proposes.

        Returns:
            The absolute tilt configuration in degrees, in cell-band table
            order.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Act greedily — take the policy mean, not a sample. A stochastic
            action makes the reported result irreproducible for no benefit,
            because the deliverable is one configuration to deploy.

        Example:
            >>> theta_star = trainer.extract_policy_configuration()
        """
        # TODO(1): reset the environment at the baseline configuration
        # TODO(2): roll out deterministically, taking the policy mean
        # TODO(3): return the final theta, asserted within bounds
        raise NotImplementedError("src.optim.marl.train.MARLTrainer.extract_policy_configuration")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Train MARL as a script, so the DVC stage matches notebook 05b.

    Args:
        cfg: Composed by Hydra from ``configs/``, with ``optim=marl``.

    Returns:
        The final validated objective, for Hydra sweepers.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Run over ``cfg.optim.seeds`` and report mean, standard deviation, best
        and worst (PROJECT.md section 25.3). A single-seed result for a
        stochastic method is an anecdote, and MARL varies more across seeds than
        BO does.

    Example:
        >>> # task marl
    """
    # TODO(1): raise unless cfg.optim is the marl config — guard against a default compose
    # TODO(2): build the space, surrogate-backed objective and TiltEnv
    # TODO(3): train per seed in cfg.optim.seeds
    # TODO(4): extract the configuration and validate it with source="sionna"
    # TODO(5): persist theta_star, the KPIs and the history to reports/results/
    raise NotImplementedError("src.optim.marl.train.main")


if __name__ == "__main__":
    main()
