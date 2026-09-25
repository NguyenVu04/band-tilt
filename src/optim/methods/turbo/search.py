"""TuRBO-1 over the tilt box, maximising the coverage objective.

Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian
Optimization* (NeurIPS): one trust region centred on the best point, a GP fitted
to the evaluations since the last restart, and Thompson sampling of a batch from
candidates inside the region. Written against BoTorch because neither Ax nor
BoTorch ships it; candidate construction follows BoTorch's TuRBO-1 tutorial.

The model sees one number per evaluation, the ``objective`` of its KPI vector.
The run still records every KPI, so the published shortlist is built exactly
as for every other method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import ATTACHED, INCUMBENT, INIT, SEARCH, SOBOL, sobol

# Generation-node name of a trust-region proposal.
TURBO = "TuRBO"

# BoTorch TuRBO-1 tutorial: min(5000, max(2000, 200 d)) candidates.
_MIN_CANDIDATES = 2000
_MAX_CANDIDATES = 5000
_CANDIDATES_PER_DIMENSION = 200


@dataclass
class TrustRegion:
    """Side length and success/failure counters of one trust region.

    Lengths are in the unit cube the tilt box is rescaled to.

    Attributes:
        dim: Decision variables.
        batch_size: Evaluations per round; sets :attr:`failure_tolerance`.
        length_init: Side length after a (re)start.
        length_min: Below this the region has collapsed and the search restarts.
        length_max: Cap when expanding.
        success_tolerance: Consecutive successful rounds that double the length.
        improvement: Share of ``|best|`` a round must gain to count as a success.
        perturbed_dimensions: Expected dimensions of the centre a candidate
            perturbs; each moves with probability ``min(perturbed_dimensions / d, 1)``.
        length: Current side length.
        best: Best score since the last restart.
        successes: Consecutive successful rounds.
        failures: Consecutive failed rounds.
    """

    dim: int
    batch_size: int
    length_init: float
    length_min: float
    length_max: float
    success_tolerance: int
    improvement: float
    perturbed_dimensions: float
    length: float = field(init=False)
    best: float = field(init=False)
    successes: int = field(init=False)
    failures: int = field(init=False)

    def __post_init__(self) -> None:
        """Start at the initial length with no history."""
        self.restart()

    @classmethod
    def from_config(cls, cfg: DictConfig, dim: int, batch_size: int) -> TrustRegion:
        """Read ``cfg.optim.method.trust_region``."""
        region = cfg.optim.method.trust_region
        return cls(
            dim=dim,
            batch_size=batch_size,
            length_init=float(region.length_init),
            length_min=float(region.length_min),
            length_max=float(region.length_max),
            success_tolerance=int(region.success_tolerance),
            improvement=float(region.improvement),
            perturbed_dimensions=float(region.perturbed_dimensions),
        )

    @property
    def failure_tolerance(self) -> int:
        """Consecutive failed rounds that halve the length, ``ceil(max(4, d) / q)``."""
        return math.ceil(max(4.0, float(self.dim)) / self.batch_size)

    @property
    def collapsed(self) -> bool:
        """Whether the region has shrunk below ``length_min``."""
        return self.length < self.length_min

    def restart(self) -> None:
        """Reset to ``length_init`` and forget the best score and the counters."""
        self.length = self.length_init
        self.best = -math.inf
        self.successes = 0
        self.failures = 0

    def update(self, batch_best: float) -> None:
        """Record one round's best score, then resize on a run of successes or failures."""
        threshold = (
            self.best + self.improvement * abs(self.best) if math.isfinite(self.best) else -math.inf
        )
        if batch_best > threshold:
            self.successes, self.failures = self.successes + 1, 0
        else:
            self.successes, self.failures = 0, self.failures + 1
        if self.successes == self.success_tolerance:
            self.length, self.successes = min(2.0 * self.length, self.length_max), 0
        elif self.failures == self.failure_tolerance:
            self.length, self.failures = self.length / 2.0, 0
        self.best = max(self.best, batch_best)


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Run TuRBO-1 on the objective for ``n_init + n_iter`` evaluations.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.budget``,
            ``cfg.optim.method.trust_region`` and ``cfg.optim.seed``.

    Returns:
        The history, whose first row is always the committed incumbent. Rows are
        labelled ``init``/``Sobol`` for a (re)start design and
        ``search``/``TuRBO`` for trust-region proposals.
    """
    space = evaluator.space
    budget = cfg.optim.method.budget
    n_init = int(budget.n_init)
    n_total = n_init + int(budget.n_iter)
    batch_size = max(1, int(budget.batch_size))
    seed = int(cfg.optim.seed)
    region = TrustRegion.from_config(cfg, space.n_dim, batch_size)

    lower = space.lower
    # A dimension whose bounds coincide has nowhere to move; any span maps it back.
    span = np.where(space.upper > space.lower, space.upper - space.lower, 1.0)

    history = History(space)
    incumbent = evaluator.evaluate(space.baseline)
    history.append(incumbent, phase=INCUMBENT, generation_node=ATTACHED)
    # The GP's data since the last restart: unit-cube points and their scores.
    unit_x = [(space.baseline - lower) / span]
    score_y = [incumbent.kpi.objective]

    def evaluate(points: np.ndarray, phase: str, node: str) -> list[float]:
        new = []
        for point in points:
            tilts = space.clip(lower + point * span)
            result = evaluator.evaluate(tilts)
            history.append(result, phase=phase, generation_node=node)
            # The clipped point, not the proposal: on a dimension whose bounds
            # coincide the two differ, and a GP told the input moved when the
            # tilt could not would fit the trust region to a fictitious axis.
            unit_x.append((tilts - lower) / span)
            new.append(result.kpi.objective)
            best = history.results[history.best_index()].kpi.objective
            print(
                f"  eval {len(history) - 1}/{n_total} {node:<6} J {new[-1]:.6f}"
                f"  best {best:.6f}  region {region.length:.4g}  restarts {restarts}",
                flush=True,
            )
        score_y.extend(new)
        return new

    restarts = 0
    evaluate(sobol(space.n_dim, min(n_init, n_total), seed), INIT, SOBOL)
    region.best = max(score_y)
    while (remaining := n_total - (len(history) - 1)) > 0:
        if region.collapsed:
            # TuRBO-1 restarts from scratch: a fresh design, and a GP that does
            # not see the collapsed region's data. The global best stays in history.
            restarts += 1
            region.restart()
            unit_x.clear()
            score_y.clear()
            design = sobol(space.n_dim, min(max(n_init, 2), remaining), seed + restarts)
            evaluate(design, INIT, SOBOL)
            region.best = max(score_y)
            continue
        batch = _propose(
            np.array(unit_x),
            np.array(score_y),
            region,
            min(batch_size, remaining),
            seed + len(history),
        )
        region.update(max(evaluate(batch, SEARCH, TURBO)))
    return history


def _propose(
    unit_x: np.ndarray, score_y: np.ndarray, region: TrustRegion, q: int, seed: int
) -> np.ndarray:
    """A batch of ``q`` unit-cube points, Thompson-sampled inside the trust region.

    Fits a GP to the restart's data, centres the region on its best point with
    side lengths scaled by the GP lengthscales (geometric mean one), perturbs a
    random subset of the centre's dimensions, and keeps the ``q`` candidates
    that maximise posterior samples. Seeds a forked torch RNG, so the caller's
    random state is untouched.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation.sampling import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    dim = unit_x.shape[1]
    train_x = torch.as_tensor(unit_x, dtype=torch.float64)
    train_y = torch.as_tensor(score_y, dtype=torch.float64).unsqueeze(-1)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # CPU float64: the GPU is the ray tracer's.
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        kernel = getattr(model.covar_module, "base_kernel", model.covar_module)
        lengthscale = kernel.lengthscale.detach().reshape(-1)
        scale = lengthscale / torch.exp(torch.log(lengthscale).mean())
        centre = train_x[int(train_y.argmax())]
        low = torch.clamp(centre - scale * region.length / 2.0, 0.0, 1.0)
        high = torch.clamp(centre + scale * region.length / 2.0, 0.0, 1.0)

        n_candidates = min(_MAX_CANDIDATES, max(_MIN_CANDIDATES, _CANDIDATES_PER_DIMENSION * dim))
        pool = SobolEngine(dim, scramble=True, seed=seed).draw(n_candidates, dtype=torch.float64)
        pool = low + (high - low) * pool
        probability = min(region.perturbed_dimensions / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=torch.float64) <= probability
        # Every candidate moves in at least one dimension.
        idle = torch.nonzero(mask.sum(dim=1) == 0, as_tuple=True)[0]
        mask[idle, torch.randint(0, dim, (len(idle),))] = True
        candidates = centre.expand(n_candidates, dim).clone()
        candidates[mask] = pool[mask]

        with torch.no_grad():
            chosen = MaxPosteriorSampling(model=model, replacement=False)(candidates, num_samples=q)
    return chosen.numpy()
