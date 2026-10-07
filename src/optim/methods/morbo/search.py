"""MORBO over the tilt box, maximising the objectives jointly.

Daulton, Eriksson, Balandat and Bakshy (2022), *Multi-Objective Bayesian
Optimization over High-Dimensional Search Spaces* (UAI), ``docs/MORBO.pdf``.
Trust regions are centred on the Pareto points with the largest hypervolume
contribution; each fits its own GPs to the evaluations near it, whoever made
them; and a batch is chosen by Thompson sampling, one point at a time, to
maximise the hypervolume improvement over the observed front and the points
already picked. Neither Ax nor BoTorch ships it, so this follows the paper's
Algorithm 1 and Appendices A and D on BoTorch models.

Hypervolume is against the origin (:func:`src.optim.objective.hypervolume`).
The run records every KPI beside the objectives, so the published shortlist is
built exactly as for every other method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import ATTACHED, INCUMBENT, INIT, SEARCH, SOBOL, sobol
from src.optim.objective import hypervolume, hypervolume_contributions, pareto_mask

# Generation-node name of a trust-region proposal.
MORBO = "MORBO"

# Purposes a derived seed is drawn for; see _derived_seed.
_RESTART = 1
_PROPOSAL = 2


def _derived_seed(seed: int, purpose: int, index: int) -> int:
    """A uint32 seed for the ``index``-th draw of ``purpose`` in a run seeded ``seed``.

    Mixed by ``numpy.random.SeedSequence`` rather than offset from ``seed``: an
    offset aliases across a seed sweep, so seed 42's first restart would redraw
    seed 43's initial design.
    """
    return int(np.random.SeedSequence([seed, purpose, index]).generate_state(1)[0])


def failure_tolerance(dim: int) -> int:
    """Failed rounds in a row that halve a region: ``max(10, ceil(d / 3))`` (Section 5.2)."""
    return max(10, math.ceil(dim / 3))


def local_size(dim: int) -> int:
    """Fewest points a local model is fitted to, ``min(250, 2d)`` (Appendix D.1)."""
    return min(250, 2 * dim)


def perturbation_probability(
    dim: int, perturbed_dimensions: float, n_evaluated: int, n_init: int, n_total: int
) -> float:
    """Chance a candidate moves each dimension of its base point (Appendix A).

    ``p0 = min(k / d, 1)`` decays as ``p0 (1 - 0.5 log n / log b)``, with
    ``b = n_total - n_init`` and ``n = min(max(n_evaluated - n_init, 1), b)``,
    so it halves by the end of the budget.
    """
    p0 = min(perturbed_dimensions / dim, 1.0)
    budget = n_total - n_init
    if budget < 2:
        return p0
    n = min(max(n_evaluated - n_init, 1), budget)
    return p0 * (1.0 - 0.5 * math.log(n) / math.log(budget))


@dataclass
class TrustRegion:
    """One region: the history row it is centred on, its side and its failure streak.

    Lengths are in the unit cube the tilt box is rescaled to. A region only
    shrinks: the paper sets the success tolerance to infinity (Appendix D.1),
    because shared data already feeds a region what the others find.
    """

    centre: int
    length: float
    failures: int = 0

    def box(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Mask of the rows of ``x`` in the cube of side ``scale * length`` around the centre."""
        return (np.abs(x - x[self.centre]) <= scale * self.length / 2.0).all(axis=1)

    def record(self, improved: bool, tolerance: int) -> None:
        """Count one round; ``tolerance`` failures in a row halve the length."""
        self.failures = 0 if improved else self.failures + 1
        if self.failures >= tolerance:
            self.length, self.failures = self.length / 2.0, 0


def _recentre(regions: list[TrustRegion], x: np.ndarray, y: np.ndarray) -> None:
    """Move each region to the largest-contribution point inside it (Section 3.2).

    Greedy in region order, and a point centres one region only. A region with
    no front point inside it keeps its centre.
    """
    contribution = hypervolume_contributions(y)
    taken: set[int] = set()
    for region in regions:
        options = [int(i) for i in np.flatnonzero(region.box(x)) if int(i) not in taken]
        if options:
            best = max(options, key=lambda i: contribution[i])
            if contribution[best] > 0.0:
                region.centre = best
        taken.add(region.centre)


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Run MORBO on the objectives for ``n_init + n_iter`` evaluations.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.budget``,
            ``cfg.optim.method.trust_region``, ``cfg.optim.method.n_candidates``
            and ``cfg.optim.seed``.

    Returns:
        The history, whose first row is always the committed incumbent. Rows are
        labelled ``init``/``Sobol`` for the initial design and region restarts,
        and ``search``/``MORBO`` for trust-region proposals.
    """
    space = evaluator.space
    dim = space.n_dim
    method = cfg.optim.method
    n_init = int(method.budget.n_init)
    n_total = n_init + int(method.budget.n_iter)
    batch_size = max(1, int(method.budget.batch_size))
    seed = int(cfg.optim.seed)
    settings = method.trust_region
    length_init, length_min = float(settings.length_init), float(settings.length_min)
    tolerance = failure_tolerance(dim)

    history = History(space)
    unit_x: list[np.ndarray] = []
    values: list[np.ndarray] = []

    def record(result, phase: str, node: str) -> int:
        history.append(result, phase=phase, generation_node=node)
        # The snapped and clipped point, not the proposal: a GP told the input
        # moved when the tilt could not would fit to a fictitious axis.
        unit_x.append(space.to_unit(result.tilt_deg))
        values.append(result.kpi.objectives)
        return len(history) - 1

    def evaluate(points: np.ndarray, phase: str, node: str) -> list[int]:
        return [record(evaluator.evaluate(space.from_unit(p)), phase, node) for p in points]

    record(evaluator.evaluate(space.baseline), INCUMBENT, ATTACHED)
    evaluate(sobol(dim, min(n_init, n_total), seed), INIT, SOBOL)

    contribution = hypervolume_contributions(np.array(values))
    order = np.argsort(-contribution, kind="stable")
    regions = [
        TrustRegion(centre=int(order[k % len(order)]), length=length_init)
        for k in range(int(settings.count))
    ]

    restarts = 0
    while (remaining := n_total - (len(history) - 1)) > 0:
        collapsed = next((region for region in regions if region.length < length_min), None)
        if collapsed is not None:
            # ponytail: random restart point (MORBO Fig. 4 ablation, on par) rather than
            # Algorithm 1's HV-scalarised global-GP draw; add that if regions collapse often.
            restarts += 1
            point = sobol(dim, 1, _derived_seed(seed, _RESTART, restarts))
            collapsed.centre = evaluate(point, INIT, SOBOL)[0]
            collapsed.length, collapsed.failures = length_init, 0
            continue

        x, y = np.array(unit_x), np.array(values)
        _recentre(regions, x, y)
        probability = perturbation_probability(
            dim, float(settings.perturbed_dimensions), len(history) - 1, n_init, n_total
        )
        batch, owners = _propose(
            x,
            y,
            regions,
            q=min(batch_size, remaining),
            n_candidates=int(method.n_candidates),
            probability=probability,
            seed=_derived_seed(seed, _PROPOSAL, len(history)),
        )
        before = hypervolume(y)
        rows = evaluate(batch, SEARCH, MORBO)
        for index, region in enumerate(regions):
            mine = [values[row] for row, owner in zip(rows, owners, strict=True) if owner == index]
            if mine:
                after = hypervolume(np.vstack([y, *mine]))
                region.record(after > before * (1.0 + float(settings.improvement)), tolerance)
    return history


def _local_rows(x: np.ndarray, region: TrustRegion) -> np.ndarray:
    """The rows a region's model is fitted to (Section 3.3, Appendix D.1).

    Every point within the cube of side ``2L`` around the centre, topped up
    with the nearest points until there are :func:`local_size` of them.
    """
    inside = np.flatnonzero(region.box(x, scale=2.0))
    needed = min(local_size(x.shape[1]), len(x))
    if len(inside) >= needed:
        return inside
    distance = np.linalg.norm(x - x[region.centre], axis=1)
    return np.argsort(distance, kind="stable")[:needed]


def _fit(train_x, train_y):
    """Independent GPs, one per objective: Matérn-5/2 with ARD, constant mean (Appendix D.1).

    The lengthscales carry BoTorch's dimension-scaled log-normal prior
    (``get_covar_module_with_dim_scaled_prior``): without one, the marginal
    likelihood over 36 lengthscales and a few dozen points is ill-posed, and
    ``fit_gpytorch_mll`` raised ``ModelFittingError`` on a ray-traced run.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
    from gpytorch.mlls import SumMarginalLogLikelihood

    dim = train_x.shape[-1]
    model = ModelListGP(
        *(
            SingleTaskGP(
                train_x,
                train_y[:, [m]],
                covar_module=get_covar_module_with_dim_scaled_prior(dim, use_rbf_kernel=False),
                outcome_transform=Standardize(m=1),
            )
            for m in range(train_y.shape[-1])
        )
    )
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _candidates(x: np.ndarray, on_front: np.ndarray, region: TrustRegion, pool, probability: float):
    """Perturb random front points inside the region, each dimension with ``probability``.

    Appendix A: perturbing a few dimensions of a good point, rather than
    sampling the whole region, keeps a high-dimensional search from wandering
    to the region's boundary. The perturbed values come from ``pool``, a
    scrambled Sobol draw over the unit cube, rescaled into the region.
    """
    import torch

    n, dim = pool.shape
    inside = np.flatnonzero(on_front & region.box(x))
    bases = torch.as_tensor(x[inside if inside.size else [region.centre]], dtype=torch.float64)
    centre = torch.as_tensor(x[region.centre], dtype=torch.float64)
    low = torch.clamp(centre - region.length / 2.0, 0.0, 1.0)
    high = torch.clamp(centre + region.length / 2.0, 0.0, 1.0)
    mask = torch.rand(n, dim, dtype=torch.float64) <= probability
    # Every candidate moves in at least one dimension.
    idle = torch.nonzero(mask.sum(dim=1) == 0, as_tuple=True)[0]
    mask[idle, torch.randint(0, dim, (len(idle),))] = True
    candidates = bases[torch.randint(0, len(bases), (n,))].clone()
    candidates[mask] = (low + (high - low) * pool)[mask]
    return candidates


def _improvement(front: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Hypervolume each sampled objective vector would add to ``front``, alone.

    BoTorch's ``FastNondominatedPartitioning`` splits the region the front does
    not dominate, above the origin, into boxes once; a sample's improvement is
    then the volume it covers in each box, summed, for every sample at once.
    """
    import torch
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    front_t = torch.as_tensor(front, dtype=torch.float64)
    partitioning = FastNondominatedPartitioning(
        ref_point=torch.zeros(front_t.shape[-1], dtype=torch.float64), Y=front_t
    )
    lower, upper = partitioning.get_hypercell_bounds()
    points = torch.as_tensor(samples, dtype=torch.float64).unsqueeze(-2)
    sides = (torch.minimum(points, upper) - lower).clamp_min(0.0)
    return sides.prod(dim=-1).sum(dim=-1).numpy()


def _propose(
    x: np.ndarray,
    y: np.ndarray,
    regions: list[TrustRegion],
    q: int,
    n_candidates: int,
    probability: float,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """A batch of ``q`` unit-cube points, and the region each came from (Section 3.1).

    Sequential greedy: for each point, every region draws a fresh joint
    posterior sample over its candidates, and the candidate whose sample adds
    the most hypervolume to the observed front plus the points already picked
    (at their values in the same draw) joins the batch. Exact sampling on CPU
    float64, under a forked torch RNG so the caller's random state is untouched.
    """
    import gpytorch
    import torch
    from torch.quasirandom import SobolEngine

    on_front = pareto_mask(y)
    observed = y[on_front]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # The GPU is the ray tracer's.
        tensor_x = torch.as_tensor(x, dtype=torch.float64)
        tensor_y = torch.as_tensor(y, dtype=torch.float64)
        pools = SobolEngine(x.shape[1], scramble=True, seed=seed).draw(
            n_candidates * len(regions), dtype=torch.float64
        )
        models, candidates = [], []
        for index, region in enumerate(regions):
            rows = torch.as_tensor(_local_rows(x, region))
            models.append(_fit(tensor_x[rows], tensor_y[rows]))
            pool = pools[index * n_candidates : (index + 1) * n_candidates]
            candidates.append(_candidates(x, on_front, region, pool, probability))

        chosen: list[tuple[int, int]] = []
        for _ in range(q):
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(float("inf")):
                draws = [
                    model.posterior(points).rsample(torch.Size([1]))[0].numpy()
                    for model, points in zip(models, candidates, strict=True)
                ]
            picked = np.array([draws[r][i] for r, i in chosen]).reshape(-1, y.shape[1])
            front = np.vstack([observed, picked])
            best = (-np.inf, 0, 0)
            for region_index, draw in enumerate(draws):
                gain = _improvement(front, draw)
                gain[[i for r, i in chosen if r == region_index]] = -np.inf
                i = int(np.argmax(gain))
                if gain[i] > best[0]:
                    best = (gain[i], region_index, i)
            chosen.append(best[1:])
    batch = np.array([candidates[r][i].numpy() for r, i in chosen])
    return batch, [r for r, _ in chosen]
