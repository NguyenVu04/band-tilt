"""MORBO over the tilt box, maximising the objectives jointly.

Daulton, Eriksson, Balandat and Bakshy, "Multi-objective Bayesian optimization
over high-dimensional search spaces", UAI 2022, PMLR 180:507-517. Trust regions
are centred on the Pareto points with the largest hypervolume contribution; each
fits its own GPs to the evaluations near it, whoever made them; and a batch is
chosen by Thompson sampling, one point at a time, to maximise the hypervolume
improvement over the observed front and the points already picked. A
collapsed region restarts at the point a random hypervolume scalarisation of a
global GP draw ranks best. Neither Ax nor BoTorch ships it, so this follows the
paper and the authors' reference code on BoTorch models.

Hypervolume is against the origin (:func:`src.optim.objective.hypervolume`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods.base import ATTACHED, INCUMBENT, INIT, SEARCH, SOBOL, sobol
from src.optim.objective import hypervolume, hypervolume_contributions, pareto_mask

# Generation-node names of a trust-region proposal and of a region's restart point.
MORBO = "MORBO"
RESTART = "MORBO restart"

# Rounds a collapsed region's centre is barred from centring any region, so a
# restart cannot fall straight back onto the point it just exhausted.
TABU_TENURE = 100


def failure_tolerance(dim: int) -> int:
    """Failed evaluations in a row that halve a region: ``max(10, ceil(d / 3))`` (Section 5.2)."""
    return max(10, math.ceil(dim / 3))


def local_size(dim: int) -> int:
    """Fewest points a local model is fitted to, ``min(250, 2d)`` (Appendix D.1)."""
    return min(250, 2 * dim)


def _decay(n_evaluated: int, n_init: int, n_total: int, alpha: float) -> float:
    """``1 - alpha * log(n - n0 + 1) / log(N - n0 + 1)``.

    1 when the initial design ends, ``1 - alpha`` at the end of the budget.
    """
    n = max(n_evaluated, n_init)
    n_max = max(n_total, n)
    if n_max == n_init:
        return 1.0
    return 1.0 - alpha * math.log(n - n_init + 1) / math.log(n_max - n_init + 1)


def perturbation_probability(
    dim: int, perturbed_dimensions: float, n_evaluated: int, n_init: int, n_total: int
) -> float:
    """Chance a candidate moves each dimension of its base point (Appendix A).

    ``min(k / d, 1)``, halved by the end of the budget.
    """
    return min(perturbed_dimensions / dim, 1.0) * _decay(n_evaluated, n_init, n_total, 0.5)


def _in_cube(points: np.ndarray, centre: np.ndarray, side: float) -> np.ndarray:
    """Mask of the rows of ``points`` in the cube of side ``side`` around ``centre``."""
    return (np.abs(points - centre) <= side / 2.0).all(axis=1)


@dataclass
class TrustRegion:
    """One region: the history row it is centred on, its side and its failure streak.

    Lengths are in the unit cube the tilt box is rescaled to. A region only
    shrinks: the paper sets the success tolerance to infinity (Appendix D.1),
    because shared data already feeds a region what the others find. Failures
    count evaluations, not rounds, so the batch size does not slow the shrinking.
    """

    centre: int
    length: float
    failures: int = 0

    def box(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Mask of the rows of ``x`` in the cube of side ``scale * length`` around the centre."""
        return _in_cube(x, x[self.centre], scale * self.length)

    def record(self, improved: bool, n_new: int, tolerance: int) -> None:
        """Count ``n_new`` evaluations; ``tolerance`` failed ones in a row halve the length."""
        self.failures = 0 if improved else self.failures + n_new
        if self.failures >= tolerance:
            self.length, self.failures = self.length / 2.0, 0


def _recentre(
    region: TrustRegion,
    others: list[TrustRegion],
    contribution: np.ndarray,
    x: np.ndarray,
    tabu: Iterable[int],
) -> None:
    """Move ``region`` to the largest-contribution Pareto point it may take (Section 3.2).

    Another region's centre and a tabu point are taken. The point is the best
    untaken one inside the region, else the best untaken one anywhere, else the
    best of all. With no point on the front above the origin the centre stays.
    """
    eligible = contribution > 0.0
    taken = [other.centre for other in others if eligible[other.centre]] + list(tabu)
    free = eligible.copy()
    free[taken] = False
    for options in (free & region.box(x), free, eligible):
        if options.any():
            region.centre = int(np.flatnonzero(options)[np.argmax(contribution[options])])
            return


def search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Run MORBO on the objectives for ``n_init + n_iter`` evaluations.

    Args:
        evaluator: Scores a tilt vector.
        cfg: Composed config; reads ``cfg.optim.method.budget``,
            ``cfg.optim.method.trust_region``, ``cfg.optim.method.n_candidates``
            and ``cfg.optim.seed``.

    Returns:
        The history, whose first row is always the committed incumbent. Rows are
        labelled ``init``/``Sobol`` for the initial design, ``init``/``MORBO
        restart`` for a region's restart point, and ``search``/``MORBO`` for
        trust-region proposals.
    """
    space = evaluator.space
    dim = space.n_dim
    method = cfg.optim.method
    n_init = int(method.budget.n_init)
    n_total = n_init + int(method.budget.n_iter)
    batch_size = int(method.budget.batch_size)
    seed = int(cfg.optim.seed)
    settings = method.trust_region
    length_init, length_min = float(settings.length_init), float(settings.length_min)
    tolerance = failure_tolerance(dim)
    n_candidates = int(method.n_candidates)
    # Restart and proposal seeds come from one stream of the run's own, so no
    # draw repeats another seed's initial design.
    rng = np.random.default_rng(seed)

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
    # The restart model sees the initial design and earlier restart points only.
    restart_rows = evaluate(sobol(dim, n_init, seed), INIT, SOBOL)

    # Distinct top-contribution points; the best is shared once they run out.
    contribution = hypervolume_contributions(np.array(values))
    order = np.argsort(-contribution, kind="stable")
    tops = order[: max(1, int((contribution > 0.0).sum()))]
    regions = [
        TrustRegion(centre=int(tops[k] if k < len(tops) else tops[0]), length=length_init)
        for k in range(int(settings.count))
    ]

    tabu: dict[int, int] = {}
    while (remaining := n_total - (len(history) - 1)) > 0:
        n_evaluated = len(history) - 1
        collapsed = next((region for region in regions if region.length < length_min), None)
        if collapsed is not None:
            tabu[collapsed.centre] = TABU_TENURE
            point = _restart_point(
                np.array(unit_x)[restart_rows],
                np.array(values)[restart_rows],
                n_candidates,
                seed=int(rng.integers(2**32)),
            )
            collapsed.centre = evaluate(point, INIT, RESTART)[0]
            restart_rows.append(collapsed.centre)
            decay = _decay(n_evaluated, n_init, n_total, 0.5)
            collapsed.length = length_min + (length_init - length_min) * decay
            collapsed.failures = 0
            continue

        x, y = np.array(unit_x), np.array(values)
        contribution = hypervolume_contributions(y)
        for index, region in enumerate(regions):
            others = regions[:index] + regions[index + 1 :]
            _recentre(region, others, contribution, x, tabu)
        probability = perturbation_probability(
            dim, float(settings.perturbed_dimensions), n_evaluated, n_init, n_total
        )
        batch, owners = _propose(
            x,
            y,
            regions,
            q=min(batch_size, remaining),
            n_candidates=n_candidates,
            probability=probability,
            seed=int(rng.integers(2**32)),
        )
        before = hypervolume(y)
        rows = evaluate(batch, SEARCH, MORBO)
        for index, region in enumerate(regions):
            mine = [values[row] for row, owner in zip(rows, owners, strict=True) if owner == index]
            if mine:
                after = hypervolume(np.vstack([y, *mine]))
                improved = after > before * (1.0 + float(settings.improvement))
                region.record(improved, len(mine), tolerance)
        tabu = {row: left - 1 for row, left in tabu.items() if left > 1}
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


def _fit(train_x, train_y, bounds):
    """Independent GPs, one per objective: Matérn-5/2 with ARD, constant mean (Appendix D.1).

    Inputs are rescaled from ``bounds``, the region's ``2L`` model box, to the
    unit cube, so the lengthscale prior sees a small region's spread rather
    than a sliver of the whole box. The lengthscales carry BoTorch's
    dimension-scaled log-normal prior
    (``get_covar_module_with_dim_scaled_prior``): without one, the marginal
    likelihood over ``d`` lengthscales and a few dozen points is ill-posed, and
    ``fit_gpytorch_mll`` raises ``ModelFittingError``.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.input import Normalize
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
                input_transform=Normalize(d=dim, bounds=bounds),
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
    scrambled Sobol draw over the unit cube, rescaled into the region. A
    candidate the mask left unmoved gets ``ceil(d * probability)`` dimensions.
    """
    import torch

    n, dim = pool.shape
    inside = np.flatnonzero(on_front & region.box(x))
    bases = torch.as_tensor(x[inside if inside.size else [region.centre]], dtype=torch.float64)
    centre = torch.as_tensor(x[region.centre], dtype=torch.float64)
    low = torch.clamp(centre - region.length / 2.0, 0.0, 1.0)
    high = torch.clamp(centre + region.length / 2.0, 0.0, 1.0)
    mask = torch.rand(n, dim, dtype=torch.float64) <= probability
    idle = torch.nonzero(~mask.any(dim=1), as_tuple=True)[0]
    if len(idle):
        dims = torch.rand(len(idle), dim).argsort(dim=1)[:, : math.ceil(dim * probability)]
        mask[idle.unsqueeze(1), dims] = True
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


def _select(front: np.ndarray, samples: np.ndarray) -> tuple[int, int, float]:
    """The sample to pick from one region's draw, as ``(rule, index, score)``.

    The largest hypervolume improvement over ``front``, rule 2. When no sample
    improves on it, a random simplex scalarisation breaks the tie instead, rule
    1, so across regions an improving pick always beats a scalarised one. The
    weights come from torch's global generator.
    """
    import torch
    from botorch.utils.sampling import sample_simplex

    gain = _improvement(front, samples)
    if (gain > 0.0).any():
        index = int(np.argmax(gain))
        return 2, index, float(gain[index])
    weights = sample_simplex(d=samples.shape[1], n=1, dtype=torch.float64)[0].numpy()
    score = samples @ weights
    index = int(np.argmax(score))
    return 1, index, float(score[index])


def _hv_scalarisation(points, weights):
    """Each set's random hypervolume scalarisation against the origin.

    ``max_y min_j (max(y_j, 0) / w_j) ** m`` over the ``[..., k, m]`` points of
    a set, ``m`` objectives and ``w`` a positive unit vector: the scalarisation
    whose expectation over ``w`` is the hypervolume, up to a constant.
    """
    return (points.clamp_min(0.0) / weights).amin(dim=-1).pow(points.shape[-1]).amax(dim=-1)


def _restart_point(x: np.ndarray, y: np.ndarray, n_candidates: int, seed: int) -> np.ndarray:
    """A ``[1, d]`` restart centre: one global draw ranked by a random HV scalarisation.

    The authors' ``TRBOState.gen_new_restart_design``: GPs over the restart data
    ``(x, y)``, one joint posterior draw over ``n_candidates`` scrambled Sobol
    points of the whole unit cube, and the point whose sampled objectives, beside
    the restart data, score the most under :func:`_hv_scalarisation` with
    weights drawn from the positive unit sphere. Where no draw beats the restart
    data, the scores tie and the first Sobol point is taken, as there. Exact
    sampling on CPU float64, under a forked torch RNG.
    """
    import gpytorch
    import torch
    from botorch.utils.sampling import sample_hypersphere
    from torch.quasirandom import SobolEngine

    dim, n_objectives = x.shape[1], y.shape[1]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        tensor_y = torch.as_tensor(y, dtype=torch.float64)
        unit = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(torch.float64)
        model = _fit(torch.as_tensor(x, dtype=torch.float64), tensor_y, unit)
        weights = sample_hypersphere(d=n_objectives, n=1, qmc=True, dtype=torch.float64).abs()
        pool = SobolEngine(dim, scramble=True, seed=seed).draw(n_candidates, dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(float("inf")):
            draw = model.posterior(pool).rsample(torch.Size([1]))[0]
        sets = torch.cat(
            [draw.unsqueeze(-2), tensor_y.expand(n_candidates, *tensor_y.shape)], dim=-2
        )
        best = int(_hv_scalarisation(sets, weights).argmax())
    return pool[best : best + 1].numpy()


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

    Sequential greedy: for each point, every region draws fresh candidates and
    one joint posterior sample over them and the points already picked inside
    it. Those picked points' sampled values join the observed front, and the
    candidate :func:`_select` prefers across regions joins the batch. Exact
    sampling on CPU float64, under a forked torch RNG so the caller's random
    state is untouched.
    """
    import gpytorch
    import torch
    from torch.quasirandom import SobolEngine

    on_front = pareto_mask(y)
    observed = y[on_front]
    n_regions = len(regions)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # The GPU is the ray tracer's.
        tensor_x = torch.as_tensor(x, dtype=torch.float64)
        tensor_y = torch.as_tensor(y, dtype=torch.float64)
        models = []
        for region in regions:
            rows = torch.as_tensor(_local_rows(x, region))
            centre = x[region.centre]
            bounds = np.clip(np.stack([centre - region.length, centre + region.length]), 0.0, 1.0)
            models.append(
                _fit(tensor_x[rows], tensor_y[rows], torch.as_tensor(bounds, dtype=torch.float64))
            )
        pools = SobolEngine(x.shape[1], scramble=True, seed=seed).draw(
            q * n_regions * n_candidates, dtype=torch.float64
        )

        picked = np.empty((0, x.shape[1]))
        owners: list[int] = []
        for step in range(q):
            best = None
            for index, (region, model) in enumerate(zip(regions, models, strict=True)):
                start = (step * n_regions + index) * n_candidates
                pool = pools[start : start + n_candidates]
                candidates = _candidates(x, on_front, region, pool, probability)
                pending = picked[_in_cube(picked, x[region.centre], region.length)]
                points = torch.cat([candidates, torch.as_tensor(pending, dtype=torch.float64)])
                with torch.no_grad(), gpytorch.settings.max_cholesky_size(float("inf")):
                    draw = model.posterior(points).rsample(torch.Size([1]))[0].numpy()
                front = np.vstack([observed, draw[n_candidates:]])
                rule, choice, score = _select(front, draw[:n_candidates])
                if best is None or (rule, score) > best[:2]:
                    best = (rule, score, index, candidates[choice].numpy())
            picked = np.vstack([picked, best[3]])
            owners.append(best[2])
    return picked, owners
