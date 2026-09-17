# 3. TuRBO on a weighted KPI score

- **Status:** Accepted
- **Date:** 2026-09-13
- **Rewritten:** 2026-09-14 — amendments folded in; the superseded ADR 0002
  (multi-objective BO on Ax) is removed and lives in Git history.
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** partly, by
  [ADR 0006](0006-radio-coverage-objective.md) — TuRBO, the matched budgets
  and highest-score selection stand; the weighted score it optimizes and
  selects by is replaced by the coverage objective, and `kpi.weights`
  and the weight-sensitivity table are deleted. The intermediate ADRs 0004 and
  0005 are in Git history.

## Context

The Bayesian arm was first multi-objective BO on Ax over all four KPIs. On the
committed scenario its hypervolume acquisition (qLogNEHVI) made a 160-trial run
infeasible: model time per round grew several-fold while ray tracing a batch
stayed roughly constant, and botorch's fused kernel could not compile without
MSVC on `PATH`.

The maintainer also asked for KPI weights that drive both the search and the
choice of winner.

## Decision

**The Bayesian arm is TuRBO-1** (Eriksson et al., 2019, *Scalable Global
Optimization via Local Bayesian Optimization*) in `src/optim/methods/turbo/`,
written against BoTorch: one trust region, a GP per round on the evaluations
since the last restart, Thompson sampling within the region, and a restart when
the region collapses.

**The objective is one weighted score**, `src/optim/objective.py::scores`: the
raw sum of `kpi.weights` times each KPI, signed so larger is better. Defaults
are hole 4, overlap 3, served ratio 2, weak 1.

**The winner is the highest score** for every method: inside a run
(`History.best_index`), in the rule sweep, and across methods
(`compare.best_method`). A tie keeps the earlier row, so the incumbent holds.
`kpi.tolerance` is for reporting deltas only.

**No Pareto front is computed.** The published shortlist is the incumbent plus
the highest weighted scores (`optim.n_solutions`).

**Budgets stay matched.** TuRBO and random search both spend `n_init + n_iter`
evaluations plus the incumbent, and random search draws from the same seeded
torch Sobol sequence as TuRBO's initial design, so their first `n_init` points
are identical.

**Every candidate is ray-traced** at the configured fidelity with a fixed solver
seed (common random numbers); nothing is scored by a surrogate.

## Consequences

**Positive**

- Model cost per round is one GP on one output.
- One number decides every comparison, so a method, a run and a notebook cannot
  disagree about which configuration is best.

**Negative**

- The weights are judgement values with nothing to check them against.
- Raw, unnormalised values let a KPI with a wide range dominate the score.
- A gain on a lower-weighted KPI can outweigh a small loss on a higher one.
- Selecting the maximum of many evaluations under one fixed solver seed favours
  candidates that benefit from that seed's Monte-Carlo noise.
- Runs before 2026-09-13 used a different method and selection rule and are not
  comparable with runs after it.

## Alternatives considered

**qLogNParEGO acquisition on Ax.** Kept four objectives and removed the
hypervolume growth; set aside for a trust-region method on a weighted score.

**MORBO, a multi-objective trust region.** The most code and tuning, and still a
hypervolume computation per region.

**A normalised score** (deltas over tolerances or over the incumbent). Rejected
in favour of the raw weighted sum; relative deltas are also undefined where the
incumbent KPI is zero.
