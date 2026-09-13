# 3. TuRBO on a weighted KPI score

- **Status:** Accepted
- **Date:** 2026-09-13
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** [0002](0002-bayesian-optimization-without-a-trust-region.md)
- **Superseded by:** —

## Context

[ADR 0002](0002-bayesian-optimization-without-a-trust-region.md) chose plain
multi-objective Bayesian optimization on Ax over all four KPIs, and recorded as
a negative consequence that hypervolume cost grows with the observation count.
On the committed scenario that growth made a run infeasible:

- A 160-trial run with Ax's four-objective default, qLogNEHVI, was stopped after
  about 110 minutes and 30 model rounds. The gap between rounds grew from about
  85 s to about 450 s, while ray tracing a batch of four stayed near 30 s.
- botorch's fused qLogNEHVI kernel could not compile in the pipeline's shell
  (no MSVC `cl` on `PATH`), which made each round slower still.
- A rerun with qLogNParEGO acquisition was stopped at the maintainer's direction
  in favour of the decision below.

The maintainer also asked for KPI weights that drive both the search and the
choice of winner.

## Decision

**The Bayesian arm is TuRBO-1** (Eriksson et al., 2019, *Scalable Global
Optimization via Local Bayesian Optimization*) in `src/optim/methods/turbo/`,
written against BoTorch: one trust region, a GP per round on the evaluations
since the last restart, Thompson sampling within the region, and a restart when
the region collapses. The `mobo` method is removed.

**The objective is one weighted score**, `src/optim/objective.py::scores`: the
raw sum of `kpi.weights` times each KPI, signed so larger is better. Defaults
are hole 4, overlap 3, band priority 2, weak 1, following ADR 0001's priority
order.

**The winner is the highest score** for every method: inside a run
(`History.best_index`), in the rule sweep, and across methods
(`compare.best_method`). A tie keeps the earlier row, so the incumbent holds.
`kpi.tolerance` remains, for reporting deltas as better, worse or a tie only.

**Budgets stay matched.** TuRBO and random search both spend `n_init + n_iter`
evaluations plus the incumbent. Random search keeps its Ax Sobol loop.

## Consequences

**Positive**

- Model cost per round no longer grows with the Pareto front: one GP on one
  output, and a trust region that bounds where candidates are drawn.
- One number decides every comparison, so a method, a run and a notebook cannot
  disagree about which configuration is best.

**Negative**

- The weights are judgement values with nothing to check them against.
- Raw, unnormalised values let a KPI with a wide range dominate: at the incumbent
  overlap rate (about 0.44) and band priority score (about 0.30) move the score
  far more than weak rate (about 0.02) can.
- ADR 0001's lexicographic selection with tolerances is retired; a gain on a
  lower-priority KPI can now outweigh a small loss on a higher one.
- Runs before 2026-09-13 used a different method and selection rule and are not
  comparable with runs after it.

**Neutral**

- Every run still records all four KPIs and publishes its Pareto front and the
  `optim.n_solutions` shortlist, unchanged.

## Alternatives considered

**qLogNParEGO acquisition on Ax.** Kept four objectives and Ax's loop, and
removed the hypervolume growth; set aside because the maintainer chose a
trust-region method on a weighted score.

**MORBO, a multi-objective trust region.** The most code and tuning, and still a
hypervolume computation per region.

**A normalised score** (deltas over tolerances or over the incumbent). Rejected
by the maintainer in favour of the raw weighted sum; relative deltas are also
undefined where the incumbent KPI is zero, as hole rate is here.
