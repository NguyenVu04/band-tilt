# 2. Bayesian optimization without a trust region

- **Status:** Accepted
- **Date:** 2026-09-07
- **Revised:** 2026-09-09 — the search now uses a surrogate; see the revision
  note at the end
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

[README.md](../../README.md) names the project's comparison as **TuRBO versus
MARL**, and lists a radio-map surrogate as a prerequisite for the BO arm:
evaluating a tilt configuration means ray-tracing it, and a trust-region method
was chosen on the assumption that evaluations would be too expensive to spend
freely.

Three facts, measured on this repository's committed scenario, qualify that
assumption.

*Ray tracing is affordable here, though not cheap.* Driving
`src.simulation.radio.solve_band` against a scene loaded once, a full evaluation
— three bands, twelve transmitters, the configured `samples_per_tx: 10_000_000`
— takes roughly **thirty to forty seconds**. Loading and perturbing the scene
costs about seventeen seconds once, which is why the evaluator holds it across
candidates rather than rebuilding it per call.

That number is worth stating carefully, because the obvious way to measure it is
wrong. Dr.Jit evaluates lazily, so `RadioMapSolver.__call__` returns before the
map exists and the work lands on the first read of `radio_map.rss`. The timer
inside `radio.solve_band` stops in between and reports one to five seconds — low
by roughly an order of magnitude. `src/optim/evaluator.py` times the whole
materialisation instead, and a budget must be planned against that figure.

*Lowering fidelity buys very little and costs correctness.* Dropping
`samples_per_tx` from 1e7 to 1e6 cuts the wall clock by under a tenth, and a
further decade to 1e5 by only about a third — the cost is dominated by scene
traversal rather than by sample count. Against that, the share of tiles any ray
reaches falls by roughly a quarter at 1e6 and hole rate rises from about 0.11 to
about 0.18, a shift two orders of magnitude larger than the solver's own
run-to-run spread. A cheaper search would not be a cheaper search of this
problem; it would be a search of a different one.

*The model is a comparable cost, and a growing one.* In a 36-dimensional,
five-objective Ax loop, generating a batch of four candidates costs under a
tenth of a second while Sobol is running, about nine seconds at seventeen
observations, and over a minute past thirty. Early on the simulator dominates;
by a hundred observations the model does. Neither is negligible, and the two are
within a small factor of each other over a run of a few hundred evaluations.

A trust region is a device for spending few evaluations well. At tens of seconds
an evaluation, a few hundred of them cost hours — enough to be worth planning,
not enough to make evaluation the scarce resource TuRBO is built around. Were an
evaluation to cost minutes, this decision would deserve revisiting.

## Decision

The Bayesian arm is **plain multi-objective Bayesian optimization on Ax**, over
all four KPIs of [ADR 0001](0001-five-kpis-under-lexicographic-priority.md), with
**no trust region**.

> **Revised 2026-09-09.** As written, this also said "and **no surrogate**. It
> optimizes the ray tracer directly." That is no longer how the arm runs. The
> search scores candidates with a learned surrogate and Sionna-RT re-solves the
> front it proposes; see the revision note. The no-trust-region decision, and
> everything below it, stands.

Consequences of that, each a decision in its own right:

**All four KPIs are optimized jointly.** Ax receives them as a multi-objective
goal, so the run produces a Pareto front rather than one point. No scalarization
is introduced, and ADR 0001's scalarized fallback with its `lambda` weights
remains unbuilt and unneeded.

**The lexicographic rule applies once, at the end.** ADR 0001's priority order,
with the tolerances now in `configs/kpi.yaml`, selects the single configuration
to report out of the front. That is the role ADR 0001 reserves for it — "choosing
from a five-dimensional front requires exactly the priority this record states".

**Every reported KPI is measured at full configured fidelity.** No
`samples_per_tx` override exists. Since the revision, the KPIs an optimizer
*sees* during the search are predictions, but no prediction reaches a report:
`src/optim/report.py` re-solves before anything is published.

**The solver seed is fixed across candidates.** Common random numbers: the
Monte-Carlo noise is shared, so the KPI *differences* the optimizer compares are
less noisy than the KPIs themselves.

**Ax's objective thresholds are anchored on the incumbent**, measured at the
start of every run, so hypervolume is credited only for beating the deployed
tilts.

**Random search and a rule-based sweep are the baselines.** Random search runs
the same loop at the same evaluation budget with Ax's Sobol strategy, which makes
it a control on the model rather than a separate experiment.

## Consequences

**Positive**

- The BO arm exists and runs today, instead of waiting on a surrogate that has
  no code and would need its own acceptance criteria.
- Every reported KPI is ray-traced ground truth, so there is no surrogate error
  between the reported objective and the deliverable. This survives the
  revision: the surrogate moved into the search, not into the report.
- Dropping the trust region removes the tuning surface that TuRBO's behaviour is
  most sensitive to — region length, success and failure tolerances, restart
  policy — none of which this project could have justified from evidence.
- Random search costs the same evaluations and a fraction of the wall clock, so
  "did the model help" is answerable rather than assumed.

**Negative**

- **The project's stated comparison changes.** README frames it as TuRBO versus
  MARL; it is now multi-objective BO versus MARL. Any claim about trust-region
  methods on this problem is out of scope until someone implements one.
- Neither Ax nor BoTorch ships TuRBO, so restoring it later means writing and
  maintaining the trust-region loop directly against BoTorch.
- Hypervolume in five objectives is expensive and grows with the observation
  count. Together with tens of seconds per evaluation, the default budget is a
  run of hours. Past a few hundred evaluations this approach needs revisiting,
  and that is the point at which a trust region starts to earn its keep.
- Optimizing the simulator directly ties a run to one scenario. Nothing here
  shows the winning tilts transfer, and only one scenario is on disk to check
  against.
- ADR 0001 notes mean overlap neighbours is largely redundant with overlap rate.
  Multi-objective optimization pays for that redundancy in hypervolume cost,
  where a scalarization would merely have double-counted it.

**Neutral**

- The surrogate remains worth building, for MARL and for scaling past this
  scenario. This record does not cancel it; it removes it from the BO arm's
  critical path.
- `src/optim/search.py` depends on an `ObjectiveEvaluator` protocol rather than
  on the ray tracer, so a surrogate becomes a second implementation of that
  protocol and no search code changes when it arrives.

## Alternatives considered

**TuRBO with a hypervolume acquisition, written against BoTorch** — a
single-region MORBO. Rejected as a first step: it is the most code, the most
tuning parameters, and its central premise is that evaluations are scarce, which
the measurements above contradict. It remains the natural thing to build if the
evaluation cost rises — a larger scene, more cells, or diffraction enabled would
each do it, and at minutes per evaluation the argument above reverses.

**Keep the surrogate as a prerequisite, as the roadmap has it.** Rejected
because the measurement shows what the surrogate was meant to buy is already
affordable, and building it first would have deferred the BO arm behind a model
whose acceptance criteria are not written.

**Scalarize the four KPIs and run single-objective BO.** Rejected because ADR
0001 rejects a weighted sum as the definition of the objective and admits it only
as a fallback where an optimizer cannot express a multi-objective goal. Ax can,
so the fallback is not needed, and its `lambda` weights would have been chosen
by judgement with nothing to check them against.

**Reduce `samples_per_tx` during the search and re-solve the winner at full
fidelity.** Rejected on measurement: a decade of fidelity buys under a tenth of
the wall clock, while shifting every KPI by orders of magnitude more than the
solver's noise. The search would optimize a different objective than the one
reported, and would barely finish sooner for it.


## Revision note — 2026-09-09

Revised in place at the maintainer's direction, under the exception
[docs/adr/README.md](README.md) records. The Context above is a measurement
record and is unchanged; what changed is the Decision's "no surrogate" clause
and the standing of one rejected alternative.

**What changed.** Optimization is now two phases. `src/optim/run.py` searches
with `src/surrogate/`, scoring a few hundred candidates in minutes and needing
no GPU. `src/optim/report.py` then re-solves that run's Pareto front with
Sionna-RT, re-derives the front from what it measured, and publishes it. A run
that has only been searched carries `verified: false`, and `src/evaluation`
refuses to load one — surrogate predictions cannot reach a comparison as though
they were measurements.

**Why this is not the alternative rejected below.** The last alternative in this
record — reduce `samples_per_tx` during the search and re-solve the winner —
was rejected because "the search would optimize a different objective than the
one reported". That objection was about *fidelity*: the reported number would
itself have been low-fidelity, and a decade of fidelity bought under a tenth of
the wall clock. Neither half applies here.

- Every published KPI comes from `Evaluator` at the configured fidelity. The
  surrogate decides only **which** candidates get measured.
- What it therefore risks is **ranking**, not measurement — the same class of
  risk as a weak acquisition function, and bounded the same way: a solution the
  surrogate ranks out of the shortlist is a solution not found, not a number
  reported wrongly.
- The saving is not a tenth. A 161-candidate search falls from roughly an hour
  and a half of ray tracing to nine solves.

**What is reported changed with it.** This record's Decision says the
lexicographic rule "applies once, at the end", selecting *the* configuration to
report. It still runs, but on measured KPIs and to mark one row `recommended`.
The deliverable is now the **verified front** — `pareto_<method>.csv` and
`tilt_options_<method>.csv` — because five objectives do not have a best, and
with a model proposing the front, collapsing it to one point would trust that
model both to find the front and to rank within it. Choosing among measured
trade-offs is a judgement this project should put to a person.

**What this costs.** A fully ray-traced search is no longer reachable through
`task bo`, so the arm this record originally described cannot be reproduced
without reverting the code. The number of solutions measured per run
(`optim.report.n_solutions`, default 8) is a budget chosen by judgement, not
from evidence about how often the surrogate's ranking is wrong — the run
records its own KPI error on the solutions it offered, and that is the evidence
to revisit this with.
