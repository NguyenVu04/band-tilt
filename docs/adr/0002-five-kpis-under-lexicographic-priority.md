# 2. Five KPIs under lexicographic priority

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

"Better coverage" has to become a number before anything can be optimised, and
the choice of number *is* the research question — every result the project
produces is a statement about whichever quantity gets picked here.

Radio network optimization has many candidate KPIs: coverage rates at various
thresholds, interference measures, throughput proxies, accessibility, retainability,
handover statistics. Including more looks safer and is not: each additional
objective dilutes the others, widens the Pareto front, and makes the BO/MARL
comparison harder to read.

Three further pressures shape the choice.

*The objectives conflict.* Tilting a cell down shrinks its footprint — that
reduces overlap with neighbours and simultaneously risks opening holes at the
cell edge. There is no configuration that minimises everything, so the
formulation has to say what wins.

*They are on incomparable scales.* Hole rate is a percentage in `[0, 100]`. Mean
overlap neighbours is a small count, typically between 1 and 4. Band Priority
Score is on whatever scale the band weights use. A weighted sum over raw values
has its effective priority set by the scales, not by the weights.

*A strict order is unimplementable as literally stated.* Lexicographic
optimization compares the second objective only when the first ties. Hole rate is
continuous, so exact ties essentially never happen, and a literal implementation
is single-objective optimization on hole rate wearing a costume.

There is also a KPI that was available and deliberately left out. Accessibility —
whether a UE can actually connect — is standard in RAN optimization and was
considered for both the serving-cell rule and a dominant-band criterion.

## Decision

The objective is exactly five KPIs:

| KPI | Definition | Direction |
|---|---|---|
| Hole rate | fraction of grid with `R_max <= -120` dBm | minimise |
| Overlap rate | fraction of grid with any neighbour within 6 dB of the serving cell | minimise |
| Weak rate | fraction of grid with `-120 < R_max <= -90` dBm | minimise |
| Mean overlap neighbours | average neighbour count, over overlapping locations only | minimise |
| UE-weighted Band Priority Score | UE-weighted average of dominant-band priority weights | **maximise** |

They are ordered lexicographically: **Hole > Overlap > Weak**, then overlap
severity, then band coordination.

Each objective carries a **tolerance** in `configs/kpi.yaml`. A difference smaller
than its tolerance is treated as a tie and the comparison moves to the next
objective. Without this the order does not bind.

Where an optimizer cannot express a lexicographic goal, a scalarized fallback is
provided. It operates on **normalised** KPIs, and the weights are checked to
satisfy `lambda_H > lambda_O > lambda_W` rather than trusted.

**Accessibility is excluded** — not as a KPI, not as an objective term, not as a
serving-cell or dominant-band criterion.

The definitions live in `src/kpi/` and nowhere else. Bayesian Optimization, MARL,
the surrogate's training labels and the final validation all call the same
functions, and no call site re-derives a threshold.

## Consequences

**Positive**

- The objective is small enough to reason about and to plot: five columns, one
  comparison table.
- The priority is explicit, so a configuration that fills holes at the cost of
  overlap is unambiguously better rather than a matter of taste.
- One implementation means BO and MARL cannot be scoring subtly different things,
  which is what makes the PROJECT.md section 25 comparison valid.
- Excluding accessibility keeps the formulation about radio coverage, where the
  MDT data and the ray-tracing simulation both have something to say.

**Negative**

- The tolerances are consequential and have no principled value. Too tight and
  the formulation collapses to hole-rate minimisation; too loose and the priority
  stops binding. They will be chosen by judgement and must be reported.
- The lexicographic relation with slack is **not transitive**, so it is not a
  valid sort key. Selecting a best candidate needs a single pass, and any
  "ranking" of candidates is order-dependent.
- Excluding accessibility means the optimizer can produce a configuration with
  excellent RSRP coverage that is worse to actually connect to, and nothing in
  the formulation will notice.
- Throughput and interference are not represented, so an overlap reduction that
  costs capacity looks like a pure win.
- Changing any threshold makes every previously produced result incomparable.

**Neutral**

- The scalarized path exists and is lossy by construction. It is a compatibility
  shim for optimizers that need one number, and results produced with it must say
  so.
- Adding a sixth KPI later is possible but supersedes this record and invalidates
  the existing comparisons.

## Alternatives considered

**A single weighted-sum objective.** Simple, works with every optimizer, and
yields a total order. Rejected as the *definition* because it cannot express a
strict priority: for any weights there is a trade that sacrifices hole rate for
enough of the other four, which is exactly the outcome the priority forbids. Kept
as a fallback where an optimizer requires it.

**Full multi-objective optimization, reporting a Pareto front.** Makes the
conflicts explicit and imposes no priority. Rejected as the primary formulation
because the deliverable is one tilt configuration to deploy, and choosing from a
five-dimensional front requires exactly the priority this record states — so the
decision reappears, less visibly. Multi-objective acquisition remains available in
`configs/optim/bo.yaml`, with the priority applied when selecting from the front.

**Constrained optimization: maximise Band Priority Score subject to hole rate
below a cap.** Clean, standard, and directly deployable. Rejected because the caps
are as arbitrary as the tolerances but bind much harder — a configuration one
hundredth of a point over the cap is infeasible rather than slightly worse, and
the sensible cap is not known before seeing what the tilt space can achieve.

**Include accessibility.** Standard in RAN optimization and operationally
meaningful. Rejected because it is not derivable from what this project has: the
MDT export carries RSRP and position, not connection outcomes, and Sionna-RT
models propagation rather than random access. Including it would mean modelling
accessibility from RSRP, which adds an assumption without adding information.

**Include throughput or SINR.** Closer to user experience than coverage rates.
Rejected for scope: it needs a load model and a scheduler assumption, neither of
which the available data supports, and the resulting number would be dominated by
those assumptions rather than by the tilt configuration under study.
