# 1. Five KPIs under lexicographic priority

- **Status:** Accepted
- **Date:** 2026-08-28
- **Revised:** 2026-08-28 — revised in place to follow a change in the
  formulation. The priority order and the Mean Overlap Neighbors denominator
  changed; see *Revision note — 2026-08-28* below.
- **Revised:** 2026-08-29 — revised in place to remove the citations to a
  specification document that is no longer treated as a source of truth. No
  decision changed.
- **Revised:** 2026-09-08 — revised in place, again at the maintainer's
  direction rather than superseded. Expected RSRP Improvement replaces Mean
  Overlap Neighbours in the third priority slot; see *Revision note — 2026-09-08*
  below.
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
objective dilutes the others, widens the Pareto front, and makes the TuRBO/MARL
comparison harder to read.

Three further pressures shape the choice.

*The objectives conflict.* Tilting a cell down shrinks its footprint — that
reduces overlap with neighbours and simultaneously risks opening holes at the
cell edge. There is no configuration that minimises everything, so the
formulation has to say what wins.

*They are on incomparable scales.* Hole rate is a percentage in `[0, 100]`. Mean
overlap neighbours is a small count. Band Priority Score is on whatever scale the
band weights use. A weighted sum over raw values has its effective priority set by
the scales, not by the weights.

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
| Overlap rate | fraction of grid with any co-band neighbour within 6 dB of that band's serving cell | minimise |
| Expected RSRP Improvement | `mean_u σ((R_sim(u) − R_real(u)) / τ)` over the MDT reports, `τ = 3` dB | **maximise** |
| UE-weighted Band Priority Score | UE-weighted fraction of UEs served by higher-priority bands | **maximise** |
| Weak rate | fraction of grid with `-120 < R_max <= -90` dBm | minimise |

They are ordered lexicographically: **Hole > Overlap > Expected RSRP Improvement
> Band Priority Score > Weak**. Coverage holes come first, then how often layers
collide, then whether the users who actually reported are better off than they
are today, then whether the right frequency layer is serving them, and finally
the marginal quality of what is already covered.

Two of the five are maximised, and they are the same two that are weighted by
the UE reports rather than uniformly over the grid. The three minimised KPIs are
shares of the map; the two maximised ones are shares of the traffic.

Each objective carries a **tolerance** in `configs/kpi.yaml`. A difference smaller
than its tolerance is treated as a tie and the comparison moves to the next
objective. Without this the order does not bind.

Where an optimizer cannot express a lexicographic goal, a scalarized fallback is
provided. It operates on **normalised** KPIs, and the weights are checked to
satisfy `lambda_H > lambda_O > lambda_EI > lambda_BPS > lambda_W` rather than
trusted.

**Accessibility is excluded** — not as a KPI, not as an objective term, not as a
serving-cell or dominant-band criterion.

The definitions live in `src/kpi/` and nowhere else. TuRBO, MARL, the
surrogate-predicted radio maps and the final Sionna-RT validation all score
through the same functions, and no call site re-derives a threshold. The
surrogate predicts a radio map rather than these KPIs, so there is one
evaluator and it sits downstream of both the simulator and the model.

## Consequences

**Positive**

- The objective is small enough to reason about and to plot: five columns, one
  comparison table.
- The priority is explicit, so a configuration that fills holes at the cost of
  overlap is unambiguously better rather than a matter of taste.
- One implementation means TuRBO and MARL cannot be scoring subtly different
  things, which is what makes the TuRBO-versus-MARL comparison valid.
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
- **The objective now mixes two notions of where the map matters.** Hole,
  overlap and weak rate weight every tile equally; Expected RSRP Improvement and
  Band Priority Score weight tiles by how many UE reports fall on them. A
  configuration can therefore improve on slots 3 and 4 while making ground the
  MDT never sampled worse, and slots 1, 2 and 5 are what has to catch that.
- **`τ` is a free parameter with nothing behind it.** It is set to 3 dB because
  3 dB is the conventional just-noticeable RF step and sits above the 2 dB
  measurement noise of the synthetic MDT — a defensible choice, not a derived
  one. It also bounds what the KPI can see: at `τ = 3` dB, a 10 dB loss and a
  30 dB loss both score near zero, so Expected RSRP Improvement cannot tell bad
  from catastrophic and hole rate has to.
- **`R_real` carries the reporting model's bias.** It is the strongest RSRP each
  UE actually reported, so it inherits the MDT's measurement noise and its
  censoring. The baseline configuration therefore scores slightly under 0.5
  rather than exactly on it. The offset is identical for every candidate and
  cannot change a ranking, but 0.5 must not be read as "no change".
- Throughput and interference are not represented, so an overlap reduction that
  costs capacity looks like a pure win.
- Changing any threshold makes every previously produced result incomparable.

**Neutral**

- The scalarized path exists and is lossy by construction. It is a compatibility
  shim for optimizers that need one number, and results produced with it must say
  so.
- Adding a sixth KPI later is possible but supersedes this record and invalidates
  the existing comparisons.
- Expected RSRP Improvement is the only KPI that compares a candidate against
  measured data rather than scoring it on its own terms. It is therefore the
  first thing to fail if the MDT and the radio map stop describing the same
  scenario.

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
decision reappears, less visibly. A multi-objective acquisition remains available
in `configs/optim/method/mobo.yaml`, with the priority applied when selecting from the
front.

**Constrained optimization: maximise Band Priority Score subject to hole rate
below a cap.** Clean, standard, and directly deployable. Rejected because the caps
are as arbitrary as the tolerances but bind much harder — a configuration one
hundredth of a point over the cap is infeasible rather than slightly worse, and
the sensible cap is not known before seeing what the tilt space can achieve.

**Include accessibility.** Standard in RAN optimization and operationally
meaningful. Rejected because it is not derivable from what this project has:
synthetic MDT carries RSRP and position, not connection outcomes, and Sionna-RT
models propagation rather than random access. Including it would mean modelling
accessibility from RSRP, which adds an assumption without adding information.

**Include throughput or SINR.** Closer to user experience than coverage rates.
Rejected for scope: it needs a load model and a scheduler assumption, neither of
which the available data supports, and the resulting number would be dominated by
those assumptions rather than by the tilt configuration under study.

## Revision note — 2026-08-28

This record was **revised in place** rather than superseded, at the maintainer's
direction. That departs from the rule in [README.md](README.md) that an accepted
record is never rewritten; the original text is recoverable from Git history at
`abcdf6c`. What changed:

| | Originally recorded | Now |
|---|---|---|
| Priority | Hole > Overlap > **Weak** > overlap severity > band coordination | Hole > Overlap > **Mean overlap neighbours > BPS > Weak** |
| Mean overlap neighbours | mean of `N_ov` over **overlapping locations only** | mean of `N_ov` over **all** locations, `(1/\|G\|)·Σ N_ov(g)` |
| Scalarized constraint | `lambda_H > lambda_O > lambda_W` | `lambda_H > lambda_O > lambda_ON > lambda_BPS > lambda_W` |

Both changes follow the 2026-08-28 change in the formulation. The demotion of
weak rate to last is the larger practical change: under the original order a
configuration could not trade weak coverage for band coordination, and now it
can. The denominator change is recorded as a cost under *Consequences →
Negative* above — it is the one part of this revision that removes information
from the objective rather than reordering it.

The Mean Overlap Neighbours denominator recorded here was superseded before it
was ever reimplemented; the KPI itself was removed by the 2026-09-08 revision
below.

## Revision note — 2026-09-08

**Revised in place** rather than superseded, at the maintainer's direction, for
the same reason and with the same caveat as the note above: this departs from
the rule in [README.md](README.md) that an accepted record is never rewritten,
and the previous text is recoverable from Git history.

| | Previously recorded | Now |
|---|---|---|
| Slot 3 | Mean overlap neighbours, minimise | Expected RSRP Improvement, **maximise** |
| Maximised KPIs | one (BPS) | two (Expected RSRP Improvement, BPS) |
| Scalarized constraint | `lambda_ON` | `lambda_EI` |

Mean overlap neighbours is **removed**, not demoted. The *Consequences →
Negative* section of the 2026-08-28 revision already recorded it as "partly
redundant with overlap rate" once its denominator became all `|G|`: any
configuration lowering overlap rate lowered it too, almost mechanically, so the
third slot was carrying a rescaling of the second. Expected RSRP Improvement
puts something independent there — it is the only KPI that reads what the UEs
measured rather than scoring the candidate map on its own terms.

The sigmoid is not decoration. `mean(1[ΔR > 0])` — the fraction of reports
improved — is the quantity of interest and needs no `τ`, but it is a step
function and gives a GP surrogate nothing to follow. `σ(ΔR/τ)` is its smooth
relaxation, which is what makes the KPI usable as a Bayesian-optimization
objective.

`kpi.tolerance.expected_rsrp_improvement` is **unmeasured**. Every other
tolerance sits at the ray tracer's run-to-run spread under a changed solver
seed; this one is a placeholder carried in `configs/kpi.yaml` with that stated,
and it must be derived the same way before any result is reported against it.
