# 1. Four KPIs under lexicographic priority

- **Status:** Accepted
- **Date:** 2026-08-28
- **Revised:** 2026-08-28 — revised in place to follow a change in the
  formulation. The priority order and the Mean Overlap Neighbors denominator
  changed; see *Revision note — 2026-08-28* below.
- **Revised:** 2026-08-29 — revised in place to remove the citations to a
  specification document that is no longer treated as a source of truth. No
  decision changed.
- **Revised:** 2026-09-09 — revised in place at the maintainer's direction.
  Band Priority Score and Expected RSRP Improvement swap priority slots;
  see *Revision note — 2026-09-09* below.
- **Revised:** 2026-09-08 — revised in place, again at the maintainer's
  direction rather than superseded. Expected RSRP Improvement replaces Mean
  Overlap Neighbours in the third priority slot; see *Revision note — 2026-09-08*
  below.
- **Revised:** 2026-09-10 — revised in place at the maintainer's direction.
  Expected RSRP Improvement is removed, leaving four KPIs; see *Revision note —
  2026-09-10* below. The file name keeps "five" so existing links still resolve.
- **Revised:** 2026-09-11 — revised in place at the maintainer's direction.
  Band Priority Score counts each UE's serving band under the capacity model's
  serving rule rather than the strongest band on its tile; see *Revision note —
  2026-09-11* below.
- **Revised:** 2026-09-12 — revised in place at the maintainer's direction. The
  synthetic MDT no longer reports SINR and the radio map no longer stores it;
  PRBs per UE derive from SINR recomputed from the reported RSRP. No KPI
  definition changed; see *Revision note — 2026-09-12* below.
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

The objective is exactly four KPIs:

| KPI | Definition | Direction |
|---|---|---|
| Hole rate | fraction of grid with `R_max <= -120` dBm | minimise |
| Overlap rate | fraction of grid with any co-band neighbour within 6 dB of that band's serving cell | minimise |
| Band Priority Score | mean normalised priority weight `w̃` of the band each covered UE is served by under the serving rule (`src/kpi/capacity.py`); a PRB-blocked UE scores 0 | **maximise** |
| Weak rate | fraction of grid with `-120 < R_max <= -90` dBm | minimise |

They are ordered lexicographically: **Hole > Overlap > Band Priority Score >
Weak**. Coverage holes come first, then how often layers collide, then whether
the right frequency layer is serving the users who reported, and finally the
marginal quality of what is already covered.

One of the four is maximised, and it is the one weighted by the UE reports
rather than uniformly over the grid. The three minimised KPIs are shares of the
map; the maximised one is a share of the traffic.

Each objective carries a **tolerance** in `configs/kpi.yaml`. A difference smaller
than its tolerance is treated as a tie and the comparison moves to the next
objective. Without this the order does not bind.

Where an optimizer cannot express a lexicographic goal, a scalarized fallback is
provided. It operates on **normalised** KPIs, and the weights are checked to
satisfy `lambda_H > lambda_O > lambda_BPS > lambda_W` rather than trusted.

**Accessibility is excluded** — not as a KPI, not as an objective term, not as a
serving-cell or dominant-band criterion.

The definitions live in `src/kpi/` and nowhere else. TuRBO, MARL, the
surrogate-predicted radio maps and the final Sionna-RT validation all score
through the same functions, and no call site re-derives a threshold. The
surrogate predicts a radio map rather than these KPIs, so there is one
evaluator and it sits downstream of both the simulator and the model.

## Consequences

**Positive**

- The objective is small enough to reason about and to plot: four columns, one
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
- **The objective mixes two notions of where the map matters.** Hole, overlap
  and weak rate weight every tile equally; Band Priority Score scores each UE
  report. A configuration can therefore improve slot 3
  while making ground the MDT never sampled worse, and slots 1, 2 and 4 are what
  has to catch that.
- Throughput is not in the objective, so an overlap reduction that costs
  capacity looks like a pure win. SINR and PRB limits enter only indirectly,
  through the serving rule Band Priority Score counts by; SINR is derived from
  RSRP, never measured or stored (see the 2026-09-12 note). Every `kpi.capacity`
  value is a placeholder — slot 3 is only as good as those values; see the
  2026-09-11 note.
- Changing any threshold makes every previously produced result incomparable.

**Neutral**

- The scalarized path exists and is lossy by construction. It is a compatibility
  shim for optimizers that need one number, and results produced with it must say
  so.
- Adding a fifth KPI later is possible but supersedes this record and invalidates
  the existing comparisons.
- No KPI compares a candidate against measured data any more; every one scores
  the candidate map on its own terms, with the MDT supplying only UE positions.

## Alternatives considered

**A single weighted-sum objective.** Simple, works with every optimizer, and
yields a total order. Rejected as the *definition* because it cannot express a
strict priority: for any weights there is a trade that sacrifices hole rate for
enough of the other three, which is exactly the outcome the priority forbids. Kept
as a fallback where an optimizer requires it.

**Full multi-objective optimization, reporting a Pareto front.** Makes the
conflicts explicit and imposes no priority. Rejected as the primary formulation
because the deliverable is one tilt configuration to deploy, and choosing from a
four-dimensional front requires exactly the priority this record states — so the
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


## Revision note — 2026-09-09

Revised in place at the maintainer's direction, as this record has been three
times before. One thing changed: **Band Priority Score and Expected RSRP
Improvement swap the third and fourth priority slots.** The five KPIs, their
definitions, their directions and their thresholds are untouched, as is the
lexicographic mechanism and its tolerances.

The order is now **Hole > Overlap > Band Priority Score > Expected RSRP
Improvement > Weak**.

**What this costs.** Every result produced before this change is incomparable
with every result after it, which is the standing consequence of any reordering
and the reason this is a record rather than a config key. Runs on disk are not
migrated: their `run.json` carries the config that scored them, so which order a
past run used is recoverable, but its winner is not the winner the new order
would pick.

**One consequence worth stating, because it was already a known gap.** This
record's *Consequences* section flags
`kpi.tolerance.expected_rsrp_improvement` as **unmeasured** — a placeholder,
where the other four sit at the ray tracer's run-to-run spread. Until now that
unmeasured tolerance sat in the third slot, deciding ties before any measured
one was consulted. It now sits fourth, behind Band Priority Score, whose
tolerance is measured. That narrows the reach of the gap; it does not close it,
and measuring the tolerance remains on the roadmap.

The reasoning for preferring the frequency layer over the reported improvement
is the maintainer's, and is recorded here as their direction rather than
reconstructed after the fact.

## Revision note — 2026-09-10

Revised in place at the maintainer's direction. **Expected RSRP Improvement is
removed.** The other four KPIs keep their definitions, directions, thresholds
and tolerances; the order is now **Hole > Overlap > Band Priority Score > Weak**.

| | Previously recorded | Now |
|---|---|---|
| KPIs | five | four |
| Maximised KPIs | two (Expected RSRP Improvement, BPS) | one (BPS) |
| Scalarized constraint | `... > lambda_BPS > lambda_EI > lambda_W` | `... > lambda_BPS > lambda_W` |

The same change removes the MDT's synthetic censoring, the per-scenario building
perturbation and the per-scenario material draw from the simulator. Expected
RSRP Improvement was the only KPI that read the reported RSRP values, and the
only one whose tolerance was unmeasured; both gaps close with it.

**What this costs.** As with every revision, results before and after are
incomparable. Runs on disk are not migrated; their `run.json` records the five
KPIs that scored them.

**SINR and PRB demand enter as a diagnostic, not an objective.**
`src/kpi/capacity.py` picks a serving cell-band per UE (band preference above an
RSRP threshold, else the strongest, under per-cell-band PRB limits) and turns
full-load co-band SINR into PRBs per UE. The evaluation's demand map is now PRBs
required per tile. The *Include throughput or SINR* alternative above still
holds for the objective: the model's load and scheduler assumptions are
placeholders, and no optimizer sees the result. (The SINR here was the ray
tracer's at the time; since the 2026-09-12 note it is recomputed from RSRP.)

## Revision note — 2026-09-11

Revised in place at the maintainer's direction. **Band Priority Score changes
definition**; its slot, direction, weights and tolerance value are unchanged.

| | Previously recorded | Now |
|---|---|---|
| Band counted | strongest layer on the UE's tile (`dominant_band`) | the UE's serving band from `src/kpi/capacity.py`: band preference above `kpi.capacity.rsrp_threshold_dbm`, else the strongest, under per-cell-band PRB limits |
| Unit summed | tiles, weighted by UE report count | UE reports, one term each |
| PRB-blocked UE | not modelled | in the denominator at weight 0 |
| UE on a hole | excluded | excluded |

The score is `mean over covered UEs of w̃[band_u]`, with `w̃` the min-max
normalised `kpi.band_priority`. It asks how many UEs the prioritised bands
actually serve, not how many stand where a prioritised band is loudest.

**What this costs.** Band Priority Score values before and after are
incomparable, and so is any winner that slot 3 decided. Runs on disk are not
migrated.

**Two statements above no longer hold.** The 2026-09-10 note's "no optimizer
sees the result" is now false: the serving rule, its SINR and its PRB limits
decide slot 3. The *Include throughput or SINR* alternative is correspondingly
walked back in part — SINR and load now shape the objective, on placeholder
values.

**`kpi.tolerance.band_priority_score` is unmeasured for the new definition.**
It was set at the ray tracer's run-to-run spread of the old score. The value is
carried unchanged and must be re-derived the same way before any result is
reported against it — the gap the 2026-09-08 note recorded for Expected RSRP
Improvement, now in slot 3.

## Revision note — 2026-09-12

Revised in place at the maintainer's direction. **No KPI definition, slot,
direction, threshold or tolerance changes.** What changes is where slot 3's SINR
comes from.

SINR existed twice. `src/simulation/radio.py` stored sionna-rt's
`RadioMap.sinr` in `radio_map.npz`, and `src/simulation/mdt.py` sampled it,
added an independent Gaussian error, and wrote 36 `sinr_*` columns into the MDT.
Separately, `src/kpi/capacity.py` derives SINR from RSRP alone — full-load
co-band interference plus `k·T·B`. Every KPI reader already used the derived
one, because it has to: an optimizer's map and the surrogate's prediction carry
RSRP and nothing else.

| | Previously recorded | Now |
|---|---|---|
| `radio_map.npz` | `rsrp_dbm` and `sinr_db` | `rsrp_dbm` only |
| MDT columns | `rsrp_*` and `sinr_*` per cell-band | `rsrp_*` only |
| Noise knobs | `rsrp_noise_sigma_db`, `sinr_noise_sigma_db` | `rsrp_noise_sigma_db` |
| PRBs per UE at the MDT stage | from the reported SINR | from SINR recomputed off the reported RSRP |
| KPI readers | already recomputed from RSRP | unchanged |

The two agreed to **MAE 7e-06 dB, max 0.0086 dB** over the stored map, recorded
in `reports/surrogate_architecture_2026-09-11/baseline_results.json`. That
measurement is what showed the stored copy to be redundant, and it is also the
last time it can be taken: the comparison is retired with the array it compared
against, and `check_baselines.py` no longer carries it.

**What this costs.** `demand_map.npz` shifts. The measurement error now reaches
PRB demand once, through RSRP, instead of twice through two independent draws,
so PRB demand and any Band Priority Score that PRB blocking decided are not
comparable with earlier runs. Runs on disk are not migrated. The reported
`rsrp_*` values themselves are unchanged for a given seed — the RSRP noise draw
was left exactly as it was.

**What this buys.** One definition of SINR, in `src/kpi/capacity.py`, exercised
by every scorer. The previous arrangement could drift: nothing compared the
reported SINR against the clean SINR, and `sinr_noise_sigma_db` was applied and
then read by only the demand map, so a wrong value there was invisible.

**The *Include throughput or SINR* alternative and the 2026-09-11 walk-back both
still stand.** SINR still shapes the objective through the serving rule, on
placeholder `kpi.capacity` values. Only its source narrowed.
