# 1. Four KPIs

- **Status:** Accepted
- **Date:** 2026-08-28
- **Rewritten:** 2026-09-14 — rewritten at the maintainer's direction to describe
  the current system. Earlier revisions (lexicographic selection, Mean Overlap
  Neighbours, Expected RSRP Improvement, Band Priority Score) are in Git history.
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** partly, by
  [ADR 0003](0003-contraharmonic-objective-and-kpi-set.md) — the four definitions below
  stand and are reported, but the objective is not built from them, the served
  ratio is renamed `served_rate`, six more measures join them, and the admission
  cap is a ceiling; see the Amendments. The amendments for deleted records are
  in Git history.
- **Amended:** 2026-09-22 — references repointed when the superseded objective
  records were deleted and the survivors renumbered; see [the README](README.md).

## Context

"Better coverage" has to become a number before anything can be optimised, and
the choice of number *is* the research question — every result the project
produces is a statement about whichever quantity gets picked here.

Radio network optimization has many candidate KPIs: coverage rates at various
thresholds, interference measures, throughput proxies, accessibility,
retainability, handover statistics. Each additional one dilutes the others and
makes method comparisons harder to read.

The objectives conflict. Tilting a cell down shrinks its footprint — that
reduces overlap with neighbours and risks opening holes at the cell edge, and it
changes which UEs a cell can carry within its PRBs. No configuration minimises
everything, so the formulation has to say how they trade.

## Decision

The objective is exactly four KPIs, defined in `src/kpi/` and nowhere else:

| KPI | Definition | Direction |
|---|---|---|
| Hole rate | fraction of grid tiles with `R_max <= kpi.hole_dbm` | minimise |
| Overlap rate | fraction of grid tiles with any co-band neighbour within `kpi.overlap_margin_db` of that band's serving cell | minimise |
| Served ratio | fraction of UEs admitted by the serving rule (`src/kpi/capacity.py`) to a cell-band whose RSRP is above `kpi.hole_dbm` | **maximise** |
| Weak rate | fraction of grid tiles with `hole_dbm < R_max <= kpi.weak_dbm` | minimise |

The column order, **Hole > Overlap > Served > Weak**, is the reporting order
and the order of the default weights. Selection is the weighted score of
[ADR 0002](0002-turbo-on-a-weighted-kpi-score.md), not a lexicographic rule.

## Amendment (2026-09-17)

The four definitions above are unchanged. What changed around them:

- **None of them is the objective.** The search maximises the objective of
  [ADR 0003](0003-contraharmonic-objective-and-kpi-set.md); the KPIs are measured and reported beside it.
- **A fifth KPI is measured**: `edge_rsrp_dbm`, the 5th-percentile serving
  RSRP over covered locations. It is conditional
  on coverage, so it is read beside the hole rate.

**The serving rule.** Each UE takes the most preferred band
(`kpi.capacity.band_preference`) whose layer clears
`kpi.capacity.rsrp_threshold_dbm`, else the strongest layer; a cell-band out of
PRBs (`max_prb`), or already loaded past `kpi.capacity.max_admission_utilisation`
of them, passes the UE to the next candidate. PRBs per UE are
`throughput_per_ue_bps / (12 · SCS · log2(1 + SINR))`, with the solver's
full-load co-band SINR. UEs within an interval are admitted in `t_s` order,
so a cell fills as its reports arrive; ties are broken by strongest RSRP over
every layer at the UE, then by row order. A layer at or below `kpi.hole_dbm`
never serves.

A UE on a hole, and a UE blocked everywhere, both count as not served. The
three rates are shares of the map; the served ratio is a share of the traffic.

**Accessibility is excluded** as a KPI: synthetic MDT carries RSRP and position,
not connection outcomes, and Sionna-RT models propagation, not random access.

## Amendment 2 (2026-09-18)

The four definitions above still stand; what is reported around them changed.

- **The names.** `served_ratio` is `served_rate`; `edge_rsrp_dbm` is
  `rsrp_p05_dbm`, and the percentile behind it is a constant in
  `src/kpi/quality.py` rather than a config value, because the column is named
  after it.
- **Six more measures**: `overlap_neighbor_mean`, `rsrp_p50_dbm`,
  `sinr_p05_db`, `sinr_p50_db`, `prb_utilisation_max` and `load_imbalance`.
  Every one is also reported per frequency layer. `prb_utilisation_max` has
  since left the reported set ([ADR 0003](0003-contraharmonic-objective-and-kpi-set.md)).
- **The admission rule** is a ceiling, not a gate: a cell-band refuses a UE
  whose PRBs would carry it past `kpi.capacity.max_admission_utilisation` of
  `max_prb`, so no cell-band ever ends an interval above that share. The
  paragraph above describing a cell-band "already loaded past" the share
  describes the superseded rule.
- **The objective** is [ADR 0003](0003-contraharmonic-objective-and-kpi-set.md)'s.

## Consequences

**Positive**

- Four columns, one comparison table, one implementation scored by every method.
- The served ratio makes capacity visible: a tilt that covers a hotspot from a
  cell with no PRBs left does not look like a win.

**Negative**

- **Every `kpi.capacity` value is a placeholder.** The served ratio is only as
  good as the per-UE throughput, PRB limits, SCS and band preference behind it.
- The objective mixes two notions of where the map matters: three KPIs weight
  every tile equally, the served ratio weights UE reports. It also overlaps hole
  rate, since a UE on a hole is unserved.
- Nothing band-aware is in the objective. Band preference shapes the serving
  rule and therefore PRB blocking, but no KPI rewards a particular band.
- The capacity model is optimistic and inconsistent: Shannon rate with no MCS
  cap or overhead, full-load interference beside partial PRB load.
- Changing any threshold or capacity value makes earlier results incomparable;
  `src/evaluation/runs.py` refuses to compare such runs.

## Alternatives considered

**Band Priority Score** — the mean normalised weight of each UE's serving band.
Replaced because it rewarded *which* band served, on judgement weights, rather
than *whether* the UE was served.

**Full multi-objective optimization, reporting a Pareto front.** Rejected
because the deliverable is one tilt configuration, and picking from a
four-dimensional front needs a preference anyway; see ADR 0002.

**Constrained optimization** (maximise served ratio subject to a hole-rate cap).
Clean and directly deployable, but the sensible cap is not known before seeing
what the tilt space can achieve.

**Include throughput directly.** Closer to user experience; left out because the
load and scheduler assumptions would dominate the number.
