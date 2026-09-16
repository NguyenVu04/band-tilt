# 5. Separate reported and targeted KPIs, and soften the overlap margin

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the soft-threshold configuration, the hard audit score and the
  overlap desirability of [ADR 0004](0004-soft-threshold-desirability-objective.md)
- **Superseded by:** —

## Context

ADR 0004 left three things unresolved, and they turned out to be one thing.

**The overlap curve softened the wrong comparison.** ADR 0004 softened the `> 0`
of the neighbour *count* and said so plainly: "the overlap margin is still hard
— only the `> 0` indicator was softened, so a neighbour 6.1 dB down still
contributes nothing". The step it removed was not the step that mattered. A
tilt change that pushes an interferer from 5.9 dB down to 6.1 dB down registers
as a whole neighbour vanishing, and one that pushes it from 6.1 to 20 dB
registers as nothing at all. The physics is in the RSRP difference; that is what
the curve has to see. ADR 0004 listed this under Alternatives considered and
deferred it because it changes the meaning of every overlap figure ever
reported. That cost is now accepted.

**The soft thresholds were a second set of numbers for the same physics.**
`kpi.soft` carried a target and a temperature per KPI, and the one honest target
among them — `kpi.soft.hole_rate.target` — was required to equal `kpi.hole_dbm`
so that the two definitions of a hole would agree. Two config keys that must
match are one config key and a rule nothing enforces. The temperatures were
judgement values ADR 0004 already flagged as its main risk, they were never in
`dvc.yaml`'s params or in `runs._kpi_definition`, and so a changed temperature —
which ADR 0004 says invalidates every archived measurement — silently escaped
both the pipeline re-run trigger and the comparability check.

**Two scores meant the reported number and the searched-for number could
disagree.** `selection_scores` dispatched on `optim.objective`, every table
carried `score` and `score_hard`, and `src/evaluation/compare.py` had already
drifted: `seed_summary` and `method_table` read the hard sum while
`winner_vs_candidates` and `convergence` read whatever was configured. A reader
comparing two tables in the same report could be comparing two objectives.

## Decision

**Each threshold is read twice, hard and soft, from one config key.** The
desirability of a KPI softens the same constant its rate steps on, with unit
temperature:

| KPI | Soft form | Config key |
|---|---|---|
| `hole_desirability` | `sigmoid(R_max − hole_dbm)`, per tile, dB | `kpi.hole_dbm` |
| `overlap_desirability` | `sigmoid(Δ − overlap_margin_db)`, per co-band neighbour, dB | `kpi.overlap_margin_db` |
| `served_desirability` | the tile's served share, averaged over occupied tiles | — |

`kpi.soft`, `src/kpi/soft.py`, `soft_spec` and `soften` are deleted. There is no
temperature left to set, so there is none to leave stale.

The served term keeps its per-tile averaging — that was never about a curve, but
about giving a quiet tile the same vote as a hotspot — and drops its sigmoid: the
per-tile share is already in `[0, 1]`, and a unit-temperature sigmoid on it would
be indistinguishable from a straight line.

**`Δ` is the neighbour's RSRP below the serving transmitter**, and the per-band
score is the **mean over the co-band neighbours** of that tile. A tile with no
neighbour, including an uncovered one, scores exactly one — the same convention
`overlap_rate` uses when it counts an uncovered tile as not overlapping.

**Two families of KPI, neither derived from the other.** `src/optim/objective.py`
replaces `WEIGHTED_NAMES`, `OBJECTIVE_NAMES` and `DESIRABILITY_OF` with:

- `REPORT_NAMES` — `hole_rate`, `overlap_rate`, `served_ratio`, `weak_rate`,
  `edge_rsrp_dbm`. What a deployment reads. **No score reads them.**
- `TARGET_NAMES` — the three desirabilities. What the search maximises, and the
  key order `kpi.weights` is read in.

`KPI_NAMES` is still their concatenation, so every table keeps its column order.

**One score.** `quality_index` — the weighted geometric mean of ADR 0004, kept
unchanged — is the only score. `scores`, `selection_scores`, `tolerances`,
`normalised_weights`, `as_maximised` and `optim.objective` are deleted, along
with the `score_hard` column. `kpi.weights` shrinks to the three targeted KPIs
and is keyed by their names, so a reported rate cannot be weighted into anything.

**`kpi.tolerance` is deleted and verdicts read the sign of the delta.** A tie
class needs a measured noise floor per KPI; five of the eight entries were
marked UNMEASURED in the config and were never derived. A verdict that reports
an unmeasured threshold as if it were measured is worse than no verdict, so
`delta_table` and `seed_summary` now report `better`, `worse` or `unchanged`
beside the delta itself and let the reader judge the size. `seed_summary` still
carries the 95 % interval over seeds, which is a *measured* spread.

## Consequences

**Positive**

- One constant per threshold. `kpi.hole_dbm` and `kpi.overlap_margin_db` cannot
  drift from their soft twins, because there are no twins.
- Every threshold the KPIs read is already a `dvc.yaml` param and already in
  `runs._kpi_definition`, so ADR 0004's two silent-escape holes close without
  new machinery.
- The overlap KPI is now differentiable in the quantity a tilt change moves.
  ADR 0004's fourth Negative entry is removed rather than documented.
- A report cannot quote two different objectives, because there is only one.

**Negative**

- **Every archived run is incomparable.** The overlap desirability is a
  different measurement and `score_hard` no longer exists. Worse, the
  fingerprinted thresholds are numerically unchanged — only the formula reading
  them moved — so `runs.verify` will *not* catch it. The nine runs under
  `outputs/optim/` must be re-run.
- **The mean over neighbours is not a count.** A further neighbour raises the
  average, so a tile crowded by one layer can score below a tile crowded by one
  and shadowed by another. The KPI reads typical separation, not crowding;
  crowding is what `overlap_rate` reports. The alternative is below.
- **Unit temperature is itself a judgement**, now implicit in the formula rather
  than explicit in a config key. One dB is the scale the fading depth and the
  margin are both quoted in, so it is a defensible one — but changing it is now
  a code change, which is the point and also the cost.
- **No tie class.** Every nonzero delta reads as better or worse, including
  deltas well inside solver noise. The 95 % interval beside it is what a reader
  must use instead.
- `weak_rate` now carries no weight at all, having carried one only for the
  deleted audit score. It stays measured and reported.

## Alternatives considered

**Keeping `kpi.soft` with the overlap target moved to the RSRP difference.**
The smallest change, and rejected because it keeps the two-keys-that-must-agree
problem for the hole term and keeps the temperatures out of the param list. The
defect was the second set of numbers, not the numbers in it.

**Taking the worst neighbour** (`min` over neighbours, equivalently the
strongest interferer) rather than the mean. This is the exact soft form of
`overlap_rate`, which fires on whether *any* neighbour is inside the margin —
always the strongest one — and it cannot be diluted by a distant layer. Rejected
in favour of the mean at the maintainer's direction: the mean reads the whole
band rather than one offender. If the dilution above shows up in practice, this
is the one-line change, in `src/kpi/overlap.py` `_separation`.

**A product over neighbours** (`prod` of `sigmoid(Δ − margin)`), which is
non-compensatory in the same way the quality index is: every extra neighbour
multiplies in a factor at most one, so more neighbours is always worse and a
distant one costs almost nothing. It has the count sensitivity the mean lacks,
at the cost of a score that falls off fast where several layers overlap. Left
open.

**Keeping a tie class with a measured tolerance.** Legitimate, and it needs the
measurement ADR 0004 asked for and never got: repeated solves of one
configuration under different `simulation.seed` values, per KPI. That work is
not done, so the honest state is no tie class rather than an invented one. A
future ADR can reinstate it once the noise floor is measured.
