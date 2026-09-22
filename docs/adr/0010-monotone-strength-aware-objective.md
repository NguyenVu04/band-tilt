# 10. A monotone, strength-aware objective

- **Status:** Proposed
- **Date:** 2026-09-20
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** [ADR 0009](0009-effective-coverage-objective.md) entirely, and
  the demand map of [ADR 0007](0007-demand-weighted-objective.md) §1, which was
  0007's last surviving piece. What still stands from 0007 is the admission
  ceiling and the reported KPI set, now one measure shorter
- **Superseded by:** —

## Context

> The motivation is the decider's to state. What follows is the mechanical
> account of what changed and what it costs.

ADR 0009's objective was

```
J = sum_g w_g * lambda_bg * exp(1 - lambda_bg) / sum_g w_g
```

with `b(g)` the most preferred band clearing `T_cov` and `w_g = 1 + r_g`. On the
2026-09-20 run it behaved as 0009 predicted: TuRBO won `J` while the rule sweep
won most of the reported KPIs. 0009 recorded that disagreement and left it.
Measured over the 318 archived candidates and the three archived winner maps, it
is worse than a disagreement.

**`J` barely tracks the report card.** Over random search's 145 Sobol candidates —
an unbiased sample of the tilt box — Spearman's rho between `J` and an
equal-weight rank score over the ten KPIs is **0.428**, and between `J` and any
single KPI, 0.21 to 0.38. Maximising `J` gives close to no guarantee on anything
reported.

**The search was not the problem; the selection was.** TuRBO *evaluated* the
best-KPI configuration in the whole study (rank score 0.902, `J` 0.8223) and `J`
told it to pick another (rank score 0.789, `J` 0.8231) — 0.11 of rank score
traded for 0.0008 of `J`.

**It could be raised by destroying coverage.** Because `lambda` was counted on
whichever band the tilts left standing, dropping a crowded preferred layer below
`T_cov` moved a tile onto a cleaner lower band and paid. A tile at −80 dBm on
b2600 with one neighbour scored 0.736; downtilt b2600 below −120 dBm there and it
scored 1.000 on b1800 at −118 dBm — `J` +0.264 for 38 dB less signal. On the
archived winner maps that surface covered **7.0–7.6 % of the grid and was worth
+0.024 to +0.027 of `J`**, against TuRBO's entire +0.0151 gain over the incumbent.
No run took it, so it was latent, not the observed mechanism. ADR 0009 rejected
`alpha_b` because "a weight must not depend on the decision variable", then made
the band *selection* depend on it.

**The demand weighting did nothing.** `r = 0` on **97.0 %** of tiles, mean `w` was
**1.0035**, 0.70 % of tiles exceeded `w = 1.1`, and reported tiles carried 3.3 %
of total weight. `J` was a plain area mean with a 0.35 % perturbation, while three
documents described it as demand-weighted.

**`prb_utilisation_max` was never a KPI.** Over the same 318 candidates it spanned
0.798284 to 0.799999 (std 2.7e-4), 98.1 % within 0.001 of the 0.8 admission
ceiling, and took three distinct values at 3 d.p. It is
`kpi.capacity.max_admission_utilisation` read back.

## Decision

Score **every band, keep the best, and make strength count.**

```
s_bg      = clip((R_b,max(g) - T_cov) / (T_weak - T_cov), 0, 1)
m_bg      = #{ i != argmax on band b : R_bi(g) > T_cov
                                       and R_b,max(g) - R_bi(g) <= Delta_R }
lambda_bg = 1 + m_bg   where band b clears T_cov at g,  else 0
u_bg      = lambda_bg * exp(1 - lambda_bg) * s_bg

J         = mean_g max_b u_bg
```

`T_cov` is `kpi.hole_dbm`, `T_weak` is `kpi.weak_dbm` and `Delta_R` is
`kpi.overlap_margin_db` — the same three cuts `hole_rate`, `weak_rate` and
`overlap_rate` use. **The objective still has no parameters of its own**, and it
no longer reads `kpi.capacity.band_preference` or any demand map.

`m_b` is unchanged and remains **co-band**: it counts transmitters within band `b`
only, never across bands.

`prb_utilisation_max` leaves the reported set, which becomes ten KPIs.
`src.kpi.load.prb_utilisation_max` stays as the check that the admission rule
held, and the per-band table and the per-cell-band figure keep printing it.

### What was decided along the way

**The maximum over bands, not the preferred band.** This is the whole of the
monotonicity fix. Removing a layer sets its `u_b` to 0 and cannot raise a maximum,
so no tilt can pay by going dark. Adding a clean layer never lowers `J`. Adding a
*cell* to a band still can, which is interference and is correct.

**`T_weak` is not a new parameter.** ADR 0009 rejected a strength term as one that
"reintroduces a free parameter", having in mind `u * min(1, (R_1 - T_cov) / 20)`
with an invented 20 dB. `T_weak - T_cov` is the width `weak_rate` already
declares, so the knee is read from config that exists and is already reviewed.

**The demand map is deleted, not repaired.** The alternative was a spatial kernel
so `r` is non-zero off the 3.0 % of reported tiles. Rejected: it reintroduces the
bandwidth parameter ADR 0009 deleted, and it invents traffic where none was
measured. A weight that shifts `J` by 0.0005 does not earn an artifact, a DVC
stage, a grid-shape check and three documents that overstate it.

**`objective_version` in `kpi.yaml`.** ADR 0009 noted that `src/evaluation/runs.py`
caught its change only "by accident of the block disappearing", and that the
utility's form lives in code where the comparability check cannot see it. This
change would not have been caught at all: the config is otherwise unchanged, and
dropping a `KpiVector` field does not help either, because `from_mapping` ignores
keys it does not need. The version is bumped by hand, compared with the rest of
the `kpi` block, and also compared against the running config — checking runs only
against each other passes a set that is uniformly stale.

## Consequences

**Positive**

- **Monotone in coverage.** The 7 %-of-grid exploit is gone by construction, not
  by hoping no search finds it.
- **Six of the ten KPIs are priced**, against 1.5 before: hole rate, weak rate,
  overlap rate, overlap neighbours, and both RSRP percentiles through `s`.
- **Still no free parameters**, and one fewer input than 0009 — no band
  preference, no demand map.
- **Honest naming.** `J` is a tile-uniform measure and is now described as one.

**Negative**

- **The objective and the serving rule no longer agree.** `J` scores the best
  layer; the serving rule admits on `kpi.capacity.band_preference`. A tile can be
  scored on a band no UE would be served on. Band preference now shows up in
  `served_rate` and the serving mix alone. This is the direct cost of dropping
  selection, and dropping selection is the fix.
- **Inter-band interference is still priced nowhere.** `m_b` is co-band, so a
  clean b1800 layer scores full marks with b2600 covering the same tile. The
  reported `overlap_rate` does not measure it either — it sums the three per-band
  counts. Only the solver's SINR sees it, and `J` does not read SINR. Under a
  maximum over bands this is marginally more exposed than under band selection.
  Fixing it needs an interference model, not a reweighting.
- **No demand weighting at all.** ADR 0007's premise that the objective should
  follow the traffic is abandoned rather than approximated. A hole where nobody
  stands now costs exactly what a hole in a hotspot costs. The UE-weighted
  coverage view in `reports/` is the only place demand appears, and nothing
  optimises it.
- **Every objective value recorded before this change is incomparable.** Nothing
  in the record itself says so: `KpiVector.from_mapping` ignores the extra
  `prb_utilisation_max` key and reads an old `run.json` without complaint. The
  `objective_version` check in `src/evaluation/runs.py` is the only thing that
  catches it, and it is checked against the running config as well as run-to-run,
  because a uniformly stale set agrees with itself. Old runs were deleted rather
  than pooled.
- **`u` is still not monotone in `lambda`.** A hole and a four-way overlap score
  alike (0 against 0.199 at full strength). `hole_rate` against `overlap_rate` is
  what separates them in the report.
- **A marginal network is now penalised twice**, once through `s` and once through
  the reported `weak_rate`. That is intended, but it means `J` and `weak_rate` are
  no longer independent readings.

## Measured outcome

The three methods were re-run at seed 42 on the same scenario, same solver settings
and same 145/145/28 budgets, and the study rebuilt.

| | before (ADR 0009) | after (ADR 0010) |
|---|---:|---:|
| Spearman(`J`, equal-weight rank over the 10 KPIs), 145 Sobol candidates | 0.428 | **0.902** |
| same, all 318 candidates | 0.779 | 0.948 |
| exploit surface on the winner maps (share of grid) | 7.0–7.6 % | **0.0000** |
| available `J` gain from that surface | +0.024 to +0.027 | **+0.0000** |
| TuRBO: best KPI rank score evaluated − score picked | 0.11 | **0.037** |
| rule sweep vs TuRBO, head to head on the KPIs | 10–1 | 6–4 |

Per KPI, rho against `J` over the Sobol sample rose on eight of the ten:
`weak_rate` 0.286 → 0.877, `rsrp_p05_dbm` 0.321 → 0.918, `sinr_p05_db` 0.327 →
0.902, `sinr_p50_db` 0.277 → 0.878, `rsrp_p50_dbm` 0.264 → 0.861, `hole_rate`
0.359 → 0.738, `served_rate` 0.205 → 0.565, `load_imbalance` 0.259 → 0.385.

TuRBO now wins `J` by 0.0140 over the rule sweep (0.7650 against 0.7510, from an
incumbent of 0.7333) rather than 0.0042, and it passes the sweep's final answer at
evaluation 25 rather than somewhere past 50. The band-collapsed overlap rate
improves on 700 MHz under TuRBO (0.2396 → 0.2224), where under ADR 0009 both
non-priority layers drifted worse under every method. Cell load imbalance improves
under TuRBO (0.9098 → 0.8818), the first configuration in the study to move it the
right way.

**`s` inverts the hole-versus-crowding exchange rate, and nobody intended that.**
ADR 0009's utility paid the full 1.000 for any covered tile, so closing a hole
bought 1.000 against the 0.264 that splitting a clean tile costs — 3.78 holes to
the crowded tile, strongly hole-seeking. With `s`, a tile that has just crossed
`T_cov` sits near −120 dBm where `s` is near 0. Measured on this run the average
newly covered tile is worth **0.09**, so the rate runs the other way: one newly
crowded strong tile costs about three closed holes.

Decomposing each method's gain against the incumbent shows how little hole-closing
now contributes:

| | ΔJ | from newly covered | from lost coverage | from stronger `s` | from cleaner `lambda` |
|---|---:|---:|---:|---:|---:|
| random | +0.0169 | +0.0003 | −0.0005 | +0.0114 | +0.0058 |
| rule | +0.0177 | +0.0007 | −0.0000 | +0.0169 | +0.0001 |
| turbo | +0.0317 | +0.0005 | −0.0002 | +0.0176 | +0.0138 |

Hole-closing is under 2 % of TuRBO's gain and 4 % of the sweep's. This is why the
sweep beats TuRBO on `hole_rate` (+7.2 % against +3.8 %): it closes 820 tiles to
TuRBO's 559 as a *side effect* of uptilting whole bands uniformly, not because the
objective asked. TuRBO takes +0.0138 from reducing contention on each tile's best
band instead — a move a three-variable sweep cannot make at all, and the whole of
its margin. `J` is now more a signal-strength measure than a coverage one: rho
0.918 against `rsrp_p05_dbm` and 0.877 against `weak_rate`, but 0.738 against
`hole_rate`.

**The reported overlap rate punishes spreading crowding out, which is not
obviously wrong of the searches.** TuRBO has the *least* total co-band crowding of
all four configurations (per-band rates summing to 0.6213 against the incumbent's
0.6472 and the sweep's 0.6449) and the best `overlap_neighbor_mean`. Its
band-collapsed rate is nonetheless the worst, because the three bands' crowded
regions stop coinciding: their redundancy — how much the same ground is crowded on
more than one band — falls from 0.3308 to 0.2613. `overlap_rate` counts tiles
touched by any crowding, so thinning the same crowding over more tiles scores
worse. Whether a tile crowded on one band of three is better or worse than a tile
crowded on all three is a question the KPI cannot express and
`overlap_neighbor_mean` can.

**One measure went backwards, and it is the one this change was always going to
cost.** Rho against the band-collapsed `overlap_rate` fell from 0.347 to **0.251**,
and against `overlap_neighbor_mean` from 0.381 to **0.235** — the two weakest of
the ten. Taking each tile's best band is what removes the exploit, and it is the
same thing that makes a crowded layer free wherever another layer at that tile is
clean. All three methods still raise the band-collapsed overlap rate, most under
TuRBO (0.3164 → 0.3600, against 0.3402 before). Pricing the non-best layers'
crowding without reintroducing a band selection that depends on the tilts is the
open problem this ADR leaves behind, and it is the first thing to revisit.

## Alternatives considered

**Keep band selection and document the exploit.** Rejected: it leaves the largest
gradient in the objective pointing at destroying coverage, and the whole reason
for this ADR is that the previous one documented a defect instead of fixing it.

**Sum or average `u_b` over covered bands** instead of taking the maximum.
Rejected: it scores layers that would never serve the tile — one of the three
faults ADR 0009 was written to fix — and it puts `J`'s bound back at `n_band`.

**A spatial kernel on the demand map**, so the weight does something. Rejected
above.

**Select the winner by a KPI scalarisation and leave `J` as the search signal.**
Rejected: the search would still spend its budget optimising the wrong thing, and
an equal-weight rank score over ten KPIs is a scalarisation with ten implicit
weights, which is worse than the no-parameter objective it would replace.

**Drop `prb_utilisation_max`'s computation too.** Rejected: at three distinct
values it is useless as a KPI, but it is the only assertion that the admission
ceiling held, and it costs one line.
