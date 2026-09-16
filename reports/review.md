# Review: Multi-Band Tilt Coordination with TuRBO

| | |
|---|---|
| Scenario | `scn_28d06a5bfb4c02e6`: the bundled Sionna-RT **munich** scene, 3 nodes, 9 cells, 3 bands (700 / 1800 / 2600 MHz), **27 tilt variables** |
| Layout | Equilateral triangle, `node_spacing_m` 500 (UMa ISD, 3GPP TR 38.901 Table 7.2-1), masts 25 m, every tilt starting at 5 deg in [0, 15] |
| Objective | Soft-threshold desirability index `D` (ADR 0004), softened **per tile** inside each KPI; the ADR 0003 weighted sum retained as `score_hard` |
| Runs | **One search seed per method** (`optim.seed` = 42); 145 evaluations for TuRBO and random search, 30 for the rule sweep |
| Evidence | `reports/tables/04_evaluation/`, produced 2026-09-16 |

This review is written as a hostile referee would write it: every claim is attacked first, then a fix is proposed.
Insight interpretation is left to the authors; this document states what the numbers support.

---

## 1. Summary

1. **TuRBO wins on the objective it was given.** `D` rises 0.7017 to 0.7099. That margin is small in absolute terms because the objective was rebuilt on per-tile softening, which is not an amplifier; measured separately, the spread of `D` across the tilt box is about **29 times the solver-seed noise**, so +0.0082 is roughly 8 standard deviations and is real.
2. **The search itself did the work, not luck.** TuRBO's *median* candidate (0.7053) beats both the current configuration (0.7017) and random search's *best* (0.7041). That is the clearest evidence in this report that the model adds value.
3. **The recommendation should still not be deployed, and softening the threshold did not fix that.** TuRBO moves **6.0% of UE demand into coverage holes, from 0.0%** — slightly worse than the 5.7% the previous, aggregate-softened objective produced. The defect is that coverage is weighted by **area**, not by demand; sharpening or softening the threshold is orthogonal to it and cannot help.
4. **TuRBO's lead is a co-band overlap result.** Overlap falls 0.3292 to 0.2969. Re-weighted to hole-only, the three methods are effectively tied (0.8264 / 0.8260 / 0.8260) and the rule sweep ranks first; TuRBO leads only where overlap carries weight.
5. **The rule sweep is the only configuration that worsens nothing.** It improves 2 KPIs and worsens 0, in 30 evaluations and 1.5 minutes, while putting the least demand into holes (2.8%). TuRBO improves 3 and worsens 4.
6. **Solver noise is no longer measured anywhere in the repository.** The re-trace that produced it was removed at the maintainer's direction, so every `tolerance` in `configs/kpi.yaml` — and therefore every better/worse/tie verdict in the scoreboard — is now an unvalidated judgement value.
7. **Nothing here supports a generalisation claim.** One scenario, one search seed, no held-out evaluation, and the title's Multi-Agent Reinforcement Learning arm still does not exist.

---

## 2. Results

### Best configuration per method (seed 42)

| | Current | Random search | Rule-based sweep | TuRBO |
|---|---|---|---|---|
| **`D` (quality index)** | 0.7017 | 0.7041 | 0.7022 | **0.7099** |
| `score_hard` (ADR 0003) | -0.0027 | 0.0107 | 0.0073 | **0.0979** |
| Coverage hole rate (min) | **0.1728** | 0.1755 | 0.1737 | 0.1752 |
| Co-band overlap rate (min) | 0.3292 | 0.3186 | 0.3265 | **0.2969** |
| Served UE ratio (max) | 0.9102 | 0.9115 | 0.9130 | **0.9194** |
| Weak coverage rate (min) | **0.1444** | 0.1549 | 0.1442 | 0.1493 |
| Cell-edge RSRP [dBm] (max) | **-103.99** | -104.70 | -103.93 | -104.76 |
| `hole_desirability` (max) | **0.8260** | 0.8229 | 0.8254 | 0.8235 |
| `overlap_desirability` (max) | 0.5253 | 0.5342 | 0.5264 | **0.5467** |
| `served_desirability` (max) | 0.7816 | 0.7801 | **0.7832** | 0.7805 |

The two scores now agree on the ranking, which they did not under the previous objective: `score_hard` orders TuRBO > random > rule > current, as `D` does.

**Every method still worsens cell-edge RSRP except the rule sweep**, which holds it at -103.93 dBm. `edge_rsrp_dbm` is reported and never optimised, and it keeps earning that role: it is the only KPI that says plainly what the overlap gains cost.

### Where the coverage went

| Coverage class | Current: area / demand | Rule: area / demand | TuRBO: area / demand |
|---|---|---|---|
| hole | 0.1728 / **0.0000** | 0.1737 / 0.0280 | 0.1752 / **0.0604** |
| weak | 0.1444 / 0.7333 | 0.1442 / 0.7054 | 0.1493 / 0.6779 |
| good | 0.6828 / 0.2667 | 0.6821 / 0.2666 | 0.6755 / 0.2617 |

The hole *area* moves by +0.0024. The hole *demand* goes from nothing to 6.0%. This is the most important number in the report and it survived the objective being rebuilt, which is the point: **a soft threshold changes how sharply a tile counts, not which tiles count.** Both `hole_rate` and `hole_desirability` average over the grid, so empty ground and a hotspot weigh the same.

The service figures show what the trade actually was:

| | Current | TuRBO |
|---|---|---|
| Share of UE reports not served | 0.0898 | **0.0806** |
| — of which on a hole tile | 0.0000 | 0.0604 |
| Served SINR, median [dB] | 6.07 | **6.74** |
| PRBs per served UE, median | 47.6 | **44.1** |

TuRBO serves *more* traffic overall and serves it more efficiently. It did so by switching sectors off: congestion fell, and the users who had been queuing were replaced by users with no signal at all. A smaller unserved number, a worse failure mode.

### Robustness checks

| Check | Result |
|---|---|
| Comparability | **21 of 21 pass**: same scenario, grid, solver fidelity, band order and KPI definition across all three runs |
| KPI reproducibility | **32 of 32 pass**, largest absolute gap 2.6e-06 — every recorded KPI recomputes from the archived map |
| Objective responsiveness | no desirability term pinned at a rail; `d_hole` 0.801-0.827, `d_overlap` 0.510-0.530, `d_served` 0.699-0.782 across a 13-point probe |
| Signal against noise | `D` spread across the tilt box is **28.8x** the standard deviation of `D` over five solver seeds on one configuration |

The last row was measured by hand for this review. It is **not** reproduced by any committed code, because the solver-seed re-trace was removed; see critique 6.

### Cost

| Method | Evaluations | Best at | Ray tracing [min] | Wall clock [min] |
|---|---|---|---|---|
| TuRBO | 145 | **141** | 10.3 | 19.0 |
| Random search | 145 | 18 | 8.5 | 12.3 |
| Rule-based sweep | 30 | 3 | 1.2 | 1.5 |

TuRBO's best arrived at evaluation 141 of 145, so **its result is a lower bound and the comparison is budget-dependent**. Its wall clock is nearly double its ray tracing; the gap is GP fitting.

---

## 3. Insights from the data

### The network and the scene

- The munich scene is 1475 x 1206 m; the grid is 74 x 61 = 4514 tiles, **22x smaller than the previous scene's 101 060**. One ray-traced evaluation costs about 4.3 s, which is what made three method runs plus two full probes affordable in one sitting.
- 12.7% of tiles receive no path at all, and the hole rate is 0.173, of which only about 0.020 is reachable by tilt.
- The incumbent puts **73% of demand in "weak" coverage and 0% in holes**: the network is weak-signal-limited, not coverage-limited.
- Setting every tilt to 5 deg put the incumbent essentially on the geometric optimum `atan(23.5 / 292) = 4.6 deg`. It was the best of 13 probe points before any search ran, so the previous review's "straw-man reference" critique is resolved — and inverted. Gains are small because the start is good.

### What the optimizers changed

- **All 27 cell-band tilts moved**, mean absolute change 3.2-4.3 deg per band, largest 9.99 deg. No operator would deploy 27 simultaneous changes of this size.
- Every band's mean signed change is a *downtilt* (+3.42 deg at 2600 MHz, +3.35 at 1800, +1.61 at 700). The previous review's uptilt-everything artefact, caused by a starting tilt pinned at its maximum, is gone.
- TuRBO's win comes from driving a minority of sectors hard down to remove their overlap contribution, which is also what opens the holes. The mechanism is unchanged from the previous objective.
- No band-role differentiation is visible. 700 MHz moves least, which is the right direction for a coverage layer, but the per-band means are within 1.8 deg of each other and no sector shows a coherent band split.

---

## 4. Reviewer's critique

Severity: **Critical** blocks publication; **Major** changes a conclusion; **Minor** weakens it.

| # | Severity | Attack | Evidence | Status |
|---|---|---|---|---|
| 1 | Critical | **The recommendation moves demand into holes.** 0.0% to 6.0% of UE demand lands on hole tiles. Rebuilding the objective on per-tile soft thresholds did not help and slightly worsened it, because both hole KPIs weight area rather than demand. | `coverage_by_area_and_demand.csv` | **Open.** Blocks deployment. Sharpness was the wrong lever |
| 2 | Critical | **No generalisation.** One city, one population; nothing measures transfer. | One scenario on disk | Open |
| 3 | Critical | **No statistics.** One search seed per method; `paired_gain_turbo_vs_random.csv` has one pair, no interval, no p-value. | `paired_gain_turbo_vs_random.csv` | Open, by instruction |
| 4 | Critical | **The claimed contribution does not exist.** The title names Multi-Agent Reinforcement Learning; only TuRBO and two baselines are implemented. | `src/optim/methods/` | Open |
| 5 | Major | **Solver noise is no longer measured.** The re-trace was removed at the maintainer's direction, so nothing in the repository validates any `kpi.tolerance`. Every better/worse/tie verdict rests on unchecked judgement values, and the previous run had already shown three of them to be about 4x too tight. | Removal of `retrace` / `solver_noise`, 2026-09-16 | **New. Open** |
| 6 | Major | **TuRBO's lead depends on the weights.** Under hole-only weighting the three methods tie to four decimals and the rule sweep ranks first; TuRBO leads only where overlap carries weight. | `weight_sensitivity.csv` | Reported |
| 7 | Major | **The served ratio depends on MDT row order.** Five row shuffles of the same map and UEs span 0.001195, larger than its own 0.001 tolerance. `serve_rows` permutes positions derived from the frame's order, so the shuffle randomises admission but does not make the result order-invariant. | Row-order test, 2026-09-16 | Open |
| 8 | Major | **Unconstrained, operationally unrealistic changes.** All 27 tilts move, up to 9.99 deg, with no movement cost or change budget. | `tilt_movement_summary.csv` | Reported |
| 9 | Major | **The served term barely discriminates.** Its desirability spans 0.699-0.782 across the probe and is called a tie at TuRBO's winner. `D` is in practice hole (4/9) plus overlap (3/9). | `kpi_scoreboard.csv` | Open |
| 10 | Minor | **Budget too small for TuRBO.** Its best came at evaluation 141 of 145, so the result is a lower bound. | `method_cost.csv` | Open |
| 11 | Minor | **The rule sweep converged in 3 evaluations** of its 30, so its budget is mostly wasted and its result may be a local artefact of the coordinate order. | `method_cost.csv` | **New. Minor** |
| 12 | Minor | **Physical simplifications.** Full-load co-band SINR, Shannon rate without MCS, a single UE height, no field calibration. Placeholder `kpi.capacity` values persist. | ADR 0001, `configs/kpi.yaml` | Open |
| 13 | Minor | **The overlap margin is still hard.** Only the `> 0` indicator was softened, so a neighbour 6.1 dB down contributes nothing. The 6 dB cliff remains inside the KPI. | ADR 0004 | Reported |

---

## 5. Improvement suggestions

Ordered by impact per unit of effort.

### Fix before any further run

1. **Weight coverage by demand.** Add a demand-weighted hole term to the objective, or restrict the hole KPIs to tiles that carry UE reports. This is critique 1, it is the difference between a deployable recommendation and one that abandons served users, and it is now demonstrated to be independent of how sharp the threshold is. The tables already exist in `src/evaluation/maps.py`.
2. **Restore some measurement of solver noise**, or delete `kpi.tolerance` and stop printing verdicts. Repeated solves of one configuration under different seeds is the cheapest form; at 4.3 s an evaluation it costs under a minute. Reporting better/worse against unvalidated thresholds is worse than reporting deltas alone.
3. **Make the served ratio order-invariant.** Sort by a stable key (`t_index`, `tile_row`, `tile_col`) before the seeded permutation in `serve_rows`. Changes every served-ratio value, so re-run all methods afterwards.

### Fix the objective

4. **Add a tilt-change cost or budget** — a penalty per degree, or a cap on how many cell-bands may move. 27 simultaneous changes is not a deliverable.
5. **Re-target the served term or drop it.** It is called a tie at the winner and spans 0.08 across the whole probe; either lower its target so more tiles sit on the responsive part of the curve, or spend its weight elsewhere.
6. **Reconsider `hole_dbm = -120`.** It sits near the RSRP noise floor; the practical coverage edge is -110 to -105 dBm. Both the hard rate and the soft target read from it, so one change moves both.
7. **Consider the pilot-pollution definition of overlap** (at least three servers within the margin rather than at least one). "Any neighbour within 6 dB" covers a third of the map, which is why overlap dominates the score.
8. **Replace the placeholder `kpi.capacity` values** with operator or 3GPP-based numbers.

### Strengthen the evidence

9. **Run multiple seeds.** Notebook 04 already computes means, intervals and a paired Wilcoxon test; they are empty with one seed.
10. **Raise TuRBO's budget or show a budget curve.** Its best at evaluation 141 of 145 makes the comparison budget-dependent.
11. **Add held-out scenarios.** Generate more with different `simulation.seed`, tune on one and re-evaluate without re-optimizing.
12. **Seed TuRBO from the rule sweep.** The sweep reaches 0.7022 in 3 evaluations; starting the trust region there would test whether the model beats the heuristic from the same footing.

### Clean up the scope

13. **Either implement the MARL arm or remove it from the title, README and architecture diagram.**

---

## 6. Deferred work

| Item | Command or location |
|---|---|
| Multi-seed comparison | one seed was run by instruction; notebooks 03a/03b reuse runs already on disk |
| Solver-noise measurement | removed by instruction; see critique 5 |
| Held-out scenarios | not implemented |
| Demand-weighted coverage KPI | not implemented; see suggestion 1 |
| Insight text in the notebooks | `**Observations.** _To be written._` cells in notebooks 00-04 |
