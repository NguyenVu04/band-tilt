# Review: Multi-Band Tilt Coordination with TuRBO

| | |
|---|---|
| Scenario | `scn_61e174bef723f9c4`: the delivered scene `data/external/scene/scene.xml` (about 6.2 x 6.5 km), 3 nodes, 9 cells, 3 bands (700 / 1800 / 2600 MHz), **27 tilt variables** |
| Layout | Equilateral triangle, `node_spacing_m` 1732 (RMa ISD, 3GPP TR 38.901 Table 7.2-1), masts 20 m; starting tilts **700 MHz 12 deg, 1800 MHz 9 deg, 2600 MHz 6 deg**, all in [0, 15] |
| Objective | Soft-threshold desirability index `D` (ADR 0004), weights hole 3 / overlap 2 / served 1 (working-tree `configs/kpi.yaml`) |
| Runs | **One search seed per method** (`optim.seed` = 42); 145 evaluations for TuRBO and random search, 29 for the rule sweep |
| Evidence | `reports/tables/04_evaluation/`, produced 2026-09-16 by `task pipeline` |

This review is written as a hostile referee would write it: every claim is attacked first, then a fix is proposed.
Insight interpretation is left to the authors; this document states what the numbers support.

---

## 1. Summary

1. **TuRBO wins on the objective it was given.** `D` rises 0.7930 to 0.8151 (+0.0221). The rule sweep reaches 0.8124 and random search 0.8086. TuRBO's lead over the rule sweep is 0.0027, and nothing in the repository measures solver noise, so whether that lead is real is **not established**.
2. **The search did the work, not luck.** TuRBO's *median* candidate (0.8133) beats the *best* of both the rule sweep (0.8124) and random search (0.8086).
3. **Most of the gain is undoing the starting tilts.** At a 1000 m cell radius and 18.5 m antenna-over-UE height, the geometric optimum is `atan(18.5 / 1000) = 1.1 deg`; the incumbent starts at 6-12 deg. Every method uptilts, and the rule sweep captures **88% of TuRBO's gain in 29 evaluations**.
4. **The hole-demand defect did not reproduce here, but it is not fixed.** TuRBO and the rule sweep put 0.0% of demand into holes, random search 0.26%. Both hole KPIs still weight area rather than demand; this scenario simply did not trigger the failure.
5. **The network is service-limited, and no method comes close to fixing that.** Even at TuRBO's optimum, **45.2% of UE reports are not served**, and 83.5% of demand sits on weak-coverage tiles. The capacity model behind these figures is still placeholder values.
6. **Every method worsens overlap.** All three improve 6 KPIs and worsen 2, and the 2 are always the co-band overlap rate and its desirability. The better/worse verdicts are now plain signs of the delta with no tolerance.
7. **Most of the grid is territory no site serves.** The grid spans the full 6.2 x 6.5 km scene; the nodes sit within about 1 km of its centre. **61% of tiles are more than 1.5 km from every node**, so the area-weighted KPIs are dominated by ground the tilts barely reach.
8. **Nothing here supports a generalisation claim.** One scenario, one search seed, no held-out evaluation, and the title's Multi-Agent Reinforcement Learning arm still does not exist.

---

## 2. Results

### Best configuration per method (seed 42)

| | Current | Random search | Rule-based sweep | TuRBO |
|---|---|---|---|---|
| **`D` (quality index)** | 0.7930 | 0.8086 | 0.8124 | **0.8151** |
| Coverage hole rate (min) | 0.2106 | 0.1979 | **0.1934** | 0.1948 |
| Co-band overlap rate (min) | **0.2329** | 0.2388 | 0.2454 | 0.2380 |
| Served UE ratio (max) | 0.4761 | 0.5184 | 0.5468 | **0.5477** |
| Weak coverage rate (min) | 0.3959 | 0.2842 | **0.2569** | 0.2616 |
| Cell-edge RSRP [dBm] (max) | -112.65 | -110.64 | **-110.21** | -110.24 |
| Coverage desirability (max) | 0.7890 | 0.8019 | **0.8063** | 0.8049 |
| Layer separation desirability (max) | **0.9054** | 0.9024 | 0.8971 | 0.9043 |
| Served desirability, per tile (max) | 0.6174 | 0.6656 | 0.6815 | **0.6878** |

Unlike the munich run, **cell-edge RSRP improves under every method** (+2.0 to +2.4 dB). That is what uptilting from a steep start should do, and it agrees with the reading that the starting tilts, not the optimizer, set the size of the gain.

TuRBO and the rule sweep differ by at most 0.0074 on any rate KPI, and by 0.03 dB on edge RSRP. TuRBO gets its `D` lead by doing less damage to overlap; the rule sweep leaves fewer holes and less weak coverage.

### Where the coverage went

| Coverage class | Current: area / demand | Rule: area / demand | TuRBO: area / demand |
|---|---|---|---|
| hole | 0.2106 / **0.0000** | 0.1934 / 0.0000 | 0.1948 / **0.0000** |
| weak | 0.3959 / 0.9430 | 0.2569 / 0.8213 | 0.2616 / 0.8354 |
| good | 0.3935 / 0.0570 | 0.5496 / 0.1787 | 0.5435 / 0.1646 |

Hole demand stays at zero for both leading methods, and good-coverage demand roughly triples. Random search is the only one that moves demand into holes (0.0026). Why the area-weighted hole KPI did no harm this time has not been established. The hotspot centres sit roughly 700 m from the nearest node, not close to one. Either way, a clean result on one population does not make the KPI safe.

The service figures:

| | Current | Rule | TuRBO |
|---|---|---|---|
| Share of UE reports not served | 0.5239 | 0.4532 | **0.4523** |
| Served SINR, 10th percentile [dB] | -2.14 | **-1.65** | -1.75 |
| Served SINR, median [dB] | 4.63 | **5.39** | 5.31 |
| PRBs per served UE, median | 56.6 | **51.5** | 52.0 |
| Served on 2600 / 1800 / 700 MHz | 0.371 / 0.070 / 0.035 | 0.347 / 0.129 / 0.071 | 0.372 / 0.112 / 0.063 |

The optimizers roughly double the traffic carried on 1800 and 700 MHz. Offloading from 2600 MHz is where the served-ratio gain comes from.

### Robustness checks

| Check | Result |
|---|---|
| Comparability | **21 of 21 pass**: same scenario, grid, solver fidelity, band order and KPI definition across all three runs |
| KPI reproducibility | **32 of 32 pass**, largest absolute gap 2.9e-06 |
| Signal against noise | **Not measured.** The 28.8x ratio in the previous review was hand-measured on munich and does not carry over to this scene |

### Cost

| Method | Evaluations | Best at | Ray tracing [min] | Wall clock [min] |
|---|---|---|---|---|
| TuRBO | 145 | **141** | 8.2 | 16.9 |
| Random search | 145 | 107 | 7.4 | 11.5 |
| Rule-based sweep | 29 | 18 | 0.9 | 1.9 |

Despite a grid 22x the tiles of munich, one ray-traced evaluation costs about 3.4 s, less than munich's 4.3 s. TuRBO's best again arrived at evaluation 141 of 145, so **its result is a lower bound**.

---

## 3. Insights from the data

### The network and the scene

- The grid is 326 x 310 = 101,060 tiles at 20 m, of which 97,910 are open ground eligible for UEs. The MDT set has 9,708 reports after 358 no-signal UEs were dropped.
- The inter-node distances are 1732, 1745 and 1767 m; the layout generator moved node n0 to the nearest open ground, which stretched two sides.
- **12.2% of tiles receive no path at all** at the incumbent. That share is 19.4% among tiles more than 1.5 km from every node and 0.3% within 1 km. Tilt cannot reach this part of the hole rate.
- The incumbent puts **94.3% of demand on weak-coverage tiles** and 0% in holes: at 1732 m spacing with 20 m masts the network is weak-signal-limited.
- The starting tilts point the 700 MHz main lobe at about `18.5 / tan(12 deg) = 87 m` from the mast and the 2600 MHz lobe at about 176 m, against a 1000 m cell radius. This is the reverse of the `default_tilts` docstring convention (higher bands tilted harder), and it makes the incumbent a weak reference.

### What the optimizers changed

- **All 27 cell-band tilts moved.** Mean absolute change is 8.9 deg at 700 MHz, 6.8 deg at 1800 MHz and 3.4 deg at 2600 MHz, and the largest is 11.7 deg.
- The mean signed change is an **uptilt** in every band (-8.9, -6.8, -1.3 deg). Only three 2600 MHz cells were downtilted (n2c0 +4.1, n2c1 +3.6, n0c2 +2.1).
- TuRBO's proposed means are 3.1 deg at 700 MHz, 2.2 deg at 1800 MHz and 4.7 deg at 2600 MHz. The optimizer has put 2600 MHz on the steepest tilt, which is the conventional capacity-layer role, and left 700 and 1800 MHz close to the geometric optimum.
- The largest per-cell effects: n1c1 700 MHz, uptilted 12.0 to 0.6 deg, gains 107 served reports; n2c1 2600 MHz, downtilted 6.0 to 9.6 deg, loses 205. The single largest loss comes from one of the three downtilts.

---

## 4. Reviewer's critique

Severity: **Critical** blocks publication; **Major** changes a conclusion; **Minor** weakens it.

| # | Severity | Attack | Evidence | Status |
|---|---|---|---|---|
| 1 | Critical | **The hole KPIs still weight area, not demand.** The munich run moved 6.0% of demand into holes; this scenario put 0.0% there for TuRBO and the rule sweep, and 0.26% for random search. The mechanism is unchanged, and why this run escaped it has not been established. | `coverage_by_area_and_demand.csv` | **Open.** Not reproduced in this scenario |
| 2 | Critical | **No generalisation.** One city, one population; nothing measures transfer. | One scenario on disk | Open |
| 3 | Critical | **No statistics.** One search seed per method; `paired_gain_turbo_vs_random.csv` has one pair, no interval, no p-value. | `paired_gain_turbo_vs_random.csv` | Open, by instruction |
| 4 | Critical | **The claimed contribution does not exist.** The title names Multi-Agent Reinforcement Learning; only TuRBO and two baselines are implemented. | `src/optim/methods/` | Open |
| 5 | Major | **The incumbent is a straw man again.** Starting tilts of 6-12 deg against a 1.1 deg geometric optimum make every method's gain mostly an uptilt back towards geometry. The rule sweep takes 88% of TuRBO's gain in 29 evaluations. | `tilt_movement_summary.csv`, `method_cost.csv` | **New. Open** |
| 6 | Major | **The grid is mostly out of range.** 61% of tiles are more than 1.5 km from every node, and the no-path share there is 19.4% against 0.3% within 1 km. Area-weighted KPIs average over ground the three sites were never meant to cover. | `data/interim/radio_map.npz` | **New. Open** |
| 7 | Major | **Solver noise is not measured, and verdicts now have no tolerance at all.** Better/worse is the sign of the delta, so a 1e-6 change counts as a verdict. TuRBO's 0.0027 lead over the rule sweep cannot be checked against noise. | `src/evaluation/compare.py` | Open, worsened |
| 8 | Major | **More than 45% of UE reports are unserved under every method.** Served ratio peaks at 0.548. The capacity model behind it is marked PLACEHOLDER, so the served term, and one third of `D`'s weight, rest on unvalidated numbers. | `ue_service_summary.csv`, `configs/kpi.yaml` | **New. Open** |
| 9 | Major | **TuRBO's lead depends on the weights.** Under hole-only weighting the rule sweep ranks first (0.8067 vs 0.8054); under overlap-only the two tie to four decimals. TuRBO leads under the configured, equal and served-only weightings. | `weight_sensitivity.csv` | Reported |
| 10 | Major | **The served ratio depends on MDT row order.** Found on munich; `serve_rows` is unchanged and was not re-tested on this scenario. | Row-order test, earlier run | Open, not re-tested |
| 11 | Major | **Unconstrained, operationally unrealistic changes.** All 27 tilts move, up to 11.7 deg, with no movement cost or change budget. | `tilt_movement_summary.csv` | Reported |
| 12 | Major | **Every method trades overlap for coverage.** Overlap rate and layer-separation desirability are the two KPIs worsened by all three methods. | `kpi_scoreboard.csv` | Reported |
| 13 | Minor | **Budget too small for TuRBO.** Its best came at evaluation 141 of 145. | `method_cost.csv` | Open |
| 14 | Minor | **Physical simplifications.** Full-load co-band SINR, Shannon rate without MCS, a single UE height, no field calibration, one scene exported from OSM without validated materials. | ADR 0001, `scene.xml` | Open |
| 15 | Minor | **The overlap margin is still hard.** Only the `> 0` indicator was softened, so a neighbour 6.1 dB down contributes nothing. | ADR 0004 | Reported |

Closed from the previous review: the rule sweep converging in 3 of 30 evaluations (it now takes 18 of 29).

---

## 5. Improvement suggestions

Ordered by impact per unit of effort.

### Fix before any further run

1. **Restrict the grid to the service area.** Clip the KPI grid to, say, 1.5x the cell radius around the nodes, or add nodes so the site layout covers the scene. Without this the area KPIs mostly measure distance from the sites (critique 6).
2. **Start from a defensible incumbent.** Either set the starting tilts from geometry (`atan((mast_height_m - ue.height_m) / R)` is about 1 deg here) or report gains against both the configured start and a geometric start. Otherwise the reported gain is mostly correcting the start (critique 5).
3. **Weight coverage by demand.** Add a demand-weighted hole term, or restrict the hole KPIs to tiles that carry UE reports. The tables already exist in `src/evaluation/maps.py` (critique 1).
4. **Replace the placeholder `kpi.capacity` values** before reading anything into the served ratio (critique 8).
5. **Measure solver noise and put a tolerance back on verdicts.** Repeated solves of one configuration under different seeds cost under a minute at 3.4 s an evaluation (critique 7).

### Fix the objective

6. **Add a tilt-change cost or budget.** 27 simultaneous changes of up to 11.7 deg is not a deliverable.
7. **Make the served ratio order-invariant.** Sort by a stable key before the seeded permutation in `serve_rows`.
8. **Reconsider `hole_dbm = -120`** and the overlap definition, as in the previous review; neither was changed.

### Strengthen the evidence

9. **Run multiple seeds.** Notebook 04 already computes means, intervals and a paired Wilcoxon test; they are empty with one seed.
10. **Raise TuRBO's budget or show a budget curve.**
11. **Add held-out scenarios** with different `simulation.seed`.
12. **Seed TuRBO from the rule sweep** to test whether the model beats the heuristic from the same footing.

### Clean up the scope

13. **Either implement the MARL arm or remove it from the title, README and architecture diagram.**

---

## 6. Deferred work

| Item | Command or location |
|---|---|
| Multi-seed comparison | one seed was run by instruction |
| Solver-noise measurement | not implemented; see critique 7 |
| Row-order sensitivity on this scenario | not re-run; see critique 10 |
| Held-out scenarios | not implemented |
| Demand-weighted coverage KPI | not implemented; see suggestion 3 |
| Notebooks 00-04 | not re-executed; `task pipeline` does not run them, so their embedded outputs still show the munich run |
