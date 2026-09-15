# Review: Multi-Band Tilt Coordination with TuRBO

| | |
|---|---|
| Scenario | `scn_61e174bef723f9c4`: 4 sites, 12 cells, 3 bands (700 / 1800 / 2600 MHz), 36 tilt variables |
| Code | commit `ec82036` plus the uncommitted refactor of 2026-09-15 |
| Runs | one search seed per method, `optim.seed` = the global `seed: 42` in `configs/config.yaml`; 145 evaluations for TuRBO and random search; the multi-seed comparison is deferred |
| Evidence | `reports/tables/` and `reports/figures/`, produced by notebooks 00–04 on 2026-09-15 |

This review is written as a hostile referee would write it: every claim is attacked first, then a fix is proposed.
Insight interpretation is left to the authors; this document states what the numbers support.

---

## 1. Summary

1. **All three methods beat the current configuration by a wide margin, and the gain is not solver noise.** Re-traced under five other solver seeds, TuRBO's score gain is +0.221 (95% CI 0.209–0.234), the rule-based sweep +0.165, random search +0.135; every seed improves.
2. **TuRBO found the highest weighted score (−0.277 vs −0.331 rule, −0.358 random), and it is the only method that reduced co-band overlap.** Its lead depends on the weights: with coverage-only or hole-only weights the rule-based sweep ranks first and TuRBO second.
3. **The rule-based sweep reached about three quarters of TuRBO's gain (75% re-traced, 76% as selected) with 27 evaluations and 1.1 minutes, against 145 evaluations and 8.5 minutes, and it closed the most coverage holes.**
4. **The current configuration is a weak reference.** Every tilt starts at a uniform default, and 2600 MHz starts at its maximum downtilt. Nearly every proposed change is an uptilt (32 of 36), and the median random candidate already beats the current configuration.
5. **The KPIs do not measure what users experience.** Coverage holes carry 0% of traffic demand, and the network is capacity-saturated: 22% of UE reports are not served, and almost every cell-band peaks at 90–100% PRB utilisation. The served ratio therefore mostly reflects placeholder capacity settings.
6. **Nothing here supports a generalisation claim.** One scenario, one search seed, no held-out evaluation, and the title's Multi-Agent Reinforcement Learning arm does not exist.

---

## 2. Results

### Best configuration per method (seed 42)

| | Current | Random search | Rule-based sweep | TuRBO |
|---|---:|---:|---:|---:|
| Coverage hole rate (↓) | 0.1618 | 0.1549 | **0.1516** | 0.1548 |
| Co-band overlap rate (↓) | 0.3531 | 0.3578 (worse) | 0.3615 (worse) | **0.3407** |
| Served UE ratio (↑) | 0.7758 | 0.8208 | 0.8279 | **0.8335** |
| Weak coverage rate (↓) | 0.3491 | 0.3064 | **0.2958** | 0.3021 |
| Weighted score (↑) | −0.5041 | −0.3579 | −0.3310 | **−0.2766** |
| Coverage holes closed / opened | — | 720 / 20 | **1037 / 4** | 802 / 94 |
| Evaluations | — | 145 | 27 | 145 |
| Evaluation of the best | — | 125 | 11 | 142 |
| Wall clock [min] (ray tracing) | — | 13.2 (11.2) | 1.1 (0.8) | 8.5 (5.6) |

Source: `kpi_scoreboard.csv`, `method_cost.csv`; hole counts from `coverage_before_after` logic (`src/evaluation/maps.py` `change_mask`).

### Robustness checks

| Check | Result | Source |
|---|---|---|
| Score gain re-measured under 5 other solver seeds | TuRBO +0.221 [0.209, 0.234]; rule +0.165 [0.155, 0.176]; random +0.135 [0.127, 0.143]; 100% of seeds positive for all | `retraced_gain.csv` |
| Ray-tracer noise vs configured tolerance (standard deviation) | hole 0.0008 (tol. 0.002); overlap 0.0004 (0.001); **served 0.0063 (0.001)**; weak 0.0006 (0.004) | `solver_noise.csv` |
| KPIs recomputed from archived maps | 16 of 16 within tolerance | `kpi_reproducibility.csv` |
| Comparability of runs | 21 of 21 checks hold | `comparability_checks.csv` |
| Ranking under other weights | TuRBO first under configured, equal and served-only weights; **rule-based sweep first and TuRBO second under coverage-only and hole-only weights**; random search last everywhere except equal weights | `weight_sensitivity.csv` |
| Best score vs typical candidate | Initial Sobol design median −0.432 (+0.072 over current); TuRBO's median candidate −0.331 exceeds random search's best −0.358 | `winner_vs_candidates.csv` |
| TuRBO vs random search, paired by seed | 1 pair only (gain +0.081); no test possible | `paired_gain_turbo_vs_random.csv` |
| Notebook 04 vs `task evaluate` | identical tables | re-run log |

---

## 3. Insights from the data

### Network and demand (notebooks 00–01)

| Finding | Evidence |
|---|---|
| The low band is the coverage floor, as the problem statement assumes | Coverage holes per band alone: 2600 MHz 35.0%, 1800 MHz 30.0%, 700 MHz 19.9%; all bands combined 16.2% |
| Holes are where nobody is | Holes cover 16.2% of the area and carry 0.0% of peak PRB demand; weak-coverage tiles carry 76.7% of demand |
| The network is capacity-limited | 22.4% of UE reports not served; almost every cell-band reaches 90–100% peak PRB utilisation |
| Serving follows the preference rule, not propagation | 53.8% of reports on 2600 MHz, 14.0% on 1800 MHz, 9.8% on 700 MHz |
| Measurement noise is consistent with its setting | MDT minus radio map: mean 0.00 dB, standard deviation 4.00 dB (configured 4 dB) |

### What the optimizers changed (notebooks 03–04)

| Finding | Evidence |
|---|---|
| The optimizers mainly undo excessive downtilt | 32 of 36 cell-band tilts move up; mean change 2600 MHz −7.1°, 1800 MHz −5.5°, 700 MHz −4.3°; largest 9.7° |
| Uptilting trades near-site signal for reach | Best-server RSRP drops in rings around every site and rises over most of the remaining area, for all three methods (`rsrp_change_maps.png`) |
| No distinct band roles emerge | All three bands are uptilted in the same direction, the high band most; nothing resembles "low band to the edge, high band near the site" |
| Coverage improves mostly where users are not | Hole area falls from 16.2% to 15.2–15.5%, while the share of demand in holes stays near zero (0.29% after TuRBO) |
| Users gain through weak coverage and capacity | Demand in weak coverage falls from 76.7% to 69.6–70.6%; not-served reports fall from 22.4% to 16.7% (TuRBO), 17.2% (rule) and 17.9% (random); TuRBO raises median served SINR from 5.1 to 6.3 dB and cuts median PRBs per served UE from 53.5 to 46.2 |
| Traffic shifts to the high band | The share served on 2600 MHz rises from 53.8% to about 62% for all methods |
| TuRBO had not converged | Its best configuration came at evaluation 142 of 145, and its progress curve was still rising |
| Random search spends most of its budget without progress | Its best score did not change between evaluation 7 and 124 |

---

## 4. Reviewer's critique

Severity: **Critical** blocks publication; **Major** changes a conclusion; **Minor** weakens it.

| # | Severity | Attack | Evidence | Status |
|---|---|---|---|---|
| 1 | Critical | **No generalisation.** Every configuration is tuned and scored on one city and one population; no result measures transfer to another scenario. | One scenario on disk | Open |
| 2 | Critical | **No statistics.** One search seed per method; the method comparison is anecdotal. | `paired_gain_turbo_vs_random.csv` has one pair | Open; multi-seed experiment to be set up by the authors |
| 3 | Critical | **The claimed contribution does not exist.** The title names Multi-Agent Reinforcement Learning; only TuRBO and two baselines are implemented. | `src/optim/methods/` | Open |
| 4 | Major | **Straw-man reference.** Uniform default tilts, with 2600 MHz at its maximum downtilt, can only be improved by uptilting; the median random candidate already gains +0.07. The headline "improvement over the current configuration" mostly measures a bad starting point. | `winner_vs_candidates.csv`, `decision_variables.csv` | Mitigated by reporting gains over candidates and baselines; a realistic operator-tuned reference is still missing |
| 5 | Major | **TuRBO's lead depends on the weights.** Its advantage comes from overlap, which carries weight 3; under coverage-only or hole-only weights the rule-based sweep beats it with a fifth of the evaluations, and the rule-based sweep closes the most holes. | `weight_sensitivity.csv`, hole counts in section 2 | Reported |
| 6 | Major | **The hole rate optimizes empty ground.** Holes carry no demand, yet the hole rate has the largest weight (4). | `coverage_by_area_and_demand.csv` | Reported |
| 7 | Major | **The served ratio is not a trustworthy KPI.** (a) Placeholder capacity settings saturate every cell. (b) Its tolerance (0.001) is 6× smaller than the ray tracer's own noise (0.0063), so every served-ratio "better/worse" verdict is meaningless. (c) It depends on MDT row order: the same map and UEs give 22.35%–22.61% not served for different row orders, a spread 2.6× its tolerance. | `solver_noise.csv`; row-order test in this review | Open |
| 8 | Major | **Unconstrained, operationally unrealistic changes.** All 36 tilts move, by up to 9.7°, with no movement cost or change budget; no operator would roll this out in one step. | `tilt_movement_summary.csv` | Reported |
| 9 | Minor | **Budget too small for TuRBO.** Its best came at evaluation 142 of 145, so its result is a lower bound and the method comparison is budget-dependent. | `search_progress.png` | Open |
| 10 | Minor | **Unrealistic measurements are kept.** 4.6% of reported RSRP values lie below −156 dBm, outside the NR reporting range (3GPP TS 38.133, to be verified). | `data_quality.csv` | Open |
| 11 | Minor | **Physical simplifications.** Full-load co-band SINR, Shannon rate without MCS, a single UE height, no calibration against field measurements. | ADR 0001 | Open |
| 12 | Minor | **Bit-level non-reproducibility.** Re-solving the same radio map on the GPU changes values by up to 1.5×10⁻⁵ dB; negligible for the KPIs, but byte-level checksums of artifacts fail. | re-trace comparison, 2026-09-15 | Reported |

---

## 5. Improvement suggestions

Ordered by impact per unit of effort.

### Fix before any further run

1. **Make the served ratio independent of row order.** In `src/kpi/capacity.py` `serve_rows`, sort UEs by a stable key (`t_index`, `tile_row`, `tile_col`, `x`, `y`) before the seeded permutation. This changes every served-ratio value, so re-run all methods afterwards.
2. **Set `kpi.tolerance.served_ratio` from the measured noise** (`solver_noise.csv`: about 0.006, rounded up), and review the other three tolerances against the same table.
3. **Replace the reference configuration.** Use the rule-based sweep's result, or a documented operator-style tuning, as the "current configuration". Report gains against it, not against uniform defaults.

### Fix the objective

4. **Weight coverage by demand.** Replace or complement the tile-weighted hole and weak rates with demand-weighted versions (the tables already exist in `src/evaluation/maps.py`), or restrict them to tiles with demand.
5. **Normalise the score.** Divide each KPI change by its measured noise before weighting, so no KPI dominates through its range (ADR 0003 lists this as a rejected alternative; the evidence now favours it).
6. **Add a tilt-change cost or budget**, for example a penalty per degree moved or a maximum number of changed cell-bands, and report the trade-off between gain and number of changes.
7. **Replace placeholder capacity values** with operator or 3GPP-based numbers, then check that cells are no longer uniformly saturated.

### Strengthen the evidence

8. **Run a multi-seed comparison.** Once several seeds exist per method, notebook 04 already reports means, 95% intervals and a paired Wilcoxon test.
9. **Increase TuRBO's budget or show a budget curve** (score against evaluations for 100, 200 and 400 evaluations) so the comparison does not depend on where the budget was cut.
10. **Add held-out scenarios.** Generate at least three more scenarios with different `simulation.seed`, tune on one and re-evaluate the recommended tilts on the others without re-optimizing.
11. **Test the band-role hypothesis directly.** Report per-band tilt and per-band coverage footprint of the best configurations; the current results show uniform uptilting, not differentiated roles.
12. **Seed TuRBO's initial design with the rule-based result.** The sweep reaches a strong region in 11 evaluations; starting the trust region there would test whether the model adds value beyond the heuristic.

### Clean up the scope

13. **Either implement the MARL arm or remove it from the title, README and architecture diagram.**
14. **Clip or flag RSRP below the 3GPP reporting range in the synthetic MDT**, citing the exact clause once verified.

---

## 6. Deferred work

| Item | Command or location |
|---|---|
| Multi-seed comparison | To be set up by the authors; notebooks 03a and 03b accept `BAND_TILT_SEEDS` and reuse runs already on disk |
| Held-out scenarios | not implemented |
| Insight and observation text in the notebooks | `**Observations.** _To be written._` cells in notebooks 00–04 |
