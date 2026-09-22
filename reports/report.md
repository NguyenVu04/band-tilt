# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-22: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0010](../docs/adr/0010-monotone-strength-aware-objective.md)): the share of the grid served *cleanly by exactly one cell at usable strength*, on whichever layer serves it best. J lies in [0, 1], is 1 only when every tile has a single dominant server at or above the weak threshold, and has no free parameters. Ten KPIs were reported beside it, none of them weighted into it.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations; the rule sweep was allowed up to 120.

**Results.**
- **TuRBO** scored highest on J: 0.7648 against 0.7333 currently, a 4.30 % gain. It reached the highest *effective coverage* — tiles with one dominant server at usable strength — at 65.8 % against 58.5 %, and improved 8 of the 10 reported KPIs.
- **The rule sweep** reached J = 0.7512 (+2.43 %) in 112 evaluations, improving 8 KPIs, and took the largest share of the hole, weak-coverage, RSRP, cell-edge SINR and served-rate gains.
- **Random search** reached J = 0.7503 (+2.31 %), also improving 8 KPIs and worsening 3.

Two findings shape how these should be read. First, **the objective and the reported KPI set now largely agree**, which they did not under the objective this study replaced: TuRBO wins J, and the candidate with the best equal-weight rank score across the ten KPIs is also a TuRBO candidate, 0.0043 of J below the one it picked. Head to head the rule sweep still takes 8 of the 10 individual KPIs. Second, **all three methods worsen co-band overlap** on the band-collapsed KPI, even though all three improve it on 2600 MHz and TuRBO improves it on 1800 and 700 MHz too. Section 6.4 works through why, and Section 3.4 explains why this is the one measure the new objective tracks worse than the old.

**Caveats.** The results come from one scenario, one search seed per method, and a simplified capacity model. No configuration is recommended for deployment.

---

## 1. Introduction

A 5G/6G site commonly radiates several frequency bands from the same mast, and the bands differ physically:
- Low bands such as 700 MHz propagate farther and penetrate buildings better.
- High bands such as 2600 MHz carry more capacity over a smaller footprint.
- A mid band such as 1800 MHz sits between the two.

A well-tuned network gives each layer the role its propagation suits.

In practice, antenna tilt is often set per band from a static planning value and adjusted by hand. When each layer is tilted without regard to the others, two failures become likely:
- Several bands may cover the same area strongly, which wastes resources and raises co-channel interference.
- The cell edge may develop coverage holes where no layer reaches.

This project treats the tilts of every (cell, band) pair as one coordinated optimization problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) scores every proposed configuration. Three searches of increasing sophistication are compared on the same objective:
- an operator-style rule,
- random search,
- trust-region Bayesian optimization (TuRBO [2]).

The project also plans a Multi-Agent Reinforcement Learning arm. It is not implemented, so this report does not evaluate it.

## 2. Problem Definition

**Network.** The study area is a local city scene (`data/external/scene/scene.xml`), rasterised into a 326 × 310 grid of 20 m tiles: 6,200 × 6,520 m, or 101,060 tiles.
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side, the RMa inter-site distance of 3GPP TR 38.901 Table 7.2-1 [5].
- **Cells.** Each node has three sectors at azimuths 45°, 165° and 285°, on 25 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve cells and 36 cell-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{36}$$

Results are reported as offsets from the current configuration: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell (`tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another cell of the same band within 6 dB of that band's strongest cell;
- serves more UEs within each cell-band's PRB limit;
- preserves each band's physical role.

**Starting condition** (`tables/01_eda/`):
- **Coverage.** 11.2 % of the grid is a coverage hole, 30.7 % is weakly covered (−120 to −90 dBm) and 58.1 % has good coverage. Holes are a periphery effect: 0.4 % of tiles within 1 km of a node are holes, against 14.3 % beyond (`hole_summary.csv`). Half the hole area — 5,755 tiles of 11,368 — has no propagation path on any band at all.
- **Band behaviour.**
  - Alone, 700 MHz leaves 13.8 % of the grid in a hole, 1800 MHz 23.5 % and 2600 MHz 28.8 % (`coverage_classes_per_band.csv`).
  - By raw signal, 700 MHz is the strongest layer on 91.9 % of the covered area.
  - 2600 MHz is preferred whenever it clears the threshold, so the serving rule puts 80.2 % of covered area on 2600 MHz before PRB limits (`serving_area_per_band.csv`). The gap between which layer is strongest and which layer serves is the central tension in this configuration.
- **Overlap.** 31.6 % of tiles have at least one overlapping co-band neighbour, with a mean of 0.96 neighbours per covered tile (`overlap_neighbour_summary.csv`).
- **Service.** 47.3 % of UE rows are not served (`serving_band_mix.csv`). 21.2 % of UE rows stand on hole tiles, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Weighted by peak PRB demand, 83.6 % sits on weak tiles and none in holes (Table 3). A UE with no candidate cell-band adds no PRB demand, so demand in holes is invisible in that measure, not absent.

![Study area](figures/00_simulation/study_area.png)

*Figure 1. Study area: twelve cells on four nodes and a sample of UE positions over the scene. Source: `figures/00_simulation/study_area.png`.*

![RSRP per band](figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the current tilts. Source: `figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](figures/01_eda/demand_vs_coverage.png)

*Figure 3. Peak PRB demand beside signal strength at the current tilts. Source: `figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. The PRB limits are N_RB at 15 kHz SCS, TS 38.101-1 Table 5.3.2-1 [6]. Source: [`tables/00_simulation/frequency_bands.csv`](tables/00_simulation/frequency_bands.csv).*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per cell |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

*Table 2. Scenario. Sources: [`tables/00_simulation/study_area.csv`](tables/00_simulation/study_area.csv), [`ue_distribution.csv`](tables/00_simulation/ue_distribution.csv), [`ue_measurement_summary.csv`](tables/00_simulation/ue_measurement_summary.csv), [`decision_variables.csv`](tables/00_simulation/decision_variables.csv).*

| Property | Value |
|---|---|
| Scenario ID | `scn_7d938e15f9ac4618` |
| Grid | 326 × 310 tiles, 20 m (6,200 × 6,520 m) |
| Nodes / cells / cell-band pairs | 4 / 12 / 36 |
| Mast height | 25 m |
| Current tilt (2600 / 1800 / 700 MHz) | 12° / 10° / 8° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | 10,087 |
| UE positions with no path to any cell | 17.5 % |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`tables/01_eda/coverage_by_area_and_demand.csv`](tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of peak PRB demand |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 11,368 | 11.2 % | 0.0 % |
| Weak (−120 to −90 dBm) | 31,017 | 30.7 % | 83.6 % |
| Good (> −90 dBm) | 58,675 | 58.1 % | 16.4 % |

## 3. Proposed Solutions

All three solutions search the same bounded tilt box (`src/optim/space.py`) with the same Sionna-RT evaluator (`src/optim/evaluator.py`). The evaluator builds the scene once and uses one fixed solver seed, so every candidate shares the same Monte-Carlo noise. The solutions also share one definition of "better": the objective J (Section 3.4). They differ only in where they look.

Each run has four steps (`src/optim/run.py`, `src/optim/report.py`):
1. Evaluate the current configuration.
2. Search, scoring every candidate on all UEs.
3. Re-trace the winner once, to archive its radio map.
4. Publish a shortlist of the `optim.n_solutions` = 4 highest-J configurations, always including the current one.

### 3.1 Rule-based per-band sweep

This mimics the heuristic an operator would use. Every cell on a band shares one tilt, which collapses the 36 dimensions to three. Coordinate descent then passes over the bands: it tries `n_steps = 10` evenly spaced tilts across the band's range, keeps the best, and moves to the next band, for `n_rounds = 4` passes. A value equal to the current one is skipped, so the budget is at most 3 × 10 × 4 = 120 evaluations.

The run is deterministic and spent 111 sweep evaluations plus the incumbent: 30 in the first pass, then 27 per pass once each band's current value is on the grid. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. It perturbs a random subset of the centre's dimensions and Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. It doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0003 and ADR 0010.

### 3.4 The objective every solution maximises

The objective is [ADR 0010](../docs/adr/0010-monotone-strength-aware-objective.md), implemented in `src/optim/objective.py`:

$$J = rac{1}{|G|}\sum_{g\in G} \max_b\, u_{bg},
\qquad u_{bg} = \lambda_{bg}\, e^{1-\lambda_{bg}}\, s_{bg},
\qquad \lambda_{bg} = 1 + m_{bg},
\qquad s_{bg} = \mathrm{clip}\!\left(rac{R_{b,\max}(g) - T_{	ext{cov}}}{T_{	ext{weak}} - T_{	ext{cov}}},\, 0,\, 1
ight)$$

- **Every band is scored, and the tile takes its best layer.** There is no band selection. `kpi.capacity.band_preference` belongs to the serving rule and the objective does not read it, so a tile can be scored on a layer no UE would be admitted on.
- $m_{bg}$ counts the other cells **on band $b$ alone** that are above $T_{	ext{cov}}$ and within $\Delta_R$ of that band's strongest (`src/kpi/overlap.py::overlap_neighbors_per_band`). It is the same count the overlap rate thresholds, read per band instead of summed over all three. It is co-band: nothing crosses the band axis.
- $\lambda_{bg}$ is therefore the number of cells contending to serve tile $g$ on band $b$, and $\lambda_{bg} = 0$ where that band does not cover it — which includes every tile the ray tracer found no path to.
- $s_{bg}$ is how far that band's strongest cell sits between the hole and weak thresholds. A server at $-90$ dBm or better keeps all of its utility, one just out of a hole keeps almost none, and power beyond $-90$ dBm buys nothing.
- $T_{	ext{cov}}$ = `kpi.hole_dbm` = −120 dBm, $T_{	ext{weak}}$ = `kpi.weak_dbm` = −90 dBm and $\Delta_R$ = `kpi.overlap_margin_db` = 6 dB, so each physical quantity keeps one threshold shared with the KPIs.

**The shape of the utility.** At full strength, $\lambda e^{1-\lambda}$ is the whole of the objective's preference over crowding:

| $\lambda$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| $u = \lambda e^{1-\lambda}$ | 0.000 | **1.000** | 0.736 | 0.406 | 0.199 | 0.092 |

It peaks at exactly 1 when one cell dominates the tile. A tile scoring 1 is what this report calls **effectively covered**: at or above the weak threshold on some band, with nothing else on that band within 6 dB. So $J \in [0, 1]$, **higher is better**, and $J$ reads as the share of the grid that is effectively covered, discounted for how badly the rest is crowded and how marginal its signal is.

**The exchange rate this implies, and why it is not what it looks like.** Splitting a clean, strong tile between two cells moves it from λ = 1 to λ = 2 and costs $1 - 2e^{-1} = 0.264$. Closing a hole looks like it should gain the full 1.000 — but it cannot, because a tile that has just crossed $T_{\text{cov}}$ sits near −120 dBm, where $s \approx 0$. At −119 dBm a newly covered tile is worth 0.033; at −110 dBm, 0.333. **Measured on this run, the average tile TuRBO newly covers is worth 0.085** (Section 6.4).

So the real exchange rate runs the other way: **one newly crowded strong tile costs about three closed holes.** Under ADR 0009, where every covered tile scored 1.000 regardless of strength, one closed hole paid for 3.78 crowded ones and the objective was strongly hole-seeking. The strength factor inverts that. J is now primarily a *signal-strength and cleanliness* measure that treats hole-closing as a minor bonus, which is visible in its rank correlations: 0.918 against cell-edge RSRP and 0.877 against weak rate, but only 0.738 against the hole rate.

**Why the maximum over bands.** The previous objective ([ADR 0009](../docs/adr/0009-effective-coverage-objective.md)) scored the most preferred band clearing $T_{	ext{cov}}$, which made the score depend on which band the tilts left standing. Dropping a crowded 2600 MHz layer below the threshold moved a tile onto a cleaner 1800 MHz layer and *raised* $J$: a tile at −80 dBm with one neighbour scored 0.736, and the same tile alone on 1800 MHz at −118 dBm scored 1.000 — 38 dB worse for +0.264. Measured on that study's three winner maps, this surface covered 7.0–7.6 % of the grid and was worth +0.024 to +0.027 of $J$, against TuRBO's entire +0.0151 gain over the incumbent. No run took it, but it was the largest gradient in the objective. A maximum over bands cannot be raised by removing a layer.

**What it does not read.** Cell load, where UEs stand, and inter-band interference. That last one is a real gap: $m$ is co-band, the reported overlap rate merely sums the three per-band counts, and $J$ does not read SINR — so a clean 1800 MHz layer scores full marks with 2600 MHz covering the same tile. The objective has **no free parameters**: it reads three KPI thresholds, all of which the reported KPIs already define. There is nothing to tune and nothing to re-derive per scenario.

**Nothing is weighted by demand.** ADR 0007 weighted each tile $w_g = 1 + r_g$ by its MDT report count. ADR 0010 deleted it, on measurement: `r = 0` on 97.0 % of tiles, the mean weight was 1.0035, and the whole weighting moved $J$ by 0.0005. A hole where nobody stands now costs exactly what a hole in a hotspot costs. Where the traffic stands is still reported, in Table 8b, and optimised by nothing.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. It drives the served ratio and every capacity table, and it is now the only thing that reads the band preference.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, over all ten:
   - coverage hole rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓, weak-coverage rate ↓
   - cell-edge and median RSRP ↑ (5th and 50th percentiles of best-server RSRP over covered tiles [4], read beside the hole rate)
   - cell-edge and median best-server SINR ↑
   - served UE ratio ↑
   - cell load imbalance ↓

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the ten is weighted into J, so agreement between J and the KPIs is a finding, not a construction. Peak PRB utilisation was an eleventh until [ADR 0010](../docs/adr/0010-monotone-strength-aware-objective.md) dropped it: bounded by the 0.8 admission ceiling by construction, it took three distinct values at 3 d.p. across the previous study's 318 candidates. It is still computed, as the check that the serving rule held.
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by the winner against the median candidate, sample efficiency, and TuRBO paired with random search on the same seed.
4. **Robustness.** Where the configuration moves demand, not only area, and how it trades one KPI against another. Nothing in J reads demand, so this criterion is now entirely a check on the objective rather than a reflection of it.
5. **Cost.** Ray-tracing evaluations, ray-tracing minutes and wall-clock minutes per run.

## 5. Research Methodology

**Data generation.** No operator data was available. All data was produced synthetically (`notebooks/00_simulation.ipynb`, `src/simulation/`):

1. **Scenario.** The scene was rasterised onto the 20 m grid, and a population of 10 to 20 UEs was drawn every 15 minutes for 7 days.
   - Each draw mixed four elliptical Gaussian hotspots with a uniform open-ground background. The hotspots sit where surrounding building volume is high, at least 500 m apart.
   - The hotspots hold 70 % of UEs on average, modulated by a diurnal profile and AR(1) noise (`configs/simulation.yaml` `time`, `density`).
   - UEs are independent per interval, with no mobility.
2. **Radio map.** Each band was ray-traced separately at 10⁷ rays per transmitter and maximum depth 8.
   - Line of sight, specular reflection and refraction were on; diffuse reflection and diffraction were off.
   - Materials were ITU-R P.2040 and frequency-static.
   - The output was per-cell RSRP and SINR on the grid (`tables/00_simulation/propagation_parameters.csv`).
   - At 25 m masts, 2600 MHz reaches 88.1 % of tiles, 1800 MHz 87.8 % and 700 MHz 91.1 % (`tables/00_simulation/reach_per_band.csv`).
3. **MDT.** The serving rule was run on the current radio map, and the 5,311 admitted UE rows (52.7 % of the population, on 3,017 distinct tiles) form the MDT. It is kept for reference and plots; no score reads it, and no measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 22 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`).
- The checks cover grid, scenario ID, band and cell order, bounds, schedule, duplicate rows, the MDT subset and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 112.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map and `run.json` under `outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands, band carrier frequencies and KPI definition (24 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the current configuration.
4. Compares each winner with the candidates its own search evaluated.
5. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets, so criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took 2.4 to 3.8 s per candidate (Table 12).

## 6. Analysis and Interpretation

### 6.1 Comparability and correctness

All 24 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off: over all 44 recorded measures the largest absolute gap is 4.0 × 10⁻⁶ dB on median SINR, and **on J it is at most 1.3 × 10⁻¹⁰**. The differences discussed below therefore come from the configurations, not from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **Objective J ↑** | 0.7333 | 0.7512 | 0.7503 | **0.7648** |
| Effective coverage (u > 0.75) ↑ | 0.5848 | 0.6335 | 0.6332 | **0.6583** |
| Coverage hole rate ↓ | 0.1125 | **0.1046** | 0.1093 | 0.1085 |
| Co-band overlap rate ↓ | **0.3164** | 0.3249 *(worse)* | 0.3356 *(worse)* | 0.3546 *(worse)* |
| Overlap neighbours per covered tile ↓ | **0.9625** | 1.1110 *(worse)* | 1.0352 *(worse)* | 1.0352 *(worse)* |
| Weak coverage rate ↓ | 0.3069 | **0.2535** | 0.2707 | 0.2639 |
| Cell-edge RSRP p05 [dBm] ↑ | −108.56 | **−106.98** | −107.66 | −107.44 |
| Median RSRP p50 [dBm] ↑ | −84.11 | **−80.80** | −81.92 | −81.53 |
| Cell-edge SINR p05 [dB] ↑ | −5.87 | **−4.97** | −5.30 | −5.01 |
| Median SINR p50 [dB] ↑ | 8.49 | 9.90 | 10.00 | **11.03** |
| Served UE rate ↑ | 0.5265 | **0.5907** | 0.5774 | 0.5882 |
| Cell load imbalance ↓ | **0.9098** | 0.9393 *(worse)* | 1.0295 *(worse)* | 0.9976 *(worse)* |
| **KPIs better / worse** | — | 8 / 3 | 8 / 3 | 8 / 3 |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per KPI and method. Source: `figures/04_evaluation/kpi_improvement.png`.*

**The objective and the KPI set broadly agree about the winner.** TuRBO wins J by a clear margin (+4.30 % against the sweep's +2.43 % and random's +2.31 %), and it wins the quantity J is built from: effective coverage rises from 58.5 % to 65.8 % of the grid, the best of the three. The configuration with the best equal-weight rank score over the ten KPIs, out of all 402 evaluated, is a TuRBO candidate scoring J = 0.7605, 0.0043 below the one TuRBO actually picked.

Head to head, though, the rule sweep takes 8 of the 10 individual KPIs — hole rate, overlap rate, weak rate, both RSRP percentiles, cell-edge SINR, the served rate and load imbalance — while TuRBO takes only median SINR, and random search edges TuRBO on overlap neighbours by 4 × 10⁻⁷. The sweep's advantage is concentrated in area coverage, where its three shared per-band tilts widen every footprint at once.

**Measured as a proxy, the objective improved by a factor of two.** Over random search's 145 Sobol candidates — an unbiased sample of the tilt box — Spearman's rho between J and an equal-weight rank score across the ten KPIs is **0.902**, against **0.428** for the objective [ADR 0010](../docs/adr/0010-monotone-strength-aware-objective.md) replaced. Per KPI it rose on eight of the ten: weak rate 0.286 to 0.877, cell-edge RSRP 0.321 to 0.918, cell-edge SINR 0.327 to 0.902, hole rate 0.359 to 0.738. It fell on the two overlap measures, which the next paragraph takes up.

**All three methods worsen co-band overlap** — 0.3164 to 0.3249 / 0.3356 / 0.3546 — and all three worsen overlapping neighbours per covered tile. This is the one dimension on which the new objective is a *worse* proxy than the old: rho against the overlap rate fell from 0.347 to 0.251 and against overlap neighbours from 0.381 to 0.235. The cause is structural and is the direct cost of the fix in Section 3.4 — because each tile takes its best band, a crowded layer costs nothing wherever another layer at the same tile is clean. Section 6.4 shows that per band, overlap improves on 2600 MHz under every method and on all three bands under TuRBO, and explains why the band-collapsed union still rose. Pricing the non-best layers' crowding is the first thing to revisit.

**Cell load imbalance worsens under all three methods** — least under the rule sweep (−3.2 % relative), then TuRBO (−9.7 %) and random search (−13.2 %). The previous run of this study found TuRBO *improving* it (0.9098 to 0.8818); this rerun, same seed and same code for TuRBO, did not (Section 6.8, limitation 3). Nothing in J asks for even load, so whether a search lands on an even-load configuration is incidental.

### 6.3 Did the search matter?

*Table 6. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.7333 | 0.7365 | 0.7367 | 0.7455 | 0.7503 |
| Rule-based sweep | 0.7333 | — | 0.7467 | 0.7504 | 0.7512 |
| TuRBO | 0.7333 | 0.7365 | **0.7605** | **0.7640** | **0.7648** |

*Table 7. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | 0.7479 | 0.7377 | 0.7479 |
| 25 | 0.7493 | 0.7483 | **0.7536** |
| 50 | 0.7493 | 0.7512 | **0.7605** |
| 100 | 0.7503 | 0.7512 | **0.7630** |
| 145 | 0.7503 | — | **0.7648** |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **better** than the current configuration: their median is 0.7367 against 0.7333, and their 90th percentile reaches 0.7455. Under the previous objective they sat below the incumbent; the difference is that the strength factor rewards the wider footprints a random tilt tends to produce.

The evidence that the model earned TuRBO's margin is strong and takes the same form as before:

- TuRBO's **median** candidate (0.7605) scores above both baselines' single **best** (0.7503 and 0.7512).
- The two runs share the same 16 Sobol points and diverge only once the model proposes: that shared design has a median of 0.7365 for both, while TuRBO's all-candidate median is 0.7605 against random search's 0.7367.

Per method:

- **Random search** found its best at evaluation 95 and did not improve in the remaining 50.
- **TuRBO** found its best at **evaluation 143 of 145** — effectively the end of its budget. It passed the rule sweep's final answer at evaluation 18, its first model-proposed batch. A larger budget would plausibly still improve it; this is the clearest single lever left.
- **The rule sweep** found its best, 0.7512, at evaluation 49 and spent the remaining 62 evaluations without improving. Its first pass starts at the bottom of each band's range, so it trails both other methods at 10 and 25 evaluations. Quadrupling its budget from 28 to 112 evaluations bought +0.0001 of J over the previous 5-step, 2-pass sweep: the shared-tilt space is exhausted, not under-searched.

The paired gain of TuRBO over random search is **+0.0146** on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Every method lowered the hole rate and raised the band-collapsed overlap rate — but not because J traded one for the other.** Section 3.4 shows a newly covered tile is worth little under the strength factor. Decomposing each winner's gain against the incumbent: hole-closing contributes +0.0005 of TuRBO's +0.0315 and +0.0007 of the sweep's +0.0178. TuRBO takes +0.0182 from stronger signal and +0.0131 from reducing contention on each tile's best band; the sweep takes +0.0166 from stronger signal and only +0.0006 from contention, a move its three shared tilts can barely make. The sweep closes 796 hole tiles to TuRBO's 576 as a side effect of uniform uptilts.

It is not that overlap-reducing configurations were unavailable. The rule sweep's own candidate set contained one reaching an overlap rate of **0.3061**, better than the incumbent's 0.3164 (`tables/04_evaluation/sample_efficiency.csv`); J simply did not pick it. Across all 145 candidates, neither random search nor TuRBO ever found a configuration whose overlap rate beat the incumbent's at all.

**Per band, overlap improved where it matters most.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`src/kpi/overlap.py::overlap_neighbors_per_band`):

*Table 8a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **2600 MHz** (best layer on most tiles) | 0.2010 | **0.1904** | 0.1967 | 0.1928 |
| 1800 MHz | 0.2067 | 0.2092 | 0.2065 | **0.2033** |
| 700 MHz | 0.2396 | 0.2441 | 0.2304 | **0.2197** |
| Band-collapsed (the reported KPI) | **0.3164** | 0.3249 | 0.3356 | 0.3546 |

Every search cleans up 2600 MHz, and under ADR 0010 the lower layers are no longer left out: **TuRBO improves all three bands** (700 MHz 0.2396 to 0.2197, 1800 MHz 0.2067 to 0.2033), where under the previous objective the lower layers drifted worse under all three methods. The sweep worsens 1800 and 700 MHz. The collapsed rate still rises, mostly because former hole tiles joined the covered set at all and arrived with a neighbour — the searches uptilt on every band (Section 6.6), widening footprints.

**This is where the new objective is a worse proxy than the old one, and it is worth stating plainly.** Scoring each tile on its best band is what removed the exploit of Section 3.4, but it also means a crowded layer costs nothing wherever another layer at the same tile is clean. Measured over the Sobol sample, rho between J and the band-collapsed overlap rate fell from 0.347 to 0.251, and between J and overlap neighbours per covered tile from 0.381 to 0.235 — the only two of the ten KPIs that went backwards, against eight that improved. Pricing the non-best layers' crowding, without reintroducing a selection that depends on the tilts, is the open problem this study leaves behind.

*Table 8b. Coverage class by area and by demand (demand weighted by the current configuration's peak PRB demand). Nothing in J reads this view — ADR 0010 deleted the demand weighting — so it is a check on the result, not a reflection of it. Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 11.2 % / 0.0 % | 10.5 % / 0.0 % | 10.9 % / 0.4 % | 10.8 % / 1.0 % |
| Weak | 30.7 % / 83.6 % | 25.3 % / 76.8 % | 27.1 % / 79.2 % | 26.4 % / 78.1 % |
| Good | 58.1 % / 16.4 % | 64.2 % / 23.2 % | 62.0 % / 20.5 % | 62.8 % / 20.9 % |

*Table 9. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | **0.96** | **64.4 %** | 17.6 % |
| Rule-based sweep | 1.11 | 63.7 % | 17.8 % |
| Random search | 1.04 | 62.3 % | 16.1 % |
| TuRBO | 1.04 | 60.2 % | **14.8 %** |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand in holes.** The rule sweep moves **no** peak demand onto hole tiles; random search moves 0.4 % and TuRBO 1.0 %. The share of demand on weak tiles falls under every method, most under the rule sweep (83.6 % to 76.8 %, against TuRBO's 78.1 %), and the share on good tiles rises from 16.4 % to 23.2 % under the sweep and 20.9 % under TuRBO. On the demand-weighted view the rule sweep is the strongest of the three. Nothing in J reads this view any more, so it is the check catching something the objective cannot see — a small amount of served traffic pushed onto ground the network no longer covers.

**Pile-ups shrink even as the mean rises.** TuRBO cuts the share of covered tiles with three or more neighbours from 17.6 % to 14.8 %, the best of the three, while raising the mean neighbour count from 0.96 to 1.04. It is converting a few badly contested tiles into many mildly contested ones — which is what λe^(1−λ) asks for, since the penalty is steepest between λ = 1 and λ = 3 and nearly flat beyond λ = 4.

**No objective parameters to vary, and no band priority either.** Earlier versions of this report noted that the pick's robustness to τ_R and β was untested, and then that the band-priority order was a judgement the objective depended on. Both are closed by construction: the objective has no parameters, and under ADR 0010 it does not read `kpi.capacity.band_preference` at all — that order now belongs to the serving rule alone. What remains untested is the single search seed, which Section 6.8 lists.

### 6.5 Capacity impact

*Table 10. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 47.3 % | −0.38 | 5.13 | 53.2 | 34.1 % / 10.5 % / 8.1 % |
| Rule-based sweep | **40.9 %** | **+0.17** | 6.81 | 43.8 | 42.7 % / 9.0 % / 7.4 % |
| Random search | 42.3 % | +0.11 | 7.22 | 42.0 | 43.6 % / 5.8 % / 8.3 % |
| TuRBO | 41.2 % | −0.65 *(worse)* | **8.31** | **37.5** | 34.5 % / 16.4 % / 8.0 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

**Service.** Every method serves substantially more UEs: 5.1 to 6.4 points. The rule sweep serves the most (59.1 % against 52.7 %), TuRBO 58.8 % and random search 57.7 %. Median served SINR rises under all three, and the PRBs a served UE needs fall by 18 to 29 %.

**TuRBO does something the baselines do not: it spreads traffic across layers.** Both baselines push traffic onto the preferred 2600 MHz layer in the usual way — 34.1 % of reports to about 43 %. TuRBO leaves 2600 MHz at 34.5 % while lifting **1800 MHz from 10.5 % to 16.4 %**. The cell-impact table shows the mechanism: TuRBO *downtilts* several 2600 MHz sectors toward the 15° bound (n2c1 to 14.76°, n3c1 to 14.31°, n1c1 to 13.72°), shrinking those footprints — n2c1's 2600 MHz layer loses 470 served reports — while uptilting 1800 MHz almost to 0° across most of the network (n1c1 to 0.13°, n0c0 to 0.61°, n1c0 to 0.90°).

That redistribution is why TuRBO has the best median served SINR (8.31 dB against the incumbent's 5.13) and needs the fewest PRBs per UE (37.5 against 53.2): traffic moved off the most contended layer. It is also why its **10th-percentile served SINR is the only one to get worse** (−0.38 to −0.65 dB) — the UEs at the bottom of the distribution are the ones the shrunken 2600 MHz footprints dropped.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- Area covered by 2600 MHz rises 71.2 % → 72.4 %, 1800 MHz 76.5 % → 77.7 %, 700 MHz 86.2 % → 86.3 %.
- Mean RSRP where covered improves on every layer: 2600 MHz −97.2 → −94.3 dBm, 1800 MHz −91.5 → −88.4 dBm, 700 MHz −84.2 → −82.5 dBm. No band loses mean signal.
- Median served SINR on 2600 MHz rises 3.5 → 5.3 dB, and on 1800 MHz 5.2 → 9.6 dB.

About 41 % of UE reports remain unserved. Much of that is out of reach: 17.5 % of UE positions have no path to any cell (Table 2) and 21.2 % stand on hole tiles. The rest is PRB exhaustion — peak utilisation sits at the 0.8 admission ceiling in every configuration (`tables/04_evaluation/cell_impact.csv`), so the 2600 MHz layer remains the binding constraint on service.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 11. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 5.47 | 11.73 | −4.13 |
| 1800 MHz | 12 / 12 | 7.53 | 9.87 | −6.36 |
| 700 MHz | 12 / 12 | 5.18 | 7.74 | −2.99 |

Every one of the 36 cell-bands moved. The configuration is a net uptilt overall, but it is **not** uniform, and the structure is the point:

- **Every band has a negative mean change**, so footprints widen across the network: −6.36° on 1800 MHz, −4.13° on 2600 MHz, −2.99° on 700 MHz. That widening is what lifts the hole and weak rates, and it is also what brings new tiles into coverage carrying a neighbour, which is why the band-collapsed overlap rate rises.
- **Nine of the 36 are downtilted, spread across all three bands** (four on 2600 MHz, three on 700 MHz, two on 1800 MHz). Under the previous objective every downtilt was on 2600 MHz; now that each band is scored on its own, the search trims contention wherever it finds it.
- **1800 MHz moves most**, averaging 7.53° of absolute change against 5.47° and 5.18°, so the mid layer is where the search does its restructuring.
- Proposed tilts span **0.13° to 14.82°**, so the [0°, 15°] box is binding at both ends. The largest single change is 11.73°.

This reads as the search re-dividing the layers cell by cell rather than band by band. The clearest case is n2c1, whose 2600 MHz carrier is downtilted 2.76° and loses 470 served reports while its 1800 MHz carrier is uptilted 8.76° and gains 269 — the same sector handing traffic from the high band down to the mid band, with median SINR moving −3.8 dB and +4.3 dB to match. n1c1 does the same (2600 MHz −241, 1800 MHz +339). n3c2 and n1c0 run the other way, uptilting 2600 MHz by 11.7° and 10.5° and gaining 357 and 227 reports. That is a strategy a per-band sweep cannot express, and it is where TuRBO's J margin comes from.

The rule sweep instead sets every cell on a band to one value (`outputs/tilt_change_rule.csv`) — a uniform, and very large, uptilt that drives 1800 MHz to the bottom of the box and 700 MHz to 1.67°. With one seed it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended.

The largest traffic shifts are on 2600 MHz sectors (`tables/04_evaluation/cell_impact.csv`): n2c1 −470 served reports, n1c1 −241, n3c1 −158, against n3c2 +357 and n1c0 +227. Those are the cells to watch after a rollout.

### 6.7 Cost

*Table 12. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 112 | 49 | 4.46 | 7.11 | 2.4 | +0.0178 | **0.0025** |
| Random search | 145 | 95 | 8.10 | 11.20 | 3.4 | +0.0169 | 0.0015 |
| TuRBO | 145 | 143 | 9.27 | 19.59 | 3.8 | **+0.0315** | 0.0016 |

TuRBO's GP fitting and acquisition added about 10.3 minutes of wall clock on top of its ray tracing, roughly doubling it. Ray-tracing cost per evaluation is higher for TuRBO than random search (3.8 vs 3.4 s) because its proposals reach further into the box, where more rays survive.

**By J gain per minute the rule sweep is about 1.6 times as cost-effective as TuRBO**, reaching 57 % of TuRBO's gain for about a third of the wall clock. At the previous 28-evaluation budget that ratio was five: the extra 84 evaluations cost minutes and bought almost nothing (Section 6.3). The sweep does not compete on quality: TuRBO passes its final J at evaluation 18 of 145 and ends 0.0137 above it. Its remaining case is cost, and the 8 of 10 individual KPIs it wins (Section 6.2).

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed. The TuRBO–rule margin is +0.0137 on J, but it still rests on one pair.
3. **Winner's curse, and TuRBO is not bit-reproducible.** Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured. Rerunning the same seed reproduced random search exactly (J to 10⁻¹¹), but TuRBO diverged at its first model proposal: the GPU ray tracer is reproducible only to about 10⁻¹¹ in J, and the GP fit amplifies that into different proposals. The rerun reached J = 0.7648 against the previous 0.7650 — but its winner worsens load imbalance where the previous one improved it. Any per-KPI claim about TuRBO's winner beyond J should be read as one draw.
4. **The objective prices crowding worse than the KPI does.** Because each tile is scored on its best band, a crowded layer costs nothing wherever another layer at the same tile is clean. Rho between J and the band-collapsed overlap rate is 0.251, and 0.235 against overlap neighbours per covered tile — the two weakest of the ten, and both *worse* than under the objective ADR 0010 replaced. This is the direct cost of removing the band selection, and it is the first thing to revisit.
5. **Inter-band interference is priced nowhere.** The overlap count is co-band by definition, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. A clean 1800 MHz layer scores full marks with 2600 MHz covering the same tile. Fixing this needs an interference model, not a reweighting.
6. **Nothing is weighted by demand.** ADR 0010 deleted the weighting as measurably inert (it moved J by 0.0005), so a hole where nobody stands now costs exactly what a hole in a hotspot costs. Table 8b is the only place demand appears, and it shows 1.0 % of peak demand landing on hole tiles under the winner where none did before.
7. **The capacity model is a simplification.** It drives the served ratio and every capacity figure. It uses a Shannon rate with no MCS cap, full-load co-band interference against partial PRB load, and no receiver noise figure.
8. **Tilts at both bounds.** The winner places cell-bands at 0.13° and 14.82°, so [0°, 15°] is binding at both ends and the box may be too narrow.
9. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability with earlier runs.** No J in this report is comparable with any figure recorded before 2026-09-20. The objective was replaced (ADR 0010), one KPI was dropped and the demand map was deleted. `kpi.objective_version` now refuses to pool scores across that change, checked against the running config as well as run to run. Every run traced under the previous objectives was deleted rather than archived; `src/evaluation/runs.py` would now refuse to pool them, because the deleted `kpi.objective` config block is part of the KPI definition it compares.

## 7. Conclusions and Recommendations

*Table 13. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0178 (2nd) | +0.0169 (3rd) | **+0.0315 (1st)** |
| 2. Reported KPIs | **8 better, 3 worse; best on 8 of 10**, including every coverage and service measure | 8 better, 3 worse | 8 better, 3 worse; best effective coverage and median SINR, fewest pile-ups |
| 3. Search effectiveness | Best at evaluation 49 of 112; beaten by TuRBO from evaluation 18 onward | Best at 95, no gain in the next 50 | **Median candidate above both baselines' best; still improving at 143 of 145** |
| 4. Robustness | **No demand onto holes**; largest demand shift out of weak coverage | 0.4 % of demand onto holes | 1.0 % of demand onto holes; fewest 3+ pile-ups |
| 5. Cost | **7.1 min, 112 evaluations; 1.6× the J per minute** | 11.2 min, 145 evaluations | 19.6 min, 145 evaluations |

**Conclusions.**

- **TuRBO reached the highest J**, and there is good evidence the model earned it: its median candidate (0.7605) scored above both baselines' best, and it shares its first 16 Sobol points with random search, diverging only once the model starts proposing.
- **On the individual KPIs the rule sweep is the stronger practical result.** All three improve 8 of the 10 reported KPIs, but the sweep wins 8 of them head to head — every coverage and service measure, cell-edge SINR and load imbalance — against TuRBO's one (median SINR), and costs about a third of TuRBO's wall clock. TuRBO beats the sweep's final J from evaluation 18 onward. Quadrupling the sweep's budget did not change this picture: it added +0.0001 of J.
- **The objective and the report card largely agree about the winner.** Out of all 402 evaluated configurations, the one with the best equal-weight rank score across the ten KPIs is a TuRBO candidate scoring J = 0.7605, 0.0043 below the one TuRBO picked. Under the previous objective TuRBO had evaluated the best-KPI configuration and J told it to pick another, giving up 0.11 of rank score for 0.0008 of J.
- **Measured as a proxy, the objective is twice as good.** Spearman's rho between J and that rank score, over random search's 145 Sobol candidates, is 0.902 against 0.428 for the objective [ADR 0010](../docs/adr/0010-monotone-strength-aware-objective.md) replaced. It rose on eight of the ten KPIs.
- **The objective barely pays for closing holes, and that is why TuRBO does less of it.** A newly covered tile arrives near −120 dBm where the strength factor is near zero, so it is worth about 0.09, not 1.0. Across all of TuRBO's gain, hole-closing accounts for +0.0005 of +0.0315 — under 2 %. The rule sweep closes more holes (796 tiles against 576) as a side effect of uniformly uptilting whole bands, not because the objective asked for it. TuRBO instead takes +0.0131 from reducing contention on each tile's best band, which a three-variable sweep barely can (+0.0006).
- **Every layer is now priced, and it shows.** 2600 MHz overlap falls under all three methods and **TuRBO lowers it on every band** (700 MHz 0.2396 → 0.2197), where under the previous objective both lower layers drifted worse under every method. Effective coverage rises from 58.5 % to 65.8 %.
- **TuRBO found a multi-band strategy the baselines did not**: spread traffic across layers rather than concentrating it on 2600 MHz, moving 5.9 points onto 1800 MHz while both baselines pushed 2600 MHz to about 43 %. It gives the best median served SINR and the lowest PRBs per served UE, at the cost of the worst 10th-percentile served SINR.
- **Every result** depends on one scenario, one seed, an objective that under-prices crowding and prices inter-band interference not at all, and a simplified capacity model. A same-seed rerun of TuRBO reproduced its J to 2 × 10⁻⁴ but not its per-KPI profile (Section 6.8, limitation 3).

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Decide how the objective should price crowding across layers.** This is the single most consequential open question left. Taking each tile's best band removed the exploit of Section 3.4, but it also made a crowded layer free wherever another layer at the same tile is clean: rho against the band-collapsed overlap rate fell to 0.251, the weakest of the ten and worse than before. A term over the non-best layers would price it — the constraint is that it must not depend on which band the tilts leave standing, or the exploit returns.
3. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`), and re-trace the shortlists under other solver seeds. The TuRBO–rule margin of +0.0137 still needs an interval before it can be called decisive.
4. **Give TuRBO a larger budget.** It found its best at evaluation 143 of 145 and never restarted, so the trust region was still productive when the budget ended.
5. **Widen the tilt box.** The winner sits at 0.13° and 14.82°, so [0°, 15°] is binding at both ends.
6. **Model inter-band interference.** The overlap count is co-band and J does not read SINR, so nothing in the study prices a strong neighbour on another layer. This needs a model, not a reweighting.
7. **Address the 2600 MHz PRB ceiling.** Peak utilisation sits at the 0.8 admission ceiling in every configuration, so tilt alone cannot raise the served ratio much further; that is a capacity decision, not a tilt one.
8. **Build held-out scenario validation and the planned MARL arm** before drawing a method-level conclusion.

---

## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/external/scene/scene.xml` (not in Git) |
| Layout | 4 nodes, 1,732 m triangle plus centroid, 3 sectors at 45° / 165° / 285°, 25 m masts |
| Tilt | Current 12° (2600 MHz), 10° (1800 MHz), 8° (700 MHz); bounds [0°, 15°], every cell-band |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6, edge percentile 5 |
| Objective | no parameters; reads `hole_dbm`, `weak_dbm` and `overlap_margin_db`; `objective_version` 2 (ADR 0010) |
| Capacity | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission ceiling 0.8, 20 Mbps per UE, SCS 15 kHz, admission in report-time order |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12; rule 10 steps × 4 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-22_06-28-22/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-22_06-39-35/` |
| TuRBO | `outputs/optim/turbo/2026-09-22_06-47-34/` |

The previous runs (rule sweep at 5 steps × 2 rounds) are archived under `outputs/optim_archive/2026-09-20/`.

To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`reports/tables/…`) | Figures (`reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `mdt_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `mdt_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `mdt_overview`, `mdt_share_by_component`, `mdt_share_by_coverage`, `mdt_rsrp_statistics`, `cell_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `mdt_over_time`, `mdt_vs_ue_positions`, `mdt_rsrp_vs_ue_rsrp`, `cell_band_utilisation`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `template_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `decision_variables`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_<method>`, `tilt_movement_<method>`, `kpi_comparison_<method>`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary` | `search_progress`, `kpi_progress`, `tilt_movement_<method>`, `coverage_before_after_<method>`, `rsrp_change_maps`, `serving_band_mix` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `ue_service_summary`, `cell_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_served_ratio`, `tradeoff_overlap_rate_vs_served_ratio`, `rsrp_change_maps`, `coverage_before_after`, `coverage_class_maps`, `overlap_neighbour_maps`, `serving_band_mix`, `cell_band_utilisation`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell; bounds are [0°, 15°]. Negative Δ is an uptilt.*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 8.47 (-3.53) | 0.61 (-9.39) | 4.70 (-3.30) |
| n0c1 | 10.78 (-1.22) | 2.54 (-7.46) | 1.93 (-6.07) |
| n0c2 | 4.61 (-7.39) | 3.11 (-6.89) | 13.49 (+5.49) |
| n1c0 | 1.48 (-10.52) | 0.90 (-9.10) | 14.45 (+6.45) |
| n1c1 | 13.72 (+1.72) | 0.13 (-9.87) | 1.07 (-6.93) |
| n1c2 | 1.01 (-10.99) | 2.90 (-7.10) | 2.53 (-5.47) |
| n2c0 | 8.74 (-3.26) | 2.25 (-7.75) | 2.46 (-5.54) |
| n2c1 | 14.76 (+2.76) | 1.24 (-8.76) | 9.22 (+1.22) |
| n2c2 | 13.26 (+1.26) | 1.06 (-8.94) | 0.68 (-7.32) |
| n3c0 | 3.07 (-8.93) | 12.20 (+2.20) | 6.97 (-1.03) |
| n3c1 | 14.31 (+2.31) | 1.98 (-8.02) | 2.41 (-5.59) |
| n3c2 | 0.27 (-11.73) | 14.82 (+4.82) | 0.26 (-7.74) |

Nine of the 36 cell-bands are downtilted: four on 2600 MHz, two on 1800 MHz and three on 700 MHz.

The rule-based sweep's best configuration sets every cell to 8.33° on 2600 MHz, 0.00° on 1800 MHz and 1.67° on 700 MHz (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[5] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[6] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[7] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
