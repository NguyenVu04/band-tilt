# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-20: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0009](../docs/adr/0009-effective-coverage-objective.md)): the demand-weighted share of the grid served *cleanly by exactly one cell*, on the band that would actually serve it. J lies in [0, 1], is 1 only when every tile has a single dominant server, and has no free parameters. Eleven KPIs were reported beside it, none of them weighted into it.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations.

**Results.**
- **TuRBO** scored highest on J: 0.8231 against 0.8080 currently, a 1.86 % gain. It reached the highest *effective coverage* — tiles with exactly one serving cell — at 69.6 % against 66.1 %, and improved 9 of the 11 reported KPIs.
- **The rule sweep** reached J = 0.8189 (+1.35 %) in 28 evaluations, also 9 KPIs better, and took the largest share of the hole, weak-coverage, RSRP, SINR and served-rate gains.
- **Random search** reached J = 0.8155 (+0.92 %), improving 8 KPIs and worsening 4.

Two findings shape how these should be read. First, **the objective and the reported KPI set disagree**: TuRBO wins J, but the rule sweep beats it on 8 of the 11 KPIs, including every coverage and service measure. Second, **all three methods worsen co-band overlap** on the band-collapsed KPI, even though all three *improve* it on the one band the objective scores. Both follow from what J prices, and Section 6.4 works through why.

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

This mimics the heuristic an operator would use. Every cell on a band shares one tilt, which collapses the 36 dimensions to three. Coordinate descent then passes over the bands: it tries `n_steps = 5` evenly spaced tilts across the band's range, keeps the best, and moves to the next band, for `n_rounds = 2` passes. A value equal to the current one is skipped.

The run is deterministic and spent 27 sweep evaluations plus the incumbent. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. It perturbs a random subset of the centre's dimensions and Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. It doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0003 and ADR 0009.

### 3.4 The objective every solution maximises

The objective is [ADR 0009](../docs/adr/0009-effective-coverage-objective.md), implemented in `src/optim/objective.py`:

$$J = \frac{\sum_{g\in G} w_g\, \lambda_g\, e^{1-\lambda_g}}{\sum_{g\in G} w_g},
\qquad \lambda_g = 1 + m_{b(g)}(g), \qquad w_g = 1 + r_g$$

- $b(g)$ is the **most preferred band whose strongest cell clears** $T_{\text{cov}}$, in the order `kpi.capacity.band_preference` = 2600 → 1800 → 700 MHz. This is the same order the serving rule admits UEs by, so the objective judges each tile on the layer that would really carry it.
- $m_{b(g)}(g)$ counts the other cells **on that band alone** that are above $T_{\text{cov}}$ and within $\Delta_R$ of its strongest (`src/kpi/overlap.py::overlap_neighbors_per_band`). It is the same count the overlap rate thresholds, read on one band instead of summed over all three.
- $\lambda_g$ is therefore the number of cells contending to serve tile $g$, and $\lambda_g = 0$ where no band covers it — which includes every tile the ray tracer found no path to.
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm and $\Delta_R$ = `kpi.overlap_margin_db` = 6 dB, so each physical quantity keeps one threshold shared with the KPIs.
- $r_g$ is tile $g$'s MDT report count over the busiest tile's, so $w_g \in [1, 2]$ (`src/data/demand.py`).

**The shape of the utility.** $\lambda e^{1-\lambda}$ is the whole of the objective's preference:

| $\lambda$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| $u = \lambda e^{1-\lambda}$ | 0.000 | **1.000** | 0.736 | 0.406 | 0.199 | 0.092 |

It peaks at exactly 1 when one cell dominates the tile. A tile scoring 1 is what this report calls **effectively covered**: above the hole threshold, with nothing else on its serving band within 6 dB. So $J \in [0, 1]$, **higher is better**, and $J$ reads directly as "the demand-weighted share of the grid that is effectively covered", discounted for how badly the rest is crowded.

**The exchange rate this implies.** Closing a hole moves a tile from 0 to 1 and gains 1.000. Splitting a clean tile between two cells moves it from 1 to 2 and costs $1 - 2e^{-1} = 0.264$. **One closed hole therefore pays for 3.78 newly crowded tiles.** The objective is strongly hole-seeking by construction, and Section 6.4 shows the searches acting on exactly that.

**What it does not read.** Signal strength above $T_{\text{cov}}$ (a tile at −119 dBm and one at −70 dBm score alike), the non-serving layers, cell load, and anything about where UEs stand beyond the weight $w_g$. The objective has **no free parameters**: it reads two KPI thresholds and the band preference, all of which the KPIs or the serving rule already define. There is nothing to tune and nothing to re-derive per scenario.

**How much the demand weight actually moves it.** Only 2.99 % of tiles carry an MDT report at all, and because the weight is floored at 1, those tiles hold just 3.32 % of the total weight (`tables/02_preprocessing/demand_tile_weight.csv`). On this scenario the demand weighting shifts J by about 0.0005 — the unweighted mean utility is 0.8076 against the weighted 0.8080. The weighting is a statement of intent that this scenario's sparse MDT cannot make much of.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. It drives the served ratio and every capacity table, and it shares the band preference with the objective.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, over all eleven:
   - coverage hole rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓, weak-coverage rate ↓
   - cell-edge and median RSRP ↑ (5th and 50th percentiles of best-server RSRP over covered tiles [4], read beside the hole rate)
   - cell-edge and median best-server SINR ↑
   - served UE ratio ↑
   - peak PRB utilisation ↓, cell load imbalance ↓

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the eleven is weighted into J, so agreement between J and the KPIs is a finding, not a construction.
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by the winner against the median candidate, sample efficiency, and TuRBO paired with random search on the same seed.
4. **Robustness.** Where the configuration moves demand, not only area, and how it trades one KPI against another.
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
3. **MDT.** The serving rule was run on the current radio map, and the 5,311 admitted UE rows (52.7 % of the population, on 3,017 distinct tiles) form the MDT. The objective's tile weights are built from it. It is kept for later use; no measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 22 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`).
- The checks cover grid, scenario ID, band and cell order, bounds, schedule, duplicate rows, the MDT subset and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** `task optim` ran each method once with search seed 42 (Appendix A); notebooks `03a_baseline` and `03b_turbo` read those finished runs rather than re-searching.
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 28.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map and `run.json` under `outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands, band carrier frequencies and KPI definition (23 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the current configuration.
4. Compares each winner with the candidates its own search evaluated.
5. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets, so criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took about 4 s per candidate (Table 12).

## 6. Analysis and Interpretation

### 6.1 Comparability and correctness

All 23 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off: over all 48 recorded measures the largest absolute gap is 2.3 × 10⁻⁶ dB on median RSRP, and **on J it is exactly zero**. The differences discussed below therefore come from the configurations, not from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **Objective J ↑** | 0.8080 | 0.8189 | 0.8155 | **0.8231** |
| Effective coverage (λ = 1) ↑ | 0.6608 | 0.6802 | 0.6881 | **0.6964** |
| Coverage hole rate ↓ | 0.1125 | **0.1044** | 0.1094 | 0.1058 |
| Co-band overlap rate ↓ | **0.3164** | 0.3247 *(worse)* | 0.3368 *(worse)* | 0.3402 *(worse)* |
| Overlap neighbours per covered tile ↓ | **0.9625** | 1.1203 *(worse)* | 1.0501 *(worse)* | 1.0643 *(worse)* |
| Weak coverage rate ↓ | 0.3069 | **0.2538** | 0.3063 | 0.2638 |
| Cell-edge RSRP p05 [dBm] ↑ | −108.56 | **−106.98** | −108.46 | −107.42 |
| Median RSRP p50 [dBm] ↑ | −84.11 | **−80.84** | −83.81 | −81.40 |
| Cell-edge SINR p05 [dB] ↑ | −5.87 | **−5.00** | −5.85 | −5.15 |
| Median SINR p50 [dB] ↑ | 8.49 | **9.83** | 8.30 *(worse)* | 9.64 |
| Served UE rate ↑ | 0.5265 | **0.5930** | 0.5758 | 0.5787 |
| Peak PRB utilisation ↓ | 0.7999 | **0.7992** | 0.7999 | 0.7997 |
| Cell load imbalance ↓ | **0.9098** | 0.9416 *(worse)* | 1.0805 *(worse)* | 1.0325 *(worse)* |
| **KPIs better / worse** | — | **9 / 3** | 8 / 4 | **9 / 3** |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per KPI and method. Source: `figures/04_evaluation/kpi_improvement.png`.*

**The objective and the KPI set disagree, and that is the headline.** TuRBO wins J by a clear margin (+1.86 % against the sweep's +1.35 % and random's +0.92 %), and it wins the one quantity J is built from: effective coverage rises from 66.1 % to 69.6 % of the grid, the best of the three. But **the rule sweep beats TuRBO on 8 of the 11 reported KPIs**, including every coverage and service measure — hole rate (0.1044 against 0.1058), weak rate (0.2538 against 0.2638), both RSRP percentiles, both SINR percentiles and the served rate (0.5930 against 0.5787).

This is a genuine tension, not a bookkeeping artefact, and it is the objective's doing. J does not read signal strength above the hole threshold, so the rule sweep's larger RSRP and SINR gains earn it nothing; J counts only whether each tile has exactly one serving cell on its serving band, and TuRBO produces more such tiles. In the previous study the objective and the KPIs agreed, which made the pick uncontroversial. Here they do not, so **which configuration is "best" depends on whether you accept J's definition of the goal.**

**All three methods worsen co-band overlap** — 0.3164 to 0.3247 / 0.3368 / 0.3402 — and all three worsen overlapping neighbours per covered tile. None of the three has a term pushing against it: a hole closed is worth 3.78 tiles crowded (Section 3.4). Section 6.4 shows that on the band the objective actually reads, overlap *improved* under every method, and explains why the band-collapsed KPI still rose.

**Cell load imbalance worsens under all three**, most under random search (+18.8 % relative). Nothing in J asks for even load, and ADR 0007 records that as a deliberate omission.

### 6.3 Did the search matter?

*Table 6. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.8080 | 0.8017 | 0.8019 | 0.8082 | 0.8155 |
| Rule-based sweep | 0.8080 | — | 0.8128 | 0.8184 | 0.8189 |
| TuRBO | 0.8080 | 0.8017 | **0.8180** | **0.8217** | **0.8231** |

*Table 7. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | 0.8080 | 0.8173 | 0.8080 |
| 25 | 0.8127 | **0.8189** | 0.8126 |
| 50 | 0.8127 | — | 0.8186 |
| 100 | 0.8155 | — | 0.8211 |
| 145 | 0.8155 | — | **0.8231** |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **worse** than the current configuration: their median is 0.8019 against 0.8080, and even their 90th percentile (0.8082) only just clears it. The per-band starting tilts are a reasonable configuration, not a weak one.

The evidence that the model earned TuRBO's margin is strong and takes the same form as before:

- TuRBO's **median** candidate (0.8180) scores above random search's single **best** (0.8155).
- TuRBO's 90th percentile (0.8217) scores above the rule sweep's best (0.8189).
- The two runs share the same 16 Sobol points and diverge only once the model proposes. Those Sobol points averaged 0.8019; TuRBO's own 128 proposals averaged **0.8176** (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`).

Per method:

- **Random search** found its best at evaluation 62 and did not improve in the remaining 83.
- **TuRBO** never restarted and found its best at **evaluation 144 of 145** — the very end of its budget. It passed the rule sweep's best somewhere between evaluations 50 and 100. A larger budget would plausibly still improve it; this is the clearest single lever available.
- **The rule sweep** reached 0.8173 within 10 evaluations and its best 0.8189 by evaluation 11, then found nothing better in its remaining 17. Note that it beats TuRBO at every budget up to 50 evaluations.

The paired gain of TuRBO over random search is **+0.0076** on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**The coverage-versus-overlap trade is back, and the objective chose a side.** Every method lowered the hole rate and raised the band-collapsed overlap rate. That is the exchange rate of Section 3.4 acting exactly as designed: a tile moved out of a hole is worth 1.000, a clean tile split in two costs 0.264, so a search will accept nearly four newly crowded tiles to close one hole.

It is not that overlap-reducing configurations were unavailable. The rule sweep's own candidate set contained one reaching an overlap rate of **0.3067**, better than the incumbent's 0.3164 (`tables/04_evaluation/sample_efficiency.csv`); J simply did not pick it. Across all 145 candidates, neither random search nor TuRBO ever found a configuration whose overlap rate beat the incumbent's at all.

**On the band the objective reads, overlap improved under every method.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`src/kpi/overlap.py::overlap_neighbors_per_band`):

*Table 8a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **2600 MHz** (serving band on 71 % of tiles) | 0.2010 | 0.1901 | 0.1788 | **0.1718** |
| 1800 MHz | 0.2067 | 0.2092 | 0.2064 | 0.2082 |
| 700 MHz | 0.2396 | 0.2456 | 0.2496 | 0.2440 |
| Band-collapsed (the reported KPI) | 0.3164 | 0.3247 | 0.3368 | 0.3402 |

Each search cleaned up the layer J scores — 2600 MHz falls from 20.1 % to 17.2 % under TuRBO — and let the two unscored layers drift flat or slightly worse. The collapsed rate still rose, mostly because former hole tiles joined the covered set at all and arrived with neighbours - 814 of them under the rule sweep, 677 under TuRBO and 312 under random search. **This is ADR 0009's stated negative consequence, now measured rather than predicted**: only the priority band is scored, so the lower layers are free to degrade.

*Table 8b. Coverage class by area and by demand (demand weighted by the current configuration's peak PRB demand). Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 11.2 % / 0.0 % | 10.4 % / 0.0 % | 10.9 % / 0.4 % | 10.6 % / 0.0 % |
| Weak | 30.7 % / 83.6 % | 25.4 % / 77.0 % | 30.6 % / 80.4 % | 26.4 % / 77.8 % |
| Good | 58.1 % / 16.4 % | 64.2 % / 23.0 % | 58.4 % / 19.3 % | 63.0 % / 22.2 % |

*Table 9. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | 0.96 | 64.4 % | 17.6 % |
| Rule-based sweep | 1.12 | 63.7 % | 17.9 % |
| Random search | 1.05 | 62.2 % | 16.5 % |
| TuRBO | 1.06 | 62.0 % | **16.0 %** |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand in holes.** TuRBO and the rule sweep move **no** peak demand onto hole tiles; random search moves 0.4 %. The share of demand on weak tiles falls under every method, most under the rule sweep (83.6 % to 77.0 %), and the share on good tiles rises from 16.4 % to 23.0 %. On the demand-weighted view the rule sweep is the strongest of the three, which is the same disagreement Section 6.2 records.

**Pile-ups shrink even as the mean rises.** TuRBO cuts the share of covered tiles with three or more neighbours from 17.6 % to 16.0 %, the best of the three, while raising the mean neighbour count from 0.96 to 1.06. It is converting a few badly contested tiles into many mildly contested ones — which is what $\lambda e^{1-\lambda}$ asks for, since the penalty is steepest between λ = 1 and λ = 3 and nearly flat beyond λ = 4.

**No objective parameters to vary.** The previous version of this report noted that the pick's robustness to τ_R and β was untested. That concern is now closed by construction: the objective has no parameters (Section 3.4). What replaces it is the band-priority order, which *is* a judgement, and which Section 6.8 lists as a limitation.

### 6.5 Capacity impact

*Table 10. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 47.3 % | −0.38 | 5.13 | 53.2 | 34.1 % / 10.5 % / 8.1 % |
| Rule-based sweep | **40.7 %** | **+0.25** | 6.95 | 43.2 | 43.8 % / 8.2 % / 7.4 % |
| Random search | 42.4 % | +0.19 | 6.85 | 43.6 | 41.1 % / 9.5 % / 6.9 % |
| TuRBO | 42.1 % | −0.10 *(worse)* | **8.62** | **36.4** | 33.1 % / 17.2 % / 7.6 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

**Service.** Every method serves substantially more UEs: 4.9 to 6.6 points. The rule sweep serves the most (59.3 % against 52.7 %), TuRBO and random search about 57.9 %. Median served SINR rises under all three, and the PRBs a served UE needs fall by 18 to 32 %.

**TuRBO does something the baselines do not: it offloads to 1800 MHz.** The two baselines push traffic onto the preferred 2600 MHz layer in the usual way — 34.1 % of reports to 41–44 %. TuRBO moves it the other way, dropping 2600 MHz to 33.1 % while lifting **1800 MHz from 10.5 % to 17.2 %**. The cell-impact table shows the mechanism: TuRBO *downtilts* several 2600 MHz sectors hard toward the 15° bound (n0c2 to 14.95°, n1c1 to 14.75°, n1c0 to 14.52°), shrinking those footprints — n1c1's 2600 MHz layer loses 495 served reports — while uptilting 1800 MHz almost to 0° across the network (n0c0 to 0.74°, n0c2 to 0.84°, n1c2 to 1.00°).

That redistribution is why TuRBO has the best median served SINR (8.62 dB against the incumbent's 5.13) and needs the fewest PRBs per UE (36.4 against 53.2): traffic moved off the most contended layer. It is also why its **10th-percentile served SINR is the only one to get worse** (−0.38 to −0.10 dB is a small improvement; it is the one method that does not clear zero) — the UEs at the bottom of the distribution are the ones the shrunken 2600 MHz footprints dropped.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- Area covered by 2600 MHz rises 71.2 % → 72.4 %, 1800 MHz 76.5 % → 78.0 %, 700 MHz 86.2 % → 86.7 %.
- Mean RSRP where covered improves on every layer: 2600 MHz −97.2 → −92.9 dBm, 1800 MHz −91.5 → −88.2 dBm, 700 MHz −84.2 → −82.2 dBm. No band loses mean signal.
- Median served SINR on 2600 MHz rises 3.5 → 6.1 dB, and on 1800 MHz 5.2 → 9.9 dB.

About 42 % of UE reports remain unserved. Much of that is out of reach: 17.5 % of UE positions have no path to any cell (Table 2) and 21.2 % stand on hole tiles. The rest is PRB exhaustion — peak utilisation sits at the 0.8 admission ceiling in every configuration (`tables/04_evaluation/cell_impact.csv`), so the 2600 MHz layer remains the binding constraint on service.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 11. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 4.90 | 9.63 | −3.21 |
| 1800 MHz | 12 / 12 | 6.04 | 9.26 | −6.04 |
| 700 MHz | 12 / 12 | 4.62 | 6.92 | −4.62 |

Every one of the 36 cell-bands moved. The configuration is a net uptilt overall, but it is **not** uniform, and the structure is the point:

- **1800 MHz and 700 MHz are uptilted everywhere** — all 12 cells on each, with mean \|Δ\| equal to mean Δ, so not one cell on either band was downtilted. The 1800 MHz layer is pushed hardest, averaging 6.04° of uptilt and landing several cells below 1°.
- **2600 MHz is mixed**: mean \|Δ\| 4.90° against mean Δ −3.21°, so a minority of cells were downtilted while most were uptilted. The five downtilted cell-bands are all on 2600 MHz, three of them within 0.5° of the 15° bound.
- Proposed tilts span **0.74° to 14.95°**, so the [0°, 15°] box is binding at both ends.

This reads as the search deliberately re-dividing the layers: pull a few 2600 MHz sectors in tight to stop them contending with each other, and open 1800 MHz out to pick up the traffic and the ground they give away. That is a strategy a per-band sweep cannot express, and it is where TuRBO's J margin comes from.

The rule sweep instead sets every cell on a band to one value: **7.50° on 2600 MHz** (Δ = −4.50°), **0.00° on 1800 MHz** (Δ = −10.00°) and **0.00° on 700 MHz** (Δ = −8.00°) (`outputs/tilt_change_rule.csv`) — a uniform, and very large, uptilt that drives both lower layers to the bottom of the box. With one seed it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended.

The largest traffic shifts are on 2600 MHz sectors (`tables/04_evaluation/cell_impact.csv`): n1c1 −495 served reports, n1c0 −169, n0c2 −80.

### 6.7 Cost

*Table 12. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 28 | 11 | 1.07 | 1.85 | 2.3 | +0.0109 | **0.0059** |
| Random search | 145 | 62 | 7.49 | 10.22 | 3.1 | +0.0075 | 0.0007 |
| TuRBO | 145 | 144 | 9.01 | 19.59 | 3.7 | **+0.0151** | 0.0008 |

TuRBO's GP fitting and acquisition added about 10.6 minutes of wall clock on top of its ray tracing, slightly more than doubling it. Ray-tracing cost per evaluation is higher for TuRBO than random search (3.7 vs 3.1 s) because its proposals reach further into the box, where more rays survive.

**By J gain per minute the rule sweep is roughly eight times as cost-effective as TuRBO**, and it reaches a J only 0.0042 lower for a fifteenth of the wall clock. Given that it also beats TuRBO on 8 of the 11 reported KPIs (Section 6.2), the sweep is the stronger practical result on this scenario; TuRBO's case rests on the objective, on the headroom its late-budget improvement implies, and on the per-cell structure a sweep cannot produce.

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed, and the TuRBO–rule margin (+0.0042 on J) in particular may lie within seed-to-seed variation.
3. **Winner's curse.** Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
4. **The objective scores only the priority band.** J reads the most preferred band above −120 dBm and ignores the others, so the 1800 and 700 MHz layers are unpriced. Table 8a shows them drifting worse while 2600 MHz improves. Twelve of the 36 decision variables act on a small slice of the map.
5. **The objective ignores signal strength above the threshold.** A tile at −119 dBm with one server scores the same as one at −70 dBm. The rule sweep's larger RSRP and SINR gains earn it nothing in J, which is most of why J and the KPIs disagree here.
6. **The band priority is a judgement.** 2600 → 1800 → 700 MHz is asserted in `kpi.capacity.band_preference`, it decides which layer every tile is scored on, and no alternative order was tested.
7. **The capacity model is a simplification.** It drives the served ratio and every capacity figure. It uses a Shannon rate with no MCS cap, full-load co-band interference against partial PRB load, and no receiver noise figure.
8. **Tilts at both bounds.** The winner places cell-bands at 0.74° and 14.95°, and the rule sweep drives two whole bands to 0.00°, so [0°, 15°] is binding at both ends and the box may be too narrow.
9. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability with earlier runs.** No J in this report is comparable with any figure recorded before 2026-09-20. The objective was replaced (ADR 0009) and its range changed from unbounded to [0, 1]. Every run traced under the previous objectives was deleted rather than archived; `src/evaluation/runs.py` would now refuse to pool them, because the deleted `kpi.objective` config block is part of the KPI definition it compares.

## 7. Conclusions and Recommendations

*Table 13. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0109 (2nd) | +0.0075 (3rd) | **+0.0151 (1st)** |
| 2. Reported KPIs | **9 better, 3 worse**; best on 8 of 11, including every coverage and service measure | 8 better, 4 worse | **9 better, 3 worse**; best effective coverage and fewest pile-ups |
| 3. Search effectiveness | Best at evaluation 11 of 28; ahead of TuRBO up to 50 evaluations | Best at 62, no gain in the next 83 | **Median candidate above random's best; still improving at 144 of 145** |
| 4. Robustness | **No demand onto holes**; largest demand shift out of weak coverage | 0.4 % of demand onto holes | **No demand onto holes**; fewest 3+ pile-ups |
| 5. Cost | **1.9 min, 28 evaluations; 8× the J per minute** | 10.2 min, 145 evaluations | 19.6 min, 145 evaluations |

**Conclusions.**

- **TuRBO reached the highest J**, and there is good evidence the model earned it: its median candidate scored above random search's best, and its own proposals averaged 0.8176 against the shared Sobol design's 0.8019.
- **But the rule-based sweep is the stronger practical result on this scenario.** It beats TuRBO on 8 of the 11 reported KPIs — hole rate, weak rate, both RSRP percentiles, both SINR percentiles, served rate and peak PRB utilisation — for a fifteenth of the wall clock, and reaches a J only 0.0042 lower. The honest summary is that **the objective and the KPI set disagree about the winner**, and the disagreement is structural: J does not read signal strength above the threshold, which is where most of the sweep's advantage lies.
- **The coverage-versus-overlap trade is real and the objective takes a side.** All three methods close holes and raise the band-collapsed overlap rate, because one closed hole is worth 3.78 newly crowded tiles. Overlap-reducing candidates existed — the sweep evaluated one at 0.3067 — and J did not select them.
- **On the band it actually scores, the objective works.** 2600 MHz overlap falls under all three methods, most under TuRBO (20.1 % → 17.2 %), and effective coverage rises from 66.1 % to 69.6 %. The unscored 1800 and 700 MHz layers drift slightly worse, exactly as ADR 0009 predicted.
- **TuRBO found a multi-band strategy the baselines did not**: shrink a few 2600 MHz sectors, open 1800 MHz wide, and move 6.7 points of traffic onto the mid band. It gives the best median served SINR and the lowest PRBs per served UE of the three.
- **Every result** depends on one scenario, one seed, an objective that prices only the priority band, and a simplified capacity model.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Decide whether the objective should price the non-serving layers.** This is the single most consequential open question. Table 8a shows 1800 and 700 MHz drifting worse while J is indifferent; a per-band term would price that, at the cost of the property that makes J interpretable.
3. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`), and re-trace the shortlists under other solver seeds. The TuRBO–rule margin of +0.0042 needs an interval before it can be called a margin at all.
4. **Give TuRBO a larger budget.** It found its best at evaluation 144 of 145 and never restarted, so the trust region was still productive when the budget ended.
5. **Widen the tilt box.** The winner sits at 0.74° and 14.95°, and the rule sweep drives two bands to 0.00°; [0°, 15°] is binding at both ends.
6. **Test an alternative band priority.** The 2600 → 1800 → 700 order decides which layer every tile is scored on, and TuRBO's own answer was to move traffic *to* 1800 MHz — which the objective's priority order does not reward.
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
| Objective | no parameters; reads `hole_dbm`, `overlap_margin_db` and the band preference (ADR 0009) |
| Capacity | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission ceiling 0.8, 20 Mbps per UE, SCS 15 kHz, admission in report-time order |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12; rule 5 steps × 2 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-20_04-09-52/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-20_04-20-14/` |
| TuRBO | `outputs/optim/turbo/2026-09-20_03-50-06/` |

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
| n0c0 | 5.17 (-6.83) | 0.74 (-9.26) | 5.72 (-2.28) |
| n0c1 | 5.52 (-6.48) | 1.85 (-8.15) | 2.51 (-5.49) |
| n0c2 | 14.95 (+2.95) | 0.84 (-9.16) | 3.23 (-4.77) |
| n1c0 | 14.52 (+2.52) | 5.18 (-4.82) | 3.44 (-4.56) |
| n1c1 | 14.75 (+2.75) | 1.70 (-8.30) | 6.50 (-1.50) |
| n1c2 | 2.37 (-9.63) | 1.00 (-9.00) | 1.86 (-6.14) |
| n2c0 | 7.79 (-4.21) | 5.03 (-4.97) | 3.61 (-4.39) |
| n2c1 | 13.36 (+1.36) | 4.78 (-5.22) | 1.81 (-6.19) |
| n2c2 | 3.79 (-8.21) | 9.91 (-0.09) | 4.81 (-3.19) |
| n3c0 | 12.54 (+0.54) | 2.64 (-7.36) | 1.09 (-6.91) |
| n3c1 | 3.55 (-8.45) | 8.66 (-1.34) | 4.93 (-3.07) |
| n3c2 | 7.18 (-4.82) | 5.16 (-4.84) | 1.08 (-6.92) |

Every 1800 MHz and 700 MHz cell is uptilted; the only downtilts are on 2600 MHz, three of them within 0.5° of the 15° bound.

The rule-based sweep's best configuration sets every cell to 7.50° on 2600 MHz, 0.00° on 1800 MHz and 0.00° on 700 MHz (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[5] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[6] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[7] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
