# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every figure and table below comes from the notebook pipeline run of 2026-09-16 (`notebooks/00_simulation` through `04_evaluation`), using the working-tree configuration: baseline tilt 10° for every cell and band, quality-index weights 3 / 2 / 1. Paths are relative to `reports/`.*

---

## Abstract

This study asked whether network-wide search over antenna tilts could improve the coverage of a multi-band urban cell layout when every band on every cell is tuned jointly, not one band at a time. The study area was the bundled Sionna-RT Munich scene: three nodes, nine sectors and three bands (700, 1800 and 2600 MHz), giving 27 absolute-tilt decision variables in [0°, 15°]. A week-long, time-varying UE population produced 9,202 synthetic MDT reports. Every candidate configuration was ray-traced and scored on four reported KPIs (hole rate, co-band overlap rate, served UE ratio and weak-coverage rate) and on a weighted geometric-mean quality index *D* over three soft desirabilities. Three searches were compared from the same incumbent (*D* = 0.8499): a rule-based per-band sweep, Sobol random search and TuRBO-1 Bayesian optimization, the last two with matched budgets of 145 evaluations. TuRBO scored highest (*D* = 0.8610, +0.0111). It improved all five reported measures, including served UE ratio (0.907 → 0.918) and weak-coverage rate (0.174 → 0.144). The median TuRBO candidate beat the best candidate of either baseline. The rule-based sweep reached *D* = 0.8574 in 28 evaluations but worsened overlap. Every result comes from one scenario and one search seed, the capacity model uses placeholder values, and TuRBO's recommendation put 2.2 % of demand into coverage holes where the incumbent put none. The TuRBO configuration is therefore recommended as a candidate for held-out and multi-seed validation, not for deployment.

---

## 1. Introduction

A 5G/6G site commonly radiates several frequency bands from the same mast. The bands differ physically. Low bands such as 700 MHz propagate farther and penetrate buildings better. High bands such as 2600 MHz carry more capacity over a smaller footprint. A mid band such as 1800 MHz bridges the two. A well-tuned network gives each layer the role its propagation suits: the low band provides the coverage floor and reaches the cell edge, the high band concentrates capacity near the node, and the mid band fills in between.

In practice, antenna tilt is often set per band, from a static planning value, and adjusted by hand. When each layer is tilted without regard to the others, two failures become likely. First, several bands may cover the same near-node area strongly, which wastes radio resources and raises co-channel interference. Second, the cell edge may develop a coverage hole where the high band has faded before the low band takes over. Manual, band-by-band tuning is slow and can miss both interactions.

This project treats the tilts of every (cell, band) pair as one coordinated optimization problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) scores every proposed configuration. Three search strategies of increasing sophistication are compared on the same objective: an operator-style rule, random search and trust-region Bayesian optimization (TuRBO [2]). The project also planned a Multi-Agent Reinforcement Learning arm, but it has not been implemented, so this report does not evaluate it.

## 2. Problem Definition

**Network.** The study area is the Sionna-RT Munich scene, rasterised into a 61 × 74 grid of 20 m tiles (4,514 tiles). Three nodes, each 25 m high with three sectors (azimuths 30°, 150°, 270°), carry three bands each: nine cells and 27 cell-band pairs (Table 1, Table 2, Figure 1).

**Decision variable.** For N = 9 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{27}$$

It reports each result as an offset from the incumbent, where every tilt is 10° (`tables/02_preprocessing/decision_variables.csv`).

**Goal.** The goal is a configuration that:
- reduces coverage holes (tiles whose best-server RSRP is ≤ −120 dBm);
- reduces redundant co-band overlap (a neighbour within 6 dB of the serving cell);
- raises the share of UEs a cell-band can admit under its PRB limit;
- preserves each band's physical role.

**Starting condition.** At the incumbent tilts, 17.7 % of the area is a coverage hole and 17.4 % is weakly covered. No demand falls in the holes, but 76.1 % of peak PRB demand sits on weakly covered tiles (Table 3). Each layer alone leaves 20.6 % (700 MHz) to 27.7 % (2600 MHz) of tiles in a hole (`tables/01_eda/coverage_classes_per_band.csv`). The layers therefore complement each other only partly, which is the gap coordinated tilting targets. The serving rule puts 76 % of UE reports on 2600 MHz and 3 % on 700 MHz, and blocks 9.4 % of reports on PRB limits (`tables/01_eda/serving_band_mix.csv`, `tables/01_eda/data_quality.csv`).

![Study area](figures/00_simulation/study_area.png)

*Figure 1. Study area: nine cells on three nodes (red) and a sample of 300 UE positions (green) over the Munich scene. Source: `figures/00_simulation/study_area.png`.*

![RSRP per band](figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the incumbent tilts. Source: `figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](figures/01_eda/demand_vs_coverage.png)

*Figure 3. Peak PRB demand beside signal strength at the incumbent tilts. Source: `figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. Source: [`tables/00_simulation/frequency_bands.csv`](tables/00_simulation/frequency_bands.csv).*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per cell |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

*Table 2. Scenario. Source: [`tables/00_simulation/scenario_summary.csv`](tables/00_simulation/scenario_summary.csv), [`tables/00_simulation/mdt_summary.csv`](tables/00_simulation/mdt_summary.csv).*

| Property | Value |
|---|---|
| Scenario ID | `scn_28d06a5bfb4c02e6` |
| Grid | 61 × 74 tiles, 20 m |
| Time intervals | 672 × 15 min (7 days) |
| Demand hotspots | 4 |
| UE positions drawn | 10,066 |
| MDT reports (UEs reached by a cell) | 9,202 |
| UEs dropped, no cell reached | 864 |
| Measurement error σ | 4 dB |

*Table 3. Coverage class by area and by demand at the incumbent tilts. Source: [`tables/01_eda/coverage_by_area_and_demand.csv`](tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of peak PRB demand |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 798 | 17.7 % | 0.0 % |
| Weak (−120 to −90 dBm) | 785 | 17.4 % | 76.1 % |
| Good (> −90 dBm) | 2,931 | 64.9 % | 23.9 % |

## 3. Proposed Solutions

All three solutions search the same bounded tilt box (`src/optim/space.py`) and use the same Sionna-RT evaluator with a fixed solver seed (`src/optim/evaluator.py`). They also share one definition of "better": the quality index *D* (Section 4). They differ only in where they look. Each run first evaluates the incumbent, then searches, then publishes a shortlist of the eight highest-*D* configurations, including the incumbent (`src/optim/run.py`, `src/optim/report.py`).

### 3.1 Rule-based per-band sweep

This solution mimics the heuristic an operator would use. Every cell on a band shares one tilt, which collapses the 27 dimensions to three. Coordinate descent then passes over the bands: it tries `n_steps = 5` evenly spaced tilts across the band's range, keeps the best, and moves to the next band, for `n_rounds = 2` passes. It is deterministic and needs at most 3 × 5 × 2 evaluations. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 quasi-random points from a seeded Sobol sequence over the full 27-dimensional box. The first 16 points are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, any gap between the two measures what the surrogate model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found so far. Each round, it fits a Gaussian process to the evaluations since the last restart and uses Thompson sampling to draw a batch of three candidates inside the region. The region expands after three consecutive successes, shrinks after repeated failures, and restarts when its side falls below 0.5⁷ of the unit cube. The budget is 16 Sobol initial points plus 128 trust-region evaluations. The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision record: `docs/adr/0003-turbo-on-a-weighted-kpi-score.md`.

#### 3.3.1 The objective every solution maximises

A weighted sum of KPIs is compensatory: a large gain on one KPI can pay for a collapse on another. The project therefore scores candidates with a desirability index [4] (ADR 0004, ADR 0005, `src/optim/objective.py::quality_index`). Each targeted KPI is softened per tile before aggregation, around the same threshold its reported rate uses:

| Desirability | Per-tile form | Weight |
|---|---|---:|
| Coverage, `hole_desirability` | sigmoid(R_max − (−120 dBm)) | 3 |
| Layer separation, `overlap_desirability` | mean over co-band neighbours of sigmoid(Δ − 6 dB) | 2 |
| Served, `served_desirability` | the tile's served share, averaged over occupied tiles | 1 |

The index is their weighted geometric mean, $D = \prod_k d_k^{w_k}$ with weights normalised to sum to 1. *D* lies in [0, 1], and one collapsed term pulls the whole index towards zero.

## 4. Criteria for Assessing Solutions

The solutions were judged against the following criteria. Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The best *D* found, as a change from the incumbent. This is the objective every method optimizes.
2. **Reported KPIs.** The direction of change against the incumbent for the four ADR 0001 rates, plus 5th-percentile cell-edge RSRP [5]:
   - coverage hole rate ↓
   - co-band overlap rate ↓
   - served UE ratio ↑
   - weak-coverage rate ↓
   - cell-edge RSRP ↑

   Following ADR 0005, a change is labelled only as better or worse. No tie band is applied, because solver noise per KPI has not been measured.
3. **Search effectiveness.** Whether the search itself earned the gain: the best score against the median candidate evaluated, and TuRBO paired with random search on the same seed.
4. **Robustness.** Whether the ranking survives alternative weightings (equal weights and each desirability alone). Also, where the configuration moves demand, not only area, into or out of holes.
5. **Cost.** Ray-tracing evaluations, ray-tracing minutes and wall-clock minutes per run.

## 5. Research Methodology

**Data generation.** No operator data was available, so all data was produced synthetically by the notebook pipeline (`notebooks/00_simulation.ipynb`, `src/simulation/`):

1. **Scenario.** The scene was rasterised onto the 20 m grid. A UE population was then drawn every 15 minutes for 7 days. Each draw mixed 90 % hotspot demand (four Gaussian hotspots near built-up volume) with 10 % uniform background, modulated by a diurnal profile and AR(1) noise (`configs/simulation.yaml` `time`, `density`).
2. **Radio map.** Each band was ray-traced separately with 10⁷ rays per transmitter, a maximum depth of 8, line of sight, specular reflection and refraction, 8 × 8 cross-polarised TR 38.901 panels [6] and a 4.85 dBm reference-signal power. The output was per-cell RSRP and SINR on the grid.
3. **Synthetic MDT.** The radio map was sampled at each UE position with 4 dB Gaussian measurement error added. UEs that no cell reached were dropped.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 20 contract checks: grid, scenario ID, band and cell order, bounds, duplicate keys and baseline tilts (`tables/02_preprocessing/verification_checks.csv`). It then wrote typed Parquet tables without dropping or altering a row (`tables/02_preprocessing/preprocessing_audit.csv`). `notebooks/01_eda.ipynb` recorded data-quality measures without removing anything.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42. Before the run, earlier runs made under a different configuration were moved to `outputs/optim_archive/2026-09-16_pre-rerun/`, so none were reused. Random search and TuRBO each spent 145 evaluations (incumbent + 16 + 128). The rule sweep stopped at 28 evaluations. Every candidate was fully ray-traced and scored; no surrogate prediction entered a reported number. Each run wrote its history, best tilt, best radio map and `run.json` under `outputs/optim/<method>/<timestamp>/`, and was logged to MLflow (`mlflow.db`).

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It performs these steps:

1. Checks that all runs share the scenario, grid, solver settings, bands and KPI definition (21 checks).
2. Recomputes each archived best configuration's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the incumbent.
4. Compares each winner with the distribution of candidates.
5. Re-ranks methods under four alternative weightings.
6. Maps coverage and demand change, the serving-band mix, cell utilisation and tilt movement.
7. Records cost and convergence.

**Relevance, criteria and practicality.** Ray tracing on a real city geometry was chosen over a statistical path-loss model because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away. Budgets were matched between random search and TuRBO so that criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost. The runs used a single RTX 3050 (4 GB). Budget-matched random search and TuRBO runs took about 9 and 15 minutes of wall-clock time, which made one seed per method practical within this study.

## 6. Analysis and Interpretation

### 6.1 Comparability and correctness

All 21 comparability checks held, and each KPI recomputed from the archived radio maps matched the recorded value. Gaps were at most 3.5 × 10⁻⁶ dB for cell-edge RSRP and 0 or below 2 × 10⁻⁹ for rates and desirabilities (`tables/04_evaluation/comparability_checks.csv`, `tables/04_evaluation/kpi_reproducibility.csv`). The differences below therefore appear to come from the configurations rather than from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the incumbent (seed 42). Arrows show the better direction. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Incumbent | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| Quality index *D* ↑ | 0.8499 | 0.8574 | 0.8562 | **0.8610** |
| Coverage hole rate ↓ | 0.1768 | **0.1719** | 0.1755 | 0.1739 |
| Co-band overlap rate ↓ | 0.3212 | 0.3314 *(worse)* | 0.3186 | **0.3161** |
| Served UE ratio ↑ | 0.9065 | 0.9100 | 0.9115 | **0.9181** |
| Weak-coverage rate ↓ | 0.1739 | **0.1422** | 0.1549 | 0.1436 |
| Cell-edge RSRP [dBm] ↑ | −105.92 | **−103.96** | −104.70 | −104.18 |
| Reported and targeted KPIs improved / worsened | — | 7 / 1 | 8 / 0 | 8 / 0 |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Change of every KPI against the incumbent, oriented so that higher is better. Source: `figures/04_evaluation/kpi_improvement.png`.*

All three methods improved *D*, and TuRBO improved it most (+0.0111, against +0.0076 for the rule sweep and +0.0063 for random search). TuRBO and random search improved every reported KPI. The rule sweep closed the most area holes and lifted the cell edge most, but raised co-band overlap by about one percentage point. This seems to be the expected cost of one shared tilt per band: pulling every cell up to 3.75° extends all footprints together, so neighbours meet more often (`outputs/tilt_change_rule.csv`).

The absolute gains are small. The hole rate moved by 0.3 to 0.5 percentage points of area. It therefore appears that, in this layout, most of the 17.7 % hole area lies in building shadow that tilt alone cannot reach. The largest relative movements were in weak coverage (−3 points) and served ratio (+1.2 points for TuRBO).

### 6.3 Did the search matter?

*Table 5. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Incumbent | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.8499 | 0.8516 | 0.8514 | 0.8543 | 0.8562 |
| Rule-based sweep | 0.8499 | — | 0.8528 | 0.8568 | 0.8574 |
| TuRBO | 0.8499 | 0.8516 | **0.8589** | **0.8607** | **0.8610** |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best quality index found so far against evaluations. Source: `figures/04_evaluation/search_progress.png` (also `figures/03b_turbo/search_progress.png`).*

TuRBO's median candidate (0.8589) scored above the single best candidate of either baseline. This is perhaps the strongest evidence here that the trust-region model, and not a lucky draw, produced the gain: the two runs share the same 16 initial points, and they diverge only once the model starts proposing candidates. Random search found its best point at evaluation 18 and made no progress in the remaining 127. TuRBO was still improving at evaluation 140, so a larger budget might yield more. The paired gain of TuRBO over random search was +0.0048 on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test could be computed, so the margin cannot be separated statistically from seed-to-seed variation.

### 6.4 Is the result robust?

*Table 6. Best score and rank under alternative weightings. Source: [`tables/04_evaluation/weight_sensitivity.csv`](tables/04_evaluation/weight_sensitivity.csv).*

| Weighting | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|
| Configured (3/2/1) | 0.8574 (2) | 0.8562 (3) | **0.8610 (1)** |
| Equal | 0.8647 (2) | 0.8632 (3) | **0.8696 (1)** |
| Coverage only | **0.8289 (1)** | 0.8283 (2) | 0.8279 (3) |
| Layer separation only | 0.8988 (3) | 0.9004 (2) | **0.9085 (1)** |
| Served only | 0.8711 (2) | 0.8702 (3) | **0.8774 (1)** |

TuRBO ranked first under four of the five weightings. Under coverage-only weighting the order reversed, and the three methods fell within 0.001 of one another. TuRBO's lead therefore seems to come from layer separation and service, not from coverage. The "share with unchanged best configuration" is 0 for every alternative weighting. In other words, re-weighting would select a different configuration from each run's archive, so the specific recommended tilt set depends on the configured weights.

*Table 7. Coverage class by area and by demand. Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Incumbent area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 17.7 % / 0.0 % | 17.2 % / 1.2 % | 17.5 % / 0.2 % | 17.4 % / **2.2 %** |
| Weak | 17.4 % / 76.0 % | 14.2 % / 72.2 % | 15.5 % / 75.5 % | 14.4 % / 70.2 % |
| Good | 64.9 % / 24.0 % | 68.6 % / 26.6 % | 67.0 % / 24.3 % | 68.3 % / 27.6 % |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 6. Best-server RSRP before and after TuRBO, and the holes it closed (18 tiles) or opened (5 tiles). Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 7. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

Table 7 raises the most important caution in this report. All three methods reduced hole area, but each moved some demand into holes, and TuRBO moved the most (2.2 % of peak PRB demand, from 0 %). TuRBO closed 18 hole tiles and opened 5 (Figure 6). The opened tiles appear to carry traffic, while several of the closed tiles appear to carry none. This is almost certainly a consequence of the coverage terms being weighted by area, not demand (ADR 0001, "two notions of where the map matters"). The served ratio partly offsets it, because TuRBO's net served UE ratio still rose. The data cannot show whether that trade would be acceptable to users in the new holes.

### 6.5 Capacity impact

*Table 8. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Incumbent | 9.35 % | −0.59 | 5.97 | 48.1 | 75.7 % / 11.9 % / 3.1 % |
| Rule-based sweep | 9.00 % | −0.38 | 5.97 | 48.1 | 76.7 % / 11.8 % / 2.5 % |
| Random search | 8.85 % | −0.34 | 6.12 | 47.3 | 76.4 % / 12.0 % / 2.8 % |
| TuRBO | **8.19 %** | **−0.15** | **7.12** | **42.4** | 77.8 % / 11.4 % / 2.6 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 8. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 9. Peak PRB utilisation per cell-band. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

TuRBO's configuration raised the median served SINR by about 1.1 dB and cut the median PRBs a served UE needs by about 12 %. This appears consistent with its lower co-band overlap: less interference means higher spectral efficiency per UE. The serving mix shifted slightly further towards 2600 MHz and away from 700 MHz under every method. The low band was therefore not pushed further into a coverage-floor role. The capacity model uses placeholder PRB, throughput and SCS values (`configs/kpi.yaml` `capacity`), so these figures should be read as directions of change, not as absolute capacity.

### 6.6 Recommended tilt changes

![Tilt movement](figures/04_evaluation/tilt_movement.png)

*Figure 10. Tilt changes in the recommended (TuRBO) configuration. Source: `figures/04_evaluation/tilt_movement.png`.*

*Table 9. Tilt movement summary for the recommended configuration. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 9 / 9 | 4.91 | 9.79 | −3.83 |
| 1800 MHz | 9 / 9 | 5.24 | 8.97 | −4.07 |
| 700 MHz | 9 / 9 | 5.65 | 9.74 | −4.86 |

The recommended configuration moved all 27 tilts, on average by about 5°, and mostly upward (less downtilt). This suggests that the 10° starting downtilt was too steep for a 500 m inter-site distance with 25 m masts. The 700 MHz layer moved furthest up on average, consistent with a low band extending towards the cell edge. The individual moves are large and not uniform: n0c2 on 2600 MHz and n2c2 on 700 MHz tilted down while their co-sited layers tilted up. Some of this dispersion may reflect the solver noise the search exploited, not physics (the winner's-curse risk noted in ADR 0003). Per-cell served-report, utilisation and SINR changes are in `tables/04_evaluation/cell_impact.csv`.

### 6.7 Cost

*Table 10. Search cost. Source: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | *D* gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|
| Rule-based sweep | 28 | 12 | 1.14 | 1.45 | 0.0052 |
| Random search | 145 | 18 | 7.31 | 8.88 | 0.0007 |
| TuRBO | 145 | 140 | 9.50 | 15.48 | 0.0007 |

TuRBO's model fitting added about six minutes of wall-clock time over its ray tracing. Measured by gain per minute, the rule sweep was clearly the most cost-effective, and TuRBO bought its extra 0.0036 of *D* at roughly ten times the time. At about 4 s per ray-traced candidate on this GPU, none of these costs seems prohibitive for an offline planning study.

### 6.8 Limitations

These results should be read tentatively, for five reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence intervals or significance tests could be computed, and the TuRBO-versus-random margin could lie within seed-to-seed variation.
3. **Solver noise is unmeasured.** Every candidate used the same ray-tracer seed, so selecting the maximum of 145 candidates may favour configurations that benefit from that seed's Monte-Carlo noise.
4. **The capacity model is a placeholder.** It drives both the served ratio in the objective and every capacity figure.
5. **The Multi-Agent RL arm and held-out validation were not built.** The planned comparison against RL therefore could not be made.

Running several search seeds, repeating the solve of the best configurations under different solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

## 7. Conclusions and Recommendations

*Table 11. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Overall quality *D* | +0.0076 (2nd) | +0.0063 (3rd) | **+0.0111 (1st)** |
| 2. Reported KPIs | 4 of 5 better; overlap worse | 5 of 5 better | **5 of 5 better**; best overlap and served ratio |
| 3. Search effectiveness | Plateaued after 12 evaluations | Plateaued after 18 evaluations | **Median candidate beat both baselines' best**; still improving at 140 |
| 4. Robustness | First on coverage-only weighting; 1.2 % of demand into holes | Least demand into holes (0.2 %) | First under 4 of 5 weightings; **most demand into holes (2.2 %)** |
| 5. Cost | **1.5 min, 28 evaluations** | 8.9 min, 145 evaluations | 15.5 min, 145 evaluations |

**Conclusions.**

- **TuRBO** was the most effective of the three on the objective it was given. It improved every reported KPI, reduced co-band overlap and raised served SINR. The comparison of its median candidate with the baselines' best points suggests the gain came from the model-guided search.
- **The rule-based sweep**, though it cannot coordinate neighbouring cells, was by far the most cost-effective. It improved coverage and cell-edge RSRP the most, at the price of more overlap.
- **Random search**, at TuRBO's budget, was dominated by TuRBO on almost every criterion. Its only advantage was that it moved the least demand into holes.
- **Every method's gains were modest** in absolute terms. All of them depend on one scenario, one seed and a placeholder capacity model.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** In particular, TuRBO's configuration moves 2.2 % of peak demand into new coverage holes, and no result has been validated beyond the scenario it was tuned on.
2. **Carry TuRBO forward as the preferred optimizer, and use the rule-based sweep as a cheap warm start.** Its 3.75° all-band setting could seed TuRBO's initial design.
3. **Weight the coverage term by demand, not only by area** (or add a demand-in-holes constraint) before the next run, so the objective penalises the failure identified in Section 6.4.
4. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS`) and measure solver-seed noise on the shortlisted configurations, to put confidence intervals on the margins above.
5. **Replace the placeholder `kpi.capacity` values with operator figures**, then build held-out scenario validation and the planned MARL arm before drawing a method-level conclusion.

---

## Appendices

### Appendix A. Configuration and reproduction

The run used the working-tree configuration of 2026-09-16, which differs from the last commit (`1549b0d`) in two places:

| Setting | Committed | Used in this run |
|---|---|---|
| `configs/simulation.yaml` baseline tilt, every cell and band | 5° | 10° |
| `configs/kpi.yaml` `weights` (coverage / separation / served) | 4 / 3 / 2 | 3 / 2 / 1 |

Unchanged settings: tilt bounds [0°, 15°], `hole_dbm = −120`, `weak_dbm = −90`, `overlap_margin_db = 6`, global seed 42, TuRBO and random budgets 16 + 128, TuRBO batch 3, rule sweep 5 steps × 2 rounds.

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-16_08-25-34/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-16_08-34-27/` |
| TuRBO | `outputs/optim/turbo/2026-09-16_08-36-09/` |

Earlier runs were archived under `outputs/optim_archive/2026-09-16_pre-rerun/`. To reproduce, run notebooks `00` through `04` in order, or run `task pipeline` from the Taskfile. Both call the same functions in `src/`.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`reports/tables/…`) | Figures (`reports/figures/…`) |
|---|---|---|
| 00 simulation | `frequency_bands`, `node_layout`, `scenario_summary`, `reach_per_band`, `mdt_summary` | `study_area`, `traffic_model`, `rsrp_per_band`, `mdt_rsrp_distribution` |
| 01 EDA | `dataset_overview`, `coverage_classes_per_band`, `coverage_by_area_and_demand`, `serving_band_mix`, `data_quality` | `band_propagation`, `demand_vs_coverage`, `cell_band_utilisation` |
| 02 preprocessing | `verification_checks`, `decision_variables`, `preprocessing_audit` | — |
| 03a baseline | `search_space`, `baseline_results` | — |
| 03b TuRBO | `turbo_results` | `search_progress` |
| 04 evaluation | `comparability_checks`, `kpi_scoreboard`, `kpi_reproducibility`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `weight_sensitivity`, `coverage_by_area_and_demand`, `ue_service_summary`, `recommended_tilt`, `tilt_movement_summary`, `cell_impact`, `method_cost`, `convergence` | `kpi_improvement`, `rsrp_change_maps`, `coverage_before_after`, `serving_band_mix`, `cell_band_utilisation`, `tilt_movement`, `search_progress` |

Deliverables per method are in `reports/outputs/`: `solutions_<method>.csv` (the shortlist with every KPI and its delta), `tilt_options_<method>.csv` and `tilt_change_<method>.csv` (the recommended row).

Two files in these folders were **not** regenerated by this run and are older artifacts: `tables/00_simulation/site_layout.csv` and `figures/04_evaluation/kpi_improvement_vs_tolerance.png`.

### Appendix C. Recommended tilt configuration (TuRBO)

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 10° for every entry; bounds are [0°, 15°].*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 4.40 (−5.60) | 6.46 (−3.54) | 3.81 (−6.19) |
| n0c1 | 0.21 (−9.79) | 1.46 (−8.54) | 1.75 (−8.25) |
| n0c2 | 14.85 (+4.85) | 11.77 (+1.77) | 7.66 (−2.34) |
| n1c0 | 6.01 (−3.99) | 5.37 (−4.63) | 8.30 (−1.70) |
| n1c1 | 9.64 (−0.36) | 6.48 (−3.52) | 4.31 (−5.69) |
| n1c2 | 3.57 (−6.43) | 2.96 (−7.04) | 3.68 (−6.32) |
| n2c0 | 4.90 (−5.10) | 4.35 (−5.65) | 2.94 (−7.06) |
| n2c1 | 6.08 (−3.92) | 13.49 (+3.49) | 0.26 (−9.74) |
| n2c2 | 5.88 (−4.12) | 1.03 (−8.97) | 13.57 (+3.57) |

The rule-based sweep's best configuration sets every cell on every band to 3.75° (Δ = −6.25°) (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] G. Derringer and R. Suich, "Simultaneous optimization of several response variables," *Journal of Quality Technology*, vol. 12, no. 4, 1980.

[5] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[6] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[7] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[8] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
