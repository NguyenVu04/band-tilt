# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-17: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage and load in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time. The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes (three sectors each, twelve cells) carrying three bands (700, 1800 and 2600 MHz), which gives 36 absolute-tilt decision variables in [0°, 15°], all starting at 12°. A week-long, time-varying population of 10,066 UE positions was drawn over the scene. The 4,145 UEs served at the current tilts form the MDT the search is scored on.

Candidates were ray-traced and scored on one objective, J = J_radio^γ · J_load^(1−γ) (ADR 0006). J_radio is a coverage utility discounted per overlapping co-band neighbour. J_load penalises the CVaR tail of PRB utilisation. Five KPIs were reported beside J: hole rate, co-band overlap rate, served UE ratio, weak-coverage rate and cell-edge RSRP.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

The last two had matched budgets of 145 evaluations. Their published winners were then re-traced and scored on every UE.

**Results.** TuRBO scored highest on J (0.7124 against 0.6818 currently, +4.5 %), but the rule sweep reached practically the same value (0.7122) in 28 evaluations. Random search reached 0.7055. All three methods improved six of the seven measures: hole rate by 9–11 %, weak coverage by 30–36 % and served ratio by 23–25 % (relative). All three worsened co-band overlap, by 1.5–4.5 %. The gain is dominated by a broad uptilt of about 6–8° that every method found.

**Caveats.** The results come from one scenario, one search seed per method, and a placeholder capacity model. No configuration is recommended for deployment. TuRBO and the rule sweep are both carried forward to a multi-seed comparison.

---

## 1. Introduction

A 5G/6G site commonly radiates several frequency bands from the same mast, and the bands differ physically:
- Low bands such as 700 MHz propagate farther and penetrate buildings better.
- High bands such as 2600 MHz carry more capacity over a smaller footprint.
- A mid band such as 1800 MHz sits between the two.

A well-tuned network gives each layer the role its propagation suits.

In practice, antenna tilt is often set per band from a static planning value and adjusted by hand. When each layer is tilted without regard to the others, two failures become likely. Several bands may cover the same area strongly, which wastes resources and raises co-channel interference. And the cell edge may develop coverage holes where no layer reaches. Load adds a third failure: a tilt that pulls traffic onto a cell already at its PRB limit blocks users even where the signal is good.

This project treats the tilts of every (cell, band) pair as one coordinated optimization problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) scores every proposed configuration. Three searches of increasing sophistication are compared on the same objective: an operator-style rule, random search and trust-region Bayesian optimization (TuRBO [2]). The project also plans a Multi-Agent Reinforcement Learning arm. It is not implemented, so this report does not evaluate it.

## 2. Problem Definition

**Network.** The study area is a local city scene (`data/external/scene/scene.xml`), rasterised into a 326 × 310 grid of 20 m tiles: 6,200 × 6,520 m, or 101,060 tiles.
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side, the RMa inter-site distance of 3GPP TR 38.901 Table 7.2-1 [6].
- **Cells.** Each node has three sectors at azimuths 45°, 165° and 285°, on 20 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve cells and 36 cell-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{36}$$

Results are reported as offsets from the current configuration, where every tilt is 12° (`tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another cell of the same band within 6 dB of that band's strongest cell;
- serves more UEs within each cell-band's PRB limit, and keeps the busiest cell-bands out of overload;
- preserves each band's physical role.

**Starting condition** (`tables/01_eda/`):
- **Coverage.** 18.4 % of the grid is a coverage hole, 40.6 % is weakly covered (−120 to −90 dBm) and 41.0 % has good coverage. Holes are a periphery effect: 0.9 % of tiles within 1 km of a node against 23.4 % beyond it (`hole_summary.csv`).
- **Band behaviour.** 700 MHz alone leaves 21.9 % of the grid in a hole, 1800 MHz 32.0 % and 2600 MHz 37.0 % (`coverage_classes_per_band.csv`). By raw signal, 700 MHz is the strongest layer on 89.9 % of the covered area. Because 2600 MHz is preferred whenever it clears the threshold, the serving rule puts 77.2 % of covered area on 2600 MHz before PRB limits (`serving_area_per_band.csv`).
- **Service.** 58.8 % of UE rows are not served (`serving_band_mix.csv`). A third of UE rows (33.3 %) stand on hole tiles, mostly one hotspot 3.4 km from the nearest node with no propagation path (`hotspots.csv`, `mdt_share_by_coverage.csv`).
- **Demand.** Weighted by peak PRB demand, 92.3 % sits on weak tiles and none in holes (Table 3). A UE with no candidate cell-band adds no PRB demand, so demand in holes is invisible in that measure, not absent.

![Study area](figures/00_simulation/study_area.png)

*Figure 1. Study area: twelve cells on four nodes and a sample of UE positions over the scene. Source: `figures/00_simulation/study_area.png`.*

![RSRP per band](figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the current tilts. Source: `figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](figures/01_eda/demand_vs_coverage.png)

*Figure 3. Peak PRB demand beside signal strength at the current tilts. Source: `figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. The PRB limits are N_RB at 15 kHz SCS, TS 38.101-1 Table 5.3.2-1 [7]. Source: [`tables/00_simulation/frequency_bands.csv`](tables/00_simulation/frequency_bands.csv).*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per cell |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

*Table 2. Scenario. Sources: [`tables/00_simulation/study_area.csv`](tables/00_simulation/study_area.csv), [`network_configuration.csv`](tables/00_simulation/network_configuration.csv), [`ue_distribution.csv`](tables/00_simulation/ue_distribution.csv), [`mdt_summary.csv`](tables/00_simulation/mdt_summary.csv).*

| Property | Value |
|---|---|
| Scenario ID | `scn_7d938e15f9ac4618` |
| Grid | 326 × 310 tiles, 20 m (6,200 × 6,520 m) |
| Nodes / cells / cell-band pairs | 4 / 12 / 36 |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | 10,066 |
| MDT rows (served at the current tilts) | 4,145 (41.2 %) |
| UE positions with no path to any cell | 25.2 % |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`tables/01_eda/coverage_by_area_and_demand.csv`](tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of peak PRB demand |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 18,598 | 18.4 % | 0.0 % |
| Weak (−120 to −90 dBm) | 41,061 | 40.6 % | 92.3 % |
| Good (> −90 dBm) | 41,401 | 41.0 % | 7.7 % |

## 3. Proposed Solutions

All three solutions search the same bounded tilt box (`src/optim/space.py`) with the same Sionna-RT evaluator (`src/optim/evaluator.py`). The evaluator builds the scene once and uses one fixed solver seed, so every candidate shares the same Monte-Carlo noise. The solutions also share one definition of "better": the objective J (Section 3.4). They differ only in where they look.

Each run has four steps (`src/optim/run.py`, `src/optim/report.py`):
1. Evaluate the current configuration.
2. Search.
3. Publish a shortlist of the `optim.n_solutions` = 4 highest-J configurations, always including the current one.
4. Re-trace every published solution and score it on all UEs.

### 3.1 Rule-based per-band sweep

This mimics the heuristic an operator would use. Every cell on a band shares one tilt, which collapses the 36 dimensions to three. Coordinate descent then passes over the bands: it tries `n_steps = 5` evenly spaced tilts across the band's range, keeps the best, and moves to the next band, for `n_rounds = 2` passes. A value equal to the current one is skipped. The run is deterministic and spent 27 sweep evaluations plus the incumbent. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. It perturbs a random subset of the centre's dimensions and Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. It doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0003 and ADR 0006.

### 3.4 The objective every solution maximises

The objective is ADR 0006, implemented in `src/optim/objective.py`:

$$J = J_{\text{radio}}^{\gamma}\; J_{\text{load}}^{1-\gamma}$$

$$J_{\text{radio}} = \frac{1}{|G|}\sum_{g\in G} \sigma\!\left(\frac{R_s(g) - T_{\text{cov}}}{\tau_R}\right) e^{-\beta\, m_g}, \qquad J_{\text{load}} = 1 - \frac{\mathrm{CVaR}_\alpha\bigl([\rho - \rho_0]_+\bigr)}{1-\rho_0}$$

- $R_s(g)$ is the strongest cell-band RSRP at tile $g$, and $-\infty$ where no path exists.
- $m_g$ counts, on every band, the other cells of that band above $T_{\text{cov}}$ and within $\Delta_R$ of the band's strongest cell (`src/kpi/overlap.py::overlap_neighbors`). This is the same count the overlap rate thresholds.
- $\rho$ is the PRBs admitted to one cell-band in one interval over its PRB limit. Every cell-band in every interval counts, idle ones at zero.
- $\mathrm{CVaR}_\alpha$ is the mean of the largest ⌈(1 − α) N⌉ samples [4].
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm and $\Delta_R$ = `kpi.overlap_margin_db` = 6 dB, so each physical quantity has one threshold shared with the KPIs.
- `kpi.objective` sets τ_R = 10 dB, β = 1, ρ₀ = 0.8, α = 0.9 and γ = 0.5.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs load which cell-band:
- **Candidates.** Each UE ranks its cell-bands. Those at or above −120 dBm come first, in band preference 2600 > 1800 > 700 MHz and then by RSRP; the rest follow by RSRP. Layers at or below `kpi.hole_dbm` are never candidates.
- **PRB need.** A UE needs PRBs = 20 Mbps / (12 · SCS · log₂(1 + SINR)), with 15 kHz SCS [8] and the solver's full-load co-band SINR.
- **Admission.** The UE takes the first candidate still at or under 80 % of its PRB limit (`max_admission_utilisation`) with room for its PRBs.
- **Order.** Within an interval, UEs are admitted in a seeded random order.

Every `kpi.capacity` value is a placeholder.

**Which UEs count.**
- **Search.** The UE-counted terms (served ratio and J_load) are scored on the MDT, the UEs served at the current tilts.
- **Evaluation.** Every UE counts: the published solutions are re-traced and re-scored on all 10,066 positions (`evaluation.parquet`). Sections 6.1–6.2 and 6.4–6.7 use those all-UE numbers.
- **Search traces.** Convergence, sample efficiency, trade-off scatters and γ sensitivity stay on the MDT.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J on all UEs, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, on all UEs:
   - coverage hole rate ↓
   - co-band overlap rate ↓
   - served UE ratio ↑
   - weak-coverage rate ↓
   - cell-edge RSRP ↑ (5th percentile of best-server RSRP over covered tiles [5], read beside the hole rate)

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied.
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by the winner against the median candidate, sample efficiency, and TuRBO paired with random search on the same seed.
4. **Robustness.** Whether the pick survives another γ, and where the configuration moves demand, not only area.
5. **Cost.** Ray-tracing evaluations, ray-tracing minutes and wall-clock minutes per run.

## 5. Research Methodology

**Data generation.** No operator data was available. All data was produced synthetically (`notebooks/00_simulation.ipynb`, `src/simulation/`):

1. **Scenario.** The scene was rasterised onto the 20 m grid. A UE population of 10 to 20 UEs was drawn every 15 minutes for 7 days. Each draw mixed four elliptical Gaussian hotspots, placed where surrounding building volume is high and at least 500 m apart, with a uniform open-ground background. The hotspots hold 70 % of UEs on average, modulated by a diurnal profile and AR(1) noise (`configs/simulation.yaml` `time`, `density`). UEs are independent per interval, with no mobility.
2. **Radio map.** Each band was ray-traced separately at 10⁷ rays per transmitter and maximum depth 8, with line of sight, specular reflection and refraction, but no diffuse reflection or diffraction. Materials were ITU-R P.2040 and frequency-static. The output was per-cell RSRP and SINR on the grid (`tables/00_simulation/propagation_parameters.csv`).
3. **MDT.** The serving rule was run on the current radio map. The 4,145 admitted UE rows, with their serving RSRP, form the MDT. No measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 22 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`). The checks cover grid, scenario ID, band and cell order, bounds, schedule, duplicate rows, the MDT subset and baseline tilts. The notebook then wrote typed Parquet tables without dropping or altering a row. `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 28.
- Every candidate was fully ray-traced and scored. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map, all-UE re-score and `run.json` under `outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands and KPI definition (22 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the current configuration on all UEs.
4. Compares each winner with the candidates its own search evaluated.
5. Re-picks each run's winner under five values of γ.
6. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
7. Records cost and convergence.

**Relevance, criteria and practicality.** Ray tracing on real city geometry was chosen over a statistical path-loss model because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away. Budgets were matched between random search and TuRBO so that criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost. One seed per method kept the study within a single GPU session: ray tracing took about 3–5 s per candidate (Table 12).

## 6. Analysis and Interpretation

### 6.1 Comparability and correctness

All 22 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). Each KPI recomputed from the archived radio maps matched the recorded value to float round-off (`tables/04_evaluation/kpi_reproducibility.csv`). The differences below therefore come from the configurations, not from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, all UEs, seed 42. Arrows show the better direction. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| Objective J ↑ | 0.6818 | 0.7122 | 0.7055 | **0.7124** |
| Radio utility J_radio ↑ | 0.5568 | 0.5843 | 0.5777 | **0.5864** |
| Load utility J_load ↑ | 0.8349 | **0.8681** | 0.8616 | 0.8656 |
| Coverage hole rate ↓ | 0.1840 | **0.1639** | 0.1680 | 0.1659 |
| Co-band overlap rate ↓ | **0.2689** | 0.2744 *(worse)* | 0.2810 *(worse)* | 0.2730 *(worse)* |
| Served UE ratio ↑ | 0.4121 | **0.5163** | 0.5069 | 0.5133 |
| Weak-coverage rate ↓ | 0.4063 | **0.2594** | 0.2847 | 0.2632 |
| Cell-edge RSRP [dBm] ↑ | −112.27 | **−109.34** | −110.00 | −109.39 |
| Measures improved / worsened | — | 6 / 1 | 6 / 1 | 6 / 1 |

*Table 5. Relative improvement over the current configuration, signed so that positive is better. Source: [`tables/04_evaluation/kpi_relative_improvement.csv`](tables/04_evaluation/kpi_relative_improvement.csv).*

| Method | Hole | Overlap | Served | Weak | Edge RSRP | J_radio | J_load | J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | +10.9 % | −2.1 % | +25.3 % | +36.2 % | +2.6 % | +4.9 % | +4.0 % | +4.45 % |
| Random search | +8.7 % | −4.5 % | +23.0 % | +29.9 % | +2.0 % | +3.8 % | +3.2 % | +3.47 % |
| TuRBO | +9.8 % | −1.5 % | +24.6 % | +35.2 % | +2.6 % | +5.3 % | +3.7 % | +4.49 % |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change of every KPI and objective term against the current configuration, oriented so that higher is better. Source: `figures/04_evaluation/kpi_improvement.png`.*

All three methods improved J, and TuRBO and the rule sweep are practically tied (+0.0306 and +0.0304). Every method improved the hole, weak, served and edge measures and both objective terms, and every method worsened co-band overlap.

The rule sweep is slightly ahead on holes, weak coverage, served ratio, cell-edge RSRP and J_load. TuRBO is ahead on J_radio and has the smallest overlap penalty. Random search trails both on every measure.

The largest movements are in weak coverage (−14.3 points of area for TuRBO) and the served ratio (+10.1 points). The hole rate moves by only 1.6–2.0 points. That is consistent with Section 2: most hole area lies at the grid periphery and in no-path shadow, beyond the reach of a tilt change.

### 6.3 Did the search matter?

*Table 6. Winner against the candidates each run evaluated (search scores, MDT). Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.6851 | 0.7192 | 0.7198 | 0.7275 | 0.7393 |
| Rule-based sweep | 0.6851 | — | 0.7254 | 0.7476 | 0.7480 |
| TuRBO | 0.6851 | 0.7192 | **0.7454** | **0.7504** | **0.7521** |

*Table 7. Best J reached after a fixed number of evaluations (search scores, MDT). Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | 0.7393 | 0.7254 | 0.7393 |
| 25 | 0.7393 | 0.7480 | 0.7407 |
| 50 | 0.7393 | — | 0.7482 |
| 100 | 0.7393 | — | 0.7493 |
| 145 | 0.7393 | — | 0.7521 |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective score found so far against evaluations (MDT). Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

TuRBO's median candidate (0.7454) scored above random search's single best (0.7393). The two runs share the same 16 initial points and diverge only once the model proposes. The trust-region proposals averaged 0.7447 against 0.7186 for the Sobol design (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`). This is the strongest evidence here that the model, not a lucky draw, produced TuRBO's gain over random search.

- **Random search** found its best point at evaluation 7, inside the shared design, and never improved on it.
- **TuRBO** never restarted, and its best point was its last evaluation (index 144), so a larger budget may still improve it.
- **Rule sweep** reached 0.7480 within 25 evaluations. TuRBO needed about 50 to match that.

The paired all-UE gain of TuRBO over random search is +0.0069 on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation. The TuRBO–rule margin (+0.0003 on all UEs) is smaller still.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate (MDT), with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Objective weight γ.** Only γ can be varied without re-tracing: the history stores J_radio and J_load, and every other objective parameter is inside them. Re-picking each run's winner from its own history (`tables/04_evaluation/gamma_sensitivity.csv`, MDT) gives:
- **γ = 0.75.** Every method keeps its pick, as at the configured 0.5.
- **γ = 0 and 0.25.** Every method picks a different configuration, one with higher J_load.
- **γ = 1 (radio only).** TuRBO's pick moves to evaluation 117, with slightly lower overlap (0.2691 against 0.2730) and a slightly higher hole rate (0.1677 against 0.1659). The rule and random picks stay.

The recommendation holds near the configured γ but is not invariant to it. τ_R, β, ρ₀ and α were not varied, because each needs new ray tracing.

**Search score against all-UE score.** The search ranks the shortlist on the MDT, and the all-UE re-score does not always agree:
- **Rule sweep.** The recommended solution scores 0.7122 on all UEs, while solution 3 of the same shortlist scores 0.7159.
- **TuRBO.** The recommended solution scores 0.7124; solution 3 scores 0.7154.
- **Random search.** The recommended solution is also the best on all UEs.

(Source: each run's `evaluation.parquet`, Appendix A.) The published deliverables (`reports/outputs/solutions_<method>.csv`) carry the search's MDT-scored measures.

*Table 8. Coverage class by area and by demand (demand weighted by the current configuration's peak PRB demand). The current weak-tile demand share reads 92.1 % here against 92.3 % in Table 3, and the current served ratio 0.4121 against 0.4118 in `tables/00_simulation/baseline_kpis.csv`: the serving rule admits UEs in a seeded order over the UE table's rows, and notebooks 00/01 read the raw UE table while evaluation reads the sorted processed one. Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 18.4 % / 0.0 % | 16.4 % / 0.0 % | 16.8 % / 0.3 % | 16.6 % / 0.0 % |
| Weak | 40.6 % / 92.1 % | 25.9 % / 77.3 % | 28.5 % / 80.7 % | 26.3 % / 77.8 % |
| Good | 41.0 % / 7.9 % | 57.7 % / 22.7 % | 54.7 % / 19.0 % | 57.1 % / 22.2 % |

*Table 9. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | 0.88 | 67.0 % | 16.3 % |
| Rule-based sweep | 1.02 | 67.2 % | 15.5 % |
| Random search | 0.90 | 66.2 % | 14.5 % |
| TuRBO | 0.94 | 67.3 % | 14.5 % |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

The hole-rate gain lands where there are no users: holes already carry no peak demand at the current tilts, and neither TuRBO nor the rule sweep moves any demand into holes. Random search moves 0.3 % of demand into holes. The demand-weighted change is on weak tiles, which fall from 92.1 % of demand to 77.8 % under TuRBO.

Overlap is the one measure every method worsens. Uptilting extends every footprint, so the mean neighbour count over covered tiles rises (0.88 → 0.94 for TuRBO, → 1.02 for the rule sweep). Pile-ups of three or more neighbours nevertheless fall (16.3 % → 14.5 % for TuRBO). Giving cells on one band different tilts keeps TuRBO's neighbour count below the rule sweep's.

### 6.5 Capacity impact

*Table 10. UE service, all UEs. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 58.8 % | −1.58 | 4.03 | 61.1 | 28.3 % / 5.3 % / 7.5 % |
| Rule-based sweep | **48.4 %** | −0.86 | **6.22** | **46.8** | 38.6 % / 5.0 % / 8.0 % |
| Random search | 49.3 % | −0.97 | 6.06 | 47.6 | 37.0 % / 5.7 % / 8.0 % |
| TuRBO | 48.7 % | **−0.69** | 5.97 | 48.1 | 37.8 % / 5.3 % / 8.2 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

Every method cut the unserved share by about ten points. The extra served UEs land almost entirely on the preferred 2600 MHz layer: 28.3 % → 37.8 % of UE reports under TuRBO, while 1800 and 700 MHz barely move. Median served SINR rises by about 2 dB and the PRBs a served UE needs fall by about a fifth. TuRBO has the best 10th-percentile served SINR.

Per band, every layer covers more area after optimization, and its mean RSRP where covered rises by 5–7 dB (`tables/04_evaluation/band_layer_summary.csv`). The low band's median served SINR rises most (8.99 → 12.44 dB under TuRBO).

Nearly half of the UE reports remain unserved. Part is out of reach: a quarter of UE positions have no path to any cell at the current tilts (Table 2). The rest reflects the placeholder capacity model, under which most cell-bands already peak near their PRB limit. These figures should be read as directions of change, not absolute capacity.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 11. Tilt movement for the highest-J configuration. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 7.03 | 10.61 | −7.03 |
| 1800 MHz | 12 / 12 | 7.78 | 10.63 | −7.78 |
| 700 MHz | 12 / 12 | 7.48 | 11.94 | −7.47 |

The TuRBO configuration uptilts 35 of 36 cell-bands, by 7–8° on average per band. Only n2c1 on 700 MHz stays practically where it was (+0.06°).

Several proposed tilts sit close to the 0° lower bound: n3c1 on 700 MHz at 0.06° and n1c1 on 700 MHz at 0.25°. Together with the rule sweep's choice of 3.75° on every cell and band (Δ = −8.25°), this suggests the uniform 12° starting tilt is too much downtilt for this layout, and the lower bound may be constraining the search. With one seed, it is not established which per-cell differences matter and which reflect where the trust region happened to be when the budget ended.

The largest traffic shifts are on 2600 MHz sectors facing 165° and 285° (`tables/04_evaluation/cell_impact.csv`):
- n3c1: +187 served reports
- n2c1: +176
- n0c2: +158

### 6.7 Cost

*Table 12. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 28 | 20 | 1.20 | 2.21 | 2.6 | 0.0137 |
| Random search | 145 | 7 | 12.14 | 15.37 | 5.0 | 0.0015 |
| TuRBO | 145 | 144 | 9.86 | 19.43 | 4.1 | 0.0016 |

TuRBO's GP fitting and acquisition added about 9.6 minutes of wall clock to its ray tracing, nearly doubling it. For the same number of evaluations it spent less ray-tracing time than random search; this run does not show why. By J gain per minute, the rule sweep was about nine times as cost-effective as either other method. It reached practically TuRBO's result in about a ninth of the wall clock.

### 6.8 Limitations

These results should be read tentatively, for eight reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed, and the TuRBO–rule and TuRBO–random margins may lie within seed-to-seed variation.
3. **Winner's curse.** Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
4. **The search sees a biased population.** The MDT holds 41 % of UE rows, none of them in holes. The search optimizes load on well-covered UEs, and its ranking of the shortlist can disagree with the all-UE re-score (Section 6.4). The published deliverable CSVs carry the MDT scores.
5. **The capacity model is a placeholder.** It drives the served ratio, J_load and every capacity figure. It uses a Shannon rate with no MCS cap and full-load interference beside partial PRB load.
6. **The objective parameters are judgement values.** Only γ was varied, and the grid KPIs and J_radio count area, including area without users.
7. **Tilts near the bound.** The search space may be too narrow at 0°.
8. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

## 7. Conclusions and Recommendations

*Table 13. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J (all UEs) | +0.0304 (2nd, practically tied) | +0.0237 (3rd) | **+0.0306 (1st)** |
| 2. Reported KPIs | 4 of 5 better, overlap worse; best on holes, weak, served, edge | 4 of 5 better, overlap worst | 4 of 5 better, overlap worse; smallest overlap penalty |
| 3. Search effectiveness | Best at evaluation 20 of 28 | Best at evaluation 7, no later gain | **Median candidate above random's best; still improving at 144** |
| 4. Robustness | Pick stable for γ ≥ 0.5; a runner-up scores higher on all UEs | Pick stable for γ ≥ 0.5; 0.3 % of demand into holes | Pick stable for γ 0.5–0.75; a runner-up scores higher on all UEs |
| 5. Cost | **2.2 min, 28 evaluations** | 15.4 min, 145 evaluations | 19.4 min, 145 evaluations |

**Conclusions.**

- **A coordinated uptilt explains most of the gain.** Every method moved the uniform 12° tilts up by 6–8° on average. That cut weak coverage by about a third, raised the served ratio by about a quarter and lifted the cell edge by 2–3 dB, at a small cost in co-band overlap.
- **TuRBO** reached the highest J and beat random search at the same budget and seed, with evidence that the model, not chance, earned that margin. Its advantage over the three-variable rule sweep is negligible on this scenario.
- **The rule-based sweep** was by far the most cost-effective, reaching practically TuRBO's objective in 28 evaluations and a ninth of the wall clock.
- **Random search** was dominated by both.
- **Every result** depends on one scenario, one seed, an MDT-scored search and a placeholder capacity model.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on, and the shortlist ranking changes between the MDT and all UEs.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`). Re-trace the shortlists under other solver seeds, to put intervals on the TuRBO–rule margin.
3. **Seed TuRBO's initial design with the rule sweep's result** (3.75° on every cell-band), and consider a larger budget: TuRBO was still improving when its budget ran out.
4. **Revisit the tilt lower bound** if the hardware allows, since several proposed tilts sit near 0°.
5. **Decide which score the deliverable should rank by.** Either rank the published shortlist by the all-UE re-score, or state the MDT basis wherever the deliverable is read.
6. **Replace the placeholder `kpi.capacity` values with operator figures**, then build held-out scenario validation and the planned MARL arm before drawing a method-level conclusion.

---

## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/external/scene/scene.xml` (not in Git) |
| Layout | 4 nodes, 1,732 m triangle plus centroid, 3 sectors at 45° / 165° / 285°, 20 m masts |
| Tilt | 12° current, bounds [0°, 15°], every cell-band |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6, edge percentile 5 |
| Objective | τ_R 10 dB, β 1, ρ₀ 0.8, α 0.9, γ 0.5 |
| Capacity (placeholders) | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission cap 0.8, 20 Mbps per UE, SCS 15 kHz |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3; rule 5 steps × 2 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-17_07-25-42/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-17_07-41-04/` |
| TuRBO | `outputs/optim/turbo/2026-09-17_07-57-05/` |

Older run directories under `outputs/optim/` lack `evaluation.parquet` and were skipped by the evaluation. To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`reports/tables/…`) | Figures (`reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `mdt_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `mdt_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `mdt_overview`, `mdt_share_by_component`, `mdt_share_by_coverage`, `mdt_rsrp_statistics`, `cell_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `mdt_over_time`, `mdt_vs_ue_positions`, `mdt_rsrp_vs_ue_rsrp`, `cell_band_utilisation`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `template_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `decision_variables`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_<method>`, `tilt_movement_<method>`, `kpi_comparison_<method>`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary` | `search_progress`, `kpi_progress`, `tilt_movement_<method>`, `coverage_before_after_<method>`, `rsrp_change_maps`, `serving_band_mix` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `gamma_sensitivity`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `ue_service_summary`, `cell_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_served_ratio`, `tradeoff_overlap_rate_vs_served_ratio`, `rsrp_change_maps`, `coverage_before_after`, `coverage_class_maps`, `overlap_neighbour_maps`, `serving_band_mix`, `cell_band_utilisation`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta, MDT-scored), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° for every entry; bounds are [0°, 15°].*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 8.65 (−3.35) | 1.37 (−10.63) | 2.68 (−9.32) |
| n0c1 | 2.34 (−9.66) | 4.74 (−7.26) | 3.60 (−8.40) |
| n0c2 | 3.32 (−8.68) | 2.12 (−9.88) | 5.15 (−6.85) |
| n1c0 | 4.22 (−7.78) | 6.23 (−5.77) | 10.13 (−1.87) |
| n1c1 | 8.66 (−3.34) | 2.95 (−9.05) | 0.25 (−11.75) |
| n1c2 | 7.88 (−4.12) | 4.70 (−7.30) | 2.03 (−9.97) |
| n2c0 | 6.97 (−5.03) | 7.59 (−4.41) | 2.40 (−9.60) |
| n2c1 | 5.26 (−6.74) | 4.21 (−7.79) | 12.06 (+0.06) |
| n2c2 | 5.48 (−6.52) | 4.57 (−7.43) | 2.10 (−9.90) |
| n3c0 | 1.39 (−10.61) | 4.93 (−7.07) | 6.60 (−5.40) |
| n3c1 | 1.46 (−10.54) | 4.13 (−7.87) | 0.06 (−11.94) |
| n3c2 | 4.01 (−7.99) | 3.08 (−8.92) | 7.30 (−4.70) |

The rule-based sweep's best configuration sets every cell on every band to 3.75° (Δ = −8.25°) (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] R. T. Rockafellar and S. Uryasev, "Optimization of conditional value-at-risk," *Journal of Risk*, vol. 2, no. 3, 2000.

[5] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[6] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[7] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[8] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
