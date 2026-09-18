# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-18: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J (ADR 0006): a per-tile coverage utility, discounted for each overlapping co-band neighbour and averaged over the grid, with τ_R = 30 dB and β = 0.5. Five KPIs were reported beside J: hole rate, co-band overlap rate, served UE ratio, weak-coverage rate and cell-edge RSRP.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations.

**Results.**
- **TuRBO** scored highest on J: 0.5575 against 0.5380 currently, a 3.6 % gain. It improved **every one of the five KPIs**, and was the only method to cut co-band overlap materially, from 0.316 to 0.294 (6.9 % relative).
- **The rule sweep** reached J = 0.5469 (+1.7 %) in 28 evaluations, also improving all five KPIs, and took the largest share of the weak-coverage and cell-edge gains.
- **Random search** reached J = 0.5404 (+0.5 %), improving four KPIs and worsening co-band overlap.

Unlike the previous run at 30 m masts, the winning configurations no longer trade coverage away for separation: TuRBO and the rule sweep improved all five reported measures at once.

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
- **Service.** 42.4 % of UE rows are not served (`serving_band_mix.csv`). 21.2 % of UE rows stand on hole tiles, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Weighted by peak PRB demand, 82.9 % sits on weak tiles and none in holes (Table 3). A UE with no candidate cell-band adds no PRB demand, so demand in holes is invisible in that measure, not absent.

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
| Weak (−120 to −90 dBm) | 31,017 | 30.7 % | 82.9 % |
| Good (> −90 dBm) | 58,675 | 58.1 % | 17.1 % |

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

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0003 and ADR 0006.

### 3.4 The objective every solution maximises

The objective is ADR 0006, implemented in `src/optim/objective.py`:

$$J = \frac{1}{|G|}\sum_{g\in G} \sigma\!\left(\frac{R_s(g) - T_{\text{cov}}}{\tau_R}\right) e^{-\beta\, m_g}$$

- $R_s(g)$ is the strongest cell-band RSRP at tile $g$, and $-\infty$ where no path exists.
- $m_g$ counts, on every band, the other cells of that band above $T_{\text{cov}}$ and within $\Delta_R$ of the band's strongest cell (`src/kpi/overlap.py::overlap_neighbors`). This is the same count the overlap rate thresholds.
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm and $\Delta_R$ = `kpi.overlap_margin_db` = 6 dB, so each physical quantity has one threshold shared with the KPIs.
- `kpi.objective` sets τ_R = 30 dB and β = 0.5 (`tables/03a_baseline/objective_parameters.csv`).

J depends on the radio map alone. It counts area, not UEs.

**What τ_R and β do.** τ_R is the width of the coverage sigmoid: at 30 dB a tile 30 dB above the hole threshold scores 0.73 rather than the 0.95 a 10 dB width would give, so utility accrues gradually across the whole usable RSRP range instead of saturating just above −120 dBm. β sets the overlap discount: each co-band neighbour retains $q_{ov} = e^{-\beta} = 0.607$ of a tile's utility. Together they make J reward reaching more area more strongly, and punish overlap more weakly, than the earlier τ_R = 10, β = 1 setting.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. It drives the served ratio and every capacity table.
- **Candidates.** Each UE ranks its cell-bands. Those at or above −120 dBm come first, in band preference 2600 > 1800 > 700 MHz and then by RSRP; the rest follow by RSRP. Layers at or below `kpi.hole_dbm` are never candidates. Because `kpi.capacity.rsrp_threshold_dbm` equals `kpi.hole_dbm` in the committed config, the documented "skip the preferred band" fallback cannot fire: band preference always decides among candidates.
- **PRB need.** A UE needs PRBs = 20 Mbps / (12 · SCS · log₂(1 + SINR)), with 15 kHz SCS [7] and the solver's full-load co-band SINR.
- **Admission.** The UE takes the first candidate whose load is still at or under 70 % of its PRB limit (`max_admission_utilisation`) and that has room for its PRBs. This is a gate applied before admission, not a ceiling: a UE admitted at 69 % load may take the cell-band above 70 %, but never above `max_prb`.
- **Order.** Within an interval, UEs are admitted in report-time (`t_s`) order — a cell fills as its reports arrive, not best-first. Reports at the same instant are taken strongest RSRP first over every layer at the UE, then by row order; since `t_s` is drawn continuously per UE, that tiebreak is effectively never reached.

**Which UEs count.** Every UE position counts, both in the search and in evaluation. The MDT (the 5,815 UE rows served at the current tilts, `tables/00_simulation/mdt_summary.csv`) is still built and kept for later use, but nothing scores on it.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration:
   - coverage hole rate ↓
   - co-band overlap rate ↓
   - served UE ratio ↑
   - weak-coverage rate ↓
   - cell-edge RSRP ↑ (5th percentile of best-server RSRP over covered tiles [4], read beside the hole rate)

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied.
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
3. **MDT.** The serving rule was run on the current radio map, and the 5,815 admitted UE rows form the MDT. It is kept for later use; no measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 22 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`).
- The checks cover grid, scenario ID, band and cell order, bounds, schedule, duplicate rows, the MDT subset and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
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

All 23 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off: the largest absolute gap on any measure is 1.7 × 10⁻⁶ dB on cell-edge RSRP, and on J it is 4.4 × 10⁻¹¹. The differences discussed below therefore come from the configurations, not from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| Objective J ↑ | 0.5380 | 0.5469 | 0.5404 | **0.5575** |
| Coverage hole rate ↓ | 0.1125 | **0.1072** | 0.1083 | 0.1091 |
| Co-band overlap rate ↓ | 0.3164 | 0.3133 | 0.3254 *(worse)* | **0.2944** |
| Served UE ratio ↑ | 0.5765 | 0.6219 | **0.6240** | 0.6144 |
| Weak-coverage rate ↓ | 0.3069 | **0.2660** | 0.2864 | 0.2706 |
| Cell-edge RSRP [dBm] ↑ | −108.56 | **−107.32** | −107.98 | −107.53 |
| Measures improved / worsened | — | **5 / 0** | 4 / 1 | **5 / 0** |

*Table 5. Relative improvement over the current configuration, signed so that positive is better. Source: [`tables/04_evaluation/kpi_relative_improvement.csv`](tables/04_evaluation/kpi_relative_improvement.csv).*

| Method | Hole | Overlap | Served | Weak | Edge RSRP | J |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | +4.7 % | +1.0 % | +7.9 % | +13.3 % | +1.1 % | +1.65 % |
| Random search | +3.8 % | −2.8 % | +8.2 % | +6.7 % | +0.5 % | +0.46 % |
| TuRBO | +3.0 % | +6.9 % | +6.6 % | +11.8 % | +1.0 % | +3.63 % |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change of every measure against the current configuration, oriented so that higher is better. Source: `figures/04_evaluation/kpi_improvement.png`.*

TuRBO improved J the most (+0.0195), ahead of the rule sweep (+0.0089) and random search (+0.0025).

The striking result is that **there is no longer a trade-off to describe**. TuRBO and the rule sweep improved all five reported KPIs simultaneously; only random search worsened one (overlap). At 30 m masts and τ_R = 10, β = 1, the winning configuration bought a 13 % overlap reduction by giving up holes, weak coverage and the cell edge. Here every method gains on coverage *and* service, and TuRBO gains on overlap as well.

Two changes explain this jointly. Lowering the masts to 25 m left the current configuration with substantially more slack to recover — 11.2 % holes and 30.7 % weak coverage, against 7.3 % and 27.1 % at 30 m — so tilting into better coverage is available in a way it was not before. And the flatter sigmoid (τ_R = 30) with the weaker overlap discount (β = 0.5) means J now pays for that recovered coverage instead of spending it on separation.

The methods split on *which* improvement they chased. The rule sweep took the largest coverage gains (weak −13.3 %, hole −4.7 %, edge +1.1 dB) but barely moved overlap (+1.0 %). TuRBO took most of the overlap (+6.9 %) while keeping nearly all of the coverage gain. That difference is what the extra 117 evaluations bought.

### 6.3 Did the search matter?

*Table 6. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.5380 | 0.5226 | 0.5211 | 0.5301 | 0.5404 |
| Rule-based sweep | 0.5380 | — | 0.5413 | 0.5457 | 0.5469 |
| TuRBO | 0.5380 | 0.5226 | **0.5458** | **0.5541** | **0.5575** |

*Table 7. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | 0.5404 | 0.5443 | 0.5404 |
| 25 | 0.5404 | 0.5469 | 0.5411 |
| 50 | 0.5404 | — | 0.5429 |
| 100 | 0.5404 | — | 0.5543 |
| 145 | 0.5404 | — | 0.5575 |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly worse than the current configuration: their median is 0.5211 against 0.5380. The per-band starting tilts are therefore a reasonable configuration, not a weak one.

TuRBO's median candidate (0.5458) scored above random search's single best (0.5404), and its 90th percentile (0.5541) above the rule sweep's best (0.5469). The two runs share the same 16 initial points and diverge only once the model proposes. The trust-region proposals averaged 0.5458, against 0.5217 for the Sobol design (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`). This is the strongest evidence here that the model, not a lucky draw, produced TuRBO's gain over random search.

- **Random search** found its best point at evaluation 7, inside the shared design, and never improved on it in the remaining 138.
- **TuRBO** never restarted. It passed the rule sweep's best between evaluations 50 and 100, and found its own best at evaluation 136 of 144 — the last eighth of its budget — so a larger budget may still improve it.
- **Rule sweep** reached 0.5443 within 10 evaluations and its best 0.5469 by evaluation 12, then found nothing better in its remaining 16.

The paired gain of TuRBO over random search is +0.0170 on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Trade-off between KPIs.** J is maximised here by recovering coverage and, for TuRBO, cutting overlap at the same time. The objective counts area and never reads the hole rate, the weak rate or where UEs stand, so the agreement between J and the reported KPIs in this run is a fact about this operating point, not a guarantee. At a lower τ_R or a higher β, J would again prefer separation over reach.

*Table 8. Coverage class by area and by demand (demand weighted by the current configuration's peak PRB demand). Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 11.2 % / 0.0 % | 10.7 % / 0.0 % | 10.8 % / 0.3 % | 10.9 % / 0.5 % |
| Weak | 30.7 % / 82.9 % | 26.6 % / 77.7 % | 28.6 % / 79.5 % | 27.1 % / 79.3 % |
| Good | 58.1 % / 17.1 % | 62.7 % / 22.3 % | 60.5 % / 20.2 % | 62.0 % / 20.2 % |

*Table 9. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | 0.96 | 64.4 % | 17.6 % |
| Rule-based sweep | 1.03 | 64.9 % | 17.5 % |
| Random search | 1.03 | 63.5 % | 16.8 % |
| TuRBO | 0.93 | 67.0 % | 16.4 % |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand in holes.** The current configuration has no demand on hole tiles, and the rule sweep keeps it that way. TuRBO moves 0.5 % of peak demand onto hole tiles and random search 0.3 % — far less than the 1.6 % the previous run's winner moved. The share of demand on weak tiles falls under every method, most under the rule sweep (82.9 % to 77.7 %).

**Overlap.** TuRBO is the only method that lowers it on every view. The mean neighbour count over covered tiles falls from 0.96 to 0.93, the share of covered tiles with no neighbour rises from 64.4 % to 67.0 %, and pile-ups of three or more fall from 17.6 % to 16.4 %. The rule sweep and random search both *raise* the mean neighbour count while covering more area — they gain coverage by illuminating more ground with more layers, which is exactly the effect the overlap term exists to price.

**Objective parameters.** The robustness of the pick to the objective parameters (τ_R, β) was not tested at a third setting, because each needs new ray tracing. The history stores only the measured objective, so no parameter can be varied offline. The comparison against the previous run at τ_R = 10, β = 1 is confounded with the mast-height change and cannot isolate either.

### 6.5 Capacity impact

*Table 10. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 42.4 % | −1.14 | 4.41 | 58.1 | 37.6 % / 11.0 % / 9.0 % |
| Rule-based sweep | 37.8 % | **−0.41** | 6.47 | 45.5 | 46.2 % / 7.0 % / 9.0 % |
| Random search | **37.6 %** | −0.55 | **6.92** | **43.3** | 47.4 % / 7.7 % / 7.3 % |
| TuRBO | 38.6 % | −1.01 | 6.43 | 45.7 | 43.5 % / 9.6 % / 8.3 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

**Service.** Every method serves substantially more UEs: 4.6 to 4.8 points, against 0.5 to 2.2 points in the previous run. The extra served UEs land on the preferred 2600 MHz layer, which rises from 37.6 % to 43.5–47.4 % of UE reports, while 1800 MHz falls from 11.0 % to 7.0–9.6 %. Median served SINR rises by about 2 dB under every method, and the PRBs a served UE needs fall by 21–25 %. TuRBO's 10th-percentile served SINR is essentially unchanged (−1.14 to −1.01 dB) where the other two improve it more.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`):
- Area covered rises by 2.6 points on 2600 MHz under TuRBO, 1.2 points on 1800 MHz, and is flat on 700 MHz.
- Mean RSRP where covered rises on 2600 MHz (−97.2 to −92.3 dBm) and 1800 MHz (−91.5 to −88.2 dBm), and rises slightly on 700 MHz (−84.2 to −82.6 dBm). Unlike the previous run, no band loses mean signal.
- The median served SINR on 2600 MHz nearly doubles, from 2.9 to 5.6 dB.

About 38 % of UE reports remain unserved. Much of that is out of reach: 17.5 % of UE positions have no path to any cell (Table 2) and 21.2 % stand on hole tiles. The rest reflects PRB exhaustion — most 2600 MHz cell-bands peak within a percent of their limit (`tables/04_evaluation/cell_impact.csv`), so the 2600 MHz layer is the binding constraint on service in every configuration tested.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 11. Tilt movement for the highest-J configuration. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 5.51 | 11.97 | −4.78 |
| 1800 MHz | 12 / 12 | 5.04 | 9.86 | −4.24 |
| 700 MHz | 12 / 12 | 5.07 | 7.67 | −2.42 |

The TuRBO configuration is a net uptilt on every band, the opposite of the previous run's net downtilt on the high bands:
- It uptilts 26 of 36 cell-bands and downtilts 10.
- Three cells (n0c2, n1c0, n2c1) are downtilted on every band, with n1c0's 700 MHz layer pushed to 14.55°, close to the 15° upper bound.
- The remaining nine cells are mostly uptilted, and n2c0's 2600 MHz layer reaches 0.03°, effectively at the 0° lower bound.

This reads as the search concentrating a few sectors downward to serve near-in demand while opening the rest of the network outward to recover the coverage the 25 m masts lost.

The rule sweep instead sets every cell on a band to one value: 7.5° on 2600 MHz (Δ = −4.5°), 7.5° on 1800 MHz (Δ = −2.5°) and 3.75° on 700 MHz (Δ = −4.25°) (`outputs/tilt_change_rule.csv`) — a uniform uptilt, which is most of the coverage recovery and none of the overlap reduction. With one seed, it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended. Proposed tilts at both bounds again suggest the [0°, 15°] box may constrain the search.

The largest traffic shifts are on 2600 MHz sectors (`tables/04_evaluation/cell_impact.csv`):
- n2c1: −335 served reports
- n3c1: +312
- n3c2: +293

### 6.7 Cost

*Table 12. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 28 | 12 | 1.18 | 2.02 | 2.5 | 0.0044 |
| Random search | 145 | 7 | 9.26 | 11.21 | 3.8 | 0.0002 |
| TuRBO | 145 | 136 | 9.53 | 18.03 | 3.9 | 0.0011 |

TuRBO's GP fitting and acquisition added about 8.5 minutes of wall clock to its ray tracing, roughly 90 % on top of the trace time. Ray-tracing cost per evaluation is near-identical between random search and TuRBO (3.8 vs 3.9 s), as expected. By J gain per minute, the rule sweep was about four times as cost-effective as TuRBO, but it reached a J 0.011 lower and left overlap essentially untouched.

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed, and the TuRBO–rule and TuRBO–random margins may lie within seed-to-seed variation.
3. **Winner's curse.** Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
4. **The objective counts area only.** J reads coverage and overlap per tile, never where UEs stand or how cells are loaded. That it agreed with all five KPIs in this run is a property of this operating point, not of the objective.
5. **The capacity model is a simplification.** It drives the served ratio and every capacity figure. It uses a Shannon rate with no MCS cap, full-load co-band interference against partial PRB load, and no receiver noise figure.
6. **The objective parameters are judgement values.** τ_R = 30 dB and β = 0.5 were chosen, not derived, and were not varied within this run.
7. **Tilts near both bounds.** The search space may be too narrow at 0° and at 15°: the winner places one cell-band at 0.03° and another at 14.55°.
8. **This run is not comparable to the 2026-09-17 run.** Besides the mast height and the objective parameters, the seed-stream derivation changed (`src/simulation/seeds.py` now hashes `(seed, name)` instead of adding a per-stream offset), so the UE population differs: 10,087 rows against 10,066. The mast positions happen to be identical, but the scenario is not the same one. Note also that `scenario_id` hashes the *config values* only, so it still reads `scn_7d938e15f9ac4618` — it is not a sufficient key for "same population".
9. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

## 7. Conclusions and Recommendations

*Table 13. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0089 (2nd) | +0.0025 (3rd) | **+0.0195 (1st)** |
| 2. Reported KPIs | **5 better, 0 worse**; largest coverage and cell-edge gains | 4 better, 1 worse; best served ratio, more overlap | **5 better, 0 worse**; only method to cut overlap |
| 3. Search effectiveness | Best at evaluation 12 of 28 | Best at evaluation 7, no gain in the next 138 | **Median candidate above random's best; best at 136 of 144** |
| 4. Robustness | **No demand onto holes** | 0.3 % of demand onto holes | 0.5 % of demand onto holes |
| 5. Cost | **2.0 min, 28 evaluations** | 11.2 min, 145 evaluations | 18.0 min, 145 evaluations |

**Conclusions.**

- **TuRBO** reached the highest J, clearly above both the rule sweep and random search at the same seed. There is evidence that the model, not chance, earned its margin over random search.
- **There is no coverage-versus-overlap trade in this configuration.** At 25 m masts with τ_R = 30 and β = 0.5, TuRBO and the rule sweep improved all five reported KPIs at once. The 25 m baseline left more coverage to recover than the 30 m one did, and the flatter, less overlap-averse objective directed the search at recovering it.
- **The methods differ in what they recover.** The rule sweep took most of the coverage available with a uniform per-band uptilt, in 28 evaluations. TuRBO matched most of that coverage gain *and* cut overlap by 6.9 %, which the rule sweep's three-dimensional search cannot express.
- **Random search barely improved on the current per-band tilts** and was the only method to worsen a KPI.
- **Every result** depends on one scenario, one seed, an area-only objective and a simplified capacity model.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`). Re-trace the shortlists under other solver seeds, to put intervals on the TuRBO margins. The seed-stream fix in `src/simulation/seeds.py` is a prerequisite: additive offsets aliased streams across a seed sweep.
3. **Give TuRBO a larger budget.** It found its best at evaluation 136 of 144 and never restarted, so the trust region was still productive when the budget ended — the clearest single lever available.
4. **Widen the tilt box.** The winner sits at 0.03° and 14.55°, so [0°, 15°] is binding at both ends.
5. **Decide whether J should read demand.** The current objective counts area only; it agreed with the KPIs here, but that agreement is not structural.
6. **Address the 2600 MHz PRB ceiling.** Most 2600 MHz cell-bands peak within a percent of `max_prb` in every configuration, so tilt alone cannot raise the served ratio much further; that is a capacity decision, not a tilt one.
7. **Build held-out scenario validation and the planned MARL arm** before drawing a method-level conclusion.

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
| Objective | τ_R 30 dB, β 0.5 |
| Capacity | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission gate 0.7, 20 Mbps per UE, SCS 15 kHz, admission in report-time order |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12; rule 5 steps × 2 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-18_01-55-26/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-18_02-06-39/` |
| TuRBO | `outputs/optim/turbo/2026-09-18_02-09-22/` |

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

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell; bounds are [0°, 15°].*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 3.91 (−8.09) | 3.43 (−6.57) | 1.04 (−6.96) |
| n0c1 | 8.61 (−3.39) | 3.37 (−6.63) | 2.12 (−5.88) |
| n0c2 | 14.08 (+2.08) | 10.41 (+0.41) | 11.47 (+3.47) |
| n1c0 | 12.90 (+0.90) | 12.44 (+2.44) | 14.55 (+6.55) |
| n1c1 | 9.30 (−2.70) | 0.14 (−9.86) | 3.18 (−4.82) |
| n1c2 | 4.17 (−7.83) | 3.57 (−6.43) | 2.49 (−5.51) |
| n2c0 | 0.03 (−11.97) | 4.05 (−5.95) | 0.33 (−7.67) |
| n2c1 | 13.38 (+1.38) | 11.93 (+1.93) | 13.14 (+5.14) |
| n2c2 | 5.01 (−6.99) | 3.79 (−6.21) | 4.77 (−3.23) |
| n3c0 | 9.84 (−2.16) | 8.86 (−1.14) | 8.69 (+0.69) |
| n3c1 | 3.10 (−8.90) | 3.31 (−6.69) | 2.15 (−5.85) |
| n3c2 | 2.28 (−9.72) | 3.78 (−6.22) | 2.98 (−5.02) |

The rule-based sweep's best configuration sets every cell to 7.5° on 2600 MHz, 7.5° on 1800 MHz and 3.75° on 700 MHz (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[5] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[6] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[7] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
