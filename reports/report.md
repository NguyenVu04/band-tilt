# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-17: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 30 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 9° on 1800 MHz and 6° on 700 MHz. A week-long, time-varying population of 10,066 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J (ADR 0006): a per-tile coverage utility, discounted for each overlapping co-band neighbour and averaged over the grid. Five KPIs were reported beside J: hole rate, co-band overlap rate, served UE ratio, weak-coverage rate and cell-edge RSRP.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations.

**Results.**
- **TuRBO** scored highest on J: 0.6305 against 0.5913 currently, a 6.6 % gain. It got there mainly by cutting the co-band overlap rate from 0.369 to 0.321 (13 % relative). The price was slightly more holes (+1.6 %), slightly more weak coverage (+3.3 %) and a 0.2 dB lower cell edge.
- **The rule sweep** reached J = 0.6052 (+2.3 %) in 28 evaluations.
- **Random search** reached J = 0.5920 (+0.1 %).

No method improved every measure.

**Caveats.** The results come from one scenario, one search seed per method, and a placeholder capacity model. No configuration is recommended for deployment.

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
- **Cells.** Each node has three sectors at azimuths 45°, 165° and 285°, on 30 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve cells and 36 cell-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{36}$$

Results are reported as offsets from the current configuration: 12° on 2600 MHz, 9° on 1800 MHz and 6° on 700 MHz for every cell (`tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another cell of the same band within 6 dB of that band's strongest cell;
- serves more UEs within each cell-band's PRB limit;
- preserves each band's physical role.

**Starting condition** (`tables/01_eda/`):
- **Coverage.** 7.3 % of the grid is a coverage hole, 27.1 % is weakly covered (−120 to −90 dBm) and 65.6 % has good coverage. Holes are a periphery effect: 0.3 % of tiles within 1 km of a node are holes, against 9.3 % beyond (`hole_summary.csv`).
- **Band behaviour.**
  - Alone, 700 MHz leaves 9.1 % of the grid in a hole, 1800 MHz 17.1 % and 2600 MHz 22.2 % (`coverage_classes_per_band.csv`).
  - By raw signal, 700 MHz is the strongest layer on 93 % of the covered area.
  - 2600 MHz is preferred whenever it clears the threshold, so the serving rule puts 84 % of covered area on 2600 MHz before PRB limits (`serving_area_per_band.csv`).
- **Overlap.** 36.9 % of tiles have at least one overlapping co-band neighbour, with a mean of 1.19 neighbours per covered tile (`overlap_neighbour_summary.csv`).
- **Service.** 42.8 % of UE rows are not served (`serving_band_mix.csv`). 20.6 % of UE rows stand on hole tiles, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Weighted by peak PRB demand, 81.0 % sits on weak tiles and none in holes (Table 3). A UE with no candidate cell-band adds no PRB demand, so demand in holes is invisible in that measure, not absent.

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

*Table 2. Scenario. Sources: [`tables/00_simulation/study_area.csv`](tables/00_simulation/study_area.csv), [`network_configuration.csv`](tables/00_simulation/network_configuration.csv), [`ue_distribution.csv`](tables/00_simulation/ue_distribution.csv), [`ue_measurement_summary.csv`](tables/00_simulation/ue_measurement_summary.csv).*

| Property | Value |
|---|---|
| Scenario ID | `scn_7d938e15f9ac4618` |
| Grid | 326 × 310 tiles, 20 m (6,200 × 6,520 m) |
| Nodes / cells / cell-band pairs | 4 / 12 / 36 |
| Mast height | 30 m |
| Current tilt (2600 / 1800 / 700 MHz) | 12° / 9° / 6° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | 10,066 |
| UE positions with no path to any cell | 14.5 % |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`tables/01_eda/coverage_by_area_and_demand.csv`](tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of peak PRB demand |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 7,408 | 7.3 % | 0.0 % |
| Weak (−120 to −90 dBm) | 27,375 | 27.1 % | 81.0 % |
| Good (> −90 dBm) | 66,277 | 65.6 % | 19.0 % |

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
- `kpi.objective` sets τ_R = 10 dB and β = 1.

J depends on the radio map alone. It counts area, not UEs.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. It drives the served ratio and every capacity table.
- **Candidates.** Each UE ranks its cell-bands. Those at or above −120 dBm come first, in band preference 2600 > 1800 > 700 MHz and then by RSRP; the rest follow by RSRP. Layers at or below `kpi.hole_dbm` are never candidates.
- **PRB need.** A UE needs PRBs = 20 Mbps / (12 · SCS · log₂(1 + SINR)), with 15 kHz SCS [7] and the solver's full-load co-band SINR.
- **Admission.** The UE takes the first candidate still at or under 80 % of its PRB limit (`max_admission_utilisation`) with room for its PRBs.
- **Order.** Within an interval, UEs are admitted strongest RSRP first, over every layer at the UE. Ties keep the UE table's row order.

Every `kpi.capacity` value is a placeholder.

**Which UEs count.** Every UE position counts, both in the search and in evaluation. The MDT (the 5,756 UE rows served at the current tilts, `tables/00_simulation/mdt_summary.csv`) is still built and kept for later use, but nothing scores on it.

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
3. **MDT.** The serving rule was run on the current radio map, and the 5,756 admitted UE rows form the MDT. It is kept for later use; no measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 22 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`).
- The checks cover grid, scenario ID, band and cell order, bounds, schedule, duplicate rows, the MDT subset and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 28.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map and `run.json` under `outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands and KPI definition (22 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the current configuration.
4. Compares each winner with the candidates its own search evaluated.
5. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets, so criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took about 2–3 s per candidate (Table 12).

## 6. Analysis and Interpretation

### 6.1 Comparability and correctness

All 22 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off, with one exception: TuRBO's weak coverage rate is 0.27980 as recorded and 0.27981 recomputed, one tile in 101,060. The archived map is a second ray trace of the winning tilts, and GPU ray tracing is not bit-reproducible. The gap is far below every difference discussed below, so the differences come from the configurations, not from bookkeeping.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction. Source: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| Objective J ↑ | 0.5913 | 0.6052 | 0.5920 | **0.6305** |
| Coverage hole rate ↓ | 0.0733 | 0.0732 | **0.0725** | 0.0745 *(worse)* |
| Co-band overlap rate ↓ | 0.3695 | 0.3513 | 0.3740 *(worse)* | **0.3215** |
| Served UE ratio ↑ | 0.5718 | 0.5862 | **0.5939** | 0.5767 |
| Weak-coverage rate ↓ | **0.2709** | 0.2845 *(worse)* | 0.2775 *(worse)* | 0.2798 *(worse)* |
| Cell-edge RSRP [dBm] ↑ | **−105.66** | −105.86 *(worse)* | −105.81 *(worse)* | −105.87 *(worse)* |
| Measures improved / worsened | — | 4 / 2 | 3 / 3 | 3 / 3 |

*Table 5. Relative improvement over the current configuration, signed so that positive is better. Source: [`tables/04_evaluation/kpi_relative_improvement.csv`](tables/04_evaluation/kpi_relative_improvement.csv).*

| Method | Hole | Overlap | Served | Weak | Edge RSRP | J |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | +0.1 % | +4.9 % | +2.5 % | −5.0 % | −0.2 % | +2.34 % |
| Random search | +1.0 % | −1.2 % | +3.9 % | −2.4 % | −0.1 % | +0.12 % |
| TuRBO | −1.6 % | +13.0 % | +0.9 % | −3.3 % | −0.2 % | +6.62 % |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change of every measure against the current configuration, oriented so that higher is better. Source: `figures/04_evaluation/kpi_improvement.png`.*

TuRBO improved J the most (+0.0392), ahead of the rule sweep (+0.0139) and random search (+0.0007). Its gain comes almost entirely from co-band overlap, which falls by 4.8 points of area. It gives up a little on holes (+0.12 points), weak coverage (+0.9 points) and cell-edge RSRP (−0.21 dB).

The rule sweep also gains through less overlap, but less of it, and pays more in weak coverage. Random search moves J barely at all. It has the best served ratio and hole rate of the three, but more overlap than the current configuration.

Every KPI moves by only a few percent. The objective rewards less overlap more strongly than it penalises the small coverage losses that come with it.

### 6.3 Did the search matter?

*Table 6. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.5913 | 0.5641 | 0.5623 | 0.5758 | 0.5920 |
| Rule-based sweep | 0.5913 | — | 0.5985 | 0.6037 | 0.6052 |
| TuRBO | 0.5913 | 0.5641 | **0.6018** | **0.6232** | **0.6305** |

*Table 7. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | 0.5920 | 0.6043 | 0.5920 |
| 25 | 0.5920 | 0.6052 | 0.5975 |
| 50 | 0.5920 | — | 0.6042 |
| 100 | 0.5920 | — | 0.6167 |
| 145 | 0.5920 | — | 0.6305 |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly worse than the current configuration: their median is 0.562 against 0.591. The per-band starting tilts are therefore a reasonable configuration, not a weak one.

TuRBO's median candidate (0.6018) scored above random search's single best (0.5920), and its 90th percentile above the rule sweep's best. The two runs share the same 16 initial points and diverge only once the model proposes. The trust-region proposals averaged 0.6056, against 0.5648 for the Sobol design (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`). This is the strongest evidence here that the model, not a lucky draw, produced TuRBO's gain over random search.

- **Random search** found its best point at evaluation 7, inside the shared design, and never improved on it.
- **TuRBO** never restarted. It passed random search's best at evaluation 18 and the rule sweep's at evaluation 61, and found its best at evaluation 133 of 144, so a larger budget may still improve it.
- **Rule sweep** reached 0.6043 within 10 evaluations and 0.6052 by evaluation 13.

The paired gain of TuRBO over random search is +0.0385 on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

Hole rate and overlap rate conflict along the Pareto front:
- The low-hole end is made up of rule-sweep candidates (hole rate 0.071, overlap 0.356–0.364).
- The low-overlap end is TuRBO's late candidates, ending in TuRBO's own pick (hole rate 0.074, overlap 0.321).
- The rule sweep's and random search's picks are dominated.

### 6.4 Is the result robust?

**Trade-off between KPIs.** J is maximised by cutting overlap, and the configurations that do so give up a little coverage. The objective counts area and never reads the hole rate, the weak rate or where UEs stand, so this trade is a property of the objective, not an accident of one search.

*Table 8. Coverage class by area and by demand (demand weighted by the current configuration's peak PRB demand). Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 7.3 % / 0.0 % | 7.3 % / 0.4 % | 7.3 % / 0.7 % | 7.4 % / 1.6 % |
| Weak | 27.1 % / 81.0 % | 28.4 % / 83.0 % | 27.8 % / 80.5 % | 28.0 % / 81.2 % |
| Good | 65.6 % / 19.0 % | 64.2 % / 16.6 % | 65.0 % / 18.7 % | 64.6 % / 17.2 % |

*Table 9. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | 1.19 | 60.1 % | 20.9 % |
| Rule-based sweep | 1.11 | 62.1 % | 20.5 % |
| Random search | 1.21 | 59.7 % | 19.4 % |
| TuRBO | 1.05 | 65.3 % | 18.3 % |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand in holes.** The current configuration has no demand on hole tiles. Every method moves some onto them: TuRBO 1.6 % of peak demand, random search 0.7 % and the rule sweep 0.4 %. The share of demand on weak tiles barely changes under TuRBO (81.0 % to 81.2 %).

**Overlap.** TuRBO lowers it on every view. The mean neighbour count over covered tiles falls from 1.19 to 1.05, the share of covered tiles with no neighbour rises from 60.1 % to 65.3 %, and pile-ups of three or more fall from 20.9 % to 18.3 %.

**Objective parameters.** The robustness of the pick to the objective parameters (τ_R, β) was not tested, because each needs new ray tracing. The history stores only the measured objective, so no parameter can be varied offline.

### 6.5 Capacity impact

*Table 10. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 42.8 % | −0.75 | 4.52 | 57.3 | 37.2 % / 11.1 % / 8.9 % |
| Rule-based sweep | 41.4 % | −0.56 | 5.73 | 49.5 | 46.1 % / 5.4 % / 7.1 % |
| Random search | **40.6 %** | **−0.31** | **6.70** | **44.4** | 45.9 % / 6.2 % / 7.3 % |
| TuRBO | 42.3 % | −1.00 | 5.97 | 48.1 | 41.9 % / 7.6 % / 8.1 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

**Service.** Every method serves slightly more UEs, most of all random search (2.2 points) and least TuRBO (0.5 points). The extra served UEs land on the preferred 2600 MHz layer, and part of the 1800 MHz traffic moves there as well: 1800 MHz falls from 11.1 % to 5–8 % of UE reports. Median served SINR rises under every method, and the PRBs a served UE needs fall by 14–23 %. TuRBO is the only method whose 10th-percentile served SINR falls.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`):
- Area covered rises by 1.9 points on 2600 MHz under TuRBO and moves by less than half a point on the other bands.
- Mean RSRP where covered rises on 2600 MHz (−96.8 to −92.8 dBm) and 1800 MHz (−90.0 to −88.6 dBm) and falls slightly on 700 MHz (−82.1 to −82.7 dBm).
- The median served SINR on 700 MHz falls from 12.4 to 10.9 dB.

About 42 % of UE reports remain unserved. Part is out of reach: 14.5 % of UE positions have no path to any cell (Table 2). The rest reflects the placeholder capacity model, under which most cell-bands already peak near their PRB limit. These figures should be read as directions of change, not absolute capacity.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 11. Tilt movement for the highest-J configuration. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 4.88 | 10.98 | −3.62 |
| 1800 MHz | 12 / 12 | 4.33 | 6.83 | −1.69 |
| 700 MHz | 12 / 12 | 3.97 | 8.29 | +1.12 |

The TuRBO configuration is not a uniform shift:
- It uptilts 22 of 36 cell-bands and downtilts 14.
- Four cells (n0c2, n1c0, n2c1, n3c0) are downtilted on every band, five of their cell-bands to above 14°, close to the 15° upper bound.
- Most other cell-bands are uptilted, n0c0 on 2600 MHz to 1.0°, close to the 0° lower bound.

The rule sweep instead sets every cell to 7.5° on every band: Δ = −4.5° on 2600 MHz, −1.5° on 1800 MHz and +1.5° on 700 MHz (`outputs/tilt_change_rule.csv`). With one seed, it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended. Proposed tilts at both bounds suggest the [0°, 15°] box may constrain the search.

The largest traffic shifts are on 2600 MHz sectors (`tables/04_evaluation/cell_impact.csv`):
- n3c1: +330 served reports
- n2c1: −251
- n1c1: +198

### 6.7 Cost

*Table 12. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 28 | 13 | 0.91 | 1.18 | 1.9 | 0.0118 |
| Random search | 145 | 7 | 7.60 | 8.68 | 3.1 | 0.0001 |
| TuRBO | 145 | 133 | 7.03 | 10.76 | 2.9 | 0.0036 |

TuRBO's GP fitting and acquisition added about 3.7 minutes of wall clock to its ray tracing. For the same number of evaluations it spent less ray-tracing time than random search; this run does not show why. By J gain per minute, the rule sweep was about three times as cost-effective as TuRBO, but it reached a J 0.025 lower.

### 6.8 Limitations

These results should be read tentatively, for eight reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed, and the TuRBO–rule and TuRBO–random margins may lie within seed-to-seed variation.
3. **Winner's curse.** Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured, and a re-trace of the same tilts already differs by one tile (Section 6.1).
4. **The objective counts area only.** J reads coverage and overlap per tile, never where UEs stand or how cells are loaded. The winner moves 1.6 % of demand onto hole tiles and worsens the hole and weak rates slightly while J rises.
5. **The capacity model is a placeholder.** It drives the served ratio and every capacity figure. It uses a Shannon rate with no MCS cap and full-load interference beside partial PRB load.
6. **The objective parameters are judgement values.** τ_R and β were not varied.
7. **Tilts near both bounds.** The search space may be too narrow at 0° and at 15°.
8. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

## 7. Conclusions and Recommendations

*Table 13. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0139 (2nd) | +0.0007 (3rd) | **+0.0392 (1st)** |
| 2. Reported KPIs | 4 better, 2 worse; less overlap, more weak coverage | 3 better, 3 worse; best served ratio and hole rate, more overlap | 3 better, 3 worse; least overlap, slightly more holes and weak coverage |
| 3. Search effectiveness | Best at evaluation 13 of 28 | Best at evaluation 7, no later gain | **Median candidate above random's best; best at 133 of 144** |
| 4. Robustness | 0.4 % of demand onto holes | 0.7 % of demand onto holes | 1.6 % of demand onto holes |
| 5. Cost | **1.2 min, 28 evaluations** | 8.7 min, 145 evaluations | 10.8 min, 145 evaluations |

**Conclusions.**

- **TuRBO** reached the highest J, clearly above both the rule sweep and random search at the same seed. There is evidence that the model, not chance, earned its margin over random search.
- **Its gain is overlap.** TuRBO cut the co-band overlap rate by 13 % with a mixed configuration: four cells downtilted on every band, most others uptilted. It gave up a little on holes, weak coverage and the cell edge.
- **The rule-based sweep** was the most cost-effective per minute, but moving whole bands limited it to +2.3 % on J.
- **Random search** barely improved on the current per-band tilts.
- **Every result** depends on one scenario, one seed, an area-only objective and a placeholder capacity model.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on, and the winner worsens coverage slightly while it cuts overlap.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`). Re-trace the shortlists under other solver seeds, to put intervals on the TuRBO margins.
3. **Decide whether J should weigh the coverage losses and the demand moved onto holes.** The current objective does not read them.
4. **Consider a larger TuRBO budget and a wider tilt box** if the hardware allows: TuRBO was still improving late in its budget, and proposed tilts sit near both bounds.
5. **Replace the placeholder `kpi.capacity` values with operator figures**, then build held-out scenario validation and the planned MARL arm before drawing a method-level conclusion.

---

## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/external/scene/scene.xml` (not in Git) |
| Layout | 4 nodes, 1,732 m triangle plus centroid, 3 sectors at 45° / 165° / 285°, 30 m masts |
| Tilt | Current 12° (2600 MHz), 9° (1800 MHz), 6° (700 MHz); bounds [0°, 15°], every cell-band |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6, edge percentile 5 |
| Objective | τ_R 10 dB, β 1 |
| Capacity (placeholders) | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission cap 0.8, 20 Mbps per UE, SCS 15 kHz, admission strongest RSRP first |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3; rule 5 steps × 2 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-17_13-46-51/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-17_13-55-32/` |
| TuRBO | `outputs/optim/turbo/2026-09-17_13-57-03/` |

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

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 9° on 1800 MHz and 6° on 700 MHz for every cell; bounds are [0°, 15°].*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 1.02 (−10.98) | 4.18 (−4.82) | 1.63 (−4.37) |
| n0c1 | 3.76 (−8.24) | 4.63 (−4.37) | 2.94 (−3.06) |
| n0c2 | 14.67 (+2.67) | 14.72 (+5.72) | 12.35 (+6.35) |
| n1c0 | 13.66 (+1.66) | 12.34 (+3.34) | 12.26 (+6.26) |
| n1c1 | 4.55 (−7.45) | 2.24 (−6.76) | 3.82 (−2.18) |
| n1c2 | 9.29 (−2.71) | 2.17 (−6.83) | 4.30 (−1.70) |
| n2c0 | 4.14 (−7.86) | 6.21 (−2.79) | 2.28 (−3.72) |
| n2c1 | 13.11 (+1.11) | 10.29 (+1.29) | 14.29 (+8.29) |
| n2c2 | 7.37 (−4.63) | 4.45 (−4.55) | 3.91 (−2.09) |
| n3c0 | 14.11 (+2.11) | 14.53 (+5.53) | 13.05 (+7.05) |
| n3c1 | 4.99 (−7.01) | 5.61 (−3.39) | 6.94 (+0.94) |
| n3c2 | 9.82 (−2.18) | 6.39 (−2.61) | 7.61 (+1.61) |

The rule-based sweep's best configuration sets every cell on every band to 7.5° (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[5] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[6] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[7] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
