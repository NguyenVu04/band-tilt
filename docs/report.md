# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report — template. Every number, table and figure belongs to the optimization runs listed in Appendix A, read by notebooks `00_simulation` through `04_evaluation`, and to the configuration in `configs/`. Placeholders `<!-- TODO -->` name the generated file each value comes from. Paths are relative to this file (`docs/`): generated tables and figures are under `../reports/`, run directories under `../outputs/`. RSRP, interference and noise are all per resource element: thermal noise is k·T over one 15 kHz subcarrier, not over the channel bandwidth. Eleven KPIs are recorded for every candidate; the evaluation reports seven over all bands and the coverage, overlap, RSRP and SINR measures per band ([ADR 0003](adr/0003-three-objectives-and-morbo.md)).*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band sector layout when every band on every sector is tuned jointly, not one band at a time.

The study area is an urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve in all), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 20°]. Every sector-band starts at 10°. A week-long, time-varying UE population was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates are ray-traced and scored on two objectives, coverage and separation, searched jointly; throughput is measured and recorded beside them ([ADR 0003](adr/0003-three-objectives-and-morbo.md)). Eleven KPIs are measured beside the objectives, none of them weighted into them.

Two searches start from the same current configuration with matched budgets and the same seed:
- Sobol random search,
- MORBO, multi-objective trust-region Bayesian optimization.

**Results.** <!-- TODO: hypervolume and KPI outcome per method, from ../reports/tables/04_evaluation/hypervolume.csv and kpi_scoreboard.csv -->

**Caveats.** <!-- TODO: scenario count, seeds per method, capacity model -->

---

## 1. Introduction

### 1.1 The problem as a RAN engineer meets it

A 5G/6G site commonly radiates several frequency bands from the same mast, and the bands differ physically:
- Low bands such as 700 MHz propagate farther and penetrate buildings better. They are the coverage floor.
- High bands such as 2600 MHz carry more capacity over a smaller footprint. They are the capacity layer.
- A mid band such as 1800 MHz bridges the two.

In day-to-day optimisation the tilts of those layers are rarely set together. A typical life cycle is:
1. **Planning** sets one electrical tilt per band from a link budget, often the same value on every sector (here 10° on every band).
2. **Tuning** then happens one complaint at a time. A drive test, a KPI alarm or a ticket points at a sector with poor edge RSRP, pilot pollution, a coverage hole or a congested carrier.
3. **The fix** is a remote electrical tilt (RET) change on one band of that sector, followed by days of counter collection to see whether it helped and what it broke next door.

Each change is reasonable locally, but the layers interact:
- Uptilting a capacity carrier widens its footprint into its neighbours' and raises co-band overlap and interference.
- Downtilting it pulls its edge in and can open a hole that only the low band then covers, at lower throughput.
- Because the serving layer is chosen by load and signal quality, moving one band's tilt moves traffic between layers, so the counters of carriers that were never touched change too.

The result is a loop that is slow, expensive in field time, and blind to cross-band effects until they show up in the counters. Most networks are left at a configuration that nobody chose as a whole.

### 1.2 What this study does

It treats the tilts of every (sector, band) pair as one coordinated optimisation problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) over real city geometry acts as a digital twin: it scores every proposed configuration on coverage, overlap, signal quality and the throughput the simulated UEs would get. Two searches are compared on the same objectives and budget:
- random search, as the model-free control,
- MORBO [2], which keeps trust regions around the best trade-offs found and learns where to look.

### 1.3 What it offers a RAN engineer, and how it changes the work

**What you get from a run.** Not a single magic setting but a reviewable package:
- The Pareto front of measured configurations (`../reports/outputs/solutions_<method>.csv`), the recommendation first, each with all eleven KPIs and its delta against today. Every number is a ray-traced measurement, not a model prediction.
- The tilt table of each of those configurations (`../reports/outputs/tilt_options_<method>.csv`), which is the content of a RET change request.
- A sector-impact table (Section 6.5) ranking which carriers gain or lose traffic, so you know where to look first after a change.
- The trade-offs made explicit: where the recommendation gains coverage, where it adds overlap, which layer the traffic moves to.

**What it changes in your work.**
- **From trial-and-error to choosing between measured options.** <!-- TODO: evaluations and wall-clock per run, from ../reports/tables/04_evaluation/method_cost.csv -->
- **From band-by-band to layer-aware decisions.** <!-- TODO: the layer pattern of the recommendation, Section 6.6 -->
- **From reacting to complaints to planning a rollout.** The sector-impact table says which carriers to watch, and the front offers alternatives when a proposed tilt is mechanically or operationally unacceptable.

**What stays your job.** The study does not replace engineering judgement, and Section 6.8 lists why:
- **Field validation.** One simulated city is not your network. Results are uncalibrated against drive tests, and nothing has been tested on a held-out scenario.
- **Operational limits.** The search does not penalise antenna movement. <!-- TODO: how many antennas the recommendation moves, and by how much, from tilt_movement_summary.csv -->
- **The model's blind spots.** There is no MCS cap, no scheduler, no mobility and no inter-band interference in the objectives.

<!-- TODO: the narrow, testable claim this run supports -->

## 2. Problem Definition

**Operational statement.** Given a multi-band site layout and its current tilts, propose new electrical tilts for every sector and band that close coverage holes and weak areas, keep co-band overlap under control, and raise the signal quality and throughput users get, while each band keeps the role its propagation suits. The proposal must come with its measured effect on every KPI and its tilt delta per antenna, so an engineer can review it before any RET change.

**Network.** The study area is a local city scene (`data/scenes/hanoi/scene.xml`), rasterised onto a grid of 20 m tiles. <!-- TODO: grid size and extent, from ../reports/tables/00_simulation/study_area.csv -->
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side.
- **Sectors.** Each node has three sectors at azimuths 45°, 165° and 285°, on 25 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve sectors and 36 sector-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 sectors and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 20^\circ]^{36}$$

Proposals are snapped to the 0.1° lattice (`optim.tilt_resolution_deg`). Results are reported as offsets from the current configuration: 10° on every band and sector (`../reports/tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another sector of the same band within 6 dB of that band's strongest sector;
- raises the throughput UEs can expect, counting a UE no layer reaches at 0 Mbit/s;
- preserves each band's physical role.

**Starting condition** (`../reports/tables/01_eda/`):
- **Coverage.** <!-- TODO: hole / weak / good shares, from coverage_classes_per_band.csv and hole_summary.csv -->
- **Band behaviour.** <!-- TODO: per-band hole shares and serving area, from coverage_classes_per_band.csv and serving_area_per_band.csv -->
- **Overlap.** <!-- TODO: from overlap_neighbour_summary.csv -->
- **Service.** <!-- TODO: share of UE reports on hole tiles, from ../reports/tables/00_simulation/serving_band_mix.csv and hotspots.csv -->
- **Demand.** <!-- TODO: from coverage_by_area_and_demand.csv (Table 3) -->

![Study area](../reports/figures/00_simulation/study_area.png)

*Figure 1. Study area: twelve sectors on four nodes and a sample of UE positions over the scene. Source: `../reports/figures/00_simulation/study_area.png`.*

![RSRP per band](../reports/figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the current tilts. Source: `../reports/figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](../reports/figures/01_eda/demand_vs_coverage.png)

*Figure 3. UE demand beside signal strength at the current tilts. Source: `../reports/figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. The PRB limits are N_RB at 15 kHz SCS, TS 38.101-1 Table 5.3.2-1 [4]. Source: [`frequency_bands.csv`](../reports/tables/00_simulation/frequency_bands.csv).*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per sector |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

*Table 2. Scenario. Sources: [`study_area.csv`](../reports/tables/00_simulation/study_area.csv), [`ue_distribution.csv`](../reports/tables/00_simulation/ue_distribution.csv), [`ue_measurement_summary.csv`](../reports/tables/00_simulation/ue_measurement_summary.csv), [`decision_variables.csv`](../reports/tables/00_simulation/decision_variables.csv).*

| Property | Value |
|---|---|
| Scenario ID | <!-- TODO --> |
| Grid | <!-- TODO --> |
| Nodes / sectors / sector-band pairs | 4 / 12 / 36 |
| Mast height | 25 m |
| Current tilt (2600 / 1800 / 700 MHz) | 10° / 10° / 10° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | <!-- TODO --> |
| UE positions with no path to any sector | <!-- TODO --> |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | <!-- TODO --> | | |
| Weak (−120 to −90 dBm) | <!-- TODO --> | | |
| Good (> −90 dBm) | <!-- TODO --> | | |

## 3. Proposed Solutions

Both solutions search the same bounded tilt box (`src/optim/space.py`) with the same Sionna-RT evaluator (`src/optim/evaluator.py`). The evaluator builds the scene once and uses one fixed solver seed, so every candidate shares the same Monte-Carlo noise. The solutions also share one definition of "better": the objectives of Section 3.3, compared by hypervolume. They differ only in where they look.

Each run has four steps (`src/optim/run.py`, `src/optim/report.py`):
1. Evaluate the current configuration.
2. Search, scoring every candidate on all UEs.
3. Re-trace the recommendation once, to archive its radio map.
4. Publish the Pareto front, largest hypervolume contribution first. The first row is the recommendation; the current configuration is listed only when it is on the front.

### 3.1 Sobol random search

Random search is the model-free control. It draws 8 + 64 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 8 are identical to MORBO's initial design. Because the budget and seed match MORBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.2 MORBO

MORBO [2] keeps trust regions centred on the Pareto points with the largest hypervolume contribution.
- **Each round.** Each region fits one GP per objective (Matérn-5/2 with ARD, under BoTorch's dimension-scaled log-normal lengthscale prior) to every evaluation inside a cube twice its side, topped up with the nearest points, whoever proposed them. Candidates perturb a subset of the dimensions of the Pareto points inside the region, each dimension with probability min(20 / 36, 1), halved over the budget. A batch of three is chosen one point at a time by Thompson sampling, each point maximising the hypervolume improvement over the observed front and the points already picked.
- **Region size.** A region starts at side 0.8 of the unit cube and only shrinks. It halves after max(10, ⌈36 / 3⌉) = 12 consecutive failed evaluations, a success being a hypervolume gain above 10⁻³ of the current hypervolume.
- **Restart.** Below side 0.01 the region restarts around the point a random hypervolume scalarisation ranks best in one draw of a global GP over the initial design and earlier restart points, at a side that decays with the budget used, and its old centre is barred from centring a region for 100 rounds.
- **Budget.** 8 Sobol initial points plus 64 evaluations, one trust region.
- **Seeds.** The initial design is the Sobol sequence seeded with `optim.seed`, the same one random search draws from. Restart and proposal seeds come from one `numpy.random.default_rng(optim.seed)` stream of the run's own.

The implementation uses BoTorch/GPyTorch [3]. The GPs only choose where to look; every reported number is ray-traced. Implementation: `src/optim/methods/morbo/search.py`; configuration: `configs/optim/method/morbo.yaml`; decision record: [ADR 0003](adr/0003-three-objectives-and-morbo.md).

### 3.3 The objectives

The objectives are [ADR 0003](adr/0003-three-objectives-and-morbo.md), implemented in `src/optim/objective.py`. With $R_{bs}(g)$ band $b$'s strongest sector at tile $g$, $i$ every other co-band sector, all in linear power, and $G_{\text{cov}}$ the tiles some sector-band reaches above $T_{\text{hole}}$ = `kpi.hole_dbm`:

- **Coverage** $|G_{\text{cov}}| / |G|$, that is $1 - \text{HoleRate}$.
- **Separation** $\frac{1}{|G_{\text{cov}}|}\sum_{g \in G_{\text{cov}}} \prod_b R_{bs}(g) / \left(R_{bs}(g) + \sum_i R_{bi}(g)\right)$: an equal rival halves a band; a band whose strongest sector is not above $T_{\text{hole}}$ counts 1.
- **Throughput** (recorded, not searched) $\frac{1}{|U|}\sum_u \ln(1 + R_u)$ over every UE report, $R_u$ in Mbit/s and 0 for a UE no layer reaches.

Coverage and separation are searched (`OBJECTIVE_NAMES`); each is maximised and rounded to 6 significant digits before a search reads it. A run is ranked by hypervolume against the origin, and its recommendation is the evaluated point with the largest hypervolume contribution.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a sector-band serves. Within an interval UEs connect one at a time, each to the sector-band above −120 dBm where an equal share of 0.8 of its `max_prb`, split over the UEs already there and itself, carries the most Shannon throughput. Nobody is refused. It drives the estimated-throughput KPIs, the throughput objective and every per-sector-band table.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The hypervolume of each run's evaluations, as a gain over the current configuration's.
2. **Reported KPIs.** The direction of change against the current configuration, over the seven network KPIs (`NETWORK_KPIS` in `src/evaluation/compare.py`):
   - coverage hole rate ↓, weak-coverage rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓
   - cell-edge, median and mean estimated UE throughput ↑ (5th and 50th percentiles and mean over every UE report, 0 Mbit/s for a UE no layer reaches)

   Per band, the hole, weak and overlap rates and the cell-edge and median RSRP and SINR ↑ (5th and 50th percentiles over the band's covered tiles) are reported for every layer. Best-server RSRP and SINR are per band only: the strongest layer across bands is not one any UE is measured on.

   A change is labelled better, worse, unchanged, or undefined when the delta is not a number (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the KPIs is weighted into the objectives, so agreement between the objectives and the KPIs is a finding, not a construction.
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by sample efficiency and MORBO paired with random search on the same seed.
4. **Robustness.** Where the configuration moves demand, not only area, and how it trades one KPI against another.
5. **Cost.** Ray-tracing evaluations, ray-tracing minutes and wall-clock minutes per run.

## 5. Research Methodology

**Data generation.** No operator data was available. All data was produced synthetically by the generator (`notebooks/00_simulation.ipynb`, `src/scenario/`), which writes the files `simulation.input` names: the UE table, the sector table and the manifest. The radio stage (`src/simulation/`) and everything after it read only those files, so real data can replace the generator.

1. **Scenario.** The scene was rasterised onto the 20 m grid, and a population of 10 to 20 UEs was drawn every 15 minutes for 7 days.
   - Each draw mixed four elliptical Gaussian hotspots with a uniform open-ground background. The hotspots sit where surrounding building volume is high, at least 500 m apart.
   - The hotspots hold 70 % of UEs on average, modulated by a diurnal profile and AR(1) noise (`configs/scenario.yaml` `time`, `density`).
   - UEs are independent per interval, with no mobility.
2. **Radio map.** Each band was ray-traced separately (`../reports/tables/00_simulation/propagation_parameters.csv`).
   - The output was per-sector RSRP and SINR on the grid.
   - <!-- TODO: reach per band, from ../reports/tables/00_simulation/reach_per_band.csv -->
3. **Service.** Every UE row is served from the radio map at its tile, in the search and in the evaluation alike. No measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the UE table, the manifest and the radio map against the contract checks in `../reports/tables/02_preprocessing/verification_checks.csv`. <!-- TODO: how many held -->
- The notebook then wrote the typed UE table without dropping or altering a row. The sector table is not copied: every stage reads it from the scene folder.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_morbo` ran each method with the search seeds in Appendix A.
- Each spent 73 evaluations: the incumbent, 8 initial points and 64 more.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, front, best tilt, best radio map and `run.json` under `../outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands, band carrier frequencies and KPI definition.
2. Recomputes each archived recommendation's KPIs from its saved radio map, to confirm they were recorded correctly.
3. Builds the scoreboard against the current configuration.
4. Compares the methods by hypervolume and sample efficiency.
5. Maps coverage, overlap, the serving-band mix, sector utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and MORBO had matched budgets and a shared initial design, so criterion 3 isolates the model's contribution.
- **Seeds.** <!-- TODO: seeds per method and the per-evaluation ray-tracing time, Table 11 -->

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

<!-- TODO: from ../reports/tables/04_evaluation/comparability_checks.csv and kpi_reproducibility.csv -->

### 6.2 Overall quality and reported KPIs

*Table 4. Recommendation per method against the current configuration. Sources: [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv), [`hypervolume.csv`](../reports/tables/04_evaluation/hypervolume.csv).*

<!-- TODO: table -->

*Table 4b. Best-server RSRP and SINR per band. Source: [`band_kpis.csv`](../reports/tables/04_evaluation/band_kpis.csv).*

<!-- TODO: table -->

![KPI comparison](../reports/figures/04_evaluation/kpi_comparison.png)

*Figure 4. Network KPIs per method against the current configuration. Source: `../reports/figures/04_evaluation/kpi_comparison.png`.*

![Objective comparison](../reports/figures/04_evaluation/objective_comparison.png)

*Figure 5. Objectives per method against the current configuration. Source: `../reports/figures/04_evaluation/objective_comparison.png`.*

<!-- TODO: findings -->

### 6.3 Did the search matter?

*Table 5. Hypervolume gain per method. Source: [`hypervolume.csv`](../reports/tables/04_evaluation/hypervolume.csv).*

<!-- TODO: table -->

*Table 6. Best hypervolume, hole rate and overlap rate reached after a fixed number of evaluations. Source: [`sample_efficiency.csv`](../reports/tables/04_evaluation/sample_efficiency.csv).*

<!-- TODO: table -->

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Figure 6. Hypervolume found so far against evaluations. Source: `../reports/figures/04_evaluation/search_progress.png`.*

![MORBO evaluations](../reports/figures/03b_morbo/morbo_evaluations.png)

*Figure 7. Every MORBO evaluation, by what proposed it. Source: `../reports/figures/03b_morbo/morbo_evaluations.png`.*

<!-- TODO: paired gain, from paired_gain_morbo_vs_random.csv; the MORBO proposals against its Sobol points, from ../reports/tables/03b_morbo/morbo_evaluations_by_proposer.csv -->

![Coverage vs separation trade-off](../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png)

*Figure 8. Every evaluated configuration on the two searched objectives, with each method's front and recommendation. Source: `../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png`.*

### 6.4 Is the result robust?

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band. Source: [`band_kpis.csv`](../reports/tables/04_evaluation/band_kpis.csv).*

<!-- TODO: table -->

*Table 7b. Coverage class by area and by demand, with demand counted in UE reports. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/04_evaluation/coverage_by_area_and_demand.csv).*

<!-- TODO: table -->

*Table 8. Overlapping co-band neighbours per configuration. Source: [`overlap_neighbour_summary.csv`](../reports/tables/04_evaluation/overlap_neighbour_summary.csv).*

<!-- TODO: table -->

![Coverage before and after, 2600 MHz](../reports/figures/04_evaluation/coverage_before_after_b2600.png)

*Figure 9. 2600 MHz best-server RSRP before and after the recommendation, and the tiles that crossed the hole threshold; `coverage_before_after_b1800.png` and `coverage_before_after_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/coverage_before_after_b2600.png`.*

![RSRP change maps, 2600 MHz](../reports/figures/04_evaluation/rsrp_change_maps_b2600.png)

*Figure 10. Change in 2600 MHz best-server RSRP for each method's recommendation; `rsrp_change_maps_b1800.png` and `rsrp_change_maps_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/rsrp_change_maps_b2600.png`.*

<!-- TODO: findings -->

### 6.5 Capacity impact

*Table 9. UE service and median served SINR per band. Sources: [`ue_service_summary.csv`](../reports/tables/04_evaluation/ue_service_summary.csv), [`band_layer_summary.csv`](../reports/tables/04_evaluation/band_layer_summary.csv).*

<!-- TODO: table -->

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Figure 11. Serving-band mix per configuration. Source: `../reports/figures/04_evaluation/serving_band_mix.png`.*

![Sector-band throughput](../reports/figures/04_evaluation/sector_band_throughput.png)

*Figure 12. Median estimated throughput per sector-band, current and recommended. Source: `../reports/figures/04_evaluation/sector_band_throughput.png`.*

<!-- TODO: findings, from sector_impact.csv and sector_band_load.csv -->

### 6.6 Recommended tilt changes

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 13. Tilt change per sector and band in the recommendation. Source: `../reports/figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the recommendation. Negative Δ is an uptilt. Source: [`tilt_movement_summary.csv`](../reports/tables/04_evaluation/tilt_movement_summary.csv).*

<!-- TODO: table; per-sector values in ../reports/tables/04_evaluation/recommended_tilt.csv and ../reports/outputs/tilt_options_<method>.csv -->

<!-- TODO: findings -->

### 6.7 Cost

*Table 11. Search cost. Source: [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv).*

<!-- TODO: table -->

### 6.8 Limitations

<!-- TODO: one scenario; seeds per method; winner's curse under one solver seed; inter-band interference priced nowhere; nothing weighted by demand in the searched objectives; the simplified capacity model; tilts at the bounds and no movement penalty; no held-out validation -->

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Random search | MORBO |
|---|---|---|
| 1. Hypervolume | <!-- TODO --> | <!-- TODO --> |
| 2. Reported KPIs | <!-- TODO --> | <!-- TODO --> |
| 3. Search effectiveness | <!-- TODO --> | <!-- TODO --> |
| 4. Robustness | <!-- TODO --> | <!-- TODO --> |
| 5. Cost | <!-- TODO --> | <!-- TODO --> |

**Conclusions.** <!-- TODO -->

**What this means for a RAN engineer today.** <!-- TODO -->

**Recommendations.** <!-- TODO -->

---

## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/scenes/hanoi/scene.xml` (not in Git), with the sector table `sectors.csv`, the manifest `scenario.json` and the generator record `synthetic.json` beside it |
| Layout | 4 nodes, 1,732 m triangle plus centroid (1,000 m centre to corner), 3 sectors at 45° / 165° / 285°, 25 m masts |
| Tilt | Current 10° on every band and sector; bounds [0°, 20°]; 0.1° lattice |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6; edge percentile 5 (`LOW_PERCENTILE` in `src/kpi/quality.py`, a code constant) |
| Objectives | coverage and separation searched, throughput recorded; hypervolume against the origin; rounded to 6 significant digits (ADR 0003) |
| Capacity | max-throughput sector selection over an equal share of 0.8 × `max_prb`, candidates above −120 dBm, SCS 15 kHz, connection in report-time order |
| Noise | k·T·SCS per resource element at 298.15 K (`simulation.radio_map.bands[].scs_hz`), no receiver noise figure |
| Search | seed 42 (`optim.seed`); random and MORBO 8 + 64; MORBO batch 3, one trust region, side 0.8 / 0.01, failure tolerance 12 evaluations, improvement 10⁻³, perturbed dimensions 20, 2048 candidates per region and point |

Runs used in this report: <!-- TODO: run directories under ../outputs/optim/ -->

To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`. Notebooks 03a and 03b skip any method and seed that already has a run under `optim.output.dir` (`outputs/optim/`), so clear that directory first to re-search.

Each run's `run.json` records the resolved configuration, the scenario ID and a `provenance` block: the Git commit, whether the working tree was dirty, and the numpy, scipy, torch, botorch, gpytorch and sionna-rt versions.

Every path a stage writes is configurable, so a trial run can be kept apart from the real one. The notebooks read extra Hydra overrides from `BAND_TILT_OVERRIDES`, and `reports.figures_dir` and `reports.tables_dir` (`configs/config.yaml`) set where their tables and figures go. A small-budget check keeps every output under `outputs/smoke/`, executes each notebook to a copy (`jupyter nbconvert --to notebook --execute notebooks/<nb>.ipynb --output-dir outputs/smoke/notebooks`, 00 through 04 in order), then runs `python -m src.evaluation.run` with the same overrides:

```text
BAND_TILT_OVERRIDES="simulation.input.ue_file=outputs/smoke/data/ue_positions.csv
  simulation.input.sectors_file=outputs/smoke/data/sectors.csv
  simulation.input.manifest_file=outputs/smoke/data/scenario.json
  scenario.output.record_file=outputs/smoke/data/synthetic.json
  simulation.output.radio_map_file=outputs/smoke/data/radio_map.npz
  data.output.ue_file=outputs/smoke/data/ue.parquet
  optim.output.dir=outputs/smoke/optim optim.output.deliverable_dir=outputs/smoke/reports/outputs
  reports.figures_dir=outputs/smoke/reports/figures reports.tables_dir=outputs/smoke/reports/tables
  ++optim.method.budget.n_init=4 ++optim.method.budget.n_iter=4"
```

The variable is one line, space-separated. The scene, ray-tracing fidelity and seeds keep their defaults.

### Appendix B. Index of generated tables and figures

<!-- TODO: list per stage, from ../reports/tables/ and ../reports/figures/ -->

Deliverables per method are in `../reports/outputs/`: `solutions_<method>.csv` (the Pareto front with every measure and its delta, the recommendation first) and `tilt_options_<method>.csv` (the tilt table of each of those solutions).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy, "Multi-objective Bayesian optimization over high-dimensional search spaces," in *Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 180, pp. 507–517, 2022.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[5] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
