# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report, 3GPP SMa layout. Every number, table and figure belongs to the optimization runs listed in Appendix A, read by notebooks `00_simulation` through `04_evaluation`, and to the configuration in `configs/`. Paths are relative to this file (`docs/`): generated tables and figures are under `../reports/`, run directories under `../outputs/`. RSRP, interference and noise are all per resource element: thermal noise is k·T over one 15 kHz subcarrier, not over the channel bandwidth. Eleven KPIs are recorded for every candidate; the evaluation reports seven over all bands and the coverage, overlap, RSRP and SINR measures per band ([ADR 0003](adr/0003-three-objectives-and-morbo.md)).*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band sector layout when every band on every sector is tuned jointly, not one band at a time.

The study area is an urban scene ray-traced with Sionna-RT. It holds four nodes laid out to the 3GPP Suburban Macro (SMa) scenario of TR 38.901 [6], a 1,299 m inter-site distance and 35 m masts, with three sectors each (twelve in all), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 20°]. Every sector-band starts at 10°. A week-long, time-varying UE population was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates are ray-traced and scored on two objectives, coverage and separation, searched jointly; throughput is measured and recorded beside them ([ADR 0003](adr/0003-three-objectives-and-morbo.md)). Eleven KPIs are measured beside the objectives, none of them weighted into them.

Two searches start from the same current configuration with matched budgets and the same seed:
- Sobol random search,
- MORBO, multi-objective trust-region Bayesian optimization.

**Results.** The SMa geometry alone holds the current configuration at a 4.40 % hole rate. From there, MORBO raised the hypervolume of its evaluations 6.9 % over the current configuration's, and random search 3.4 %, from the same initial design. MORBO's recommendation improves five of the seven network KPIs: hole rate 4.40 % → 4.30 %, weak-coverage rate 29.4 % → 24.6 %, overlapping neighbours per covered tile 1.09 → 1.01, and median and mean estimated throughput 51.7 → 55.5 and 80.4 → 85.0 Mbit/s. Co-band overlap rises 34.9 % → 36.6 %, and cell-edge throughput stays at 0 Mbit/s. Random search's recommendation improves only the two throughput KPIs and worsens four, overlap most (34.9 % → 48.7 %).

**Caveats.** One scenario, one search seed per method and one solver seed, so no confidence interval is available and solver noise is unmeasured: a 0.1-point change in hole rate may be inside it. The layout takes four of SMa's 19 sites over an urban scene, at one outdoor UE height. The capacity model is an equal-share Shannon estimate, with no scheduler or MCS cap.

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
- **From trial-and-error to choosing between measured options.** A run is 73 ray-traced evaluations: 2.9 minutes of ray tracing and 7.2 minutes of wall clock for MORBO, 3.4 for random search, on one 4 GB RTX 3050 laptop GPU. MORBO's run offers 18 Pareto-optimal configurations to choose from.
- **From band-by-band to layer-aware decisions.** The recommendation uptilts 700 MHz most on average (−1.6°), widening the coverage floor, while 2600 MHz nets out near zero (−0.2°) through large opposite moves per sector that shift its traffic between sectors and onto the lower layers (Sections 6.5 and 6.6).
- **From reacting to complaints to planning a rollout.** The sector-impact table says which carriers to watch, and the front offers alternatives when a proposed tilt is mechanically or operationally unacceptable.

**What stays your job.** The study does not replace engineering judgement, and Section 6.8 lists why:
- **Field validation.** One simulated city is not your network. Results are uncalibrated against drive tests, and nothing has been tested on a held-out scenario.
- **Operational limits.** The search does not penalise antenna movement. The recommendation moves all 36 sector-band tilts, by 4.5° to 5.4° on average per band and by up to 9.4°.
- **The model's blind spots.** There is no MCS cap, no scheduler, no mobility and no inter-band interference in the objectives.

The narrow claim this run supports: on one SMa-geometry scenario, at a budget of 73 evaluations and one seed, MORBO found a joint tilt configuration that improves five of seven network KPIs at the cost of 1.7 points of co-band overlap, and random search at the same budget did not.

## 2. Problem Definition

**Operational statement.** Given a multi-band site layout and its current tilts, propose new electrical tilts for every sector and band that close coverage holes and weak areas, keep co-band overlap under control, and raise the signal quality and throughput users get, while each band keeps the role its propagation suits. The proposal must come with its measured effect on every KPI and its tilt delta per antenna, so an engineer can review it before any RET change.

**Network.** The study area is a local city scene (`data/scenes/hanoi/scene.xml`), rasterised onto a grid of 20 m tiles: 326 × 310 tiles (101,060) over 6,200 × 6,520 m.
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 2,250 m side, 1,299 m from centre to corner: the SMa inter-site distance (TR 38.901 Table 7.2-5 [6]). SMa specifies a 19-site hexagonal grid; these four nodes are a subset of it. Each mast is snapped to the nearest open ground, which puts one corner node 1,310 m from the centre.
- **Sectors.** Each node has three sectors at azimuths 45°, 165° and 285°, on 35 m masts, the SMa base-station height. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
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
- **Coverage.** 4.40 % of tiles are holes, 29.4 % weak and 66.2 % good. The holes are fragmented: 1,708 separate regions, 1,247 of them a single tile, the largest holding 22.7 % of the hole area. 1,876 of the 4,445 hole tiles have no path to any sector at all. Within 1 km of a node the hole rate is 0.13 %; beyond it, 5.87 %.
- **Band behaviour.** Per layer, the hole share is 5.6 % at 700 MHz, 11.3 % at 1800 MHz and 14.1 % at 2600 MHz. 700 MHz is the strongest band on 93.9 % of the covered area, but the serving rule puts 88.1 % of it on 2600 MHz, the band with the most PRBs.
- **Overlap.** 34.9 % of tiles have at least one co-band neighbour within 6 dB, 1.09 neighbours per covered tile on average (median 0, 90th percentile 3). 20.9 % of covered tiles have three or more, at a median 1,371 m from the nearest node.
- **Service.** 16.4 % of UE reports fall on hole tiles, 13.9 % with no path to any sector at all. Of all UE reports, 62.5 % are served on 2600 MHz, 12.9 % on 1800 MHz and 8.2 % on 700 MHz. One of the four demand hotspots (1,866 UE reports) is centred 3,297 m from the nearest node, and no layer reaches its centre at the current tilts (`../reports/tables/01_eda/hotspots.csv`).
- **Demand.** Holes are 4.4 % of the area but 16.4 % of the demand (Table 3): the hole rate understates what users see.

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
| Scenario ID | `scn_6966862f76b0ae52` |
| Grid | 326 × 310 tiles of 20 m (6,200 × 6,520 m) |
| Nodes / sectors / sector-band pairs | 4 / 12 / 36 |
| Inter-site distance | 1,299 m (SMa, TR 38.901 Table 7.2-5) |
| Mast height | 35 m (SMa) |
| Current tilt (2600 / 1800 / 700 MHz) | 10° / 10° / 10° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | 10,087 |
| UE positions with no path to any sector | 13.9 % |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 4,445 | 4.4 % | 16.4 % |
| Weak (−120 to −90 dBm) | 29,735 | 29.4 % | 22.2 % |
| Good (> −90 dBm) | 66,880 | 66.2 % | 61.4 % |

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
   - The share of tiles each band reaches is 97.1 % at 700 MHz, 95.1 % at 1800 MHz and 95.2 % at 2600 MHz; the median RSRP over those tiles is −84.1, −91.3 and −95.5 dBm.
3. **Service.** Every UE row is served from the radio map at its tile, in the search and in the evaluation alike. No measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the UE table, the manifest and the radio map against the contract checks in `../reports/tables/02_preprocessing/verification_checks.csv`. All 28 held.
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
- **Seeds.** One search seed per method, 42, and one solver seed shared by every evaluation. Ray tracing took about 2.4 s per evaluation (Table 11), so each extra seed costs a few minutes.

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

All 23 comparability checks hold: both runs optimised the baseline's scenario, kept their radio maps, and match the baseline's grid, solver settings, bands, carrier frequencies and KPI definition (`comparability_checks.csv`). Recomputing each recommendation's KPIs from its archived radio map reproduces the recorded values: the hole, weak and overlap rates and the objectives exactly, the throughput KPIs to within 3 × 10⁻⁶ Mbit/s (`kpi_reproducibility.csv`). Repeatability across seeds is untested: each method ran one seed.

### 6.2 Overall quality and reported KPIs

*Table 4. Recommendation per method against the current configuration. Sources: [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv), [`hypervolume.csv`](../reports/tables/04_evaluation/hypervolume.csv).*

| KPI | Direction | Current | MORBO | Random search |
|---|:-:|---:|---:|---:|
| Coverage hole rate | ↓ | 4.40 % | **4.30 %** | 4.50 % |
| Weak-coverage rate | ↓ | 29.4 % | **24.6 %** | 29.6 % |
| Co-band overlap rate | ↓ | **34.9 %** | 36.6 % | 48.7 % |
| Overlap neighbours per covered tile | ↓ | 1.09 | **1.01** | 1.20 |
| Cell-edge estimated throughput, p05 [Mbit/s] | ↑ | 0.0 | 0.0 | 0.0 |
| Median estimated throughput [Mbit/s] | ↑ | 51.7 | 55.5 | **61.5** |
| Mean estimated throughput [Mbit/s] | ↑ | 80.4 | 85.0 | **88.6** |
| Coverage objective | ↑ | 0.9560 | **0.9570** | 0.9550 |
| Separation objective | ↑ | 0.6778 | **0.6994** | 0.6388 |
| Throughput objective, mean ln(1 + R) | ↑ | 3.513 | 3.566 | **3.628** |
| Network KPIs better / worse / unchanged | | | 5 / 1 / 1 | 2 / 4 / 1 |
| Hypervolume, every evaluation | ↑ | 2.276 | **2.433** | 2.353 |

*Table 4b. Best-server RSRP and SINR per band. Source: [`band_kpis.csv`](../reports/tables/04_evaluation/band_kpis.csv).*

| Band | Configuration | RSRP p50 [dBm] | RSRP p05 [dBm] | SINR p50 [dB] | SINR p05 [dB] |
|---|---|---:|---:|---:|---:|
| 2600 MHz | Current | −93.4 | −113.2 | 12.7 | −0.93 |
| | MORBO | −93.0 | −113.9 | 15.0 | −0.92 |
| | Random search | −94.1 | −114.5 | 12.1 | −1.32 |
| 1800 MHz | Current | −89.9 | −110.3 | 12.2 | −0.98 |
| | MORBO | −89.0 | −110.6 | 13.9 | −1.10 |
| | Random search | −89.0 | −110.1 | 13.3 | −1.05 |
| 700 MHz | Current | −83.6 | −105.3 | 11.5 | −1.12 |
| | MORBO | −82.0 | −104.5 | 13.9 | −0.92 |
| | Random search | −86.0 | −108.2 | 9.9 | −1.52 |

![KPI comparison](../reports/figures/04_evaluation/kpi_comparison.png)

*Figure 4. Network KPIs per method against the current configuration. Source: `../reports/figures/04_evaluation/kpi_comparison.png`.*

![Objective comparison](../reports/figures/04_evaluation/objective_comparison.png)

*Figure 5. Objectives per method against the current configuration. Source: `../reports/figures/04_evaluation/objective_comparison.png`.*

**Findings.**
- **The geometry did most of the work.** At the SMa spacing and mast height the current configuration is already under a 5 % hole rate. Tilt moves it by tenths of a point: MORBO's recommendation by −0.10 points, random search's by +0.10.
- **MORBO trades a little overlap for everything else.** Its recommendation improves weak coverage most (−4.8 points) and raises median SINR on every band, by 1.6 to 2.4 dB. Overlap rises 1.7 points, though Table 7a shows every band's own overlap falls; the all-bands KPI rises because the remaining overlap is spread over more tiles.
- **Random search bought throughput with overlap.** Its recommendation has the highest median and mean throughput, but overlap rises 13.9 points and the separation objective falls 0.039. Its 700 MHz layer loses 2.4 dB of median RSRP and 1.5 dB of median SINR, the coverage floor weakening.
- **Cell-edge throughput cannot move.** 16.4 % of UE reports sit on hole tiles in every configuration, more than 5 %, so the 5th percentile is 0 Mbit/s throughout.

### 6.3 Did the search matter?

*Table 5. Hypervolume gain per method. Source: [`hypervolume.csv`](../reports/tables/04_evaluation/hypervolume.csv).*

| Method | Seed | Current configuration | Initial design | Every evaluation | Gain over current | Pareto points | Best evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|
| MORBO | 42 | 2.2762 | 2.3388 | 2.4325 | +6.9 % | 18 | 66 |
| Random search | 42 | 2.2762 | 2.3388 | 2.3528 | +3.4 % | 9 | 65 |

*Table 6. Best hypervolume, hole rate and overlap rate reached after a fixed number of evaluations. Source: [`sample_efficiency.csv`](../reports/tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Hypervolume, MORBO | Hypervolume, random | Hole rate, MORBO | Hole rate, random | Overlap rate, MORBO | Overlap rate, random |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.3388 | 2.3388 | 4.40 % | 4.40 % | 34.9 % | 34.9 % |
| 25 | 2.3544 | 2.3388 | 4.25 % | 4.40 % | 34.9 % | 34.9 % |
| 50 | 2.4010 | 2.3468 | 4.15 % | 4.26 % | 34.9 % | 34.9 % |
| 73 | 2.4325 | 2.3528 | 4.15 % | 4.26 % | 34.9 % | 34.9 % |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Figure 6. Hypervolume found so far against evaluations. Source: `../reports/figures/04_evaluation/search_progress.png`.*

![MORBO evaluations](../reports/figures/03b_morbo/morbo_evaluations.png)

*Figure 7. Every MORBO evaluation, by what proposed it. Source: `../reports/figures/03b_morbo/morbo_evaluations.png`.*

**Findings.**
- **The model earned the gain.** Both methods share the same 8-point initial design, so everything after it is the search's own. MORBO's hypervolume exceeds random search's by 0.080 on the one seed pair (`paired_gain_morbo_vs_random.csv`); with one pair there is no confidence interval or test.
- **MORBO pulls ahead early and keeps climbing.** By 25 evaluations it is already above the hypervolume random search ends at after 73, and its best hole rate (4.15 %) is lower than random search's (4.26 %) by evaluation 50.
- **Every Pareto point is a MORBO proposal.** None of the 8 Sobol points is on MORBO's front, and all 18 front points are among its 64 proposals, whose mean coverage, separation and throughput objectives all exceed the Sobol points' (`../reports/tables/03b_morbo/morbo_evaluations_by_proposer.csv`).
- **No evaluation beat the current overlap.** The best overlap rate found by either method stays at the current 34.9 %: in this geometry no configuration the searches tried reduced the all-bands overlap rate.
- **The recommendation is not the lowest hole rate found.** MORBO's lowest hole rate is 4.15 %, but its recommendation is the evaluated point with the largest hypervolume contribution, at 4.30 %. `../reports/outputs/solutions_morbo.csv` lists the alternatives on the front.

![Coverage vs separation trade-off](../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png)

*Figure 8. Every evaluated configuration on the two searched objectives, with each method's front and recommendation. Source: `../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png`.*

### 6.4 Is the result robust?

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band. Source: [`band_kpis.csv`](../reports/tables/04_evaluation/band_kpis.csv).*

| Band | Current | MORBO | Random search |
|---|---:|---:|---:|
| 2600 MHz | 23.8 % | 20.8 % | 24.2 % |
| 1800 MHz | 25.2 % | 23.4 % | 24.5 % |
| 700 MHz | 28.2 % | 24.9 % | 31.2 % |
| All bands (KPI) | 34.9 % | 36.6 % | 48.7 % |

*Table 7b. Coverage class by area and by demand, with demand counted in UE reports. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Coverage class | Current: area | Current: demand | MORBO: area | MORBO: demand | Random: area | Random: demand |
|---|---:|---:|---:|---:|---:|---:|
| Hole | 4.40 % | 16.40 % | 4.30 % | 16.58 % | 4.50 % | 16.36 % |
| Weak | 29.4 % | 22.2 % | 24.6 % | 19.4 % | 29.6 % | 20.5 % |
| Good | 66.2 % | 61.4 % | 71.1 % | 64.1 % | 65.9 % | 63.2 % |

*Table 8. Overlapping co-band neighbours per configuration. Source: [`overlap_neighbour_summary.csv`](../reports/tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean, covered tiles | 0 neighbours | 1 | 2 | 3 or more |
|---|---:|---:|---:|---:|---:|
| Current | 1.09 | 63.5 % | 9.4 % | 6.2 % | 20.9 % |
| MORBO | 1.01 | 61.8 % | 13.8 % | 8.2 % | 16.3 % |
| Random search | 1.20 | 49.0 % | 22.6 % | 11.7 % | 16.8 % |

![Coverage before and after, 2600 MHz](../reports/figures/04_evaluation/coverage_before_after_b2600.png)

*Figure 9. 2600 MHz best-server RSRP before and after the recommendation, and the tiles that crossed the hole threshold; `coverage_before_after_b1800.png` and `coverage_before_after_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/coverage_before_after_b2600.png`.*

![RSRP change maps, 2600 MHz](../reports/figures/04_evaluation/rsrp_change_maps_b2600.png)

*Figure 10. Change in 2600 MHz best-server RSRP for each method's recommendation; `rsrp_change_maps_b1800.png` and `rsrp_change_maps_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/rsrp_change_maps_b2600.png`.*

**Findings.**
- **Area gains did not reach demand in the holes.** MORBO's recommendation shrinks hole area 0.10 points but raises the share of UE reports on hole tiles from 16.40 % to 16.58 %: the tiles it closed carry little demand, and none of the searched objectives is weighted by demand.
- **Weak coverage improves by area and by demand.** MORBO moves 4.8 points of area and 2.9 points of demand from weak to good.
- **Overlap is thinner but wider under MORBO.** The share of covered tiles with three or more neighbours falls from 20.9 % to 16.3 %, while tiles with one or two neighbours grow. Random search shifts far more tiles from zero neighbours to one or two.

### 6.5 Capacity impact

*Table 9. UE service and median served SINR per band. Sources: [`ue_service_summary.csv`](../reports/tables/04_evaluation/ue_service_summary.csv), [`band_layer_summary.csv`](../reports/tables/04_evaluation/band_layer_summary.csv).*

| Configuration | Served on 2600 MHz | Served on 1800 MHz | Served on 700 MHz | Not served | Served SINR p50, 2600 / 1800 / 700 MHz [dB] |
|---|---:|---:|---:|---:|---|
| Current | 62.5 % | 12.9 % | 8.2 % | 16.4 % | 11.2 / 15.3 / 19.8 |
| MORBO | 55.7 % | 16.1 % | 11.6 % | 16.6 % | 13.1 / 16.9 / 17.9 |
| Random search | 56.0 % | 19.1 % | 8.6 % | 16.4 % | 17.4 / 17.7 / 20.5 |

UE reports: 10,087 in every configuration. "Not served" is one minus the three served shares.

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Figure 11. Serving-band mix per configuration. Source: `../reports/figures/04_evaluation/serving_band_mix.png`.*

![Sector-band throughput](../reports/figures/04_evaluation/sector_band_throughput.png)

*Figure 12. Median estimated throughput per sector-band, current and recommended. Source: `../reports/figures/04_evaluation/sector_band_throughput.png`.*

**Findings** (`sector_impact.csv`, MORBO's recommendation).
- **Traffic moves off 2600 MHz onto the lower layers.** The 2600 MHz share falls 6.8 points; 1800 MHz gains 3.2 and 700 MHz 3.4. Served SINR on 2600 MHz rises 1.9 dB, while on 700 MHz it falls 1.8 dB as the layer takes more users.
- **The largest moves are 2600 MHz downtilts that shed load.** n2s1 at 2600 MHz, downtilted 4.2°, serves 664 fewer reports (963 → 299) and n1s0, downtilted 5.4°, 384 fewer (521 → 137); the median throughput of the UEs that stay rises 18 to 21 Mbit/s.
- **Uptilted 2600 MHz carriers pick that load up.** n3s2 at 2600 MHz, uptilted 8.1°, serves 363 more reports (769 → 1,132) at 4.3 dB better median SINR, and n0s2, uptilted 7.4°, 214 more.
- **Watch first** the sector-bands at the top of `sector_impact.csv`, which is ranked by the size of the change in served reports.

### 6.6 Recommended tilt changes

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 13. Tilt change per sector and band in the recommendation. Source: `../reports/figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the recommendation. Negative Δ is an uptilt. Source: [`tilt_movement_summary.csv`](../reports/tables/04_evaluation/tilt_movement_summary.csv).*

The recommendation shown is MORBO's, the method with the larger hypervolume. Per-sector values are in `../reports/tables/04_evaluation/recommended_tilt.csv` and `../reports/outputs/tilt_options_morbo.csv`.

| Band | Sectors moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 of 12 | 4.66 | 9.2 | −0.24 |
| 1800 MHz | 12 of 12 | 4.51 | 8.9 | −1.31 |
| 700 MHz | 12 of 12 | 5.41 | 9.4 | −1.57 |

**Findings.**
- **Every antenna moves.** All 36 tilts change, 24 up and 12 down, by up to 9.4°. No tilt sits at a bound (0° or 20°), so the box did not constrain the recommendation.
- **The low band uptilts most.** 700 MHz moves furthest on average and toward the horizon (−1.6°), consistent with its role as the coverage floor; 2600 MHz nets out near zero through large per-sector moves in both directions.
- **The movement is unpenalised.** A rollout of 36 RET changes is a large change request; the front in `../reports/outputs/solutions_morbo.csv` is the place to look for a configuration that moves less.

### 6.7 Cost

*Table 11. Search cost. Source: [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv).*

| Method | Seed | Run | Evaluations | Best evaluation | Ray tracing [min] | Wall clock [min] |
|---|---:|---|---:|---:|---:|---:|
| MORBO | 42 | `2026-10-08_07-42-52` | 73 | 66 | 2.85 | 7.24 |
| Random search | 42 | `2026-10-08_07-38-57` | 73 | 65 | 2.87 | 3.42 |

Ray tracing costs the same for both, about 2.4 s per evaluation. The rest of MORBO's wall clock is spent outside the ray tracer, in its model fitting and candidate selection.

### 6.8 Limitations

- **One scenario.** One scene, one UE draw and one layout. Nothing has been validated on a held-out scenario.
- **A partial SMa.** The layout follows SMa's inter-site distance and mast height (TR 38.901 Table 7.2-5 [6]) but uses 4 of its 19 sites, over an urban scene rather than a suburban one. UEs sit on one outdoor plane at 1.5 m; SMa's 80 % indoor UEs on several floors are not modelled. The incumbent tilt is the project's 10° on every band.
- **One seed per method.** No confidence interval, test or repeatability measure is available (Tables 4 and 5).
- **The winner's curse.** Every evaluation shares one solver seed, and the recommendation is the best of 73 under that noise, so its scores are biased upward. Solver noise per KPI is unmeasured, so changes of a tenth of a point in hole rate may be within it.
- **Inter-band interference priced nowhere.** The objectives and the KPIs treat each band's interference separately.
- **Demand is not in the searched objectives.** The hole-rate gain came on tiles with little demand: the share of UE reports on hole tiles rose slightly (Table 7b).
- **A simplified capacity model.** Equal-share Shannon throughput with no scheduler, MCS cap or mobility. Cell-edge throughput is 0 Mbit/s in every configuration because more than 5 % of UE reports are on hole tiles.
- **No movement penalty.** The recommendation moves every antenna (Table 10). No tilt reached a bound.
- **Unreachable demand.** One demand hotspot is centred 3.3 km from the nearest node, where no layer reaches at the current tilts. In every configuration reported here, at least 16 % of UE reports remain on hole tiles.

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Random search | MORBO |
|---|---|---|
| 1. Hypervolume | +3.4 % over current | **+6.9 % over current** |
| 2. Reported KPIs | 2 better, 4 worse; overlap +13.9 points | **5 better, 1 worse**; overlap +1.7 points |
| 3. Search effectiveness | Best hole rate 4.26 %; 9 Pareto points | **Past random search's final hypervolume by evaluation 25**; 18 Pareto points, all its own proposals |
| 4. Robustness | Weakens the 700 MHz floor (−2.4 dB median RSRP) | Better weak coverage by area and demand; hole-tile demand share up 0.18 points |
| 5. Cost | 73 evaluations, 3.4 min | 73 evaluations, 7.2 min |

**Conclusions.**
1. Laid out to SMa's 1,299 m inter-site distance with 35 m masts, the current 10° configuration already holds the hole rate at 4.40 %, under a 5 % target, before any tilt search.
2. Joint tilt search adds little to the hole rate in this geometry: the lowest any search found was 4.15 %, and MORBO's recommendation is at 4.30 %. Its value is elsewhere: 4.8 points less weak coverage, 1.6 to 2.4 dB more median SINR per band and 4.5 Mbit/s more mean throughput, for 1.7 points more co-band overlap.
3. At an equal budget MORBO beat random search on hypervolume and on five of seven KPIs, with one seed each.
4. The hole rate understates the service gap: about 16 % of UE reports sit on hole tiles in every configuration, and one demand hotspot is centred where no layer reaches at the current tilts.

**What this means for a RAN engineer today.** In an SMa-like geometry, site spacing and mast height set the hole rate, and tilt tuning did not close the holes that matter to users. Use the tilt search for weak coverage, SINR and load balance between layers, review the whole front rather than only the first row, and treat a 36-antenna change as a staged rollout, watching the sector-bands at the top of the sector-impact table first.

**Recommendations.**
1. Measure solver noise, by re-tracing the recommendation under several solver seeds, before reading meaning into changes of tenths of a point.
2. Run more search seeds per method, so Tables 4 and 5 carry confidence intervals.
3. Weight the coverage objective by demand, or report a demand-weighted hole rate beside it, so a search cannot close empty tiles while the busy ones stay dark.
4. Treat the unreachable hotspot as a site question rather than a tilt question: model the full 19-site SMa grid, or a site placed for that demand.
5. Add a movement penalty or cap, so a recommendation fits a practical RET change window.

---

## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/scenes/hanoi/scene.xml` (not in Git), with the sector table `sectors.csv`, the manifest `scenario.json` and the generator record `synthetic.json` beside it |
| Layout | 3GPP SMa (TR 38.901 Table 7.2-5): 4 nodes, 2,250 m triangle plus centroid (1,299 m centre to corner), 3 sectors at 45° / 165° / 285°, 35 m masts (`scenario.layout.node_spacing_m`, `mast_height_m`) |
| Tilt | Current 10° on every band and sector; bounds [0°, 20°]; 0.1° lattice |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6; edge percentile 5 (`LOW_PERCENTILE` in `src/kpi/quality.py`, a code constant) |
| Objectives | coverage and separation searched, throughput recorded; hypervolume against the origin; rounded to 6 significant digits (ADR 0003) |
| Capacity | max-throughput sector selection over an equal share of 0.8 × `max_prb`, candidates above −120 dBm, SCS 15 kHz, connection in report-time order |
| Noise | k·T·SCS per resource element at 298.15 K (`simulation.radio_map.bands[].scs_hz`), no receiver noise figure |
| Search | seed 42 (`optim.seed`); random and MORBO 8 + 64; MORBO batch 3, one trust region, side 0.8 / 0.01, failure tolerance 12 evaluations, improvement 10⁻³, perturbed dimensions 20, 2048 candidates per region and point |

Runs used in this report: `../outputs/optim/morbo/2026-10-08_07-42-52/` and `../outputs/optim/random/2026-10-08_07-38-57/`, both seed 42, scenario `scn_6966862f76b0ae52`.

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

| Stage | Tables (`../reports/tables/<stage>/`) | Figures (`../reports/figures/<stage>/`) |
|---|---|---|
| `00_simulation` | `baseline_kpis`, `coverage_by_area_and_demand`, `decision_variables`, `frequency_bands`, `network_configuration`, `node_layout`, `propagation_parameters`, `reach_per_band`, `serving_band_mix`, `study_area`, `ue_distribution`, `ue_measurement_summary` | `coverage_and_overlap_maps`, `rsrp_per_band`, `serving_band_map`, `study_area`, `traffic_model`, `ue_rsrp_distribution` |
| `01_eda` | `band_complementarity`, `band_representation`, `coverage_by_area_and_demand`, `coverage_classes_per_band`, `cross_band_correlation`, `dataset_overview`, `duplicates`, `hole_summary`, `hotspots`, `kpi_summary`, `missing_values`, `overlap_neighbour_summary`, `overlap_per_band`, `physical_checks`, `rsrp_outliers`, `rsrp_statistics`, `schema_checks`, `sector_band_configuration`, `serving_area_per_band`, `serving_band_mix`, `signal_vs_ue_density`, `tilt_summary`, `ue_schema`, `weak_by_band` | `band_complementarity`, `band_propagation`, `coverage_class_map`, `coverage_per_band`, `cross_band_scatter`, `demand_vs_coverage`, `overlap_neighbours`, `rsrp_distribution`, `sector_band_throughput`, `serving_maps`, `signal_vs_ue_density`, `sinr_distribution`, `ue_distribution` |
| `02_preprocessing` | `baseline_kpis`, `baseline_objectives`, `coverage_classes`, `data_quality_summary`, `no_path_by_band`, `optimizer_features`, `overlap_neighbours`, `ue_overview`, `ue_weighted_indicators`, `verification_checks` | `coverage_map`, `network_layout`, `overlap_map`, `rsrp_map`, `serving_multiplicity` |
| `03a_baseline` | `band_kpis`, `baseline_configuration`, `baseline_results`, `best_tilt_random`, `coverage_by_area_and_demand`, `initial_state`, `kpi_comparison_random`, `objective_parameters`, `overlap`, `setup_network`, `setup_simulation`, `setup_users`, `tilt_movement_random`, `ue_service_summary` | `band_kpis`, `coverage_before_after_random`, `kpi_progress`, `rsrp_change_maps`, `search_progress`, `serving_band_mix`, `tilt_movement_random` |
| `03b_morbo` | `band_kpis`, `best_tilt_morbo`, `coverage_by_area_and_demand`, `hypervolume`, `kpi_comparison`, `method_results`, `morbo_configuration`, `morbo_evaluations_by_proposer`, `morbo_pareto_front`, `overlap`, `tilt_movement_morbo`, `ue_service_summary` | `band_kpis`, `coverage_before_after_morbo`, `morbo_evaluations`, `rsrp_change_maps`, `search_progress`, `serving_band_mix`, `tilt_movement_morbo`, `tradeoff_coverage_objective_vs_separation_objective` |
| `04_evaluation` | `band_kpis`, `band_layer_summary`, `candidates`, `comparability_checks`, `convergence`, `coverage_by_area_and_demand`, `experiment_setup`, `hypervolume`, `kpi_reproducibility`, `kpi_scoreboard`, `method_cost`, `overlap_neighbour_summary`, `paired_gain_morbo_vs_random`, `recommended_tilt`, `sample_efficiency`, `sector_band_load`, `sector_impact`, `tilt_movement_summary`, `ue_service_summary` | `band_kpi_panels`, `coverage_before_after_b2600`, `_b1800`, `_b700`, `coverage_class_maps`, `kpi_comparison`, `objective_comparison`, `overlap_neighbour_maps`, `rsrp_change_maps_b2600`, `_b1800`, `_b700`, `search_progress`, `sector_band_throughput`, `serving_band_mix`, `tilt_delta_heatmap`, `tilt_movement`, `tradeoff_coverage_objective_vs_separation_objective`, `tradeoff_hole_rate_vs_overlap_rate`, `ue_throughput_maps` |

Tables are `.csv`, figures `.png`.

Deliverables per method are in `../reports/outputs/`: `solutions_<method>.csv` (the Pareto front with every measure and its delta, the recommendation first) and `tilt_options_<method>.csv` (the tilt table of each of those solutions).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy, "Multi-objective Bayesian optimization over high-dimensional search spaces," in *Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 180, pp. 507–517, 2022.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[5] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.

[6] 3GPP TR 38.901 V19.2.0 (ETSI TR 138 901 V19.2.0, 2026-02), *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-5, evaluation parameters for SMa scenarios.
