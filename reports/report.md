# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-30: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the configuration in `configs/`. Paths are relative to `reports/`. RSRP, interference and noise are all per resource element: thermal noise is k·T over one 15 kHz subcarrier, not over the channel bandwidth. Twelve KPIs are reported ([ADR 0002](../docs/adr/0002-contraharmonic-objective-and-kpi-set.md)). Each band's utility in J is now its strongest cell's share of the band's received power; every J is from runs searched under it, and none compares with an earlier report.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0002](../docs/adr/0002-contraharmonic-objective-and-kpi-set.md)). Per tile, each band scores its strongest cell's share of the band's received power, scaled by how usable that cell's signal is. The tile then takes the *contraharmonic mean* of those band scores, so every covered layer counts in proportion to how well it serves. J lies in [0, 1], is 1 only when every covered band on every tile has one server at or above the weak threshold and no co-band rival above the hole threshold, and has no free parameters. Twelve KPIs were reported beside it, none of them weighted into it.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations; the rule sweep was allowed up to 120.

**Results.**
- **TuRBO** scored highest on J: 0.6706 against 0.6235 currently, a 7.55 % gain. It improved 10 of the 12 reported KPIs, and it has the best overlapping-neighbour count, both SINR percentiles and all three estimated-throughput statistics of the three methods.
- **The rule sweep** reached J = 0.6648 (+6.61 %) in 111 evaluations and improved 8 of the 12 KPIs. It has the best hole, weak-coverage, overlap-rate and RSRP figures and the lowest UE service failure rate.
- **Random search** reached J = 0.6545 (+4.96 %), also improving 8 KPIs and worsening 4. It is not the best of the three on any KPI.

Every method worsens the band-collapsed co-band overlap rate and overlapping neighbours per covered tile. The rule sweep and random search also worsen cell-edge SINR and cell-edge throughput; TuRBO improves both, and has the lowest overlap on each of the three bands taken separately.

Two findings shape how these should be read.
- **TuRBO's KPI profile is one draw.** Its two runners-up sit within 0.00013 of the winner's J, with overlap rates from 0.3213 to 0.3285 (Section 6.4).
- **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.4). Nothing in the objective stops a search from buying J by switching a weak layer off. On this run's TuRBO winner, 76.7 % of tiles have such a band (Section 6.8). The pipeline does not compute it.

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
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side.
- **Cells.** Each node has three sectors at azimuths 45°, 165° and 285°, on 25 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve cells and 36 cell-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{36}$$

Results are reported as offsets from the current configuration: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell (`tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another cell of the same band within 6 dB of that band's strongest cell;
- fails fewer UEs and raises the throughput the served ones can expect;
- preserves each band's physical role.

**Starting condition** (`tables/01_eda/`):
- **Coverage.** 11.2 % of the grid is a coverage hole, 30.7 % is weakly covered (−120 to −90 dBm) and 58.1 % has good coverage. Holes are a periphery effect: 0.4 % of tiles within 1 km of a node are holes, against 14.3 % beyond (`hole_summary.csv`). Half the hole area — 5,755 tiles of 11,368 — has no propagation path on any band at all.
- **Band behaviour.**
  - Alone, 700 MHz leaves 13.8 % of the grid in a hole, 1800 MHz 23.5 % and 2600 MHz 28.8 % (`coverage_classes_per_band.csv`).
  - By raw signal, 700 MHz is the strongest layer on 91.9 % of the covered area.
  - The max-throughput serving rule still puts 77.2 % of the covered area on 2600 MHz, whose 216-PRB carriers offer the most throughput, against 15.8 % on 700 MHz (`serving_area_per_band.csv`). The gap between which layer is strongest and which layer serves is the central tension in this configuration.
- **Overlap.** 31.6 % of tiles have at least one overlapping co-band neighbour, with a mean of 0.96 neighbours per covered tile (`overlap_neighbour_summary.csv`).
- **Service.** The UE service failure rate is 21.2 % of UE rows (`tables/00_simulation/serving_band_mix.csv`). The serving rule refuses nobody, so every one of them stands on a hole tile, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Counted in UE reports, 21.2 % stands in holes, 23.7 % on weak tiles and 55.1 % on good ones (Table 3).

![Study area](figures/00_simulation/study_area.png)

*Figure 1. Study area: twelve cells on four nodes and a sample of UE positions over the scene. Source: `figures/00_simulation/study_area.png`.*

![RSRP per band](figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the current tilts. Source: `figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](figures/01_eda/demand_vs_coverage.png)

*Figure 3. UE demand beside signal strength at the current tilts. Source: `figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. The PRB limits are N_RB at 15 kHz SCS, TS 38.101-1 Table 5.3.2-1 [4]. Source: [`tables/00_simulation/frequency_bands.csv`](tables/00_simulation/frequency_bands.csv).*

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

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 11,368 | 11.2 % | 21.2 % |
| Weak (−120 to −90 dBm) | 31,017 | 30.7 % | 23.7 % |
| Good (> −90 dBm) | 58,675 | 58.1 % | 55.1 % |

## 3. Proposed Solutions

All three solutions search the same bounded tilt box (`src/optim/space.py`) with the same Sionna-RT evaluator (`src/optim/evaluator.py`). The evaluator builds the scene once and uses one fixed solver seed, so every candidate shares the same Monte-Carlo noise. The solutions also share one definition of "better": the objective J (Section 3.4). They differ only in where they look.

Each run has four steps (`src/optim/run.py`, `src/optim/report.py`):
1. Evaluate the current configuration.
2. Search, scoring every candidate on all UEs.
3. Re-trace the winner once, to archive its radio map.
4. Publish a shortlist of the `optim.n_solutions` = 4 highest-J configurations, always including the current one.

### 3.1 Rule-based per-band sweep

This mimics the heuristic an operator would use. Every cell on a band shares one tilt, which collapses the 36 dimensions to three. Coordinate descent then passes over the bands: it tries `n_steps = 10` evenly spaced tilts across the band's range, keeps the best, and moves to the next band, for `n_rounds = 4` passes. A value equal to the current one is skipped, so the budget is at most 3 × 10 × 4 = 120 evaluations.

The run is deterministic and spent 110 sweep evaluations plus the incumbent. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. Each candidate perturbs every dimension of the centre with probability min(k / 36, 1), k = `perturbed_dimensions` = 5, and the search Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. A round improves when it beats the best by 10⁻³ of its magnitude (`improvement`). The region doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.
- **Seeds.** The initial design is the Sobol sequence seeded with `optim.seed`, the same one random search draws from. Each restart design, and each round's GP fit, candidate pool and perturbation mask, is seeded from `numpy.random.SeedSequence([optim.seed, purpose, index])`, keyed by the restart count or the history length. A derived seed, unlike an offset from `optim.seed`, cannot repeat another search seed's draws, so the runs of a seed sweep stay independent replicates.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0001 and ADR 0002.

### 3.4 The objective every solution maximises

The objective is [ADR 0002](../docs/adr/0002-contraharmonic-objective-and-kpi-set.md), implemented in `src/optim/objective.py` on top of `src/kpi/overlap.py::effective_coverage`:

$$J = \frac{1}{|G|}\sum_{g\in G} \frac{\sum_b u_{bg}^2}{\sum_b u_{bg}},
\qquad u_{bg} = \frac{s_{bg}}{1 + \sum_{i} 10^{(R_{bi}(g) - R_{bs}(g))/10}},
\qquad s_{bg} = \mathrm{clip}\!\left(\frac{R_{b,\max}(g) - T_{\text{cov}}}{T_{\text{weak}} - T_{\text{cov}}},\, 0,\, 1\right)$$

A tile scores 0 where no band covers it, that is, where $\sum_b u_{bg} = 0$.

- **Every band is scored, and the tile takes the contraharmonic mean of the band scores.** Each band is weighted by its own score, so the result lies between the plain mean and the best band, and a band that does not cover the tile carries no weight. There is no band selection, and the objective reads no band order.
- $s$ is band $b$'s strongest cell at tile $g$, and $i$ runs over every other cell **on band $b$ alone** above $T_{\text{cov}}$. The fraction is $s$'s share of the power the band delivers to the tile: 1 when it is alone, 1/2 with an equal rival. Every rival costs in proportion to its linear power, so there is no overlap margin and no step. It is co-band: nothing crosses the band axis.
- $u_{bg}$ is 0 where band $b$ does not cover the tile, which includes every tile the ray tracer found no path to.
- $s_{bg}$ is how far that band's strongest cell sits between the hole and weak thresholds. A server at −90 dBm or better keeps all of its utility, one just out of a hole keeps almost none, and power beyond −90 dBm buys nothing.
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm and $T_{\text{weak}}$ = `kpi.weak_dbm` = −90 dBm. Each physical quantity keeps one threshold, shared with the KPIs.

**The shape of the utility.** At full strength, the power share is the whole of a band's preference over crowding. With one rival $\Delta$ dB below the server:

| $\Delta$ [dB] | alone | 20 | 10 | 6 | 3 | 0 |
|---|---:|---:|---:|---:|---:|---:|
| $u$ | **1.000** | 0.990 | 0.909 | 0.799 | 0.666 | 0.500 |

Two equal rivals leave 0.333. It is exactly 1 only when the server is alone on the band. This report calls a tile **effectively covered** when its score exceeds 0.75, which a single band at full strength reaches only when its rivals together sit more than 4.8 dB below the server. So $J \in [0, 1]$ and **higher is better**. J reads as how much of the grid is effectively covered, discounted for how crowded or marginal the rest is, on every layer that covers it.

**How the mean combines layers.** At full strength on the first band:

| Band scores $u_b$ | Tile score |
|---|---:|
| 1 | 1.000 |
| 1, 1 | 1.000 |
| 1, 0.1 | 0.918 |
| 1, 0.799 (second band with a rival 6 dB down) | 0.911 |
| 1, 0.5 (second band shared by two equal cells) | 0.833 |
| 1, 0.333 (second band at −110 dBm) | 0.833 |
| 0.5 | 0.500 |

A second layer lowers a tile only by scoring below the first. The cost is largest, 0.172, when the second layer scores $\sqrt{2} - 1 \approx 0.414$. It vanishes both as that layer becomes as good as the first and as it fades out.

**The consequence: J is not monotone in the layers present.** Removing band $b$ from a tile raises the tile's score whenever $0 < u_{bg} <$ the tile's score. So darkening a weak layer can raise J, which a maximum over bands would rule out by construction. Section 6.8 measures how much of the grid this touches on this run's TuRBO winner; the pipeline does not compute it.

**The exchange rate this implies.** Splitting a clean, strong, single-band tile evenly between two cells costs 0.5; a rival 6 dB down costs 0.201, and one 20 dB down 0.010. Closing a hole looks as if it should gain the full 1.000, but it cannot. A tile that has just crossed $T_{\text{cov}}$ sits near −120 dBm, where $s \approx 0$. At −119 dBm a newly covered tile is worth 0.033, and at −110 dBm, 0.333. So one strong tile split evenly between two cells costs more than a hole closed at −110 dBm gains.

J is therefore primarily a *signal-strength and cleanliness* measure that treats hole-closing as a minor bonus. Over random search's 145 candidates, signed so that a positive value means J and the KPI improve together, J tracks cell-edge RSRP (Spearman 0.87), the weak rate, median RSRP and the hole rate (0.78 to 0.80) most closely. It tracks the UE service failure rate (0.56), the overlap rate (0.42), median SINR (0.41), median and mean throughput (0.38 and 0.28) and overlap neighbours (0.27) loosely, and cell-edge throughput (0.10) and cell-edge SINR (0.08) hardly at all. The pipeline does not produce these correlations; they were computed for this report from random search's history (`outputs/optim/random/2026-09-30_09-02-36/history.parquet`).

**Why not a preferred band or the best band.** Scoring the most preferred band that clears $T_{\text{cov}}$ would let a tilt raise J by dropping a crowded preferred layer below the threshold. Scoring the best band closes that, but prices nothing on a tile's other layers, so a crowded layer costs nothing wherever another layer is clean. The contraharmonic mean prices every covered layer, and it pays for that with monotonicity.

**What it does not read.** Cell load, where UEs stand, and inter-band interference. That last one is a real gap. The power share is co-band, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The mean does lower a tile for a weak or crowded second layer, but it does so whether or not that layer interferes with the first. The objective has **no free parameters**: it reads two KPI thresholds, both of which the reported KPIs already define.

**Nothing is weighted by demand.** Every tile counts equally. A hole where nobody stands costs exactly what a hole in a hotspot costs. Where the traffic stands is still reported, in Table 7b, but nothing optimises it.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. Within an interval UEs connect one at a time, each to the cell-band above −120 dBm where an equal share of 0.8 of its `max_prb`, split over the UEs already there and itself, carries the most Shannon throughput. Nobody is refused. It drives the estimated-throughput KPIs and every per-cell-band table; J does not read it.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, over all twelve:
   - coverage hole rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓, weak-coverage rate ↓
   - cell-edge and median RSRP ↑ (5th and 50th percentiles of best-server RSRP over covered tiles, read beside the hole rate)
   - cell-edge and median best-server SINR ↑
   - UE service failure rate ↓ (share of all UE rows with no cell-band above −120 dBm)
   - cell-edge, median and mean estimated UE throughput ↑ (5th and 50th percentiles and mean over the served UE rows, at each cell-band's equal PRB share once the interval's last UE has connected)

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the twelve is weighted into J, so agreement between J and the KPIs is a finding, not a construction. PRB load is not reported: every cell-band with a UE uses its whole usable pool by construction ([ADR 0002](../docs/adr/0002-contraharmonic-objective-and-kpi-set.md)).
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
3. **Service.** Every UE row is served from the radio map at its tile, in the search and in the evaluation alike. No measurement noise, report censoring or position error is modelled.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 17 contract checks, all of which held (`tables/02_preprocessing/verification_checks.csv`), and against the template checks (`template_checks.csv`).
- The checks cover band and cell order, scenario ID, grid, UE height and columns, SINR shape, tile and extent bounds, the RSRP bound, schedule, duplicate rows and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 111.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map and `run.json` under `outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands, band carrier frequencies and KPI definition (23 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly. The archived map is a second solve of the winning tilts at the same solver seed, stored as float32, so the check also bounds the GPU ray tracer's run-to-run noise.
3. Builds the scoreboard against the current configuration.
4. Compares each winner with the candidates its own search evaluated.
5. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets, so criterion 3 isolates the model's contribution. The rule sweep was left unmatched because its practical appeal is low cost.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took 2.3 to 2.5 s per candidate (Table 11).

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

All 23 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off. Over all 52 recorded measures, the largest absolute gap is 3.5 × 10⁻⁶ dBm, on random search's median RSRP, and **on J and every rate it is zero**. The differences discussed below therefore come from the configurations, not from bookkeeping.

**J is rounded before any search reads it.** The ray tracer's Monte-Carlo stream is seeded, but the GPU adds path contributions into a tile in a varying order, so the same tilts traced in two processes can differ in J's trailing digits. TuRBO's GP fit turns any such difference into a different proposal. `src/optim/objective.py` rounds J to 10⁻⁶, below the printed precision. Rounding makes a mismatch unlikely, not impossible: a J that lands within the noise of a rounding boundary can still round differently.

**No J here is comparable with an earlier run.** Every run was re-searched under the power-share utility of Section 3.4; the runs recorded under the former $\lambda e^{1-\lambda}$ utility were deleted. The radio-map KPIs of the current configuration are unchanged, since the scene and tilts are; its J fell from 0.6651 to 0.6235 only because the utility changed.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Sources: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **Objective J ↑** | 0.6235 | 0.6648 | 0.6545 | **0.6706** |
| Coverage hole rate ↓ | 0.1125 | **0.1044** | 0.1093 | 0.1086 |
| Weak coverage rate ↓ | 0.3069 | **0.2535** | 0.2707 | 0.2662 |
| Co-band overlap rate ↓ | **0.3164** | 0.3181 *(worse)* | 0.3356 *(worse)* | 0.3215 *(worse)* |
| Overlap neighbours per covered tile ↓ | **0.9625** | 1.1866 *(worse)* | 1.0352 *(worse)* | 1.0014 *(worse)* |
| Median RSRP p50 [dBm] ↑ | −84.11 | **−80.80** | −81.92 | −81.86 |
| Cell-edge RSRP p05 [dBm] ↑ | −108.56 | **−106.97** | −107.66 | −107.52 |
| Median SINR p50 [dB] ↑ | 14.26 | 14.87 | 15.79 | **17.42** |
| Cell-edge SINR p05 [dB] ↑ | −0.88 | −1.50 *(worse)* | −1.02 *(worse)* | **−0.57** |
| UE service failure rate ↓ | 0.2124 | **0.2064** | 0.2088 | 0.2085 |
| Cell-edge throughput p05 [Mbit/s] ↑ | 14.64 | 14.25 *(worse)* | 14.58 *(worse)* | **15.81** |
| Median throughput p50 [Mbit/s] ↑ | 64.06 | 68.05 | 68.17 | **72.88** |
| Mean throughput [Mbit/s] ↑ | 92.08 | 99.28 | 99.74 | **107.49** |
| **KPIs better / worse, of 12** | — | 8 / 4 | 8 / 4 | 10 / 2 |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per KPI and method; J is not among the panels. Source: `figures/04_evaluation/kpi_improvement.png`.*

**TuRBO wins J:** +7.55 %, against +6.61 % for the sweep and +4.96 % for random search. On the twelve KPIs, TuRBO improves ten and worsens two: the co-band overlap rate and overlapping neighbours per covered tile. The sweep and random search improve eight and worsen those two plus cell-edge SINR and cell-edge throughput.

**Head to head, the two strongest methods split the KPIs 6 to 6.**
- **The rule sweep takes six:** hole rate, weak rate, the co-band overlap rate, both RSRP percentiles and the UE service failure rate. Its coverage advantage comes from three shared per-band tilts that widen every footprint at once.
  - The failure-rate win is 0.2064 against 0.2085: 21 UE reports of 10,087.
- **TuRBO takes six:** overlapping neighbours, both SINR percentiles and all three throughput statistics. It is the only method that raises cell-edge SINR and cell-edge throughput, and it raises mean throughput by 16.7 %, against 7.8 % for the sweep.
- **Random search**, third on J, is not the best of the three on any KPI.

**No method lowers co-band overlap.** The band-collapsed overlap rate:
- rises 0.5 % under the sweep;
- rises 1.6 % under TuRBO;
- rises 6.1 % under random search.

Overlapping neighbours per covered tile rises under all three:
- TuRBO by 4.0 %;
- random search by 7.6 %;
- the sweep by 23.3 %.

### 6.3 Did the search matter?

*Table 5. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.6235 | 0.6352 | 0.6352 | 0.6456 | 0.6545 |
| Rule-based sweep | 0.6235 | — | 0.6538 | 0.6643 | 0.6648 |
| TuRBO | 0.6235 | 0.6352 | **0.6671** | **0.6703** | **0.6706** |

*Table 6. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | **0.6544** | 0.6414 | **0.6544** |
| 25 | 0.6544 | **0.6648** | 0.6572 |
| 50 | 0.6544 | **0.6648** | 0.6632 |
| 100 | 0.6545 | 0.6648 | **0.6697** |
| 145 | 0.6545 | — | **0.6706** |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **better** than the current configuration: their median is 0.6352 against 0.6235, and their 90th percentile reaches 0.6456. The strength factor rewards the wider footprints a random tilt tends to produce.

The evidence that the model earned TuRBO's margin:
- TuRBO's **median** candidate (0.6671) scores above both baselines' single **best** (0.6545 and 0.6648).
- TuRBO and random search share the same 16 Sobol points and diverge only once the model proposes. That shared design has a median of 0.6352 for both, while TuRBO's all-candidate median is 0.6671 against random search's 0.6352.
- TuRBO's 128 trust-region proposals average 0.6656, against 0.6335 for its 16 Sobol points (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`).

Per method:
- **Random search** reached 0.6544 at evaluation 7; its only later gain, at evaluation 95, adds less than 10⁻⁴.
- **TuRBO** found its best at **evaluation 131 of 145**, after improvements at 98, 102, 107, 118 and 125, and its trust region never collapsed into a restart. It passed the rule sweep's final answer at evaluation 57. A larger budget would plausibly still improve it.
- **The rule sweep** found its best, 0.6648, at evaluation 21, then spent its remaining 90 evaluations without improving. Its first pass starts at the bottom of each band's range, so it trails at 10 evaluations and leads from 25 to 50.

The paired gain of TuRBO over random search is **+0.0162** on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Every method lowered the hole rate, but not because J asked.** A tile that has just crossed the hole threshold sits near −120 dBm, where the strength factor is near zero, so closing holes buys J almost nothing (Section 3.4). The hole-rate gains in Table 4 are a side effect of uptilting, which widens every footprint.

**Overlap-reducing configurations were available to all but random search.** The rule sweep's candidate set contained one with an overlap rate of **0.3061**, and TuRBO's one at **0.3034**. Both are below the incumbent's 0.3164 (`tables/04_evaluation/sample_efficiency.csv`). Random search never beat the incumbent's overlap rate in 145 candidates. J did not pick the lowest-overlap candidates, which is consistent with J tracking the overlap rate only loosely (Section 3.4). TuRBO's published shortlist (`outputs/solutions_turbo.csv`) holds two runners-up within 0.00013 of J below the winner, at overlap rates of 0.3285 and 0.3213 against the winner's 0.3215. On overlap the winner and its near-ties differ, so which of them is picked is effectively arbitrary.

**Per band, TuRBO lowers overlap on every layer, yet the collapsed rate rises.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`tables/04_evaluation/band_kpis.csv`):

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| 2600 MHz | 0.2010 | 0.1951 | 0.1967 | **0.1905** |
| 1800 MHz | 0.2067 | 0.2077 | 0.2065 | **0.1900** |
| 700 MHz | 0.2396 | 0.2441 | 0.2304 | **0.2158** |
| Band-collapsed (the reported KPI) | **0.3164** | 0.3181 | 0.3356 | 0.3215 |

TuRBO has the lowest overlap on each band, cutting 700 MHz and 1800 MHz most. The collapsed rate still rises, because what crowding remains lands on fewer shared tiles: the three layers' crowded areas coincide less than before. Random search lowers every band, 1800 MHz only marginally, yet raises the collapsed rate most. The sweep improves 2600 MHz only. J prices each band's crowding separately, so it sees TuRBO's per-band gains and not the union the collapsed KPI counts.

*Table 7b. Coverage class by area and by demand, with demand counted in UE reports. Nothing in J reads this view, so it is a check on the result, not a reflection of it. Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 11.2 % / 21.2 % | 10.4 % / 20.6 % | 10.9 % / 20.9 % | 10.9 % / 20.8 % |
| Weak | 30.7 % / 23.7 % | 25.4 % / 19.8 % | 27.1 % / 21.8 % | 26.6 % / 22.0 % |
| Good | 58.1 % / 55.1 % | 64.2 % / 59.6 % | 62.0 % / 57.4 % | 62.5 % / 57.1 % |

*Table 8. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | **0.96** | 64.4 % | 17.6 % |
| Rule-based sweep | 1.19 | **64.5 %** | 18.1 % |
| Random search | 1.04 | 62.3 % | 16.1 % |
| TuRBO | 1.00 | 63.9 % | **15.2 %** |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand moves out of holes and weak coverage under every method.** The share of UE reports on hole tiles falls from 21.2 % to 20.6 % under the sweep and to 20.8 % under TuRBO; the share on good tiles rises from 55.1 % to 59.6 % and 57.1 %. On the demand-weighted view the rule sweep is the strongest of the three, as it is on area. Nothing in J reads this view.

**TuRBO thins the pile-ups but not the mean.** TuRBO cuts the share of covered tiles with three or more overlapping neighbours from 17.6 % to 15.2 %, the best of the three. The share with none falls slightly, from 64.4 % to 63.9 %, and the mean rises from 0.96 to 1.00. The sweep raises both the mean (to 1.19) and the share of pile-ups (to 18.1 %). The power share prices a pile-up far more than a single rival, which is what TuRBO trades on.

**No objective parameters to vary, and no band priority either.** The objective has no parameters and reads no band order. What remains untested is the single search seed, which Section 6.8 lists.

### 6.5 Capacity impact

*Table 9. UE service. Throughput statistics are in Table 4. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---|
| Current | 21.2 % | 0.44 | 12.30 | 53.0 % / 14.2 % / 11.6 % |
| Rule-based sweep | **20.6 %** | 0.48 | 13.23 | 57.4 % / 11.0 % / 11.0 % |
| Random search | 20.9 % | 0.88 | 13.53 | 54.1 % / 12.5 % / 12.5 % |
| TuRBO | 20.8 % | **1.90** | **15.20** | 51.9 % / 15.1 % / 12.1 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band throughput](figures/04_evaluation/cell_band_throughput.png)

*Figure 11. Median estimated throughput per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_throughput.png`.*

**Service.** Every method fails 0.4 to 0.6 points fewer UE reports than the current 21.2 %: the rule sweep 20.6 %, random search and TuRBO 20.9 % and 20.8 %. Every failure stands on a hole tile, since the serving rule refuses nobody. Served SINR rises under all three, most under TuRBO: 1.46 dB at the 10th percentile and 2.90 dB at the median.

**The methods split on 2600 MHz.** The sweep's uniform uptilt pulls UE reports onto 2600 MHz, from 53.0 % to 57.4 %, and off 1800 MHz. TuRBO moves the other way: 2600 MHz falls to 51.9 % and 1800 MHz rises to 15.1 %. That spread is where TuRBO's throughput gain comes from, and it shows in the cell-impact table (`tables/04_evaluation/cell_impact.csv`):
- **Downtilted 2600 MHz carriers shed load:** n3c2's, downtilted 1.61°, drops from 792 served reports to 165, and n1c0's, downtilted 2.60°, from 517 to 148. Their remaining UEs gain 16.7 and 8.2 Mbit/s of median throughput.
- **Uptilted carriers pick it up:** n2c1's 2600 MHz carrier, uptilted 10.56°, gains 366 reports and 5.5 dB of median SINR, and on n3c2's own sector the 1800 MHz carrier, uptilted 8.37°, gains 307 and 7.1 dB.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- **Area covered** rises on 2600 MHz, 71.2 % → 74.3 %, and 1800 MHz, 76.5 % → 77.6 %, and holds on 700 MHz at 86.2 %.
- **Mean RSRP where covered** improves on every layer: 2600 MHz −97.2 → −91.6 dBm, 1800 MHz −91.5 → −88.0 dBm, 700 MHz −84.2 → −82.5 dBm.
- **Median served SINR** rises on 2600 MHz, from 10.0 to 14.0 dB, and on 700 MHz, from 20.9 to 21.7 dB. It falls on 1800 MHz, from 16.0 to 15.0 dB, as that layer takes on more UEs.

About 21 % of UE reports remain unserved, all of them on hole tiles, and 17.5 % of UE positions have no path to any cell (Table 2). Tilt alone cannot serve them.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 8.68 | 11.98 | −7.98 |
| 1800 MHz | 12 / 12 | 7.34 | 9.72 | −5.84 |
| 700 MHz | 12 / 12 | 6.41 | 7.69 | −3.26 |

Every one of the 36 cell-bands moved. The configuration is a net uptilt on every band, and the structure is the point:
- **Most carriers uptilt hard.** 28 of 36 are uptilted, 2600 MHz most. That widening is what lifts the hole and weak rates.
- **Eight of the 36 are downtilted, on four sectors.**

  | Sector | Downtilted carriers |
  |---|---|
  | n0c2 | 1800 MHz +4.87°, 700 MHz +6.85° |
  | n1c0 | 2600 MHz +2.60°, 1800 MHz +1.13°, 700 MHz +5.97° |
  | n2c1 | 1800 MHz +2.99°, 700 MHz +6.04° |
  | n3c2 | 2600 MHz +1.61° |

  On those sectors the search pulls the carriers in while their neighbours widen, which is consistent with TuRBO having the lowest overlap on every band (Table 7a) and with the load shift of Section 6.5.
- **The box binds at both ends.** Proposed tilts span **0.02° to 14.87°**: n1c2's 2600 MHz carrier sits 0.02° from the 0° bound, and n0c2's 1800 MHz carrier 0.13° from the 15° bound. The largest single change is 11.98°, n1c2 on 2600 MHz.

The rule sweep instead sets every cell on every band to one value, 1.67° (`outputs/tilt_change_rule.csv`). That is a uniform uptilt of 10.33° on 2600 MHz, 8.33° on 1800 MHz and 6.33° on 700 MHz. With one seed, it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended.

The largest traffic shifts under TuRBO (`tables/04_evaluation/cell_impact.csv`):
- n3c2 on 2600 MHz, −627 served reports;
- n1c0 on 2600 MHz, −369;
- n2c1 on 2600 MHz, +366;
- n3c2 on 1800 MHz, +307;
- n2c1 on 1800 MHz, −182.

Those are the cells to watch after a rollout.

### 6.7 Cost

*Table 11. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 111 | 21 | 4.17 | 6.24 | 2.3 | +0.0412 | **0.0066** |
| Random search | 145 | 95 | 5.60 | 7.74 | 2.3 | +0.0309 | 0.0040 |
| TuRBO | 145 | 131 | 6.09 | 11.05 | 2.5 | **+0.0471** | 0.0043 |

TuRBO's GP fitting and acquisition, which run on the CPU, added about 5.0 minutes of wall clock on top of its 6.1 minutes of ray tracing. Timings are only indicative.

**By J gain per minute, the rule sweep is about 1.6 times as cost-effective as TuRBO.** It reaches 88 % of TuRBO's gain in 56 % of the wall clock. It does not win on J: TuRBO passes its final J at evaluation 57 of 145 and ends 0.0059 above it. The sweep's remaining case is cost, and the six individual KPIs it wins (Section 6.2).

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed. The TuRBO–rule margin on J is +0.0059, and the TuRBO–random paired gain is +0.0162, each resting on one pair.
3. **Winner's curse, and TuRBO's sensitivity to J.**
   - Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
   - The GPU ray tracer is reproducible only to round-off, and the GP fit amplifies any difference in J into different proposals. Rounding J to 10⁻⁶ lowers the odds of a mismatch rather than ruling it out (Section 6.1).
   - TuRBO's two runners-up sit within 0.00013 of the winner's J with overlap rates from 0.3213 to 0.3285, so any per-KPI claim about TuRBO's winner beyond J is one draw.
4. **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.4).
   - On this run's TuRBO winner, 76.7 % of tiles have such a band. If each could shed it independently, the ceiling on the gain would be 0.032, against the winner's whole improvement of 0.047.
   - The power share makes this common: almost any covered second layer with a co-band rival scores below 1, and so below a clean first layer.
   - That ceiling is not reachable, because darkening a band on one tile changes it on many.
   - Both numbers come from a one-off analysis of the archived radio map, not from the pipeline.
   - Nothing in the objective stops a search from buying J by switching a weak layer off.
5. **Inter-band interference is priced nowhere.** The power share is co-band by definition, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The contraharmonic mean does lower a tile for a weak second layer, but it does so whether or not that layer interferes. Fixing this needs an interference model, not a reweighting.
6. **Nothing is weighted by demand.** A hole where nobody stands costs exactly what a hole in a hotspot costs. Table 7b is the only place demand appears.
7. **The capacity model is a simplification.** It drives the estimated throughput and every per-cell-band figure. It uses a placeholder usable PRB share of 0.8, an equal share with no scheduler, a Shannon rate with no MCS cap, full-load co-band interference, and per-RE thermal noise with no receiver noise figure.
8. **Tilts at both bounds.** The winner places one cell-band at 0.02° and another at 14.87°, within 0.2° of each end of the box.
9. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability.** No J in this report is comparable with a value scored under any earlier objective. `src/evaluation/runs.py` refuses to pool runs recorded under a different KPI set, but there is no version guard on the objective's functional form, so a change to J alone would not be detected.

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0412 (2nd) | +0.0309 (3rd) | **+0.0471 (1st)** |
| 2. Reported KPIs | 8 better, 4 worse; best on hole, weak, overlap rate, both RSRP percentiles and the UE service failure rate | 8 better, 4 worse; best on none | **10 better, 2 worse**; best on overlap neighbours, both SINR percentiles and all three throughput statistics |
| 3. Search effectiveness | Best at evaluation 21 of 111; beaten by TuRBO from evaluation 57 | Best at 7 (to within 10⁻⁴), nothing more in the next 138 | **Median candidate above both baselines' best; still improving at 131 of 145** |
| 4. Robustness | **Most demand out of holes and weak coverage**; pulls load onto 2600 MHz | Largest rise in the overlap rate | Lowest overlap on every band, but the collapsed rate still rises; spreads load off 2600 MHz |
| 5. Cost | **6.2 min, 111 evaluations; about 1.6× the J per minute** | 7.7 min, 145 evaluations | 11.1 min, 145 evaluations |

**Conclusions.**

- **TuRBO reached the highest J, and there is good evidence the model earned it.** Its median candidate (0.6671) scored above both baselines' best. It also shares its first 16 Sobol points with random search and diverges only once the model starts proposing.
- **On the individual KPIs, the rule sweep and TuRBO split the practical result 6 to 6.** The sweep wins coverage: hole, weak, both RSRP measures, the collapsed overlap rate and the UE service failure rate (by 21 UE reports). TuRBO wins signal quality and throughput: both SINR percentiles, all three throughput statistics and overlapping neighbours, and it is the only method that raises cell-edge SINR and cell-edge throughput. The sweep costs about 56 % of TuRBO's wall clock.
- **No method lowers co-band overlap overall.** TuRBO has the lowest overlap on each band and the fewest 3+ pile-ups, but the band-collapsed rate and the mean neighbour count still rise under every method.
- **The objective barely pays for closing holes.** A newly covered tile arrives near −120 dBm, where the strength factor is near zero. The hole-rate gains are a side effect of uptilting.
- **The methods differ in where traffic goes.** The sweep concentrates UE reports on 2600 MHz; TuRBO downtilts two busy 2600 MHz carriers and spreads their load onto uptilted neighbours and 1800 MHz, which is where its throughput gain comes from.
- **Every result** depends on:
  - one scenario and one seed;
  - a simplified capacity model;
  - an objective that is not monotone in the layers present and does not price inter-band interference at all.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`), and re-trace the shortlists under other solver seeds. The TuRBO–rule margin of +0.0059 needs an interval before it can be called decisive.
3. **Decide whether the objective's non-monotonicity is acceptable.** This is the most consequential open question left, and the power share makes it reach most of the grid. Measure it as part of the pipeline rather than ad hoc, and re-trace the shortlisted configurations with the weakest layer at each tile removed, to measure whether any reachable tilt actually collects it.
4. **Give TuRBO a larger budget.** It found its best at evaluation 131 of 145 and never restarted, so the trust region was still productive when the budget ended.
5. **Widen the tilt box.** The winner sits within 0.2° of the 0° bound on one cell-band and of the 15° bound on another.
6. **Model inter-band interference.** The overlap count and the power share are co-band and J does not read SINR, so nothing in the study prices a strong neighbour on another layer. This needs a model, not a reweighting.
7. **Address the out-of-reach demand.** Every UE still unserved stands on a hole tile, largely one hotspot with no propagation path. Tilt alone cannot lower the UE service failure rate much further. That is a coverage decision, not a tilt one.
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
| Objective | co-band power share × strength per band, contraharmonic mean over bands; no parameters; reads `hole_dbm` and `weak_dbm`; rounded to 10⁻⁶ (ADR 0002) |
| Capacity | max-throughput cell selection over an equal share of 0.8 × `max_prb`, candidates above −120 dBm, SCS 15 kHz, connection in report-time order |
| Noise | k·T·SCS per resource element at 298.15 K (`simulation.radio_map.bands[].scs_hz`), no receiver noise figure |
| Search | seed 42 (`optim.seed`); TuRBO's restart and proposal seeds derived from it by `numpy.random.SeedSequence`; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12, improvement 10⁻³, perturbed dimensions 5; rule 10 steps × 4 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-30_09-02-36/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-30_09-10-21/` |
| TuRBO | `outputs/optim/turbo/2026-09-30_09-17-11/` |

To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`. Notebooks 03a and 03b skip any method and seed that already has a run under `optim.output.dir` (`outputs/optim/`), so clear that directory first to re-search.

Each run's `run.json` records the resolved configuration, the scenario ID and a `provenance` block: the Git commit, whether the working tree was dirty, and the numpy, scipy, torch, botorch, gpytorch and sionna-rt versions.

Every path a stage writes is configurable, so a trial run can be kept apart from the real one. The notebooks read extra Hydra overrides from `BAND_TILT_OVERRIDES`, and `reports.figures_dir` and `reports.tables_dir` (`configs/config.yaml`) set where their tables and figures go. The small-budget check of this pipeline used the settings below, with every output under `outputs/smoke/` and each notebook executed to a copy (`jupyter nbconvert --to notebook --execute notebooks/<nb>.ipynb --output-dir outputs/smoke/notebooks`, 00 through 04 in order):

```text
BAND_TILT_OVERRIDES="simulation.output.ue_file=outputs/smoke/data/external/ue_positions.csv
  simulation.output.manifest_file=outputs/smoke/data/external/scenario.json
  simulation.output.radio_map_file=outputs/smoke/data/interim/radio_map.npz
  data.output.ue_file=outputs/smoke/data/processed/ue.parquet
  optim.output.dir=outputs/smoke/optim optim.output.deliverable_dir=outputs/smoke/reports/outputs
  reports.figures_dir=outputs/smoke/reports/figures reports.tables_dir=outputs/smoke/reports/tables
  ++optim.method.budget.n_init=4 ++optim.method.budget.n_iter=8
  ++optim.method.n_steps=3 ++optim.method.n_rounds=1"
```

The variable is one line, space-separated. The scene, ray-tracing fidelity and seeds keep their defaults.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`reports/tables/…`) | Figures (`reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `cell_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `cell_band_throughput`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `template_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `decision_variables`, `optimizer_features`, `objective_decomposition`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map`, `effective_coverage`, `serving_multiplicity` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_<method>`, `tilt_movement_<method>`, `kpi_comparison_<method>`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `kpi_progress`, `tilt_movement_<method>`, `coverage_before_after_<method>`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `band_kpis`, `ue_service_summary`, `cell_band_load`, `cell_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_ue_service_failure_rate`, `tradeoff_overlap_rate_vs_ue_service_failure_rate`, `rsrp_change_maps`, `coverage_before_after`, `coverage_class_maps`, `overlap_neighbour_maps`, `ue_throughput_maps`, `band_kpi_panels`, `serving_band_mix`, `cell_band_throughput`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell; bounds are [0°, 15°]. Negative Δ is an uptilt.*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 1.88 (-10.12) | 0.47 (-9.53) | 1.80 (-6.20) |
| n0c1 | 0.28 (-11.72) | 3.36 (-6.64) | 0.70 (-7.30) |
| n0c2 | 5.02 (-6.98) | 14.87 (+4.87) | 14.85 (+6.85) |
| n1c0 | 14.60 (+2.60) | 11.13 (+1.13) | 13.97 (+5.97) |
| n1c1 | 1.25 (-10.75) | 1.71 (-8.29) | 2.15 (-5.85) |
| n1c2 | 0.02 (-11.98) | 0.28 (-9.72) | 0.88 (-7.12) |
| n2c0 | 2.50 (-9.50) | 1.39 (-8.61) | 1.15 (-6.85) |
| n2c1 | 1.44 (-10.56) | 12.99 (+2.99) | 14.04 (+6.04) |
| n2c2 | 2.20 (-9.80) | 1.06 (-8.94) | 0.53 (-7.47) |
| n3c0 | 3.52 (-8.48) | 0.37 (-9.63) | 5.27 (-2.73) |
| n3c1 | 1.95 (-10.05) | 0.63 (-9.37) | 1.19 (-6.81) |
| n3c2 | 13.61 (+1.61) | 1.63 (-8.37) | 0.31 (-7.69) |

Eight of the 36 cell-bands are downtilted: two on 2600 MHz, three on 1800 MHz and three on 700 MHz, all on sectors n0c2, n1c0, n2c1 and n3c2.

The rule-based sweep's best configuration sets every cell to 1.67° on all three bands (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[5] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
