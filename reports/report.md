# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-09-23: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the committed configuration in `configs/`. Paths are relative to `reports/`.*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0003](../docs/adr/0003-contraharmonic-objective-and-kpi-set.md)). Per tile, each band scores how cleanly one cell dominates it at usable strength. The tile then takes the *contraharmonic mean* of those band scores, so every covered layer counts in proportion to how well it serves. J lies in [0, 1], is 1 only when every covered band on every tile has a single dominant server at or above the weak threshold, and has no free parameters. Ten KPIs were reported beside it, none of them weighted into it.

Three searches started from the same current configuration:
- a rule-based per-band sweep,
- Sobol random search,
- TuRBO-1 Bayesian optimization.

Random search and TuRBO had matched budgets of 145 evaluations; the rule sweep was allowed up to 120.

**Results.**
- **TuRBO** scored highest on J: 0.7054 against 0.6651 currently, a 6.05 % gain. It improved 8 of the 10 reported KPIs, and it has the best co-band overlap rate, overlapping-neighbour count, cell-edge SINR and median SINR of the three methods.
- **The rule sweep** reached J = 0.6963 (+4.68 %) in 111 evaluations and improved 7 of the 10 KPIs. It has the best hole, weak-coverage, RSRP and served-UE figures, and the smallest worsening in load imbalance.
- **Random search** reached J = 0.6911 (+3.90 %), also improving 7 KPIs and worsening 3.

Every method worsens overlapping neighbours per covered tile and cell load imbalance. The rule sweep and random search also worsen the band-collapsed co-band overlap rate; TuRBO lowers it, and lowers overlap on each of the three bands taken separately.

Two findings shape how these should be read.
- **TuRBO's KPI profile is one draw.** Its earlier runs did not reproduce, because GPU ray tracing varies J in the trailing digits and the GP fit turns that into different proposals. J is now rounded to 10⁻⁶ before any search reads it, and two short TuRBO runs then matched on every evaluation (Section 6.1). The earlier runs show how far the profile moves under that tiny a perturbation: at J 0.7049, 0.7059 and now 0.7054, the overlap rate was 0.3070, 0.3183 and 0.3090.
- **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.4). Nothing in the objective stops a search from buying J by switching a weak layer off. ADR 0003 measures how much of the grid this touches on this run's TuRBO winner; the pipeline does not regenerate it.

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

The run is deterministic and spent 110 sweep evaluations plus the incumbent. It cannot give neighbouring cells different tilts. Implementation: `src/optim/methods/rule/search.py`; configuration: `configs/optim/method/rule.yaml`.

### 3.2 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.3 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. It perturbs a random subset of the centre's dimensions and Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. It doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: ADR 0002 and ADR 0003.

### 3.4 The objective every solution maximises

The objective is [ADR 0003](../docs/adr/0003-contraharmonic-objective-and-kpi-set.md), implemented in `src/optim/objective.py` on top of `src/kpi/overlap.py::effective_coverage`:

$$J = \frac{1}{|G|}\sum_{g\in G} \frac{\sum_b u_{bg}^2}{\sum_b u_{bg}},
\qquad u_{bg} = \lambda_{bg}\, e^{1-\lambda_{bg}}\, s_{bg},
\qquad \lambda_{bg} = 1 + m_{bg},
\qquad s_{bg} = \mathrm{clip}\!\left(\frac{R_{b,\max}(g) - T_{\text{cov}}}{T_{\text{weak}} - T_{\text{cov}}},\, 0,\, 1\right)$$

A tile scores 0 where no band covers it, that is, where $\sum_b u_{bg} = 0$.

- **Every band is scored, and the tile takes the contraharmonic mean of the band scores.** Each band is weighted by its own score, so the result lies between the plain mean and the best band, and a band that does not cover the tile carries no weight. There is no band selection. `kpi.capacity.band_preference` belongs to the serving rule, and the objective does not read it.
- $m_{bg}$ counts the other cells **on band $b$ alone** that are above $T_{\text{cov}}$ and within $\Delta_R$ of that band's strongest cell (`src/kpi/overlap.py::overlap_neighbors_per_band`). It is the same count the overlap rate thresholds, read per band instead of summed over all three. It is co-band: nothing crosses the band axis.
- $\lambda_{bg}$ is therefore the number of cells contending to serve tile $g$ on band $b$. It is 0 where that band does not cover the tile, which includes every tile the ray tracer found no path to.
- $s_{bg}$ is how far that band's strongest cell sits between the hole and weak thresholds. A server at −90 dBm or better keeps all of its utility, one just out of a hole keeps almost none, and power beyond −90 dBm buys nothing.
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm, $T_{\text{weak}}$ = `kpi.weak_dbm` = −90 dBm and $\Delta_R$ = `kpi.overlap_margin_db` = 6 dB. Each physical quantity keeps one threshold, shared with the KPIs.

**The shape of the utility.** At full strength, $\lambda e^{1-\lambda}$ is the whole of a band's preference over crowding:

| $\lambda$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| $u = \lambda e^{1-\lambda}$ | 0.000 | **1.000** | 0.736 | 0.406 | 0.199 | 0.092 |

It peaks at exactly 1 when one cell dominates the band. This report calls a tile **effectively covered** when its score exceeds 0.75, which is more than a single band with one co-band neighbour at full strength can reach (0.736). So $J \in [0, 1]$ and **higher is better**. J reads as how much of the grid is effectively covered, discounted for how crowded or marginal the rest is, on every layer that covers it.

**How the mean combines layers.** At full strength on the first band:

| Band scores $u_b$ | Tile score |
|---|---:|
| 1 | 1.000 |
| 1, 1 | 1.000 |
| 1, 0.1 | 0.918 |
| 1, 0.736 (second band crowded by one neighbour) | 0.888 |
| 1, 0.333 (second band at −110 dBm) | 0.833 |
| 0.736 | 0.736 |

A second layer lowers a tile only by scoring below the first. The cost is largest, 0.172, when the second layer scores $\sqrt{2} - 1 \approx 0.414$. It vanishes both as that layer becomes as good as the first and as it fades out.

**The consequence: J is not monotone in the layers present.** Removing band $b$ from a tile raises the tile's score whenever $0 < u_{bg} <$ the tile's score. So darkening a weak layer can raise J, which a maximum over bands would rule out by construction. ADR 0003 measures how much of the grid this touches on this run's TuRBO winner. The pipeline does not recompute it (Section 6.8).

**The exchange rate this implies.** Splitting a clean, strong, single-band tile between two cells moves it from λ = 1 to λ = 2 and costs $1 - 2e^{-1} = 0.264$. Closing a hole looks as if it should gain the full 1.000, but it cannot. A tile that has just crossed $T_{\text{cov}}$ sits near −120 dBm, where $s \approx 0$. At −119 dBm a newly covered tile is worth 0.033, and at −110 dBm, 0.333. So one newly crowded strong tile costs as much as several closed holes.

J is therefore primarily a *signal-strength and cleanliness* measure that treats hole-closing as a minor bonus. ADR 0003's "Measured outcome" records its rank correlation with each KPI over random search's candidates on the previous run. There, J tracked the strength and SINR percentiles most closely and the two overlap measures least. Those correlations are not produced by the pipeline and were not recomputed for this report.

**Why not a preferred band or the best band.** Scoring the most preferred band that clears $T_{\text{cov}}$ would let a tilt raise J by dropping a crowded preferred layer below the threshold. Scoring the best band closes that, but prices nothing on a tile's other layers, so a crowded layer costs nothing wherever another layer is clean. The contraharmonic mean prices every covered layer, and it pays for that with monotonicity.

**What it does not read.** Cell load, where UEs stand, and inter-band interference. That last one is a real gap. $m$ is co-band, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The mean does lower a tile for a weak or crowded second layer, but it does so whether or not that layer interferes with the first. The objective has **no free parameters**: it reads three KPI thresholds, all of which the reported KPIs already define.

**Nothing is weighted by demand.** Every tile counts equally. A hole where nobody stands costs exactly what a hole in a hotspot costs. Where the traffic stands is still reported, in Table 7b, but nothing optimises it.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a cell-band serves. It drives the served ratio and every capacity table, and it is the only thing that reads the band preference.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, over all ten:
   - coverage hole rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓, weak-coverage rate ↓
   - cell-edge and median RSRP ↑ (5th and 50th percentiles of best-server RSRP over covered tiles [4], read beside the hole rate)
   - cell-edge and median best-server SINR ↑
   - served UE ratio ↑
   - cell load imbalance ↓

   A change is labelled only as better or worse (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the ten is weighted into J, so agreement between J and the KPIs is a finding, not a construction. Peak PRB utilisation is not among them: it is bounded by the 0.8 admission ceiling by construction ([ADR 0003](../docs/adr/0003-contraharmonic-objective-and-kpi-set.md)), and it is still computed, as the check that the serving rule held.
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
- Random search and TuRBO each spent 145 evaluations: the incumbent, 16 initial points and 128 more. The rule sweep spent 111.
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
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took 2.3 to 3.6 s per candidate (Table 11).

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

All 24 comparability checks held (`tables/04_evaluation/comparability_checks.csv`). These include the check that every run was scored by `kpi.objective_version` 3, the objective this code implements. The KPIs recomputed from the archived radio maps (`tables/04_evaluation/kpi_reproducibility.csv`) match the recorded values to float round-off. Over all 44 recorded measures, the largest absolute gap is 3.9 × 10⁻⁶ dB, on the rule sweep's median SINR, and **on J it is zero**. The differences discussed below therefore come from the configurations, not from bookkeeping.

**Why TuRBO did not reproduce before, and what changed.** The ray tracer's Monte-Carlo stream is seeded, but the GPU adds path contributions into a tile in a varying order. Traced in two processes, the same tilts gave identical threshold KPIs, but J differed by up to about 4 × 10⁻¹² on every one of 17 configurations. Random search and the rule sweep never feed J back into where they look, so that was invisible to them. TuRBO's GP fit turned it into a different first proposal, and from there a different run. `src/optim/objective.py` now rounds J to 10⁻⁶, below the printed precision, so the trailing digits never reach a search.
- **The fix, tested directly.** Two TuRBO runs of 47 evaluations each (the incumbent, 16 Sobol points and 30 proposals) matched on every tilt and every J. The full 145-evaluation run was made once, so its repeat is not measured directly. Rounding makes a mismatch unlikely, not impossible: a J that lands within the noise of a rounding boundary can still round differently.
- **Deterministic stages reproduced.** This rerun rebuilt every artifact with the committed configuration and seed 42, and compared it with the earlier run of the same day. The incumbent's threshold KPIs were identical. Random search and the rule sweep evaluated the same tilts at every step and found the same winners at the same evaluations. J moved only by the rounding, at most 5 × 10⁻⁷.
- **TuRBO diverged from the earlier run at its first proposal**, as expected, since that run's GP saw unrounded scores. It ended at J 0.7054 against 0.7059, at a different configuration.
- **Its KPI profile moved, some of it by more than the method gaps.** Against the earlier run, the co-band overlap rate went from 0.3183 to 0.3090, load imbalance from 1.0401 to 1.0075, and the served rate from 0.6010 to 0.5976. The run of 2026-09-22 had 0.3070, 0.9746 and 0.5898. Section 6.8 (limitation 3) takes this up.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Sources: [`tables/04_evaluation/kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](tables/04_evaluation/method_cost.csv).*

| | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| **Objective J ↑** | 0.6651 | 0.6963 | 0.6911 | **0.7054** |
| Coverage hole rate ↓ | 0.1125 | **0.1044** | 0.1083 | 0.1077 |
| Co-band overlap rate ↓ | 0.3164 | 0.3181 *(worse)* | 0.3254 *(worse)* | **0.3090** |
| Overlap neighbours per covered tile ↓ | **0.9625** | 1.1866 *(worse)* | 1.0303 *(worse)* | 0.9988 *(worse)* |
| Weak coverage rate ↓ | 0.3069 | **0.2535** | 0.2864 | 0.2639 |
| Cell-edge RSRP p05 [dBm] ↑ | −108.56 | **−106.97** | −107.98 | −107.34 |
| Median RSRP p50 [dBm] ↑ | −84.11 | **−80.80** | −82.72 | −81.62 |
| Cell-edge SINR p05 [dB] ↑ | −5.87 | −4.97 | −5.54 | **−4.90** |
| Median SINR p50 [dB] ↑ | 8.49 | 9.90 | 9.21 | **10.57** |
| Served UE rate ↑ | 0.5265 | **0.5988** | 0.5882 | 0.5976 |
| Cell load imbalance ↓ | **0.9098** | 0.9963 *(worse)* | 1.0092 *(worse)* | 1.0075 *(worse)* |
| **KPIs better / worse, of 10** | — | 7 / 3 | 7 / 3 | 8 / 2 |

![KPI improvement](figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per KPI and method. Source: `figures/04_evaluation/kpi_improvement.png`.*

**TuRBO wins J by a clear margin:** +6.05 %, against +4.68 % for the sweep and +3.90 % for random search. On the ten KPIs, TuRBO improves eight and worsens two: overlapping neighbours per covered tile and load imbalance. The sweep and random search improve seven and worsen those two plus the co-band overlap rate.

**Head to head, the two strongest methods split the KPIs 6 to 4.**
- **The rule sweep takes six:** hole rate, weak rate, both RSRP percentiles, the served rate and load imbalance. Its coverage advantage comes from three shared per-band tilts that widen every footprint at once.
  - The served-rate win is 0.5988 against 0.5976, well inside the run-to-run shift TuRBO showed in Section 6.1.
  - The load-imbalance win is between two configurations that are both worse than the current one.
- **TuRBO takes four:** the overlap rate, overlapping neighbours and both SINR percentiles.
- Random search takes none.

**Only TuRBO lowers co-band overlap.** The band-collapsed overlap rate:
- falls 2.3 % under TuRBO;
- rises 0.5 % under the sweep;
- rises 2.8 % under random search.

Overlapping neighbours per covered tile rises under all three:
- TuRBO by 3.8 %;
- random search by 7.0 %;
- the sweep by 23.3 %.

TuRBO's earlier runs moved the overlap rate by −3.0 % and +0.6 %. That is the clearest single sign that its KPI profile is one draw (Section 6.1).

**Cell load imbalance worsens under all three methods:**
- the sweep by 9.5 %;
- TuRBO by 10.7 %;
- random search by 10.9 %.

Nothing in J asks for even load, so whether a search lands on an even-load configuration is incidental.

### 6.3 Did the search matter?

*Table 5. Winner against the candidates each run evaluated. Source: [`tables/04_evaluation/winner_vs_candidates.csv`](tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.6651 | 0.6701 | 0.6711 | 0.6819 | 0.6911 |
| Rule-based sweep | 0.6651 | — | 0.6917 | 0.6960 | 0.6963 |
| TuRBO | 0.6651 | 0.6701 | **0.7019** | **0.7046** | **0.7054** |

*Table 6. Best J reached after a fixed number of evaluations. Source: [`tables/04_evaluation/sample_efficiency.csv`](tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | Rule-based sweep | TuRBO |
|---:|---:|---:|---:|
| 10 | **0.6911** | 0.6824 | **0.6911** |
| 25 | 0.6911 | **0.6963** | 0.6939 |
| 50 | 0.6911 | 0.6963 | **0.7019** |
| 100 | 0.6911 | 0.6963 | **0.7043** |
| 145 | 0.6911 | — | **0.7054** |

![Search progress](figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **better** than the current configuration: their median is 0.6711 against 0.6651, and their 90th percentile reaches 0.6819. The strength factor rewards the wider footprints a random tilt tends to produce.

The evidence that the model earned TuRBO's margin:
- TuRBO's **median** candidate (0.7019) scores above both baselines' single **best** (0.6911 and 0.6963).
- TuRBO and random search share the same 16 Sobol points and diverge only once the model proposes. That shared design has a median of 0.6701 for both, while TuRBO's all-candidate median is 0.7019 against random search's 0.6711.
- TuRBO's 128 trust-region proposals average 0.7005, against 0.6693 for its 16 Sobol points (`tables/03b_turbo/turbo_evaluations_by_proposer.csv`).

Per method:
- **Random search** found its best at evaluation 7 and did not improve in the remaining 138.
- **TuRBO** found its best at **evaluation 143 of 145**, and its trust region never collapsed into a restart. It passed the rule sweep's final answer at evaluation 32. A larger budget would plausibly still improve it, and this is the clearest single lever left.
- **The rule sweep** found its best, 0.6963, at evaluation 21, then spent its remaining 90 evaluations without improving. Its first pass starts at the bottom of each band's range, so it trails at 10 evaluations and leads at 25.

The paired gain of TuRBO over random search is **+0.0142** on the one seed (`tables/04_evaluation/paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Every method lowered the hole rate, but not because J asked.** A tile that has just crossed the hole threshold sits near −120 dBm, where the strength factor is near zero, so closing holes buys J almost nothing (Section 3.4). The hole-rate gains in Table 4 are a side effect of uptilting, which widens every footprint.

**Overlap-reducing configurations were available to all but random search.** The rule sweep's candidate set contained one with an overlap rate of **0.3061**, and TuRBO's one at **0.3021**. Both are below the incumbent's 0.3164 (`tables/04_evaluation/sample_efficiency.csv`). Random search never beat the incumbent's overlap rate in 145 candidates. J did not pick the lowest-overlap candidates, which is consistent with overlap being the measure J tracks least closely (Section 3.4). TuRBO's winner lowers the rate to 0.3090. Its published shortlist (`outputs/solutions_turbo.csv`) holds two runners-up within 0.0004 of J below it, at 0.3084 and 0.3059. On overlap the winner and its near-ties differ, so which of them is picked is effectively arbitrary.

**Per band, TuRBO improves overlap on every layer.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`tables/04_evaluation/band_kpis.csv`):

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Rule-based sweep | Random search | TuRBO |
|---|---:|---:|---:|---:|
| 2600 MHz | 0.2010 | 0.1951 | **0.1908** | 0.1913 |
| 1800 MHz | 0.2067 | 0.2077 | 0.1952 | **0.1940** |
| 700 MHz | 0.2396 | 0.2441 | 0.2422 | **0.2237** |
| Band-collapsed (the reported KPI) | 0.3164 | 0.3181 | 0.3254 | **0.3090** |

TuRBO lowers overlap on all three bands, with the largest cut on 700 MHz, and here the collapsed rate falls too. Random search improves 2600 and 1800 MHz, and makes the largest cut on 2600 MHz, but worsens 700 MHz; its collapsed rate rises. The sweep improves 2600 MHz only.

*Table 7b. Coverage class by area and by demand, with demand weighted by the current configuration's peak PRB demand. Nothing in J reads this view, so it is a check on the result, not a reflection of it. Source: [`tables/04_evaluation/coverage_by_area_and_demand.csv`](tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Rule sweep area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|---|
| Hole | 11.2 % / 0.0 % | 10.4 % / 0.0 % | 10.8 % / 0.3 % | 10.8 % / 0.4 % |
| Weak | 30.7 % / 83.6 % | 25.4 % / 76.7 % | 28.6 % / 79.7 % | 26.4 % / 78.5 % |
| Good | 58.1 % / 16.4 % | 64.2 % / 23.3 % | 60.5 % / 20.1 % | 62.8 % / 21.0 % |

*Table 8. Overlapping co-band neighbours per configuration. Source: [`tables/04_evaluation/overlap_neighbour_summary.csv`](tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | **0.96** | 64.4 % | 17.6 % |
| Rule-based sweep | 1.19 | 64.5 % | 18.1 % |
| Random search | 1.03 | 63.5 % | 16.8 % |
| TuRBO | 1.00 | **65.4 %** | **16.5 %** |

![Coverage before and after](figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `figures/04_evaluation/rsrp_change_maps.png`.*

**Demand in holes.**
- **Onto hole tiles:** the rule sweep moves **no** peak demand there, random search moves 0.3 % and TuRBO 0.4 %.
- **Off weak tiles:** the share of demand on weak tiles falls under every method, most under the rule sweep (83.6 % to 76.7 %, against TuRBO's 78.5 %).
- **Onto good tiles:** the share on good tiles rises from 16.4 % to 23.3 % under the sweep and to 21.0 % under TuRBO.

On the demand-weighted view the rule sweep is the strongest of the three. Nothing in J reads this view, so this is the check catching something the objective cannot see: a small amount of served traffic pushed onto ground the network no longer covers.

**Crowding thins at both ends under TuRBO, but the mean still rises.** TuRBO cuts the share of covered tiles with three or more overlapping neighbours from 17.6 % to 16.5 %, and raises the share with none from 64.4 % to 65.4 %, the best of the three methods on both. The mean still rises slightly, from 0.96 to 1.00. The sweep raises both the mean (to 1.19) and the share of pile-ups (to 18.1 %).

**No objective parameters to vary, and no band priority either.** The objective has no parameters, and it does not read `kpi.capacity.band_preference`; that order belongs to the serving rule alone. What remains untested is the single search seed, which Section 6.8 lists.

### 6.5 Capacity impact

*Table 9. UE service. Source: [`tables/04_evaluation/ue_service_summary.csv`](tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | PRBs per served UE, median | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---:|---|
| Current | 47.3 % | −0.38 | 5.13 | 53.2 | 34.1 % / 10.5 % / 8.1 % |
| Rule-based sweep | **40.1 %** | +0.36 | **7.86** | **39.3** | 46.3 % / 6.4 % / 7.2 % |
| Random search | 41.2 % | +0.24 | 7.73 | 39.8 | 45.1 % / 7.1 % / 6.6 % |
| TuRBO | 40.2 % | **+0.46** | 7.77 | 39.6 | 45.9 % / 6.7 % / 7.2 % |

![Serving band mix](figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `figures/04_evaluation/serving_band_mix.png`.*

![Cell-band utilisation](figures/04_evaluation/cell_band_utilisation.png)

*Figure 11. Peak PRB utilisation per cell-band, current and recommended. Source: `figures/04_evaluation/cell_band_utilisation.png`.*

**Service.** Every method serves substantially more UEs, by 6.2 to 7.2 points. The rule sweep serves the most (59.9 % against 52.7 %), then TuRBO (59.8 %) and random search (58.8 %). Served SINR rises under all three at both the median and the 10th percentile, and the PRBs a served UE needs fall by 25 to 26 %.

**All three methods concentrate traffic on 2600 MHz.** Its share of UE reports rises from 34.1 % to between 45.1 % and 46.3 %, while 1800 MHz falls from 10.5 % to between 6.4 % and 7.1 %. The mechanism under TuRBO is visible in the cell-impact table (`tables/04_evaluation/cell_impact.csv`). All twelve 2600 MHz carriers are uptilted, which widens footprints that the serving rule prefers:
- **Gains on 2600 MHz:** n1c1's 2600 MHz carrier, uptilted 8.95°, gains 262 served reports and 7.9 dB of median SINR. n2c1's gains 214 and n3c1's 113.
- **Losses on 1800 MHz:** n2c1's 1800 MHz carrier is downtilted 4.82° and loses all 165 of its reports. n1c1's, uptilted 9.79°, loses 76.

**Per band** (`tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- **Area covered** rises on every layer: 2600 MHz 71.2 % → 74.7 %, 1800 MHz 76.5 % → 77.7 %, 700 MHz 86.2 % → 86.4 %.
- **Mean RSRP where covered** improves on every layer: 2600 MHz −97.2 → −90.8 dBm, 1800 MHz −91.5 → −88.0 dBm, 700 MHz −84.2 → −82.3 dBm. No band loses mean signal.
- **Median served SINR** rises on 2600 MHz, from 3.5 to 7.1 dB. It falls on 1800 MHz, from 5.2 to 4.7 dB, and on 700 MHz, from 13.3 to 12.8 dB.

About 40 % of UE reports remain unserved, and much of that is out of reach: 17.5 % of UE positions have no path to any cell (Table 2) and 21.2 % stand on hole tiles. The rest is PRB exhaustion. The busiest cell-bands sit at or just below the 0.8 admission ceiling in every configuration (`tables/04_evaluation/cell_impact.csv`), so the 2600 MHz layer remains the binding constraint on service.

### 6.6 Recommended tilt changes

![Tilt change heatmap](figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tables/04_evaluation/tilt_movement_summary.csv`](tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 9.28 | 11.81 | −9.28 |
| 1800 MHz | 12 / 12 | 6.27 | 9.79 | −5.12 |
| 700 MHz | 12 / 12 | 4.63 | 7.66 | −3.27 |

Every one of the 36 cell-bands moved. The configuration is a net uptilt on every band, and the structure is the point:
- **2600 MHz moves most, all of it upward.** All twelve carriers are uptilted, averaging 9.28°. That widening is what lifts the hole and weak rates, and it is what pulls traffic onto 2600 MHz (Section 6.5).
- **Five of the 36 are downtilted, and they cluster on three sectors.**

  | Sector | Downtilted carriers |
  |---|---|
  | n0c2 | 700 MHz +1.90° |
  | n1c0 | 1800 MHz +2.06°, 700 MHz +5.72° |
  | n2c1 | 1800 MHz +4.82°, 700 MHz +0.56° |

  On those sectors the search pulls the lower layers in while the others widen, which is consistent with TuRBO's largest per-band overlap cut being on 700 MHz (Table 7a).
- **The box nearly binds at both ends.** Proposed tilts span **0.19° to 14.82°**: n0c2's 2600 MHz carrier sits 0.19° from the 0° bound and n2c1's 1800 MHz carrier 0.18° from the 15° bound. The largest single change is 11.81°, n0c2 on 2600 MHz.

The rule sweep instead sets every cell on every band to one value, 1.67° (`outputs/tilt_change_rule.csv`). That is a uniform uptilt of 10.33° on 2600 MHz, 8.33° on 1800 MHz and 6.33° on 700 MHz. With one seed, it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended. TuRBO's runs so far make the point concretely: three winners within 0.001 of each other on J had four, six and five downtilts.

The largest traffic shifts under TuRBO (`tables/04_evaluation/cell_impact.csv`):
- n1c1 on 2600 MHz, +262 served reports;
- n2c1 on 2600 MHz, +214;
- n2c1 on 1800 MHz, −165;
- n3c1 on 2600 MHz, +113;
- n0c2 on 2600 MHz, +108.

Those are the cells to watch after a rollout.

### 6.7 Cost

*Table 11. Search cost. Sources: [`tables/04_evaluation/method_cost.csv`](tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based sweep | 111 | 21 | 4.28 | 6.47 | 2.3 | +0.0312 | **0.0048** |
| Random search | 145 | 7 | 7.30 | 9.90 | 3.0 | +0.0260 | 0.0026 |
| TuRBO | 145 | 143 | 8.76 | 16.61 | 3.6 | **+0.0402** | 0.0024 |

TuRBO's GP fitting and acquisition added about 7.9 minutes of wall clock on top of its 8.8 minutes of ray tracing, nearly doubling it. Timings are only indicative: random search's ray tracing took 7.3 minutes here against 8.3 minutes in the earlier run of the identical configuration.

**By J gain per minute, the rule sweep is about twice as cost-effective as TuRBO.** It reaches 78 % of TuRBO's gain in 39 % of the wall clock. It does not compete on quality: TuRBO passes its final J at evaluation 32 of 145 and ends 0.0091 above it. The sweep's remaining case is cost, and the six individual KPIs it wins (Section 6.2).

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed. The TuRBO–rule margin on J is +0.0091, and the TuRBO–random paired gain is +0.0142, each resting on one pair.
3. **Winner's curse, and TuRBO's sensitivity to J.**
   - Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
   - The GPU ray tracer is reproducible only to round-off, and the GP fit amplifies any difference in J into different proposals. Rounding J to 10⁻⁶ keeps that round-off out of the search. Two short TuRBO runs then matched exactly, but rounding lowers the odds of a mismatch rather than ruling it out (Section 6.1).
   - TuRBO's runs before the rounding measured the consequence: the same seed gave winners with J within 0.001 but different KPI profiles. The overlap rate went from improved to worsened and back.
   - A replay of the earlier run's own recorded scores, on this machine, did not reproduce its first proposal, so a second, unidentified source of difference exists between that run and this environment.
   - Any per-KPI claim about TuRBO's winner beyond J is one draw.
4. **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.4).
   - ADR 0003 measures the reach of this on this run's TuRBO winner. 49.3 % of tiles have such a band. If each could shed it independently, the ceiling on the gain would be 0.054, more than TuRBO's whole improvement of 0.040.
   - That ceiling is not reachable, because darkening a band on one tile changes it on many.
   - Both numbers come from a one-off analysis of the archived radio map, not from the pipeline.
   - Nothing in the objective stops a search from buying J by switching a weak layer off.
5. **Inter-band interference is priced nowhere.** The overlap count is co-band by definition, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The contraharmonic mean does lower a tile for a weak second layer, but it does so whether or not that layer interferes. Fixing this needs an interference model, not a reweighting.
6. **Nothing is weighted by demand.** A hole where nobody stands costs exactly what a hole in a hotspot costs. Table 7b is the only place demand appears, and it shows 0.4 % of peak demand landing on hole tiles under the winner, where none did before.
7. **The capacity model is a simplification.** It drives the served ratio and every capacity figure. It uses a Shannon rate with no MCS cap, full-load co-band interference against partial PRB load, and no receiver noise figure.
8. **Tilts near both bounds.** The winner places one cell-band at 0.19° and another at 14.82°, within 0.2° of each end of the box.
9. **No MARL arm and no held-out validation.** The planned comparison against reinforcement learning could not be made.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability.** No J in this report is comparable with a value scored under any earlier objective. `kpi.objective_version` is 3, and `src/evaluation/runs.py` refuses to pool runs across versions, checking against the running config as well as run to run.

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Rule-based sweep | Random search | TuRBO |
|---|---|---|---|
| 1. Objective J | +0.0312 (2nd) | +0.0260 (3rd) | **+0.0402 (1st)** |
| 2. Reported KPIs | 7 better, 3 worse; **best on 6 of 10**: hole, weak, both RSRP percentiles, served rate, and the smallest load worsening | 7 better, 3 worse | **8 better, 2 worse**; best on overlap rate, overlap neighbours and both SINR percentiles |
| 3. Search effectiveness | Best at evaluation 21 of 111; beaten by TuRBO from evaluation 32 | Best at 7, no gain in the next 138 | **Median candidate above both baselines' best; still improving at 143 of 145** |
| 4. Robustness | **No demand onto holes**; largest demand shift out of weak coverage | 0.3 % of demand onto holes | 0.4 % of demand onto holes; overlap lowered on every band and overall; KPI profile moved between runs before J was rounded |
| 5. Cost | **6.5 min, 111 evaluations; about 2× the J per minute** | 9.9 min, 145 evaluations | 16.6 min, 145 evaluations |

**Conclusions.**

- **TuRBO reached the highest J, and there is good evidence the model earned it.** Its median candidate (0.7019) scored above both baselines' best. It also shares its first 16 Sobol points with random search and diverges only once the model starts proposing.
- **On the individual KPIs, the rule sweep and TuRBO split the practical result.** The sweep wins six of the ten head to head: hole, weak and both RSRP measures, the served rate, and the narrowest load-imbalance loss. TuRBO wins the overlap rate, overlapping neighbours and both SINR percentiles. The sweep costs about 39 % of TuRBO's wall clock.
- **Only TuRBO lowers co-band overlap overall, and not robustly.** It lowers the band-collapsed rate and the rate on every band, and has the fewest 3+ pile-ups. The sweep and random search raise the collapsed rate, and every method raises overlapping neighbours per covered tile and load imbalance. TuRBO's runs so far have moved the collapsed rate both ways, so which way that KPI goes under TuRBO is not settled by one seed.
- **The objective barely pays for closing holes.** A newly covered tile arrives near −120 dBm, where the strength factor is near zero. The hole-rate gains are a side effect of uptilting.
- **Every method concentrates traffic on 2600 MHz.** All twelve of TuRBO's 2600 MHz carriers are uptilted. The layer's share of served reports rises from 34.1 % to 45.9 %, and 1800 MHz loses share under every method.
- **Every result** depends on:
  - one scenario and one seed;
  - a simplified capacity model;
  - an objective that is not monotone in the layers present and does not price inter-band interference at all.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b, or `task sweep`), and re-trace the shortlists under other solver seeds.
   - The TuRBO–rule margin of +0.0091 needs an interval before it can be called decisive.
   - TuRBO's runs before J was rounded show its per-KPI profile moving under perturbations of J around 10⁻¹².
3. **Decide whether the objective's non-monotonicity is acceptable.** This is the most consequential open question left. Measure it as part of the pipeline rather than ad hoc, and re-trace the shortlisted configurations with the weakest layer at each tile removed, to measure whether any reachable tilt actually collects it.
4. **Give TuRBO a larger budget.** It found its best at evaluation 143 of 145 and never restarted, so the trust region was still productive when the budget ended.
5. **Widen the tilt box.** The winner sits within 0.2° of the 0° bound on one cell-band and of the 15° bound on another.
6. **Model inter-band interference.** The overlap count is co-band and J does not read SINR, so nothing in the study prices a strong neighbour on another layer. This needs a model, not a reweighting.
7. **Address the 2600 MHz PRB ceiling.** Every method concentrates traffic on 2600 MHz, and its busiest cell-bands sit at the 0.8 admission ceiling, so tilt alone cannot raise the served ratio much further. That is a capacity decision, not a tilt one.
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
| Objective | contraharmonic mean over bands; no parameters; reads `hole_dbm`, `weak_dbm` and `overlap_margin_db`; rounded to 10⁻⁶; `objective_version` 3 (ADR 0003) |
| Capacity | preference 2600 > 1800 > 700 MHz, serving threshold −120 dBm, admission ceiling 0.8, 20 Mbps per UE, SCS 15 kHz, admission in report-time order |
| Search | seed 42; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12; rule 10 steps × 4 rounds; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `outputs/optim/random/2026-09-23_09-13-50/` |
| Rule-based sweep | `outputs/optim/rule/2026-09-23_09-23-44/` |
| TuRBO | `outputs/optim/turbo/2026-09-23_09-31-10/` |

To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`. Notebooks 03a and 03b skip any method and seed that already has a run under `outputs/optim/`, so clear that directory first to re-search.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`reports/tables/…`) | Figures (`reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `mdt_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `mdt_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `mdt_overview`, `mdt_share_by_component`, `mdt_share_by_coverage`, `mdt_rsrp_statistics`, `cell_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `mdt_over_time`, `mdt_vs_ue_positions`, `mdt_rsrp_vs_ue_rsrp`, `cell_band_utilisation`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `template_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `decision_variables`, `optimizer_features`, `objective_decomposition`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map`, `effective_coverage`, `serving_multiplicity` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_<method>`, `tilt_movement_<method>`, `kpi_comparison_<method>`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis`, `prb_usage_by_time` | `search_progress`, `kpi_progress`, `tilt_movement_<method>`, `coverage_before_after_<method>`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis`, `prb_usage` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis`, `prb_usage_by_time` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis`, `prb_usage` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `band_kpis`, `ue_service_summary`, `cell_band_load`, `prb_usage_by_time`, `cell_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_served_rate`, `tradeoff_overlap_rate_vs_served_rate`, `rsrp_change_maps`, `coverage_before_after`, `coverage_class_maps`, `overlap_neighbour_maps`, `band_kpi_panels`, `serving_band_mix`, `cell_band_utilisation`, `prb_usage_heatmaps`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`tables/04_evaluation/recommended_tilt.csv`](tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`outputs/tilt_change_turbo.csv`](outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell; bounds are [0°, 15°]. Negative Δ is an uptilt.*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 1.09 (-10.91) | 2.55 (-7.45) | 2.12 (-5.88) |
| n0c1 | 1.75 (-10.25) | 4.09 (-5.91) | 4.25 (-3.75) |
| n0c2 | 0.19 (-11.81) | 6.01 (-3.99) | 9.90 (+1.90) |
| n1c0 | 5.21 (-6.79) | 12.06 (+2.06) | 13.72 (+5.72) |
| n1c1 | 3.05 (-8.95) | 0.21 (-9.79) | 2.75 (-5.25) |
| n1c2 | 0.87 (-11.13) | 1.28 (-8.72) | 1.88 (-6.12) |
| n2c0 | 4.36 (-7.64) | 3.79 (-6.21) | 0.34 (-7.66) |
| n2c1 | 5.63 (-6.37) | 14.82 (+4.82) | 8.56 (+0.56) |
| n2c2 | 0.57 (-11.43) | 4.43 (-5.57) | 3.18 (-4.82) |
| n3c0 | 3.00 (-9.00) | 7.07 (-2.93) | 7.42 (-0.58) |
| n3c1 | 4.86 (-7.14) | 1.16 (-8.84) | 1.37 (-6.63) |
| n3c2 | 2.10 (-9.90) | 1.08 (-8.92) | 1.29 (-6.71) |

Five of the 36 cell-bands are downtilted: two on 1800 MHz and three on 700 MHz, all on sectors n0c2, n1c0 and n2c1.

The rule-based sweep's best configuration sets every cell to 1.67° on all three bands (`outputs/tilt_change_rule.csv`).

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TR 36.814, *Further advancements for E-UTRA physical layer aspects*, Annex A.2.1.4.

[5] 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1.

[6] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[7] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
