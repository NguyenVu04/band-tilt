# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the pipeline run of 2026-10-02: notebooks `00_simulation` through `04_evaluation`, the optimization runs listed in Appendix A, and the configuration in `configs/`. Paths are relative to this file (`docs/`): generated tables and figures are under `../reports/`, run directories under `../outputs/`. RSRP, interference and noise are all per resource element: thermal noise is k·T over one 15 kHz subcarrier, not over the channel bandwidth. Twelve KPIs are reported ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)).*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band cell layout when every band on every cell is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve cells), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 15°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)). Per tile, each band scores its strongest cell's share of the band's received power, scaled by how usable that cell's signal is. The tile then takes the *contraharmonic mean* of those band scores, so every covered layer counts in proportion to how well it serves. J lies in [0, 1], is 1 only when every covered band on every tile has one server at or above the weak threshold and no co-band rival above the hole threshold, and has no free parameters. Twelve KPIs were reported beside it, none of them weighted into it.

Two searches started from the same current configuration with matched budgets of 145 evaluations and the same seed:
- Sobol random search,
- TuRBO-1 Bayesian optimization.

**Results.**
- **TuRBO** scored highest on J: 0.6721 against 0.6235 currently, a 7.78 % gain. It improved 10 of the 12 reported KPIs and is better than random search on all twelve: coverage, signal strength, both SINR percentiles and all three estimated-throughput statistics. It is the only method that raises cell-edge SINR and cell-edge throughput.
- **Random search** reached J = 0.6545 (+4.96 %), improving 8 KPIs and worsening 4. It found that level by evaluation 7 and added less than 10⁻⁴ in the next 138.

Both methods worsen the band-collapsed co-band overlap rate and overlapping neighbours per covered tile. TuRBO worsens them less and has the lower overlap on each of the three bands taken separately.

Three findings shape how these should be read.
- **TuRBO's KPI profile is one draw.** A rerun of the same seed found a different winner than the 2026-09-30 run (J 0.6721 against 0.6706): GPU round-off is amplified by the GP fit into different proposals. Its two runners-up sit within 0.00005 of the winner's J (Section 6.4).
- **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.3). Nothing in the objective stops a search from buying J by switching a weak layer off. On this run's TuRBO winner, 77.0 % of tiles have such a band (Section 6.8). The pipeline does not compute it.
- **The proposal is a large move.** Every one of the 36 cell-bands changes, by up to 11.4°, and two tilts sit within 0.12° of the box edges.

**Caveats.** The results come from one scenario, one search seed per method, and a simplified capacity model. No configuration is recommended for deployment.

---

## 1. Introduction

### 1.1 The problem as a RAN engineer meets it

A 5G/6G site commonly radiates several frequency bands from the same mast, and the bands differ physically:
- Low bands such as 700 MHz propagate farther and penetrate buildings better. They are the coverage floor.
- High bands such as 2600 MHz carry more capacity over a smaller footprint. They are the capacity layer.
- A mid band such as 1800 MHz bridges the two.

In day-to-day optimisation the tilts of those layers are rarely set together. A typical life cycle is:
1. **Planning** sets one electrical tilt per band from a link budget, often the same value on every sector (here 12°, 10° and 8°).
2. **Tuning** then happens one complaint at a time. A drive test, a KPI alarm or a ticket points at a sector with poor edge RSRP, pilot pollution, a coverage hole or a congested carrier.
3. **The fix** is a remote electrical tilt (RET) change on one band of that sector, followed by days of counter collection to see whether it helped and what it broke next door.

Each change is reasonable locally, but the layers interact:
- Uptilting a capacity carrier widens its footprint into its neighbours' and raises co-band overlap and interference.
- Downtilting it pulls its edge in and can open a hole that only the low band then covers, at lower throughput.
- Because the serving layer is chosen by load and signal quality, moving one band's tilt moves traffic between layers, so the counters of carriers that were never touched change too.

The result is a loop that is slow, expensive in field time, and blind to cross-band effects until they show up in the counters. Most networks are left at a configuration that nobody chose as a whole.

### 1.2 What this study does

It treats the tilts of every (cell, band) pair as one coordinated optimisation problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) over real city geometry acts as a digital twin: it scores every proposed configuration on coverage, overlap, signal quality and the throughput the simulated UEs would get. Two searches are compared on the same objective and budget:
- random search, as the model-free control,
- trust-region Bayesian optimization (TuRBO [2]), which learns where to look.

### 1.3 What it offers a RAN engineer, and how it changes the work

**What you get from a run.** Not a single magic setting but a reviewable package:
- A shortlist of measured configurations (`../reports/outputs/solutions_<method>.csv`): the current one, the recommendation and the runners-up, each with all twelve KPIs and its delta against today. Every number is a ray-traced measurement, not a model prediction.
- The tilt change per cell and band (`tilt_change_<method>.csv`, Appendix C), which is the content of a RET change request.
- A cell-impact table (Section 6.6) ranking which carriers gain or lose traffic, so you know where to look first after a change.
- The trade-offs made explicit: where the recommendation gains coverage, where it adds overlap, which layer the traffic moves to.

**What it changes in your work.**
- **From trial-and-error to choosing between measured options.** On this scenario, 145 ray-traced candidates took about 12 minutes on a laptop GPU. The equivalent field loop is one change, then days of counters, per sector.
- **From band-by-band to layer-aware decisions.** The search found that the best configuration widens most carriers but pulls one layer in on four sectors, handing that sector's near area to one band and its reach to another (Section 6.6). That pattern is hard to reach one ticket at a time.
- **From reacting to complaints to planning a rollout.** The cell-impact table says which carriers to watch, and the shortlist offers near-equivalent alternatives when a proposed tilt is mechanically or operationally unacceptable.

**What stays your job.** The study does not replace engineering judgement, and Section 6.8 lists why:
- **Field validation.** One simulated city is not your network. Results are uncalibrated against drive tests, and nothing has been tested on a held-out scenario.
- **Operational limits.** The search does not penalise antenna movement. The recommendation moves every antenna, some by more than 10°, and you would phase such a change or reject parts of it.
- **The model's blind spots.** There is no MCS cap, no scheduler, no mobility and no inter-band interference in the objective.

The value claimed here is therefore narrow and testable: **on one realistic scenario, a joint search over all layers found configurations that improve every reported KPI except co-band overlap, at a computational cost of minutes, and it did so consistently better than an unstructured search with the same budget.**

## 2. Problem Definition

**Operational statement.** Given a multi-band site layout and its current tilts, propose new electrical tilts for every cell and band that close coverage holes and weak areas, keep co-band overlap under control, and raise the signal quality and throughput users get, while each band keeps the role its propagation suits. The proposal must come with its measured effect on every KPI and its tilt delta per antenna, so an engineer can review it before any RET change.

**Network.** The study area is a local city scene (`data/external/scene/scene.xml`), rasterised into a 326 × 310 grid of 20 m tiles: 6,200 × 6,520 m, or 101,060 tiles.
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side.
- **Cells.** Each node has three sectors at azimuths 45°, 165° and 285°, on 25 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve cells and 36 cell-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 cells and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 15^\circ]^{36}$$

Results are reported as offsets from the current configuration: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell (`../reports/tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another cell of the same band within 6 dB of that band's strongest cell;
- fails fewer UEs and raises the throughput the served ones can expect;
- preserves each band's physical role.

**Starting condition** (`../reports/tables/01_eda/`):
- **Coverage.** 11.2 % of the grid is a coverage hole, 30.7 % is weakly covered (−120 to −90 dBm) and 58.1 % has good coverage. Holes are a periphery effect: 0.4 % of tiles within 1 km of a node are holes, against 14.3 % beyond (`hole_summary.csv`). Half the hole area — 5,755 tiles of 11,368 — has no propagation path on any band at all.
- **Band behaviour.**
  - Alone, 700 MHz leaves 13.8 % of the grid in a hole, 1800 MHz 23.5 % and 2600 MHz 28.8 % (`coverage_classes_per_band.csv`).
  - By raw signal, 700 MHz is the strongest layer on 91.9 % of the covered area.
  - The max-throughput serving rule still puts 77.2 % of the covered area on 2600 MHz, whose 216-PRB carriers offer the most throughput, against 15.8 % on 700 MHz (`serving_area_per_band.csv`). The gap between which layer is strongest and which layer serves is the central tension in this configuration.
- **Overlap.** 31.6 % of tiles have at least one overlapping co-band neighbour, with a mean of 0.96 neighbours per covered tile (`overlap_neighbour_summary.csv`).
- **Service.** The UE service failure rate is 21.2 % of UE rows (`../reports/tables/00_simulation/serving_band_mix.csv`). The serving rule refuses nobody, so every one of them stands on a hole tile, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Counted in UE reports, 21.2 % stands in holes, 23.7 % on weak tiles and 55.1 % on good ones (Table 3).

![Study area](../reports/figures/00_simulation/study_area.png)

*Figure 1. Study area: twelve cells on four nodes and a sample of UE positions over the scene. Source: `../reports/figures/00_simulation/study_area.png`.*

![RSRP per band](../reports/figures/00_simulation/rsrp_per_band.png)

*Figure 2. Best-server RSRP per band at the current tilts. Source: `../reports/figures/00_simulation/rsrp_per_band.png`.*

![Demand vs coverage](../reports/figures/01_eda/demand_vs_coverage.png)

*Figure 3. UE demand beside signal strength at the current tilts. Source: `../reports/figures/01_eda/demand_vs_coverage.png`.*

*Table 1. Frequency bands. The PRB limits are N_RB at 15 kHz SCS, TS 38.101-1 Table 5.3.2-1 [4]. Source: [`frequency_bands.csv`](../reports/tables/00_simulation/frequency_bands.csv).*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per cell |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

*Table 2. Scenario. Sources: [`study_area.csv`](../reports/tables/00_simulation/study_area.csv), [`ue_distribution.csv`](../reports/tables/00_simulation/ue_distribution.csv), [`ue_measurement_summary.csv`](../reports/tables/00_simulation/ue_measurement_summary.csv), [`decision_variables.csv`](../reports/tables/00_simulation/decision_variables.csv).*

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

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 11,368 | 11.2 % | 21.2 % |
| Weak (−120 to −90 dBm) | 31,017 | 30.7 % | 23.7 % |
| Good (> −90 dBm) | 58,675 | 58.1 % | 55.1 % |

## 3. Proposed Solutions

Both solutions search the same bounded tilt box (`src/optim/space.py`) with the same Sionna-RT evaluator (`src/optim/evaluator.py`). The evaluator builds the scene once and uses one fixed solver seed, so every candidate shares the same Monte-Carlo noise. The solutions also share one definition of "better": the objective J (Section 3.3). They differ only in where they look.

Each run has four steps (`src/optim/run.py`, `src/optim/report.py`):
1. Evaluate the current configuration.
2. Search, scoring every candidate on all UEs.
3. Re-trace the winner once, to archive its radio map.
4. Publish a shortlist of `optim.n_solutions` = 4 configurations: the current one, the winner, then the highest J.

### 3.1 Sobol random search

Random search is the model-free control. It draws 16 + 128 scrambled Sobol points from a seeded sequence over the full 36-dimensional box. The first 16 are identical to TuRBO's initial design. Because the budget and seed match TuRBO's, the gap between the two measures what the model contributes. Implementation: `src/optim/methods/random/search.py`; configuration: `configs/optim/method/random.yaml`.

### 3.2 TuRBO-1 Bayesian optimization

TuRBO-1 [2] keeps one trust region centred on the best point found since the last restart.
- **Each round.** It fits a Gaussian process to the evaluations since the last restart in the unit cube, and stretches the region along the GP lengthscales. Each candidate perturbs every dimension of the centre with probability min(k / 36, 1), k = `perturbed_dimensions` = 5, and the search Thompson-samples a batch of three candidates.
- **Region size.** The region starts at side 0.8. A round improves when it beats the best by 10⁻³ of its magnitude (`improvement`). The region doubles (up to 1.6) after three consecutive improving rounds and halves after ⌈max(4, 36) / 3⌉ = 12 failed rounds.
- **Restart.** It restarts with a fresh Sobol design when the side falls below 0.5⁷.
- **Budget.** 16 Sobol initial points plus 128 trust-region evaluations.
- **Seeds.** The initial design is the Sobol sequence seeded with `optim.seed`, the same one random search draws from. Each restart design, and each round's GP fit, candidate pool and perturbation mask, is seeded from `numpy.random.SeedSequence([optim.seed, purpose, index])`, keyed by the restart count or the history length. A derived seed, unlike an offset from `optim.seed`, cannot repeat another search seed's draws, so the runs of a seed sweep stay independent replicates.

The implementation uses BoTorch/GPyTorch [3] and follows the BoTorch TuRBO-1 tutorial. The GP only chooses where to look; every reported number is ray-traced. Implementation: `src/optim/methods/turbo/search.py`; configuration: `configs/optim/method/turbo.yaml`; decision records: [ADR 0001](adr/0001-turbo-on-a-weighted-kpi-score.md) and [ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md).

### 3.3 The objective both solutions maximise

The objective is [ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md), implemented in `src/optim/objective.py` on top of `src/kpi/overlap.py::effective_coverage`:

$$J = \frac{1}{|G|}\sum_{g\in G} \frac{\sum_b u_{bg}^2}{\sum_b u_{bg}},
\qquad u_{bg} = \frac{s_{bg}}{1 + \sum_{i} 10^{(R_{bi}(g) - R_{bs}(g))/10}},
\qquad s_{bg} = \mathrm{clip}\!\left(\frac{R_{b,\max}(g) - T_{\text{cov}}}{T_{\text{weak}} - T_{\text{cov}}},\, 0,\, 1\right)$$

A tile scores 0 where no band covers it, that is, where $\sum_b u_{bg} = 0$.

- **Every band is scored, and the tile takes the contraharmonic mean of the band scores.** Each band is weighted by its own score, so the result lies between the plain mean and the best band, and a band that does not cover the tile carries no weight. There is no band selection, and the objective reads no band order.
- $s$ is band $b$'s strongest cell at tile $g$, and $i$ runs over every other cell **on band $b$ alone** above $T_{\text{cov}}$. The fraction is $s$'s share of the power the band delivers to the tile: 1 when it is alone, 1/2 with an equal rival. Every rival costs in proportion to its linear power, so there is no overlap margin and no step. It is co-band: nothing crosses the band axis.
- $u_{bg}$ is 0 where band $b$ does not cover the tile, which includes every tile the ray tracer found no path to.
- $s_{bg}$ is how far that band's strongest cell sits between the hole and weak thresholds. A server at −90 dBm or better keeps all of its utility, one just out of a hole keeps almost none, and power beyond −90 dBm buys nothing.
- $T_{\text{cov}}$ = `kpi.hole_dbm` = −120 dBm and $T_{\text{weak}}$ = `kpi.weak_dbm` = −90 dBm. Each physical quantity keeps one threshold, shared with the KPIs.

In RAN terms, $u$ is a per-layer "dominance at usable level" score: a carrier that is the only strong server on its band where it serves scores 1, pilot pollution on that band costs in proportion to the polluters' power, and a server barely above the coverage threshold scores little.

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

J is therefore primarily a *signal-strength and cleanliness* measure that treats hole-closing as a minor bonus. Over random search's 145 candidates, signed so that a positive value means J and the KPI improve together, J tracks cell-edge RSRP (Spearman 0.87), the weak rate, median RSRP and the hole rate (0.78 to 0.80) most closely. It tracks the UE service failure rate (0.56), the overlap rate (0.42), median SINR (0.41), median and mean throughput (0.38 and 0.28) and overlap neighbours (0.27) loosely, and cell-edge throughput (0.10) and cell-edge SINR (0.08) hardly at all. The pipeline does not produce these correlations; they were computed for this report from random search's history (`../outputs/optim/random/2026-10-02_04-17-10/history.parquet`).

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

   A change is labelled better, worse, unchanged, or undefined when the delta is not a number (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the twelve is weighted into J, so agreement between J and the KPIs is a finding, not a construction. PRB load is not reported: every cell-band with a UE uses its whole usable pool by construction ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)).
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by the winner against the median candidate, sample efficiency, and TuRBO paired with random search on the same seed.
4. **Robustness.** Where the configuration moves demand, not only area, and how it trades one KPI against another. Nothing in J reads demand, so this criterion is entirely a check on the objective rather than a reflection of it.
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
   - The output was per-cell RSRP and SINR on the grid (`../reports/tables/00_simulation/propagation_parameters.csv`).
   - At 25 m masts, 2600 MHz reaches 88.1 % of tiles, 1800 MHz 87.8 % and 700 MHz 91.1 % (`../reports/tables/00_simulation/reach_per_band.csv`).
3. **Service.** Every UE row is served from the radio map at its tile, in the search and in the evaluation alike. No measurement noise, report censoring or position error is modelled.

The 2026-10-02 run regenerated the scenario and baseline radio map from scratch and reproduced the earlier ones: the same scenario ID and the same baseline KPIs.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the artifacts against 17 contract checks, all of which held (`../reports/tables/02_preprocessing/verification_checks.csv`), and against the template checks (`template_checks.csv`).
- The checks cover band and cell order, scenario ID, grid, UE height and columns, SINR shape, tile and extent bounds, the RSRP bound, schedule, duplicate rows and baseline tilts.
- The notebook then wrote typed Parquet tables without dropping or altering a row.
- `notebooks/01_eda.ipynb` recorded data-quality measures and removed nothing.

**Optimization runs.** Notebooks `03a_baseline` and `03b_turbo` ran each method once with search seed 42 (Appendix A).
- Each spent 145 evaluations: the incumbent, 16 initial points and 128 more.
- Every candidate was fully ray-traced and scored on all UEs. No surrogate prediction entered a reported number.
- Each run wrote its history, shortlist, best tilt, best radio map and `run.json` under `../outputs/optim/<method>/<timestamp>/`.

**Evaluation.** `notebooks/04_evaluation.ipynb` calls `src/evaluation/run.py::evaluate`, which reads the finished runs without re-solving anything. It:

1. Checks that all runs share the baseline's scenario, grid, solver settings, bands, band carrier frequencies and KPI definition (23 checks).
2. Recomputes each archived winner's KPIs from its saved radio map, to confirm they were recorded correctly. The archived map is a second solve of the winning tilts at the same solver seed, stored as float32, so the check also bounds the GPU ray tracer's run-to-run noise.
3. Builds the scoreboard against the current configuration.
4. Compares each winner with the candidates its own search evaluated.
5. Maps coverage, overlap, the serving-band mix, cell utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets and a shared initial design, so criterion 3 isolates the model's contribution.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took 2.6 to 3.1 s per candidate (Table 11).

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

All 23 comparability checks held (`../reports/tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`kpi_reproducibility.csv`) match the recorded values to float round-off. Over all 39 recorded measures, the largest absolute gap is 8.2 × 10⁻⁶ Mbit/s, on TuRBO's median throughput, and **on J and every rate it is zero**. The differences discussed below therefore come from the configurations, not from bookkeeping.

**J is rounded before any search reads it.** The ray tracer's Monte-Carlo stream is seeded, but the GPU adds path contributions into a tile in a varying order, so the same tilts traced in two processes can differ in J's trailing digits. TuRBO's GP fit turns any such difference into a different proposal. `src/optim/objective.py` rounds J to 10⁻⁶, below the printed precision. Rounding makes a mismatch unlikely, not impossible.

**What the rerun showed.** Random search, which has no model, reproduced its 2026-09-30 result exactly: the same winner at the same evaluation, J 0.6545, the same twelve KPIs. TuRBO did not: from the same seed it found a different winner, J 0.6721 at evaluation 141 against 0.6706 at evaluation 131. Its KPI profile moved with it. That is a direct measurement of limitation 3 (Section 6.8): one TuRBO run is one draw even at a fixed seed.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Sources: [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv).*

| | Current | Random search | TuRBO |
|---|---:|---:|---:|
| **Objective J ↑** | 0.6235 | 0.6545 | **0.6721** |
| Coverage hole rate ↓ | 0.1125 | 0.1093 | **0.1088** |
| Weak coverage rate ↓ | 0.3069 | 0.2707 | **0.2632** |
| Co-band overlap rate ↓ | **0.3164** | 0.3356 *(worse)* | 0.3216 *(worse)* |
| Overlap neighbours per covered tile ↓ | **0.9625** | 1.0352 *(worse)* | 1.0037 *(worse)* |
| Median RSRP p50 [dBm] ↑ | −84.11 | −81.92 | **−81.43** |
| Cell-edge RSRP p05 [dBm] ↑ | −108.56 | −107.66 | **−107.45** |
| Median SINR p50 [dB] ↑ | 14.26 | 15.79 | **17.62** |
| Cell-edge SINR p05 [dB] ↑ | −0.88 | −1.02 *(worse)* | **−0.52** |
| UE service failure rate ↓ | 0.2124 | 0.2088 | **0.2087** |
| Cell-edge throughput p05 [Mbit/s] ↑ | 14.64 | 14.58 *(worse)* | **16.11** |
| Median throughput p50 [Mbit/s] ↑ | 64.06 | 68.17 | **74.27** |
| Mean throughput [Mbit/s] ↑ | 92.08 | 99.74 | **108.10** |
| **KPIs better / worse, of 12** | — | 8 / 4 | 10 / 2 |

![KPI improvement](../reports/figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per KPI and method; J is not among the panels. Source: `../reports/figures/04_evaluation/kpi_improvement.png`.*

**TuRBO wins J:** +7.78 %, against +4.96 % for random search. On the twelve KPIs, TuRBO improves ten and worsens two: the co-band overlap rate and overlapping neighbours per covered tile. Random search improves eight and worsens those two plus cell-edge SINR and cell-edge throughput.

**Head to head, TuRBO is better than random search on all twelve KPIs.** The margins that matter operationally:
- **Coverage.** Weak coverage 26.3 % against 27.1 % of the grid, median RSRP 0.5 dB higher. The hole rate is close (0.1088 against 0.1093), and the UE service failure rate differs by one UE report of 10,087.
- **Signal quality.** Median SINR 1.8 dB higher; cell-edge SINR 0.5 dB higher, and above today's where random search's is below it.
- **Throughput.** Mean +17.4 % against +8.3 %, and the cell edge rises (+10.0 %) where random search's falls.
- **Overlap.** Both worsen the collapsed rate, TuRBO by 1.6 % and random search by 6.1 %; overlap neighbours rise 4.3 % against 7.6 %.

### 6.3 Did the search matter?

*Table 5. Winner against the candidates each run evaluated. Source: [`winner_vs_candidates.csv`](../reports/tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.6235 | 0.6352 | 0.6352 | 0.6456 | 0.6545 |
| TuRBO | 0.6235 | 0.6352 | **0.6687** | **0.6716** | **0.6721** |

*Table 6. Best J reached after a fixed number of evaluations. Source: [`sample_efficiency.csv`](../reports/tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | TuRBO |
|---:|---:|---:|
| 10 | **0.6544** | **0.6544** |
| 25 | 0.6544 | **0.6565** |
| 50 | 0.6544 | **0.6657** |
| 100 | 0.6545 | **0.6711** |
| 145 | 0.6545 | **0.6721** |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `../reports/figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](../reports/figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `../reports/figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **better** than the current configuration: their median is 0.6352 against 0.6235, and their 90th percentile reaches 0.6456. The strength factor rewards the wider footprints a random tilt tends to produce. For an engineer, that says the committed per-band tilts are a weak starting point: almost any uptilt helps.

The evidence that the model earned TuRBO's margin:
- TuRBO's **median** candidate (0.6687) scores above random search's single **best** (0.6545).
- TuRBO and random search share the same 16 Sobol points and diverge only once the model proposes. That shared design has a median of 0.6352 for both, while TuRBO's all-candidate median is 0.6687 against random search's 0.6352.
- TuRBO's 128 trust-region proposals average 0.6669, against 0.6335 for its 16 Sobol points (`../reports/tables/03b_turbo/turbo_evaluations_by_proposer.csv`).

Per method:
- **Random search** reached 0.6544 at evaluation 7; its only later gain, at evaluation 95, adds less than 10⁻⁴.
- **TuRBO** passed random search's final best at evaluation 17, its first trust-region round, and found its best at **evaluation 141 of 145**, after improvements at 136, 139 and 140. Its trust region never collapsed into a restart. A larger budget would plausibly still improve it.

The paired gain of TuRBO over random search is **+0.0176** on the one seed (`paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.

![Hole vs overlap trade-off](../reports/figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `../reports/figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Both methods lowered the hole rate, but not because J asked.** A tile that has just crossed the hole threshold sits near −120 dBm, where the strength factor is near zero, so closing holes buys J almost nothing (Section 3.3). The hole-rate gains in Table 4 are a side effect of uptilting, which widens every footprint.

**Overlap-reducing configurations were available to TuRBO only.** TuRBO's candidate set contained one with an overlap rate of **0.2975** (evaluation 66, J 0.6673), below the incumbent's 0.3164 (`sample_efficiency.csv`). Random search never beat the incumbent's overlap rate in 145 candidates. J did not pick the lowest-overlap candidate, which is consistent with J tracking the overlap rate only loosely (Section 3.3). TuRBO's published shortlist (`../reports/outputs/solutions_turbo.csv`) holds two runners-up within 0.00005 of J below the winner, at overlap rates of 0.3214 and 0.3212 against the winner's 0.3216. An engineer who prefers a lower overlap rate can take a runner-up at no measurable cost in J.

**Per band, TuRBO lowers overlap on every layer, yet the collapsed rate rises.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`../reports/tables/04_evaluation/band_kpis.csv`):

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Random search | TuRBO |
|---|---:|---:|---:|
| 2600 MHz | 0.2010 | 0.1967 | **0.1840** |
| 1800 MHz | 0.2067 | 0.2065 | **0.1946** |
| 700 MHz | 0.2396 | 0.2304 | **0.2149** |
| Band-collapsed (the reported KPI) | **0.3164** | 0.3356 | 0.3216 |

TuRBO has the lower overlap on each band, cutting 700 MHz and 2600 MHz most. The collapsed rate still rises, because what crowding remains lands on fewer shared tiles: the three layers' crowded areas coincide less than before. Random search lowers every band, 1800 MHz only marginally, yet raises the collapsed rate most. J prices each band's crowding separately, so it sees TuRBO's per-band gains and not the union the collapsed KPI counts.

*Table 7b. Coverage class by area and by demand, with demand counted in UE reports. Nothing in J reads this view, so it is a check on the result, not a reflection of it. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|
| Hole | 11.2 % / 21.2 % | 10.9 % / 20.9 % | 10.9 % / 20.9 % |
| Weak | 30.7 % / 23.7 % | 27.1 % / 21.8 % | 26.3 % / 21.3 % |
| Good | 58.1 % / 55.1 % | 62.0 % / 57.4 % | 62.8 % / 57.8 % |

*Table 8. Overlapping co-band neighbours per configuration. Source: [`overlap_neighbour_summary.csv`](../reports/tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | **0.96** | **64.4 %** | 17.6 % |
| Random search | 1.04 | 62.3 % | 16.1 % |
| TuRBO | 1.00 | 63.9 % | **15.1 %** |

![Coverage before and after](../reports/figures/04_evaluation/coverage_before_after.png)

*Figure 8. Best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold. Source: `../reports/figures/04_evaluation/coverage_before_after.png`.*

![RSRP change maps](../reports/figures/04_evaluation/rsrp_change_maps.png)

*Figure 9. Change in best-server RSRP for each method's best configuration. Source: `../reports/figures/04_evaluation/rsrp_change_maps.png`.*

**TuRBO's map change is larger and less uniform.** It raises best-server RSRP on 88.0 % of reached tiles by a median of 3.0 dB, closing 636 hole tiles and opening 265. Random search improves 93.1 % by a median of 2.1 dB, closing 477 and opening 156. The holes TuRBO opens sit where it pulls a carrier in (Section 6.6); they are the tiles a drive test after rollout should cover first.

**Demand moves out of weak coverage under both methods.** The share of UE reports on weak tiles falls from 23.7 % to 21.3 % under TuRBO and 21.8 % under random search; the share on good tiles rises from 55.1 % to 57.8 % and 57.4 %. The share on hole tiles falls only from 21.2 % to 20.9 % under both, because most of it is one hotspot no tilt reaches.

**TuRBO thins the pile-ups but not the mean.** TuRBO cuts the share of covered tiles with three or more overlapping neighbours from 17.6 % to 15.1 %. The share with none falls slightly, from 64.4 % to 63.9 %, and the mean rises from 0.96 to 1.00. The power share prices a pile-up far more than a single rival, which is what TuRBO trades on.

**No objective parameters to vary, and no band priority either.** The objective has no parameters and reads no band order. What remains untested is the single search seed, which Section 6.8 lists.

### 6.5 Capacity impact

*Table 9. UE service. Throughput statistics are in Table 4. Source: [`ue_service_summary.csv`](../reports/tables/04_evaluation/ue_service_summary.csv).*

| Configuration | Not served | Served SINR p10 [dB] | Served SINR median [dB] | On 2600 / 1800 / 700 MHz |
|---|---:|---:|---:|---|
| Current | 21.2 % | 0.44 | 12.30 | 53.0 % / 14.2 % / 11.6 % |
| Random search | 20.9 % | 0.88 | 13.53 | 54.1 % / 12.5 % / 12.5 % |
| TuRBO | **20.9 %** | **1.73** | **15.01** | 50.6 % / 16.1 % / 12.5 % |

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `../reports/figures/04_evaluation/serving_band_mix.png`.*

![Cell-band throughput](../reports/figures/04_evaluation/cell_band_throughput.png)

*Figure 11. Median estimated throughput per cell-band, current and recommended. Source: `../reports/figures/04_evaluation/cell_band_throughput.png`.*

**Service.** Both methods fail 0.4 points fewer UE reports than the current 21.2 %: 2,105 of 10,087 under TuRBO against 2,142 today. Every failure stands on a hole tile, since the serving rule refuses nobody. Served SINR rises under both, most under TuRBO: 1.29 dB at the 10th percentile and 2.71 dB at the median.

**TuRBO moves traffic off the capacity layer and onto 1800 MHz.** 2600 MHz falls from 53.0 % to 50.6 % of UE reports and 1800 MHz rises from 14.2 % to 16.1 %; random search moves slightly the other way. That spread is where TuRBO's throughput gain comes from, and it shows in the cell-impact table (`../reports/tables/04_evaluation/cell_impact.csv`):
- **Uptilted carriers pick up load:** n2c1's 1800 MHz carrier, uptilted 9.89°, goes from 194 to 418 served reports with 5.3 dB more median SINR and 13.1 Mbit/s more median throughput.
- **Downtilted carriers shed it:** n0c2's 2600 MHz carrier, downtilted 2.96°, drops from 206 reports to 29, and n3c2's 1800 MHz carrier, downtilted 4.80°, from 137 to 5.

**Per band** (`../reports/tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- **Area covered** rises on 2600 MHz, 71.2 % → 74.4 %, and 1800 MHz, 76.5 % → 77.7 %, and holds on 700 MHz at 86.2 %. The low band keeps its coverage-floor role.
- **Mean RSRP where covered** improves on every layer: 2600 MHz −97.2 → −91.1 dBm, 1800 MHz −91.5 → −88.3 dBm, 700 MHz −84.2 → −82.5 dBm.
- **Median served SINR** rises on 2600 MHz, from 10.0 to 13.6 dB. It falls slightly on 1800 MHz, from 16.0 to 15.3 dB, as that layer takes on more UEs, and holds on 700 MHz (20.9 → 20.8 dB).
- **Per-cell-band median throughput** rises on 2600 MHz (82.2 → 111.5 Mbit/s) and 1800 MHz (67.8 → 68.1) and falls on 700 MHz (45.5 → 41.2). The slowest carrier, n2c1's 700 MHz, slows from 21.6 to 17.9 Mbit/s (`cell_band_load.csv`).

About 21 % of UE reports remain unserved, all of them on hole tiles, and 17.5 % of UE positions have no path to any cell (Table 2). Tilt alone cannot serve them; that is a site or coverage decision.

### 6.6 Recommended tilt changes

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per cell and band in the highest-J (TuRBO) configuration. Source: `../reports/figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tilt_movement_summary.csv`](../reports/tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Cells moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 8.69 | 11.37 | −8.19 |
| 1800 MHz | 12 / 12 | 7.76 | 9.89 | −6.19 |
| 700 MHz | 12 / 12 | 6.59 | 7.10 | −3.25 |

Every one of the 36 cell-bands moved. The configuration is a net uptilt on every band, and the structure is the point:
- **Most carriers uptilt hard.** 30 of 36 are uptilted, 2600 MHz most. That widening is what lifts the hole and weak rates.
- **Six of the 36 are downtilted, on four sectors.**

  | Sector | Downtilted carriers | Uptilted hard on the same sector |
  |---|---|---|
  | n0c2 | 2600 MHz +2.96°, 700 MHz +6.29° | 1800 MHz −8.39° |
  | n1c0 | 1800 MHz +4.62°, 700 MHz +6.76° | 2600 MHz −6.53° |
  | n2c1 | 700 MHz +6.98° | 1800 MHz −9.89° |
  | n3c2 | 1800 MHz +4.80° | 2600 MHz −9.85° |

  On each of those sectors the search hands the near area to one layer and the reach to another, which is a layer-role decision a per-band procedure does not make. It is consistent with TuRBO having the lower overlap on every band (Table 7a) and with the load shift of Section 6.5.
- **The box binds at both ends.** Proposed tilts span **0.11° to 14.98°**, both on sector n2c1: its 1800 MHz carrier sits 0.11° from the 0° bound and its 700 MHz carrier 0.02° from the 15° bound. The largest single change is 11.37°, n1c2 on 2600 MHz.

With one seed, it is not established which of TuRBO's per-cell differences matter and which reflect where the trust region happened to be when the budget ended. The 2026-09-30 run downtilted a different set of carriers (eight, also on four sectors) for a J within 0.0015 of this one.

The largest traffic shifts under TuRBO (`../reports/tables/04_evaluation/cell_impact.csv`):
- n2c1 on 1800 MHz, +224 served reports;
- n0c2 on 2600 MHz, −177;
- n2c1 on 2600 MHz, −166;
- n3c2 on 1800 MHz, −132;
- n3c2 on 700 MHz, +127.

Those are the cells to watch after a rollout.

### 6.7 Cost

*Table 11. Search cost. Sources: [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random search | 145 | 95 | 7.49 | 9.45 | 3.1 | +0.0309 | 0.0033 |
| TuRBO | 145 | 141 | 6.33 | 12.03 | 2.6 | **+0.0485** | **0.0040** |

TuRBO's GP fitting and acquisition, which run on the CPU, added about 5.7 minutes of wall clock on top of its 6.3 minutes of ray tracing. Timings are only indicative: the same configurations traced at 2.3 to 2.5 s per evaluation on 2026-09-30, on the same RTX 3050 Ti laptop GPU.

**TuRBO is the more cost-effective of the two by J per minute**, and it overtakes random search's final result within its first 17 evaluations.

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed. The TuRBO–random paired gain is +0.0176, resting on one pair.
3. **Winner's curse, and TuRBO's sensitivity to J.**
   - Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
   - The GPU ray tracer is reproducible only to round-off, and the GP fit amplifies any difference in J into different proposals. The rerun of seed 42 found a different winner (Section 6.1).
   - TuRBO's two runners-up sit within 0.00005 of the winner's J with overlap rates from 0.3212 to 0.3214, so any per-KPI claim about TuRBO's winner beyond J is one draw.
4. **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.3).
   - On this run's TuRBO winner, 77.0 % of tiles have such a band. If each could shed its most costly band independently, the ceiling on the gain would be 0.033, against the winner's whole improvement of 0.049.
   - The power share makes this common: almost any covered second layer with a co-band rival scores below 1, and so below a clean first layer.
   - That ceiling is not reachable, because darkening a band on one tile changes it on many.
   - Both numbers come from a one-off analysis of the archived radio map, not from the pipeline.
   - Nothing in the objective stops a search from buying J by switching a weak layer off.
5. **Inter-band interference is priced nowhere.** The power share is co-band by definition, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The contraharmonic mean does lower a tile for a weak second layer, but it does so whether or not that layer interferes. Fixing this needs an interference model, not a reweighting.
6. **Nothing is weighted by demand.** A hole where nobody stands costs exactly what a hole in a hotspot costs. Table 7b is the only place demand appears.
7. **The capacity model is a simplification.** It drives the estimated throughput and every per-cell-band figure. It uses a placeholder usable PRB share of 0.8, an equal share with no scheduler, a Shannon rate with no MCS cap, full-load co-band interference, and per-RE thermal noise with no receiver noise figure.
8. **Tilts at both bounds, and no movement penalty.** The winner places one cell-band at 0.11° and another at 14.98°, and moves all 36 antennas, by up to 11.37°. A real RET range or a change-management policy may not allow either.
9. **No held-out validation.** Nothing re-solves an optimized tilt on an unseen scenario, so no number here measures transfer.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability.** No J in this report is comparable with a value scored under any earlier objective. `src/evaluation/runs.py` refuses to pool runs recorded under a different KPI set, but there is no version guard on the objective's functional form, so a change to J alone would not be detected.

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Random search | TuRBO |
|---|---|---|
| 1. Objective J | +0.0309 | **+0.0485** |
| 2. Reported KPIs | 8 better, 4 worse | **10 better, 2 worse; better than random search on all 12** |
| 3. Search effectiveness | Best at 7 (to within 10⁻⁴), nothing more in the next 138 | **Median candidate above random search's best; still improving at 141 of 145** |
| 4. Robustness | Largest rise in the overlap rate; load pulled slightly onto 2600 MHz | Lower overlap on every band, but the collapsed rate still rises; load spread off 2600 MHz; a rerun of the seed gives a different winner |
| 5. Cost | 9.4 min, 145 evaluations | 12.0 min, 145 evaluations; **more J per minute** |

**Conclusions.**

- **TuRBO reached the highest J, and there is good evidence the model earned it.** Its median candidate (0.6687) scored above random search's best. It also shares its first 16 Sobol points with random search and diverges only once the model starts proposing.
- **At equal budget, TuRBO is better than random search on every reported KPI**, and it is the only method that raises cell-edge SINR and cell-edge throughput.
- **Neither method lowers co-band overlap overall.** TuRBO has the lower overlap on each band and fewer 3+ pile-ups, but the band-collapsed rate and the mean neighbour count still rise.
- **The objective barely pays for closing holes.** A newly covered tile arrives near −120 dBm, where the strength factor is near zero. The hole-rate gains are a side effect of uptilting.
- **The winning pattern is a layer-role decision.** TuRBO widens most carriers and, on four sectors, pulls one layer in while another on the same sector widens, moving traffic from 2600 MHz onto 1800 MHz. That is where its throughput gain comes from.
- **Every result** depends on:
  - one scenario and one seed, and a TuRBO run that is not reproducible bit for bit;
  - a simplified capacity model;
  - an objective that is not monotone in the layers present and does not price inter-band interference at all.

**What this means for a RAN engineer today.**
- Treat the output as a **ranked set of measured options**, not an instruction. The shortlist and the cell-impact table are the useful artefacts: they say what each option buys and where traffic will move.
- **The direction is clearer than the exact values.** Both searches agree that the committed tilts are too steep for this layout, and TuRBO's per-sector pattern (one layer near, one far) is the hypothesis to test first in the field, on one sector, before any network-wide change.
- **Phase the change.** No movement penalty is applied, so the raw recommendation moves every antenna. Start from the sectors with the largest traffic shifts (Section 6.6) and validate with a drive test around the tiles where holes open.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b), and re-trace the shortlists under other solver seeds. The TuRBO–random margin of +0.0176 needs an interval before it can be called decisive, and the 2026-10-02 rerun already shows TuRBO's winner moves between runs.
3. **Decide whether the objective's non-monotonicity is acceptable.** This is the most consequential open question left, and the power share makes it reach most of the grid. Measure it as part of the pipeline rather than ad hoc, and re-trace the shortlisted configurations with the weakest layer at each tile removed, to measure whether any reachable tilt actually collects it.
4. **Give TuRBO a larger budget.** It found its best at evaluation 141 of 145 and never restarted, so the trust region was still productive when the budget ended.
5. **Add a movement constraint or penalty**, and widen or confirm the tilt box against the real RET range. The winner sits within 0.12° of both bounds and moves all 36 antennas.
6. **Model inter-band interference.** The overlap count and the power share are co-band and J does not read SINR, so nothing in the study prices a strong neighbour on another layer. This needs a model, not a reweighting.
7. **Address the out-of-reach demand.** Every UE still unserved stands on a hole tile, largely one hotspot with no propagation path. Tilt alone cannot lower the UE service failure rate much further. That is a coverage decision, not a tilt one.
8. **Build held-out scenario validation** before drawing a method-level conclusion.

---


## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/external/scene/scene.xml` (not in Git) |
| Layout | 4 nodes, 1,732 m triangle plus centroid, 3 sectors at 45° / 165° / 285°, 25 m masts |
| Tilt | Current 12° (2600 MHz), 10° (1800 MHz), 8° (700 MHz); bounds [0°, 15°], every cell-band |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6; edge percentile 5 (`LOW_PERCENTILE` in `src/kpi/quality.py`, a code constant) |
| Objective | co-band power share × strength per band, contraharmonic mean over bands; no parameters; reads `hole_dbm` and `weak_dbm`; rounded to 10⁻⁶ (ADR 0002) |
| Capacity | max-throughput cell selection over an equal share of 0.8 × `max_prb`, candidates above −120 dBm, SCS 15 kHz, connection in report-time order |
| Noise | k·T·SCS per resource element at 298.15 K (`simulation.radio_map.bands[].scs_hz`), no receiver noise figure |
| Search | seed 42 (`optim.seed`); TuRBO's restart and proposal seeds derived from it by `numpy.random.SeedSequence`; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12, improvement 10⁻³, perturbed dimensions 5; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `../outputs/optim/random/2026-10-02_04-17-10/` |
| TuRBO | `../outputs/optim/turbo/2026-10-02_04-27-19/` |

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
  ++optim.method.budget.n_init=4 ++optim.method.budget.n_iter=8"
```

The variable is one line, space-separated. The scene, ray-tracing fidelity and seeds keep their defaults.

### Appendix B. Index of generated tables and figures

| Stage | Tables (`../reports/tables/…`) | Figures (`../reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `cell_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `cell_band_throughput`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `template_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `decision_variables`, `optimizer_features`, `objective_decomposition`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map`, `effective_coverage`, `serving_multiplicity` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_random`, `tilt_movement_random`, `kpi_comparison_random`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `kpi_progress`, `tilt_movement_random`, `coverage_before_after_random`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `band_kpis`, `ue_service_summary`, `cell_band_load`, `cell_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_ue_service_failure_rate`, `tradeoff_overlap_rate_vs_ue_service_failure_rate`, `rsrp_change_maps`, `coverage_before_after`, `coverage_class_maps`, `overlap_neighbour_maps`, `ue_throughput_maps`, `band_kpi_panels`, `serving_band_mix`, `cell_band_throughput`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `../reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`recommended_tilt.csv`](../reports/tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`tilt_change_turbo.csv`](../reports/outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every cell; bounds are [0°, 15°]. Negative Δ is an uptilt.*

| Cell | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0c0 | 1.62 (-10.38) | 1.58 (-8.42) | 1.75 (-6.25) |
| n0c1 | 1.44 (-10.56) | 2.44 (-7.56) | 1.40 (-6.60) |
| n0c2 | 14.96 (+2.96) | 1.61 (-8.39) | 14.29 (+6.29) |
| n1c0 | 5.47 (-6.53) | 14.62 (+4.62) | 14.76 (+6.76) |
| n1c1 | 1.59 (-10.41) | 2.08 (-7.92) | 1.03 (-6.97) |
| n1c2 | 0.63 (-11.37) | 2.07 (-7.93) | 1.33 (-6.67) |
| n2c0 | 0.90 (-11.10) | 0.52 (-9.48) | 1.08 (-6.92) |
| n2c1 | 8.56 (-3.44) | 0.11 (-9.89) | 14.98 (+6.98) |
| n2c2 | 0.75 (-11.25) | 1.26 (-8.74) | 2.71 (-5.29) |
| n3c0 | 6.19 (-5.81) | 1.36 (-8.64) | 1.02 (-6.98) |
| n3c1 | 1.42 (-10.58) | 3.31 (-6.69) | 0.90 (-7.10) |
| n3c2 | 2.15 (-9.85) | 14.80 (+4.80) | 1.73 (-6.27) |

Six of the 36 cell-bands are downtilted: one on 2600 MHz, two on 1800 MHz and three on 700 MHz, all on sectors n0c2, n1c0, n2c1 and n3c2.

---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[5] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
