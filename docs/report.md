# Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

*Band-tilt project report. Every number, table and figure below comes from the optimization runs of 2026-10-06 listed in Appendix A, read by notebooks `00_simulation` through `04_evaluation` as re-executed on 2026-10-06, and the configuration in `configs/`. Paths are relative to this file (`docs/`): generated tables and figures are under `../reports/`, run directories under `../outputs/`. RSRP, interference and noise are all per resource element: thermal noise is k·T over one 15 kHz subcarrier, not over the channel bandwidth. Twelve KPIs are recorded for every candidate; the evaluation reports eight over all bands and the coverage, overlap, RSRP and SINR measures per band ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)).*

---

## Abstract

This study asks whether a network-wide search over antenna tilts can improve coverage in a multi-band sector layout when every band on every sector is tuned jointly, not one band at a time.

The study area is a 6.2 × 6.5 km urban scene ray-traced with Sionna-RT. It holds four nodes on 25 m masts, with three sectors each (twelve in all), carrying three bands: 700, 1800 and 2600 MHz. That gives 36 absolute-tilt decision variables in [0°, 20°]. The starting tilts are one value per band: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz. A week-long, time-varying population of 10,087 UE positions was drawn over the scene, and every UE counts, in the search and in the evaluation.

Candidates were ray-traced and scored on one objective J ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)). Per tile, each band scores its strongest sector's share of the band's received power, scaled by how usable that sector's signal is. The tile then takes the *contraharmonic mean* of those band scores, so every covered layer counts in proportion to how well it serves. J lies in [0, 1], is 1 only when every covered band on every tile has one server at or above the weak threshold and no co-band rival above the hole threshold, and has no free parameters. Twelve KPIs were measured beside it, none of them weighted into it.

Two searches started from the same current configuration with matched budgets of 145 evaluations and the same seed:
- Sobol random search,
- TuRBO-1 Bayesian optimization.

**Results.**
- **TuRBO** scored highest on J: 0.6727 against 0.6236 currently, a 7.87 % gain. It improved 7 of the 8 network KPIs, worsening only the co-band overlap rate, and is better than random search on all eight and on every per-band RSRP and SINR percentile. It is the only method that raises cell-edge and median throughput and lowers the UE service failure rate.
- **Random search** reached J = 0.6438 (+3.24 %), improving 2 network KPIs and worsening 6, among them the hole rate, the UE service failure rate and cell-edge throughput. Most of its Sobol candidates score below the current configuration.

Both methods worsen the band-collapsed co-band overlap rate. TuRBO worsens it less, has the lower overlap on each of the three bands taken separately, and leaves overlapping neighbours per covered tile slightly lower than today.

Three findings shape how these should be read.
- **TuRBO's KPI profile is one draw.** One seed, one run: the GPU ray tracer is reproducible only to round-off, and the GP fit can amplify that into different proposals. Its two runners-up sit within 0.0002 of the winner's J at overlap rates from 0.3350 to 0.3419 (Section 6.4).
- **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.3). Nothing in the objective stops a search from buying J by switching a weak layer off. On this run's TuRBO winner, 77.2 % of tiles have such a band (Section 6.8). The pipeline does not compute it.
- **The proposal is a large move.** Every one of the 36 sector-bands changes, by up to 11.9°, and two tilts sit within 0.15° of the box edges.

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

It treats the tilts of every (sector, band) pair as one coordinated optimisation problem and evaluates it entirely in simulation. A ray tracer (Sionna-RT [1]) over real city geometry acts as a digital twin: it scores every proposed configuration on coverage, overlap, signal quality and the throughput the simulated UEs would get. Two searches are compared on the same objective and budget:
- random search, as the model-free control,
- trust-region Bayesian optimization (TuRBO [2]), which learns where to look.

### 1.3 What it offers a RAN engineer, and how it changes the work

**What you get from a run.** Not a single magic setting but a reviewable package:
- A shortlist of measured configurations (`../reports/outputs/solutions_<method>.csv`): the current one, the recommendation and the runners-up, each with all twelve KPIs and its delta against today. Every number is a ray-traced measurement, not a model prediction.
- The tilt change per sector and band (`tilt_change_<method>.csv`, Appendix C), which is the content of a RET change request.
- A sector-impact table (Section 6.5) ranking which carriers gain or lose traffic, so you know where to look first after a change.
- The trade-offs made explicit: where the recommendation gains coverage, where it adds overlap, which layer the traffic moves to.

**What it changes in your work.**
- **From trial-and-error to choosing between measured options.** On this scenario, 145 ray-traced candidates took 10 to 13 minutes on a laptop GPU. The equivalent field loop is one change, then days of counters, per sector.
- **From band-by-band to layer-aware decisions.** The search found that the best configuration widens most carriers but pulls one or two layers in on six sectors, handing that sector's near area to one band and its reach to another (Section 6.6). That pattern is hard to reach one ticket at a time.
- **From reacting to complaints to planning a rollout.** The sector-impact table says which carriers to watch, and the shortlist offers near-equivalent alternatives when a proposed tilt is mechanically or operationally unacceptable.

**What stays your job.** The study does not replace engineering judgement, and Section 6.8 lists why:
- **Field validation.** One simulated city is not your network. Results are uncalibrated against drive tests, and nothing has been tested on a held-out scenario.
- **Operational limits.** The search does not penalise antenna movement. The recommendation moves every antenna, six of them by more than 10°, and you would phase such a change or reject parts of it.
- **The model's blind spots.** There is no MCS cap, no scheduler, no mobility and no inter-band interference in the objective.

The value claimed here is therefore narrow and testable: **on one realistic scenario, a joint search over all layers found a configuration that improves every network KPI except the band-collapsed co-band overlap rate, at a computational cost of minutes, and it did so better than an unstructured search with the same budget, which on this box mostly made the network worse.**


## 2. Problem Definition

**Operational statement.** Given a multi-band site layout and its current tilts, propose new electrical tilts for every sector and band that close coverage holes and weak areas, keep co-band overlap under control, and raise the signal quality and throughput users get, while each band keeps the role its propagation suits. The proposal must come with its measured effect on every KPI and its tilt delta per antenna, so an engineer can review it before any RET change.

**Network.** The study area is a local city scene (`data/scenes/hanoi/scene.xml`), rasterised into a 326 × 310 grid of 20 m tiles: 6,200 × 6,520 m, or 101,060 tiles.
- **Nodes.** Four nodes sit on the corners and centroid of an equilateral triangle with a 1,732 m side.
- **Sectors.** Each node has three sectors at azimuths 45°, 165° and 285°, on 25 m masts. The antenna is an 8 × 8 cross-polarised TR 38.901 panel with 4.85 dBm reference-signal power per resource element.
- **Bands.** Every sector carries three bands, giving twelve sectors and 36 sector-band pairs (Tables 1 and 2, Figure 1).

**Decision variable.** For N = 12 sectors and B = 3 bands, the optimizer chooses one absolute electrical tilt per pair:

$$\boldsymbol{\theta} = [\theta_{1,1},\dots,\theta_{1,B},\dots,\theta_{N,B}] \in [0^\circ, 20^\circ]^{36}$$

Results are reported as offsets from the current configuration: 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every sector (`../reports/tables/00_simulation/decision_variables.csv`). No step size or maximum change is imposed. Tilt movement is reported, not penalised.

**Goal.** A configuration that:
- reduces coverage holes, meaning tiles whose strongest layer is at or below −120 dBm;
- reduces co-band overlap, meaning another sector of the same band within 6 dB of that band's strongest sector;
- fails fewer UEs and raises the throughput the served ones can expect;
- preserves each band's physical role.

**Starting condition** (`../reports/tables/01_eda/`):
- **Coverage.** 11.3 % of the grid is a coverage hole, 30.7 % is weakly covered (−120 to −90 dBm) and 58.1 % has good coverage. Holes are a periphery effect: 0.4 % of tiles within 1 km of a node are holes, against 14.3 % beyond (`hole_summary.csv`). Half the hole area — 5,747 tiles of 11,370 — has no propagation path on any band at all.
- **Band behaviour.**
  - Alone, 700 MHz leaves 13.8 % of the grid in a hole, 1800 MHz 23.5 % and 2600 MHz 28.8 % (`coverage_classes_per_band.csv`).
  - By raw signal, 700 MHz is the strongest layer on 91.9 % of the covered area.
  - The max-throughput serving rule still puts 77.2 % of the covered area on 2600 MHz, whose 216-PRB carriers offer the most throughput, against 15.8 % on 700 MHz (`serving_area_per_band.csv`). The gap between which layer is strongest and which layer serves is the central tension in this configuration.
- **Overlap.** 31.6 % of tiles have at least one overlapping co-band neighbour, with a mean of 0.96 neighbours per covered tile (`overlap_neighbour_summary.csv`).
- **Service.** The UE service failure rate is 21.2 % of UE rows (`../reports/tables/00_simulation/serving_band_mix.csv`). The serving rule refuses nobody, so every one of them stands on a hole tile, mostly one hotspot 3.4 km from the nearest node with no propagation path at its centre (`hotspots.csv`, `hole_summary.csv`).
- **Demand.** Counted in UE reports, 21.2 % stands in holes, 23.7 % on weak tiles and 55.1 % on good ones (Table 3).

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
| Scenario ID | `scn_b7b162aabf88a3ee` |
| Grid | 326 × 310 tiles, 20 m (6,200 × 6,520 m) |
| Nodes / sectors / sector-band pairs | 4 / 12 / 36 |
| Mast height | 25 m |
| Current tilt (2600 / 1800 / 700 MHz) | 12° / 10° / 8° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE positions drawn | 10,087 |
| UE positions with no path to any sector | 17.5 % |

*Table 3. Coverage class by area and by demand at the current tilts. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/01_eda/coverage_by_area_and_demand.csv).*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole (≤ −120 dBm) | 11,370 | 11.3 % | 21.2 % |
| Weak (−120 to −90 dBm) | 31,021 | 30.7 % | 23.7 % |
| Good (> −90 dBm) | 58,669 | 58.1 % | 55.1 % |

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
- $s$ is band $b$'s strongest sector at tile $g$, and $i$ runs over every other sector **on band $b$ alone** above $T_{\text{cov}}$. The fraction is $s$'s share of the power the band delivers to the tile: 1 when it is alone, 1/2 with an equal rival. Every rival costs in proportion to its linear power, so there is no overlap margin and no step. It is co-band: nothing crosses the band axis.
- $u_{bg}$ is 0 where band $b$ does not cover the tile, which includes every tile the ray tracer found no path to.
- $s_{bg}$ is how far that band's strongest sector sits between the hole and weak thresholds. A server at −90 dBm or better keeps all of its utility, one just out of a hole keeps almost none, and power beyond −90 dBm buys nothing.
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
| 1, 0.5 (second band shared by two equal sectors) | 0.833 |
| 1, 0.333 (second band at −110 dBm) | 0.833 |
| 0.5 | 0.500 |

A second layer lowers a tile only by scoring below the first. The cost is largest, 0.172, when the second layer scores $\sqrt{2} - 1 \approx 0.414$. It vanishes both as that layer becomes as good as the first and as it fades out.

**The consequence: J is not monotone in the layers present.** Removing band $b$ from a tile raises the tile's score whenever $0 < u_{bg} <$ the tile's score. So darkening a weak layer can raise J, which a maximum over bands would rule out by construction. Section 6.8 measures how much of the grid this touches on this run's TuRBO winner; the pipeline does not compute it.

**The exchange rate this implies.** Splitting a clean, strong, single-band tile evenly between two sectors costs 0.5; a rival 6 dB down costs 0.201, and one 20 dB down 0.010. Closing a hole looks as if it should gain the full 1.000, but it cannot. A tile that has just crossed $T_{\text{cov}}$ sits near −120 dBm, where $s \approx 0$. At −119 dBm a newly covered tile is worth 0.033, and at −110 dBm, 0.333. So one strong tile split evenly between two sectors costs more than a hole closed at −110 dBm gains.

J is therefore primarily a *signal-strength and cleanliness* measure that treats hole-closing as a minor bonus. Over random search's 144 Sobol candidates, signed so that a positive value means J and the KPI improve together, J tracks cell-edge RSRP (Spearman 0.90), the weak rate, median RSRP and the hole rate (0.86 each) most closely. It tracks the UE service failure rate (0.57), the overlap rate (0.50) and overlap neighbours (0.44) moderately, median SINR (0.29) and median, mean and cell-edge throughput (0.25, 0.15 and 0.11) loosely, and cell-edge SINR (−0.02) not at all. The pipeline does not produce these correlations; they were computed for this report from random search's history (`../outputs/optim/random/2026-10-02_09-59-04/history.parquet`).


**Why not a preferred band or the best band.** Scoring the most preferred band that clears $T_{\text{cov}}$ would let a tilt raise J by dropping a crowded preferred layer below the threshold. Scoring the best band closes that, but prices nothing on a tile's other layers, so a crowded layer costs nothing wherever another layer is clean. The contraharmonic mean prices every covered layer, and it pays for that with monotonicity.

**What it does not read.** Sector load, where UEs stand, and inter-band interference. That last one is a real gap. The power share is co-band, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The mean does lower a tile for a weak or crowded second layer, but it does so whether or not that layer interferes with the first. The objective has **no free parameters**: it reads two KPI thresholds, both of which the reported KPIs already define.

**Nothing is weighted by demand.** Every tile counts equally. A hole where nobody stands costs exactly what a hole in a hotspot costs. Where the traffic stands is still reported, in Table 7b, but nothing optimises it.

**The serving rule** (`src/kpi/capacity.py`) decides which UEs a sector-band serves. Within an interval UEs connect one at a time, each to the sector-band above −120 dBm where an equal share of 0.8 of its `max_prb`, split over the UEs already there and itself, carries the most Shannon throughput. Nobody is refused. It drives the estimated-throughput KPIs and every per-sector-band table; J does not read it.

## 4. Criteria for Assessing Solutions

Criteria 1 and 2 decide effectiveness, criteria 3 and 4 decide whether the result can be trusted, and criterion 5 decides practicality.

1. **Overall quality.** The winner's J, as a change from the current configuration.
2. **Reported KPIs.** The direction of change against the current configuration, over the eight network KPIs (`NETWORK_KPIS` in `src/evaluation/compare.py`):
   - coverage hole rate ↓, weak-coverage rate ↓, co-band overlap rate ↓, overlapping neighbours per covered tile ↓
   - UE service failure rate ↓ (share of all UE rows with no sector-band above −120 dBm)
   - cell-edge, median and mean estimated UE throughput ↑ (5th and 50th percentiles and mean over the served UE rows, at each sector-band's equal PRB share once the interval's last UE has connected)

   Per band, the hole, weak and overlap rates and the cell-edge and median RSRP and SINR ↑ (5th and 50th percentiles over the band's covered tiles) are reported for every layer. Best-server RSRP and SINR are per band only: the strongest layer across bands is not one any UE is measured on, so their all-band values are recorded with every candidate but not reported.

   A change is labelled better, worse, unchanged, or undefined when the delta is not a number (`src/evaluation/compare.py`). Solver noise per KPI has not been measured, so no tie band is applied. None of the KPIs is weighted into J, so agreement between J and the KPIs is a finding, not a construction. PRB load is not reported: every sector-band with a UE uses its whole usable pool by construction ([ADR 0002](adr/0002-contraharmonic-objective-and-kpi-set.md)).
3. **Search effectiveness.** Whether the search itself earned the gain. Measured by the winner against the median candidate, sample efficiency, and TuRBO paired with random search on the same seed.
4. **Robustness.** Where the configuration moves demand, not only area, and how it trades one KPI against another. Nothing in J reads demand, so this criterion is entirely a check on the objective rather than a reflection of it.
5. **Cost.** Ray-tracing evaluations, ray-tracing minutes and wall-clock minutes per run.

## 5. Research Methodology

**Data generation.** No operator data was available. All data was produced synthetically by the generator (`notebooks/00_simulation.ipynb`, `src/scenario/`), which writes the files `simulation.input` names: the UE table, the sector table and the manifest. The radio stage (`src/simulation/`) and everything after it read only those files, so real data can replace the generator.

1. **Scenario.** The scene was rasterised onto the 20 m grid, and a population of 10 to 20 UEs was drawn every 15 minutes for 7 days.
   - Each draw mixed four elliptical Gaussian hotspots with a uniform open-ground background. The hotspots sit where surrounding building volume is high, at least 500 m apart.
   - The hotspots hold 70 % of UEs on average, modulated by a diurnal profile and AR(1) noise (`configs/scenario.yaml` `time`, `density`).
   - UEs are independent per interval, with no mobility.
2. **Radio map.** Each band was ray-traced separately at 10⁷ rays per transmitter and maximum depth 8.
   - Line of sight, specular reflection and refraction were on; diffuse reflection and diffraction were off.
   - Materials were ITU-R P.2040 and frequency-static.
   - The output was per-sector RSRP and SINR on the grid (`../reports/tables/00_simulation/propagation_parameters.csv`).
   - At 25 m masts, 2600 MHz reaches 88.1 % of tiles, 1800 MHz 87.8 % and 700 MHz 91.1 % (`../reports/tables/00_simulation/reach_per_band.csv`).
3. **Service.** Every UE row is served from the radio map at its tile, in the search and in the evaluation alike. No measurement noise, report censoring or position error is modelled.

The 2026-10-06 re-execution regenerated the scenario and the baseline radio map from scratch and re-ran both searches, after deleting the earlier runs. It reproduced the previous execution (2026-10-05): the UE table is byte-identical (SHA-256), the baseline KPIs agree to GPU round-off (largest gap 4.6 × 10⁻⁶ Mbit/s, on median throughput), and both searches found the same winners, at evaluations 107 and 119, with the same tilts and the same KPIs and J to round-off. The scenario ID changed from `scn_d7899e238887de67` to `scn_b7b162aabf88a3ee` only because the layout key `cells_per_node` was renamed `sectors_per_node` and that key is part of the ID's hash; the scenario itself is unchanged.

**Verification.** Before optimization, `notebooks/02_preprocessing.ipynb` checked the UE table, the manifest and the radio map against 25 contract checks, all of which held (`../reports/tables/02_preprocessing/verification_checks.csv`).
- The checks cover the manifest's keys, the UE table's columns, integer and numeric dtypes and missing values, band and sector order, band carriers, scenario ID, grid, UE height, the SINR shape and its no-path pattern, tile and extent bounds, each UE's tile against its coordinates, the RSRP bound, the schedule, duplicate rows and baseline tilts.
- The notebook then wrote the typed UE table without dropping or altering a row. The sector table is not copied: every stage reads it from the scene folder.
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
5. Maps coverage, overlap, the serving-band mix, sector utilisation and tilt movement.
6. Records cost and convergence.

**Relevance, criteria and practicality.**
- **Ray tracing over a statistical model.** Real city geometry was ray-traced rather than using a statistical path-loss model, because tilt changes act mainly through building shadowing and reflections, which a statistical model averages away.
- **Budgets.** Random search and TuRBO had matched budgets and a shared initial design, so criterion 3 isolates the model's contribution.
- **Seeds.** One seed per method kept the study within a single GPU session: ray tracing took 2.4 to 3.2 s per candidate (Table 11).

## 6. Analysis and Interpretation

### 6.1 Comparability, correctness and repeatability

All 23 comparability checks held (`../reports/tables/04_evaluation/comparability_checks.csv`). The KPIs recomputed from the archived radio maps (`kpi_reproducibility.csv`) match the recorded values to float round-off. Over all 24 recorded network measures, the largest absolute gap is 2.7 × 10⁻⁶ Mbit/s, on the current configuration's median throughput against the re-traced baseline, and **on every rate it is zero**. The differences discussed below therefore come from the configurations, not from bookkeeping.

**J is rounded before any search reads it.** The ray tracer's Monte-Carlo stream is seeded, but the GPU adds path contributions into a tile in a varying order, so the same tilts traced in two processes can differ in J's trailing digits. TuRBO's GP fit turns any such difference into a different proposal. `src/optim/objective.py` rounds J to 10⁻⁶, below the printed precision. Rounding makes a mismatch unlikely, not impossible.

**The tilt box changed.** These runs search [0°, 20°] per sector-band. Earlier runs under [0°, 15°] (scenario `scn_7d938e15f9ac4618`) are a different scenario ID and are not comparable; they were deleted.

### 6.2 Overall quality and reported KPIs

*Table 4. Best configuration per method against the current configuration, seed 42. Arrows show the better direction; bold marks the best value in the row. Sources: [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv), [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv).*

| | Current | Random search | TuRBO |
|---|---:|---:|---:|
| **Objective J ↑** | 0.6236 | 0.6438 | **0.6727** |
| Coverage hole rate ↓ | 0.1125 | 0.1134 *(worse)* | **0.1096** |
| Weak coverage rate ↓ | 0.3070 | 0.3005 | **0.2635** |
| Co-band overlap rate ↓ | **0.3165** | 0.3580 *(worse)* | 0.3366 *(worse)* |
| Overlap neighbours per covered tile ↓ | 0.9626 | 1.0552 *(worse)* | **0.9606** |
| UE service failure rate ↓ | 0.2124 | 0.2126 *(worse)* | **0.2101** |
| Cell-edge throughput p05 [Mbit/s] ↑ | 14.66 | 13.79 *(worse)* | **17.03** |
| Median throughput p50 [Mbit/s] ↑ | 64.12 | 63.74 *(worse)* | **76.27** |
| Mean throughput [Mbit/s] ↑ | 92.11 | 96.80 | **107.25** |
| **Network KPIs better / worse, of 8** | — | 2 / 6 | 7 / 1 |

*Table 4b. Best-server RSRP and SINR per band, the percentiles over each band's covered tiles. Source: [`band_kpis.csv`](../reports/tables/04_evaluation/band_kpis.csv).*

| Band | Measure | Current | Random search | TuRBO |
|---|---|---:|---:|---:|
| 2600 MHz | RSRP p50 / p05 [dBm] ↑ | −98.09 / −115.41 | −93.58 / −114.30 | **−92.76 / −114.24** |
| 2600 MHz | SINR p50 / p05 [dB] ↑ | 12.78 / **−0.79** | 14.53 / −1.31 | **15.87** / −0.99 |
| 1800 MHz | RSRP p50 / p05 [dBm] ↑ | −91.73 / −112.45 | −90.55 / −113.49 | **−87.74 / −111.51** |
| 1800 MHz | SINR p50 / p05 [dB] ↑ | 13.13 / **−0.80** | 12.87 / −1.17 | **14.59** / −0.98 |
| 700 MHz | RSRP p50 / p05 [dBm] ↑ | −83.92 / −108.51 | −84.12 / −109.21 | **−82.23 / −107.57** |
| 700 MHz | SINR p50 / p05 [dB] ↑ | 13.21 / −1.10 | 14.33 / −1.07 | **15.92 / −0.95** |

![KPI improvement](../reports/figures/04_evaluation/kpi_improvement.png)

*Figure 4. Relative change per network KPI and method; J is not among the panels. Source: `../reports/figures/04_evaluation/kpi_improvement.png`.*

**TuRBO wins J:** +7.87 %, against +3.24 % for random search. On the eight network KPIs, TuRBO improves seven and worsens one, the co-band overlap rate. Random search improves two, the weak rate and mean throughput, and worsens six: the hole rate, both overlap measures, the UE service failure rate and cell-edge and median throughput.

**Head to head, TuRBO is better than random search on all eight network KPIs and on every per-band RSRP and SINR percentile.** The margins that matter operationally:
- **Coverage.** Weak coverage 26.4 % against 30.1 % of the grid, and the hole rate 0.1096 against 0.1134, where random search's is above today's. TuRBO fails 2,119 of 10,087 UE reports, 23 fewer than today; random search fails 2,145, three more.
- **Signal quality.** Median SINR rises on every band under TuRBO, by 1.5 to 3.1 dB. Cell-edge SINR is the one measure it does not improve everywhere: it falls by about 0.2 dB on 2600 and 1800 MHz and rises 0.15 dB on 700 MHz, while random search lowers it on the two upper bands by 0.4 to 0.5 dB.
- **Throughput.** Mean +16.4 % against +5.1 %, median +19.0 % against −0.6 %, and the cell edge rises (+16.2 %) where random search's falls (−5.9 %).
- **Overlap.** Both worsen the collapsed rate, TuRBO by 6.4 % and random search by 13.1 %. Overlap neighbours per covered tile fall 0.2 % under TuRBO and rise 9.6 % under random search.

### 6.3 Did the search matter?

*Table 5. Winner against the candidates each run evaluated. Source: [`winner_vs_candidates.csv`](../reports/tables/04_evaluation/winner_vs_candidates.csv).*

| Method | Current | Initial design, median | All candidates, median | All candidates, 90th pct. | Best |
|---|---:|---:|---:|---:|---:|
| Random search | 0.6236 | 0.6166 | 0.6127 | 0.6310 | 0.6438 |
| TuRBO | 0.6236 | 0.6166 | **0.6682** | **0.6719** | **0.6727** |

*Table 6. Best J reached after a fixed number of evaluations. Source: [`sample_efficiency.csv`](../reports/tables/04_evaluation/sample_efficiency.csv).*

| Evaluations | Random search | TuRBO |
|---:|---:|---:|
| 10 | **0.6359** | **0.6359** |
| 25 | 0.6359 | **0.6374** |
| 50 | 0.6359 | **0.6663** |
| 100 | 0.6424 | **0.6711** |
| 145 | 0.6438 | **0.6727** |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Figure 5. Best objective found so far against evaluations. Source: `../reports/figures/04_evaluation/search_progress.png`.*

![TuRBO evaluations](../reports/figures/03b_turbo/turbo_evaluations.png)

*Figure 6. Every TuRBO evaluation, by what proposed it. Source: `../reports/figures/03b_turbo/turbo_evaluations.png`.*

Random search's Sobol candidates are mostly **worse** than the current configuration: their median is 0.6127 against 0.6236, and their 90th percentile reaches only 0.6310. With bounds of [0°, 20°] a random draw is as likely to steepen a tilt well past today's as to flatten it. For an engineer, that says no direction of change is safe by default: the gain has to be searched for.

The evidence that the model earned TuRBO's margin:
- TuRBO's **median** candidate (0.6682) scores above random search's single **best** (0.6438).
- TuRBO and random search share the same 16 Sobol points and diverge only once the model proposes. That shared design has a median of 0.6166 for both, while TuRBO's all-candidate median is 0.6682 against random search's 0.6127.
- TuRBO's 128 trust-region proposals average 0.6639, against 0.6112 for its 16 Sobol points (`../reports/tables/03b_turbo/turbo_evaluations_by_proposer.csv`).

Per method:
- **Random search** reached 0.6359 at evaluation 7, then 0.6389 at 56, 0.6424 at 95 and its best, 0.6438, at evaluation 107.
- **TuRBO** passed random search's final best at evaluation 34, and found its best at **evaluation 119 of 145**, after improvements at 104, 107, 113 and 116. Its trust region never collapsed into a restart.

The paired gain of TuRBO over random search is **+0.0289** on the one seed (`paired_gain_turbo_vs_random.csv`). With one pair, no confidence interval or Wilcoxon test can be computed, so the margin cannot be separated from seed-to-seed variation.


![Hole vs overlap trade-off](../reports/figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png)

*Figure 7. Every evaluated configuration on hole rate against overlap rate, with each method's pick and the Pareto front. Source: `../reports/figures/04_evaluation/tradeoff_hole_rate_vs_overlap_rate.png`.*

### 6.4 Is the result robust?

**Both methods lowered the hole rate, but not because J asked.** A tile that has just crossed the hole threshold sits near −120 dBm, where the strength factor is near zero, so closing holes buys J almost nothing (Section 3.3). The hole-rate gains in Table 4 are a side effect of uptilting, which widens every footprint.

**Overlap-reducing configurations were available to TuRBO only.** 18 of TuRBO's candidates beat the incumbent's overlap rate of 0.3165, the lowest at **0.2992** (evaluation 69, J 0.6655; `sample_efficiency.csv`). Random search never beat it in 145 evaluations. J did not pick the lowest-overlap candidate, which is consistent with J tracking the overlap rate only moderately (Section 3.3). TuRBO's published shortlist (`../reports/outputs/solutions_turbo.csv`) holds two runners-up within 0.0002 of the winner's J: one at an overlap rate of 0.3350 (J 0.672658), below the winner's 0.3366, and one at 0.3419. An engineer who prefers a lower overlap rate can take the first runner-up at no measurable cost in J.

**Per band, TuRBO lowers overlap on every layer, yet the collapsed rate rises.** The band-collapsed KPI counts a tile if *any* of the three layers is crowded there. Split by band (`../reports/tables/04_evaluation/band_kpis.csv`):

*Table 7a. Share of tiles with at least one overlapping co-band neighbour, per band.*

| Band | Current | Random search | TuRBO |
|---|---:|---:|---:|
| 2600 MHz | 0.2010 | 0.1999 | **0.1692** |
| 1800 MHz | 0.2069 | 0.2111 | **0.1980** |
| 700 MHz | 0.2394 | 0.2341 | **0.2167** |
| Band-collapsed (the reported KPI) | **0.3165** | 0.3580 | 0.3366 |

TuRBO has the lower overlap on each band, cutting 2600 MHz and 700 MHz most. The collapsed rate still rises, because what crowding remains lands on fewer shared tiles: the three layers' crowded areas coincide less than before. Random search lowers 2600 and 700 MHz slightly and raises 1800 MHz, yet raises the collapsed rate most. J prices each band's crowding separately, so it sees TuRBO's per-band gains and not the union the collapsed KPI counts.

*Table 7b. Coverage class by area and by demand, with demand counted in UE reports. Nothing in J reads this view, so it is a check on the result, not a reflection of it. Source: [`coverage_by_area_and_demand.csv`](../reports/tables/04_evaluation/coverage_by_area_and_demand.csv).*

| Class | Current area / demand | Random area / demand | TuRBO area / demand |
|---|---|---|---|
| Hole | 11.3 % / 21.2 % | 11.3 % / 21.3 % | 11.0 % / 21.0 % |
| Weak | 30.7 % / 23.7 % | 30.1 % / 26.2 % | 26.4 % / 21.2 % |
| Good | 58.1 % / 55.1 % | 58.6 % / 52.6 % | 62.7 % / 57.7 % |

*Table 8. Overlapping co-band neighbours per configuration. Source: [`overlap_neighbour_summary.csv`](../reports/tables/04_evaluation/overlap_neighbour_summary.csv).*

| Configuration | Mean neighbours, covered tiles | Share with 0 | Share with 3+ |
|---|---:|---:|---:|
| Current | 0.96 | **64.3 %** | 17.6 % |
| Random search | 1.06 | 59.6 % | 16.2 % |
| TuRBO | **0.96** | 62.2 % | **14.5 %** |

![Coverage before and after, 2600 MHz](../reports/figures/04_evaluation/coverage_before_after_b2600.png)

*Figure 8. 2600 MHz best-server RSRP before and after TuRBO, and the tiles that crossed the hole threshold; `coverage_before_after_b1800.png` and `coverage_before_after_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/coverage_before_after_b2600.png`.*

![RSRP change maps, 2600 MHz](../reports/figures/04_evaluation/rsrp_change_maps_b2600.png)

*Figure 9. Change in 2600 MHz best-server RSRP for each method's best configuration; `rsrp_change_maps_b1800.png` and `rsrp_change_maps_b700.png` show the other bands. Source: `../reports/figures/04_evaluation/rsrp_change_maps_b2600.png`.*

**TuRBO's map change is larger and closes more holes than it opens.** Over all bands, it raises best-server RSRP on 87.5 % of the tiles reached in both maps, by a median of 2.9 dB, closing 625 hole tiles and opening 332. Random search raises 67.0 % by a median of 1.4 dB, but closes 374 and opens 463, which is why its hole rate rises. The holes TuRBO opens sit where it pulls a carrier in (Section 6.6); they are the tiles a drive test after rollout should cover first. These all-band figures were computed for this report from the archived radio maps; the pipeline draws the per-band maps only.

**Demand moves out of weak coverage under TuRBO only.** The share of UE reports on weak tiles falls from 23.7 % to 21.2 % under TuRBO and rises to 26.2 % under random search; the share on good tiles rises from 55.1 % to 57.7 % under TuRBO and falls to 52.6 % under random search. The share on hole tiles barely moves, 21.2 % to 21.0 % and 21.3 %, because most of it is one hotspot no tilt reaches.

**TuRBO thins the pile-ups and holds the mean.** TuRBO cuts the share of covered tiles with three or more overlapping neighbours from 17.6 % to 14.5 %. The share with none falls from 64.3 % to 62.2 %, and the mean stays at 0.96. The power share prices a pile-up far more than a single rival, which is what TuRBO trades on.

**No objective parameters to vary, and no band priority either.** The objective has no parameters and reads no band order. What remains untested is the single search seed, which Section 6.8 lists.

### 6.5 Capacity impact

*Table 9. UE service and median served SINR per band. Throughput statistics are in Table 4. Sources: [`ue_service_summary.csv`](../reports/tables/04_evaluation/ue_service_summary.csv), [`band_layer_summary.csv`](../reports/tables/04_evaluation/band_layer_summary.csv).*

| Configuration | Not served | On 2600 / 1800 / 700 MHz | Median served SINR, 2600 / 1800 / 700 MHz [dB] |
|---|---:|---|---|
| Current | 21.2 % | 52.9 % / 14.2 % / 11.6 % | 9.96 / 16.03 / 20.77 |
| Random search | 21.3 % | 49.0 % / 19.0 % / 10.7 % | 11.66 / 13.46 / 20.48 |
| TuRBO | **21.0 %** | 47.5 % / 19.3 % / 12.2 % | **15.46 / 19.82 / 23.95** |

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Figure 10. Serving-band mix per configuration. Source: `../reports/figures/04_evaluation/serving_band_mix.png`.*

![Sector-band throughput](../reports/figures/04_evaluation/sector_band_throughput.png)

*Figure 11. Median estimated throughput per sector-band, current and recommended. Source: `../reports/figures/04_evaluation/sector_band_throughput.png`.*

**Service.** TuRBO fails 2,119 of 10,087 UE reports against 2,142 today; random search fails 2,145. Every failure stands on a hole tile, since the serving rule refuses nobody. Median served SINR rises on every band under TuRBO, by 3.2 to 5.5 dB; under random search it rises on 2600 MHz only.

**Both methods move traffic off the capacity layer and onto 1800 MHz.** 2600 MHz falls from 52.9 % of UE reports to 47.5 % under TuRBO and 49.0 % under random search, and 1800 MHz rises from 14.2 % to 19.3 % and 19.0 %. Under TuRBO that spread comes with higher SINR on every layer, which is where its throughput gain comes from; under random search it does not. It shows in the sector-impact table (`../reports/tables/04_evaluation/sector_impact.csv`):
- **Uptilted carriers pick up load:** n2s1's 2600 MHz carrier, uptilted 11.82°, goes from 717 to 1,175 served reports with 8.3 dB more median SINR and 9.7 Mbit/s more median throughput; n3s2's 1800 MHz carrier, uptilted 8.72°, from 136 to 525, with 11.9 dB more median SINR.
- **Downtilted carriers shed it:** n3s2's 2600 MHz carrier, downtilted 4.67°, drops from 793 reports to 46, and n1s0's 2600 MHz carrier, downtilted 4.03°, from 512 to 118.

**Per band** (`../reports/tables/04_evaluation/band_layer_summary.csv`), under TuRBO:
- **Area covered** rises on 2600 MHz, 71.2 % → 72.6 %, and 1800 MHz, 76.5 % → 77.6 %, and holds on 700 MHz at 86.1 %. The low band keeps its coverage-floor role.
- **Mean RSRP where covered** improves on every layer: 2600 MHz −97.2 → −92.6 dBm, 1800 MHz −91.5 → −88.1 dBm, 700 MHz −84.2 → −82.6 dBm.
- **Median served SINR** rises on every layer: 2600 MHz 10.0 → 15.5 dB, 1800 MHz 16.0 → 19.8 dB and 700 MHz 20.8 → 23.9 dB.
- **Per-sector-band median throughput**, the median over a band's twelve carriers, rises on every layer: 2600 MHz 82.1 → 86.1 Mbit/s, 1800 MHz 69.8 → 84.5 and 700 MHz 46.6 → 52.2. The slowest carrier is now n0s2's 700 MHz at 24.5 Mbit/s, where today's slowest, n2s1's 700 MHz, runs at 20.6 Mbit/s (`sector_band_load.csv`).

About 21 % of UE reports remain unserved, all of them on hole tiles, and 17.5 % of UE positions have no path to any sector (Table 2). Tilt alone cannot serve them; that is a site or coverage decision.

### 6.6 Recommended tilt changes

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Figure 12. Tilt change per sector and band in the highest-J (TuRBO) configuration. Source: `../reports/figures/04_evaluation/tilt_delta_heatmap.png`.*

*Table 10. Tilt movement for the highest-J configuration. Negative Δ is an uptilt. Source: [`tilt_movement_summary.csv`](../reports/tables/04_evaluation/tilt_movement_summary.csv).*

| Band | Sectors moved | Mean \|Δ\| [°] | Largest \|Δ\| [°] | Mean Δ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 / 12 | 8.34 | 11.82 | −4.75 |
| 1800 MHz | 12 / 12 | 8.38 | 9.85 | −5.83 |
| 700 MHz | 12 / 12 | 7.30 | 11.94 | −1.94 |

Every one of the 36 sector-bands moved. The configuration is a net uptilt on every band, and the structure is the point:
- **Most carriers uptilt hard.** 27 of 36 are uptilted, 1800 MHz most. That widening is what lifts the weak rate and the RSRP percentiles.
- **Nine of the 36 are downtilted, on six sectors.**

  | Sector | Downtilted carriers | Uptilted hard on the same sector |
  |---|---|---|
  | n0s2 | 2600 MHz +5.26°, 700 MHz +11.94° | 1800 MHz −9.08° |
  | n1s0 | 2600 MHz +4.03°, 700 MHz +8.82° | 1800 MHz −5.65° |
  | n1s1 | 2600 MHz +7.60° | 1800 MHz −9.04° |
  | n2s1 | 1800 MHz +5.83°, 700 MHz +11.36° | 2600 MHz −11.82° |
  | n3s1 | 1800 MHz +9.49° | 2600 MHz −9.70° |
  | n3s2 | 2600 MHz +4.67° | 1800 MHz −8.72° |

  On each of those sectors the search hands the near area to one layer and the reach to another, which is a layer-role decision a per-band procedure does not make. It is consistent with TuRBO having the lower overlap on every band (Table 7a) and with the load shift of Section 6.5.
- **The box binds at both ends.** Proposed tilts span **0.15° to 19.94°**: n2s0's 1800 MHz carrier sits 0.15° from the 0° bound and n0s2's 700 MHz carrier 0.06° from the 20° bound. The largest single change is 11.94°, n0s2 on 700 MHz.

With one seed, it is not established which of TuRBO's per-sector differences matter and which reflect where the trust region happened to be when the budget ended.

The largest traffic shifts under TuRBO (`../reports/tables/04_evaluation/sector_impact.csv`):
- n3s2 on 2600 MHz, −747 served reports;
- n2s1 on 2600 MHz, +458;
- n1s0 on 2600 MHz, −394;
- n3s2 on 1800 MHz, +389;
- n1s1 on 2600 MHz, −246.

Those are the sectors to watch after a rollout.

### 6.7 Cost

*Table 11. Search cost. Sources: [`method_cost.csv`](../reports/tables/04_evaluation/method_cost.csv), [`kpi_scoreboard.csv`](../reports/tables/04_evaluation/kpi_scoreboard.csv).*

| Method | Evaluations | Best found at | Ray tracing [min] | Wall clock [min] | Ray tracing per evaluation [s] | J gain | J gain per wall-clock minute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random search | 145 | 107 | 7.79 | 9.53 | 3.2 | +0.0202 | 0.0021 |
| TuRBO | 145 | 119 | 5.77 | 12.54 | 2.4 | **+0.0491** | **0.0039** |

TuRBO's GP fitting and acquisition, which run on the CPU, added about 6.8 minutes of wall clock on top of its 5.8 minutes of ray tracing. Timings are only indicative: both runs traced the same scene on the same RTX 3050 Ti laptop GPU, yet random search's evaluations took 3.2 s against TuRBO's 2.4 s, so GPU state between runs varies more than the methods do.

**TuRBO is the more cost-effective of the two by J per minute**, and it overtakes random search's final result within its first 34 evaluations.

### 6.8 Limitations

These results should be read tentatively, for nine reasons:

1. **One scenario.** Every configuration was tuned and scored on the same city, layout and UE population, so nothing here measures generalisation.
2. **One search seed per method.** No confidence interval or significance test could be computed. The TuRBO–random paired gain is +0.0289, resting on one pair.
3. **Winner's curse, and TuRBO's sensitivity to J.**
   - Every candidate used the same ray-tracer seed, so the maximum of many candidates may favour configurations that benefit from that seed's Monte-Carlo noise. Solver noise is unmeasured.
   - The GPU ray tracer is reproducible only to round-off, and the GP fit can amplify any difference in J into different proposals, so a rerun of seed 42 need not find the same winner. An earlier rerun under the previous bounds did not; the reruns of 2026-10-05 and 2026-10-06 under the current bounds did, at the same evaluation (119) and the same J to six digits.
   - TuRBO's two runners-up sit within 0.0002 of the winner's J with overlap rates from 0.3350 to 0.3419, so any per-KPI claim about TuRBO's winner beyond J is one draw.
4. **J is not monotone in the layers present.** A tile's score rises when it loses a covered band scoring below the tile's own score (Section 3.3).
   - On this run's TuRBO winner, 77.2 % of tiles have such a band. If each could keep only its best band, the ceiling on the gain would be 0.063, against the winner's whole improvement of 0.049.
   - The power share makes this common: almost any covered second layer with a co-band rival scores below 1, and so below a clean first layer.
   - That ceiling is not reachable, because darkening a band on one tile changes it on many.
   - Both numbers come from a one-off analysis of the archived radio map, not from the pipeline.
   - Nothing in the objective stops a search from buying J by switching a weak layer off.
5. **Inter-band interference is priced nowhere.** The power share is co-band by definition, the reported overlap rate merely sums the three per-band counts, and J does not read SINR. The contraharmonic mean does lower a tile for a weak second layer, but it does so whether or not that layer interferes. Fixing this needs an interference model, not a reweighting.
6. **Nothing is weighted by demand.** A hole where nobody stands costs exactly what a hole in a hotspot costs. Table 7b is the only place demand appears.
7. **The capacity model is a simplification.** It drives the estimated throughput and every per-sector-band figure. It uses a placeholder usable PRB share of 0.8, an equal share with no scheduler, a Shannon rate with no MCS cap, full-load co-band interference, and per-RE thermal noise with no receiver noise figure.
8. **Tilts at both bounds, and no movement penalty.** The winner places one sector-band at 0.15° and another at 19.94°, and moves all 36 antennas, by up to 11.94°. A real RET range or a change-management policy may not allow either.

9. **No held-out validation.** Nothing re-solves an optimized tilt on an unseen scenario, so no number here measures transfer.

Running several search seeds, re-tracing the shortlisted configurations under other solver seeds, and evaluating on held-out scenarios would address limitations 1–3.

**Comparability.** No J in this report is comparable with a value scored under any earlier objective. `src/evaluation/runs.py` refuses to pool runs recorded under a different KPI set, but there is no version guard on the objective's functional form, so a change to J alone would not be detected.

## 7. Conclusions and Recommendations

*Table 12. Summary against the assessment criteria (Section 4).*

| Criterion | Random search | TuRBO |
|---|---|---|
| 1. Objective J | +0.0202 | **+0.0491** |
| 2. Reported KPIs | 2 of 8 network KPIs better, 6 worse; median SINR down on two bands | **7 of 8 better, only the overlap rate worse; better than random search on all 8 and on every per-band RSRP and SINR percentile** |
| 3. Search effectiveness | Median candidate below the current configuration; best at 107 of 145 | **Median candidate above random search's best; best at 119 of 145** |
| 4. Robustness | Largest rise in the overlap rate; more holes opened than closed; more demand on weak tiles | Lower overlap on every band, but the collapsed rate still rises; demand moves onto good tiles; one draw of a GP-driven search |
| 5. Cost | 12.9 min, 145 evaluations | 11.1 min, 145 evaluations; **more J per minute** |

**Conclusions.**

- **TuRBO reached the highest J, and there is good evidence the model earned it.** Its median candidate (0.6682) scored above random search's best (0.6438). It also shares its first 16 Sobol points with random search and diverges only once the model starts proposing.
- **At equal budget, TuRBO is better than random search on every reported KPI**, network and per band, and it is the only method that raises cell-edge and median throughput and lowers the UE service failure rate.
- **Unstructured search is not safe on this box.** With bounds of [0°, 20°] most random candidates score below today's configuration, and random search's best still worsens six of the eight network KPIs.
- **Neither method lowers co-band overlap overall.** TuRBO has the lower overlap on each band, fewer 3+ pile-ups and an unchanged mean neighbour count, but the band-collapsed rate still rises.
- **The objective barely pays for closing holes.** A newly covered tile arrives near −120 dBm, where the strength factor is near zero. TuRBO's hole-rate gain is a side effect of widening footprints.
- **The winning pattern is a layer-role decision.** TuRBO widens most carriers and, on six sectors, pulls one or two layers in while another on the same sector widens, moving traffic from 2600 MHz onto 1800 MHz with higher SINR on every layer. That is where its throughput gain comes from.
- **Every result** depends on:
  - one scenario and one seed, and a TuRBO run that need not reproduce bit for bit;
  - a simplified capacity model;
  - an objective that is not monotone in the layers present and does not price inter-band interference at all.

**What this means for a RAN engineer today.**
- Treat the output as a **ranked set of measured options**, not an instruction. The shortlist and the sector-impact table are the useful artefacts: they say what each option buys and where traffic will move.
- **The direction is clearer than the exact values.** TuRBO's net uptilt on every band says the committed tilts are too steep for this layout, and its per-sector pattern (one layer near, one far) is the hypothesis to test first in the field, on one sector, before any network-wide change. Random search shows that an arbitrary change is more likely to hurt than help.
- **Phase the change.** No movement penalty is applied, so the raw recommendation moves every antenna. Start from the sectors with the largest traffic shifts (Section 6.6) and validate with a drive test around the tiles where holes open.

**Recommendations.**

1. **Do not deploy any recommended tilt set yet.** No result has been validated beyond the scenario it was tuned on.
2. **Repeat the comparison over several search seeds** (`BAND_TILT_SEEDS` in notebooks 03a/03b), and re-trace the shortlists under other solver seeds. The TuRBO–random margin of +0.0289 needs an interval before it can be called decisive, and an earlier rerun showed TuRBO's winner can move between runs of the same seed.
3. **Decide whether the objective's non-monotonicity is acceptable.** This is the most consequential open question left, and the power share makes it reach most of the grid. Measure it as part of the pipeline rather than ad hoc, and re-trace the shortlisted configurations with the weakest layer at each tile removed, to measure whether any reachable tilt actually collects it.
4. **Give TuRBO a larger budget.** It was still improving at evaluation 119 of 145 and never restarted, so the trust region was still productive late in the budget.
5. **Add a movement constraint or penalty**, and confirm the tilt box against the real RET range. The winner sits within 0.15° of both bounds and moves all 36 antennas.
6. **Model inter-band interference.** The overlap count and the power share are co-band and J does not read SINR, so nothing in the study prices a strong neighbour on another layer. This needs a model, not a reweighting.
7. **Address the out-of-reach demand.** Every UE still unserved stands on a hole tile, largely one hotspot with no propagation path. Tilt alone cannot lower the UE service failure rate much further. That is a coverage decision, not a tilt one.
8. **Build held-out scenario validation** before drawing a method-level conclusion.

---


## Appendices

### Appendix A. Configuration and reproduction

The runs used the committed configuration in `configs/`:

| Setting | Value |
|---|---|
| Scene | `data/scenes/hanoi/scene.xml` (not in Git), with the sector table `sectors.csv`, the manifest `scenario.json` and the generator record `synthetic.json` beside it |
| Layout | 4 nodes, 1,732 m triangle plus centroid (1,000 m centre to corner), 3 sectors at 45° / 165° / 285°, 25 m masts |
| Tilt | Current 12° (2600 MHz), 10° (1800 MHz), 8° (700 MHz); bounds [0°, 20°], every sector-band |
| KPI thresholds | `hole_dbm` −120, `weak_dbm` −90, `overlap_margin_db` 6; edge percentile 5 (`LOW_PERCENTILE` in `src/kpi/quality.py`, a code constant) |
| Objective | co-band power share × strength per band, contraharmonic mean over bands; no parameters; reads `hole_dbm` and `weak_dbm`; rounded to 10⁻⁶ (ADR 0002) |
| Capacity | max-throughput sector selection over an equal share of 0.8 × `max_prb`, candidates above −120 dBm, SCS 15 kHz, connection in report-time order |
| Noise | k·T·SCS per resource element at 298.15 K (`simulation.radio_map.bands[].scs_hz`), no receiver noise figure |
| Search | seed 42 (`optim.seed`); TuRBO's restart and proposal seeds derived from it by `numpy.random.SeedSequence`; random and TuRBO 16 + 128; TuRBO batch 3, trust region 0.8 / 0.5⁷ / 1.6, success tolerance 3, failure tolerance 12, improvement 10⁻³, perturbed dimensions 5; 4 solutions published |

Runs used in this report:

| Method | Run directory |
|---|---|
| Random search | `../outputs/optim/random/2026-10-06_03-48-41/` |
| TuRBO | `../outputs/optim/turbo/2026-10-06_03-58-44/` |

To reproduce, run notebooks `00` through `04` in order, or `task pipeline`; both call the same functions in `src/`. Notebooks 03a and 03b skip any method and seed that already has a run under `optim.output.dir` (`outputs/optim/`), so clear that directory first to re-search.

Each run's `run.json` records the resolved configuration, the scenario ID and a `provenance` block: the Git commit, whether the working tree was dirty, and the numpy, scipy, torch, botorch, gpytorch and sionna-rt versions.

Every path a stage writes is configurable, so a trial run can be kept apart from the real one. The notebooks read extra Hydra overrides from `BAND_TILT_OVERRIDES`, and `reports.figures_dir` and `reports.tables_dir` (`configs/config.yaml`) set where their tables and figures go. The small-budget check of this pipeline (2026-10-03) used the settings below, with every output under `outputs/smoke/` and each notebook executed to a copy (`jupyter nbconvert --to notebook --execute notebooks/<nb>.ipynb --output-dir outputs/smoke/notebooks`, 00 through 04 in order), then `python -m src.evaluation.run` with the same overrides:

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

| Stage | Tables (`../reports/tables/…`) | Figures (`../reports/figures/…`) |
|---|---|---|
| 00 simulation | `study_area`, `network_configuration`, `node_layout`, `frequency_bands`, `ue_distribution`, `propagation_parameters`, `reach_per_band`, `ue_measurement_summary`, `serving_band_mix`, `decision_variables`, `baseline_kpis`, `coverage_by_area_and_demand` | `study_area`, `traffic_model`, `rsrp_per_band`, `ue_rsrp_distribution`, `serving_band_map`, `coverage_and_overlap_maps` |
| 01 EDA | `dataset_overview`, `ue_schema`, `missing_values`, `duplicates`, `schema_checks`, `physical_checks`, `band_representation`, `tilt_summary`, `rsrp_statistics`, `coverage_classes_per_band`, `serving_area_per_band`, `serving_band_mix`, `hole_summary`, `weak_by_band`, `overlap_per_band`, `overlap_neighbour_summary`, `cross_band_correlation`, `band_complementarity`, `hotspots`, `coverage_by_area_and_demand`, `signal_vs_ue_density`, `sector_band_configuration`, `kpi_summary`, `rsrp_outliers` | `rsrp_distribution`, `coverage_per_band`, `band_propagation`, `serving_maps`, `coverage_class_map`, `overlap_neighbours`, `cross_band_scatter`, `band_complementarity`, `ue_distribution`, `demand_vs_coverage`, `signal_vs_ue_density`, `sector_band_throughput`, `sinr_distribution` |
| 02 preprocessing | `ue_overview`, `verification_checks`, `no_path_by_band`, `coverage_classes`, `overlap_neighbours`, `ue_weighted_indicators`, `baseline_kpis`, `optimizer_features`, `objective_decomposition`, `data_quality_summary` | `network_layout`, `rsrp_map`, `overlap_map`, `coverage_map`, `effective_coverage`, `serving_multiplicity` |
| 03a baseline | `setup_network`, `setup_simulation`, `setup_users`, `baseline_configuration`, `initial_state`, `objective_parameters`, `best_tilt_random`, `tilt_movement_random`, `kpi_comparison_random`, `baseline_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `kpi_progress`, `tilt_movement_random`, `coverage_before_after_random`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 03b TuRBO | `turbo_configuration`, `turbo_evaluations_by_proposer`, `best_tilt_turbo`, `tilt_movement_turbo`, `kpi_comparison_turbo`, `method_results`, `coverage_by_area_and_demand`, `overlap`, `ue_service_summary`, `band_kpis` | `search_progress`, `turbo_evaluations`, `tilt_movement_turbo`, `coverage_before_after_turbo`, `rsrp_change_maps`, `serving_band_mix`, `band_kpis` |
| 04 evaluation | `comparability_checks`, `experiment_setup`, `kpi_scoreboard`, `kpi_relative_improvement`, `winner_vs_candidates`, `paired_gain_turbo_vs_random`, `candidates`, `kpi_reproducibility`, `coverage_by_area_and_demand`, `overlap_neighbour_summary`, `band_layer_summary`, `band_kpis`, `ue_service_summary`, `sector_band_load`, `sector_impact`, `recommended_tilt`, `tilt_movement_summary`, `method_cost`, `convergence`, `sample_efficiency` | `kpi_improvement`, `tradeoff_hole_rate_vs_overlap_rate`, `tradeoff_hole_rate_vs_ue_service_failure_rate`, `tradeoff_overlap_rate_vs_ue_service_failure_rate`, `rsrp_change_maps_<band>`, `coverage_before_after_<band>`, `coverage_class_maps`, `overlap_neighbour_maps`, `ue_throughput_maps`, `band_kpi_panels`, `serving_band_mix`, `sector_band_throughput`, `tilt_movement`, `tilt_delta_heatmap`, `search_progress` |

Deliverables per method are in `../reports/outputs/`: `solutions_<method>.csv` (the shortlist with every measure and its delta), `tilt_options_<method>.csv`, and `tilt_change_<method>.csv` (the recommended row).

### Appendix C. Highest-J tilt configuration (TuRBO)

*Source: [`recommended_tilt.csv`](../reports/tables/04_evaluation/recommended_tilt.csv); machine-readable form: [`tilt_change_turbo.csv`](../reports/outputs/tilt_change_turbo.csv). Current tilt is 12° on 2600 MHz, 10° on 1800 MHz and 8° on 700 MHz for every sector; bounds are [0°, 20°]. Negative Δ is an uptilt.*

| Sector | 2600 MHz [°] (Δ) | 1800 MHz [°] (Δ) | 700 MHz [°] (Δ) |
|---|---|---|---|
| n0s0 | 2.25 (-9.75) | 0.35 (-9.65) | 0.66 (-7.34) |
| n0s1 | 0.53 (-11.47) | 0.39 (-9.61) | 2.09 (-5.91) |
| n0s2 | 17.26 (+5.26) | 0.92 (-9.08) | 19.94 (+11.94) |
| n1s0 | 16.03 (+4.03) | 4.35 (-5.65) | 16.82 (+8.82) |
| n1s1 | 19.60 (+7.60) | 0.96 (-9.04) | 2.66 (-5.34) |
| n1s2 | 3.77 (-8.23) | 2.10 (-7.90) | 2.46 (-5.54) |
| n2s0 | 0.56 (-11.44) | 0.15 (-9.85) | 1.68 (-6.32) |
| n2s1 | 0.18 (-11.82) | 15.83 (+5.83) | 19.36 (+11.36) |
| n2s2 | 1.00 (-11.00) | 0.67 (-9.33) | 0.34 (-7.66) |
| n3s0 | 6.85 (-5.15) | 3.58 (-6.42) | 4.23 (-3.77) |
| n3s1 | 2.30 (-9.70) | 19.49 (+9.49) | 1.20 (-6.80) |
| n3s2 | 16.67 (+4.67) | 1.28 (-8.72) | 1.22 (-6.78) |

Nine of the 36 sector-bands are downtilted: four on 2600 MHz, two on 1800 MHz and three on 700 MHz, on sectors n0s2, n1s0, n1s1, n2s1, n3s1 and n3s2.


---

## References

[1] NVIDIA, *Sionna RT: Ray tracing for radio propagation modeling*. Available: https://nvlabs.github.io/sionna/

[2] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[3] *BoTorch: Bayesian optimization in PyTorch*, with GPyTorch. Available: https://botorch.org/

[4] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[5] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.4.1.
