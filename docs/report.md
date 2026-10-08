# Joint Multi-Band Antenna Tilt Optimization with High-Dimensional Multi-Objective Bayesian Optimization over a Ray-Traced Digital Twin

---

## Abstract

Multi-band base stations transmit several frequency bands from the same mast, yet their antenna tilts are often tuned one band and one sector at a time, without accounting for how the bands interact. This paper optimizes all tilts jointly for coverage, interference and user throughput, scoring each candidate on a ray-traced digital twin and selecting candidates with multi-objective Bayesian optimization. In a single-seed comparison with random search at the same cost, the method finds better trade-offs with fewer evaluations and recommends a configuration that reduces weak coverage and raises median signal quality and throughput, at a small cost in cell overlap. In the studied layout, tilt has little effect on the coverage-hole rate. The evaluation is limited to one simulated scenario.

**Index Terms** — 5G, 6G, radio access network, antenna tilt optimization, coverage and capacity optimization, self-organizing networks, multi-band networks, Bayesian optimization, multi-objective optimization, ray tracing, digital twin.

---

## I. Introduction

### A. Motivation

**What users experience.** For a mobile user, network quality is decided at the margins. A call drops when a commuter crosses from one cell into the next, a video stalls at the far end of a street, and a phone shows a strong signal yet delivers little data because several cells are competing for it. Users perceive none of the network's architecture, only that service is unreliable in particular places, and they may react by complaining, by abandoning the application or by changing provider. One of the main levers an operator has over these effects is the downward angle, or *tilt*, of the base-station antennas. Tilted too high, a cell spills into its neighbours and creates interference; tilted too low, it leaves gaps at its edge.

**Why the problem has become harder.** Modern 5G sites, and the 6G sites that will follow, no longer transmit on a single frequency. A mast commonly carries several bands with complementary roles. Low bands travel far and penetrate buildings, forming a *coverage layer*; high bands carry much more data over a shorter range, forming a *capacity layer*; mid bands bridge the two. Every band on every sector has its own tilt, so the number of settings grows with each band an operator adds. These settings are also coupled. Raising the tilt of a capacity band widens its footprint into neighbouring cells and increases interference there. Lowering it pulls the cell edge inward and may leave users relying on a slower band. Because the serving band is selected according to signal quality and load, changing one band's tilt moves traffic onto other bands, changing the performance of carriers that nobody touched.

**What engineers face.** In practice, tilts are set once at planning time, often to the same value everywhere, and then adjusted reactively. A drive test, a performance alarm or a customer complaint points to a problem sector; an engineer changes the tilt of one band on that sector remotely; and the team then typically waits days for performance counters to show whether the change helped, and whether it quietly degraded a neighbouring cell. Each step is reasonable on its own, but the loop is slow, costly in field effort and blind to cross-band side effects until they appear in the statistics. Engineers are left addressing individual complaints rather than optimizing the network as a whole, and the network-wide configuration becomes the accumulation of local decisions rather than one chosen jointly.

**What is missing.** Engineers need a way to evaluate the *whole* network's tilt configuration, across every band and sector at once, before touching any antenna; to see the trade-offs between coverage, interference and user throughput explicitly rather than discovering them afterwards; and to obtain a small set of concrete, reviewable options rather than a single opaque answer. Such a tool must be economical in the number of evaluations, since each site-specific evaluation of a city-scale network is far more costly than a closed-form model, and transparent, so that an engineer can see where each option gains, where it loses and which antennas must change.

This paper addresses that need. It combines a ray-traced model of a city, which estimates how every candidate configuration would affect coverage, interference and user throughput, with a sample-efficient multi-objective optimizer that learns which configurations are worth testing. The outcome is a shortlist of simulated options, each accompanied by its estimated effect on every performance indicator, the antenna changes it requires and the cells to monitor after rollout, produced in minutes of computation before any antenna is changed. Field validation of the chosen option remains necessary.

### B. Approach and Contributions

This paper treats the tilts of all (sector, band) pairs as a single coordinated optimization problem and evaluates it entirely in simulation. A ray tracer applied to explicit city geometry serves as a digital twin that scores each candidate configuration on coverage, co-band separation and the throughput the simulated UE population would obtain. The contributions are as follows:

1. **Problem formulation.** Joint multi-band tilt configuration is cast as a 36-dimensional, three-objective black-box problem with a co-band separation objective and a proportional-fair throughput utility that counts unserved UEs at zero rate (Section III).
2. **Sample-efficient search.** MORBO [6], a trust-region multi-objective Bayesian optimization method designed for high-dimensional spaces, is applied to the problem and compared with Sobol random search under a matched budget, a shared initial design and a common seed, so that the difference reflects the search strategy rather than the starting points (Section IV).
3. **Operational evaluation.** The recommended configurations are assessed not only on the searched objectives but on eleven KPIs, per frequency layer, by area and by demand, and in terms of inter-layer load redistribution and per-sector impact. Apart from the hole rate, which equals one minus the coverage objective, none of these is optimized directly (Sections V–VI).
4. **Empirical findings.** In the studied SMa-like layout, the coverage-hole rate is nearly insensitive to tilt; joint tilt optimization instead yields gains in weak coverage, median SINR, throughput and the distribution of load across layers (Section VI).

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Section IV describes the search methods. Section V details the experimental methodology and assessment criteria. Section VI reports and interprets the results. Section VII discusses implications and limitations, and Section VIII concludes.

## II. Related Work

**Coverage and capacity optimization.** Antenna tilt is a principal control in coverage and capacity optimization (CCO), one of the self-organizing network (SON) use cases identified by 3GPP [1]. Early automated approaches adjusted tilt per cell with rule-based, fuzzy or reinforcement-learning controllers driven by local measurements [2], [3]. Many such controllers act on one cell, or one carrier, at a time, and therefore share the locality of manual tuning.

**Black-box optimization of RAN parameters.** Because network KPIs are expensive to evaluate and non-differentiable with respect to configuration, tilt and power settings have been optimized with Bayesian optimization and reinforcement learning over simulators [4]. Standard Gaussian-process Bayesian optimization degrades in high dimensions; trust-region methods such as TuRBO [5] restore sample efficiency by restricting the search to adaptively sized local regions. MORBO [6] extends this idea to multiple objectives, maintaining trust regions centred on points of large hypervolume contribution and selecting batches by Thompson-sampled hypervolume improvement [7], [15].

**Ray-traced digital twins.** Statistical path-loss models represent building shadowing and multipath only in distribution, whereas the effect of a tilt change at a given location depends on site-specific blockage and reflections. GPU-accelerated ray tracers such as Sionna RT [10] make site-specific evaluation of a full network configuration fast enough — about two seconds per evaluation in this study — for optimization loops of tens to hundreds of full-network evaluations.

**Positioning of this work.** The present study differs from per-cell controllers in optimizing every carrier of every sector jointly, and from single-objective formulations in exposing the trade-off between coverage, co-band separation and throughput as a Pareto front from which an operator selects. It further evaluates the outcome on layer-level and demand-level KPIs that, apart from the hole rate, are not optimized directly, so that agreement between objectives and those KPIs is a finding rather than a construction.

## III. System Model and Problem Formulation

### A. Network Layout

The study area is an urban scene rasterized onto a grid $G$ of 20 m tiles, $326 \times 310 = 101{,}060$ tiles covering $6{,}200 \times 6{,}520$ m. Four sites are placed at the vertices and centroid of an equilateral triangle with a 2,250 m side, giving 1,299 m from centre to vertex, the SMa inter-site distance of 3GPP TR 38.901 [11]. SMa specifies a 19-site hexagonal grid, of which these four sites form a subset. Each mast is snapped to the nearest open ground, which places one vertex site 1,310 m from the centre. Each site hosts three sectors at azimuths of 45°, 165° and 285° on 35 m masts, the SMa base-station height, yielding $N = 12$ sectors. Each sector is equipped with an $8 \times 8$ cross-polarized planar array with the TR 38.901 element pattern, transmitting 4.85 dBm reference-signal power per resource element. Every sector carries $B = 3$ bands (Table I), giving 36 sector-band pairs.

*Table I. Frequency bands. PRB limits are $N_{RB}$ at 15 kHz subcarrier spacing per 3GPP TS 38.101-1, Table 5.3.2-1 [12].*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per sector |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

![Study area](../reports/figures/00_simulation/study_area.png)

*Fig. 1. Study area: twelve sectors on four sites and a sample of UE positions over the scene.*

### B. Propagation Model

The radio model is not calibrated against measurements. Each band is ray-traced separately with Sionna RT [10], with a maximum path depth of 8 and $10^7$ ray samples per transmitter. Line-of-sight, specular reflection and refraction are modelled; diffuse scattering and diffraction are disabled, and material properties are frequency-static. The receiver is a single vertically polarized dipole at a UE height of 1.5 m. The ray tracer yields, for every sector $i$, band $b$ and tile $g$, the RSRP $R_{i,b}(g)$ and the SINR. All powers are expressed per resource element (RE) [13]: interference comprises every other co-band sector transmitting at full power, and thermal noise is $kT\Delta f$ with $T = 298.15$ K over a single $\Delta f = 15$ kHz subcarrier rather than the channel bandwidth, without a receiver noise figure. Bands are orthogonal, so no inter-band interference is modelled.

### C. Traffic Model

No operator data was available; a synthetic UE population was therefore generated. Every 15 minutes over seven days (672 intervals), 10 to 20 UEs are drawn from a mixture of four elliptical Gaussian demand hotspots and a uniform background over open ground. The hotspots are placed where the surrounding building volume is high, at least 500 m apart, and hold on average 70 % of UEs, modulated by a diurnal profile and first-order autoregressive noise. UEs are independent between intervals, without mobility. This yields 10,087 UE reports, each served from the radio map at its tile. Measurement noise, report censoring and positioning error are not modelled.

### D. Decision Variables

For $N = 12$ sectors and $B = 3$ bands, the decision vector is the absolute electrical tilt of every sector-band pair,

$$
\boldsymbol{\theta} = [\theta_{1,1}, \dots, \theta_{1,B}, \dots, \theta_{N,B}] \in \Theta = [0^\circ, 20^\circ]^{36},
$$

with every proposal snapped to a 0.1° lattice so that each proposal is a discrete, implementable setting. The incumbent configuration $\boldsymbol{\theta}^{(0)}$ sets 10° on every band and sector. No step-size limit or maximum change from $\boldsymbol{\theta}^{(0)}$ is imposed; tilt movement is reported but not penalized.

### E. Serving and Capacity Model

Within each interval, UEs attach sequentially in report-time order. Each UE attaches to the sector-band, among those with RSRP above $T_{\text{hole}} = -120$ dBm at its tile, that offers it the highest Shannon-bound rate under an equal share of $0.8 \, N_{RB}$ PRBs divided among the UEs already attached and itself. No UE is refused admission. A UE with no sector-band above $T_{\text{hole}}$ is unserved and assigned a rate of 0 Mbit/s. The resulting per-UE rate $R_u$ drives the throughput KPIs, the throughput objective and all per-layer service statistics.

### F. Key Performance Indicators

Let $R_s(g) = \max_{i,b} R_{i,b}(g)$ be the best-server RSRP over all bands, $s_b(g)$ the strongest sector of band $b$ at tile $g$, $T_{\text{weak}} = -90$ dBm and $\Delta = 6$ dB the overlap margin. Eleven KPIs are computed for every candidate; seven are reported over all bands:

- **Coverage-hole rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[R_s(g) \le T_{\text{hole}}]$; a tile with no ray-traced path is a hole.
- **Weak-coverage rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[T_{\text{hole}} < R_s(g) \le T_{\text{weak}}]$.
- **Co-band overlap rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[N_{\text{ov}}(g) > 0]$, where
  $$N_{\text{ov}}(g) = \sum_b \mathbb{1}[R_{s_b,b}(g) > T_{\text{hole}}] \cdot \left|\{ i \ne s_b : R_{i,b}(g) \ge R_{s_b,b}(g) - \Delta,\ R_{i,b}(g) > T_{\text{hole}} \}\right|$$
  counts overlapping co-band neighbours, summed over bands. Two carriers of the same sector are never neighbours.
- **Overlapping neighbours per covered tile** (↓): the mean of $N_{\text{ov}}(g)$ over tiles with $R_s(g) > T_{\text{hole}}$.
- **Cell-edge, median and mean estimated UE throughput** (↑): the 5th and 50th percentiles and the mean of $R_u$ over every UE report, unserved UEs counting 0 Mbit/s.

Per band, the hole, weak and overlap rates and the 5th- and 50th-percentile best-server RSRP and SINR over the band's covered tiles are additionally reported. Best-server RSRP and SINR are reported per band only, since the strongest layer across bands is not one on which any UE is measured.

### G. Optimization Problem

Let $G_{\text{cov}} = \{g : R_s(g) > T_{\text{hole}}\}$ be the covered tiles and $U$ the set of UE reports. With RSRP in linear power, three objectives are maximized:

$$
f_{\text{cov}}(\boldsymbol{\theta}) = \frac{|G_{\text{cov}}|}{|G|} = 1 - \text{HoleRate},
$$

$$
f_{\text{sep}}(\boldsymbol{\theta}) = \frac{1}{|G_{\text{cov}}|} \sum_{g \in G_{\text{cov}}} \prod_{b=1}^{B} \frac{R_{s_b,b}(g)}{R_{s_b,b}(g) + \sum_{i \ne s_b} R_{i,b}(g)},
$$

$$
f_{\text{thr}}(\boldsymbol{\theta}) = \frac{1}{|U|} \sum_{u \in U} \ln(1 + R_u), \quad R_u \text{ in Mbit/s}.
$$

The separation objective is a soft, co-band analogue of the overlap KPI: an equal-power rival halves a band's factor, and a band whose strongest sector does not exceed $T_{\text{hole}}$ contributes a factor of 1, leaving holes to the coverage objective. The throughput objective is a proportional-fair utility that rewards raising slow UEs more than fast ones and, by counting unserved UEs at zero rate, penalizes coverage loss under demand. Each objective is quantized to six significant digits before the search observes it, suppressing non-determinism from GPU accumulation order.

The problem is

$$
\max_{\boldsymbol{\theta} \in \Theta} \ \mathbf{F}(\boldsymbol{\theta}) = \left(f_{\text{cov}}, f_{\text{sep}}, f_{\text{thr}}\right),
$$

and a set of evaluated points $P$ is scored by its hypervolume [14] with respect to the origin, the natural floor of every objective:

$$
\mathrm{HV}(P) = \lambda\left(\{\mathbf{z} \in \mathbb{R}^3 : \mathbf{0} \le \mathbf{z} \le \mathbf{y} \text{ for some } \mathbf{y} \in P\}\right),
$$

where $\lambda$ is the Lebesgue measure. The recommended configuration of a run is the evaluated point with the largest hypervolume contribution; the remaining Pareto-optimal points, ordered by contribution, form a shortlist for the operator. No KPI is weighted into the objectives, but the coverage objective is by definition one minus the hole-rate KPI, and the separation objective is related to, though not identical with, the overlap KPIs.

## IV. Search Methods

Both methods search the same bounded, discretized tilt space with the same ray-tracing evaluator. The evaluator constructs the scene once and uses a single fixed solver seed, so all candidates share the same Monte-Carlo noise realization. Each run (i) evaluates the incumbent configuration, (ii) evaluates 8 initial points followed by 64 search points, all scored on every UE report, and (iii) publishes the Pareto front ordered by hypervolume contribution. Every reported quantity is a ray-traced evaluation; no surrogate prediction enters any reported result.

### A. Sobol Random Search

Random search [16] serves as the model-free control. It draws $8 + 64$ points from a seeded, scrambled Sobol sequence [17] over the full 36-dimensional box. Its first eight points are identical to MORBO's initial design, so that, under a matched budget and seed, the difference between the two methods reflects MORBO's search strategy — its surrogate models, trust region and candidate generation together — rather than its starting points.

### B. MORBO

MORBO [6] maintains trust regions centred on Pareto points with the largest hypervolume contribution. The tilt box is rescaled to the unit cube, and the following configuration is used:

- **Local models.** In each round, one Gaussian process (GP) per objective — constant mean, Matérn-5/2 kernel with automatic relevance determination, under a dimension-scaled log-normal lengthscale prior, standardized outputs — is fitted to every evaluation within a cube of twice the trust-region side, supplemented with the nearest points. Evaluations are shared across regions. The models are implemented with BoTorch [8] and GPyTorch [9].
- **Candidate generation.** 2,048 candidates per region and batch point are generated by perturbing a random subset of the coordinates of Pareto points inside the region. Each coordinate is perturbed with probability $p_n = p_0\left(1 - 0.5 \log(n - n_0 + 1) / \log(N_{\text{eval}} - n_0 + 1)\right)$, with $p_0 = \min(20/36, 1)$, so that the perturbation rate halves over the budget.
- **Acquisition.** A batch of three points is selected sequentially by Thompson sampling [15]: each point maximizes the hypervolume improvement of a joint posterior sample over the observed front and the points already selected [7].
- **Trust-region adaptation.** A single trust region is used. It is initialized with side 0.8 and halves after $\max(10, \lceil 36/3 \rceil) = 12$ consecutive failed evaluations, a success being a hypervolume gain exceeding $10^{-3}$ of the current hypervolume. Below side 0.01 the region restarts at the point ranked best by a random hypervolume scalarization in a single draw of a global GP, and the former centre is barred from re-centring for 100 rounds.
- **Budget and seeding.** 8 Sobol initial points and 64 further evaluations. The initial design uses the same seeded Sobol sequence as random search; proposal and restart randomness derive from an independent stream seeded identically.

The dimension-scaled lengthscale prior departs from the original MORBO configuration; without it, GP fitting failed on the ray-traced objectives.

## V. Experimental Methodology

### A. Scenario Summary

*Table II. Scenario parameters.*

| Property | Value |
|---|---|
| Grid | 326 × 310 tiles of 20 m (6,200 × 6,520 m) |
| Sites / sectors / sector-band pairs | 4 / 12 / 36 |
| Inter-site distance | 1,299 m (SMa, TR 38.901 [11]) |
| Mast height | 35 m (SMa) |
| Incumbent tilt (2600 / 1800 / 700 MHz) | 10° / 10° / 10° |
| Tilt bounds and resolution | [0°, 20°], 0.1° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 4, holding 70 % of UEs on average |
| UE reports | 10,087 |
| UE reports with no path to any sector | 13.9 % |
| KPI thresholds | $T_{\text{hole}} = -120$ dBm, $T_{\text{weak}} = -90$ dBm, $\Delta = 6$ dB |
| Search seed | 42 (both methods) |

The fraction of tiles reached by any path is 97.1 % at 700 MHz, 95.1 % at 1800 MHz and 95.2 % at 2600 MHz, with median RSRP over reached tiles of −84.1, −91.3 and −95.5 dBm respectively.

### B. Characterization of the Incumbent Configuration

At the incumbent tilts, 4.40 % of tiles are holes, 29.4 % weak and 66.2 % well covered (Table III). The holes are fragmented into 1,708 connected regions, 1,247 of which comprise a single tile; the largest holds 22.7 % of the hole area, and 1,876 of the 4,445 hole tiles receive no propagation path from any sector. The hole rate is 0.13 % within 1 km of a site and 5.87 % beyond.

Per layer, the hole share is 5.6 % at 700 MHz, 11.3 % at 1800 MHz and 14.1 % at 2600 MHz. Although 700 MHz is the strongest band on 93.9 % of the covered area, the serving rule places 88.1 % of that area on 2600 MHz, the band with the most PRBs. Of all UE reports, 62.5 % are served on 2600 MHz, 12.9 % on 1800 MHz and 8.2 % on 700 MHz. Co-band overlap affects 34.9 % of tiles, with 1.09 overlapping neighbours per covered tile on average (median 0, 90th percentile 3); 20.9 % of covered tiles have three or more, at a median distance of 1,371 m from the nearest site.

Demand and coverage are misaligned: holes occupy 4.4 % of the area but carry 16.4 % of UE reports (Table III), and 13.9 % of reports fall on tiles with no path to any sector. One of the four demand hotspots (1,866 UE reports) is centred 3,297 m from the nearest site, where no layer provides a path at the incumbent tilts. The area-based hole rate therefore understates the service deficit experienced by users.

*Table III. Coverage class by area and by demand at the incumbent configuration.*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole ($\le -120$ dBm) | 4,445 | 4.4 % | 16.4 % |
| Weak ($-120$ to $-90$ dBm) | 29,735 | 29.4 % | 22.2 % |
| Good ($> -90$ dBm) | 66,880 | 66.2 % | 61.4 % |

![RSRP per band](../reports/figures/00_simulation/rsrp_per_band.png)

*Fig. 2. Best-server RSRP per band at the incumbent tilts.*

![Demand vs coverage](../reports/figures/01_eda/demand_vs_coverage.png)

*Fig. 3. UE demand alongside signal strength at the incumbent tilts.*

### C. Data Verification

Prior to optimization, the UE table, scenario manifest and radio map were validated against 28 contract checks covering schema, value ranges, grid consistency, band and frequency agreement, physical plausibility (no RSRP above the transmitted reference-signal power) and the absence of duplicates; all 28 held. No record was removed or altered.

### D. Assessment Criteria

Five criteria are applied. Criteria 1 and 2 assess effectiveness, criteria 3 and 4 trustworthiness, and criterion 5 practicality.

1. **Overall quality:** hypervolume of each run's evaluated set, expressed as a gain over the incumbent's.
2. **Reported KPIs:** direction of change against the incumbent over the seven network-level KPIs and the per-band measures of Section III-F. A change is labelled better, worse or unchanged; since solver noise per KPI has not been measured, no tie tolerance is applied.
3. **Search effectiveness:** sample efficiency, and the paired hypervolume difference between MORBO and random search on the same seed.
4. **Robustness:** where a configuration moves demand, not only area, and how it trades one KPI against another.
5. **Cost:** ray-tracing evaluations, ray-tracing time and wall-clock time per run.

The evaluation operates exclusively on archived run outputs without re-solving. It first verifies that all runs share the incumbent's scenario, grid, solver settings, bands, carrier frequencies and KPI definitions, then recomputes each recommendation's KPIs from its archived radio map to confirm that they were recorded correctly.

## VI. Results and Analysis

### A. Comparability and Reproducibility

All 23 comparability checks hold: both runs optimized the incumbent's scenario, retained their radio maps and match the incumbent's grid, solver settings, bands, carrier frequencies and KPI definitions. Recomputing each recommendation's KPIs from its archived radio map reproduces the recorded values exactly for the hole, weak and overlap rates and the objectives, and to within $3 \times 10^{-6}$ Mbit/s for the throughput KPIs. Repeatability across seeds remains untested, as each method was executed with one seed.

### B. Network-Level KPIs

*Table IV. Recommended configuration of each method against the incumbent. Bold marks the best value per row.*

| KPI | Direction | Incumbent | MORBO | Random search |
|---|:-:|---:|---:|---:|
| Coverage-hole rate | ↓ | 4.40 % | **4.30 %** | 4.50 % |
| Weak-coverage rate | ↓ | 29.4 % | **24.6 %** | 29.6 % |
| Co-band overlap rate | ↓ | **34.9 %** | 36.6 % | 48.7 % |
| Overlapping neighbours per covered tile | ↓ | 1.09 | **1.01** | 1.20 |
| Cell-edge throughput, p05 [Mbit/s] | ↑ | 0.0 | 0.0 | 0.0 |
| Median throughput [Mbit/s] | ↑ | 51.7 | 55.5 | **61.5** |
| Mean throughput [Mbit/s] | ↑ | 80.4 | 85.0 | **88.6** |
| Coverage objective $f_{\text{cov}}$ | ↑ | 0.9560 | **0.9570** | 0.9550 |
| Separation objective $f_{\text{sep}}$ | ↑ | 0.6778 | **0.6994** | 0.6388 |
| Throughput objective $f_{\text{thr}}$ | ↑ | 3.513 | 3.566 | **3.628** |
| Network KPIs better / worse / unchanged | | | 5 / 1 / 1 | 2 / 4 / 1 |
| Hypervolume of all evaluations | ↑ | 2.276 | **2.433** | 2.353 |

*Table V. Best-server RSRP and SINR per band over each band's covered tiles.*

| Band | Configuration | RSRP p50 [dBm] | RSRP p05 [dBm] | SINR p50 [dB] | SINR p05 [dB] |
|---|---|---:|---:|---:|---:|
| 2600 MHz | Incumbent | −93.4 | −113.2 | 12.7 | −0.93 |
| | MORBO | −93.0 | −113.9 | 15.0 | −0.92 |
| | Random search | −94.1 | −114.5 | 12.1 | −1.32 |
| 1800 MHz | Incumbent | −89.9 | −110.3 | 12.2 | −0.98 |
| | MORBO | −89.0 | −110.6 | 13.9 | −1.10 |
| | Random search | −89.0 | −110.1 | 13.3 | −1.05 |
| 700 MHz | Incumbent | −83.6 | −105.3 | 11.5 | −1.12 |
| | MORBO | −82.0 | −104.5 | 13.9 | −0.92 |
| | Random search | −86.0 | −108.2 | 9.9 | −1.52 |

![KPI comparison](../reports/figures/04_evaluation/kpi_comparison.png)

*Fig. 4. Network-level KPIs of each method's recommendation against the incumbent.*

![Objective comparison](../reports/figures/04_evaluation/objective_comparison.png)

*Fig. 5. Objectives of each method's recommendation against the incumbent.*

Four observations follow from Tables IV and V.

*The hole rate is nearly insensitive to tilt.* In the studied layout, the incumbent configuration already holds the hole rate below 5 %. The recommendations shift it by a tenth of a percentage point (−0.10 for MORBO, +0.10 for random search), and the lowest value found by either search is 4.15 %. One reason is structural: 1,876 of the 4,445 incumbent hole tiles receive no propagation path from any sector, and tilt alters antenna gain, not the existence of a path. Whether a different site layout would close these holes was not tested.

*MORBO trades a small overlap increase for gains elsewhere.* Its recommendation achieves the largest reduction in weak coverage (−4.8 points) and raises median SINR on every band, by 2.3 dB at 2600 MHz, 1.6 dB at 1800 MHz and 2.4 dB at 700 MHz. The tails do not all improve: 5th-percentile RSRP falls by 0.7 dB at 2600 MHz and 0.2 dB at 1800 MHz, and 5th-percentile SINR falls by 0.1 dB at 1800 MHz. The all-band overlap rate rises by 1.7 points, even though the overlap rate of every individual band falls (Table VIII); since the all-band rate counts a tile if any band overlaps there, the overlapping tiles of different bands must coincide less than before.

*Random search buys throughput with overlap.* Its recommendation attains the highest median and mean throughput, but the overlap rate rises by 13.9 points and the separation objective falls by 0.039. Its 700 MHz layer loses 2.4 dB of median RSRP and 1.5 dB of median SINR, weakening the coverage layer.

*Cell-edge throughput is pinned at zero.* Between 16.4 % and 16.6 % of UE reports lie on hole tiles in every configuration, exceeding 5 %, so the 5th-percentile throughput is 0 Mbit/s throughout and cannot discriminate between configurations.

### C. Search Effectiveness

*Table VI. Hypervolume per method (seed 42).*

| Method | Incumbent | Initial design | All evaluations | Gain over incumbent | Pareto points | Recommended evaluation |
|---|---:|---:|---:|---:|---:|---:|
| MORBO | 2.2762 | 2.3388 | 2.4325 | +6.9 % | 18 | 66 |
| Random search | 2.2762 | 2.3388 | 2.3528 | +3.4 % | 9 | 65 |

*Table VII. Best hypervolume and hole rate reached after a given number of evaluations.*

| Evaluations | HV, MORBO | HV, random | Hole rate, MORBO | Hole rate, random |
|---:|---:|---:|---:|---:|
| 10 | 2.3388 | 2.3388 | 4.40 % | 4.40 % |
| 25 | 2.3544 | 2.3388 | 4.25 % | 4.40 % |
| 50 | 2.4010 | 2.3468 | 4.15 % | 4.26 % |
| 73 | 2.4325 | 2.3528 | 4.15 % | 4.26 % |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Fig. 6. Best hypervolume found against the number of evaluations.*

![MORBO evaluations](../reports/figures/03b_morbo/morbo_evaluations.png)

*Fig. 7. Every MORBO evaluation, grouped by proposer (Sobol initial design or trust-region proposal).*

*The difference arises in the search, on this seed.* Because both methods share the same eight-point initial design, the difference between their final hypervolumes arises after it. MORBO's final hypervolume exceeds that of random search by 0.080 on the single seed pair; with one pair, no confidence interval or significance test can be reported, and the margin may not generalize across seeds.

*MORBO is more sample-efficient on this seed.* After 25 evaluations, MORBO already exceeds the hypervolume that random search reaches after 73, and by evaluation 50 its best hole rate (4.15 %) is below random search's (4.26 %). MORBO's hypervolume was still rising between evaluations 50 and 73, indicating that the budget did not exhaust its progress.

*The Pareto front consists entirely of model-guided proposals.* None of the eight Sobol points lies on MORBO's front; all 18 Pareto-optimal points are among its 64 trust-region proposals. These proposals also exceed the initial design on average in all three objectives: mean coverage 0.9570 against 0.9522, mean separation 0.6789 against 0.6497 and mean throughput utility 3.589 against 3.529.

*No evaluation reduced the all-band overlap rate.* The best overlap rate found by either method equals the incumbent's 34.9 %: within the explored space, no configuration lowered the aggregate overlap.

*The recommendation is not the minimum-hole configuration.* MORBO's lowest observed hole rate is 4.15 %, whereas its recommendation — the point with the largest hypervolume contribution — lies at 4.30 %. The remaining front points constitute alternatives for an operator who prioritizes hole reduction.

![Coverage vs separation trade-off](../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png)

*Fig. 8. All evaluated configurations projected onto the coverage and separation objectives, with each method's front and recommendation.*

### D. Robustness: Per-Layer and Demand-Weighted Effects

*Table VIII. Per-band coverage-hole, weak-coverage and co-band overlap rates.*

| Band | Measure | Incumbent | MORBO | Random search |
|---|---|---:|---:|---:|
| 2600 MHz | Hole / weak / overlap | 14.1 / 50.2 / 23.8 % | 15.4 / 49.5 / 20.8 % | 16.2 / 52.1 / 24.2 % |
| 1800 MHz | Hole / weak / overlap | 11.3 / 44.1 / 25.2 % | 11.5 / 41.9 / 23.4 % | 11.2 / 41.5 / 24.5 % |
| 700 MHz | Hole / weak / overlap | 5.6 / 29.4 / 28.2 % | 5.5 / 25.2 / 24.9 % | 5.9 / 36.1 / 31.2 % |
| All bands | Overlap (KPI) | 34.9 % | 36.6 % | 48.7 % |

*Table IX. Coverage class by area and by demand (share of UE reports).*

| Coverage class | Incumbent: area | Incumbent: demand | MORBO: area | MORBO: demand | Random: area | Random: demand |
|---|---:|---:|---:|---:|---:|---:|
| Hole | 4.40 % | 16.40 % | 4.30 % | 16.58 % | 4.50 % | 16.36 % |
| Weak | 29.4 % | 22.2 % | 24.6 % | 19.4 % | 29.6 % | 20.5 % |
| Good | 66.2 % | 61.4 % | 71.1 % | 64.1 % | 65.9 % | 63.2 % |

*Table X. Distribution of overlapping co-band neighbours over covered tiles.*

| Configuration | Mean | 0 neighbours | 1 | 2 | 3 or more |
|---|---:|---:|---:|---:|---:|
| Incumbent | 1.09 | 63.5 % | 9.4 % | 6.2 % | 20.9 % |
| MORBO | 1.01 | 61.8 % | 13.8 % | 8.2 % | 16.3 % |
| Random search | 1.20 | 49.0 % | 22.6 % | 11.7 % | 16.8 % |

![Coverage before and after, 2600 MHz](../reports/figures/04_evaluation/coverage_before_after_b2600.png)

*Fig. 9. 2600 MHz best-server RSRP before and after MORBO's recommendation, and the tiles that crossed the hole threshold.*

![RSRP change maps, 2600 MHz](../reports/figures/04_evaluation/rsrp_change_maps_b2600.png)

*Fig. 10. Change in 2600 MHz best-server RSRP under each method's recommendation.*

*Area gains do not reach demand in the holes.* MORBO's recommendation reduces hole area by 0.10 points but raises the share of UE reports on hole tiles from 16.40 % to 16.58 %: the hole tiles it opens carry more demand than those it closes. The coverage and separation objectives are tile-uniform and do not weight by demand. The throughput objective does count demand, but the 13.9 % of UE reports on tiles with no propagation path to any sector contribute zero regardless of tilt and therefore exert no pressure on the search.

*Weak coverage improves by both area and demand.* MORBO reduces the weak-coverage share by 4.8 points of area and 2.9 points of demand, and raises the good-coverage share by 4.9 and 2.7 points respectively.

*The capacity layer contracts while the coverage layer strengthens.* Under MORBO, the 2600 MHz hole rate rises from 14.1 % to 15.4 %, whereas the 700 MHz weak rate falls from 29.4 % to 25.2 %. Each band's own overlap rate falls, by 3.0 points at 2600 MHz, 1.8 at 1800 MHz and 3.3 at 700 MHz. Random search, in contrast, increases the 700 MHz weak rate by 6.7 points and its overlap by 3.0 points.

*Overlap becomes shallower but more widespread under MORBO.* The share of covered tiles with three or more overlapping neighbours falls from 20.9 % to 16.3 %, while tiles with one or two neighbours increase. Random search shifts a far larger share of tiles from zero to one or two neighbours.

### E. Inter-Layer Load Redistribution

*Table XI. UE service and median served SINR per band.*

| Configuration | Served on 2600 MHz | Served on 1800 MHz | Served on 700 MHz | Not served | Served SINR p50, 2600 / 1800 / 700 MHz [dB] |
|---|---:|---:|---:|---:|---|
| Incumbent | 62.5 % | 12.9 % | 8.2 % | 16.4 % | 11.2 / 15.3 / 19.8 |
| MORBO | 55.7 % | 16.1 % | 11.6 % | 16.6 % | 13.1 / 16.9 / 17.9 |
| Random search | 56.0 % | 19.1 % | 8.6 % | 16.4 % | 17.4 / 17.7 / 20.5 |

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Fig. 11. Serving-band mix per configuration.*

![Sector-band throughput](../reports/figures/04_evaluation/sector_band_throughput.png)

*Fig. 12. Median estimated throughput per sector-band, incumbent and recommended.*

*Traffic migrates from 2600 MHz to the lower layers.* Under MORBO's recommendation, the 2600 MHz share of UE reports falls by 6.8 points, while 1800 MHz gains 3.2 points and 700 MHz 3.4 points. Median served SINR on 2600 MHz rises by 1.9 dB, whereas on 700 MHz it falls by 1.8 dB while that layer serves more users.

*The largest changes are load-shedding downtilts on the capacity layer.* Sector n2s1 (site 2, sector 1) at 2600 MHz, downtilted by 4.2°, serves 664 fewer reports (963 → 299), and n1s0 at 2600 MHz, downtilted by 5.4°, serves 384 fewer (521 → 137); the median throughput of the UEs that remain rises by 18.4 and 20.8 Mbit/s respectively.

*Some uptilted capacity carriers serve more reports.* Sector n3s2 at 2600 MHz, uptilted by 8.1°, serves 363 additional reports (769 → 1,132) at a 4.3 dB higher median SINR, and n0s2 at 2600 MHz, uptilted by 7.4°, serves 214 additional reports. Which sectors previously served these reports was not traced. Ranking sector-bands by the magnitude of the change in served reports can give an operator a prioritized monitoring list for a staged rollout.

### F. Recommended Tilt Configuration

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Fig. 13. Tilt change per sector and band in MORBO's recommendation.*

*Table XII. Tilt movement of MORBO's recommendation. Negative $\Delta$ denotes an uptilt.*

| Band | Sectors moved | Mean $\lvert\Delta\rvert$ [°] | Largest $\lvert\Delta\rvert$ [°] | Mean $\Delta$ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 12 of 12 | 4.66 | 9.2 | −0.24 |
| 1800 MHz | 12 of 12 | 4.51 | 8.9 | −1.31 |
| 700 MHz | 12 of 12 | 5.41 | 9.4 | −1.57 |

All 36 tilts change, 24 upward and 12 downward, by up to 9.4°, with resulting tilts between 0.6° and 19.2°. No tilt reaches a bound, so the bounds are not active at the recommendation. The 700 MHz layer moves furthest on average and toward the horizon (mean −1.6°), which is consistent with extending the coverage layer, whereas 2600 MHz nets out near zero (−0.2°) through large opposing per-sector moves, accompanied by the load shifts of Section VI-E. Because movement is unpenalized, the recommendation corresponds to a 36-antenna RET change request. The tilt movement of the other Pareto-optimal configurations was not analyzed, so whether the front contains a comparable option with less movement remains open.

### G. Computational Cost

*Table XIII. Search cost on a single 4 GB laptop-class GPU (NVIDIA RTX 3050).*

| Method | Evaluations | Recommended evaluation | Ray tracing [min] | Wall clock [min] |
|---|---:|---:|---:|---:|
| MORBO | 73 | 66 | 2.85 | 7.24 |
| Random search | 73 | 65 | 2.87 | 3.42 |

Ray tracing costs approximately 2.4 s per full-network evaluation for both methods. The additional wall-clock time of MORBO is incurred outside the ray tracer, in GP fitting and acquisition optimization. A complete joint optimization of 36 tilts therefore requires minutes of commodity compute in this setting. This covers the simulation only; field validation of the selected configuration comes on top of it.

## VII. Discussion

### A. Implications for RAN Operation

Subject to the limitations below, the results suggest three possible changes to tilt-optimization practice. First, joint optimization can complement trial-and-error with selection among simulated alternatives: a single run produced 18 Pareto-optimal configurations, each accompanied by its full KPI vector and per-antenna tilt deltas, from which an engineer can select according to operational priorities. Second, the recommended configuration moves all bands jointly: it uptilts the coverage layer on average while making opposing per-sector moves on the capacity layer, and load shifts across sectors and frequency layers accordingly. Whether a band-by-band procedure would reach a similar configuration was not tested. Third, the per-sector impact ranking can inform a staged rollout plan with an explicit monitoring order.

In the studied layout, however, tilt optimization barely changed the coverage-hole rate and did not close the holes that carry demand; a large share of hole tiles have no propagation path to any sector. In such a setting, tilt search is better directed at weak coverage, SINR and inter-layer load balance, while demand that no layer reaches is a candidate for site planning rather than tilt tuning.

### B. Limitations and Threats to Validity

- **Single scenario.** One scene, one UE realization and one layout were studied; no result has been validated on a held-out scenario or calibrated against drive-test measurements.
- **Partial SMa layout.** The layout follows the SMa inter-site distance and mast height [11] but uses 4 of its 19 sites, over an urban rather than suburban scene. UEs lie on a single outdoor plane at 1.5 m; the indoor, multi-floor UE distribution of SMa is not modelled. The incumbent is a uniform 10° configuration rather than an operator-tuned one.
- **Limited baselines.** MORBO is compared only with random search and a uniform 10° incumbent. Neither an operator-tuned configuration, a band-by-band procedure nor another optimizer, such as an evolutionary algorithm or reinforcement learning, was evaluated.
- **Synthetic traffic and full-load interference.** The traffic is synthetic, and SINR assumes every co-band sector transmits at full power, which is pessimistic at low load.
- **Single seed per method.** No confidence interval, significance test or repeatability measure is available for Tables IV and VI.
- **Winner's curse and solver noise.** All evaluations share one solver seed, and each recommendation is the best of 73 under that noise realization, so its scores are biased upward. Solver noise per KPI is unmeasured; changes of a tenth of a point in the hole rate may lie within it.
- **Reference-point dependence.** Hypervolume is computed against the origin; a reference point at the incumbent, or a selection restricted to configurations dominating the incumbent, could yield a different recommendation.
- **Algorithmic configuration.** A single trust region was used and the budget of 73 evaluations in 36 dimensions is small; MORBO was still improving late in the budget. The original method's default number of trust regions was not evaluated.
- **Objective–KPI mismatch on overlap.** The separation objective is a soft per-tile product over bands, whereas the all-band overlap KPI counts any crowded band; the recommendation improves the former while the latter worsens.
- **Demand-agnostic area objectives.** Coverage and separation are tile-uniform; the hole-rate reduction occurred on low-demand tiles, and the share of UE reports on hole tiles rose slightly (Table IX).
- **Simplified capacity model.** Rates are equal-share Shannon bounds without scheduling, MCS limits or mobility. Inter-band interference is absent by construction, and cell-edge throughput is 0 Mbit/s in every configuration because more than 5 % of UE reports lie on hole tiles.
- **Unpenalized movement.** The recommendation moves every antenna (Table XII), which may exceed a practical RET change window.

## VIII. Conclusion and Future Work

This paper formulated the joint configuration of electrical tilts across all sectors and bands of a multi-band network as a 36-dimensional, three-objective black-box problem evaluated by a ray-traced digital twin, and compared MORBO with Sobol random search under matched conditions. Table XIV summarizes the outcome against the assessment criteria.

*Table XIV. Summary against the assessment criteria.*

| Criterion | Random search | MORBO |
|---|---|---|
| 1. Hypervolume | +3.4 % over incumbent | **+6.9 % over incumbent** |
| 2. Reported KPIs | 2 better, 4 worse; overlap +13.9 points | **5 better, 1 worse**; overlap +1.7 points |
| 3. Search effectiveness | Best hole rate 4.26 %; 9 Pareto points | **Exceeds random search's final hypervolume by evaluation 25**; 18 Pareto points, all model-guided |
| 4. Robustness | Weakens the 700 MHz layer (−2.4 dB median RSRP) | Weak coverage reduced by area and demand; hole-tile demand share +0.18 points |
| 5. Cost | 73 evaluations, 3.4 min | 73 evaluations, 7.2 min |

The principal conclusions are:

1. In the studied layout, the uniform 10° incumbent already has a hole rate of 4.40 % before any optimization, and 1,876 of its 4,445 hole tiles receive no propagation path that tilt could change.
2. Joint tilt optimization contributes little to the hole rate in this geometry — the lowest value found was 4.15 %, and MORBO's recommendation attains 4.30 % — but delivers 4.8 points less weak coverage, 1.6–2.4 dB higher median SINR per band and 4.5 Mbit/s higher mean throughput, at the cost of 1.7 points of all-band co-band overlap.
3. At an equal budget and on one seed per method, MORBO reached a larger hypervolume than random search, and its recommendation improved five of seven network KPIs against random search's two; whether this margin holds across seeds is untested.
4. The area-based hole rate understates the service deficit: about 16 % of UE reports lie on hole tiles in every configuration, and one demand hotspot is centred where no sector provides a propagation path.

Future work will (i) quantify solver noise by re-tracing recommendations under multiple solver seeds; (ii) execute multiple search seeds per method to obtain confidence intervals and significance tests; (iii) weight the coverage objective by demand, or report a demand-weighted hole rate alongside it; (iv) extend the layout to the full 19-site SMa grid with indoor, multi-floor UEs; (v) introduce a tilt-movement penalty or constraint so that recommendations fit practical RET change windows; (vi) compare against an operator-tuned configuration and further optimizers; and (vii) incorporate a scheduler-level capacity model and validate the approach against operator measurements.

---

## References

[1] 3GPP TR 36.902, *Evolved Universal Terrestrial Radio Access Network (E-UTRAN); Self-configuring and self-optimizing network (SON) use cases and solutions*, Release 9, clause 4.1.

[2] R. Razavi, S. Klein, and H. Claussen, "A fuzzy reinforcement learning approach for self-optimization of coverage in LTE networks," *Bell Labs Technical Journal*, vol. 15, no. 3, pp. 153–175, Dec. 2010, doi: 10.1002/bltj.20463.

[3] N. Dandanov, H. Al-Shatri, A. Klein, and V. Poulkov, "Dynamic self-optimization of the antenna tilt for best trade-off between coverage and capacity in mobile networks," *Wireless Personal Communications*, vol. 92, no. 1, pp. 251–278, 2017, doi: 10.1007/s11277-016-3849-9.

[4] R. M. Dreifuerst et al., "Optimizing coverage and capacity in cellular networks using machine learning," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021, pp. 8138–8142.

[5] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 5496–5507.

[6] S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy, "Multi-objective Bayesian optimization over high-dimensional search spaces," in *Proc. Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 180, pp. 507–517, 2022.

[7] S. Daulton, M. Balandat, and E. Bakshy, "Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization," in *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 2020, pp. 9851–9864.

[8] M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy, "BoTorch: A framework for efficient Monte-Carlo Bayesian optimization," in *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 2020, pp. 21524–21538.

[9] J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, and A. G. Wilson, "GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration," in *Advances in Neural Information Processing Systems 31 (NeurIPS)*, 2018, pp. 7576–7586.

[10] J. Hoydis, F. Aït Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus, and A. Keller, "Sionna RT: Differentiable ray tracing for radio propagation modeling," in *Proc. IEEE Globecom Workshops*, 2023.

[11] 3GPP TR 38.901 V19.2.0 (ETSI TR 138 901 V19.2.0, 2026-02), *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-5, evaluation parameters for SMa scenarios.

[12] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[13] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.3.

[14] E. Zitzler and L. Thiele, "Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach," *IEEE Transactions on Evolutionary Computation*, vol. 3, no. 4, pp. 257–271, 1999.

[15] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, vol. 25, no. 3–4, pp. 285–294, 1933, doi: 10.1093/biomet/25.3-4.285.

[16] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281–305, 2012.

[17] I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals," *USSR Computational Mathematics and Mathematical Physics*, vol. 7, no. 4, pp. 86–112, 1967, doi: 10.1016/0041-5553(67)90144-9.
