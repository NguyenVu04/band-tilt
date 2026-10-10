# Joint Multi-Band Antenna Tilt Optimization with High-Dimensional Multi-Objective Bayesian Optimization over a Ray-Traced Network Simulator

---

## Abstract

Multi-band base stations transmit several frequency bands from the same mast, yet their antenna tilts are often tuned one band and one sector at a time, without accounting for how the bands interact. This paper optimizes all tilts of a five-site, three-band network jointly, scoring each candidate with an uncalibrated, site-specific ray-traced simulation and searching with high-dimensional multi-objective Bayesian optimization (MORBO). Every configuration is read on three KPIs, coverage rate, co-band separation rate and median estimated user throughput, and the resulting Pareto front is published with its tilts for an engineer to choose from. Against Sobol random search at an equal, verified budget of 73 ray traces from a shared initial design, MORBO reaches a larger hypervolume on its objectives and on the three KPIs, and its searched candidates score higher on every KPI (one-sided Mann–Whitney $p \le 5 \times 10^{-5}$, Cliff's δ 0.40 to 0.80). Its chosen configuration raises the coverage rate from 0.933 to 0.942 and the median throughput from 60 to 85 Mbps, higher in 608 of 672 fifteen-minute intervals, while raising every band's own separation rate. No evaluated configuration of either method improves the band-collapsed separation rate, which exposes the trade-off a single recommendation would hide. The evaluation is limited to one simulated scenario with synthetic traffic.

**Index Terms** — 5G, radio access network, antenna tilt optimization, coverage and capacity optimization, multi-band networks, Bayesian optimization, multi-objective optimization, ray tracing.

---

## I. Introduction

### A. Motivation

**What users experience.** For a mobile user, network quality is decided at the margins. A call drops when a commuter crosses from one cell into the next, a video stalls at the far end of a street, and a phone shows a strong signal yet delivers little data because several cells are competing for it. Users perceive none of the network's architecture, only that service is unreliable in particular places. One lever an operator has over these effects is the downward angle, or *tilt*, of the base-station antennas. Tilted too high, a cell spills into its neighbours and creates interference; tilted too low, it leaves gaps at its edge.

**Why the problem has become harder.** Modern 5G sites no longer transmit on a single frequency. A mast commonly carries several bands with complementary roles. Low bands travel far and penetrate buildings, forming a *coverage layer*; high bands carry much more data over a shorter range, forming a *capacity layer*; mid bands bridge the two. Every band on every sector has its own tilt, so the number of settings grows with each band an operator adds. These settings are also coupled. Raising the tilt of a capacity band widens its footprint into neighbouring cells and increases interference there. Lowering it pulls the cell edge inward and may leave users relying on a slower band. Where the serving band depends on signal quality and load, changing one band's tilt can move traffic onto other bands, changing the performance of carriers that nobody touched.

**What engineers face.** A reactive tuning workflow illustrates the difficulty. Tilts are set at planning time and then adjusted reactively. A drive test, a performance alarm or a customer complaint points to a problem sector; an engineer changes the tilt of one band on that sector remotely; and the team then waits for performance counters to show whether the change helped, and whether it quietly degraded a neighbouring cell. Each step is reasonable on its own, but the loop is slow, costly in field effort and blind to cross-band side effects until they appear in the statistics. Engineers are left addressing individual complaints rather than optimizing the network as a whole, and the network-wide configuration becomes the accumulation of local decisions rather than one chosen jointly.

**What is missing.** Engineers need a way to evaluate the *whole* network's tilt configuration, across every band and sector at once, before touching any antenna; to see the trade-offs between coverage, interference and user throughput explicitly rather than discovering them afterwards; and to obtain a small set of concrete, reviewable options rather than a single opaque answer. Such a tool must be economical in the number of evaluations, since each site-specific evaluation of a multi-site network is far more costly than a closed-form model, and transparent, so that an engineer can see where each option gains, where it loses and which antennas must change.

This paper addresses the static part of that need: coverage, interference and throughput at fixed UE positions; mobility and handover are not modelled. It combines a ray-traced model of a five-site urban scene, which estimates how every candidate configuration would affect coverage, interference and user throughput, with a sample-efficient multi-objective optimizer that learns which configurations are worth testing. The outcome is the Pareto front of simulated options on coverage, separation and throughput, each accompanied by its estimated effect on every performance indicator, per frequency layer, and the antenna changes it requires, produced in minutes of computation in this setting before any antenna is changed. The radio model is not calibrated against measurements, and field validation of the chosen option remains necessary.

### B. Approach and Contributions

This paper treats the tilts of all (sector, band) pairs as a single coordinated optimization problem and evaluates it entirely in simulation. A ray tracer applied to explicit scene geometry, without calibration against measurements, scores each candidate configuration on coverage, co-band separation and the estimated equal-share Shannon rate of a synthetic UE population. The contributions are as follows:

1. **Problem formulation.** Joint multi-band tilt configuration is cast as a 45-dimensional, three-objective black-box problem with a co-band separation objective and a proportional-fair throughput utility that counts unserved UEs at zero rate (Section III).
2. **Sample-efficient search under a verified equal budget.** MORBO [6], a trust-region multi-objective Bayesian optimization method designed for high-dimensional spaces, is compared with Sobol random search under a matched evaluation budget, a shared initial design and a common seed, each checked on the runs themselves (Sections IV–V).
3. **KPI-level evaluation for engineers.** Every evaluated configuration is read on coverage rate, separation rate and median throughput, network-wide and per band; the methods are compared by non-parametric tests on their searched candidates and by front-set coverage, and the combined Pareto front is published with every sector-band tilt (Sections V–VI).
4. **Empirical findings.** MORBO's searched candidates beat random search's on all three KPIs with medium to large effect sizes, and its chosen configuration raises coverage and median throughput in space and over time. No configuration of either method improves the band-collapsed separation rate, although MORBO's raises every band's own separation, which shows how a band-collapsed interference KPI can hide per-layer gains (Section VI).

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Section IV describes the search methods. Section V details the experimental methodology and assessment criteria. Section VI reports and interprets the results. Section VII discusses implications and limitations, and Section VIII concludes.

## II. Related Work

**Coverage and capacity optimization.** Antenna tilt is a principal control in coverage and capacity optimization (CCO), one of the self-organizing network (SON) use cases identified by 3GPP [1]. Early automated approaches adjusted tilt per cell with rule-based, fuzzy or reinforcement-learning controllers driven by local measurements [2], [3], and later ones from call traces, again cell by cell [21]. Many such controllers act on one cell, or one carrier, at a time, and therefore share the locality of manual tuning.

**Black-box optimization of RAN parameters.** Because network KPIs are expensive to evaluate and non-differentiable with respect to configuration, tilt and power settings have been optimized with Bayesian optimization and reinforcement learning over simulators. Dreifuerst et al. [4] tune the downtilt and transmit power of all sectors jointly with multi-objective Bayesian optimization and deep reinforcement learning, and obtain Pareto fronts between coverage and capacity. Tekgul et al. [22] tune tilt and beamwidths per cell for joint uplink-downlink coverage and capacity with a site-specific, sample-efficient learning method, and Benzaghta et al. [23] use MORBO to trade ground-user rates against aerial coverage by sector tilt. Standard Gaussian-process Bayesian optimization degrades in high dimensions; trust-region methods such as TuRBO [5] restore sample efficiency by restricting the search to adaptively sized local regions. MORBO [6] extends this idea to multiple objectives, maintaining trust regions centred on points of large hypervolume contribution and selecting batches by Thompson-sampled hypervolume improvement [7], [13].

**Ray-traced propagation models.** Statistical path-loss models represent building shadowing and multipath only in distribution, whereas the effect of a tilt change at a given location depends on site-specific blockage and reflections. GPU-accelerated ray tracers such as Sionna RT [8] make site-specific evaluation of a full network configuration fast enough — about 3 s per evaluation for the 15-sector scene of this study — for optimization loops of tens to hundreds of full-network evaluations.

**Positioning of this work.** Joint optimization across sectors with a Pareto front between coverage and capacity has been shown before [4]. The present study differs in treating each band of a multi-band sector as a separate decision variable, coupled to the other bands through the serving rule; in adding a co-band separation objective; in searching 45 dimensions with a trust-region method; and in scoring candidates with a site-specific ray tracer. It further evaluates every candidate on network- and layer-level KPIs, compares the methods with non-parametric tests and front-set coverage, and publishes the front with its tilts. Agreement between the objectives and these KPIs is partly by construction: the throughput KPIs share the per-UE rates of the throughput objective, and the overlap KPIs are related to the separation objective. The per-layer and demand-level measures are the less constrained check.

## III. System Model and Problem Formulation

### A. Network Layout

The study area is an urban scene rasterized onto a grid $G$ of 20 m tiles, $326 \times 310 = 101{,}060$ tiles covering $6{,}200 \times 6{,}520$ m. Five sites stand on the corners of an axis-aligned square and at its centre, each corner 1,732 m from the centre site, so adjacent corners are 2,449 m apart. Each mast stands on open ground at its ideal position. Each site hosts three sectors at azimuths of 0°, 120° and 240° on 25 m masts, the urban-macro (UMa) base-station height of 3GPP TR 38.901 [9], yielding $N = 15$ sectors. Each sector is equipped with an $8 \times 8$ cross-polarized planar array with the TR 38.901 element pattern, transmitting 4.85 dBm reference-signal power per resource element. Every sector carries $B = 3$ bands (Table I), giving 45 sector-band pairs.

*Table I. Frequency bands. PRB limits are $N_{RB}$ at 15 kHz subcarrier spacing per 3GPP TS 38.101-1, Table 5.3.2-1 [10].*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per sector |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

![Study area](../reports/figures/00_simulation/study_area.png)

*Fig. 1. Study area: fifteen sectors on five sites and a sample of UE positions over the scene.*

### B. Propagation Model

The radio model is not calibrated against measurements. Each band is ray-traced separately with Sionna RT [8], with a maximum path depth of 8 and $10^7$ ray samples per transmitter. Line-of-sight, specular reflection and refraction are modelled; diffuse scattering and diffraction are disabled, and material properties are frequency-static. The receiver is a single vertically polarized dipole at a UE height of 1.5 m. The ray tracer yields, for every sector $i$, band $b$ and tile $g$, the RSRP $R_{i,b}(g)$ and the SINR. All powers are expressed per resource element (RE) [11]: interference comprises every other co-band sector transmitting at full power, and thermal noise is $kT\Delta f$ with $T = 298.15$ K over a single $\Delta f = 15$ kHz subcarrier rather than the channel bandwidth, without a receiver noise figure. Bands are orthogonal, so no inter-band interference is modelled.

### C. Traffic Model

No operator data was available; a synthetic UE population was therefore generated. Every 15 minutes over seven days (672 intervals), 10 to 20 UEs are drawn from a mixture of three elliptical Gaussian demand hotspots and a uniform background over open ground. The hotspots are placed where the surrounding building volume is high, at least 500 m apart, and hold on average 70 % of UEs, modulated by a diurnal profile and first-order autoregressive noise. UEs are independent between intervals, without mobility. This yields 10,087 UE reports, which are synthetic positions rather than measurements, each served from the radio map at its tile. Measurement noise, report censoring and positioning error are not modelled.

### D. Decision Variables

For $N = 15$ sectors and $B = 3$ bands, the decision vector is the absolute downtilt of every sector-band pair,

$$
\boldsymbol{\theta} = [\theta_{1,1}, \dots, \theta_{1,B}, \dots, \theta_{N,B}] \in \Theta = [0^\circ, 20^\circ]^{45},
$$

with every proposal snapped to a 0.1° lattice so that each proposal is a discrete setting. Whether a given antenna supports this range and an independent electrical tilt per band was not checked. The simulator applies each tilt as a rotation of the whole array in elevation, so side and back lobes tilt with the main beam, as under mechanical tilt. The incumbent configuration $\boldsymbol{\theta}^{(0)}$ sets 12° on every band and sector, the UMa electrical downtilt of the TR 38.901 calibration parameters [9]. No step-size limit or maximum change from $\boldsymbol{\theta}^{(0)}$ is imposed; tilt movement is reported but not penalized.

### E. Serving and Capacity Model

Within each interval, UEs attach sequentially in report-time order. Each UE attaches to the sector-band, among those with RSRP above $T_{\text{hole}} = -110$ dBm at its tile, that offers it the highest Shannon-bound rate under an equal share of $0.8 \, N_{RB}$ PRBs divided among the UEs already attached and itself. No UE is refused admission. A UE with no sector-band above $T_{\text{hole}}$ is unserved and assigned a rate of 0 Mbps. The resulting per-UE rate $R_u$ drives the throughput KPIs, the throughput objective and all per-layer service statistics. This rate-greedy rule is an idealization, not a model of any vendor's band-selection or load-balancing policy, so the inter-layer load results of Section VI-E describe this rule.

### F. Key Performance Indicators

Let $R_s(g) = \max_{i,b} R_{i,b}(g)$ be the best-server RSRP over all bands, $s_b(g)$ the strongest sector of band $b$ at tile $g$, $T_{\text{weak}} = -90$ dBm and $\Delta = 6$ dB the overlap margin. Eleven KPIs are computed for every candidate: the seven below, reported over all bands, and the 5th- and 50th-percentile best-server RSRP and SINR, reported per band only:

- **Coverage-hole rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[R_s(g) \le T_{\text{hole}}]$; a tile with no ray-traced path is a hole.
- **Weak-coverage rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[T_{\text{hole}} < R_s(g) \le T_{\text{weak}}]$.
- **Co-band overlap rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[N_{\text{ov}}(g) > 0]$, where
  $$N_{\text{ov}}(g) = \sum_b \mathbb{1}[R_{s_b,b}(g) > T_{\text{hole}}] \cdot \left|\{ i \ne s_b : R_{i,b}(g) \ge R_{s_b,b}(g) - \Delta,\ R_{i,b}(g) > T_{\text{hole}} \}\right|$$
  counts overlapping co-band neighbours, summed over bands. Two carriers of the same sector are never neighbours.
- **Overlapping neighbours per covered tile** (↓): the mean of $N_{\text{ov}}(g)$ over tiles with $R_s(g) > T_{\text{hole}}$.
- **Cell-edge, median and mean estimated UE throughput** (↑): the 5th and 50th percentiles and the mean of $R_u$ over every UE report, unserved UEs counting 0 Mbps.

The evaluation reads every configuration on three of them, coverage rate, separation rate and median throughput, network-wide and per band (Section V-D).

### G. Optimization Problem

Let $G_{\text{cov}} = \{g : R_s(g) > T_{\text{hole}}\}$ be the covered tiles and $U$ the set of UE reports. With RSRP in linear power, three objectives are maximized:

$$
f_{\text{cov}}(\boldsymbol{\theta}) = \frac{|G_{\text{cov}}|}{|G|} = 1 - \text{HoleRate},
$$

$$
f_{\text{sep}}(\boldsymbol{\theta}) = \frac{1}{|G_{\text{cov}}|} \sum_{g \in G_{\text{cov}}} \prod_{b=1}^{B} \frac{R_{s_b,b}(g)}{R_{s_b,b}(g) + \sum_{i \ne s_b} R_{i,b}(g)},
$$

$$
f_{\text{thr}}(\boldsymbol{\theta}) = \frac{1}{|U|} \sum_{u \in U} \ln(1 + R_u), \quad R_u \text{ in Mbps}.
$$

The separation objective is a soft, co-band analogue of the overlap KPI: an equal-power rival halves a band's factor, and a band whose strongest sector does not exceed $T_{\text{hole}}$ contributes a factor of 1, leaving holes to the coverage objective. The throughput objective is a proportional-fair utility that rewards raising slow UEs more than fast ones and, by counting unserved UEs at zero rate, penalizes coverage loss under demand. Each objective is quantized to six significant digits before the search observes it, suppressing non-determinism from GPU accumulation order.

The problem is

$$
\max_{\boldsymbol{\theta} \in \Theta} \ \mathbf{F}(\boldsymbol{\theta}) = \left(f_{\text{cov}}, f_{\text{sep}}, f_{\text{thr}}\right),
$$

and a set of evaluated points $P$ is scored by its hypervolume [12] with respect to the origin, the natural floor of every objective:

$$
\mathrm{HV}(P) = \lambda\left(\{\mathbf{z} \in \mathbb{R}^3 : \mathbf{0} \le \mathbf{z} \le \mathbf{y} \text{ for some } \mathbf{y} \in P\}\right),
$$

where $\lambda$ is the Lebesgue measure. The recommended configuration of a run is the evaluated point with the largest hypervolume contribution; the remaining Pareto-optimal points, ordered by contribution, form a shortlist for the operator. No KPI is weighted into the objectives, but the coverage objective is by definition one minus the hole-rate KPI, and the separation objective is related to, though not identical with, the overlap KPIs.

## IV. Search Methods

The search looks for tilt vectors $\boldsymbol{\theta}$ that make the objective vector $\mathbf{F}(\boldsymbol{\theta})$ of Section III-G as good as possible. Four properties of that problem decide how it can be searched:

1. **Expensive evaluations.** Scoring one candidate ray-traces every sector on every band and re-serves every UE report (Section VI-I). Only a small budget of evaluations is affordable.
2. **Black box.** $\mathbf{F}$ has no closed form; it is known only at the points where it has been evaluated. Its thresholds and arg-max rules (the hole threshold, the strongest sector $s_b$, the serving rule) make it discontinuous, and the tilt lattice makes the domain discrete, so gradients are unavailable.
3. **High dimension.** There is one decision variable per sector-band pair, $d = NB$ in total.
4. **Conflicting objectives.** Coverage, separation and throughput pull the tilts in different directions, so there is no single best configuration, only a set of best trade-offs.

Two methods are compared. **Sobol random search** places its evaluations evenly over the search space without learning from them; it is the control. **MORBO** [6] learns a statistical model of $\mathbf{F}$ from the evaluations made so far and uses it to decide where to evaluate next. Section IV-A defines the concepts and the protocol both methods share; Sections IV-B and IV-C explain each method in turn — what it is, why it is used, and how it proceeds step by step.

### A. Shared Concepts and Protocol

**Pareto front.** Because the objectives conflict, configurations are compared by dominance. An objective vector $\mathbf{y}$ *dominates* $\mathbf{y}'$ when it is at least as good on every objective and strictly better on at least one:

$$
\mathbf{y} \succ \mathbf{y}' \iff y_j \ge y'_j \ \text{for all } j \ \text{and} \ y_j > y'_j \ \text{for some } j.
$$

The *Pareto front* of a set of evaluated points $P$ is the subset that no other point of $P$ dominates. Each point on the front is a trade-off that cannot be improved on one objective without losing on another.

**Hypervolume contribution and improvement.** The hypervolume $\mathrm{HV}(P)$ of Section III-G measures the region of objective space that $P$ dominates; a larger value means a front that is better, wider, or both. Two derived quantities measure the value of a single point:

$$
\mathrm{HVC}(\mathbf{y}; P) = \mathrm{HV}(P) - \mathrm{HV}(P \setminus \{\mathbf{y}\}), \qquad
\mathrm{HVI}(\mathbf{y}; P) = \mathrm{HV}(P \cup \{\mathbf{y}\}) - \mathrm{HV}(P).
$$

The *contribution* $\mathrm{HVC}$ looks backward: it is the volume that only $\mathbf{y}$ dominates, the loss if $\mathbf{y}$ were removed. It is zero for a dominated point and largest for a point that fills a sparse part of the front. The *improvement* $\mathrm{HVI}$ looks forward: it is the volume that a new point $\mathbf{y}$ would add to $P$, and zero if $P$ already dominates it. Both compare points on all objectives at once without weighting one objective against another. MORBO uses $\mathrm{HVC}$ to decide where to search and $\mathrm{HVI}$ to decide what to evaluate; both methods use $\mathrm{HVC}$ to rank their final front.

**Search space.** Both methods work in the unit cube $[0,1]^d$, where every coordinate has the same scale. A point $\mathbf{u}$ of the cube is mapped to a tilt vector by scaling each coordinate to the tilt bounds and rounding to the tilt lattice of step $\delta$ (Section III-D):

$$
\theta_m = \theta_{\min} + \delta \cdot \operatorname{round}\!\left(\frac{u_m\,(\theta_{\max} - \theta_{\min})}{\delta}\right), \qquad m = 1, \dots, d.
$$

**Protocol.** Both methods spend the same budget of $N_{\text{eval}}$ evaluations after the incumbent, and both start from the same $n_0$ initial points. Every candidate is scored by the same ray-traced simulator with one fixed solver seed, so all candidates share one Monte-Carlo noise realization and differences between candidates are not masked by solver noise. A run proceeds as follows:

1. Evaluate the incumbent configuration $\boldsymbol{\theta}^{(0)}$.
2. Evaluate the $n_0$ points of the initial design.
3. Evaluate the $N_{\text{eval}} - n_0$ points the method proposes.
4. Return the Pareto front of all evaluations, ordered by $\mathrm{HVC}$; the point with the largest $\mathrm{HVC}$ is the recommendation and the rest form the shortlist (Section III-G).

Every reported quantity is a ray-traced evaluation; no model prediction enters any reported result.

### B. Sobol Random Search

**What it is.** Random search evaluates points chosen without regard to earlier results [14]. Here the points come from a Sobol sequence [15], a deterministic *low-discrepancy* sequence: its first $n$ points cover $[0,1]^d$ more evenly than $n$ independent uniform draws, in the sense that the fraction of points that falls in any axis-aligned box stays close to the volume of that box. *Scrambling* randomizes the sequence under a seed while keeping this evenness.

**Why it is used.** It is the control. Since it ignores what earlier evaluations revealed, any advantage MORBO shows over it at the same budget measures the value of learning from the evaluations. It is also a meaningful baseline rather than a straw man: when only a few of the $d$ coordinates matter, every random draw still varies all of them, which makes random search hard to beat in high dimensions [14]. The Sobol sequence is preferred over independent draws because it is reproducible from its seed and any shorter draw is a prefix of a longer one, so the first $n_0$ points can be shared exactly with MORBO's initial design. Its evenness guarantee is asymptotic, and weak when the budget is small relative to the dimension.

**How it is applied.**

1. Draw the first $N_{\text{eval}}$ points of a scrambled Sobol sequence over $[0,1]^d$ with the run's seed.
2. Map each point to a tilt vector (Section IV-A) and evaluate it.
3. Label the first $n_0$ points the initial design; they are identical to MORBO's.
4. Return the Pareto front of all evaluations, ordered by $\mathrm{HVC}$.

Because the two methods share the initial design, budget and seed, a difference between them does not stem from the starting points. The difference combines MORBO's models, trust region and candidate generation together.

### C. MORBO

**What it is.** MORBO (multi-objective Bayesian optimization over high-dimensional search spaces) [6] combines three ideas:

- **Bayesian optimization** [16]. A probabilistic *surrogate* model is fitted to the evaluations made so far. It predicts each objective at any untried point together with the uncertainty of that prediction, and an *acquisition rule* uses both to choose the next point to evaluate. Each new evaluation refines the model, and the loop repeats.
- **Trust regions**, from TuRBO [5]. Rather than modelling and searching the whole space at once, the search is confined to a box around the best point found so far. The box shrinks when it stops producing gains and is restarted elsewhere when it becomes too small.
- **Hypervolume-based Thompson sampling** [7], [13]. A batch of points is chosen to maximize the hypervolume improvement under a random draw from the surrogate, so that all objectives are improved together rather than through a fixed weighting.

**Why it is used.** Each component answers one of the four properties above. A surrogate lets every costly evaluation inform where to look next, which random search cannot do (properties 1 and 2). A single surrogate over the whole $d$-dimensional box needs far more data than a small budget provides, and its acquisition tends to favour the boundary of the box, where uncertainty is largest; confining the search to a trust region keeps the candidates where the model is informed, which TuRBO [5] showed restores sample efficiency in high dimensions (property 3). A weighted sum of the objectives would fix the trade-off between coverage, separation and throughput before the search begins; hypervolume-based selection keeps all objectives and returns a front from which the operator chooses (property 4).

**How it works.** The building blocks are described first, then the complete procedure.

**1) Surrogate model: Gaussian processes.** A Gaussian process (GP) treats an unknown function $f$ as random, such that its values at any finite set of inputs are jointly Gaussian [17]. It is specified by a mean $c$ and a *kernel* $k(\mathbf{u}, \mathbf{u}')$, the covariance between the function's values at two inputs: nearby inputs are expected to have similar values. Given observations $\mathbf{y}$ at inputs $\mathbf{u}_1, \dots, \mathbf{u}_n$, the prediction at a new input $\mathbf{u}$ is Gaussian with mean and variance

$$
\mu_n(\mathbf{u}) = c + \mathbf{k}_n(\mathbf{u})^\top \left(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I}\right)^{-1} (\mathbf{y} - c\mathbf{1}), \qquad
\sigma_n^2(\mathbf{u}) = k(\mathbf{u}, \mathbf{u}) - \mathbf{k}_n(\mathbf{u})^\top \left(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I}\right)^{-1} \mathbf{k}_n(\mathbf{u}),
$$

where $\mathbf{K}_{st} = k(\mathbf{u}_s, \mathbf{u}_t)$ is the covariance among the observed inputs, $[\mathbf{k}_n(\mathbf{u})]_t = k(\mathbf{u}, \mathbf{u}_t)$ the covariance between the new input and each observed one, and $\sigma_\varepsilon^2$ the observation noise. The mean $\mu_n$ is the best guess of the objective; the variance $\sigma_n^2$ is small near observed points and grows away from them. One independent GP is fitted to each objective. The kernel is the Matérn-5/2 kernel with automatic relevance determination (ARD):

$$
k(\mathbf{u}, \mathbf{u}') = \sigma_f^2 \left(1 + \sqrt{5}\,r + \tfrac{5}{3} r^2\right) e^{-\sqrt{5}\,r}, \qquad r^2 = \sum_{j=1}^{d} \frac{(u_j - u'_j)^2}{\ell_j^2}.
$$

The lengthscale $\ell_j$ states how far tilt $j$ must move before an objective changes appreciably: a short lengthscale marks a sector-band to which the objective is sensitive, a long one a sector-band that barely matters. ARD learns one lengthscale per dimension from the data. With few observations and many lengthscales, this estimate is ill-posed, so the lengthscales carry a dimension-scaled log-normal prior [18],

$$
\ln \ell_j \sim \mathcal{N}\!\left(\sqrt{2} + \tfrac{1}{2}\ln d,\ 3\right),
$$

whose median grows as $\sqrt{d}$: the prior assumes that an objective varies slowly along most coordinates unless the data show otherwise. The hyperparameters $c$, $\sigma_f$, $\ell_j$ and $\sigma_\varepsilon$ are estimated by maximizing the marginal likelihood of the observations under this prior. The prior departs from the original MORBO configuration; it was adopted after GP fitting failed on the ray-traced objectives without it. Those failed runs are not reported here.

Each GP is fitted to the evaluations near the trust region — those inside a box of twice its side, supplemented with the nearest evaluations when these are too few — and all evaluations are shared across trust regions. When the budget is small relative to the dimension, this training set covers most of the evaluations; the locality of the search then comes from the trust region, which bounds the candidates, rather than from the training data.

**2) Trust region.** The search is confined to a hypercube of side $L$ around a centre $\mathbf{c}$:

$$
\mathcal{T}(\mathbf{c}, L) = \left\{\mathbf{u} \in [0,1]^{d} : |u_j - c_j| \le L/2 \ \text{for every } j\right\}.
$$

The centre is the evaluated Pareto point with the largest $\mathrm{HVC}$, so the search refines the part of the front that contributes most. The side starts at $L_0$ and only shrinks, as described in block 5.

**3) Candidate generation.** In high dimensions, almost all of a box's volume lies near its boundary, so uniform samples of the trust region would fall far from the good points at its centre. Candidates are therefore made by changing only a few tilts of a good configuration:

1. Pick a Pareto point inside the trust region at random as the *base*.
2. Select each coordinate independently with probability $p_n$.
3. Replace each selected coordinate by a quasi-random value within the trust region; keep the others from the base.
4. Repeat until $n_c$ candidates are generated.

The selection probability decreases as the budget is spent:

$$
p_n = \min\!\left(\frac{\kappa}{d},\ 1\right) \rho_n, \qquad \rho_n = 1 - \alpha\,\frac{\log(n - n_0 + 1)}{\log(N_{\text{eval}} - n_0 + 1)},
$$

where $n$ is the number of evaluations made so far, $\kappa$ the expected number of coordinates changed at the start, and $\alpha \in (0, 1)$ the fraction by which that number has fallen when the budget ends. Early candidates change many tilts at once (exploration); later ones change few (refinement).

**4) Acquisition: Thompson sampling of the hypervolume improvement.** Thompson sampling [13] makes a decision under uncertainty by drawing one plausible version of the unknown function from the model and acting as if that draw were true. Where the model is confident, the draws agree and the rule exploits; where it is uncertain, the draws vary and the rule explores. Here a joint sample $\tilde{\mathbf{f}}$ of all objectives is drawn from the GP posterior over the candidates $X_{\text{cand}}$ and over the points $X_{\text{pend}}$ already chosen for the current batch, and the next point is the candidate whose sampled objectives would add the most hypervolume:

$$
\mathbf{u}^\star = \arg\max_{\mathbf{u} \in X_{\text{cand}}} \mathrm{HVI}\!\left(\tilde{\mathbf{f}}(\mathbf{u});\ P^\ast \cup \tilde{\mathbf{f}}(X_{\text{pend}})\right),
$$

where $P^\ast$ is the observed Pareto front. Including the sampled values of the pending points means that a second point in the batch is not chosen for the same improvement as the first. A batch of $q$ points is built by repeating this choice $q$ times with fresh candidates; evaluating a batch amortizes one round of model fitting over several evaluations. Because the rule is evaluated on a finite set of candidates from one posterior draw, it requires neither a continuous optimization over $d$ dimensions nor an integral of the expected hypervolume improvement [7].

**5) Trust-region adaptation and restart.** The side of the trust region adapts to the progress of the search:

- A batch is a *success* when it raises the hypervolume by more than a relative margin $\epsilon$, $\mathrm{HV}(P \cup Y_{\text{new}}) > (1 + \epsilon)\,\mathrm{HV}(P)$; a success resets the failure count.
- Otherwise each of its evaluations counts as a failure. After $\tau$ consecutive failures, a tolerance that [6] scales with the dimension, the side halves, $L \leftarrow L/2$, concentrating the search around a centre that has stopped yielding gains. Following [6], the side never grows.
- When the side falls below $L_{\min}$, the region is exhausted and restarts. Its former centre is barred from serving as a centre for a fixed number of rounds, so the search does not return to it at once.

The restart centre is chosen over the whole cube. A GP fitted to the initial design and earlier restart points is sampled once over quasi-random points of $[0,1]^d$, and the point is taken that maximizes a *random hypervolume scalarization* [19] of its sampled objectives together with those data,

$$
s_{\mathbf{w}}(Y) = \max_{\mathbf{y} \in Y} \min_{j} \left(\frac{\max(y_j, 0)}{w_j}\right)^{M},
$$

with $M$ the number of objectives and $\mathbf{w}$ drawn uniformly on the positive unit sphere. For a fixed $\mathbf{w}$, $s_{\mathbf{w}}$ measures how far the set reaches along the direction $\mathbf{w}$; averaged over all directions, it is proportional to the hypervolume. A single random direction is thus a cheap proxy for the hypervolume that still rewards different parts of the front on different restarts. The restarted region takes the side $L = L_{\min} + (L_0 - L_{\min})\,\rho_n$, smaller the later in the budget the restart occurs.

**Complete procedure (Algorithm 1).**

1. **Initialize.** Evaluate the incumbent $\boldsymbol{\theta}^{(0)}$ and the $n_0$ points of the Sobol initial design shared with random search.
2. **Place the trust region.** Centre it on the Pareto point with the largest $\mathrm{HVC}$, with side $L_0$.
3. **Iterate** until $N_{\text{eval}}$ evaluations, the initial design included, have been spent:
   1. *Restart if exhausted.* If $L < L_{\min}$, choose a new centre by the random scalarization (block 5), evaluate it, reset $L$, and begin the next iteration.
   2. *Re-centre.* Move the centre to the current Pareto point with the largest $\mathrm{HVC}$ (block 2).
   3. *Fit the models.* Fit one GP per objective to the evaluations near the trust region (block 1).
   4. *Propose a batch.* For each of $q$ points, generate $n_c$ candidates by perturbing Pareto points (block 3), draw one posterior sample, and keep the candidate with the largest sampled $\mathrm{HVI}$ (block 4).
   5. *Evaluate.* Map the batch to the tilt lattice and ray-trace it.
   6. *Adapt.* Update the failure count and halve $L$ after $\tau$ consecutive failures (block 5).
4. **Return** the Pareto front of all evaluations, ordered by $\mathrm{HVC}$.

**Hyperparameters.** The procedure is governed by the initial-design size $n_0$, the budget $N_{\text{eval}}$, the batch size $q$, the number of candidates $n_c$, the initial and minimum sides $L_0$ and $L_{\min}$, the success margin $\epsilon$, the failure tolerance $\tau$, the expected number of perturbed coordinates $\kappa$ and its decay $\alpha$, and the number of trust regions. Algorithm 1 is stated for one trust region; with several, each follows blocks 2–5 independently while sharing all evaluations.

## V. Experimental Methodology

### A. Scenario Summary

*Table II. Scenario parameters.*

| Property | Value |
|---|---|
| Grid | 326 × 310 tiles of 20 m (6,200 × 6,520 m) |
| Sites / sectors / sector-band pairs | 5 / 15 / 45 |
| Layout | Four corner sites of an axis-aligned square and one at its centre |
| Centre-to-corner distance | 1,732 m (corner-to-corner 2,449 m) |
| Mast height | 25 m (UMa, TR 38.901 [9]) |
| Sector azimuths | 0°, 120°, 240° at every site |
| Incumbent tilt (2600 / 1800 / 700 MHz) | 12° / 12° / 12° (UMa calibration, TR 38.901 [9]) |
| Tilt bounds and resolution | [0°, 20°], 0.1° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 3, holding 70 % of UEs on average |
| UE reports | 10,087 |
| UE reports with no path to any sector | 0.4 % |
| KPI thresholds | $T_{\text{hole}} = -110$ dBm, $T_{\text{weak}} = -90$ dBm, $\Delta = 6$ dB |
| Search seed | 42 (both methods) |

Each band reaches 97.8 % (700 MHz), 96.1 % (1800 MHz) and 96.2 % (2600 MHz) of the tiles through at least one path, with a median RSRP over the reached tiles of −87.0, −94.2 and −98.3 dBm. 3GPP fixes no coverage-hole threshold: TS 37.320 [20] defines a hole by the signal level needed for basic service without quantifying it, so $T_{\text{hole}}$ is a choice of this study. The site spacing is likewise a choice of this study rather than a 3GPP value.

### B. Characterization of the Incumbent Configuration

At the incumbent tilts, 6.7 % of tiles are holes, 32.7 % weak and 60.6 % well covered (Table III). The holes are fragmented into 1,733 connected regions, 1,250 of them single tiles; the largest holds 11 % of the hole area, and 1,314 of the 6,813 hole tiles receive no propagation path from any sector. The hole rate is 0.6 % within 1 km of a site and 10.4 % beyond, so most of the hole area lies at the corners of the scene, beyond the square.

Per layer, the hole share is 7.9 % at 700 MHz, 17.5 % at 1800 MHz and 24.1 % at 2600 MHz. Although 700 MHz is the strongest band on 94.7 % of the covered area, the serving rule places 80.1 % of that area on 2600 MHz, the band with the most PRBs. Of all UE reports, 65.2 % are served on 2600 MHz, 16.5 % on 1800 MHz and 14.4 % on 700 MHz. Co-band overlap affects 34.4 % of tiles, with 0.99 overlapping neighbours per covered tile on average (median 0, 90th percentile 3); 19.0 % of covered tiles have three or more, at a median distance of 1.0 km from the nearest site.

Demand and coverage are aligned better than area alone suggests: holes occupy 6.7 % of the area but carry 3.9 % of UE reports (Table III), because the three demand hotspots, placed where building volume is high, are centred 0.65 to 1.23 km from their nearest site. Each lies within 1.6 to 2.3 km of the others.

*Table III. Coverage class by area and by demand at the incumbent configuration.*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole ($\le -110$ dBm) | 6,813 | 6.7 % | 3.9 % |
| Weak ($-110$ to $-90$ dBm) | 33,037 | 32.7 % | 25.1 % |
| Good ($> -90$ dBm) | 61,210 | 60.6 % | 71.0 % |

![RSRP per band](../reports/figures/00_simulation/rsrp_per_band.png)

*Fig. 2. Best-server RSRP per band at the incumbent tilts.*

![Demand vs coverage](../reports/figures/01_eda/demand_vs_coverage.png)

*Fig. 3. UE demand alongside signal strength at the incumbent tilts.*

### C. Data Verification

Prior to optimization, the UE reports, the scenario description and the radio map at the incumbent tilts were subjected to 28 consistency checks covering data types, value ranges, grid consistency, band and frequency agreement, physical plausibility (no RSRP above the transmitted reference-signal power) and the absence of duplicates; all 28 held. No UE report was removed or altered. Because GPU ray tracing is not bit-reproducible, the radio map at the incumbent tilts is computed once and reused wherever the incumbent is assessed, so every comparison against the incumbent in Section VI refers to a single ray-traced realization.

### D. Evaluation KPIs

The searches maximize the three objectives of Section III-G. Following the coverage and capacity optimization literature, which reports the coverage–capacity trade-off as a Pareto front for an operator to choose from [4], [23], every evaluated configuration is additionally read on three KPIs, one per concern:

- **Coverage rate** $= 1 - \text{HoleRate}$, the share of tiles some sector-band covers above $T_{\text{hole}}$; it equals $f_{\text{cov}}$.
- **Separation rate** $= 1 - \text{OverlapRate}$, the share of tiles where no band has a co-band rival within $\Delta$ of its strongest sector. Unlike $f_{\text{sep}}$, it is hard and band-collapsed: a tile counts once as soon as any band is crowded.
- **Median throughput**, the median of $R_u$ over every UE report, unserved reports counting 0 Mbps.

All three are maximized. Their hypervolume is taken against the origin. Each of the three is also recorded per band for every evaluation, coverage and separation from the band's own layers and the median throughput over the reports served on the band, together with the band's share of the reports.

Each method's **chosen configuration** is, among the configurations it proposed, the one with the largest hypervolume contribution on the three KPIs. The incumbent is excluded from that choice: it is evaluation 0 of every run rather than a search result, and on this scenario it alone holds the top separation rate (Section VI-C), so it would otherwise be chosen for both methods. The combined Pareto front of both methods on the three KPIs is published with every sector-band tilt, so that an engineer can choose a configuration rather than accept one.

### E. Fairness of the Comparison

Both methods spend one incumbent evaluation, the same eight scrambled Sobol points and 64 further evaluations, 73 ray traces in all; MORBO's restart points and its last, shortened batch count against the same 64, so its batch size does not change the total. Sobol draws from one seed are prefixes of each other, so MORBO's initial design is exactly random search's first eight points. Both are scored by the same simulator under one solver seed, on one tilt lattice and with identical KPI definitions. MORBO's model fitting and acquisition are not part of the budget, which counts ray traces, and are reported as cost. Section VI-A checks each of these properties on the runs themselves.

### F. Assessment Criteria

1. **Search effectiveness:** hypervolume of each run on the objectives and on the three KPIs, and how much each method adds beyond the shared initial design.
2. **Statistical comparison of the methods:** whether the searched candidates of MORBO score higher than those of random search on each KPI, by a one-sided Mann–Whitney U test [24] with Cliff's δ as effect size [25], [26], the non-parametric pairing recommended for comparing optimizers [27]; the shared design is left out, as both runs evaluated it alike. The fronts are compared by the C-metric (set coverage) [12], [28], which needs no reference point, and by the number of candidates that dominate the incumbent.
3. **Network KPIs** of each chosen configuration against the incumbent.
4. **Per-layer, spatial and temporal effects:** the three KPIs per band, coverage and throughput maps, and the 5th percentile, median and mean throughput per 15-minute interval against the number of UEs in it.
5. **Cost:** ray-tracing and wall-clock time per run.

The evaluation first verifies that both runs used identical scenario, propagation, antenna, UE-height and KPI settings and the same sector PRB limits. The incumbent is then assessed on its single radio map (Section V-C), each chosen configuration is ray-traced again, and their KPIs are compared with the values recorded during the search.

## VI. Results and Analysis

### A. Comparability, Budget and Reproducibility

All eight comparability checks hold. Both runs evaluated 73 configurations, split identically into the incumbent, eight Sobol points and 64 search evaluations; MORBO never restarted its trust region. The shared design's tilt vectors are identical in both runs, and its measures differ by at most $9.1 \times 10^{-7}$, GPU ray-tracing non-determinism. The incumbent and the two re-traced chosen configurations reproduce the recorded measures to within $3.8 \times 10^{-6}$ (median RSRP, in dBm), so the maps analysed below are the configurations the searches scored.

### B. Search Effectiveness

*Table IV. Hypervolume against the origin, on the search objectives and on the three evaluation KPIs.*

| Method | Measures | Incumbent | Initial design | All evaluations | Pareto points |
|---|---|---:|---:|---:|---:|
| MORBO | Objectives | 2.540 | 2.662 | **2.857** | 3 |
| Random search | Objectives | 2.540 | 2.662 | 2.765 | 10 |
| MORBO | Coverage, separation, median throughput | 36.92 | 42.71 | **52.54** | 8 |
| Random search | Coverage, separation, median throughput | 36.92 | 42.71 | 47.05 | 13 |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Fig. 4. Hypervolume of the search objectives versus the number of evaluations.*

From the shared design, MORBO adds 0.195 of hypervolume on the objectives and 9.83 on the KPIs; random search adds 0.103 and 4.34. MORBO passes random search's final value on both measure sets at evaluation 36 and is still improving at evaluation 72, so the budget of 73 evaluations in 45 dimensions did not exhaust its progress. Its 64 proposals average a coverage objective of 0.9363, a separation objective of 0.6852 and a throughput objective of 4.220, against 0.9329, 0.6739 and 4.075 for the Sobol design; all three points of its front on the objectives are trust-region proposals, and 51 of its candidates beat the incumbent on all three objectives, against 10 of random search's.

### C. KPI Trade-offs

![Coverage vs separation](../reports/figures/04_evaluation/tradeoff_coverage_rate_vs_separation_rate.png)

*Fig. 5. Every evaluated configuration on coverage rate and separation rate. Outlined points lie on their method's three-KPI front; stars mark each method's chosen configuration and the cross the incumbent.*

![Coverage vs throughput](../reports/figures/04_evaluation/tradeoff_coverage_rate_vs_estimated_throughput_p50_mbps.png)

*Fig. 6. Coverage rate against median throughput.*

![Separation vs throughput](../reports/figures/04_evaluation/tradeoff_separation_rate_vs_estimated_throughput_p50_mbps.png)

*Fig. 7. Separation rate against median throughput.*

*No candidate of either method raises the separation rate.* The incumbent sits alone at the top of the separation axis (0.6563); the best candidates reach 0.6246 (MORBO) and 0.6161 (random search). Every candidate trades separation rate for coverage rate and median throughput: 64 of MORBO's 72 candidates and 37 of random search's beat the incumbent's coverage rate, and 71 and 69 its median throughput. Consequently neither method found a configuration that dominates the incumbent on all three KPIs.

*The soft objective and the hard KPI disagree.* MORBO raised the separation objective above the incumbent's in 54 of its candidates (to 0.6997 at its chosen configuration), while the band-collapsed overlap rate rose in every candidate of both methods. The objective prices each rival by its linear power share per band; the KPI counts a tile as soon as any band has a rival within 6 dB. Section VI-E shows that the chosen configuration lowers every band's own overlap, so the disagreement lies in how bands are combined.

*Chosen configurations.* MORBO's choice, evaluation 71, a trust-region proposal, raises the coverage rate from 0.9326 to 0.9416 and the median throughput from 60.3 to 84.9 Mbps at a separation rate of 0.6237. It is also MORBO's largest hypervolume contribution on the search objectives. Random search's choice, evaluation 2, a point of the shared Sobol design, reaches 0.9329, 0.6161 and 67.7 Mbps.

### D. Statistical Comparison of the Methods

*Table V. MORBO's 64 searched candidates against random search's 64, the shared design excluded.*

| KPI | Median, MORBO | Median, random | One-sided Mann–Whitney p | Cliff's δ |
|---|---:|---:|---:|---|
| Coverage rate | 0.9366 | 0.9328 | $5.0 \times 10^{-5}$ | 0.40 (medium) |
| Separation rate | 0.6050 | 0.5817 | $2.4 \times 10^{-15}$ | 0.80 (large) |
| Median throughput [Mbps] | 77.4 | 68.9 | $1.1 \times 10^{-9}$ | 0.61 (large) |

On every KPI a MORBO candidate tends to score higher than a random-search candidate, with effect sizes from medium to large under the thresholds of [26]. The fronts agree: MORBO's three-KPI front weakly dominates 76.9 % of random search's ($C = 0.769$), random search's only 12.5 % of MORBO's ($C = 0.125$). The candidates of one MORBO run are not independent draws, since the trust region concentrates them, so these tests describe the two runs rather than the methods in general.

### E. Network-Level and Per-Layer KPIs

*Table VI. Network KPIs of each chosen configuration. Bold marks the best value per row.*

| KPI | Direction | Incumbent | MORBO (eval. 71) | Random search (eval. 2) |
|---|:-:|---:|---:|---:|
| Coverage rate | ↑ | 0.9326 | **0.9416** | 0.9329 |
| Separation rate | ↑ | **0.6563** | 0.6237 | 0.6161 |
| Weak-coverage rate | ↓ | 32.7 % | **22.6 %** | 29.4 % |
| Overlapping neighbours per covered tile | ↓ | 0.99 | **0.88** | 0.97 |
| Cell-edge throughput, p05 [Mbps] | ↑ | 8.2 | **12.9** | 10.4 |
| Median throughput [Mbps] | ↑ | 60.3 | **84.9** | 67.7 |
| Mean throughput [Mbps] | ↑ | 88.7 | **111.7** | 99.2 |
| Median best-server SINR [dB] | ↑ | 11.5 | 15.2 | **15.5** |
| Coverage objective $f_{\text{cov}}$ | ↑ | 0.9326 | **0.9416** | 0.9329 |
| Separation objective $f_{\text{sep}}$ | ↑ | 0.6800 | **0.6997** | 0.6843 |
| Throughput objective $f_{\text{thr}}$ | ↑ | 4.005 | **4.318** | 4.116 |

*Table VII. The three KPIs per band, and each band's share of UE reports.*

| Band | Configuration | Coverage rate | Separation rate | Median throughput [Mbps] | Share of reports |
|---|---|---:|---:|---:|---:|
| 2600 MHz | Incumbent | 0.759 | 0.794 | 79.9 | 65.2 % |
| | MORBO | 0.758 | 0.824 | 106.7 | 62.1 % |
| | Random search | 0.756 | 0.813 | 88.7 | 62.4 % |
| 1800 MHz | Incumbent | 0.825 | 0.761 | 59.6 | 16.5 % |
| | MORBO | 0.845 | 0.792 | 79.3 | 22.0 % |
| | Random search | 0.836 | 0.782 | 70.4 | 20.3 % |
| 700 MHz | Incumbent | 0.921 | 0.722 | 32.2 | 14.4 % |
| | MORBO | 0.925 | 0.755 | 46.5 | 13.1 % |
| | Random search | 0.911 | 0.746 | 39.0 | 13.7 % |

![KPIs per band](../reports/figures/04_evaluation/band_kpi_panels.png)

*Fig. 8. The three KPIs per band, for the incumbent and each chosen configuration.*

*Coverage and capacity rise together.* MORBO's choice lowers the hole rate from 6.7 % to 5.8 % and the weak rate from 32.7 % to 22.6 %, and raises the 5th-percentile, median and mean throughput by 56 %, 41 % and 26 %.

*Every band separates better, the network does not.* The chosen configuration raises each band's own separation rate, by 0.030 to 0.033, and lowers the overlapping neighbours per covered tile from 0.99 to 0.88, yet the band-collapsed separation rate falls from 0.656 to 0.624. The tiles still crowded on one band therefore coincide less with those crowded on another, so their union grows. A planner reading only the band-collapsed KPI would see a loss where each layer gained.

*Traffic moves onto 1800 MHz.* Its share of UE reports rises from 16.5 % to 22.0 %, while 2600 MHz falls from 65.2 % to 62.1 % and 700 MHz from 14.4 % to 13.1 %; the unserved share falls from 3.9 % to 2.8 %. The median throughput of the reports each band serves rises on every band, by 33 % to 44 %.

### F. Spatial Effects

*Table VIII. Coverage class by area and by demand (share of UE reports).*

| Coverage class | Incumbent: area | Incumbent: demand | MORBO: area | MORBO: demand | Random: area | Random: demand |
|---|---:|---:|---:|---:|---:|---:|
| Hole | 6.7 % | 3.9 % | 5.8 % | 2.8 % | 6.7 % | 3.6 % |
| Weak | 32.7 % | 25.1 % | 22.6 % | 14.1 % | 29.4 % | 22.5 % |
| Good | 60.6 % | 71.0 % | 71.5 % | 83.0 % | 63.9 % | 73.9 % |

![Coverage classes](../reports/figures/04_evaluation/coverage_class_maps.png)

*Fig. 9. Coverage classes at the incumbent and at MORBO's chosen configuration.*

![Coverage before and after, all bands](../reports/figures/04_evaluation/coverage_before_after_all.png)

*Fig. 10. Best-server RSRP over all bands at the incumbent (left) and at MORBO's chosen configuration (centre), and the tiles that cross the hole threshold (right): 1,331 holes close and 419 open.*

![Throughput change](../reports/figures/04_evaluation/ue_throughput_change_map.png)

*Fig. 11. Change in estimated UE throughput under MORBO's chosen configuration: the median, over the UE reports in each 150 m hexagon, of each report's change; hexagons with fewer than five reports are left blank.*

Over all bands, 1,331 hole tiles are closed and 419 opened (Fig. 10). The weak area that turns good lies mostly between the five sites, and the remaining holes are at the corners of the scene, beyond the square's reach. By UE reports the good share rises from 71.0 % to 83.0 %, more than by area (60.6 % to 71.5 %). The largest throughput gains sit on the demand hotspot south of the centre site and the one west of the south-west site; the hotspot between the centre and the north-west site shows both gains and losses, and the losses elsewhere are scattered single tiles of the sparse background.

### G. Temporal Effects and UE Density

![Throughput by time of day](../reports/figures/04_evaluation/interval_throughput_plot.png)

*Fig. 12. Estimated UE throughput by time of day. Every UE report is assigned to its 15-minute slot with the seven days pooled; lines show the median, shaded bands the interquartile range and dotted lines the 5th percentile of each slot, and the dashed line the mean number of UEs per interval (right axis).*

![Throughput vs load](../reports/figures/04_evaluation/throughput_vs_load.png)

*Fig. 13. Interval median throughput against the number of UEs in the interval.*

![Throughput CDF](../reports/figures/04_evaluation/throughput_cdf.png)

*Fig. 14. Estimated throughput per UE report, unserved reports at 0 Mbps.*

MORBO's chosen configuration gives a higher interval median in 608 of the 672 intervals, by 24.9 Mbps at the median interval, a higher interval mean in 650 and a higher interval 5th percentile in 506, with 44 ties. Throughput falls as an interval fills, more steeply under MORBO: the interval median correlates with the UE count at −0.26 at the incumbent and −0.40 under MORBO. With 10 UEs the mean interval median is 79.3 Mbps at the incumbent and 116.9 Mbps under MORBO; with 20, 56.3 and 75.7 Mbps, so the gain shrinks from 47 % to 34 % as the interval fills but persists at the highest load observed. The mean UE count per time-of-day slot varies between about 11 and 18 with no clear diurnal pattern at this population size.

### H. Pareto Tilt Configurations

The combined three-KPI front of both runs holds eleven configurations: the incumbent, seven of MORBO's and three of random search's. The incumbent ranks first by hypervolume contribution, because it alone holds the top separation rate. Every other row trades 0.032 to 0.082 of separation rate for 0.004 to 0.012 of coverage rate and 3.1 to 26.0 Mbps of median throughput. MORBO's evaluation 71 ranks second. Random search's evaluation 35 has the highest coverage rate (0.9444 at a separation rate of 0.5884), and MORBO's evaluation 66 the highest median throughput (86.3 Mbps). The front is published with every sector-band tilt of each configuration.

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Fig. 15. Tilt change per sector and band in MORBO's chosen configuration.*

![Tilt change bars](../reports/figures/04_evaluation/tilt_change_bars.png)

*Fig. 16. Tilt change of every sector-band in MORBO's chosen configuration, grouped by sector. Negative values denote an uptilt.*

*Table IX. Tilt movement of MORBO's chosen configuration. Negative $\Delta$ denotes an uptilt.*

| Band | Sectors moved | Mean $\lvert\Delta\rvert$ [°] | Largest $\lvert\Delta\rvert$ [°] | Mean $\Delta$ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 14 of 15 | 6.95 | 11.8 | −4.19 |
| 1800 MHz | 15 of 15 | 6.37 | 12.0 | −3.66 |
| 700 MHz | 15 of 15 | 5.72 | 9.8 | −3.01 |

44 of the 45 tilts change (Fig. 16), 31 upward and 13 downward, to between 0.0° and 19.6°. Every layer is uptilted on net, the capacity layer most, which is consistent with the gain in coverage and in served SINR at the expense of co-band containment. The largest change takes the 1800 MHz carrier of the centre site's third sector from 12° to the 0° bound. Because movement is unpenalized, the configuration corresponds to a 44-antenna RET change request.

### I. Computational Cost

*Table X. Search cost on a single 4 GB laptop-class GPU (NVIDIA RTX 3050).*

| Method | Evaluations | Ray tracing [min] | Overhead [min] | Wall clock [min] |
|---|---:|---:|---:|---:|
| MORBO | 73 | 3.18 | 2.45 | 5.63 |
| Random search | 73 | 3.94 | 0.48 | 4.43 |

Ray tracing costs 2.6 s per full-network evaluation in MORBO's run and 3.2 s in random search's; the cost depends on the configurations evaluated. MORBO spends a further 2.4 min on GP fitting and acquisition. A complete joint optimization of 45 tilts therefore takes about six minutes of commodity compute in this setting, before any field validation.

## VII. Discussion

### A. Implications for RAN Operation

Subject to the limitations below, the results suggest three changes to tilt-optimization practice. First, the trade-off between coverage, co-band interference and capacity is real even in a sparse five-site layout: no evaluated configuration improved all three, so a single recommended configuration hides a choice that belongs to the operator. Publishing the front with its tilts turns that choice into an explicit one. Second, a band-collapsed interference KPI can report a loss where every layer gained; per-layer reporting is needed to see what a joint multi-band change does. Third, the throughput gains concentrate where the demand is and persist at the highest interval load observed, which is the operating condition a capacity-oriented change has to survive.

### B. Limitations and Threats to Validity

- **Single, uncalibrated scenario.** One scene, one UE realization and one layout were studied; no result has been validated on a held-out scenario or calibrated against drive-test measurements. The simulator is not a digital twin of any deployed network.
- **Layout choice.** The 1,732 m centre-to-corner spacing has no 3GPP source, and five sites leave the corners of the 6.2 × 6.5 km scene out of any tilt's reach.
- **Propagation settings.** Diffraction and diffuse scattering are disabled, which can turn shadowed tiles into no-path holes.
- **Antenna model.** The simulator tilts the whole array, side and back lobes included, whereas 3GPP specifies the 12° incumbent as an electrical downtilt. The $8 \times 8$ array with uniform weights has its first vertical null about 14.5° off boresight, so results for sectors tilted into roughly 14° to 18° partly reflect the antenna model.
- **Objective–KPI mismatch.** The searches maximized a soft per-band separation objective, the evaluation reads a hard band-collapsed separation rate; neither method searched the evaluation KPIs directly.
- **Chosen configuration.** It is the best of 72 proposed configurations, so its scores are biased upward, and the choice depends on the hypervolume reference point and on excluding the incumbent.
- **Dependent candidates.** The statistical tests treat each run's candidates as samples; MORBO's trust region makes them dependent.
- **Limited baselines.** MORBO is compared only with random search and the uniform incumbent; no band-by-band procedure, operator-tuned configuration or further optimizer was evaluated.
- **Simplified capacity model.** Rates are equal-share Shannon bounds without scheduling, MCS limits or mobility; SINR assumes full co-band load and no receiver noise figure; inter-band interference is absent by construction. The serving rule is an idealization, so the inter-layer load results describe this rule.
- **Unpenalized movement.** The chosen configuration moves 44 of 45 tilts.

## VIII. Conclusion and Future Work

This paper formulated the joint configuration of tilts across all sectors and bands of a multi-band network as a 45-dimensional, three-objective black-box problem evaluated by an uncalibrated ray-traced network simulator, compared MORBO with Sobol random search at an equal, verified budget, and read every configuration on coverage rate, separation rate and median throughput.

1. **MORBO searches better at the same budget.** It reaches a larger hypervolume on the objectives (2.857 against 2.765) and on the three KPIs (52.54 against 47.05), passing random search's final value at evaluation 36, and its searched candidates score higher on every KPI (one-sided Mann–Whitney $p \le 5 \times 10^{-5}$, Cliff's δ 0.40 to 0.80). Its front weakly dominates 77 % of random search's, against 12.5 % the other way.
2. **Coverage and capacity rise together.** MORBO's chosen configuration raises the coverage rate from 0.9326 to 0.9416, the median estimated throughput from 60.3 to 84.9 Mbps and the cell-edge throughput from 8.2 to 12.9 Mbps, and the median throughput is higher in 608 of the 672 intervals.
3. **Band-collapsed separation is the price.** No evaluated configuration raises the separation rate above the incumbent's, although the chosen one raises every band's own separation rate. The trade-off is therefore reported as a front of eleven configurations with their tilts, from which an engineer chooses.

Future work will (i) add the separation rate, or a per-band form of it, to the searched objectives so that the search and the evaluation agree; (ii) raise the evaluation budget in proportion to the dimension; (iii) extend the layout to a 3GPP site grid with indoor UEs and model electrical tilt with the TR 38.901 antenna pattern; (iv) introduce a tilt-movement constraint; (v) compare against an operator-tuned configuration, a band-by-band procedure and further optimizers; and (vi) incorporate a scheduler-level capacity model and validate the approach against operator measurements.

---

## References

[1] 3GPP TR 36.902, *Evolved Universal Terrestrial Radio Access Network (E-UTRAN); Self-configuring and self-optimizing network (SON) use cases and solutions*, Release 9, clause 4.1.

[2] R. Razavi, S. Klein, and H. Claussen, "A fuzzy reinforcement learning approach for self-optimization of coverage in LTE networks," *Bell Labs Technical Journal*, vol. 15, no. 3, pp. 153–175, Dec. 2010, doi: 10.1002/bltj.20463.

[3] N. Dandanov, H. Al-Shatri, A. Klein, and V. Poulkov, "Dynamic self-optimization of the antenna tilt for best trade-off between coverage and capacity in mobile networks," *Wireless Personal Communications*, vol. 92, no. 1, pp. 251–278, 2017, doi: 10.1007/s11277-016-3849-9.

[4] R. M. Dreifuerst et al., "Optimizing coverage and capacity in cellular networks using machine learning," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021, pp. 8138–8142.

[5] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, "Scalable global optimization via local Bayesian optimization," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 5496–5507.

[6] S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy, "Multi-objective Bayesian optimization over high-dimensional search spaces," in *Proc. Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 180, pp. 507–517, 2022.

[7] S. Daulton, M. Balandat, and E. Bakshy, "Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization," in *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 2020, pp. 9851–9864.

[8] J. Hoydis, F. Aït Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus, and A. Keller, "Sionna RT: Differentiable ray tracing for radio propagation modeling," in *Proc. IEEE Globecom Workshops*, 2023.

[9] 3GPP TR 38.901 V19.2.0 (ETSI TR 138 901 V19.2.0, 2026-02), *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1 (evaluation parameters for UMa scenarios) and Table 7.8-1 (large-scale calibration parameters).

[10] 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.3.2-1.

[11] 3GPP TS 38.211, *NR; Physical channels and modulation*, clause 4.4.3.

[12] E. Zitzler and L. Thiele, "Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach," *IEEE Transactions on Evolutionary Computation*, vol. 3, no. 4, pp. 257–271, 1999.

[13] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, vol. 25, no. 3–4, pp. 285–294, 1933, doi: 10.1093/biomet/25.3-4.285.

[14] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281–305, 2012.

[15] I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals," *USSR Computational Mathematics and Mathematical Physics*, vol. 7, no. 4, pp. 86–112, 1967, doi: 10.1016/0041-5553(67)90144-9.

[16] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global optimization of expensive black-box functions," *Journal of Global Optimization*, vol. 13, no. 4, pp. 455–492, 1998.

[17] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. Cambridge, MA, USA: MIT Press, 2006.

[18] C. Hvarfner, E. O. Hellsten, and L. Nardi, "Vanilla Bayesian optimization performs great in high dimensions," in *Proc. International Conference on Machine Learning (ICML)*, PMLR 235, 2024.

[19] D. Golovin and Q. Zhang, "Random hypervolume scalarizations for provable multi-objective black box optimization," in *Proc. International Conference on Machine Learning (ICML)*, PMLR 119, 2020.

[20] 3GPP TS 37.320, *Radio measurement collection for Minimization of Drive Tests (MDT); Overall description; Stage 2*, Release 19, Annex A.

[21] V. Buenestado, M. Toril, S. Luna-Ramírez, J. M. Ruiz-Avilés, and A. Mendo, "Self-tuning of remote electrical tilts based on call traces for coverage and capacity optimization in LTE," *IEEE Transactions on Vehicular Technology*, vol. 66, no. 5, pp. 4315–4326, 2017, doi: 10.1109/TVT.2016.2605380.

[22] E. Tekgul, T. Novlan, S. Akoum, and J. G. Andrews, "Joint uplink-downlink capacity and coverage optimization via site-specific learning of antenna settings," *IEEE Transactions on Wireless Communications*, vol. 23, no. 5, pp. 4032–4048, 2024.

[23] M. Benzaghta, G. Geraci, D. López-Pérez, and A. Valcarce, "Cellular network design for UAV corridors via data-driven high-dimensional Bayesian optimization," arXiv:2504.05176, 2025.

[24] H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other," *Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50–60, 1947.

[25] N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," *Psychological Bulletin*, vol. 114, no. 3, pp. 494–509, 1993.

[26] J. Romano, J. D. Kromrey, J. Coraggio, and J. Skowronek, "Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys?," in *Annual Meeting of the Florida Association of Institutional Research*, 2006.

[27] J. Derrac, S. García, D. Molina, and F. Herrera, "A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms," *Swarm and Evolutionary Computation*, vol. 1, no. 1, pp. 3–18, 2011, doi: 10.1016/j.swevo.2011.02.002.

[28] E. Zitzler, L. Thiele, M. Laumanns, C. M. Fonseca, and V. Grunert da Fonseca, "Performance assessment of multiobjective optimizers: An analysis and review," *IEEE Transactions on Evolutionary Computation*, vol. 7, no. 2, pp. 117–132, 2003, doi: 10.1109/TEVC.2003.810758.
