# Joint Multi-Band Antenna Tilt Optimization with High-Dimensional Multi-Objective Bayesian Optimization over a Ray-Traced Network Simulator

---

## Abstract

Multi-band base stations transmit several frequency bands from the same mast, yet their antenna tilts are often tuned one band and one sector at a time, without accounting for how the bands interact. This paper optimizes all tilts jointly for coverage, interference and estimated user throughput, scoring each candidate with an uncalibrated, site-specific ray-traced simulation and selecting candidates with multi-objective Bayesian optimization. In a single-seed comparison with random search at the same number of evaluations, the method reaches a larger hypervolume, though it takes about twice the wall-clock time. Its recommended configuration reduces weak coverage and raises median signal quality and throughput, at the cost of more all-band cell overlap, a larger coverage hole on the highest band and a slightly larger share of users on hole tiles. Across the evaluated configurations, the coverage-hole rate changes little. The evaluation is limited to one simulated scenario with synthetic traffic, and no band-by-band baseline was run.

**Index Terms** — 5G, radio access network, antenna tilt optimization, coverage and capacity optimization, multi-band networks, Bayesian optimization, multi-objective optimization, ray tracing.

---

## I. Introduction

### A. Motivation

**What users experience.** For a mobile user, network quality is decided at the margins. A call drops when a commuter crosses from one cell into the next, a video stalls at the far end of a street, and a phone shows a strong signal yet delivers little data because several cells are competing for it. Users perceive none of the network's architecture, only that service is unreliable in particular places. One lever an operator has over these effects is the downward angle, or *tilt*, of the base-station antennas. Tilted too high, a cell spills into its neighbours and creates interference; tilted too low, it leaves gaps at its edge.

**Why the problem has become harder.** Modern 5G sites no longer transmit on a single frequency. A mast commonly carries several bands with complementary roles. Low bands travel far and penetrate buildings, forming a *coverage layer*; high bands carry much more data over a shorter range, forming a *capacity layer*; mid bands bridge the two. Every band on every sector has its own tilt, so the number of settings grows with each band an operator adds. These settings are also coupled. Raising the tilt of a capacity band widens its footprint into neighbouring cells and increases interference there. Lowering it pulls the cell edge inward and may leave users relying on a slower band. Where the serving band depends on signal quality and load, changing one band's tilt can move traffic onto other bands, changing the performance of carriers that nobody touched.

**What engineers face.** A reactive tuning workflow illustrates the difficulty. Tilts are set at planning time and then adjusted reactively. A drive test, a performance alarm or a customer complaint points to a problem sector; an engineer changes the tilt of one band on that sector remotely; and the team then waits for performance counters to show whether the change helped, and whether it quietly degraded a neighbouring cell. Each step is reasonable on its own, but the loop is slow, costly in field effort and blind to cross-band side effects until they appear in the statistics. Engineers are left addressing individual complaints rather than optimizing the network as a whole, and the network-wide configuration becomes the accumulation of local decisions rather than one chosen jointly.

**What is missing.** Engineers need a way to evaluate the *whole* network's tilt configuration, across every band and sector at once, before touching any antenna; to see the trade-offs between coverage, interference and user throughput explicitly rather than discovering them afterwards; and to obtain a small set of concrete, reviewable options rather than a single opaque answer. Such a tool must be economical in the number of evaluations, since each site-specific evaluation of a multi-site network is far more costly than a closed-form model, and transparent, so that an engineer can see where each option gains, where it loses and which antennas must change.

This paper addresses the static part of that need: coverage, interference and throughput at fixed UE positions; mobility and handover are not modelled. It combines a ray-traced model of a four-site urban scene, which estimates how every candidate configuration would affect coverage, interference and user throughput, with a sample-efficient multi-objective optimizer that learns which configurations are worth testing. The outcome is a shortlist of simulated options, each accompanied by its estimated effect on every performance indicator and the antenna changes it requires, together with a per-sector impact ranking for the recommended option, produced in minutes of computation in this setting before any antenna is changed. The radio model is not calibrated against measurements, and field validation of the chosen option remains necessary.

### B. Approach and Contributions

This paper treats the tilts of all (sector, band) pairs as a single coordinated optimization problem and evaluates it entirely in simulation. A ray tracer applied to explicit scene geometry, without calibration against measurements, scores each candidate configuration on coverage, co-band separation and the estimated equal-share Shannon rate of a synthetic UE population. The contributions are as follows:

1. **Problem formulation.** Joint multi-band tilt configuration is cast as a 36-dimensional, three-objective black-box problem with a co-band separation objective and a proportional-fair throughput utility that counts unserved UEs at zero rate (Section III).
2. **Sample-efficient search.** MORBO [6], a trust-region multi-objective Bayesian optimization method designed for high-dimensional spaces, is applied to the problem and compared with Sobol random search under a matched evaluation budget, a shared initial design and a common seed, so that the difference on this seed does not stem from the starting points (Section IV).
3. **Evaluation beyond the objectives.** The recommended configurations are assessed not only on the searched objectives but on eleven KPIs, per frequency layer, by area and by demand, and in terms of inter-layer load redistribution and per-sector impact. Apart from the hole rate, which equals one minus the coverage objective, none of these is optimized directly, although the throughput KPIs share the per-UE rates of the throughput objective and the overlap KPIs are related to the separation objective (Sections V–VI).
4. **Empirical findings.** In the studied SMa-like layout, the coverage-hole rate changes little across the evaluated configurations. On one seed, MORBO's recommendation instead improves weak coverage, median SINR and throughput and shifts load from 2600 MHz to the lower bands, at the cost of all-band overlap and 2600 MHz coverage. No band-by-band baseline was run, so the benefit of optimizing the bands jointly rather than separately is not isolated (Section VI).

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Section IV describes the search methods. Section V details the experimental methodology and assessment criteria. Section VI reports and interprets the results. Section VII discusses implications and limitations, and Section VIII concludes.

## II. Related Work

**Coverage and capacity optimization.** Antenna tilt is a principal control in coverage and capacity optimization (CCO), one of the self-organizing network (SON) use cases identified by 3GPP [1]. Early automated approaches adjusted tilt per cell with rule-based, fuzzy or reinforcement-learning controllers driven by local measurements [2], [3]. Many such controllers act on one cell, or one carrier, at a time, and therefore share the locality of manual tuning.

**Black-box optimization of RAN parameters.** Because network KPIs are expensive to evaluate and non-differentiable with respect to configuration, tilt and power settings have been optimized with Bayesian optimization and reinforcement learning over simulators. Dreifuerst et al. [4] tune the downtilt and transmit power of all sectors jointly with multi-objective Bayesian optimization and deep reinforcement learning, and obtain Pareto fronts between coverage and capacity. Standard Gaussian-process Bayesian optimization degrades in high dimensions; trust-region methods such as TuRBO [5] restore sample efficiency by restricting the search to adaptively sized local regions. MORBO [6] extends this idea to multiple objectives, maintaining trust regions centred on points of large hypervolume contribution and selecting batches by Thompson-sampled hypervolume improvement [7], [15].

**Ray-traced propagation models.** Statistical path-loss models represent building shadowing and multipath only in distribution, whereas the effect of a tilt change at a given location depends on site-specific blockage and reflections. GPU-accelerated ray tracers such as Sionna RT [10] make site-specific evaluation of a full network configuration fast enough — about 2.4 s per evaluation for the twelve-sector scene of this study — for optimization loops of tens to hundreds of full-network evaluations.

**Positioning of this work.** Joint optimization across sectors with a Pareto front between coverage and capacity has been shown before [4]. The present study differs in treating each band of a multi-band sector as a separate decision variable, coupled to the other bands through the serving rule; in adding a co-band separation objective; in searching 36 dimensions with a trust-region method; and in scoring candidates with a site-specific ray tracer. It further evaluates the outcome on layer-level and demand-level KPIs. Agreement between the objectives and these KPIs is partly by construction: the throughput KPIs share the per-UE rates of the throughput objective, and the overlap KPIs are related to the separation objective. The per-layer and demand-level measures are the less constrained check.

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

No operator data was available; a synthetic UE population was therefore generated. Every 15 minutes over seven days (672 intervals), 10 to 20 UEs are drawn from a mixture of four elliptical Gaussian demand hotspots and a uniform background over open ground. The hotspots are placed where the surrounding building volume is high, at least 500 m apart, and hold on average 70 % of UEs, modulated by a diurnal profile and first-order autoregressive noise. UEs are independent between intervals, without mobility. This yields 10,087 UE reports, which are synthetic positions rather than measurements, each served from the radio map at its tile. Measurement noise, report censoring and positioning error are not modelled.

### D. Decision Variables

For $N = 12$ sectors and $B = 3$ bands, the decision vector is the absolute electrical tilt of every sector-band pair,

$$
\boldsymbol{\theta} = [\theta_{1,1}, \dots, \theta_{1,B}, \dots, \theta_{N,B}] \in \Theta = [0^\circ, 20^\circ]^{36},
$$

with every proposal snapped to a 0.1° lattice so that each proposal is a discrete setting. Whether a given antenna supports this range and an independent electrical tilt per band was not checked. The incumbent configuration $\boldsymbol{\theta}^{(0)}$ sets 10° on every band and sector. No step-size limit or maximum change from $\boldsymbol{\theta}^{(0)}$ is imposed; tilt movement is reported but not penalized.

### E. Serving and Capacity Model

Within each interval, UEs attach sequentially in report-time order. Each UE attaches to the sector-band, among those with RSRP above $T_{\text{hole}} = -120$ dBm at its tile, that offers it the highest Shannon-bound rate under an equal share of $0.8 \, N_{RB}$ PRBs divided among the UEs already attached and itself. No UE is refused admission. A UE with no sector-band above $T_{\text{hole}}$ is unserved and assigned a rate of 0 Mbit/s. The resulting per-UE rate $R_u$ drives the throughput KPIs, the throughput objective and all per-layer service statistics. This rate-greedy rule is an idealization, not a model of any vendor's band-selection or load-balancing policy, so the inter-layer load results of Section VI-E describe this rule.

### F. Key Performance Indicators

Let $R_s(g) = \max_{i,b} R_{i,b}(g)$ be the best-server RSRP over all bands, $s_b(g)$ the strongest sector of band $b$ at tile $g$, $T_{\text{weak}} = -90$ dBm and $\Delta = 6$ dB the overlap margin. Eleven KPIs are computed for every candidate: the seven below, reported over all bands, and the 5th- and 50th-percentile best-server RSRP and SINR, reported per band only:

- **Coverage-hole rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[R_s(g) \le T_{\text{hole}}]$; a tile with no ray-traced path is a hole.
- **Weak-coverage rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[T_{\text{hole}} < R_s(g) \le T_{\text{weak}}]$.
- **Co-band overlap rate** (↓): $\frac{1}{|G|}\sum_{g} \mathbb{1}[N_{\text{ov}}(g) > 0]$, where
  $$N_{\text{ov}}(g) = \sum_b \mathbb{1}[R_{s_b,b}(g) > T_{\text{hole}}] \cdot \left|\{ i \ne s_b : R_{i,b}(g) \ge R_{s_b,b}(g) - \Delta,\ R_{i,b}(g) > T_{\text{hole}} \}\right|$$
  counts overlapping co-band neighbours, summed over bands. Two carriers of the same sector are never neighbours.
- **Overlapping neighbours per covered tile** (↓): the mean of $N_{\text{ov}}(g)$ over tiles with $R_s(g) > T_{\text{hole}}$.
- **Cell-edge, median and mean estimated UE throughput** (↑): the 5th and 50th percentiles and the mean of $R_u$ over every UE report, unserved UEs counting 0 Mbit/s.

Per band, the hole, weak and overlap rates are additionally reported, with the RSRP and SINR percentiles taken over the band's covered tiles. Best-server RSRP and SINR are reported per band only, since the strongest layer across bands is not one on which any UE is measured.

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

The search looks for tilt vectors $\boldsymbol{\theta}$ that make the objective vector $\mathbf{F}(\boldsymbol{\theta})$ of Section III-G as good as possible. Four properties of that problem decide how it can be searched:

1. **Expensive evaluations.** Scoring one candidate ray-traces every sector on every band and re-serves every UE report (Section VI-G). Only a small budget of evaluations is affordable.
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

**Protocol.** Both methods spend the same budget of $N_{\text{eval}}$ evaluations after the incumbent, and both start from the same $n_0$ initial points. Every candidate is scored by the same ray-tracing evaluator with one fixed solver seed, so all candidates share one Monte-Carlo noise realization and differences between candidates are not masked by solver noise. A run proceeds as follows:

1. Evaluate the incumbent configuration $\boldsymbol{\theta}^{(0)}$.
2. Evaluate the $n_0$ points of the initial design.
3. Evaluate the $N_{\text{eval}} - n_0$ points the method proposes.
4. Return the Pareto front of all evaluations, ordered by $\mathrm{HVC}$; the point with the largest $\mathrm{HVC}$ is the recommendation and the rest form the shortlist (Section III-G).

Every reported quantity is a ray-traced evaluation; no model prediction enters any reported result.

### B. Sobol Random Search

**What it is.** Random search evaluates points chosen without regard to earlier results [16]. Here the points come from a Sobol sequence [17], a deterministic *low-discrepancy* sequence: its first $n$ points cover $[0,1]^d$ more evenly than $n$ independent uniform draws, in the sense that the fraction of points that falls in any axis-aligned box stays close to the volume of that box. *Scrambling* randomizes the sequence under a seed while keeping this evenness.

**Why it is used.** It is the control. Since it ignores what earlier evaluations revealed, any advantage MORBO shows over it at the same budget measures the value of learning from the evaluations. It is also a meaningful baseline rather than a straw man: when only a few of the $d$ coordinates matter, every random draw still varies all of them, which makes random search hard to beat in high dimensions [16]. The Sobol sequence is preferred over independent draws because it is reproducible from its seed and any shorter draw is a prefix of a longer one, so the first $n_0$ points can be shared exactly with MORBO's initial design. Its evenness guarantee is asymptotic, and weak when the budget is small relative to the dimension.

**How it is applied.**

1. Draw the first $N_{\text{eval}}$ points of a scrambled Sobol sequence over $[0,1]^d$ with the run's seed.
2. Map each point to a tilt vector (Section IV-A) and evaluate it.
3. Label the first $n_0$ points the initial design; they are identical to MORBO's.
4. Return the Pareto front of all evaluations, ordered by $\mathrm{HVC}$.

Because the two methods share the initial design, budget and seed, a difference between them does not stem from the starting points. On a single seed, the difference combines MORBO's search strategy — its models, trust region and candidate generation together — with the realization of that seed.

### C. MORBO

**What it is.** MORBO (multi-objective Bayesian optimization over high-dimensional search spaces) [6] combines three ideas:

- **Bayesian optimization** [18]. A probabilistic *surrogate* model is fitted to the evaluations made so far. It predicts each objective at any untried point together with the uncertainty of that prediction, and an *acquisition rule* uses both to choose the next point to evaluate. Each new evaluation refines the model, and the loop repeats.
- **Trust regions**, from TuRBO [5]. Rather than modelling and searching the whole space at once, the search is confined to a box around the best point found so far. The box shrinks when it stops producing gains and is restarted elsewhere when it becomes too small.
- **Hypervolume-based Thompson sampling** [7], [15]. A batch of points is chosen to maximize the hypervolume improvement under a random draw from the surrogate, so that all objectives are improved together rather than through a fixed weighting.

**Why it is used.** Each component answers one of the four properties above. A surrogate lets every costly evaluation inform where to look next, which random search cannot do (properties 1 and 2). A single surrogate over the whole $d$-dimensional box needs far more data than a small budget provides, and its acquisition tends to favour the boundary of the box, where uncertainty is largest; confining the search to a trust region keeps the candidates where the model is informed, which TuRBO [5] showed restores sample efficiency in high dimensions (property 3). A weighted sum of the objectives would fix the trade-off between coverage, separation and throughput before the search begins; hypervolume-based selection keeps all objectives and returns a front from which the operator chooses (property 4).

**How it works.** The building blocks are described first, then the complete procedure.

**1) Surrogate model: Gaussian processes.** A Gaussian process (GP) treats an unknown function $f$ as random, such that its values at any finite set of inputs are jointly Gaussian [19]. It is specified by a mean $c$ and a *kernel* $k(\mathbf{u}, \mathbf{u}')$, the covariance between the function's values at two inputs: nearby inputs are expected to have similar values. Given observations $\mathbf{y}$ at inputs $\mathbf{u}_1, \dots, \mathbf{u}_n$, the prediction at a new input $\mathbf{u}$ is Gaussian with mean and variance

$$
\mu_n(\mathbf{u}) = c + \mathbf{k}_n(\mathbf{u})^\top \left(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I}\right)^{-1} (\mathbf{y} - c\mathbf{1}), \qquad
\sigma_n^2(\mathbf{u}) = k(\mathbf{u}, \mathbf{u}) - \mathbf{k}_n(\mathbf{u})^\top \left(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I}\right)^{-1} \mathbf{k}_n(\mathbf{u}),
$$

where $\mathbf{K}_{st} = k(\mathbf{u}_s, \mathbf{u}_t)$ is the covariance among the observed inputs, $[\mathbf{k}_n(\mathbf{u})]_t = k(\mathbf{u}, \mathbf{u}_t)$ the covariance between the new input and each observed one, and $\sigma_\varepsilon^2$ the observation noise. The mean $\mu_n$ is the best guess of the objective; the variance $\sigma_n^2$ is small near observed points and grows away from them. One independent GP is fitted to each objective. The kernel is the Matérn-5/2 kernel with automatic relevance determination (ARD):

$$
k(\mathbf{u}, \mathbf{u}') = \sigma_f^2 \left(1 + \sqrt{5}\,r + \tfrac{5}{3} r^2\right) e^{-\sqrt{5}\,r}, \qquad r^2 = \sum_{j=1}^{d} \frac{(u_j - u'_j)^2}{\ell_j^2}.
$$

The lengthscale $\ell_j$ states how far tilt $j$ must move before an objective changes appreciably: a short lengthscale marks a sector-band to which the objective is sensitive, a long one a sector-band that barely matters. ARD learns one lengthscale per dimension from the data. With few observations and many lengthscales, this estimate is ill-posed, so the lengthscales carry a dimension-scaled log-normal prior [20],

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

**4) Acquisition: Thompson sampling of the hypervolume improvement.** Thompson sampling [15] makes a decision under uncertainty by drawing one plausible version of the unknown function from the model and acting as if that draw were true. Where the model is confident, the draws agree and the rule exploits; where it is uncertain, the draws vary and the rule explores. Here a joint sample $\tilde{\mathbf{f}}$ of all objectives is drawn from the GP posterior over the candidates $X_{\text{cand}}$ and over the points $X_{\text{pend}}$ already chosen for the current batch, and the next point is the candidate whose sampled objectives would add the most hypervolume:

$$
\mathbf{u}^\star = \arg\max_{\mathbf{u} \in X_{\text{cand}}} \mathrm{HVI}\!\left(\tilde{\mathbf{f}}(\mathbf{u});\ P^\ast \cup \tilde{\mathbf{f}}(X_{\text{pend}})\right),
$$

where $P^\ast$ is the observed Pareto front. Including the sampled values of the pending points means that a second point in the batch is not chosen for the same improvement as the first. A batch of $q$ points is built by repeating this choice $q$ times with fresh candidates; evaluating a batch amortizes one round of model fitting over several evaluations. Because the rule is evaluated on a finite set of candidates from one posterior draw, it requires neither a continuous optimization over $d$ dimensions nor an integral of the expected hypervolume improvement [7].

**5) Trust-region adaptation and restart.** The side of the trust region adapts to the progress of the search:

- A batch is a *success* when it raises the hypervolume by more than a relative margin $\epsilon$, $\mathrm{HV}(P \cup Y_{\text{new}}) > (1 + \epsilon)\,\mathrm{HV}(P)$; a success resets the failure count.
- Otherwise each of its evaluations counts as a failure. After $\tau$ consecutive failures, a tolerance that [6] scales with the dimension, the side halves, $L \leftarrow L/2$, concentrating the search around a centre that has stopped yielding gains. Following [6], the side never grows.
- When the side falls below $L_{\min}$, the region is exhausted and restarts. Its former centre is barred from serving as a centre for a fixed number of rounds, so the search does not return to it at once.

The restart centre is chosen over the whole cube. A GP fitted to the initial design and earlier restart points is sampled once over quasi-random points of $[0,1]^d$, and the point is taken that maximizes a *random hypervolume scalarization* [21] of its sampled objectives together with those data,

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
| Sites / sectors / sector-band pairs | 4 / 12 / 36 |
| Inter-site distance | 1,299 m centre to vertex sites (SMa, TR 38.901 [11]); 2,250 m between vertex sites |
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

Demand and coverage are misaligned: holes occupy 4.4 % of the area but carry 16.4 % of UE reports (Table III), and 13.9 % of reports fall on tiles with no path to any sector. One of the four demand hotspots (1,866 UE reports), placed by the synthetic generator where building volume is high, is centred 3,297 m from the nearest site, where no layer provides a path at the incumbent tilts. For this synthetic population, the area-based hole rate therefore understates the service deficit.

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

Five criteria are applied. Criteria 1 and 2 assess effectiveness, criterion 3 the search, criterion 4 side effects across layers and demand, and criterion 5 cost. None tests robustness to model error, traffic or seed.

1. **Overall quality:** hypervolume of each run's evaluated set, expressed as a gain over the incumbent's. Hypervolume is taken against the origin on unnormalized objectives, so these percentages depend on that reference point.
2. **Reported KPIs:** direction of change against the incumbent over the seven network-level KPIs and the per-band measures of Section III-F. A change is labelled better, worse or unchanged; since solver noise per KPI has not been measured, no tie tolerance is applied, and the hole rate, whose changes are a tenth of a point, is left out of the tally.
3. **Search effectiveness:** sample efficiency, and the paired hypervolume difference between MORBO and random search on the same seed.
4. **Layer and demand effects:** where a configuration moves demand, not only area, and how it trades one KPI against another.
5. **Cost:** ray-tracing evaluations, ray-tracing time and wall-clock time per run.

The evaluation operates exclusively on archived run outputs without re-solving. It first verifies that all runs share the incumbent's scenario, grid, solver settings, bands, carrier frequencies and KPI definitions, then recomputes each recommendation's KPIs from its archived radio map to confirm that they were recorded correctly.

## VI. Results and Analysis

### A. Comparability and Consistency

All 23 comparability checks hold: both runs optimized the incumbent's scenario, retained their radio maps and match the incumbent's grid, solver settings, bands, carrier frequencies and KPI definitions. Recomputing each recommendation's KPIs from its archived radio map reproduces the recorded values exactly for the hole, weak and overlap rates and the objectives, and to within $3 \times 10^{-6}$ Mbit/s for the throughput KPIs. This checks the bookkeeping, not the reproducibility of the search. Repeatability across seeds remains untested, as each method was executed with one seed.

### B. Network-Level KPIs

*Table IV. Recommended configuration of each method against the incumbent. Bold marks the best value per row. The tally excludes the hole rate, whose changes of a tenth of a point may lie within unmeasured solver noise, and counts median and mean throughput separately although both derive from the same per-UE rates.*

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
| Network KPIs better / worse / unchanged, hole rate excluded | | | 4 / 1 / 1 | 2 / 3 / 1 |
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

*The all-band hole rate changes little across the evaluated configurations.* In the studied layout, the incumbent configuration already holds the hole rate below 5 %. The recommendations shift it by a tenth of a percentage point (−0.10 for MORBO, +0.10 for random search), and the lowest value found by either search is 4.15 %. This describes the configurations both runs evaluated, not a sensitivity analysis; per-band hole rates move by more (Table VIII). One reason lies in the propagation model: 1,876 of the 4,445 incumbent hole tiles receive no propagation path from any sector, and tilt alters antenna gain, not the existence of a path. Since diffraction and diffuse scattering are disabled (Section III-B), some of these tiles may receive signal in a model that includes them. The remaining 2,569 hole tiles do have a path. Whether a different site layout would close these holes was not tested.

*MORBO trades a small overlap increase for gains elsewhere.* Its recommendation achieves the largest reduction in weak coverage (−4.8 points) and raises median SINR on every band, by 2.3 dB at 2600 MHz, 1.6 dB at 1800 MHz and 2.4 dB at 700 MHz. The tails do not all improve: 5th-percentile RSRP falls by 0.7 dB at 2600 MHz and 0.2 dB at 1800 MHz, and 5th-percentile SINR falls by 0.1 dB at 1800 MHz. The all-band overlap rate rises by 1.7 points, even though the overlap rate of every individual band falls (Table VIII); since the all-band rate counts a tile if any band overlaps there, the overlapping tiles of different bands must coincide less than before.

*Random search buys throughput with overlap.* Its recommendation attains the highest median and mean throughput, but the overlap rate rises by 13.9 points and the separation objective falls by 0.039. Its 700 MHz layer loses 2.4 dB of median RSRP and 1.5 dB of median SINR, weakening the coverage layer.

*Cell-edge throughput is pinned at zero.* Between 16.4 % and 16.6 % of UE reports lie on hole tiles in every configuration, exceeding 5 %, so the 5th-percentile throughput is 0 Mbit/s throughout and cannot discriminate between configurations.

### C. Search Effectiveness

*Table VI. Hypervolume per method (seed 42), against the origin on unnormalized objectives; the percentage gains depend on that reference point.*

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

*MORBO is more sample-efficient on this seed, though not faster.* After 25 evaluations, MORBO's hypervolume already exceeds the one random search reaches after 73, by a margin of 0.0016 (2.3544 against 2.3528). By evaluation 50, its best hole rate (4.15 %) is below random search's (4.26 %). MORBO's hypervolume was still rising between evaluations 50 and 73, indicating that the budget did not exhaust its progress. Sample efficiency is not wall-clock efficiency here. At about 2.4 s per evaluation, MORBO's model overhead exceeded its ray-tracing time (Section VI-G), and in MORBO's wall-clock time random search could have run about twice as many evaluations. That comparison was not run.

*The Pareto front consists entirely of trust-region proposals.* None of the eight Sobol points lies on MORBO's front; all 18 Pareto-optimal points are among its 64 trust-region proposals. These proposals also exceed the initial design on average in all three objectives: mean coverage 0.9570 against 0.9522, mean separation 0.6789 against 0.6497 and mean throughput utility 3.589 against 3.529. Because these proposals perturb the current Pareto points, any local search would be expected to show this. It does not isolate the contribution of the GP models, and no ablation without them was run.

*No evaluation reduced the all-band overlap rate.* The best overlap rate found by either method equals the incumbent's 34.9 %: within the explored space, no configuration lowered the aggregate overlap.

*The recommendation is not the minimum-hole configuration.* MORBO's lowest observed hole rate is 4.15 %, whereas its recommendation — the point with the largest hypervolume contribution — lies at 4.30 %. The remaining front points constitute alternatives for an operator who prioritizes hole reduction.

![Coverage vs separation trade-off](../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png)

*Fig. 8. All evaluated configurations projected onto the coverage and separation objectives, with each method's front and recommendation.*

### D. Per-Layer and Demand-Weighted Effects

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

*Some uptilted capacity carriers serve more reports.* Sector n3s2 at 2600 MHz, uptilted by 8.1°, serves 363 additional reports (769 → 1,132) at a 4.3 dB higher median SINR, and n0s2 at 2600 MHz, uptilted by 7.4°, serves 214 additional reports. Which sectors previously served these reports was not traced. Ranking sector-bands by the magnitude of the change in served reports could give an operator a prioritized monitoring list for a staged rollout; this ranking is computed for the recommendation only and has not been tested in a rollout.

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

Subject to the limitations below, the results suggest three possible changes to tilt-optimization practice. First, joint optimization can complement trial-and-error with selection among simulated alternatives: a single run produced 18 Pareto-optimal configurations, each accompanied by its full KPI vector and per-antenna tilt deltas, from which an engineer can select according to operational priorities. Second, the recommended configuration moves all bands jointly: it uptilts the coverage layer on average while making opposing per-sector moves on the capacity layer, and load shifts across sectors and frequency layers accordingly. Whether a band-by-band procedure would reach a similar configuration was not tested. Third, the per-sector impact ranking, computed here for the recommendation only, could inform a staged rollout plan with an explicit monitoring order.

In the studied layout, however, tilt optimization barely changed the coverage-hole rate and did not close the holes that carry demand; a large share of hole tiles have no propagation path to any sector in the ray tracer as configured. In such a setting, tilt search appears better directed at weak coverage, SINR and inter-layer load balance. With diffraction disabled, it remains open whether the demand that no layer reaches is truly unreachable or an artefact of the propagation settings; only in the former case is it a site-planning problem rather than a tilt-tuning one.

### B. Limitations and Threats to Validity

- **Single, uncalibrated scenario.** One scene, one UE realization and one layout were studied; no result has been validated on a held-out scenario or calibrated against drive-test measurements. The simulator is not a digital twin of any deployed network.
- **Propagation settings.** Diffraction and diffuse scattering are disabled, which can turn shadowed tiles into no-path holes; the hole findings depend on this choice.
- **Partial SMa layout.** The layout follows the SMa inter-site distance and mast height [11] but uses 4 of its 19 sites, over an urban rather than suburban scene. UEs lie on a single outdoor plane at 1.5 m; the indoor, multi-floor UE distribution of SMa is not modelled. The incumbent is a uniform 10° configuration rather than an operator-tuned one.
- **Limited baselines.** MORBO is compared only with random search and a uniform 10° incumbent. Neither an operator-tuned configuration, a band-by-band procedure nor another optimizer, such as an evolutionary algorithm or reinforcement learning, was evaluated. The benefit of joint over band-by-band optimization is therefore not isolated.
- **Equal evaluations, not equal time.** The methods are compared at 73 evaluations each; MORBO took about twice the wall-clock time, and a comparison at equal wall-clock time was not run.
- **No ablation.** The contribution of the GP models, as opposed to local perturbation within a trust region, was not separated.
- **Synthetic traffic and SINR biases.** The traffic is synthetic, and hotspots are placed where building volume is high, so the demand on hole tiles depends on that placement rule. SINR assumes every co-band sector transmits at full power, which is pessimistic at low load, and omits a receiver noise figure, which is optimistic.
- **Idealized serving rule.** UEs attach to the sector-band with the highest equal-share Shannon rate, so the inter-layer load results describe this rule, not a deployed band-selection policy. Mobility and handover are not modelled.
- **Single seed per method.** No confidence interval, significance test or repeatability measure is available for Tables IV and VI.
- **Winner's curse and solver noise.** All evaluations share one solver seed, and each recommendation is the best of 73 under that noise realization, so its scores are biased upward. Solver noise per KPI is unmeasured; changes of a tenth of a point in the hole rate may lie within it.
- **Reference-point dependence.** Hypervolume is computed against the origin; a reference point at the incumbent, or a selection restricted to configurations dominating the incumbent, could yield a different recommendation.
- **Algorithmic configuration.** A single trust region was used and the budget of 73 evaluations in 36 dimensions is small; MORBO was still improving late in the budget. The original method's default number of trust regions was not evaluated.
- **Objective–KPI mismatch on overlap.** The separation objective is a soft per-tile product over bands, whereas the all-band overlap KPI counts any crowded band; the recommendation improves the former while the latter worsens.
- **Demand-agnostic area objectives.** Coverage and separation are tile-uniform; the hole-rate reduction occurred on low-demand tiles, and the share of UE reports on hole tiles rose slightly (Table IX).
- **Simplified capacity model.** Rates are equal-share Shannon bounds without scheduling, MCS limits or mobility. Inter-band interference is absent by construction, and cell-edge throughput is 0 Mbit/s in every configuration because more than 5 % of UE reports lie on hole tiles.
- **Unpenalized movement.** The recommendation moves every antenna (Table XII), which may exceed a practical RET change window.

## VIII. Conclusion and Future Work

This paper formulated the joint configuration of electrical tilts across all sectors and bands of a multi-band network as a 36-dimensional, three-objective black-box problem evaluated by an uncalibrated ray-traced network simulator, and compared MORBO with Sobol random search under a matched evaluation budget and seed. Table XIV summarizes the outcome against the assessment criteria.

*Table XIV. Summary against the assessment criteria, seed 42. Hypervolume is against the origin on unnormalized objectives.*

| Criterion | Random search | MORBO |
|---|---|---|
| 1. Hypervolume | +3.4 % over incumbent | **+6.9 % over incumbent** |
| 2. Reported KPIs, hole rate excluded | 2 better, 3 worse; overlap +13.9 points | **4 better, 1 worse**; overlap +1.7 points |
| 3. Search effectiveness | Best hole rate 4.26 %; 9 Pareto points | **Exceeds random search's final hypervolume by evaluation 25** (margin 0.0016); 18 Pareto points, all trust-region proposals |
| 4. Layer and demand effects | Weakens the 700 MHz layer (−2.4 dB median RSRP) | Weak coverage reduced by area and demand; 2600 MHz hole rate +1.3 points; hole-tile demand share +0.18 points |
| 5. Cost | 73 evaluations, 3.4 min | 73 evaluations, 7.2 min |

The principal conclusions are:

1. In the studied layout, the uniform 10° incumbent already has a hole rate of 4.40 % before any optimization, and 1,876 of its 4,445 hole tiles receive no propagation path in the ray tracer as configured, without diffraction or diffuse scattering.
2. Across the evaluated configurations, the hole rate changed little in this geometry: the lowest value found was 4.15 %, and MORBO's recommendation attains 4.30 %. That recommendation, biased upward as the best of 73 under one noise realization, attains 4.8 points less weak coverage, 1.6–2.4 dB higher median SINR per band and 4.5 Mbit/s higher mean throughput. Its costs are 1.7 points more all-band co-band overlap, 1.3 points more 2600 MHz hole area, 0.18 points more UE reports on hole tiles and 1.8 dB lower median served SINR on 700 MHz.
3. At an equal number of evaluations and on one seed per method, MORBO reached a larger hypervolume than random search in about twice the wall-clock time. Leaving out the hole rate, its recommendation improved four of six network KPIs against random search's two. Whether this margin holds across seeds, or at equal wall-clock time, is untested.
4. For this synthetic population, the area-based hole rate understates the service deficit: about 16 % of UE reports lie on hole tiles in every configuration. One demand hotspot, placed by the generator where building volume is high, is centred where the ray tracer finds no propagation path.

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

[18] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global optimization of expensive black-box functions," *Journal of Global Optimization*, vol. 13, no. 4, pp. 455–492, 1998.

[19] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. Cambridge, MA, USA: MIT Press, 2006.

[20] C. Hvarfner, E. O. Hellsten, and L. Nardi, "Vanilla Bayesian optimization performs great in high dimensions," in *Proc. International Conference on Machine Learning (ICML)*, PMLR 235, 2024.

[21] D. Golovin and Q. Zhang, "Random hypervolume scalarizations for provable multi-objective black box optimization," in *Proc. International Conference on Machine Learning (ICML)*, PMLR 119, 2020.
