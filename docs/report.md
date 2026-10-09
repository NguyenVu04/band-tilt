# Joint Multi-Band Antenna Tilt Optimization with High-Dimensional Multi-Objective Bayesian Optimization over a Ray-Traced Network Simulator

---

## Abstract

Multi-band base stations transmit several frequency bands from the same mast, yet their antenna tilts are often tuned one band and one sector at a time, without accounting for how the bands interact. This paper optimizes all tilts jointly for coverage, interference and estimated user throughput, scoring each candidate with an uncalibrated, site-specific ray-traced simulation and selecting candidates with multi-objective Bayesian optimization. On a seven-site urban-macro layout with the 3GPP calibration tilt as the incumbent, and in a single-seed comparison with random search at the same number of evaluations, the method reaches a larger hypervolume in about a tenth more wall-clock time. Its recommended configuration lowers the coverage-hole rate and the co-band overlap rate together, reduces weak coverage and raises median signal quality and estimated throughput. Random search's best configuration, a point of the initial design both methods share, lowers the hole rate further but raises the overlap rate. Most of the remaining hole area lies more than 1 km from any site. The evaluation is limited to one simulated scenario with synthetic traffic and one seed, and no band-by-band baseline was run.

**Index Terms** — 5G, radio access network, antenna tilt optimization, coverage and capacity optimization, multi-band networks, Bayesian optimization, multi-objective optimization, ray tracing.

---

## I. Introduction

### A. Motivation

**What users experience.** For a mobile user, network quality is decided at the margins. A call drops when a commuter crosses from one cell into the next, a video stalls at the far end of a street, and a phone shows a strong signal yet delivers little data because several cells are competing for it. Users perceive none of the network's architecture, only that service is unreliable in particular places. One lever an operator has over these effects is the downward angle, or *tilt*, of the base-station antennas. Tilted too high, a cell spills into its neighbours and creates interference; tilted too low, it leaves gaps at its edge.

**Why the problem has become harder.** Modern 5G sites no longer transmit on a single frequency. A mast commonly carries several bands with complementary roles. Low bands travel far and penetrate buildings, forming a *coverage layer*; high bands carry much more data over a shorter range, forming a *capacity layer*; mid bands bridge the two. Every band on every sector has its own tilt, so the number of settings grows with each band an operator adds. These settings are also coupled. Raising the tilt of a capacity band widens its footprint into neighbouring cells and increases interference there. Lowering it pulls the cell edge inward and may leave users relying on a slower band. Where the serving band depends on signal quality and load, changing one band's tilt can move traffic onto other bands, changing the performance of carriers that nobody touched.

**What engineers face.** A reactive tuning workflow illustrates the difficulty. Tilts are set at planning time and then adjusted reactively. A drive test, a performance alarm or a customer complaint points to a problem sector; an engineer changes the tilt of one band on that sector remotely; and the team then waits for performance counters to show whether the change helped, and whether it quietly degraded a neighbouring cell. Each step is reasonable on its own, but the loop is slow, costly in field effort and blind to cross-band side effects until they appear in the statistics. Engineers are left addressing individual complaints rather than optimizing the network as a whole, and the network-wide configuration becomes the accumulation of local decisions rather than one chosen jointly.

**What is missing.** Engineers need a way to evaluate the *whole* network's tilt configuration, across every band and sector at once, before touching any antenna; to see the trade-offs between coverage, interference and user throughput explicitly rather than discovering them afterwards; and to obtain a small set of concrete, reviewable options rather than a single opaque answer. Such a tool must be economical in the number of evaluations, since each site-specific evaluation of a multi-site network is far more costly than a closed-form model, and transparent, so that an engineer can see where each option gains, where it loses and which antennas must change.

This paper addresses the static part of that need: coverage, interference and throughput at fixed UE positions; mobility and handover are not modelled. It combines a ray-traced model of a seven-site urban scene, which estimates how every candidate configuration would affect coverage, interference and user throughput, with a sample-efficient multi-objective optimizer that learns which configurations are worth testing. The outcome is a shortlist of simulated options, each accompanied by its estimated effect on every performance indicator and the antenna changes it requires, together with a per-sector impact ranking for the recommended option, produced in minutes of computation in this setting before any antenna is changed. The radio model is not calibrated against measurements, and field validation of the chosen option remains necessary.

### B. Approach and Contributions

This paper treats the tilts of all (sector, band) pairs as a single coordinated optimization problem and evaluates it entirely in simulation. A ray tracer applied to explicit scene geometry, without calibration against measurements, scores each candidate configuration on coverage, co-band separation and the estimated equal-share Shannon rate of a synthetic UE population. The contributions are as follows:

1. **Problem formulation.** Joint multi-band tilt configuration is cast as a 63-dimensional, three-objective black-box problem with a co-band separation objective and a proportional-fair throughput utility that counts unserved UEs at zero rate (Section III).
2. **Sample-efficient search.** MORBO [6], a trust-region multi-objective Bayesian optimization method designed for high-dimensional spaces, is applied to the problem and compared with Sobol random search under a matched evaluation budget, a shared initial design and a common seed, so that the difference on this seed does not stem from the starting points (Section IV).
3. **Evaluation beyond the objectives.** The recommended configurations are assessed not only on the searched objectives but on eleven KPIs, per frequency layer, by area and by demand, and in terms of inter-layer load redistribution and per-sector impact. Apart from the hole rate, which equals one minus the coverage objective, none of these is optimized directly, although the throughput KPIs share the per-UE rates of the throughput objective and the overlap KPIs are related to the separation objective (Sections V–VI).
4. **Empirical findings.** On one seed, MORBO's recommendation lowers the hole rate and the all-band overlap rate together, as 32 of its 73 evaluations do against none of random search's; it also improves weak coverage, median SINR and throughput and shifts load from 2600 MHz to the lower bands. Random search adds almost no hypervolume beyond the shared initial design. No band-by-band baseline was run, so the benefit of optimizing the bands jointly rather than separately is not isolated (Section VI).

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Section IV describes the search methods. Section V details the experimental methodology and assessment criteria. Section VI reports and interprets the results. Section VII discusses implications and limitations, and Section VIII concludes.

## II. Related Work

**Coverage and capacity optimization.** Antenna tilt is a principal control in coverage and capacity optimization (CCO), one of the self-organizing network (SON) use cases identified by 3GPP [1]. Early automated approaches adjusted tilt per cell with rule-based, fuzzy or reinforcement-learning controllers driven by local measurements [2], [3]. Many such controllers act on one cell, or one carrier, at a time, and therefore share the locality of manual tuning.

**Black-box optimization of RAN parameters.** Because network KPIs are expensive to evaluate and non-differentiable with respect to configuration, tilt and power settings have been optimized with Bayesian optimization and reinforcement learning over simulators. Dreifuerst et al. [4] tune the downtilt and transmit power of all sectors jointly with multi-objective Bayesian optimization and deep reinforcement learning, and obtain Pareto fronts between coverage and capacity. Standard Gaussian-process Bayesian optimization degrades in high dimensions; trust-region methods such as TuRBO [5] restore sample efficiency by restricting the search to adaptively sized local regions. MORBO [6] extends this idea to multiple objectives, maintaining trust regions centred on points of large hypervolume contribution and selecting batches by Thompson-sampled hypervolume improvement [7], [15].

**Ray-traced propagation models.** Statistical path-loss models represent building shadowing and multipath only in distribution, whereas the effect of a tilt change at a given location depends on site-specific blockage and reflections. GPU-accelerated ray tracers such as Sionna RT [10] make site-specific evaluation of a full network configuration fast enough — about 5 to 7 s per evaluation for the 21-sector scene of this study — for optimization loops of tens to hundreds of full-network evaluations.

**Positioning of this work.** Joint optimization across sectors with a Pareto front between coverage and capacity has been shown before [4]. The present study differs in treating each band of a multi-band sector as a separate decision variable, coupled to the other bands through the serving rule; in adding a co-band separation objective; in searching 63 dimensions with a trust-region method; and in scoring candidates with a site-specific ray tracer. It further evaluates the outcome on layer-level and demand-level KPIs. Agreement between the objectives and these KPIs is partly by construction: the throughput KPIs share the per-UE rates of the throughput objective, and the overlap KPIs are related to the separation objective. The per-layer and demand-level measures are the less constrained check.

## III. System Model and Problem Formulation

### A. Network Layout

The study area is an urban scene rasterized onto a grid $G$ of 20 m tiles, $326 \times 310 = 101{,}060$ tiles covering $6{,}200 \times 6{,}520$ m. Seven sites form a centre site and its first tier on a hexagonal grid with a 500 m inter-site distance, the urban-macro (UMa) layout of 3GPP TR 38.901 [11]. UMa specifies a 19-site grid, of which these seven sites are the centre and first ring. Each mast stands on open ground at its ideal position. Each site hosts three sectors at azimuths of 0°, 120° and 240° on 25 m masts, the UMa base-station height, yielding $N = 21$ sectors. Each sector is equipped with an $8 \times 8$ cross-polarized planar array with the TR 38.901 element pattern, transmitting 4.85 dBm reference-signal power per resource element. Every sector carries $B = 3$ bands (Table I), giving 63 sector-band pairs.

*Table I. Frequency bands. PRB limits are $N_{RB}$ at 15 kHz subcarrier spacing per 3GPP TS 38.101-1, Table 5.3.2-1 [12].*

| Band | Carrier [MHz] | Bandwidth [MHz] | PRB limit per sector |
|---|---:|---:|---:|
| 2600 MHz | 2600 | 40 | 216 |
| 1800 MHz | 1800 | 20 | 106 |
| 700 MHz | 700 | 10 | 52 |

![Study area](../reports/figures/00_simulation/study_area.png)

*Fig. 1. Study area: twenty-one sectors on seven sites and a sample of UE positions over the scene.*

### B. Propagation Model

The radio model is not calibrated against measurements. Each band is ray-traced separately with Sionna RT [10], with a maximum path depth of 8 and $10^7$ ray samples per transmitter. Line-of-sight, specular reflection and refraction are modelled; diffuse scattering and diffraction are disabled, and material properties are frequency-static. The receiver is a single vertically polarized dipole at a UE height of 1.5 m. The ray tracer yields, for every sector $i$, band $b$ and tile $g$, the RSRP $R_{i,b}(g)$ and the SINR. All powers are expressed per resource element (RE) [13]: interference comprises every other co-band sector transmitting at full power, and thermal noise is $kT\Delta f$ with $T = 298.15$ K over a single $\Delta f = 15$ kHz subcarrier rather than the channel bandwidth, without a receiver noise figure. Bands are orthogonal, so no inter-band interference is modelled.

### C. Traffic Model

No operator data was available; a synthetic UE population was therefore generated. Every 15 minutes over seven days (672 intervals), 10 to 20 UEs are drawn from a mixture of three elliptical Gaussian demand hotspots and a uniform background over open ground. The hotspots are placed where the surrounding building volume is high, at least 500 m apart, and hold on average 70 % of UEs, modulated by a diurnal profile and first-order autoregressive noise. UEs are independent between intervals, without mobility. This yields 10,087 UE reports, which are synthetic positions rather than measurements, each served from the radio map at its tile. Measurement noise, report censoring and positioning error are not modelled.

### D. Decision Variables

For $N = 21$ sectors and $B = 3$ bands, the decision vector is the absolute downtilt of every sector-band pair,

$$
\boldsymbol{\theta} = [\theta_{1,1}, \dots, \theta_{1,B}, \dots, \theta_{N,B}] \in \Theta = [0^\circ, 20^\circ]^{63},
$$

with every proposal snapped to a 0.1° lattice so that each proposal is a discrete setting. Whether a given antenna supports this range and an independent electrical tilt per band was not checked. The simulator applies each tilt as a rotation of the whole array in elevation, so side and back lobes tilt with the main beam, as under mechanical tilt. The incumbent configuration $\boldsymbol{\theta}^{(0)}$ sets 12° on every band and sector, the UMa electrical downtilt of the TR 38.901 calibration parameters [11]. No step-size limit or maximum change from $\boldsymbol{\theta}^{(0)}$ is imposed; tilt movement is reported but not penalized.

### E. Serving and Capacity Model

Within each interval, UEs attach sequentially in report-time order. Each UE attaches to the sector-band, among those with RSRP above $T_{\text{hole}} = -110$ dBm at its tile, that offers it the highest Shannon-bound rate under an equal share of $0.8 \, N_{RB}$ PRBs divided among the UEs already attached and itself. No UE is refused admission. A UE with no sector-band above $T_{\text{hole}}$ is unserved and assigned a rate of 0 Mbit/s. The resulting per-UE rate $R_u$ drives the throughput KPIs, the throughput objective and all per-layer service statistics. This rate-greedy rule is an idealization, not a model of any vendor's band-selection or load-balancing policy, so the inter-layer load results of Section VI-E describe this rule.

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
| Sites / sectors / sector-band pairs | 7 / 21 / 63 |
| Layout | Centre site and first tier of a hexagonal grid (UMa, TR 38.901 [11]) |
| Inter-site distance | 500 m (UMa) |
| Mast height | 25 m (UMa) |
| Sector azimuths | 0°, 120°, 240° at every site |
| Incumbent tilt (2600 / 1800 / 700 MHz) | 12° / 12° / 12° (UMa calibration, TR 38.901 [11]) |
| Tilt bounds and resolution | [0°, 20°], 0.1° |
| Time intervals | 672 × 15 min (7 days) |
| UEs per interval | 10 to 20 |
| Demand hotspots | 3, holding 70 % of UEs on average |
| UE reports | 10,087 |
| UE reports with no path to any sector | 3.8 % |
| KPI thresholds | $T_{\text{hole}} = -110$ dBm, $T_{\text{weak}} = -90$ dBm, $\Delta = 6$ dB |
| Search seed | 42 (both methods) |

The fraction of tiles reached by any path is 93.1 % at 700 MHz, 90.6 % at 1800 MHz and 90.8 % at 2600 MHz, with median RSRP over reached tiles of −90.5, −97.7 and −101.7 dBm respectively. 3GPP fixes no coverage-hole threshold: TS 37.320 [22] defines a hole by the signal level needed for basic service without quantifying it, so $T_{\text{hole}}$ is a choice of this study.

### B. Characterization of the Incumbent Configuration

At the incumbent tilts, 14.7 % of tiles are holes, 38.8 % weak and 46.6 % well covered (Table III). The holes are fragmented into 3,582 connected regions, 2,491 of which comprise a single tile; the largest holds 26.5 % of the hole area, and 3,996 of the 14,818 hole tiles receive no propagation path from any sector. The hole rate is 0.94 % within 1 km of a site and 17.4 % beyond: seven sites span about 1 km of a 6.2 × 6.5 km scene, so most of the hole area lies far from every site.

Per layer, the hole share is 17.5 % at 700 MHz, 28.9 % at 1800 MHz and 35.7 % at 2600 MHz. Although 700 MHz is the strongest band on 90.9 % of the covered area, the serving rule places 73.9 % of that area on 2600 MHz, the band with the most PRBs. Of all UE reports, 57.2 % are served on 2600 MHz, 12.0 % on 1800 MHz and 13.2 % on 700 MHz. Co-band overlap affects 46.3 % of tiles, with 2.13 overlapping neighbours per covered tile on average (median 1, 90th percentile 6); 32.9 % of covered tiles have three or more, at a median distance of 1,652 m from the nearest site.

Demand and coverage are misaligned: holes occupy 14.7 % of the area but carry 17.6 % of UE reports (Table III), and 3.8 % of reports fall on tiles with no path to any sector. One of the three demand hotspots (2,292 UE reports), placed by the synthetic generator where building volume is high, is centred 1,906 m from the nearest site, where the best server is at −145 dBm at the incumbent tilts. For this synthetic population, the area-based hole rate therefore understates the service deficit.

*Table III. Coverage class by area and by demand at the incumbent configuration.*

| Coverage class | Tiles | Share of area | Share of UE reports |
|---|---:|---:|---:|
| Hole ($\le -110$ dBm) | 14,818 | 14.7 % | 17.6 % |
| Weak ($-110$ to $-90$ dBm) | 39,163 | 38.8 % | 25.3 % |
| Good ($> -90$ dBm) | 47,079 | 46.6 % | 57.2 % |

![RSRP per band](../reports/figures/00_simulation/rsrp_per_band.png)

*Fig. 2. Best-server RSRP per band at the incumbent tilts.*

![Demand vs coverage](../reports/figures/01_eda/demand_vs_coverage.png)

*Fig. 3. UE demand alongside signal strength at the incumbent tilts.*

### C. Data Verification

Prior to optimization, the UE table, scenario manifest and radio map were validated against 28 contract checks covering schema, value ranges, grid consistency, band and frequency agreement, physical plausibility (no RSRP above the transmitted reference-signal power) and the absence of duplicates; all 28 held. No record was removed or altered.

### D. Assessment Criteria

Five criteria are applied. Criteria 1 and 2 assess effectiveness, criterion 3 the search, criterion 4 side effects across layers and demand, and criterion 5 cost. None tests robustness to model error, traffic or seed.

1. **Overall quality:** hypervolume of each run's evaluated set, expressed as a gain over the incumbent's. Hypervolume is taken against the origin on unnormalized objectives, so these percentages depend on that reference point.
2. **Reported KPIs:** direction of change against the incumbent over the seven network-level KPIs and the per-band measures of Section III-F. A change is labelled better, worse or unchanged; since solver noise per KPI has not been measured, no tie tolerance is applied.
3. **Search effectiveness:** sample efficiency, and the paired hypervolume difference between MORBO and random search on the same seed.
4. **Layer and demand effects:** where a configuration moves demand, not only area, and how it trades one KPI against another.
5. **Cost:** ray-tracing evaluations, ray-tracing time and wall-clock time per run.

The evaluation operates exclusively on archived run outputs without re-solving. It first verifies that all runs share the incumbent's scenario, grid, solver settings, bands, carrier frequencies and KPI definitions, then recomputes each recommendation's KPIs from its archived radio map to confirm that they were recorded correctly.

## VI. Results and Analysis

### A. Comparability and Consistency

All 23 comparability checks hold: both runs optimized the incumbent's scenario, retained their radio maps and match the incumbent's grid, solver settings, bands, carrier frequencies and KPI definitions. Recomputing each recommendation's KPIs from its archived radio map reproduces the recorded values exactly for the hole, weak and overlap rates and the objectives, and to within $1.1 \times 10^{-5}$ for the throughput KPIs (in Mbit/s) and the mean number of overlapping neighbours. This checks the bookkeeping, not the reproducibility of the search. Repeatability across seeds remains untested, as each method was executed with one seed.

### B. Network-Level KPIs

*Table IV. Recommended configuration of each method against the incumbent. Bold marks the best value per row. The tally counts the seven network KPIs, median and mean throughput separately although both derive from the same per-UE rates.*

| KPI | Direction | Incumbent | MORBO | Random search |
|---|:-:|---:|---:|---:|
| Coverage-hole rate | ↓ | 14.66 % | 13.83 % | **12.96 %** |
| Weak-coverage rate | ↓ | 38.8 % | 27.1 % | **24.4 %** |
| Co-band overlap rate | ↓ | 46.3 % | **43.1 %** | 51.0 % |
| Overlapping neighbours per covered tile | ↓ | 2.13 | **1.39** | 1.76 |
| Cell-edge throughput, p05 [Mbit/s] | ↑ | 0.0 | 0.0 | 0.0 |
| Median throughput [Mbit/s] | ↑ | 36.6 | **50.0** | 48.7 |
| Mean throughput [Mbit/s] | ↑ | 58.6 | **74.9** | 72.4 |
| Coverage objective $f_{\text{cov}}$ | ↑ | 0.8534 | 0.8617 | **0.8704** |
| Separation objective $f_{\text{sep}}$ | ↑ | 0.5260 | **0.6075** | 0.5544 |
| Throughput objective $f_{\text{thr}}$ | ↑ | 3.214 | 3.464 | **3.505** |
| Network KPIs better / worse / unchanged | | | 6 / 0 / 1 | 5 / 1 / 1 |
| Hypervolume of all evaluations | ↑ | 1.443 | **1.851** | 1.754 |

*Table V. Best-server RSRP and SINR per band over each band's covered tiles.*

| Band | Configuration | RSRP p50 [dBm] | RSRP p05 [dBm] | SINR p50 [dB] | SINR p05 [dB] |
|---|---|---:|---:|---:|---:|
| 2600 MHz | Incumbent | −97.7 | −107.6 | 4.8 | −3.46 |
| | MORBO | −92.8 | −107.7 | 9.2 | −2.22 |
| | Random search | −92.1 | −107.3 | 7.4 | −2.36 |
| 1800 MHz | Incumbent | −94.9 | −106.8 | 4.7 | −3.42 |
| | MORBO | −90.9 | −106.9 | 8.2 | −2.70 |
| | Random search | −90.0 | −106.3 | 6.8 | −2.68 |
| 700 MHz | Incumbent | −89.1 | −105.5 | 4.3 | −3.62 |
| | MORBO | −83.9 | −104.5 | 7.5 | −2.48 |
| | Random search | −83.8 | −103.7 | 5.1 | −3.35 |

![KPI comparison](../reports/figures/04_evaluation/kpi_comparison.png)

*Fig. 4. Network-level KPIs of each method's recommendation against the incumbent.*

![Objective comparison](../reports/figures/04_evaluation/objective_comparison.png)

*Fig. 5. Objectives of each method's recommendation against the incumbent.*

Four observations follow from Tables IV and V.

*MORBO lowers holes and overlap together.* Its recommendation reduces the hole rate by 0.83 points and the all-band overlap rate by 3.2 points, cuts the mean number of overlapping neighbours per covered tile by 35 %, and achieves an 11.6-point reduction in weak coverage. Median SINR rises on every band, by 4.4 dB at 2600 MHz, 3.5 dB at 1800 MHz and 3.2 dB at 700 MHz, and 5th-percentile SINR by 1.2, 0.7 and 1.1 dB. The 5th-percentile RSRP falls by about 0.1 dB at 2600 and 1800 MHz and rises by 1.0 dB at 700 MHz. Mean throughput rises by 16.2 Mbit/s (28 %). Of the seven network KPIs, six improve and none worsens; the seventh, cell-edge throughput, stays at zero.

*Random search buys coverage with overlap.* Its recommendation attains the lowest hole and weak-coverage rates, 1.70 and 14.4 points below the incumbent, but the all-band overlap rate rises by 4.6 points and the separation objective gains only 0.028 against MORBO's 0.081. Every band's own overlap rate falls (Table VIII) while the all-band rate rises; since the all-band rate counts a tile if any band overlaps there, the overlapping tiles of different bands must coincide less than before.

*Hole-rate changes of about a point lie near unmeasured noise.* Each recommendation is the best of 73 evaluations under one solver-noise realization (Section VII-B), and solver noise per KPI has not been measured, so hole-rate differences of this size should be read as indicative.

*Cell-edge throughput is pinned at zero.* Between 14.4 % and 17.6 % of UE reports lie on hole tiles in every configuration, exceeding 5 %, so the 5th-percentile throughput is 0 Mbit/s throughout and cannot discriminate between configurations.

### C. Search Effectiveness

*Table VI. Hypervolume per method (seed 42), against the origin on unnormalized objectives; the percentage gains depend on that reference point.*

| Method | Incumbent | Initial design | All evaluations | Gain over incumbent | Pareto points | Recommended evaluation |
|---|---:|---:|---:|---:|---:|---:|
| MORBO | 1.4428 | 1.7536 | 1.8507 | +28.3 % | 6 | 68 |
| Random search | 1.4428 | 1.7536 | 1.7540 | +21.6 % | 6 | 7 |

*Table VII. Best hypervolume and all-band overlap rate reached after a given number of evaluations. The best hole rate, 12.96 %, is reached by evaluation 10 in both runs, by a point of the shared initial design.*

| Evaluations | HV, MORBO | HV, random | Overlap rate, MORBO | Overlap rate, random |
|---:|---:|---:|---:|---:|
| 10 | 1.7537 | 1.7539 | 46.3 % | 46.3 % |
| 25 | 1.7680 | 1.7539 | 46.3 % | 46.3 % |
| 50 | 1.8320 | 1.7539 | 43.6 % | 46.3 % |
| 73 | 1.8507 | 1.7540 | 43.1 % | 46.3 % |

![Search progress](../reports/figures/04_evaluation/search_progress.png)

*Fig. 6. Best hypervolume found against the number of evaluations.*

![MORBO evaluations](../reports/figures/03b_morbo/morbo_evaluations.png)

*Fig. 7. Every MORBO evaluation, grouped by proposer (Sobol initial design or trust-region proposal).*

*The difference arises in the search, on this seed.* Because both methods share the same eight-point initial design, the difference between their final hypervolumes arises after it. From the shared 1.7536, MORBO adds 0.0971 and random search 0.0004: random search's 64 further points barely extend the front, and its recommendation is evaluation 7, a point of the shared initial design that also lies on MORBO's front. MORBO's final hypervolume exceeds random search's by 0.097 on the single seed pair; with one pair, no confidence interval or significance test can be reported, and the margin may not generalize across seeds.

*MORBO is more sample-efficient on this seed at a similar wall-clock time.* After 25 evaluations, MORBO's hypervolume already exceeds the one random search reaches after 73 (1.7680 against 1.7540). MORBO's hypervolume was still rising at the end, and its recommendation is evaluation 68 of 73, so the budget of 73 evaluations in 63 dimensions did not exhaust its progress. Its model overhead is offset here by cheaper ray tracing on the configurations it evaluated, so its wall-clock time exceeds random search's by about a tenth (Section VI-G).

*The front is mostly trust-region proposals.* Five of MORBO's six Pareto-optimal points are trust-region proposals; the sixth is the shared initial-design point with the lowest hole rate. The proposals exceed the initial design on average in all three objectives: mean coverage 0.8575 against 0.8559, mean separation 0.5867 against 0.5557 and mean throughput utility 3.422 against 3.369, whereas random search's later points match its initial design (0.8559, 0.5573 and 3.371). Because these proposals perturb the current Pareto points, any local search would be expected to show this. It does not isolate the contribution of the GP models, and no ablation without them was run.

*Only MORBO lowers the all-band overlap rate.* 32 of MORBO's 73 evaluations lower both the hole rate and the overlap rate below the incumbent's; none of random search's does, and its lowest overlap rate equals the incumbent's 46.3 %. On the three objectives, by contrast, the incumbent is a weak reference: 61 of MORBO's evaluations and 48 of random search's beat it on all three.

*The recommendation is not the minimum-hole configuration.* The lowest hole rate either method observed is 12.96 %, at a shared initial-design point whose overlap rate is 51.0 %, whereas MORBO's recommendation — the point with the largest hypervolume contribution — lies at 13.83 % with 43.1 % overlap. The remaining front points constitute alternatives for an operator who prioritizes hole reduction.

![Coverage vs separation trade-off](../reports/figures/04_evaluation/tradeoff_coverage_objective_vs_separation_objective.png)

*Fig. 8. All evaluated configurations projected onto the coverage and separation objectives, with each method's front and recommendation.*

### D. Per-Layer and Demand-Weighted Effects

*Table VIII. Per-band coverage-hole, weak-coverage and co-band overlap rates.*

| Band | Measure | Incumbent | MORBO | Random search |
|---|---|---:|---:|---:|
| 2600 MHz | Hole / weak / overlap | 35.7 / 45.5 / 28.8 % | 35.7 / 37.6 / 20.3 % | 34.3 / 36.7 / 23.6 % |
| 1800 MHz | Hole / weak / overlap | 28.9 / 44.6 / 32.2 % | 28.8 / 37.6 / 24.2 % | 27.5 / 36.3 / 28.5 % |
| 700 MHz | Hole / weak / overlap | 17.5 / 38.0 / 39.9 % | 16.8 / 26.3 / 29.5 % | 16.2 / 24.5 / 37.9 % |
| All bands | Overlap (KPI) | 46.3 % | 43.1 % | 51.0 % |

*Table IX. Coverage class by area and by demand (share of UE reports).*

| Coverage class | Incumbent: area | Incumbent: demand | MORBO: area | MORBO: demand | Random: area | Random: demand |
|---|---:|---:|---:|---:|---:|---:|
| Hole | 14.66 % | 17.56 % | 13.83 % | 15.76 % | 12.96 % | 14.41 % |
| Weak | 38.8 % | 25.3 % | 27.1 % | 19.9 % | 24.4 % | 18.5 % |
| Good | 46.6 % | 57.2 % | 59.0 % | 64.3 % | 62.7 % | 67.1 % |

*Table X. Distribution of overlapping co-band neighbours over covered tiles.*

| Configuration | Mean | 0 neighbours | 1 | 2 | 3 or more |
|---|---:|---:|---:|---:|---:|
| Incumbent | 2.13 | 45.7 % | 13.3 % | 8.1 % | 32.9 % |
| MORBO | 1.39 | 49.9 % | 20.1 % | 11.1 % | 18.8 % |
| Random search | 1.76 | 41.5 % | 20.0 % | 13.5 % | 25.1 % |

![Coverage before and after, 2600 MHz](../reports/figures/04_evaluation/coverage_before_after_b2600.png)

*Fig. 9. 2600 MHz best-server RSRP before and after MORBO's recommendation, and the tiles that crossed the hole threshold.*

![RSRP change maps, 2600 MHz](../reports/figures/04_evaluation/rsrp_change_maps_b2600.png)

*Fig. 10. Change in 2600 MHz best-server RSRP under each method's recommendation.*

*Hole gains reach demand.* MORBO's recommendation reduces hole area by 0.83 points and the share of UE reports on hole tiles by 1.80 points, from 17.56 % to 15.76 %; random search's reduces them by 1.70 and 3.14 points. The coverage and separation objectives are tile-uniform and do not weight by demand; the throughput objective does, through every UE report served at a positive rate.

*Weak coverage improves by both area and demand.* MORBO reduces the weak-coverage share by 11.6 points of area and 5.3 points of demand, and raises the good-coverage share by 12.5 and 7.1 points respectively.

*The coverage layer strengthens and every layer's overlap falls.* Under MORBO, the 700 MHz hole rate falls by 0.65 points and its weak rate by 11.6 points, while the 2600 MHz and 1800 MHz hole rates stay within 0.1 point of the incumbent's. Each band's own overlap rate falls, by 8.5 points at 2600 MHz, 8.0 at 1800 MHz and 10.3 at 700 MHz. Random search lowers every band's hole rate by more, but its 700 MHz overlap falls by only 2.0 points.

*Overlap becomes shallower under MORBO.* The share of covered tiles with three or more overlapping neighbours falls from 32.9 % to 18.8 %, and the share with none rises from 45.7 % to 49.9 %. Random search leaves 25.1 % of covered tiles with three or more and lowers the share with none to 41.5 %.

### E. Inter-Layer Load Redistribution

*Table XI. UE service and median served SINR per band.*

| Configuration | Served on 2600 MHz | Served on 1800 MHz | Served on 700 MHz | Not served | Served SINR p50, 2600 / 1800 / 700 MHz [dB] |
|---|---:|---:|---:|---:|---|
| Incumbent | 57.2 % | 12.0 % | 13.2 % | 17.6 % | 7.2 / 12.0 / 13.2 |
| MORBO | 53.0 % | 16.7 % | 14.6 % | 15.8 % | 13.1 / 14.9 / 14.2 |
| Random search | 49.4 % | 23.4 % | 12.8 % | 14.4 % | 9.6 / 14.6 / 16.3 |

![Serving band mix](../reports/figures/04_evaluation/serving_band_mix.png)

*Fig. 11. Serving-band mix per configuration.*

![Sector-band throughput](../reports/figures/04_evaluation/sector_band_throughput.png)

*Fig. 12. Median estimated throughput per sector-band, incumbent and recommended.*

*Traffic migrates from 2600 MHz to the lower layers.* Under MORBO's recommendation, the 2600 MHz share of UE reports falls by 4.2 points, while 1800 MHz gains 4.6 points and 700 MHz 1.3 points, and the unserved share falls by 1.8 points. Median served SINR rises on every layer, by 5.9 dB on 2600 MHz, 2.9 dB on 1800 MHz and 1.0 dB on 700 MHz.

*The largest changes are load-shedding downtilts on the capacity layer.* Sector n1s1 (site 1, sector 1) at 2600 MHz, downtilted by 3.6°, serves 347 fewer reports (413 → 66), and n6s2 at 2600 MHz, downtilted by 7.2°, serves 203 fewer (272 → 69); the median throughput of the UEs that remain rises by 19.8 and 104.6 Mbit/s respectively.

*Some uptilted capacity carriers serve more reports.* Sector n2s1 at 2600 MHz, uptilted by 7.3°, serves 257 additional reports (520 → 777) at a 6.5 dB higher median SINR, and n3s0 at 2600 MHz, uptilted by 11.1°, serves 142 additional reports. Which sectors previously served these reports was not traced. Ranking sector-bands by the magnitude of the change in served reports could give an operator a prioritized monitoring list for a staged rollout; this ranking is computed for the recommendation only and has not been tested in a rollout.

### F. Recommended Tilt Configuration

![Tilt change heatmap](../reports/figures/04_evaluation/tilt_delta_heatmap.png)

*Fig. 13. Tilt change per sector and band in MORBO's recommendation.*

*Table XII. Tilt movement of MORBO's recommendation. Negative $\Delta$ denotes an uptilt.*

| Band | Sectors moved | Mean $\lvert\Delta\rvert$ [°] | Largest $\lvert\Delta\rvert$ [°] | Mean $\Delta$ [°] |
|---|---:|---:|---:|---:|
| 2600 MHz | 21 of 21 | 5.94 | 11.9 | −0.49 |
| 1800 MHz | 21 of 21 | 5.35 | 11.5 | −1.19 |
| 700 MHz | 21 of 21 | 5.79 | 11.7 | −2.72 |

All 63 tilts change, 36 upward and 27 downward, by up to 11.9°, with resulting tilts between 0.1° and 19.8°. No tilt reaches a bound, so the bounds are not active at the recommendation. The 700 MHz layer moves furthest toward the horizon on average (mean −2.7°), which is consistent with extending the coverage layer, whereas 2600 MHz nets out near zero (−0.5°) through large opposing per-sector moves, accompanied by the load shifts of Section VI-E. Because movement is unpenalized, the recommendation corresponds to a 63-antenna RET change request. The tilt movement of the other Pareto-optimal configurations was not analyzed, so whether the front contains a comparable option with less movement remains open.

### G. Computational Cost

*Table XIII. Search cost on a single 4 GB laptop-class GPU (NVIDIA RTX 3050).*

| Method | Evaluations | Recommended evaluation | Ray tracing [min] | Wall clock [min] |
|---|---:|---:|---:|---:|
| MORBO | 73 | 68 | 5.69 | 10.50 |
| Random search | 73 | 7 | 8.24 | 9.54 |

Ray tracing costs about 4.7 s per full-network evaluation in MORBO's run and 6.8 s in random search's; the cost depends on the configurations evaluated, and why it differs between them was not investigated. MORBO spends a further 4.8 min outside the ray tracer, in GP fitting and acquisition optimization, against 1.3 min for random search. A complete joint optimization of 63 tilts therefore requires about ten minutes of commodity compute in this setting. This covers the simulation only; field validation of the selected configuration comes on top of it.

## VII. Discussion

### A. Implications for RAN Operation

Subject to the limitations below, the results suggest three possible changes to tilt-optimization practice. First, joint optimization can complement trial-and-error with selection among simulated alternatives: a single run produced six Pareto-optimal configurations, each accompanied by its full KPI vector and per-antenna tilt deltas, from which an engineer can select according to operational priorities. Second, the recommended configuration moves all bands jointly: it uptilts the coverage layer on average while making opposing per-sector moves on the capacity layer, and load shifts across sectors and frequency layers accordingly. Whether a band-by-band procedure would reach a similar configuration was not tested. Third, the per-sector impact ranking, computed here for the recommendation only, could inform a staged rollout plan with an explicit monitoring order.

In the studied layout, tilt optimization lowered the coverage-hole and overlap rates together, but the hole-rate gain is under a point at the recommendation: most of the hole area lies more than 1 km from every site, where the hole rate is 17.4 % against 0.94 % within 1 km, and 3,996 hole tiles have no propagation path to any sector in the ray tracer as configured. Tilt search therefore acts mainly on weak coverage, overlap, SINR and inter-layer load balance near the sites, while the distant holes are a site-planning problem. With diffraction disabled, it remains open how many of the no-path tiles are truly unreachable rather than an artefact of the propagation settings.

### B. Limitations and Threats to Validity

- **Single, uncalibrated scenario.** One scene, one UE realization and one layout were studied; no result has been validated on a held-out scenario or calibrated against drive-test measurements. The simulator is not a digital twin of any deployed network.
- **Propagation settings.** Diffraction and diffuse scattering are disabled, which can turn shadowed tiles into no-path holes; the hole findings depend on this choice.
- **Partial UMa layout.** The layout follows the UMa inter-site distance and mast height [11] but uses 7 of its 19 sites, so most of the 6.2 × 6.5 km scene lies beyond the first tier and its holes are out of any tilt's reach. UEs lie on a single outdoor plane at 1.5 m; the indoor, multi-floor UE distribution of UMa is not modelled. The incumbent is the uniform 12° calibration tilt rather than an operator-tuned configuration.
- **Antenna model.** 3GPP specifies the 12° incumbent as an electrical downtilt, whereas the simulator tilts the whole array, side and back lobes included. The $8 \times 8$ array with uniform weights has its first vertical null about 14.5° off boresight; at tilts of roughly 14° to 18° that null sweeps the cell edge, so results for sectors tilted into that range partly reflect the antenna model rather than the network.
- **Limited baselines.** MORBO is compared only with random search and the uniform 12° incumbent. Neither an operator-tuned configuration, a band-by-band procedure nor another optimizer, such as an evolutionary algorithm or reinforcement learning, was evaluated. The benefit of joint over band-by-band optimization is therefore not isolated.
- **Equal evaluations, not equal time.** The methods are compared at 73 evaluations each; MORBO took about a tenth more wall-clock time, and a comparison at equal wall-clock time was not run.
- **No ablation.** The contribution of the GP models, as opposed to local perturbation within a trust region, was not separated.
- **Synthetic traffic and SINR biases.** The traffic is synthetic, and hotspots are placed where building volume is high, so the demand on hole tiles depends on that placement rule; one of the three hotspots lies 1.9 km from the nearest site. SINR assumes every co-band sector transmits at full power, which is pessimistic at low load, and omits a receiver noise figure, which is optimistic.
- **Idealized serving rule.** UEs attach to the sector-band with the highest equal-share Shannon rate, so the inter-layer load results describe this rule, not a deployed band-selection policy. Mobility and handover are not modelled.
- **Single seed per method.** No confidence interval, significance test or repeatability measure is available for Tables IV and VI.
- **Winner's curse and solver noise.** All evaluations share one solver seed, and each recommendation is the best of 73 under that noise realization, so its scores are biased upward. Solver noise per KPI is unmeasured; hole-rate changes of about a point may lie within it.
- **Reference-point dependence.** Hypervolume is computed against the origin; a reference point at the incumbent, or a selection restricted to configurations dominating the incumbent, could yield a different recommendation. Here the choice is close: MORBO's recommendation and the shared initial-design point that random search recommends differ in hypervolume contribution by about 0.1 %.
- **Algorithmic configuration.** A single trust region was used and the budget of 73 evaluations in 63 dimensions is small; MORBO's recommendation is its 68th evaluation, so it was still improving at the end of the budget. The original method's default number of trust regions was not evaluated.
- **Objective–KPI mismatch on overlap.** The separation objective is a soft per-tile product over bands, whereas the all-band overlap KPI counts any crowded band. Both improve under MORBO's recommendation, but random search's raises separation while the all-band overlap rate worsens, so improving one does not guarantee the other.
- **Demand-agnostic area objectives.** Coverage and separation are tile-uniform; that the share of UE reports on hole tiles fell here (Table IX) follows from where this population's demand lies, not from the objectives.
- **Simplified capacity model.** Rates are equal-share Shannon bounds without scheduling, MCS limits or mobility. Inter-band interference is absent by construction, and cell-edge throughput is 0 Mbit/s in every configuration because more than 5 % of UE reports lie on hole tiles.
- **Unpenalized movement.** The recommendation moves all 63 tilts (Table XII), which may exceed a practical RET change window.

## VIII. Conclusion and Future Work

This paper formulated the joint configuration of tilts across all sectors and bands of a multi-band network as a 63-dimensional, three-objective black-box problem evaluated by an uncalibrated ray-traced network simulator, and compared MORBO with Sobol random search under a matched evaluation budget and seed. Table XIV summarizes the outcome against the assessment criteria.

*Table XIV. Summary against the assessment criteria, seed 42. Hypervolume is against the origin on unnormalized objectives.*

| Criterion | Random search | MORBO |
|---|---|---|
| 1. Hypervolume | +21.6 % over incumbent, +0.0004 over the shared initial design | **+28.3 % over incumbent**, +0.0971 over the shared initial design |
| 2. Reported KPIs | 5 better, 1 worse; overlap +4.6 points | **6 better, 0 worse**; overlap −3.2 points |
| 3. Search effectiveness | Recommendation is an initial-design point; no evaluation lowers both hole and overlap rate | **Exceeds random search's final hypervolume by evaluation 25**; 32 evaluations lower both hole and overlap rate; 5 of 6 Pareto points are trust-region proposals |
| 4. Layer and demand effects | Lowest hole rate by area and demand; all-band overlap rises | Weak coverage reduced by area and demand; every band's overlap falls; hole-tile demand share −1.8 points |
| 5. Cost | 73 evaluations, 9.5 min | 73 evaluations, 10.5 min |

The principal conclusions are:

1. In the studied layout, the uniform 12° incumbent has a hole rate of 14.7 % before any optimization, 17.4 % beyond 1 km of a site and 0.94 % within it, and 3,996 of its 14,818 hole tiles receive no propagation path in the ray tracer as configured, without diffraction or diffuse scattering.
2. MORBO's recommendation, biased upward as the best of 73 under one noise realization, lowers the hole rate by 0.83 points and the all-band overlap rate by 3.2 points, and attains 11.6 points less weak coverage, 3.2–4.4 dB higher median SINR per band and 16.2 Mbit/s higher mean throughput, with no network KPI worse than the incumbent's.
3. At an equal number of evaluations and on one seed per method, MORBO reached a larger hypervolume than random search in about a tenth more wall-clock time; random search added almost nothing beyond the shared initial design. Whether this margin holds across seeds, or at equal wall-clock time, is untested.
4. For this synthetic population, the area-based hole rate understates the service deficit: between 14 % and 18 % of UE reports lie on hole tiles in every configuration. One demand hotspot, placed by the generator where building volume is high, is centred 1.9 km from the nearest site, and its centre remains a hole under every recommended configuration.

Future work will (i) quantify solver noise by re-tracing recommendations under multiple solver seeds; (ii) execute multiple search seeds per method to obtain confidence intervals and significance tests; (iii) raise the evaluation budget in proportion to the 63 dimensions; (iv) weight the coverage objective by demand, or report a demand-weighted hole rate alongside it; (v) extend the layout to the full 19-site UMa grid with indoor, multi-floor UEs, and model electrical tilt with the parametric antenna pattern of TR 38.901; (vi) introduce a tilt-movement penalty or constraint so that recommendations fit practical RET change windows; (vii) compare against an operator-tuned configuration and further optimizers; and (viii) incorporate a scheduler-level capacity model and validate the approach against operator measurements.

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

[11] 3GPP TR 38.901 V19.2.0 (ETSI TR 138 901 V19.2.0, 2026-02), *Study on channel model for frequencies from 0.5 to 100 GHz*, Table 7.2-1 (evaluation parameters for UMa scenarios) and Table 7.8-1 (large-scale calibration parameters).

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

[22] 3GPP TS 37.320, *Radio measurement collection for Minimization of Drive Tests (MDT); Overall description; Stage 2*, Release 19, Annex A.
