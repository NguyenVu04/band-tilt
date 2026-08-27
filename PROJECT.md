# PROJECT.md — Multi-Band Tilt Coordination for Coverage-Efficient 5G/6G RAN

## 1. Project Overview

This project studies the problem of **multi-band tilt coordination** for 5G/6G Radio Access Networks (RANs).

The objective is to determine an **optimal absolute tilt configuration** for each `(cell, band)` pair such that multiple frequency bands cooperate to:

1. minimize **coverage holes (Hole)**;
2. minimize **coverage overlap (Overlap)**;
3. minimize **weak coverage (Weak)**;
4. reduce overlap severity by minimizing the number of neighboring cells simultaneously overlapping with the serving cell;
5. encourage higher-priority bands to dominate in areas with higher UE density.

Two optimization approaches are investigated and compared under the same problem formulation:

- **Bayesian Optimization (BO)**;
- **Multi-Agent Reinforcement Learning (MARL)**.

Because running Sionna-RT for every candidate tilt configuration can be computationally expensive, a **surrogate model** is introduced to approximate the relationship between tilt configurations, radio-map characteristics, and network KPIs.

---

# 2. Research Objective

## 2.1. Primary Objective

Given a multi-cell, multi-band RAN deployment, determine:

\[
\boldsymbol{\theta}^{*}
=
[
\theta_{1,1}^{*},
\theta_{1,2}^{*},
\ldots,
\theta_{N,B}^{*}
]
\]

where:

- \(N\) is the number of cells;
- \(B\) is the number of frequency bands;
- \(\theta_{i,b}\) is the absolute tilt of cell \(i\) on band \(b\).

The optimal configuration should improve coverage quality while coordinating the spatial roles of different frequency bands.

## 2.2. Optimization Priority

The primary coverage objectives follow the strict priority:

\[
\boxed{
Hole > Overlap > Weak
}
\]

Therefore:

1. eliminating or reducing holes is the highest priority;
2. reducing overlap is the second priority;
3. reducing weak-coverage areas is the third priority.

After satisfying the primary coverage objectives, the optimization should favor an appropriate spatial distribution of frequency bands according to UE density.

---

# 3. Core Design Decisions

## 3.1. Absolute Tilt Is the Optimization Variable

The optimizer directly outputs **absolute tilt**:

\[
\boxed{
\theta_{i,b}
}
\]

subject to:

\[
\boxed{
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}
}
\]

for every cell-band pair.

The optimizer does **not** directly output tilt offsets.

### Rationale

Using tilt offset as the optimization action,

\[
a_{i,b}=\Delta\theta_{i,b},
\]

would make the feasible action range depend on the current tilt:

\[
\Delta\theta_{i,b}
\in
[
\theta_{i,b}^{min}-\theta_{i,b}^{current},
\theta_{i,b}^{max}-\theta_{i,b}^{current}
].
\]

Consequently, the action space changes with the current network configuration.

With absolute tilt:

\[
a_{i,b}=\theta_{i,b},
\]

the action space is fixed:

\[
a_{i,b}
\in
[
\theta_{i,b}^{min},
\theta_{i,b}^{max}
].
\]

This formulation is more natural for both BO and MARL.

---

## 3.2. Tilt Offset Is a Derived Reporting Quantity

After optimization, the tilt offset is calculated as:

\[
\boxed{
\Delta\theta_{i,b}
=
\theta_{i,b}^{*}
-
\theta_{i,b}^{current}
}
\]

Tilt offset is:

- not an optimization variable;
- not a KPI;
- not part of the objective function.

It is retained only to communicate how much each tilt has changed from the current configuration.

The final result should therefore report:

| Cell | Band | Current Tilt | Optimal Tilt | Tilt Offset |
|---|---|---:|---:|---:|
| Cell 1 | Band 1 | \(\theta^{current}\) | \(\theta^*\) | \(\Delta\theta^*\) |
| Cell 1 | Band 2 | \(\theta^{current}\) | \(\theta^*\) | \(\Delta\theta^*\) |
| ... | ... | ... | ... | ... |

---

## 3.3. Tilt-Change Magnitude Is Not Optimized

The objective does not include:

\[
|\Delta\theta|
\]

or any explicit penalty for changing the tilt.

The rationale is that the research objective is network optimization rather than minimizing the amount of configuration change.

The physical tilt bounds already constrain the solution:

\[
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}.
\]

If a real deployment later requires a maximum reconfiguration step, that should be modeled as an **operational constraint**, rather than as a network-quality KPI.

---

## 3.4. Accessibility Is Excluded

**Accessibility is intentionally excluded** from the current formulation.

It is not used as:

- an optimization KPI;
- an objective term;
- a dominant-band criterion.

The current formulation focuses on radio coverage, overlap, and UE-aware band coordination.

---

# 4. System Inputs

The system inputs are divided into three major groups.

## 4.1. 3D Environment

The simulation environment should contain the information required to construct the Sionna-RT scene, including:

- 3D map of the target area;
- building and infrastructure geometry;
- terrain information when available;
- electromagnetic material properties of relevant surfaces;
- coordinate system;
- spatial reference information.

These data determine the propagation environment used by Sionna-RT.

---

## 4.2. Radio Configuration

For each cell-band pair, the configuration should contain the parameters required to construct the transmitter and antenna model in Sionna-RT, such as:

- antenna position;
- antenna height;
- antenna characteristics;
- carrier frequency;
- transmit power;
- azimuth;
- electrical tilt (`eTilt`);
- mechanical tilt (`mTilt`);
- other radio parameters required by the simulator.

The absolute radio tilt is defined as:

\[
\boxed{
\mathrm{tilt}
=
\mathrm{eTilt}
+
\mathrm{mTilt}
}
\]

Therefore:

\[
\theta_{i,b}
=
\mathrm{eTilt}_{i,b}
+
\mathrm{mTilt}_{i,b}.
\]

The optimization variable is the resulting absolute tilt \(\theta_{i,b}\).

---

## 4.3. MDT Data

The minimum MDT record contains:

```text
ue_id
sim_x
sim_y
date
rsrp
gcell_id
ue_height
```

Additional fields may be incorporated later if required.

MDT data are primarily used to:

- characterize UE spatial distribution;
- estimate UE density;
- identify areas with high UE concentration;
- support surrogate modeling;
- evaluate UE-aware band coordination.

---

# 5. Coordinate and Angle Conventions

The current implementation uses the following conversion between the radio configuration/MDT convention and the Sionna-RT simulation convention:

```python
yaw = deg2rad(90.0 - row.azimuth)
pitch = deg2rad(-row.tilt)
```

Therefore:

\[
\boxed{
yaw
=
\operatorname{deg2rad}
(90^\circ-azimuth)
}
\]

and:

\[
\boxed{
pitch
=
\operatorname{deg2rad}
(-tilt)
}
\]

Under this convention, a positive radio downtilt is represented by a negative pitch in the simulation coordinate system.

The conversion must be applied consistently throughout data preprocessing, scene construction, radio-map generation, and validation.

---

# 6. Sionna-RT Radio-Map Generation

For a given absolute tilt configuration:

\[
\boldsymbol{\theta},
\]

Sionna-RT is used to generate the received signal distribution over a spatial grid:

\[
\mathcal{G}
=
\{g_1,g_2,\ldots,g_M\}.
\]

For cell \(i\), band \(b\), and location \(g\):

\[
R_{i,b}(g;\boldsymbol{\theta})
=
\mathrm{RSRP}_{i,b}(g;\boldsymbol{\theta}).
\]

The basic simulation pipeline is:

\[
\boxed{
\boldsymbol{\theta}
\rightarrow
\mathrm{Sionna\text{-}RT}
\rightarrow
\mathrm{Radio\ Map}
\rightarrow
\mathrm{RSRP}
}
\]

The resulting radio maps provide the ground truth used to calculate the network KPIs.

---

# 7. UE Spatial Distribution

The target area is discretized into spatial grids:

\[
g\in\mathcal{G}.
\]

Let:

\[
\rho(g)
\]

denote the number of UE observations assigned to grid \(g\).

If all grids have equal area:

\[
\rho(g)=n_g.
\]

The total number of UE observations is:

\[
\boxed{
N_{UE}
=
\sum_{g\in\mathcal{G}}\rho(g)
}
\]

The UE distribution is used to give greater importance to areas containing more users when evaluating band priority.

---

# 8. Serving Cell and Dominant Band

## 8.1. Serving Cell

The serving cell at a location \(x\) is determined from the strongest cell signal:

\[
\boxed{
s(x)
=
\operatorname*{arg\,max}_{i}
R_i(x)
}
\]

where \(R_i(x)\) represents the RSRP used for serving-cell selection under the adopted multi-band evaluation rule.

Accessibility is not included in this formulation.

---

## 8.2. Dominant Cell-Band

For UE-aware band coordination, the dominant cell-band at grid \(g\) is:

\[
\boxed{
(i^*(g),b^*(g))
=
\operatorname*{arg\,max}_{i,b}
R_{i,b}(g)
}
\]

and the corresponding dominant band is:

\[
b^*(g).
\]

Thus, the current definition of dominant band is based directly on the largest RSRP among the considered cell-band combinations.

---

# 9. Primary KPI Set

The optimization uses exactly **five primary KPIs**:

1. **Hole Rate**;
2. **Weak Rate**;
3. **Overlap Rate**;
4. **Mean Overlap Neighbors**;
5. **UE-weighted Band Priority Score**.

Their optimization directions are:

| KPI | Objective |
|---|---|
| Hole Rate | Minimize |
| Overlap Rate | Minimize |
| Weak Rate | Minimize |
| Mean Overlap Neighbors | Minimize |
| UE-weighted Band Priority Score | Maximize |

The coverage priority remains:

\[
\boxed{
Hole > Overlap > Weak
}
\]

---

# 10. KPI 1 — Hole Rate

At location \(x\), define the strongest received signal:

\[
R_{max}(x)
=
\max_{i,b}R_{i,b}(x).
\]

A location is classified as a hole when:

\[
R_{max}(x)\leq-120\ \mathrm{dBm}.
\]

Define:

\[
H(x)
=
\mathbb{I}
[
R_{max}(x)\leq-120
].
\]

The Hole Rate is:

\[
\boxed{
K_H
=
\mathrm{HoleRate}
=
\frac{
\sum_{x\in\mathcal{G}}H(x)
}{
|\mathcal{G}|
}
\times100\%
}
\]

Objective:

\[
\boxed{
K_H\rightarrow\min
}
\]

Hole Rate is the highest-priority coverage KPI.

---

# 11. KPI 2 — Weak Rate

A location is classified as weak coverage when:

\[
-120<R_{max}(x)\leq-90\ \mathrm{dBm}.
\]

Define:

\[
W(x)
=
\mathbb{I}
[
-120<R_{max}(x)\leq-90
].
\]

The Weak Rate is:

\[
\boxed{
K_W
=
\mathrm{WeakRate}
=
\frac{
\sum_{x\in\mathcal{G}}W(x)
}{
|\mathcal{G}|
}
\times100\%
}
\]

Objective:

\[
\boxed{
K_W\rightarrow\min
}
\]

Weak coverage has lower priority than Hole and Overlap.

---

# 12. KPI 3 — Overlap Rate

Let the serving cell at location \(x\) be:

\[
s(x)
=
\operatorname*{arg\,max}_i R_i(x).
\]

A neighbor \(j\) is considered overlapping with the serving cell when:

\[
R_{s(x)}(x)>-120\ \mathrm{dBm}
\]

and:

\[
R_{s(x)}(x)-R_j(x)<6\ \mathrm{dB}.
\]

The number of overlapping neighbors is:

\[
\boxed{
N_{ov}(x)
=
\sum_{j\in\mathcal{N}_{s(x)}}
\mathbb{I}
\left[
R_{s(x)}(x)>-120
\land
R_{s(x)}(x)-R_j(x)<6
\right]
}
\]

A location is classified as an overlap location if:

\[
O(x)
=
\mathbb{I}
[
N_{ov}(x)>0
].
\]

The Overlap Rate is:

\[
\boxed{
K_O
=
\mathrm{OverlapRate}
=
\frac{
\sum_{x\in\mathcal{G}}O(x)
}{
|\mathcal{G}|
}
\times100\%
}
\]

Objective:

\[
\boxed{
K_O\rightarrow\min
}
\]

---

# 13. KPI 4 — Mean Overlap Neighbors

Overlap Rate measures how frequently overlap occurs, but does not describe how many neighbors participate in each overlap region.

Therefore:

\[
\boxed{
K_{ON}
=
\mathrm{MeanOverlapNeighbors}
=
\frac{
\sum_{x\in\mathcal{G}}N_{ov}(x)
}{
\sum_{x\in\mathcal{G}}O(x)
}
}
\]

This KPI is evaluated over locations where overlap actually occurs.

Objective:

\[
\boxed{
K_{ON}\rightarrow\min
}
\]

This provides an additional measure of overlap severity.

---

# 14. KPI 5 — UE-weighted Band Priority Score

Each frequency band \(b\) is assigned a positive priority weight:

\[
\boxed{
w_b>0
}
\]

The weight represents the desired importance of the band in spatial coverage allocation.

For example, a conceptual configuration could be:

\[
w_{\mathrm{low}}<w_{\mathrm{mid}}<w_{\mathrm{high}}.
\]

The exact values are hyperparameters and must be defined before optimization experiments.

For grid \(g\), the dominant band is:

\[
b^*(g)
=
b
\left(
\operatorname*{arg\,max}_{i,b}
R_{i,b}(g)
\right).
\]

The UE-weighted Band Priority Score is:

\[
\boxed{
K_{BPS}
=
\mathrm{BPS}
=
\frac{
\sum_{g\in\mathcal{G}}
\rho(g)w_{b^*(g)}
}{
\sum_{g\in\mathcal{G}}\rho(g)
}
}
\]

Objective:

\[
\boxed{
K_{BPS}\rightarrow\max
}
\]

## Interpretation

The number of UE observations acts as a spatial importance weight.

For example:

```text
Grid A: 100 UE, high-band dominant, weight = 3
Grid B:  10 UE, low-band dominant,  weight = 1
```

Grid A contributes substantially more to the final score.

Therefore, the optimizer is encouraged to make higher-priority bands dominant in areas with higher UE concentration.

---

# 15. KPI Vector and Optimization Direction

The complete KPI vector is:

\[
\boxed{
\mathbf{K}
=
[
K_H,
K_O,
K_W,
K_{ON},
K_{BPS}
]
}
\]

with:

\[
\boxed{
K_H\downarrow
}
\]

\[
\boxed{
K_O\downarrow
}
\]

\[
\boxed{
K_W\downarrow
}
\]

\[
\boxed{
K_{ON}\downarrow
}
\]

\[
\boxed{
K_{BPS}\uparrow
}
\]

The first three KPIs describe coverage quality, while Mean Overlap Neighbors characterizes overlap severity and Band Priority Score describes UE-aware multi-band coordination.

---

# 16. Optimization Formulation

The decision vector is:

\[
\boxed{
\boldsymbol{\theta}
=
[
\theta_{1,1},
\theta_{1,2},
\ldots,
\theta_{N,B}
]^T
}
\]

The optimization problem can be formulated as a multi-objective or lexicographic optimization:

\[
\boxed{
\boldsymbol{\theta}^{*}
=
\arg\operatorname{opt}_{\boldsymbol{\theta}}
\left(
K_H,
K_O,
K_W,
K_{ON},
-K_{BPS}
\right)
}
\]

subject to:

\[
\boxed{
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}
}
\]

for all:

\[
i\in\mathcal{C},
\qquad
b\in\mathcal{B}.
\]

---

# 17. Lexicographic Priority

Because the research requirement explicitly states:

\[
Hole>Overlap>Weak,
\]

the preferred conceptual formulation is:

\[
\boxed{
\min_{\boldsymbol{\theta}}^{lex}
\left(
K_H,
K_O,
K_W,
K_{ON},
-K_{BPS}
\right)
}
\]

The optimization therefore follows:

1. minimize Hole Rate;
2. among acceptable solutions, minimize Overlap Rate;
3. then minimize Weak Rate;
4. reduce the number of overlapping neighbors;
5. maximize UE-weighted Band Priority Score.

If a scalar objective is required for a specific optimizer, the KPI terms should first be normalized and then combined with coefficients chosen so that the intended priority order is preserved.

---

# 18. Surrogate Modeling

Directly evaluating every candidate configuration with Sionna-RT can be expensive.

A direct optimization loop would be:

```text
Optimizer
    ↓
Sionna-RT
    ↓
Radio Map
    ↓
KPI
    ↓
Optimizer
```

The proposed architecture introduces a surrogate:

```text
Sionna-RT
    ↓
Radio Maps
    ↓
Training Dataset
    ↓
Surrogate Model
    ↓
Fast KPI Prediction
```

The surrogate dataset is:

\[
\boxed{
\mathcal{D}_{sur}
=
\{
(
\mathbf{s}^{(k)},
\boldsymbol{\theta}^{(k)},
\mathbf{K}^{(k)}
)
\}_{k=1}^{N_{sur}}
}
\]

where:

- \(\mathbf{s}^{(k)}\) represents the network/environment state;
- \(\boldsymbol{\theta}^{(k)}\) is an absolute tilt configuration;
- \(\mathbf{K}^{(k)}\) contains the KPI values obtained from Sionna-RT.

The surrogate learns:

\[
\boxed{
f_{sur}
:
(\mathbf{s},\boldsymbol{\theta})
\rightarrow
\widehat{\mathbf K}
}
\]

where:

\[
\widehat{\mathbf K}
=
[
\widehat K_H,
\widehat K_O,
\widehat K_W,
\widehat K_{ON},
\widehat K_{BPS}
].
\]

The surrogate is an acceleration mechanism, not the final source of truth.

---

# 19. Overall Research Pipeline

## Step 1 — Data Exploration

Investigate:

- MDT schema;
- UE distribution;
- RSRP distribution;
- cell distribution;
- band distribution;
- missing values;
- outliers;
- coordinate ranges;
- temporal distribution.

---

## Step 2 — Data Cleaning

Perform:

- invalid-record removal;
- missing-value handling;
- duplicate checking;
- coordinate validation;
- RSRP validation;
- cell-ID validation;
- UE-height validation;
- coordinate convention normalization.

---

## Step 3 — Train/Test Split

Split the data into training and test sets.

The split should avoid information leakage, particularly when MDT observations are correlated spatially or temporally.

Depending on the experimental objective, spatial and/or temporal splitting should be considered instead of relying only on random splitting.

---

## Step 4 — Construct the Sionna-RT Scene

Build the simulation scene from:

```text
3D Map
+
Building Geometry
+
Material Properties
+
Antenna Configuration
+
Radio Configuration
+
Absolute Tilt
```

---

## Step 5 — Generate Radio Maps

Sample valid absolute tilt configurations:

\[
\boldsymbol{\theta}^{(1)},
\boldsymbol{\theta}^{(2)},
\ldots,
\boldsymbol{\theta}^{(N)}.
\]

For each configuration:

\[
\boldsymbol{\theta}^{(k)}
\rightarrow
\mathrm{Sionna\text{-}RT}
\rightarrow
R^{(k)}
\rightarrow
\mathbf{K}^{(k)}.
\]

---

## Step 6 — Train the Surrogate

Train:

\[
f_{sur}
(\mathbf{s},\boldsymbol{\theta})
\rightarrow
\widehat{\mathbf K}.
\]

Evaluate:

- prediction error;
- generalization;
- error for each KPI;
- error across different regions of the tilt search space;
- prediction reliability near promising candidate configurations.

---

# 20. Bayesian Optimization Pipeline

BO treats the optimization problem as an expensive black-box problem:

\[
J(\boldsymbol{\theta})
\]

subject to:

\[
\boldsymbol{\theta}\in\Theta
\]

where:

\[
\Theta
=
\left\{
\boldsymbol{\theta}
\mid
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}
\right\}.
\]

The conceptual loop is:

```text
Initial Samples
      ↓
Sionna-RT / Existing Dataset
      ↓
Surrogate
      ↓
Acquisition Function
      ↓
Candidate Absolute Tilt
      ↓
Sionna-RT Evaluation
      ↓
KPI
      ↓
Update Surrogate
      ↓
Repeat
```

The final BO solution is:

\[
\boldsymbol{\theta}_{BO}^{*}.
\]

When the surrogate is used during optimization, important candidate solutions should still be validated using Sionna-RT.

---

# 21. Multi-Agent Reinforcement Learning Pipeline

MARL models the optimization as a multi-agent control problem.

A possible mapping is:

- one agent per cell; or
- one agent per site.

The exact agent granularity depends on the desired coordination structure.

Each agent may generate the absolute tilts of its bands:

\[
a_i
=
[
\theta_{i,1},
\theta_{i,2},
\ldots,
\theta_{i,B}
].
\]

A policy can be represented as:

\[
\boxed{
a_i
=
\pi_{\phi_i}(s_i,c_i)
}
\]

where:

- \(s_i\) is the local state;
- \(c_i\) represents coordination information, when communication or a centralized critic is used.

All actions must satisfy:

\[
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}.
\]

---

# 22. MARL State

The state can contain information describing the local radio environment and the effect of the cell's tilt.

Potential state features include:

- current absolute tilt;
- neighboring-cell RSRP characteristics;
- local Hole/Weak/Overlap statistics;
- local UE density;
- dominant-band distribution;
- band identity;
- spatial context;
- coordination information from neighboring agents.

The exact state representation is an implementation choice and should be validated experimentally.

---

# 23. MARL Reward

The reward must reflect:

\[
Hole>Overlap>Weak
\]

while encouraging:

\[
BPS\uparrow.
\]

A possible normalized scalar reward is:

\[
\boxed{
r_t
=
-\lambda_H\widetilde K_H
-\lambda_O\widetilde K_O
-\lambda_W\widetilde K_W
-\lambda_{ON}\widetilde K_{ON}
+\lambda_{BPS}\widetilde K_{BPS}
}
\]

where each \(\widetilde K\) is a normalized KPI.

The coefficients must be chosen so that KPI scaling does not accidentally reverse the desired priority:

\[
\lambda_H>\lambda_O>\lambda_W.
\]

If strict priority is required, a hierarchical or lexicographic reward formulation may be preferable to a simple weighted sum.

---

# 24. Surrogate-Assisted MARL

Instead of evaluating every environment step with Sionna-RT:

```text
Agent
 ↓
Absolute Tilt
 ↓
Sionna-RT
 ↓
KPI
 ↓
Reward
```

the training loop can use:

```text
Agent
 ↓
Absolute Tilt
 ↓
Surrogate
 ↓
Predicted KPI
 ↓
Reward
```

Sionna-RT remains the high-fidelity environment used for:

- initial dataset generation;
- validation;
- policy evaluation;
- surrogate error monitoring;
- surrogate updates.

---

# 25. BO vs. MARL Comparison

BO and MARL should be evaluated using the same:

- input data;
- radio-map environment;
- action/search space;
- tilt constraints;
- KPI definitions;
- surrogate;
- validation procedure;
- test scenarios.

## 25.1. Solution Quality

Compare:

\[
K_H,\quad
K_O,\quad
K_W,\quad
K_{ON},\quad
K_{BPS}.
\]

## 25.2. Computational Efficiency

Measure:

- number of Sionna-RT evaluations;
- total optimization time;
- number of candidate configurations;
- surrogate inference cost;
- training cost for MARL.

## 25.3. Stability

Run each method using multiple random seeds and report:

- mean;
- standard deviation;
- best result;
- worst result.

## 25.4. Scalability

Evaluate performance as the problem size increases:

- number of cells;
- number of bands;
- geographical area;
- number of decision variables.

---

# 26. Final Validation

The final optimized configuration:

\[
\boldsymbol{\theta}^{*}
\]

must be evaluated again using Sionna-RT.

Validation flow:

```text
Optimal Absolute Tilt
        ↓
Sionna-RT
        ↓
Ground-Truth Radio Map
        ↓
Compute Five KPIs
        ↓
Compare with Surrogate Prediction
```

The final reported network performance must therefore be based on high-fidelity Sionna-RT results rather than surrogate predictions alone.

---

# 27. Final Result Reporting

## 27.1. Tilt Configuration

For every cell-band pair, report:

\[
\theta_{i,b}^{current}
\]

\[
\theta_{i,b}^{*}
\]

and:

\[
\Delta\theta_{i,b}
=
\theta_{i,b}^{*}
-
\theta_{i,b}^{current}.
\]

The recommended final table is:

| Cell | Band | Current Tilt | Optimal Tilt | Tilt Offset |
|---|---|---:|---:|---:|
| ... | ... | ... | ... | ... |

---

## 27.2. KPI Comparison

Compare:

```text
Baseline
vs.
BO
vs.
MARL
```

using:

| KPI | Baseline | BO | MARL |
|---|---:|---:|---:|
| Hole Rate | | | |
| Overlap Rate | | | |
| Mean Overlap Neighbors | | | |
| Weak Rate | | | |
| UE-weighted Band Priority Score | | | |

---

## 27.3. Radio-Map Visualization

Recommended visualizations include:

- baseline RSRP map;
- optimized RSRP map;
- Hole map;
- Weak map;
- Overlap map;
- UE-density map;
- dominant-band map.

These visualizations help demonstrate not only numerical KPI improvement but also the spatial behavior of the optimized network.

---

# 28. End-to-End Architecture

```text
                 ┌──────────────────────┐
                 │       3D Map         │
                 │ Geometry + Material  │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Radio Configuration  │
                 │ Cell / Band / Antenna│
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │       MDT Data       │
                 │ UE / Position / RSRP │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 │ + UE Density         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Sionna-RT       │
                 │   Radio-map Engine   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Radio-map Dataset   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Surrogate Model    │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 │                      │
                 ▼                      ▼
        ┌────────────────┐     ┌────────────────┐
        │       BO       │     │      MARL      │
        └───────┬────────┘     └───────┬────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                 ┌──────────────────────┐
                 │ Absolute Tilt θ*     │
                 │ per Cell × Band      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Sionna-RT Validation│
                 └──────────┬───────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │        Five Main KPIs        │
              │ Hole / Weak / Overlap        │
              │ Mean Overlap Neighbors       │
              │ UE-weighted Band Priority    │
              └──────────────┬───────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │  Final Result Report │
                 │ Absolute Tilt       │
                 │ Tilt Offset          │
                 │ KPI Comparison       │
                 └──────────────────────┘
```

---

# 29. Confirmed Design Decisions

| Component | Final Decision |
|---|---|
| Optimization variable | **Absolute tilt** |
| Tilt offset | **Derived reporting quantity only** |
| Tilt definition | \(\mathrm{tilt}=\mathrm{eTilt}+\mathrm{mTilt}\) |
| Tilt constraint | \(\theta^{min}\leq\theta\leq\theta^{max}\) |
| Delta-tilt constraint | Not included in the current formulation |
| Tilt-change penalty | **Not used** |
| Accessibility | **Not used** |
| Serving-cell criterion | RSRP-based |
| Dominant-band criterion | Maximum RSRP among cell-band pairs |
| Hole threshold | \(R_{max}\leq-120\) dBm |
| Weak range | \(-120<R_{max}\leq-90\) dBm |
| Overlap threshold | Serving RSRP - neighbor RSRP \(<6\) dB |
| Overlap coverage condition | Serving RSRP \(>-120\) dBm |
| Primary KPI count | **5** |
| Coverage priority | **Hole > Overlap > Weak** |
| Overlap severity | Mean Overlap Neighbors |
| Band coordination KPI | UE-weighted Band Priority Score |
| UE weighting | UE count/density per spatial grid |
| Separate high-band dominance KPI | **Not required** |
| Simulator | **Sionna-RT** |
| Computational acceleration | **Surrogate modeling** |
| Optimization methods | **BO + MARL** |
| Final validation | **Sionna-RT ground truth** |

---

# 30. Parameters Still Requiring Experimental Definition

The following parameters are not yet fixed by the formulation and must be determined during implementation and experimentation:

1. number and identity of frequency bands;
2. band priority weights \(w_b\);
3. minimum and maximum tilt for each cell-band;
4. spatial grid resolution;
5. precise serving-cell aggregation rule across bands;
6. radio-map sampling strategy;
7. surrogate-model architecture;
8. KPI normalization method;
9. scalar objective or reward design;
10. MARL algorithm;
11. BO surrogate and acquisition function;
12. optimization stopping criteria;
13. surrogate update strategy;
14. evaluation scenarios and random seeds.

These should be treated as **implementation and experimental choices**, rather than as fixed elements of the core problem formulation.

---

# 31. Final Mathematical Formulation

The complete optimization problem is:

\[
\boxed{
\begin{aligned}
\boldsymbol{\theta}^{*}
&=
\arg\operatorname{opt}_{\boldsymbol{\theta}}
\left(
K_H,
K_O,
K_W,
K_{ON},
-K_{BPS}
\right)
\\
\text{s.t.}\quad
&
\theta_{i,b}^{min}
\leq
\theta_{i,b}
\leq
\theta_{i,b}^{max}.
\end{aligned}
}
\]

The five objectives are:

\[
\boxed{
K_H=\mathrm{HoleRate}\rightarrow\min
}
\]

\[
\boxed{
K_O=\mathrm{OverlapRate}\rightarrow\min
}
\]

\[
\boxed{
K_W=\mathrm{WeakRate}\rightarrow\min
}
\]

\[
\boxed{
K_{ON}
=
\mathrm{MeanOverlapNeighbors}
\rightarrow\min
}
\]

\[
\boxed{
K_{BPS}
=
\mathrm{UE\text{-}weighted\ Band\ Priority\ Score}
\rightarrow\max
}
\]

The tilt offset is derived only after optimization:

\[
\boxed{
\Delta\boldsymbol{\theta}^{*}
=
\boldsymbol{\theta}^{*}
-
\boldsymbol{\theta}^{current}
}
\]

The resulting research pipeline is therefore:

\[
\boxed{
\text{MDT + 3D Environment + Radio Configuration}
\rightarrow
\text{Sionna-RT}
\rightarrow
\text{Radio Maps}
\rightarrow
\text{Surrogate}
\rightarrow
\text{BO / MARL}
\rightarrow
\text{Absolute Tilt}
\rightarrow
\text{Sionna-RT Validation}
\rightarrow
\text{Five KPIs}
}
\]

This formulation establishes a common optimization problem for both BO and MARL, with absolute tilt as the decision variable, explicit physical tilt constraints, UE-aware multi-band coordination, and high-fidelity Sionna-RT validation as the final source of truth.
