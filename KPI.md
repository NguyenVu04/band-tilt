# KPI Definitions

This document describes the five evaluation KPIs currently used for the multi-band tilt coordination problem. The definitions below follow the current LaTeX specification and use only the KPI concepts already defined in that document.

## 1. KPI Priority

The optimization uses the following lexicographic priority, from highest to lowest:

\[
\boxed{
K_H \succ K_O \succ K_{EI} \succ K_{BPS} \succ K_W
}
\]

where:

- \(K_H\): Hole Rate — **minimize**
- \(K_O\): Overlap Rate — **minimize**
- \(K_{EI}\): Expected RSRP Improvement — **maximize**
- \(K_{BPS}\): UE-weighted Band Priority Score — **maximize**
- \(K_W\): Weak Rate — **minimize**

The symbol \(\succ\) represents optimization priority and is not a numerical comparison between KPI values.

---

## 2. Hole Rate

### 2.1 Definition

At a spatial evaluation position \(g\), let

\[
R_{\max}(g)
=
\max_{(i,b)\in\mathcal{C}\times\mathcal{B}} R_{i,b}(g)
\]

be the strongest RSRP among all cell--band pairs.

A coverage hole occurs when

\[
R_{\max}(g) \leq -120\,\mathrm{dBm}.
\]

The hole indicator is therefore

\[
H(g)
=
\mathbb{I}
\left[
R_{\max}(g)\leq -120\,\mathrm{dBm}
\right].
\]

The Hole Rate is

\[
\boxed{
K_H
=
\mathrm{HoleRate}
=
\frac{
\displaystyle\sum_{g\in\mathcal{G}} H(g)
}{
|\mathcal{G}|
}
\times 100\%.
}
\]

### 2.2 Objective

\[
\boxed{\min K_H}
\]

A lower Hole Rate means fewer locations without sufficient signal coverage.

---

## 3. Overlap Rate

### 3.1 Per-Band Serving Layer

Overlap is evaluated **within each band independently**. At each evaluation position \(g\), band \(b\) has its own serving cell:

\[
i_b(g)
=
\arg\max_{i\in\mathcal{C}}
R_{i,b}(g),
\]

and its own serving RSRP

\[
S_b(g)
=
\max_{i\in\mathcal{C}} R_{i,b}(g).
\]

There are \(|\mathcal{B}|\) serving cells at each position, one per band, not a single globally strongest cell--band pair.

### 3.2 Co-Band Constraint

A cell--band pair \((j,c)\) can only be a neighbor of the serving cell of its own band \(c\). Pairs on different bands are never compared against each other, so two carriers of one cell are never neighbors of each other, and a strong layer on another band is not an overlapping neighbor.

Every band contributes its own count, and the counts are summed. A band on which the serving cell has a close competitor therefore raises \(N_{\mathrm{ov}}(g)\) even when a different band is the strongest layer at that position.

### 3.3 Overlapping Neighbor

Within band \(b\), a cell \(j\neq i_b(g)\) contributes to overlap when all of the following conditions hold:

1. That band's serving RSRP is above the hole threshold:
   \[
   S_b(g)>-120\,\mathrm{dBm}.
   \]

2. The neighboring RSRP is above the hole threshold:
   \[
   R_{j,b}(g)>-120\,\mathrm{dBm}.
   \]

3. The RSRP difference between that band's serving cell and the neighbor does not exceed \(6\,\mathrm{dB}\):
   \[
   S_b(g)-R_{j,b}(g)\leq6\,\mathrm{dB}.
   \]

The number of overlapping neighbors at \(g\) is the sum over bands

\[
\begin{aligned}
N_{\mathrm{ov}}(g)
={}&
\sum_{b\in\mathcal{B}}
\sum_{j\in\mathcal{C}\setminus\{i_b(g)\}}
\mathbb{I}[S_b(g)>-120\,\mathrm{dBm}]
\
&\times
\mathbb{I}[R_{j,b}(g)>-120\,\mathrm{dBm}]
\
&\times
\mathbb{I}
[
S_b(g)-R_{j,b}(g)\leq6\,\mathrm{dB}
].
\end{aligned}
\]

A band on which nothing is received contributes zero: it has nothing to overlap with.

A location is classified as overlapping when

\[
N_{\mathrm{ov}}(g)>0.
\]

The overlap indicator is

\[
O(g)
=
\mathbb{I}[N_{\mathrm{ov}}(g)>0].
\]

### 3.4 KPI

The Overlap Rate is

\[
\boxed{
K_O
=
\mathrm{OverlapRate}
=
\frac{
\displaystyle\sum_{g\in\mathcal{G}}O(g)
}{
|\mathcal{G}|
}
\times100\%.
}
\]

### 3.5 Objective

\[
\boxed{\min K_O}
\]

A lower Overlap Rate means fewer evaluation locations contain excessive same-band coverage overlap.

---

## 4. Expected RSRP Improvement

### 4.1 Purpose

Expected RSRP Improvement evaluates the expected change in serving RSRP after applying a new tilt configuration. It compares the current serving RSRP observed in MDT measurements with the serving RSRP expected after the tilt change.

Let \(\mathcal{U}\) be the set of MDT measurement locations.

For each \(u\in\mathcal{U}\):

- \(R_{\mathrm{real}}(u)\) is the current serving RSRP measured by the UE.
- \(R_{\mathrm{sim}}(u)\) is the serving RSRP predicted after applying the candidate tilt configuration.
- \(\tau>0\) is the sigmoid sensitivity parameter.

Both serving values are defined as the strongest RSRP at the corresponding location.

### 4.2 RSRP Difference

The expected change is

\[
\Delta R(u)
=
R_{\mathrm{sim}}(u)-R_{\mathrm{real}}(u).
\]

The change is mapped to \((0,1)\) using a sigmoid function:

\[
I_{\mathrm{RSRP}}(u)
=
\sigma
\left(
\frac{
R_{\mathrm{sim}}(u)-R_{\mathrm{real}}(u)
}{
\tau
}
\right),
\]

where

\[
\sigma(x)
=
\frac{1}{1+e^{-x}}.
\]

Equivalently,

\[
\boxed{
I_{\mathrm{RSRP}}(u)
=
\frac{1}{
1+
\exp\left(
-\dfrac{
R_{\mathrm{sim}}(u)-R_{\mathrm{real}}(u)
}{
\tau
}
\right)
}.
}
\]

### 4.3 KPI

The Expected RSRP Improvement is the mean value over the MDT measurement set:

\[
\boxed{
K_{EI}
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
I_{\mathrm{RSRP}}(u).
}
\]

The KPI satisfies

\[
0<K_{EI}<1.
\]

When

\[
R_{\mathrm{sim}}(u)>R_{\mathrm{real}}(u),
\]

the corresponding improvement score is greater than \(0.5\). Conversely, when the expected serving RSRP decreases, the score is below \(0.5\).

### 4.4 Objective

\[
\boxed{\max K_{EI}}
\]

A higher value indicates greater expected serving-RSRP improvement over the MDT measurement locations.

---

## 5. UE-Weighted Band Priority Score

### 5.1 Purpose

The UE-weighted Band Priority Score (BPS) evaluates how well the serving-band distribution matches the intended band priorities and the spatial UE distribution.

The current definition assigns each band \(b\) an initial priority weight \(w_b\). Higher-priority bands receive larger weights.

Let:

- \(\mathcal{G}\) be the spatial evaluation set.
- \(\mathcal{B}\) be the set of bands.
- \(N_{g,b}\) be the number of UEs at location or region \(g\) served by band \(b\).
- \(w_b\) be the initial priority weight of band \(b\).

The weights are normalized to \([0,1]\):

\[
\boxed{
\widetilde{w}_b
=
\frac{
w_b-w_{\min}
}{
w_{\max}-w_{\min}
}
}
\]

with

\[
w_{\min}=\min_{b\in\mathcal{B}}w_b,
\qquad
w_{\max}=\max_{b\in\mathcal{B}}w_b.
\]

### 5.2 KPI

The UE-weighted Band Priority Score is

\[
\boxed{
K_{BPS}
=
\frac{
\displaystyle
\sum_{g\in\mathcal{G}}
\sum_{b\in\mathcal{B}}
\widetilde{w}_bN_{g,b}
}{
\displaystyle
\sum_{g\in\mathcal{G}}
\sum_{b\in\mathcal{B}}
N_{g,b}
}.
}
\]

The KPI satisfies

\[
K_{BPS}\in[0,1].
\]

It represents the average normalized priority weight of the bands serving the UEs.

### 5.3 Objective

\[
\boxed{\max K_{BPS}}
\]

A higher BPS means that UEs are, on average, served by bands with higher assigned priority weights.

---

## 6. Weak Rate

### 6.1 Definition

A location is classified as weak when the strongest available RSRP is above the hole threshold but does not exceed the defined strong-coverage threshold:

\[
-120\,\mathrm{dBm}
<
R_{\max}(g)
\leq
-90\,\mathrm{dBm}.
\]

The weak indicator is

\[
W(g)
=
\mathbb{I}
\left[
-120\,\mathrm{dBm}
<
R_{\max}(g)
\leq
-90\,\mathrm{dBm}
\right].
\]

### 6.2 KPI

The Weak Rate is

\[
\boxed{
K_W
=
\mathrm{WeakRate}
=
\frac{
\displaystyle\sum_{g\in\mathcal{G}}W(g)
}{
|\mathcal{G}|
}
\times100\%.
}
\]

### 6.3 Objective

\[
\boxed{\min K_W}
\]

A lower Weak Rate means fewer locations fall within the defined weak-coverage range.

---

## 7. Overall Lexicographic Evaluation

The five KPIs are not treated as a simple weighted sum. Their optimization priority is explicitly ordered:

\[
\boxed{
K_H
\succ
K_O
\succ
K_{EI}
\succ
K_{BPS}
\succ
K_W
}
\]

with the optimization directions:

| Priority | KPI | Direction | Meaning |
|---:|---|---|---|
| 1 | Hole Rate \(K_H\) | Minimize | Reduce coverage holes |
| 2 | Overlap Rate \(K_O\) | Minimize | Reduce same-band coverage overlap |
| 3 | Expected RSRP Improvement \(K_{EI}\) | Maximize | Improve expected serving RSRP |
| 4 | BPS \(K_{BPS}\) | Maximize | Prefer higher-priority bands for UE service |
| 5 | Weak Rate \(K_W\) | Minimize | Reduce weak-coverage locations |

For an optimization formulation in which all objectives are converted to minimization, the objective vector is

\[
\boxed{
\mathbf{J}(\mathbf{tilt})
=
\left[
K_H,\,
K_O,\,
-K_{EI},\,
-K_{BPS},\,
K_W
\right].
}
\]

The optimal tilt configuration is therefore represented as

\[
\boxed{
\mathbf{tilt}^{*}
=
\arg\min_{\mathbf{tilt}\in\mathcal{X}}^{\mathrm{lex}}
\mathbf{J}(\mathbf{tilt}).
}
\]

The feasible tilt space remains subject to the operational constraint

\[
\boxed{
\mathrm{tilt}_{i,b}^{\min}
\leq
\mathrm{tilt}_{i,b}
\leq
\mathrm{tilt}_{i,b}^{\max}
}
\qquad
\forall i,b.
\]

Thus, a candidate configuration must first satisfy the highest-priority KPI before improvements in lower-priority KPIs can determine the preference between otherwise comparable configurations.
