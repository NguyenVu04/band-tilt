# 6. A radio-and-load CVaR objective

- **Status:** Accepted
- **Date:** 2026-09-17
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the desirability objective and `kpi.weights` of the deleted
  ADRs 0004 and 0005; partly, the weighted score of
  [ADR 0003](0003-turbo-on-a-weighted-kpi-score.md) and the objective role of
  [ADR 0001](0001-four-kpis-and-weighted-score.md)'s KPIs
- **Superseded by:** —

## Context

The search maximised a Derringer weighted geometric mean of three per-tile
desirabilities (coverage, co-band separation, served ratio per tile). Load
entered only through blocking in the served ratio, so a configuration that
packed cells to their PRB limit scored the same as one that spread the traffic.
The weights were judgement values with no physical reading, and the layout of
three corner nodes left the middle of the service area to cell-edge coverage.

## Decision

**The objective** (`src/optim/objective.py`) is

```
J = J_radio^gamma · J_load^(1 − gamma)

J_radio = mean_g  sigmoid((R_s(g) − T_cov) / tau_R) · q_ov^m_g
m_g     = sum_b #{ j ≠ s_b(g) on band b : R_j > T_cov and R_s_b − R_j <= Delta_R }
J_load  = 1 − CVaR_alpha([rho − rho_0]_+) / (1 − rho_0)
```

- `g` is every grid tile, equally weighted (`w_g = 1`). `s(g)` is the strongest
  cell-band at the tile; `s_b(g)` is the strongest transmitter on band `b`, and
  counts only where `R_s_b > T_cov`. Neighbours are counted **on every band,
  co-band only**: exactly the count the overlap rate thresholds
  (`src.kpi.overlap.overlap_neighbors`).
- `T_cov = kpi.hole_dbm` and `Delta_R = kpi.overlap_margin_db`: one threshold
  per physical quantity, shared with the reported KPIs.
- `q_ov = exp(−beta)`.
- `rho` is one cell-band's admitted PRBs over its `max_prb` in one
  interval. The CVaR is the mean of the largest `ceil((1 − alpha) · N)` of all
  (cell-band, interval) samples, idle ones included. The search serves the MDT
  (the UEs admitted at the committed tilts); evaluation serves every UE.
- `tau_R`, `beta`, `rho_0`, `alpha`, `gamma` live in `kpi.objective`.

**The terms are stored, not `J`.** `j_radio` and `j_load` are columns beside
the five reported KPIs, so a stored history can be re-scored under another
`gamma`. Every other parameter is inside the measurement.

**Admission cap.** A cell-band admits a UE only while its load is at most
`kpi.capacity.max_admission_utilisation` of `max_prb` and the UE still fits;
otherwise the UE passes to its next candidate.

**Layout.** A fourth node stands on the centroid of the corner triangle, and
the cell fans start at azimuth 0°.

**Reported KPIs** are hole rate, overlap rate, served ratio, weak rate and
cell-edge RSRP. The desirabilities, `kpi.weights` and the weight-sensitivity
table are deleted.

## Consequences

**Positive**

- Load enters the objective directly, through its tail, so relieving the
  busiest cell-bands scores even when nothing is blocked.
- Every parameter has a physical or statistical reading (dB width, per-neighbour
  retention, utilisation target, tail share, trade-off).

**Negative**

- `m_g` is a hard count, so `J_radio` steps as a neighbour crosses `Delta_R`.
- With `rho_0` equal to the admission cap, `rho` exceeds `rho_0` by at most
  one UE's PRBs, so `J_load` stays near one unless the two are set apart.
- The parameters are judgement values with no sensitivity analysis.
- Runs recorded before this change lack `j_radio` and `j_load` and cannot be
  loaded as a `KpiVector`; the layout change also invalidates every earlier
  radio map.

## Alternatives considered

**Keep the desirability index and add a load desirability.** Rejected: a fourth
weight, and a mean over cells hides the overloaded tail the CVaR targets.

**Cross-band overlap neighbours.** Rejected for consistency with the overlap
rate: carriers of one cell on different bands do not interfere.

**Counting neighbours on the serving band only.** Rejected: crowding on a band
that does not serve the tile was invisible to the objective while the overlap
rate still reported it.
