# 6. A radio coverage objective

- **Status:** Accepted
- **Date:** 2026-09-17
- **Rewritten:** 2026-09-17 — rewritten in place at the maintainer's direction.
  The first version combined this coverage term with a CVaR load term
  (`J = J_radio^gamma · J_load^(1 − gamma)`) and searched on the MDT; that text
  is in Git history.
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the desirability objective and `kpi.weights` of the deleted
  ADRs 0004 and 0005; partly, the weighted score of
  [ADR 0003](0003-turbo-on-a-weighted-kpi-score.md) and the objective role of
  [ADR 0001](0001-four-kpis-and-weighted-score.md)'s KPIs
- **Superseded by:** —

## Context

The search maximised a Derringer weighted geometric mean of three per-tile
desirabilities (coverage, co-band separation, served ratio per tile). The
weights were judgement values with no physical reading, and the layout of three
corner nodes left the middle of the service area to cell-edge coverage.

The first version of this record added a load term: the CVaR of PRB
utilisation above a target, weighted against coverage by `gamma` and scored on
the MDT (the UEs served at the committed tilts). It was removed at the
maintainer's direction. The drawbacks recorded against it at the time: with the
target equal to the admission cap it stayed near one, it added three judgement
parameters, and the MDT holds none of the UEs in holes, so the search and the
all-UE re-score could rank the shortlist differently.

## Decision

**The objective** (`src/optim/objective.py`, the `objective` measure) is

```
J   = mean_g  sigmoid((R_s(g) − T_cov) / tau_R) · q_ov^m_g
m_g = sum_b #{ j ≠ s_b(g) on band b : R_j > T_cov and R_s_b − R_j <= Delta_R }
```

- `g` is every grid tile, equally weighted. `s(g)` is the strongest cell-band at
  the tile; `s_b(g)` is the strongest transmitter on band `b`, and counts only
  where `R_s_b > T_cov`. Neighbours are counted **on every band, co-band only**:
  exactly the count the overlap rate thresholds
  (`src.kpi.overlap.overlap_neighbors`).
- `T_cov = kpi.hole_dbm` and `Delta_R = kpi.overlap_margin_db`: one threshold
  per physical quantity, shared with the reported KPIs.
- `q_ov = exp(−beta)`. `tau_R` and `beta` live in `kpi.objective`.

**Every UE counts.** The served ratio is measured on every UE of
`data.output.ue_file`, in the search and in evaluation alike. The MDT is still
built, and kept for later use; nothing scores on it.

**Serving order.** Within an interval, UEs are admitted in `t_s` order: a
cell fills in the order its reports arrive, not best-first. Reports at the same
instant are taken strongest RSRP first over every layer at the UE, and row order
breaks what remains. A cell-band admits a UE only while its load is at most
`kpi.capacity.max_admission_utilisation` of `max_prb` and the UE still fits;
otherwise the UE passes to its next candidate.

**Layout.** A fourth node stands on the centroid of the corner triangle, and
every node's cell fan is rotated by
`simulation.transmitters.layout.azimuth_offset_deg` (45°).

**Reported KPIs** are hole rate, overlap rate, served ratio, weak rate and
cell-edge RSRP, stored beside `objective`.

## Consequences

**Positive**

- Two parameters, each with a physical reading (dB width, per-neighbour
  retention), and no trade-off weight.
- The objective depends on the radio map alone, so the population a run is
  scored on cannot change which configuration wins. This is what makes the
  serving rule's own randomness harmless to the search: `t_s` is drawn per UE
  (`src/simulation/sample.py`), so admission order is a random draw, but it
  reaches only the reported served ratio, never `J`.

**Negative**

- Load is not in the objective. A tilt that packs cell-bands to their PRB limit
  scores the same as one that spreads the traffic; the served ratio and the
  utilisation tables report it.
- `m_g` is a hard count, so `J` steps as a neighbour crosses `Delta_R`.
- The parameters are judgement values with no sensitivity analysis.
- Every tile counts equally, including tiles where no UE stands.
- Runs recorded before this change carry `j_radio` and `j_load`, not
  `objective`, and cannot be loaded as a `KpiVector`.

## Alternatives considered

**Keep the CVaR load term.** Removed at the maintainer's direction; see Context.

**Keep the desirability index and add a load desirability.** Rejected: a fourth
weight, and a mean over cells hides the overloaded tail.

**Cross-band overlap neighbours.** Rejected for consistency with the overlap
rate: carriers of one cell on different bands do not interfere.

**Counting neighbours on the serving band only.** Rejected: crowding on a band
that does not serve the tile was invisible to the objective while the overlap
rate still reported it.
