# 3. Coverage and separation searched by MORBO, recommended by hypervolume contribution

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** [ADR 0001](0001-turbo-on-a-weighted-kpi-score.md) (TuRBO, highest-score
  selection) and sections 1 and 3 of
  [ADR 0002](0002-contraharmonic-objective-and-kpi-set.md) (the objective `J` and the
  twelve-KPI set). Sections 2 and 4 of 0002 (max-throughput sector selection over an
  equal PRB share, and the Shannon-bound capacity model) stand.
- **Superseded by:** —

## Context

`J` folded coverage strength and co-band crowding into one contraharmonic score
per tile. One number hid the trade an engineer actually makes: on the 2026-10-06
runs, the winner raised `J` while the band-collapsed overlap rate rose, and the
throughput the UEs receive was reported but read by nothing in the search. `J`
was also not monotone in the layers present, so a tilt could gain by darkening a
weak layer.

The UE service failure rate and the throughput statistics split the UEs along
one mask: a UE on a hole left the throughput statistics instead of pulling them
down, so closing a hole under a UE could lower the median throughput.

## Decision

### 1. Two searched objectives and a recorded one, each maximised

With `R_bs(g)` band `b`'s strongest sector at tile `g` and `i` the other co-band
sectors above `kpi.hole_dbm`:

- **Coverage** `mean_g [1 - prod_b (1 - sigma(R_bs - weak_dbm))]`,
  `sigma(x) = 1 / (1 + 10^(-x / 10))`. The logistic is centred on `kpi.weak_dbm`:
  centred on 0 dBm, as first written, it is about `10^(RSRP / 10)` and vanishes
  over the whole grid.
- **Separation** `mean_g prod_b 1 / (1 + sum_i 10^((m - (R_bs - R_bi)) / 10))`,
  `m = kpi.overlap_margin_db`. Co-band, so the overlap rule of the reported KPIs
  and this objective agree; a band with no server above `kpi.hole_dbm` counts 1,
  leaving holes to coverage.
- **Throughput** `mean_u ln(1 + R_u)` over every UE report, `R_u` in Mbit/s. A
  proportional-fair utility: it rewards lifting slow UEs more than fast ones.

Coverage and separation are searched (`OBJECTIVE_NAMES`). Throughput is measured
and recorded with every candidate, but no search reads it until it is reviewed.
*Amended 2026-10-08:* throughput is reviewed and searched as the third objective.

Each is rounded to 6 significant digits before a search reads it, for the same
reason `J` was rounded: GPU accumulation order varies the trailing digits.

### 2. Unserved UEs count 0 Mbit/s; the failure rate is removed

The serving rule writes 0 Mbit/s, not NaN, for a UE with no sector-band above
`kpi.hole_dbm`, so the throughput KPIs and the throughput objective see it. The
UE service failure rate is dropped: it duplicated the hole rate measured where
the UEs stand. The reported set is eleven KPIs. While more than 5 % of reports
stand on holes, `estimated_throughput_p05_mbps` reads 0; it is kept and reported
as such.

### 3. MORBO searches, hypervolume ranks

MORBO (Daulton et al., UAI 2022) replaces TuRBO-1: trust regions centred on the
largest hypervolume contribution, local GPs per objective (Matérn-5/2 with ARD,
under BoTorch's dimension-scaled log-normal lengthscale prior; without a prior
the fit failed on the first ray-traced run) fitted to shared data, and a batch chosen by Thompson-sampled hypervolume
improvement. Hyperparameters follow the paper's Appendix D; one trust region by
default (`trust_region.count`). A collapsed region restarts at the point a
random hypervolume scalarisation ranks best in one draw of a global GP over the
initial design and earlier restart points, as Algorithm 1 and the authors'
reference code do.

Hypervolume is taken against the origin, every objective's floor, so every
non-dominated point contributes. Each run's recommended configuration is the
evaluated point with the largest hypervolume contribution, the rule MORBO uses
for its centres; the shortlist is the rest of the Pareto front in that order.

### 4. Tilts on a 0.1° lattice

`optim.tilt_resolution_deg` snaps every proposal of every method, so the
deliverable is settable on an antenna and the methods search the same space.

## Consequences

- Results from before this record are not comparable: the KPI vector changed
  shape, and `KpiVector.from_mapping` refuses an old `run.json`.
- The recommendation depends on the reference point. Against the origin, the
  contribution of a point at the front's extreme extends to the reference planes,
  so extreme points can be recommended over a balanced one.
- The search no longer has a scalar to plot: progress is hypervolume.
- `src/evaluation` computes hypervolume in numpy, so a comparison still needs no
  `bo` extra; MORBO uses BoTorch's box decomposition for candidate improvement.

## Alternatives considered

- **Keep `J` and add throughput as a constraint.** One more threshold to choose,
  and the coverage–crowding trade stays hidden in `J`.
- **Reference point at the incumbent.** Only configurations better than today on
  all three would count; with a small budget and a random design mostly worse than
  the incumbent, the front above it is often empty early on.
- **Knee point as the recommendation.** Needs a normalisation of the objectives
  on different scales; the hypervolume contribution is scale-free.
