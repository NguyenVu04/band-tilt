# 7. A demand-weighted objective, a load ceiling, and the reported KPI set

- **Status:** Accepted
- **Date:** 2026-09-18
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the tile-uniform objective and the reported KPI list of
  [ADR 0006](0006-radio-coverage-objective.md); the admission rule and the
  five-KPI list of [ADR 0001](0001-four-kpis-and-weighted-score.md)
- **Superseded by:** —

## Context

Three things were wrong with what ADR 0006 left in place, and all three are
recorded in its own Consequences section.

**Every tile counted equally.** The objective averaged its per-tile utility over
the whole grid, so 101 060 tiles of open ground and the four hotspots carrying
most of the traffic had the same say. A tilt that filled a hole where nobody
stands scored exactly like one that filled a hole in a hotspot. The simulator
already produces the measurement that would fix this — the MDT is the set of
reports the network actually served — and nothing read it.

**The admission cap did not cap anything.** `kpi.capacity.max_admission_utilisation`
gated admission on the load a cell-band *already* carried: a cell-band at 0.69
of its limit still admitted a UE that took it to 1.0, because the second test
was against `max_prb` and not against the share. The measure a deployment cares
about — the load a cell ends the interval at — was therefore unbounded by the
setting named after it.

**The reported set was thin.** Five measures, with no visibility of SINR, of how
hard any cell worked, or of whether the traffic sat evenly across the network.
A multi-band study also could not say which *layer* a number came from: every
KPI was a whole-network scalar.

## Decision

### 1. The demand map

`src/data/demand.py`, built by `task preprocess` into
`data.output.demand_file`, in two steps.

```
d_g   = median{ sum of prb_per_ue on tile g in interval t : t in T_g }
w_g   = (1 - u) * K_h * d / ||K_h * d||_1  +  u / |G|
```

- `T_g` is the set of intervals in which tile `g` carries **at least one** MDT
  report. Not every interval of the horizon: no tile on this scenario is
  reported in more than 16 of 672 intervals, so a median over all of them is
  zero for every tile and carries no information.
- `prb_per_ue` is the PRBs the serving rule required for that report, and the
  MDT now carries it (`src/simulation/mdt.py`). A tile never reported has
  `d_g = 0`, which is a statement about the sample and not about the tile.
- `K_h` is an isotropic Gaussian kernel of standard deviation
  `data.demand.bandwidth_m`. On a regular lattice a Gaussian KDE evaluated at
  tile centres, over samples aggregated per tile, *is* the discrete Gaussian
  convolution of `d`, so that is how it is computed. Kernel mass falling off the
  grid edge is dropped and the result renormalised: a KDE restricted to the
  study area.
- `u` is `data.demand.uniform_share`, a floor under ground the MDT never
  reported. It is 0 in the committed config.

### 2. The objective

```
J   = sum_g w_g * sigmoid((R_s(g) - T_cov) / tau_R) * q_ov^m_g,   sum_g w_g = 1
m_g = sum_b #{ j != s_b(g) on band b : R_j > T_cov and R_s_b - R_j <= Delta_R }
```

Everything but `w_g` is ADR 0006 unchanged: `R_s` the strongest cell-band at the
tile, `m_g` the co-band neighbour count `overlap_rate` thresholds,
`T_cov = kpi.hole_dbm`, `Delta_R = kpi.overlap_margin_db`, `q_ov = exp(-beta)`.
Setting `uniform_share` to 1 recovers ADR 0006's tile mean exactly, which is how
the two are compared.

### 3. The admission ceiling

A cell-band admits a UE only when

```
PRB_required + PRB_current <= max_admission_utilisation * max_prb
```

so the cap is a **ceiling on the resulting load**, not a gate on the load before
it. No cell-band ever ends an interval above that share, and
`kpi.capacity.max_admission_utilisation` is set to **0.8**. `prb_utilisation_max`
is bounded by it by construction; a value above it is a fault, not a finding.

### 4. The reported KPIs

Eleven measures, stored beside `objective`, in this order:

| KPI | Definition | Direction |
|---|---|---|
| `hole_rate` | share of tiles with `R_max <= kpi.hole_dbm` | minimise |
| `overlap_rate` | share of tiles with any co-band neighbour within `Delta_R` | minimise |
| `overlap_neighbor_mean` | mean `m_g` over covered tiles | minimise |
| `weak_rate` | share of tiles with `hole_dbm < R_max <= weak_dbm` | minimise |
| `rsrp_p05_dbm` | 5th percentile of `R_max` over covered tiles | **maximise** |
| `rsrp_p50_dbm` | median of the same | **maximise** |
| `sinr_p05_db` | 5th percentile of the best server's SINR over covered tiles | **maximise** |
| `sinr_p50_db` | median of the same | **maximise** |
| `served_rate` | share of UE reports the serving rule admitted | **maximise** |
| `prb_utilisation_max` | highest cell-band load in any interval, over its limit | minimise |
| `load_imbalance` | coefficient of variation of cell-band utilisation | minimise |

`load_imbalance` is the population standard deviation over the mean of each
cell-band's interval-averaged utilisation. Scale-free: it does not move when the
whole network gets busier, only when the traffic sits unevenly.

**Per band.** Every one of the eleven is also reported per frequency layer
(`src.evaluation.compare.band_kpis`), by giving the same function one band's
slice of the radio map. There is no second definition. `objective` is not
reported per band: it scores a network, and one layer of a multi-band network is
not a network.

**The PRB series.** `prb_usage_by_time` reports the PRBs and utilisation of every
cell-band in every interval — the series `prb_utilisation_max` and
`load_imbalance` reduce.

## Choosing the parameters

Four judgement values now sit in `configs/`. None of them has a calibration
against operator measurements, so each is set from a stated reading rather than
from a fit, and this is the reading.

**`kpi.objective.tau_r_db` — the width of the coverage sigmoid, in dB.**
The utility runs from 0.12 to 0.88 over `±tau_R` around `T_cov`, so `tau_R` is
the dB span across which the objective can tell one grade of coverage from
another. Set it to the span you actually want distinguished, which for this
project is `hole_dbm` to `weak_dbm`, and it is **30 dB**. Read the two ends:

- Much smaller (5 dB) makes `J` a soft hole rate. Everything above about
  `T_cov + 10` scores 1, so the search stops caring how strong coverage is and
  optimises only the boundary — and the gradient nearly vanishes away from it,
  which is what a trust-region method needs to move.
- Much larger (100 dB) flattens the sigmoid towards a straight line in dB, so
  a tile at `-119` and a tile at `-60` score almost alike and the hole threshold
  stops meaning anything.

Sanity check after a change: the incumbent's `J` should sit well inside `(0, 1)`
and the spread of `J` across the search's candidates should be larger than the
solver's re-trace noise. If every candidate scores within a thousandth of the
incumbent, `tau_R` is too large to rank them.

**`kpi.objective.beta` — the price of one overlapping co-band neighbour.**
One neighbour keeps `exp(-beta)` of a tile's utility, so it is set from the
question "what fraction of a tile's coverage value would I give up to remove one
overlapping neighbour from it?" If that fraction is `f`, then
`beta = -ln(1 - f)`. The committed **0.5** is `f = 39%`. Equivalently, in dB:
one neighbour costs the same as losing `beta * tau_R` of sigmoid argument, which
at these settings is about 15 dB of coverage near the threshold — a steep price,
deliberately, because a redundant layer is exactly what the project exists to
remove.

- `beta = 0` deletes the overlap term and the objective becomes pure coverage;
  the search will then blanket the area with every layer.
- `beta >= 2` (86% per neighbour) makes two overlapping neighbours worth less
  than an uncovered tile, and the search will open holes to avoid overlap. Keep
  `beta * max(m_g)` below about 2 for the trade to stay sane, and
  `overlap_neighbor_mean` in the reported KPIs is where to read `max(m_g)`.

**`data.demand.bandwidth_m` — the KDE bandwidth, in metres.**
Do not use Scott's or Silverman's rule. Both scale the bandwidth with the spread
of the samples, which here is the size of the study area (about 6.5 km), giving
a kernel hundreds of metres wide that erases the very hotspots the map exists to
find. Two readings that do give a number, and they agree to within a factor of
two on this scenario:

1. **Sampling spacing.** With `n` reported tiles over area `A`, neighbouring
   samples sit about `sqrt(A / n)` apart — about 113 m here. A bandwidth near
   that makes adjacent kernels just touch, so the field is continuous without
   being smooth.
2. **Feature scale.** The demand the map should resolve has a physical size: in
   this simulator, `simulation.density.sigma_minor_m`/`sigma_major_m`, i.e.
   40–200 m. A bandwidth at the low end of the feature scale keeps hotspots
   separate.

The committed **150 m** sits between the two. Check it by eye on the two rasters
notebook 02 plots: the weight map should show the hotspots as distinct blobs. If
they have merged into one cloud, the bandwidth is too large; if the map looks
like scattered dots, it is too small.

**`data.demand.uniform_share` — the floor under unreported ground.**
At 0 the objective may ignore a region no UE was ever drawn in. That is correct
when the MDT is the demand you care about, and it is the committed value. Raise
it when a region matters for a reason the traffic sample cannot show — a
coverage obligation, a road, a site about to be built — and read it as "this
fraction of the score is about the map, the rest is about the traffic". Setting
it to 1 is the ADR 0006 objective.

**`kpi.capacity.max_admission_utilisation`.** Not an objective parameter, but it
now binds: it is the headroom policy, and `prb_utilisation_max` cannot exceed
it. 0.8 is a common operational ceiling; lowering it trades `served_rate` for
headroom directly and visibly.

## Consequences

**Positive**

- The objective is about the traffic, not about the map's empty quarters, which
  was ADR 0006's first recorded negative.
- The demand weighting costs the search nothing: the weights are built once,
  read once per run, and a candidate is still one ray trace.
- The load ceiling makes `max_admission_utilisation` mean what it is named, and
  gives `prb_utilisation_max` a bound a reader can check against.
- Serving the UE table once per measurement, with the served rate and both load
  measures reducing that one assignment, keeps the added KPIs free of ray-tracing
  cost.
- Per-band rows make a multi-band claim checkable: "700 MHz is the coverage
  floor" is now a number and not a design intention.

**Negative**

- The objective now depends on the UE sample through the MDT, which ADR 0006
  deliberately avoided: `J` was a function of the radio map alone. Two scenarios
  drawn from the same density give slightly different weights, so a comparison
  across scenarios must fix the demand map rather than rebuild it.
- **A hotspot nobody can reach gets almost no weight.** The MDT holds served
  reports, so ground the network already fails to cover is under-represented in
  the very map that decides where coverage matters. On this scenario the fourth
  hotspot sits far from every mast: 1 844 UE reports stand within 400 m of its
  centre and only 117 of them are ever served, so the KDE leaves it nearly
  weightless and the objective has little reason to reach for it. `hole_rate`
  still counts those tiles at full weight, which is one reason the KPIs stay
  tile-uniform; raising `data.demand.uniform_share` is the lever if the
  objective should chase such a region, and building the map from the UE table
  instead is the alternative rejected below.
- The demand map inherits every simplification of the capacity model: the PRBs
  it is a median of come from a Shannon rate with no MCS cap.
- The median is taken over the intervals a tile was reported in, so a tile
  reported once carries the same standing as a tile reported sixteen times. The
  KDE smooths the noise that creates but does not remove it.
- Two more judgement parameters, and still no sensitivity analysis behind any of
  the four.
- The 0.8 ceiling refuses admissions the previous rule accepted, so
  `served_rate` is not comparable with any figure recorded before this change —
  and neither is anything else: `src/evaluation/runs.py` refuses to compare runs
  across it.
- Runs recorded before this change carry `served_ratio` and `edge_rsrp_dbm` and
  cannot be loaded as a `KpiVector`.

## Alternatives considered

**Weight the reported KPIs by demand too.** Rejected. A rate that says how much
of the *map* is bad and an objective that says how much of the *traffic* is
badly served answer different questions, and both are worth printing. Collapsing
them would leave no tile-uniform number to notice that a configuration abandoned
a quarter of the study area.

**Use the UE table rather than the MDT.** The UE table includes UEs no cell
reaches, so its demand map would weight holes by the traffic standing in them —
which the objective is already measuring through `R_s`. The MDT is what a real
network can observe, and using it keeps the map buildable from measurements.

**Scott's / Silverman's bandwidth.** Rejected; see "Choosing the parameters".
They estimate a density over the sample's own spread, and the sample's spread
here is the study area.

**Weight by report count instead of requested PRBs.** Rejected: two UEs at
-118 dBm need far more PRBs than two at -70 dBm, and it is the PRBs a cell must
find. Counting reports would weight the easy traffic and the hard traffic alike.

**Keep the old admission gate and add a separate ceiling KPI.** Rejected: it
would report an overload the rule was configured to prevent, and leave the
config value describing nothing.
