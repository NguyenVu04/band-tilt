# 7. A per-band demand-weighted objective, a load ceiling, and the reported KPI set

- **Status:** Superseded by 0008 (the coverage utility only)
- **Date:** 2026-09-18
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the tile-uniform objective and the reported KPI list of
  [ADR 0006](0006-radio-coverage-objective.md); the admission rule and the
  five-KPI list of [ADR 0001](0001-four-kpis-and-weighted-score.md)
- **Superseded by:** [ADR 0008](0008-softplus-coverage-utility.md), for the
  coverage utility alone — the logistic became a softplus and `J` lost its
  `[0, n_band]` bound. Everything else in this record stands.
- **Rewritten in place on 2026-09-19**, at the maintainer's direction, when the
  demand map moved from requested PRBs to report counts and the objective was
  split per band. The original text, which recorded the PRB-weighted map and the
  band-collapsed objective, is in Git history. See
  [the README](README.md).

## Context

Three things were wrong with what ADR 0006 left in place, and all three are
recorded in its own Consequences section.

**Every tile counted equally.** The objective averaged its per-tile utility over
the whole grid, so 101 060 tiles of open ground and the four hotspots carrying
most of the traffic had the same say. A tilt that filled a hole where nobody
stands scored exactly like one that filled a hole in a hotspot. The simulator
already produces the measurement that would fix this — the MDT is the set of
reports the network actually served — and nothing read it.

**Every band counted as one.** `R_s(g)` was the strongest cell-band at the tile,
over all three layers at once. A tile lit by 2600 MHz therefore scored full
coverage utility even where 700 MHz had a hole, and the objective could not see
the difference. Worse for the search: on a tile where 2600 is strongest, moving
a 700 MHz tilt leaves the maximum unchanged, so 12 of the 36 decision dimensions
had no gradient over large parts of the map.

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
`data.output.demand_file`. One step:

```
c_g = |{ MDT reports on tile g, over the whole horizon }|
p_g = c_g / sum_h c_h
```

- Reports are counted as rows, with no de-duplication. This simulator redraws
  every UE position independently per interval (`src/simulation/sample.py`) and
  the schema carries no UE identity, so each row is a distinct UE by
  construction. Real MDT, where one handset can send several reports inside one
  interval, must be de-duplicated by UE id first, or a stationary chatty device
  outweighs a crowd.
- A tile never reported has `p_g = 0`, which is a statement about the sample and
  not about the tile. There is no smoothing; section 2's `alpha` is the only
  floor.
- The map has no settings of its own and carries no band. A UE standing on a
  tile is a UE standing on a tile, whichever layer ends up serving it.

**Counts, not requested PRBs.** `prb_per_ue` is a function of SINR, which is a
function of the tilt being optimized. A map built at the baseline tilts and
weighted by PRBs therefore weights the objective by the baseline network's own
coverage — the circularity behind the "hotspot nobody can reach gets almost no
weight" negative this record carried before the rewrite. Counting rows makes the
map independent of the variable being searched. `src/simulation/mdt.py` no
longer carries the column.

**Counts over the horizon, not a per-interval median.** The earlier map took the
median summed PRB over the intervals a tile was reported in. Transposed to UE
counts that statistic degenerates: with about 15 UEs per interval over 101 060
tiles, no tile on this scenario is ever occupied twice in one interval, so the
median is 1.0 on 3 016 of the 3 017 reported tiles and `p` becomes an indicator
of "ever reported" carrying no intensity at all. The count over the horizon runs
1 to 15 and does carry it.

### 2. The objective

```
J     = sum_b sum_g w_bg * sigmoid((R_sb(g) - T_cov) / tau_R) * q_ov^m_bg
w_bg  = (1 - alpha_b) / |G| + alpha_b * p_g,      sum_g w_bg = 1
m_bg  = #{ j != s_b(g) on band b : R_j > T_cov and R_s_b - R_j <= Delta_R }
```

One term per band, summed. `R_sb(g)` is the strongest transmitter **on band `b`**
at the tile, and `m_bg` its co-band overlapping neighbours — the same count
`overlap_rate` thresholds, now applied to its own band's sigmoid rather than to
a band-collapsed one. `T_cov = kpi.hole_dbm`, `Delta_R = kpi.overlap_margin_db`,
`q_ov = exp(-beta)`, all unchanged from ADR 0006.

Each band's weights sum to one, so each term lies in `[0, 1]` and **`J` lies in
`[0, n_band]`**, not in `[0, 1]`. The sum is deliberate: it weights every layer
alike, and the place to say a layer matters more is a per-band weight, which
this record does not introduce.

`alpha_b` is `kpi.objective.alpha` for that band: **0** spends the band's effort
evenly over the map, which is what a coverage layer wants, and **1** spends it
where the traffic was measured, which is what a capacity layer wants. The
committed values are `b700: 0.0`, `b1800: 0.5`, `b2600: 1.0`. Setting every
`alpha` to 0 recovers a per-band form of ADR 0006's tile mean.

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

These stay **tile-uniform and band-collapsed**, on purpose: a rate that says how
much of the *map* is bad and an objective that says how much of the *traffic* is
badly served answer different questions, and both are worth printing.

`load_imbalance` is the population standard deviation over the mean of each
cell-band's interval-averaged utilisation. Scale-free: it does not move when the
whole network gets busier, only when the traffic sits unevenly.

**Per band.** Every one of the eleven is also reported per frequency layer
(`src.evaluation.compare.band_kpis`), by giving the same function one band's
slice of the radio map. There is no second definition, and it is the same slice
the objective now takes. `objective` is still not reported per band: its terms
are, but the number that ranks a configuration scores a network.

**The PRB series.** `prb_usage_by_time` reports the PRBs and utilisation of every
cell-band in every interval — the series `prb_utilisation_max` and
`load_imbalance` reduce.

## Choosing the parameters

Three judgement values sit in `configs/kpi.yaml`, plus the admission ceiling.
None has a calibration against operator measurements, so each is set from a
stated reading rather than from a fit, and this is the reading.

**`kpi.objective.tau_r_db` — the width of the coverage sigmoid, in dB.**
The utility runs from 0.12 to 0.88 over `±tau_R` around `T_cov`, so `tau_R` is
the dB span across which the objective can tell one grade of coverage from
another. Set it to the spread the radio map actually has:

```
tau_R  ~  IQR(R_sb) / 1.349
```

`1.349 = Phi^-1(0.75) - Phi^-1(0.25)`, so this is the normal-scale robust
estimate of the standard deviation — robust because the RSRP distribution has a
long low tail from tiles at the edge of reach, which would inflate a plain
standard deviation. Take the IQR over the **per-band serving RSRP at reached
tiles**, which is exactly the quantity each term's sigmoid reads, and pool the
bands: one `tau_r_db` serves all three terms. Notebook 01's `rsrp_statistics`
table reports the per-band and pooled figures. On this scenario the three bands
agree to within a dB and the pooled value is 17.1 dB, so the committed value is
**17.0**.

Re-derive it when the scenario changes — a different scene, mast layout or
transmit power gives a different spread — and never between two runs on one
scenario, or the two stop being comparable. Read the two ends:

- Much smaller (5 dB) makes `J` a soft hole rate. Everything above about
  `T_cov + 10` scores 1, so the search stops caring how strong coverage is and
  optimises only the boundary — and the gradient nearly vanishes away from it,
  which is what a trust-region method needs to move.
- Much larger (100 dB) flattens the sigmoid towards a straight line in dB, so
  a tile at `-119` and a tile at `-60` score almost alike and the hole threshold
  stops meaning anything.

Sanity check after a change: each band's term should sit well inside `(0, 1)`
and the spread of `J` across the search's candidates should be larger than the
solver's re-trace noise. If every candidate scores within a thousandth of the
incumbent, `tau_R` is too large to rank them.

**`kpi.objective.beta` — the price of one overlapping co-band neighbour.**
One neighbour keeps `exp(-beta)` of a tile's utility on that band, so it is set
from the question "what fraction of a tile's coverage value would I give up to
remove one overlapping neighbour from it?" If that fraction is `f`, then
`beta = -ln(1 - f)`. The committed **0.25** is `f = 22%`.

It was halved from 0.5 when the objective was split. The count is now applied to
its own band's sigmoid rather than once to a band-collapsed sum, so a tile
crowded on two bands is penalised inside two terms instead of once, and the
previous price would have overstated the trade.

- `beta = 0` deletes the overlap term and the objective becomes pure coverage;
  the search will then blanket the area with every layer.
- Large `beta` makes two overlapping neighbours worth less than an uncovered
  tile, and the search will open holes to avoid overlap. Keep
  `beta * max(m_bg)` below about 2 for the trade to stay sane, and
  `overlap_neighbor_mean` in the reported KPIs is where to read the typical
  neighbour count.

**`kpi.objective.alpha` — the demand share of a band's tile weights.**
This is where a band's role in the network is stated. At `alpha = 0` the band is
scored evenly over the map: every hole costs the same wherever it is, which is
what a coverage layer is for. At `alpha = 1` the band is scored only where the
MDT reported, so it is rewarded for serving measured traffic and not for
blanketing empty ground. The committed `b700: 0.0, b1800: 0.5, b2600: 1.0` is
the usual low-band-covers / high-band-carries split.

Read the extremes before moving one. `alpha = 1` with no smoothing means the
band's term is a sum over the reported tiles alone — about 3% of this grid — so
that term is a point estimate with the variance of the UE sample, and its
optimum moves with the sampling seed. Lower `alpha` towards 0 to buy back a
floor under unreported ground; that is the lever, since the map itself no longer
has one.

**`kpi.capacity.max_admission_utilisation`.** Not an objective parameter, but it
binds: it is the headroom policy, and `prb_utilisation_max` cannot exceed it.
0.8 is a common operational ceiling; lowering it trades `served_rate` for
headroom directly and visibly.

## The capacity model is a Shannon bound, not NR link adaptation

`src/kpi/capacity.py` charges a UE
`throughput_per_ue_bps / (B_PRB * log2(1 + SINR))` PRBs, with `B_PRB = 12 * SCS`.
Accepted as a deliberate fidelity-for-smoothness trade: the search needs a
quantity that moves continuously with tilt, and this one does. The deviations
from 3GPP, all optimistic, are recorded in that module and summarised here so a
reader knows what the served rate and the load KPIs are:

- **No modulation and coding ceiling.** TS 38.214 Table 5.1.3.1-2 tops out at
  `Q_m * R = 8 * 948/1024` bit/s/Hz per layer (Table 5.1.3.1-1, without 256QAM:
  `6 * 948/1024`); CQI index 1 of Table 5.2.2.1-2 floors a schedulable UE at
  `2 * 78/1024`. `log2(1 + SINR)` obeys neither, so a high-SINR UE is charged
  too few PRBs — cells look less loaded than they would be — and a UE below the
  floor is charged a finite number and blocked by the ceiling rather than
  refused outright.
- **The rate basis is the nominal RB bandwidth.** The UE data rate of
  TS 38.306 4.1.2 uses the symbol rate `12 / T_s^mu` with
  `T_s^mu = 1e-3 / (14 * 2^mu)`, and scales by `(1 - OH)` with `OH = 0.14` for
  downlink FR1. Using `12 * SCS` and no overhead is optimistic on both counts.
- **One layer.** No `v_Layers` MIMO factor and no `R_max`.
- **SINR is the solver's wideband value applied per PRB**, so frequency-selective
  fading does not appear and the PRB figure is an interval average.

What *is* 3GPP: `12` subcarriers per resource block (TS 38.211 4.4.4.1) and the
`max_prb` limits per cell-band, which are `N_RB` from TS 38.101-1 Table 5.3.2-1
for each band's bandwidth at 15 kHz SCS.

This matters less to the objective than it used to — the demand map no longer
reads `prb_per_ue` — but it still sets `served_rate`, `prb_utilisation_max`,
`load_imbalance` and which admissions the ceiling refuses.

## Consequences

**Positive**

- The objective is about the traffic, not about the map's empty quarters, which
  was ADR 0006's first recorded negative — and it is about the traffic only as
  much as each band's role says it should be.
- The demand map no longer depends on the variable being optimized. A tilt can
  no longer make ground look unimportant by failing to cover it.
- A hole on one layer is visible. Every cell-band tilt now moves a term of `J`
  directly, which removes the flat regions the band-collapsed maximum created.
- The map has no hyperparameters left: no bandwidth to justify, no uniform
  share. The one remaining knob, `alpha`, states a band's role rather than a
  smoothing choice.
- The demand weighting still costs the search nothing: the share is built once,
  read once per run, and a candidate is still one ray trace.
- The load ceiling makes `max_admission_utilisation` mean what it is named, and
  gives `prb_utilisation_max` a bound a reader can check against.
- Per-band rows make a multi-band claim checkable: "700 MHz is the coverage
  floor" is now a number and not a design intention.

**Negative**

- **`J` is no longer in `[0, 1]`.** It is in `[0, n_band]`, and its absolute
  value is not comparable with anything recorded before this rewrite. Neither is
  its scale comparable across studies with different band counts.
- **A band at `alpha = 1` is scored on about 3% of the grid.** With no
  smoothing, `p` is zero on every tile the MDT never reported — 98 043 of
  101 060 here — so that band's term is a sum over 3 017 isolated 20 m tiles. It
  is a high-variance estimate of a spatial integral and it moves with the UE
  sampling seed, which makes `J` non-comparable across scenario seeds. Lowering
  `alpha` is the only floor; a kernel was removed deliberately and could be put
  back.
- The objective still depends on the UE sample through the MDT, which ADR 0006
  deliberately avoided: `J` was a function of the radio map alone. A comparison
  across scenarios must fix the demand map rather than rebuild it.
- **The MDT still holds served reports only**, so ground the network already
  fails to cover is under-represented in the map that decides where coverage
  matters. Weighting by counts rather than PRBs weakens this — an unserved tile
  is now under-weighted only by not being in the MDT, not also by its poor SINR
  — but it does not remove it. `hole_rate` still counts those tiles at full
  weight, which is one reason the KPIs stay tile-uniform.
- A tile reported once and a tile reported twice differ by a factor of two in
  `p`, on a sample that thin. There is no smoothing left to absorb that noise.
- Three judgement parameters, and still no sensitivity analysis behind any of
  them. `tau_r_db` now at least has a stated estimator.
- The 0.8 ceiling refuses admissions the pre-0007 rule accepted, so
  `served_rate` is not comparable with any figure recorded before it — and
  neither is anything else: `src/evaluation/runs.py` refuses to compare runs
  across `kpi.objective`.
- Every run scored under the previous form of this record is incomparable and
  was deleted rather than reconciled.

## Alternatives considered

**Weight the reported KPIs by demand too.** Rejected. A rate that says how much
of the *map* is bad and an objective that says how much of the *traffic* is
badly served answer different questions, and both are worth printing. Collapsing
them would leave no tile-uniform number to notice that a configuration abandoned
a quarter of the study area.

**Use the UE table rather than the MDT.** The UE table includes UEs no cell
reaches, so its demand map would weight holes by the traffic standing in them —
which the objective is already measuring through `R_sb`. The MDT is what a real
network can observe, and using it keeps the map buildable from measurements.

**Weight by requested PRBs rather than report count.** This was the decision
before the rewrite, on the reading that two UEs at -118 dBm need far more PRBs
than two at -70 dBm and it is the PRBs a cell must find. Reversed: that reading
is true of the *cell's* load, which is what `prb_utilisation_max` and
`load_imbalance` measure, and false of *where the demand is*, which is what the
map is for. Weighting by a function of SINR made the objective's weights depend
on the tilts it was searching over.

**Keep the KDE, applied to counts.** Rejected here, and the closest call. The
kernel was doing real variance reduction: it filled the 97% of the grid the MDT
never reaches and made the map stable under the UE seed. It was removed because
it is a hyperparameter no data selects — Scott's and Silverman's rules both
scale with the study area and erase the hotspots — and because `alpha` now
provides the floor it was partly standing in for. Recorded as a negative above;
the lever if the 3%-of-grid support proves too thin is to put it back.

**Average over bands instead of summing.** Rejected: it only divides `J` by a
constant, so it changes no ranking. Summing was kept because it makes the band
count visible in the number's range.

**Keep the old admission gate and add a separate ceiling KPI.** Rejected: it
would report an overload the rule was configured to prevent, and leave the
config value describing nothing.
