# 3. A contraharmonic, strength-aware objective, a load ceiling, and the reported KPI set

- **Status:** Proposed
- **Date:** 2026-09-22
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** every earlier objective record. Their surviving decisions are
  folded in here: the admission ceiling, the reported KPI set and the capacity
  model's fidelity trade (sections 3 to 5), and the per-band utility with its
  version guard (sections 1 and 2). The records themselves were deleted on
  2026-09-22 and are in Git history; see [the README](README.md).
- **Superseded by:** —

## Context

> The motivation is the decider's to state. What follows is the mechanical
> account of what is decided and what it costs.

The objective went through several forms before this one. Two lessons from them
are what shaped this record.

- **Scoring a preferred band made `J` depend on which band the tilts left
  standing.** Dropping a crowded preferred layer below the hole threshold moved
  a tile onto a cleaner lower layer and *raised* `J`.
- **Scoring the best band closed that, but left a tile's other layers
  unpriced.** A crowded layer cost nothing wherever another layer at the same
  tile was clean. Over random search's 145 Sobol candidates, `J` then correlated
  only 0.251 with the band-collapsed overlap rate and 0.235 with overlap
  neighbours, the two weakest of the ten KPIs.

## Decision

### 1. The objective

```
s_bg  = clip((R_b,max(g) - T_cov) / (T_weak - T_cov), 0, 1)
m_bg  = #{ i != argmax on band b : R_bi(g) > T_cov
                                   and R_b,max(g) - R_bi(g) <= Delta_R }
lambda_bg = 1 + m_bg   where band b clears T_cov at g,  else 0
u_bg  = lambda_bg * exp(1 - lambda_bg) * s_bg

J     = mean_g  sum_b u_bg^2 / sum_b u_bg            (0 where sum_b u_bg = 0)
```

`T_cov` is `kpi.hole_dbm`, `T_weak` is `kpi.weak_dbm` and `Delta_R` is
`kpi.overlap_margin_db`. These are the same three cuts that `hole_rate`,
`weak_rate` and `overlap_rate` use. **The objective has no parameters of its
own.** It reads no `kpi.capacity.band_preference` and no demand map, so every
tile counts equally.

`m_bg` is **co-band**: it counts transmitters within band `b` only, never
across bands.

The tile takes the **contraharmonic mean** of its bands' utilities. Each band is
weighted by its own utility, so the result lies between the arithmetic mean and
the maximum of the `u_bg`, and `J` stays in `[0, 1]`. A band that does not cover
the tile has `u_bg = 0` and carries no weight. With one covered band, or with
equal bands, the result equals the maximum.

Implementation: `src/kpi/overlap.py::effective_coverage`, averaged over the
grid by `src/optim/objective.py::objective`.

### 2. `objective_version`

The functional form lives in code, where the config comparison cannot see it.
`kpi.objective_version` is bumped by hand whenever the form changes. It is
compared with the rest of the `kpi` block across runs, and against the running
config as well, because a set of runs that are uniformly stale still agree with
one another. This record is version **3**.

### 3. The admission ceiling

A cell-band admits a UE only when

```
PRB_required + PRB_current <= max_admission_utilisation * max_prb
```

so the cap is a **ceiling on the resulting load**, not a gate on the load before
admission. No cell-band ever ends an interval above that share.
`kpi.capacity.max_admission_utilisation` is **0.8**, a common operational
ceiling. Lowering it trades `served_rate` for headroom directly and visibly.

### 4. The reported KPIs

Ten measures, stored beside `objective`, in this order:

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
| `load_imbalance` | coefficient of variation of cell-band utilisation | minimise |

- **They stay tile-uniform and band-collapsed, on purpose.** A rate that says
  how much of the *map* is bad answers a different question from an objective,
  and both are worth printing.
- **`load_imbalance`** is the population standard deviation over the mean of
  each cell-band's interval-averaged utilisation. It is scale-free: it does not
  move when the whole network gets busier, only when the traffic sits unevenly.
- **Per band.** Every KPI is also reported per frequency layer
  (`src.evaluation.compare.band_kpis`), by giving the same function one band's
  slice of the radio map. There is no second definition.
- **`prb_utilisation_max` is computed but not reported.** It is bounded by the
  admission ceiling by construction, so as a KPI it is the ceiling read back.
  It stays as the check that the admission rule held, and the per-band table
  and the per-cell-band figure print it. `prb_usage_by_time` reports the PRBs
  and utilisation of every cell-band in every interval.

### 5. The capacity model is a Shannon bound, not NR link adaptation

`src/kpi/capacity.py` charges a UE
`throughput_per_ue_bps / (B_PRB * log2(1 + SINR))` PRBs, with `B_PRB = 12 * SCS`.
This is accepted as a deliberate fidelity-for-smoothness trade: the search needs
a quantity that moves continuously with tilt, and this one does. The deviations
from 3GPP, all optimistic, are recorded in that module and summarised here:

- **No modulation and coding ceiling or floor.** `log2(1 + SINR)` has neither.
  So a high-SINR UE is charged too few PRBs, and cells look less loaded than
  they would be. A UE below the lowest schedulable rate is charged a finite
  number of PRBs and blocked by the admission ceiling, rather than refused
  outright.
- **The rate basis is the nominal RB bandwidth.** The UE data rate of
  TS 38.306 4.1.2 uses the symbol rate `12 / T_s^mu`, with
  `T_s^mu = 1e-3 / (14 * 2^mu)`, and scales by `(1 - OH)`, with `OH = 0.14` for
  downlink FR1. Using `12 * SCS` and no overhead is optimistic on both counts.
- **One layer.** No `v_Layers` MIMO factor and no `R_max`.
- **SINR is the solver's per-RE value applied to every PRB.** Signal,
  interference and `k * T * SCS` noise are flat across the carrier, so
  frequency-selective fading does not appear and the PRB figure is an interval
  average.

What *is* 3GPP: `12` subcarriers per resource block (TS 38.211 4.4.4.1), and the
`max_prb` limits per cell-band, which are `N_RB` from TS 38.101-1 Table 5.3.2-1
for each band's bandwidth at 15 kHz SCS.

The objective does not read the capacity model. It sets `served_rate`,
`prb_utilisation_max`, `load_imbalance` and which admissions the ceiling
refuses.

## Consequences

**Positive**

- **Every covered layer is priced.** A crowded or marginal second layer lowers
  the tile's score instead of vanishing under a clean one, and no band selection
  depends on the tilts.
- **No free parameters**, and `J` is bounded in `[0, 1]`.
- **The strength factor prices coverage quality.** Hole rate, weak rate, both
  RSRP percentiles and both overlap measures all move `J`.

**Negative**

- **`J` is not monotone in the layers present.** Removing band `b` from a tile
  raises its score exactly when `0 < u_bg < J(g)`. Dropping `u` from the sums
  changes `S2/S1` to `(S2 - u^2)/(S1 - u)`, which exceeds `S2/S1` iff
  `u < S2/S1`. So a tilt can gain `J` by darkening a weak layer. A maximum over
  bands would rule that out by construction.
- **The objective and the serving rule do not agree.** `J` scores every layer,
  while the serving rule admits by `kpi.capacity.band_preference`. Band
  preference shows up only in `served_rate` and the serving mix.
- **Inter-band interference is priced nowhere.** `m_bg` is co-band, and the
  reported `overlap_rate` sums the per-band counts. The mean penalises a weak or
  crowded second layer whether or not it interferes with the first. Only the
  solver's SINR sees interference, and `J` does not read SINR.
- **No demand weighting.** A hole where nobody stands costs exactly what a hole
  in a hotspot costs. The UE-weighted coverage view in `reports/` is the only
  place demand appears, and nothing optimises it.
- **`u` is not monotone in `lambda` either.** A hole and a four-way overlap
  score alike (0 against 0.199 at full strength). Reading `hole_rate` beside
  `overlap_rate` is what separates them.
- **A marginal network is penalised twice**, once through `s` and once through
  the reported `weak_rate`, so `J` and `weak_rate` are not independent readings.
- **Every `kpi.capacity` value is a placeholder**, and the capacity model is
  optimistic (section 5).
- **Every objective value recorded under an earlier form is incomparable.**
  `objective_version` is the only thing that catches it, and those runs were
  deleted rather than pooled.

## Measured outcome

Measured at seed 42 against the best-band maximum this record replaces, on the
same scenario, the same solver settings and the same budgets. Random search
draws the same seeded Sobol sequence under both objectives. Its 145 candidates
are therefore **the same tilt configurations with identical KPIs**, so the
correlations below are a paired comparison of the two objectives on one sample.
The contraharmonic column's TuRBO figures are from the 2026-09-25 rerun, with
J rounded to 1e-6; its searches match the 2026-09-23 run on every evaluation.
The best-band maximum column was not rerun.

| | best-band maximum | contraharmonic mean |
|---|---:|---:|
| Spearman(`J_max`, `J_ch`), shared 145 candidates | — | 0.935 |
| Spearman(`J`, equal-weight rank score over the 10 KPIs)¹ ³ | 0.930 | 0.918 (0.898) |
| rho vs band-collapsed overlap rate | 0.251 | **0.427** |
| rho vs overlap neighbours per covered tile | 0.235 | **0.297** |
| rho vs hole rate / served rate³ | 0.738 / 0.565 | 0.770 / 0.612 (0.356) |
| rho vs weak rate / RSRP p05 / SINR p05³ | 0.877 / 0.918 / 0.902 | 0.787 / 0.872 / 0.860 (0.122) |
| rho vs RSRP p50 / SINR p50 / load imbalance³ | 0.861 / 0.878 / 0.385 | 0.773 / 0.789 / 0.351 (0.412 / −0.038) |
| Tiles where dropping one covered band raises the score, TuRBO winner² | 0 by construction | 0.4934 |
| Ceiling on that gain, `mean_g (max_b u_bg - J(g))`, TuRBO winner² | 0.0000 | 0.0542 |
| TuRBO band-collapsed overlap rate (incumbent 0.3164) | 0.3546 | **0.3090** |
| TuRBO − rule sweep on `J` | +0.0136 | +0.0091 |
| Rule sweep vs TuRBO, head to head on the 10 KPIs | 8–2 | 6–4 |

¹ The rank score is the mean over the ten KPIs of each candidate's percentile
rank, oriented so that higher is better.

² Per tile, with no tilt re-traced. No configuration can collect the ceiling,
because darkening a band on one tile changes it on many, including tiles where
it is the best layer. It bounds what the non-monotonicity is worth. It is not a
measured exploit.

³ Both columns were first measured with thermal noise over the full channel
bandwidth. With per-RE noise (`k * T * scs_hz`, 2026-09-25) the contraharmonic
column's SINR, served-rate and load figures, and the rank score, are the values
in parentheses; `J` and the other six KPIs are unchanged. The best-band maximum
column was not remeasured, so on those rows the two columns compare only under
the earlier noise.

**The trade the change bought, in correlation.** Rho rose on four of the ten
KPIs: both overlap measures, the hole rate and the served rate. It fell on the
six strength and SINR measures. `J` now tracks crowding better and signal
strength worse, and its rank-score correlation is marginally lower. Those
served-rate and SINR comparisons are under the earlier noise. With per-RE noise,
`J` tracks SINR and service far less closely: rho is 0.122 against cell-edge
SINR, 0.412 against median SINR, 0.356 against the served rate and −0.038
against load imbalance.

**On the search, overlap is the measure that moved.** TuRBO lowers the
band-collapsed overlap rate by 2.3 %, where under the maximum it raised it by
12.1 %. It improves all three bands: 2600 MHz 0.2010 → 0.1913, 1800 MHz
0.2067 → 0.1940 and 700 MHz 0.2396 → 0.2237. It improves 8 of the 10 KPIs,
against 7 under the maximum, and worsens overlap neighbours per covered tile
(+3.8 %) and load imbalance (+2.9 %). Random search and the rule sweep still
raise the band-collapsed rate.

**TuRBO's margin comes from contention.** Each winner's gain over the incumbent
splits as follows. The first two columns are the tiles that changed coverage.
The rest is the change on tiles covered in both, divided into the part a
strength-only score (every covered band at `lambda = 1`) would show, and the
remainder.

| | ΔJ | newly covered | lost coverage | strength | contention |
|---|---:|---:|---:|---:|---:|
| random | +0.0260 | +0.0004 | −0.0001 | +0.0284 | −0.0027 |
| rule | +0.0312 | +0.0007 | −0.0000 | +0.0459 | −0.0155 |
| turbo | +0.0402 | +0.0005 | −0.0001 | +0.0394 | +0.0004 |

The sweep's uniform uptilts buy the most strength and pay for it in crowding,
now that crowding on every layer counts. TuRBO buys less strength and does not
pay. Hole-closing stays negligible, because a newly covered tile is worth about
0.09: 588 tiles under TuRBO, 818 under the sweep.

**What was lost.** Under the maximum, TuRBO's distinctive move was to hand
traffic from 2600 MHz to 1800 MHz: 16.4 % of UE reports were served on
1800 MHz, against 10.5 % at the incumbent, both under the earlier noise. Under
this objective, with per-RE noise, every method concentrates traffic on
2600 MHz (62.8–63.2 %, from 60.3 %), and 1800 MHz falls to 4.8–5.4 % (from
6.8 %). TuRBO's recommended configuration uptilts every 2600 MHz carrier.
Its five downtilts are three 700 MHz carriers and two 1800 MHz carriers.

## Alternatives considered

**The best-band maximum.** Monotone, but it leaves the non-best layers'
crowding unpriced (Context).

**The most preferred band clearing `T_cov`.** Rejected: the score depends on
which band the tilts leave standing, so a tilt can pay by killing a layer.

**The arithmetic mean over covered bands.** It scores layers that would never
serve the tile at full weight. The contraharmonic mean still scores them, but
weights each by `u_bg`, so a band contributes in proportion to how well it
would serve. The arithmetic mean never exceeds the contraharmonic mean, so a
weak layer drags it down harder.

**A penalty on the non-best layers added to the maximum.** Not tried. It needs
a weight, which is a free parameter. It also needs a way to choose the best
layer that does not depend on the tilts.

**A demand-weighted objective.** Tried in an earlier form, with a weight of one
plus the MDT report count per tile. It was deleted as measurably inert, since
the report count is zero on almost every tile of this grid. A spatial kernel
that spreads it would add a bandwidth parameter and invent traffic where none
was measured.

**Keep the old admission gate and add a separate ceiling KPI.** Rejected: it
would report an overload the rule was configured to prevent, and leave the
config value describing nothing.
