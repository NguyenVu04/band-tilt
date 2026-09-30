# 2. A contraharmonic, strength-aware objective, max-throughput cell selection, and the reported KPI set

- **Status:** Proposed
- **Date:** 2026-09-22
- **Rewritten:** 2026-09-28 — the reported KPI set is replaced (section 3),
  `objective_version` is removed, and the former four-KPI record is deleted;
  see [the README](README.md).
- **Revised:** 2026-09-29 — *Measured outcome* re-measured on the 2026-09-29
  runs under the nine KPIs of section 3 and 10 Mbps per UE. The comparison
  with the best-band maximum is kept as recorded.
- **Rewritten:** 2026-09-30 — sections 2 to 4 rewritten in place: the band
  preference and the admission gate are replaced by max-throughput cell
  selection over an equal PRB share, and three estimated-throughput KPIs join
  the set. *Measured outcome* predates this and is kept as recorded.
- **Rewritten:** 2026-09-30 — section 1 rewritten in place: the per-band
  utility `lambda e^(1 - lambda)` over the neighbours within
  `kpi.overlap_margin_db` is replaced by the strongest cell's share of the
  band's received power, over every co-band cell above `kpi.hole_dbm`.
  *Measured outcome* predates this and is kept as recorded.
- **Renumbered:** 2026-09-28 (formerly 0003).
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** every earlier objective and KPI record. Their surviving
  decisions are folded in here: the per-band utility (section 1), the serving
  rule, the reported KPI set and the capacity model's fidelity trade
  (sections 2 to 4). The records themselves are in Git history.
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
  neighbours, the two weakest of the ten KPIs then reported.

## Decision

### 1. The objective

```
s_bg  = clip((R_b,max(g) - T_cov) / (T_weak - T_cov), 0, 1)
p_bg  = 1 / sum_{i on band b : R_bi(g) > T_cov} 10^((R_bi(g) - R_b,max(g)) / 10)
                       where band b clears T_cov at g,  else 0
u_bg  = p_bg * s_bg

J     = mean_g  sum_b u_bg^2 / sum_b u_bg            (0 where sum_b u_bg = 0)
```

`T_cov` is `kpi.hole_dbm` and `T_weak` is `kpi.weak_dbm`, the same cuts that
`hole_rate` and `weak_rate` use. **The objective has no parameters of its
own.** It reads no band order, no capacity setting and no demand map, so every
tile counts equally.

`p_bg` is the strongest cell's share of the power band `b` delivers to the tile
from cells above `T_cov`; the strongest cell's own term is `10^0 = 1`. It is 1
for a lone server and 1/2 for two equal ones, and every rival costs in
proportion to its linear power, so there is no margin and no step. It is
**co-band**: it sums transmitters within band `b` only, never across bands.

The tile takes the **contraharmonic mean** of its bands' utilities. Each band is
weighted by its own utility, so the result lies between the arithmetic mean and
the maximum of the `u_bg`, and `J` stays in `[0, 1]`. A band that does not cover
the tile has `u_bg = 0` and carries no weight. With one covered band, or with
equal bands, the result equals the maximum.

Implementation: `src/kpi/overlap.py::effective_coverage`, averaged over the
grid by `src/optim/objective.py::objective`.

### 2. Max-throughput cell selection over an equal PRB share

Within an interval UEs connect one at a time, in report-time order, simultaneous
reports strongest RSRP first. A UE's candidates are the cell-bands with
`R > kpi.hole_dbm`. It joins

```
argmax_(i,b)  P_ib / (n_ib + 1) * 12 * SCS_b * log2(1 + SINR_ib)
P_ib = max_admission_utilisation * max_prb_ib
```

where `n_ib` is the UEs already connected there. **Nobody is refused**: the only
UE left unserved is one with no candidate. Once the interval's last UE has
connected, each UE's **estimated throughput** is `P_ib / N_ib` times its per-PRB
rate, `N_ib` the cell-band's final UE count, so a UE's figure falls as later UEs
join its cell-band.

`kpi.capacity.max_admission_utilisation` is **0.8**: the share of `max_prb` a
cell-band shares among its UEs. There is no band order, no serving threshold
above the hole threshold and no per-UE demand. Every cell-band with a UE uses its
whole pool, so PRB load carries no information and is not reported.

### 3. The reported KPIs

Twelve measures, stored beside `objective`, in this order. The best server is
the strongest layer over every band and cell, `R_max`; covered tiles are those
with `R_max > kpi.hole_dbm`.

| KPI | Definition | Direction |
|---|---|---|
| `hole_rate` | share of tiles with `R_max <= kpi.hole_dbm` | minimise |
| `weak_rate` | share of tiles with `hole_dbm < R_max <= weak_dbm` | minimise |
| `overlap_rate` | share of tiles with any co-band neighbour within `Delta_R` | minimise |
| `overlap_neighbor_mean` | mean `m_g` over covered tiles | minimise |
| `rsrp_p50_dbm` | median of `R_max` over covered tiles | **maximise** |
| `rsrp_p05_dbm` | 5th percentile of the same | **maximise** |
| `sinr_p50_db` | median of the best server's SINR over covered tiles | **maximise** |
| `sinr_p05_db` | 5th percentile of the same | **maximise** |
| `ue_service_failure_rate` | share of UE reports with no cell-band above `kpi.hole_dbm`, `1 - served share` | minimise |
| `estimated_throughput_p05_mbps` | 5th percentile of the served UE reports' estimated throughput | **maximise** |
| `estimated_throughput_p50_mbps` | median of the same | **maximise** |
| `estimated_throughput_mean_mbps` | mean of the same | **maximise** |

- **They stay tile-uniform and band-collapsed, on purpose**, except the four UE
  KPIs, which count every UE position. A rate that says how much of the *map*
  is bad answers a different question from an objective, and both are worth
  printing.
- **Per band.** Every KPI is also reported per frequency layer
  (`src.evaluation.compare.band_kpis`), by giving the same function one band's
  slice of the radio map. There is no second definition. The UE KPIs are
  blank on band rows: a UE fails only when no band reaches it, and takes its
  throughput from whichever band it chose.
- **The failure rate and the throughput split the UEs along one mask.** A report
  with no candidate counts in the failure rate and never as a zero in the
  throughput statistics, so the 5th percentile describes served UEs rather than
  pinning at zero wherever a hotspot sits in a hole.
- **Load is not a KPI.** The former `load_imbalance` and `prb_utilisation_max`
  are removed, and PRB load is no longer reported (section 2). The evaluation
  reports UEs and median estimated throughput per cell-band instead.
- **Spectral efficiency is not a KPI.** The former `se_p50_bps_hz`,
  `se_mean_bps_hz` and `se_p05_bps_hz` are removed. The Shannon rate
  `log2(1 + SINR)` stays inside the throughput estimate (section 4).

### 4. The capacity model is a Shannon bound, not NR link adaptation

`src/kpi/capacity.py` credits one PRB with `B_PRB * log2(1 + SINR)`, with
`B_PRB = 12 * SCS`. This is accepted as a deliberate fidelity-for-smoothness
trade: the search needs a quantity that moves continuously with tilt, and this
one does. The deviations from 3GPP, all optimistic, are recorded in that module
and summarised here:

- **No modulation and coding ceiling or floor.** `log2(1 + SINR)` has neither.
  So a high-SINR UE is credited more than the top MCS carries, and a UE below
  the lowest schedulable rate is credited a small positive rate rather than
  none.
- **An equal share, not a scheduler.** Every UE on a cell-band gets the same
  PRBs for the whole interval, whatever its SINR or demand.
- **The rate basis is the nominal RB bandwidth.** The UE data rate of
  TS 38.306 4.1.2 uses the symbol rate `12 / T_s^mu`, with
  `T_s^mu = 1e-3 / (14 * 2^mu)`, and scales by `(1 - OH)`, with `OH = 0.14` for
  downlink FR1. Using `12 * SCS` and no overhead is optimistic on both counts.
- **One layer.** No `v_Layers` MIMO factor and no `R_max`.
- **SINR is the solver's per-RE value applied to every PRB.** Signal,
  interference and `k * T * SCS` noise are flat across the carrier, so
  frequency-selective fading does not appear and the throughput is an interval
  average.

What *is* 3GPP: `12` subcarriers per resource block (TS 38.211 4.4.4.1), and the
`max_prb` limits per cell-band, which are `N_RB` from TS 38.101-1 Table 5.3.2-1
for each band's bandwidth at 15 kHz SCS.

The objective does not read the capacity model. It sets which cell-band serves
each UE and the three throughput KPIs; `ue_service_failure_rate` depends on the
radio map alone.

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
- **The objective and the serving rule do not agree.** `J` scores every layer
  by strength and co-band contention, while the serving rule picks by throughput
  under load. The serving rule shows up only in the throughput KPIs and the
  serving mix.
- **Inter-band interference is priced nowhere.** `p_bg` is co-band, and the
  reported `overlap_rate` sums the per-band counts. The mean penalises a weak or
  crowded second layer whether or not it interferes with the first. Only the
  solver's SINR sees interference, and `J` does not read SINR.
- **No demand weighting.** A hole where nobody stands costs exactly what a hole
  in a hotspot costs. The UE-weighted coverage view in `reports/` is the only
  place demand appears, and nothing optimises it.
- **`u` does not separate a hole from heavy contention.** A hole scores 0 and a
  band shared by `n` equal cells `1/n` (0.25 at four). Reading `hole_rate`
  beside `overlap_rate` is what separates them.
- **The objective and the overlap KPIs count contention differently.**
  `overlap_rate` and `overlap_neighbor_mean` count neighbours within `Delta_R`;
  `J` prices every co-band rival above `T_cov` by its power, however far down.
- **A marginal network is penalised twice**, once through `s` and once through
  the reported `weak_rate`, so `J` and `weak_rate` are not independent readings.
- **`kpi.capacity.max_admission_utilisation` is a placeholder**, and the
  throughput estimate is optimistic (section 4). Read its direction rather than
  its level.
- **Every run recorded under an earlier form or KPI set is incomparable.**
  Such runs are deleted rather than pooled; a run recorded before the KPI set of
  section 3 no longer loads.

## Measured outcome

Both subsections were measured under the serving rule this record replaced on
2026-09-30: band preference with an admission gate at 10 Mbps per UE. `J` and the
eight radio-map KPIs do not read the serving rule, so their figures hold; the
failure-rate, served-rate, load and serving-mix figures describe the old rule.

### Against the best-band maximum, as recorded

This comparison was measured under the former ten-KPI set (`served_rate`,
`load_imbalance`, no spectral efficiency) and 20 Mbps per UE. It is the record
of the objective decision and is not re-measured: the maximum column would need
every random-search candidate re-traced and a TuRBO run under the old objective.
The contraharmonic mean on the current KPI set is in the next subsection.

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

¹ The rank score is the mean over the KPIs of the set measured (ten here, nine
in the next subsection) of each candidate's percentile rank, oriented so that
higher is better.

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

**On the search, overlap is the measure that moved.** In that run TuRBO lowered
the band-collapsed overlap rate by 2.3 %, where under the maximum it raised it
by 12.1 %. It improved 8 of the 10 KPIs, against 7 under the maximum.

**What was lost.** Under the maximum, TuRBO's distinctive move was to hand
traffic from 2600 MHz to 1800 MHz: 16.4 % of UE reports were served on
1800 MHz, against 10.5 % at the incumbent, both under the earlier noise and
20 Mbps per UE.

### The contraharmonic mean on the current KPI set

Re-measured on the 2026-09-29 runs: seed 42, the nine KPIs of section 3,
10 Mbps per UE and per-RE noise. That TuRBO run uses
`trust_region.perturbed_dimensions` = 5 (the recorded comparison used 20), so
its winner differs from the one above. The correlations are over random search's
145 candidates and signed so that a positive value means `J` and the KPI
improve together. The pipeline does not produce the figures in this subsection;
they were computed once from the archived histories and radio maps.

| | contraharmonic mean |
|---|---:|
| Spearman(`J`, equal-weight rank score over the 9 KPIs)¹ | 0.905 |
| rho vs band-collapsed overlap rate / overlap neighbours per covered tile | 0.427 / 0.297 |
| rho vs hole rate / weak rate | 0.770 / 0.787 |
| rho vs RSRP p50 / RSRP p05 | 0.773 / 0.872 |
| rho vs SINR p50 / SINR p05 | 0.412 / 0.122 |
| rho vs UE service failure rate | 0.481 |
| Tiles where dropping one covered band raises the score, TuRBO winner² | 0.4957 |
| Ceiling on that gain, `mean_g (max_b u_bg - J(g))`, TuRBO winner² | 0.0543 |
| TuRBO band-collapsed overlap rate (incumbent 0.3164) | 0.3102 |
| TuRBO − rule sweep on `J` | +0.0098 |
| Rule sweep vs TuRBO, head to head on the 9 KPIs | 5–4 |

The correlations with the eight KPIs the two sets share match the per-RE values
recorded above, because `J` and those KPIs read the radio map alone. The ceiling on the
non-monotonicity gain, 0.054, again exceeds the winner's whole improvement of
0.041.

**Overlap still moves.** TuRBO lowers the band-collapsed overlap rate by 2.0 %
and improves all three bands: 2600 MHz 0.2010 → 0.1915, 1800 MHz
0.2067 → 0.1894 and 700 MHz 0.2396 → 0.2219. It improves 8 of the 9 KPIs and
worsens overlap neighbours per covered tile (+2.2 %). Random search and the
rule sweep still raise the band-collapsed rate.

**TuRBO's margin comes from contention.** Each winner's gain over the incumbent
splits as follows. The first two columns are the tiles that changed coverage.
The rest is the change on tiles covered in both, divided into the part a
strength-only score (every covered band at `lambda = 1`) would show, and the
remainder.

| | ΔJ | newly covered | lost coverage | strength | contention |
|---|---:|---:|---:|---:|---:|
| random | +0.0260 | +0.0004 | −0.0001 | +0.0284 | −0.0027 |
| rule | +0.0312 | +0.0007 | −0.0000 | +0.0459 | −0.0155 |
| turbo | +0.0410 | +0.0005 | −0.0001 | +0.0392 | +0.0014 |

The sweep's uniform uptilts buy the most strength and pay for it in crowding,
now that crowding on every layer counts. TuRBO buys less strength and gains on
contention. Hole-closing stays negligible, because a newly covered tile is worth
about 0.09: 591 tiles under TuRBO, 818 under the sweep.

**Traffic concentrates on 2600 MHz.** Every method raises its share of UE
reports from 64.9 % to 67.2–67.8 %, and 1800 MHz falls from 5.2 % to 3.5–3.9 %.
TuRBO's recommended configuration uptilts every 2600 MHz carrier. Its six
downtilts are three 700 MHz carriers and three 1800 MHz carriers, on four
sectors.

## Alternatives considered

**The best-band maximum.** Monotone, but it leaves the non-best layers'
crowding unpriced (Context).

**`lambda e^(1 - lambda)` over the neighbours within `Delta_R`** (in force until
2026-09-30), with `lambda = 1 + m_bg`. It counted a rival as a whole cell
whatever its power, and stepped at the margin: a rival 6 dB down cost 0.264 at
full strength and one 6.01 dB down cost nothing.

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

**Band preference with an admission gate** (in force until 2026-09-30). A UE
took the most preferred band above a serving threshold with room for a fixed
per-UE demand, and was refused where none had room. Replaced: the band order and
the demand were placeholders nothing measured, and the refusals made the failure
rate mix coverage with a configured capacity.

**Estimated throughput at connect time.** The rate a UE computed when it chose,
before later UEs diluted it. Rejected: it is an upper bound on what the UE gets,
and it would credit the first UE of an interval with a whole cell-band.

**Hole UEs as zero throughput.** Rejected: with a hotspot in a hole, the 5th
percentile would sit at zero in every configuration. The failure rate already
counts those UEs.
