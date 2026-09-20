# 9. An effective-coverage objective

- **Status:** Superseded by 0010
- **Date:** 2026-09-20
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** [ADR 0008](0008-softplus-coverage-utility.md) entirely, and
  the objective of [ADR 0007](0007-demand-weighted-objective.md): the coverage
  utility, the per-band split, the per-band weights `w_bg` and the exponential
  overlap discount are all replaced. What survives from 0007 is the demand map
  itself, the admission ceiling and the eleven reported KPIs
- **Superseded by:** [ADR 0010](0010-monotone-strength-aware-objective.md)
  entirely. The measured consequences this ADR recorded and left open — the
  unscored lower layers, the objective disagreeing with the KPI set — are what
  0010 was written to fix, together with the demand map and one KPI

## Context

> The motivation is the decider's to state. What follows is the mechanical
> account of what changed and what it costs.

ADR 0007's objective, as amended by 0008, was

```
J = sum_b sum_g w_bg * softplus((R_sb(g) - T_cov) / tau_R) * exp(-beta * m_bg)
```

Three things about it were unsatisfying enough to replace rather than tune.

**It has three free parameters and none of them is observable.** `tau_R` is a
knee width estimated from the RSRP spread, `beta` is a price nobody can quote,
and `alpha_b` is a per-band blend. Each has to be re-derived per scenario, and
two runs whose parameters differ are not comparable. ADR 0008 also left `tau_R`
"no longer derived from the property it is now mostly controlling", and the
committed `beta` had drifted from the value both records state.

**It is unbounded and rewards signal nobody needs.** 0008's own Consequences
say it: above `T_cov` the utility grows linearly in dB without ceiling, so a
configuration can buy `J` by concentrating power on tiles that are already well
covered. There is no natural scale at which a score is "good".

**It scores every band everywhere, including bands that would never serve.** A
tile where 2600 MHz is strong still carries a 700 MHz term, so the objective
spends effort on layers the serving rule would not pick at that tile.

## Decision

Score what the network is actually for: **each tile served cleanly by exactly
one cell, on the band that would really serve it.**

```
b(g)      = the most preferred band whose strongest cell clears T_cov
            (kpi.capacity.band_preference, most preferred first)
m_b(g)    = #{ i != argmax on band b : R_bi(g) > T_cov
                                       and R_b,max(g) - R_bi(g) <= Delta_R }
lambda(g) = 1 + m_b(g)(g)   where some band clears T_cov,  else 0
u(g)      = lambda(g) * exp(1 - lambda(g))
r(g)      = c(g) / max_h c(h)          c = MDT reports per tile
w(g)      = 1 + r(g)

J         = sum_g w(g) u(g) / sum_g w(g)
```

`T_cov` is `kpi.hole_dbm` and `Delta_R` is `kpi.overlap_margin_db` — the same
cuts the hole and overlap KPIs use. **The objective has no parameters of its
own**; the `kpi.objective` config block is deleted.

`lambda` is the count of cells contending to serve the tile. `u = lambda
e^(1-lambda)` is the unique smooth function that peaks at exactly 1 when
`lambda = 1`:

| `lambda` | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `u` | 0 | **1.000** | 0.736 | 0.406 | 0.199 |

So `J` is the demand-weighted share of the grid that is effectively covered, in
`[0, 1]`, and 1 means every tile has one dominant server. A hole and a tile the
ray tracer found no path to both score 0; a tile "effectively covered" — the
name for `lambda = 1` — is exactly one with `R_1 > T_cov` and no other cell on
its band both above `T_cov` and within `Delta_R`.

### What was decided along the way

**`r` counts reports, and normalises by the busiest tile, not by a per-interval
statistic.** A per-interval intensity was specified first and measured before
being rejected: on this scenario a tile is reported twice within one 15-minute
interval 8 times in 5,311 reports, so a mean over reporting intervals is exactly
1.0 on 3,009 of the 3,017 reported tiles and `r` degenerates into a binary "was
this tile ever seen". ADR 0007 §1 records the same degeneracy from the median.
The horizon total spreads over 1…15 and needs no new artifact: `r = share /
max(share)` reads off the demand map already written.

**`w = 1 + r` has no band term.** An `alpha_b` keyed on the serving band was
specified and rejected: `alpha` would make each tile's weight a function of
which band serves it, which is a function of the tilts being searched. Since `w`
sits in both the numerator and the denominator of a weighted average, the search
could raise `J` by pushing a badly-scoring high-demand tile onto a lower-`alpha`
band instead of fixing it. ADR 0007 rejected PRB-weighted demand for the same
reason — a weight must not depend on the decision variable. The surviving form
weights a tile between 1 and 2 and never to 0, so ground the MDT never stood on
is still scored, just not doubly.

**`R_2 > T_cov` was dropped from the effective-coverage definition.** As an
`and`, it excludes the ideal case: a tile whose only audible cell is its server
has `R_2 = -inf`. The neighbour rule already carries the condition in the right
place, and `lambda = 1` is the definition.

## Consequences

**Positive**

- **No free parameters.** Nothing to re-derive per scenario, nothing to drift
  from its record, and two runs on one scenario are comparable by construction.
- **Bounded and interpretable.** `J = 0.81` means what it says: 81% of the
  demand-weighted grid, in effective-coverage units. A run can be read without
  the config that produced it.
- **The objective and the serving rule agree.** `lambda` is counted on the band
  `kpi.capacity.band_preference` would actually serve the tile on, so the search
  optimises the network as the served rate measures it.
- **Holes and crowding trade against each other on one scale.** Closing a hole
  buys 1.0; splitting a clean tile between two cells costs 0.26. No price
  parameter mediates it.

**Negative**

- **Every objective value recorded before this change is incomparable**, in
  scale as well as value: the old best was ≈ 2.2 and unbounded, the new range is
  `[0, 1]`. `src/evaluation/runs.py` does catch this one, because the deleted
  `kpi.objective` block is part of the KPI definition it compares — but only by
  accident of the block disappearing. Old runs were deleted rather than pooled.
- **Coverage strength no longer scores at all.** A tile at -119 dBm with one
  server and a tile at -70 dBm with one server are worth the same. This is the
  deliberate reversal of ADR 0008, and it means `weak_rate` and `rsrp_p05_dbm`
  are now the only place a marginal network shows up. Read them before accepting
  a result.
- **The lower layers' tilts move `J` weakly, and the lower layers get worse.**
  Only the priority band is scored, and on the committed baseline b2600 is that
  band on 71.2% of tiles against b1800's 5.8% and b700's 11.8%. Twelve of the 36
  decision variables therefore act on a small slice of the map, and the search
  has a flatter surface there than the per-band sum gave it.

  Measured on the 2026-09-20 run, this is not only a gradient problem. Every
  method improved co-band overlap on the scored band and let the unscored ones
  drift:

  | share of tiles with a co-band neighbour | incumbent | rule | random | turbo |
  |---|---:|---:|---:|---:|
  | b2600 (scored on 71% of tiles) | 0.2010 | 0.1901 | 0.1788 | 0.1718 |
  | b1800 | 0.2067 | 0.2092 | 0.2064 | 0.2082 |
  | b700 | 0.2396 | 0.2456 | 0.2496 | 0.2440 |
  | band-collapsed (the reported KPI) | 0.3164 | 0.3247 | 0.3368 | 0.3402 |

  The objective does what it was built to do on the layer it reads, and the
  reported `overlap_rate` gets worse anyway. Whether to price the non-serving
  layers is the first thing to revisit.
- **The searches trade overlap for holes, as the exchange rate says they will.**
  On the 2026-09-20 run all three methods lowered the hole rate and raised the
  band-collapsed overlap rate. Overlap-reducing candidates existed - the rule
  sweep evaluated one at 0.3067 against the incumbent's 0.3164 - and `J` did not
  select them. This is the intended behaviour, stated here so it is not
  rediscovered as a bug.
- **`J` and the reported KPI set can disagree on the winner.** On the same run
  TuRBO won `J` (0.8231 against the rule sweep's 0.8189) while the rule sweep
  won 8 of the 11 KPIs, because `J` does not read signal strength above `T_cov`
  and that is where the sweep's advantage lay. Under ADR 0007/0008 the two
  agreed; they no longer have to.
- **`u` is not monotone in `lambda`.** It rises from 0 to 1 and falls after, so a
  hole and a four-way overlap can score alike (0 against 0.199). They are not
  alike operationally, and `hole_rate` against `overlap_rate` is what separates
  them in the report.

## Alternatives considered

**Keep the per-band sum and only replace the coverage utility.** Every cell-band
tilt would keep a term it moves, which is what ADR 0007 introduced the split
for. Rejected: it reintroduces scoring bands that would never serve the tile,
which is one of the three faults being fixed, and it puts `J`'s bound back at
`n_band` rather than 1.

**`alpha_b` on the tile weights**, as first specified. Rejected above: it makes
the weights a function of the decision variable.

**A per-interval demand intensity** for `r`. Rejected on measurement, above.

**Add a strength term back, e.g. `u * min(1, (R_1 - T_cov) / 20)`.** Rejected:
it reintroduces a free parameter and the unbounded-strength incentive in a
milder form, and the reported RSRP percentiles already carry strength. If the
search starts producing effectively-covered but marginal maps, this is the
first thing to revisit.

**Average over bands instead of picking one**, i.e. mean `lambda` across covered
bands. Rejected: it scores a tile on layers that would not serve it, in a way
that a tilt on the serving layer cannot fix.
