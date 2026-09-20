# 8. A softplus coverage utility

- **Status:** Superseded by 0009
- **Date:** 2026-09-19
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the coverage utility of
  [ADR 0007](0007-demand-weighted-objective.md), and with it the `[0, n_band]`
  bound on `J`. Everything else in 0007 — the per-band split, the demand blend
  `w_bg`, the co-band overlap discount, the admission ceiling and the eleven
  reported KPIs — is carried over unchanged
- **Superseded by:** [ADR 0009](0009-effective-coverage-objective.md)

## Context

> The motivation for this change is the decider's to state. What follows is the
> mechanical account of what changed and what it costs; the Context section is
> deliberately left for the maintainer rather than reconstructed.

ADR 0007's objective passed the coverage margin through a logistic:

```
u_bg  =  sigmoid((R_sb(g) - T_cov) / tau_R)
```

`sigmoid` saturates. A tile at `T_cov + 2*tau_R` and a tile at `T_cov + 6*tau_R`
score 0.88 and 0.998, a difference the search cannot act on, so above roughly
two `tau_R` the objective stops distinguishing good coverage from excellent
coverage and the gradient in the strong-signal region nearly vanishes. ADR 0007
records the same effect from the other end when `tau_R` is set too small.

## Decision

Replace the logistic with softplus, leaving every other factor as ADR 0007
defines it:

```
J     = sum_b sum_g w_bg * softplus((R_sb(g) - T_cov) / tau_R) * q_ov^m_bg
w_bg  = (1 - alpha_b) / |G| + alpha_b * p_g,      sum_g w_bg = 1
m_bg  = #{ j != s_b(g) on band b : R_j > T_cov and R_s_b - R_j <= Delta_R }
```

`softplus(x) = ln(1 + e^x)`, computed as `numpy.logaddexp(0, x)` so the
exponential cannot overflow on strong tiles. A no-path tile is `R_sb = -inf` and
still scores exactly zero.

The shape either side of the threshold:

- **Below `T_cov`** softplus decays like `e^x`, so a deep hole contributes
  essentially nothing — the same behaviour the logistic had.
- **At `T_cov`** the utility is `ln 2 ≈ 0.693`, where the logistic gave `0.5`.
- **Above `T_cov`** it grows linearly, `u ≈ (R_sb - T_cov) / tau_R`, without
  ceiling. Coverage keeps paying for every further dB.

`tau_R` therefore changes meaning. Under the logistic it was the dB span the
objective could tell apart, estimated as `IQR(R_sb) / 1.349` (ADR 0007,
"Choosing the parameters"). Under softplus it is still the knee width, but above
the knee it is a **unit of account**: `tau_R` dB of extra signal buys one unit of
utility. The estimator is retained — it still sets the knee to the spread the map
has — and the committed value is unchanged.

`beta` and `alpha` are unchanged and their readings in ADR 0007 still hold,
except that `beta` is now a price against an unbounded utility rather than
against one capped at 1: one neighbour still keeps the fraction `exp(-beta)` of a
tile's term, but that term can be several times larger than it could be before.

## Consequences

**Positive**

- The objective distinguishes strong coverage from adequate coverage. The
  logistic's saturated region is gone, so a candidate that lifts already-covered
  tiles is ranked above one that does not.
- There is a usable gradient across the whole RSRP range, not only near the
  threshold — which is what a trust-region method needs to move.
- The search still scores a hole at zero, so nothing about hole avoidance is
  weakened.

**Negative**

- **`J` is no longer bounded.** It is non-negative and unbounded above, not in
  `[0, n_band]`. Every bound statement in `configs/kpi.yaml`, the README and the
  notebooks is rewritten, and the per-band sanity check ADR 0007 prescribes —
  "each band's term should sit well inside `(0, 1)`" — no longer applies. The
  replacement check is that each term stays finite and that the spread of `J`
  across candidates exceeds the solver's re-trace noise.
- **Every objective value recorded before this change is incomparable**, both in
  absolute value and in scale. `src/evaluation/runs.py` already refuses to
  compare runs across a differing `kpi.objective`, but the block's *values* are
  unchanged here, so that guard does not fire on this change. Runs either side of
  it must not be pooled.
- **The objective now rewards raw signal strength without limit.** Where the
  logistic made the threshold the whole story, softplus lets a configuration buy
  `J` by concentrating power on tiles that are already well covered rather than
  by closing holes. `hole_rate` and `rsrp_p05_dbm` in the reported KPIs are where
  that trade shows up, and they are the numbers to read before accepting a
  result.
- **`beta` was set against a bounded utility.** ADR 0007 fixes `beta = 0.25` from
  "what fraction of a tile's coverage value would I give up to remove one
  overlapping neighbour" — a fraction, so it survives the rescaling in relative
  terms. But the *absolute* penalty per neighbour now scales with the tile's
  margin above `T_cov`, so strong tiles are charged more for crowding than weak
  ones. Whether that is the intended trade is not settled here.
- `tau_R`'s estimator was derived for a width the objective "can tell apart".
  It is retained because it still sets a sensible knee, but it is no longer
  derived from the property it is now mostly controlling.

## Alternatives considered

**A bounded softplus soft-clip**, `softplus(x + 1/2) - softplus(x - 1/2)`: the
integral of the logistic over a unit window. It keeps `u` in `(0, 1)`, keeps
`u = 1/2` at `T_cov`, keeps `J` in `[0, n_band]` and so keeps every bound
statement and the ADR 0007 sanity check intact, while giving a sharper knee than
the logistic. Rejected in favour of the unbounded form at the maintainer's
direction: the saturation is the property being removed, and a soft-clip only
moves where it starts.

**Keep the logistic and raise `tau_R`.** Flattens the curve rather than removing
the ceiling, and ADR 0007 already records where that ends: at 100 dB a tile at
`-119` and a tile at `-60` score almost alike and the hole threshold stops
meaning anything.
