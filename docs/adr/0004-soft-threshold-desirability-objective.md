# 4. A soft-threshold desirability objective

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** the objective and winner rule of [ADR 0003](0003-turbo-on-a-weighted-kpi-score.md)
- **Superseded by:** —

## Context

ADR 0003 scored every configuration by the raw weighted sum of four KPIs, and
listed two consequences it could not remove: raw values let a KPI with a wide
range dominate, and a gain on a lower-weighted KPI can outweigh a loss on a
higher one. Both follow from the same property — a weighted sum is
*compensatory*, so there is no value of a KPI bad enough that the others cannot
pay for it. A weighted sum cannot express "this must be at least this good".

The sum is also indifferent to where a KPI already stands. Improving the hole
rate from 0.60 to 0.59 pays exactly what improving it from 0.20 to 0.19 pays,
though only one of those is worth spending tilt on.

Separately, the maintainer asked for a single published figure for overall
network quality. A second scalar derived differently from the objective would be
free to disagree with it about which configuration is best.

## Decision

**Each objective KPI becomes a desirability** (Derringer & Suich, 1980,
*Simultaneous Optimization of Several Response Variables*), in `src/kpi/soft.py`:

```
d_k = sigmoid(s_k * (KPI_k - target_k) / T_k)
```

with `s_k` the sign of `MAXIMISED`, `target_k` the value scoring 0.5 and `T_k`
the units the curve turns over — both from `kpi.soft` in `configs/kpi.yaml`. It is
applied **per tile**, inside the KPI, so those units are the tile's own: dBm for
coverage, neighbours for overlap, a share for the served term.

The transform is strictly monotone, so it cannot reorder tiles on a single KPI;
what it changes is the exchange rate between KPIs once the tiles are averaged.
Past its target a KPI saturates and stops paying, so the search spends its moves
on whichever KPI is still short.

**The temperature is mandatory and validated.** `soft_spec` raises on a missing
or non-positive one. On rates that move by thousandths between candidates, a
temperature of 1 makes the sigmoid indistinguishable from a straight line, which
would silently restore the weighted sum this ADR replaces.

**The objective is three KPIs, not four.** `OBJECTIVE_NAMES` is the hole rate,
the overlap rate and the served ratio. `weak_rate` stays measured and reported —
`src/evaluation/maps.py` needs `kpi.weak_dbm` for its coverage classes, and it
is how the classic downtilt failure shows up — but it no longer steers the
search.

**Every objective KPI is softened per tile, before aggregation**, in `src/kpi/`,
each with a `*_desirability` twin beside the rate it softens:

| KPI | Softened quantity | Units of target / temperature |
|---|---|---|
| `hole_desirability` | the tile's best-server RSRP | dBm / dB |
| `overlap_desirability` | the tile's co-band neighbour count | neighbours |
| `served_desirability` | the tile's served share of UE reports | share |

Softening the aggregate *rate* instead was the first implementation and is not
enough: it leaves the hard comparison inside each KPI untouched. To `hole_rate`,
a tile 0.1 dB short of coverage and one 40 dB short are the same tile, so a tilt
change lifting a tile from −125 to −121 dBm moves the rate not at all, while the
same change across the threshold moves it by a whole tile. No curve applied
afterwards recovers what the step discarded. The transform belongs where the
comparison is made.

Two consequences of the per-tile form, both accepted:

- **Tiles carrying no UE report are excluded from the served term**: scoring them
  zero would punish a configuration for not covering ground nobody stands on, and
  scoring them one would dilute the term with empty map. A no-path tile in the
  hole term is *not* excluded — it scores exactly zero, the worst possible tile,
  matching `hole_rate`'s own convention.
- **The dynamic range is roughly ten times smaller** than the aggregate form's,
  because most tiles sit far from the threshold and contribute no gradient. This
  is cosmetic rather than a loss: the aggregate sigmoid amplified signal and noise
  by the same factor. Measured on the committed scenario, the objective's spread
  across the tilt box is about **29 times the solver-seed noise**.

**Weights are normalised, and the combination is geometric.** The score is

```
D = prod_k d_k ** w_k,   w = kpi.weights over OBJECTIVE_NAMES, renormalised to sum 1
```

a weighted geometric mean in `[0, 1]`. Multiplicative rather than additive is the
whole point: it is **non-compensatory**, so one collapsed KPI drives `D` to zero
and cannot be bought back.

**`D` is both the objective and the published quality index.** TuRBO maximises
it, the winner is the highest one, and it is the overall network quality figure.
One number, so the thing optimised and the thing reported cannot drift apart.

**A fifth KPI is measured and never optimised.** `edge_rsrp_dbm` is the
5th-percentile serving RSRP over covered locations, the cell-edge measure of
3GPP TR 36.814 Annex A.2.1.4. It is conditional on coverage, so it can improve
by covering less; it is read beside `hole_rate` and enters no score.

**The hard score is kept, unchanged, as an audit.** `objective.scores` is still
ADR 0003's raw weighted sum over all four KPIs.
`objective.selection_scores` dispatches on `optim.objective` (`soft` | `hard`),
and every search, winner and published table goes through it. Both scores appear
in every published table as `score` and `score_hard`.

## Consequences

**Positive**

- The formulation can now express a floor. A configuration that abandons
  coverage cannot win by serving its remaining UEs superbly.
- Diminishing returns past a target stop the search buying more of a KPI that is
  already comfortable, which is what the weights were being asked to do and
  could not.
- `D` is unitless and bounded, so weightings are comparable and the published
  quality figure is the objective rather than a second opinion about it.
- ADR 0003's first two Negative entries are removed rather than measured.

**Negative**

- **Targets and temperatures are new judgement values.** They are less fragile
  than in the aggregate form, where a target far from the incumbent put every
  candidate on a flat tail: a per-tile target is a physical quantity (the coverage
  edge in dBm, one crowding neighbour) rather than a value that has to be measured
  first. The risk that remains is a temperature so narrow that few tiles
  contribute, so `reports/review.md` reports the realised `d` range and the
  signal-to-noise ratio for every run.
- **The whole `kpi.soft` block is read at measure time**, not score time, because
  a per-tile softening cannot be recovered from the scalar it averages to.
  Changing any target or temperature invalidates stored measurements rather than
  merely re-scoring them. `compare.weight_sensitivity` can still re-weight an
  archived run; it cannot re-temperature one.
- **The hard rates keep their thresholds and are still measured.** `hole_rate`,
  `overlap_rate` and `served_ratio` feed the audit score and every coverage-class
  figure, so `kpi.hole_dbm`, `kpi.weak_dbm` and `kpi.overlap_margin_db` stay live.
  Two definitions of a hole now exist; they agree on where the edge is, because
  `kpi.soft.hole_rate.target` is set to `kpi.hole_dbm`, and differ only in how
  sharply.
- **The overlap margin is still hard.** Only the `> 0` indicator was softened, so
  a neighbour 6.1 dB down still contributes nothing to the count.
- A geometric mean is undefined on a negative term, so every desirability must
  stay in `[0, 1]` — which is why `edge_rsrp_dbm`, in dBm, cannot enter it.
- Runs before this ADR used a different objective and winner rule and are not
  comparable with runs after it. `runs.verify` refuses to compare them.

## Alternatives considered

**A normalised weighted sum of the desirabilities.** Same `[0, 1]` range and
easier to explain, rejected because it is still compensatory: an excellent served
ratio would mask a collapsed hole rate, which is the failure thresholds exist to
prevent.

**A hard constraint** (maximise the served ratio subject to a hole-rate cap).
Directly deployable, and rejected for the reason ADR 0001 already gave: the
sensible cap is not known before seeing what the tilt space can achieve. A soft
threshold is the same statement without needing the cap to be right.

**Softening only the aggregate rate** — `sigmoid((target - hole_rate) / T)` on the
share, leaving the per-tile comparison hard. This was the first implementation and
was replaced: it reshapes a KPI's output without removing the step inside it, so
the objective stayed blind to how far below the threshold a tile actually sat. It
survives in git history.

**Softening the 6 dB overlap margin as well.** The most faithful to the physics,
and rejected for now because it changes the meaning of `overlap_neighbors` and
therefore of every overlap figure ever reported. Only the `> 0` indicator was
softened.

**Redefining overlap as pilot pollution** (at least three servers within the
margin, rather than at least one). A redefinition rather than a softening, and it
targets a real defect — "any neighbour within 6 dB" covers most of the map — but
it would break comparability with every earlier run and is left open.
