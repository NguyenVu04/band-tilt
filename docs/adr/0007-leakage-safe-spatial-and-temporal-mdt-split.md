# 7. Leakage-safe spatial and temporal MDT split

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

The MDT export holds 41,481 measurements from 1,490 UEs over roughly two weeks.
Those records feed the project in one specific way: they become `rho(g)`, the UE
observation count per grid cell, which is the spatial weight in the UE-weighted
Band Priority Score (PROJECT.md sections 7 and 14).

So the split question is not the usual supervised-learning one. Nothing is fitted
on individual MDT rows. What the split protects is the *objective itself*: if the
density map that the optimizer maximises against is built from the same
measurements used to score the result, the optimization has been tuned to the
evaluation data.

MDT records are correlated along two axes at once, and a random split ignores
both.

*Spatially.* Consecutive points on one trajectory are metres apart. They see
almost the same buildings, the same serving cell and the same propagation paths.
A random split puts one in train and its near-twin in test.

*Temporally.* One UE contributes a burst of measurements over a single session,
during which the network configuration is fixed. Splitting within a burst leaks
the configuration as much as the position.

A plain random split is therefore optimistic for a reason that produces no error
and no warning — the numbers simply come out better than they should.

There is a further complication: "leakage-safe" is not one thing. Holding out
regions and holding out time answer different questions, and they give different
numbers.

## Decision

`configs/data.yaml` declares the split method, and `src/data/split.py` is the only
code in the project that splits data. Four methods are available:

| Method | Holds out | Answers |
|---|---|---|
| `spatial_block` | whole grid blocks of `block_size_m` | does this transfer to places with no measurements? |
| `temporal` | the tail of the observation window | does this still hold next week? |
| `group_shuffle` | whole `group_col` values, normally `ue_id` | does this transfer to unseen users? |
| `random` | individual records | *demonstration only* |

**`spatial_block` is the default**, because the Band Priority Score is a spatial
weighting and the question that matters is whether an optimized configuration
holds in areas the density map did not see.

`random` exists solely so the size of the leakage can be measured against a
defensible baseline. **It must never back a reported result.**

Consequences that follow from the split:

- UE density used during optimization is built from the **training** split
  (notebooks 02, 03, 04).
- UE density used in final validation is rebuilt from the **test** split
  (notebook 06).
- The inner train/validation split uses the same method as the outer split.
  Validating on a random fold of a spatially split training set measures a
  different kind of generalisation from the one the test set measures, so the two
  numbers could not be read together.
- `split.assert_no_leakage` checks the condition appropriate to the configured
  method, and is called after every split.

**Which method a reported result used must be stated.** The numbers are not
comparable across methods.

## Consequences

**Positive**

- The Band Priority Score reported at the end is computed against measurements
  that played no part in shaping the objective it scores.
- Making the method explicit and configurable means the choice is a stated
  research decision rather than a default nobody examined.
- Keeping `random` available means the cost of leaking can be quantified, which
  is a far stronger argument than asserting it exists.
- The inner-split rule stops a subtle version of the same error, where the outer
  split is careful and the inner one is not.

**Negative**

- Spatial blocks cannot hit the requested test fraction exactly — whole blocks
  move together — so `test_size` is approximate, and more so at coarse
  `block_size_m`.
- Holding out whole regions can leave the test set with a systematically
  different radio environment, so train and test KPI distributions may differ for
  reasons that are not error. Notebook 01 compares them, and the difference has
  to be understood rather than assumed away.
- The two density maps mean the optimizer maximised against one weighting and is
  scored against another. That is the intended protection, but it means part of
  any gap between predicted and validated Band Priority Score is the density
  difference rather than model error, and the two are not separable.
- `block_size_m` is an unforced parameter with no principled value. Too small and
  neighbouring blocks still leak; too large and the test set stops representing
  the area.

**Neutral**

- Four methods means four code paths in the splitter and four cases in the
  leakage assertion.
- `random` remains in the codebase as an attractive nuisance. The config comment
  and this record are what stand between it and a reported result.

## Alternatives considered

**A plain random split.** Simple, hits the requested fraction exactly, and keeps
train and test distributions matched. Rejected because the records are correlated
in exactly the way that makes it invalid, and the resulting optimism is invisible
— no error, no warning, just better numbers.

**`group_shuffle` on `ue_id` as the default.** The standard answer for correlated
records, and it removes the temporal correlation within a session cleanly.
Rejected as the default because two different UEs walking the same street produce
nearly identical measurements, so holding out users does not hold out places —
and places are what the Band Priority Score weights. Kept as an option, since it
is the right choice if the question ever becomes about user generalisation.

**Temporal split as the default.** Matches how the system would be deployed:
optimise on what has been seen, apply going forward. Rejected as the default
because the observation window is roughly two weeks and the cell configuration
carries a single `sync_date`, so a temporal split holds out a period during which
nothing structural changed — it tests very little. Kept as an option for when a
longer window is available.

**No split at all; use every measurement for density and report accordingly.**
Defensible on the grounds that density is a fixed input rather than a fitted
model, and it maximises the data behind the density map. Rejected because the
optimizer *does* maximise a score computed from that map, so the map is part of
the objective — and scoring the result on the same measurements makes the final
Band Priority Score partly a measure of how well the optimizer fitted the
evaluation data.
