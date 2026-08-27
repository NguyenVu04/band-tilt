# 6. Two optimizers, one problem definition

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

The research question is whether Bayesian Optimization or Multi-Agent
Reinforcement Learning is the better approach to multi-band tilt coordination.
PROJECT.md section 25 requires them to be evaluated on the same input data, radio
map, action space, tilt constraints, KPI definitions, surrogate and validation
procedure.

That requirement is easy to state and easy to violate without noticing. The two
methods have genuinely different natural idioms: BO wants a bounded box and
usually searches the unit cube; RL wants an environment with an action spec and
usually squashes a network output into range. Implemented independently, they
drift — one clips where the other squashes, one works in degrees where the other
works normalised, one reads the bounds at construction and the other per episode.
Each difference is individually defensible and none of them raises. The result is
a comparison of two search spaces reported as a comparison of two methods.

The same hazard applies to scoring. A KPI threshold re-derived inside an
optimizer, or a normalisation applied in one and not the other, produces two
objectives that are almost the same.

Framework choice adds a further question. The request named Ax, BoTorch, TorchRL
and Ray. Ax and BoTorch are complementary — Ax is the loop and BoTorch the
model/acquisition layer beneath it. TorchRL and Ray RLlib are alternatives to each
other for the MARL side, and Ray additionally offers parallel execution and
hyperparameter search.

## Decision

**BO and MARL share one problem definition and differ only in how they search
it.**

Structurally:

- `src/optim/space.py` defines the search space `Theta`. Both optimizers derive
  their bounds from it. Neither reads `configs/radio.yaml` directly.
- `src/optim/objective.py` is the single scoring interface. Both call it, and it
  delegates to `src/kpi/`. Neither re-derives a threshold.
- `src/optim/space.TiltSpace.to_unit` / `from_unit` are the only sanctioned
  normalisation, so a candidate cannot be interpreted in the wrong units at one
  end of a loop.
- The number of Sionna-RT evaluations is counted inside the objective, not
  estimated afterwards. It is a headline result (PROJECT.md section 25.2).

Framework choice:

- **Bayesian Optimization: Ax with BoTorch.** Ax provides the loop, trial storage
  and restarts; BoTorch provides the GP and the acquisition functions, including
  multi-objective `qNEHVI` for the Pareto formulation.
- **MARL: TorchRL with TensorDict.** Centralised training with decentralised
  execution, one agent per cell or per site.
- **Ray is not used.** Neither RLlib nor Tune nor its parallel execution.

## Consequences

**Positive**

- Any divergence between the two search spaces is a structural impossibility
  rather than a review responsibility.
- The comparison in notebook 06 is valid by construction: the same bounds, the
  same KPI code, the same surrogate artifact, the same validation path.
- The shared objective means the Sionna-RT evaluation count is measured the same
  way for both, which is what makes the cost comparison meaningful.
- Dropping Ray keeps the dependency surface small: two optimizer stacks, not
  three, and no second MARL implementation to keep in step with the first.

**Negative**

- The shared abstraction is a compromise for both. Ax has its own search-space
  representation and TorchRL its own action specs, so each integration adapts to
  `TiltSpace` rather than using its framework's idiom directly — more glue code,
  and less benefit from framework-native features.
- Without Ray, the expensive surrogate-dataset build in notebook 03 runs
  sequentially. That is the longest job in the project, and parallelising it is
  exactly what Ray would be good at. This is a real cost, accepted deliberately.
- Without Ray Tune, hyperparameter search for the surrogate and the optimizers
  falls back to Hydra multirun, which is sequential and has no early stopping.
- Only one MARL algorithm family gets exercised, so a result showing MARL
  underperforming is a statement about TorchRL MAPPO rather than about MARL.

**Neutral**

- BO and MARL still legitimately differ in acquisition function, policy
  architecture, evaluation count and wall-clock cost. Those differences are the
  results.
- MARL pays a training cost BO does not. Whether it amortises depends on the
  policy transferring to a network it was not trained on, which has to be
  demonstrated rather than assumed — and reported inside the method cost either
  way.
- Adding Ray later is additive: it would parallelise the dataset build without
  changing the problem definition. Replacing TorchRL with RLlib would supersede
  this record.

## Alternatives considered

**Implement each optimizer independently, against `configs/` directly.** Each
gets its framework's natural idiom with no adapter layer, and the code is simpler
to read in isolation. Rejected because it makes the central claim of the project
unverifiable: proving the two searched the same space would then be a matter of
reading both implementations and trusting they agree.

**Use Ray RLlib for MARL.** A mature multi-agent API, built-in distributed
rollouts, and Ray Tune for hyperparameters — a strong fit for the scaling study
in PROJECT.md section 25.4. Rejected in favour of TorchRL for a smaller
dependency surface and closer control over a non-standard environment whose
reward comes from a custom surrogate. The bounded network size (26 cells today)
means distributed rollouts are not yet the bottleneck; the sequential dataset
build is.

**Use both TorchRL and RLlib behind one environment interface, and compare
them.** Would separate "MARL underperformed" from "this MARL implementation
underperformed", which is a real weakness of the adopted decision. Rejected on
scope: two trainers to build, maintain and keep in step, for a comparison that is
secondary to the BO-versus-MARL question.

**Use Ray for parallelism only, keeping TorchRL for MARL.** Ray Tune for
hyperparameters and Ray for parallel Sionna-RT solves, with no RLlib. This was
the original plan and it addresses the sequential-dataset-build cost directly.
Dropped at the user's direction to keep the dependency set minimal for now. It
remains the obvious first addition if the dataset build becomes the bottleneck,
and it would not change the problem definition.
