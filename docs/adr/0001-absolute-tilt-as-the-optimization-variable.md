# 1. Absolute tilt as the optimization variable

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

The deliverable of this project is a tilt setting for every `(cell, band)` pair.
There are two ways to parameterise the thing an optimizer actually emits, and
they look almost equivalent:

**Absolute tilt.** The optimizer outputs `theta`, the tilt the antenna should be
set to. The feasible set is `[theta_min, theta_max]` per cell-band — a fixed box,
determined by the hardware and the deployment.

**Tilt offset.** The optimizer outputs `delta`, how far to move from where the
antenna currently is. This is closer to how a change is described operationally
("tilt sector 3 down by two degrees"), and it makes small changes the natural
default.

They are not equivalent, because the offset formulation makes the feasible set
depend on the current configuration:

    delta in [theta_min - theta_current, theta_max - theta_current]

A cell currently at 6° with bounds `[0°, 15°]` may move within `[-6°, +9°]`. The
same cell at 12° may move within `[-12°, +3°]`. The action space is a different
box for every cell, and it changes whenever the network changes.

That matters differently for the two methods being compared. Bayesian
Optimization can handle per-dimension bounds, but a Gaussian process fitted over
a space whose axes shift between problem instances cannot transfer between them.
For MARL it is worse: the policy's output range would have to be re-derived per
episode from the current tilt, and a policy trained on one starting configuration
learns an action space that no longer exists when the network moves.

There is a separate question tangled up with this one: whether the *size* of the
change should be penalised. Operators care about how many antennas have to be
touched, and an objective that includes `|delta|` would prefer configurations
close to the current one.

## Decision

The optimizer emits **absolute tilt**. The decision vector is

    theta = [theta_1_1, theta_1_2, ..., theta_N_B]

subject to `theta_min[i,b] <= theta[i,b] <= theta_max[i,b]`, and nothing else.

Absolute tilt is defined as `tilt = eTilt + mTilt` (PROJECT.md section 4.2). The
sum is the variable; splitting it back into a settable electrical tilt and a
fixed mechanical tilt is a deployment concern handled at reporting time.

The tilt offset is **derived after optimization, for reporting only**:

    delta = theta_star - theta_current

It is not a decision variable, not a KPI, and not a term in the objective. There
is no penalty on `|delta|`.

The search space is defined once, in `src/optim/space.py`, and both optimizers
read their bounds from it. The offset is computed once, in
`src/radio/cell_band.tilt_offset`, and appears only in the final report.

## Consequences

**Positive**

- The action space is a fixed box, identical for every problem instance. A MARL
  policy and a BO surrogate can both transfer to a network in a different state.
- BO and MARL search provably the same set, which is what PROJECT.md section 25
  requires for the comparison to mean anything.
- The physical bounds are the entire constraint set, so feasibility is a single
  elementwise check rather than a state-dependent computation.
- The formulation matches the research question, which is what tilt configuration
  is best — not how to get there from here.

**Negative**

- The optimizer has no reason to prefer a nearby configuration, so it may return
  a solution that moves every antenna in the network. That is operationally
  expensive and the result does not say so.
- "Improve this network" and "design this network" become the same problem, and
  the current configuration carries no privileged status beyond being the
  reporting baseline.
- If a deployment later caps how far a tilt may move, that constraint has to be
  expressed by narrowing the bounds in `configs/radio.yaml` and re-running —
  which is a per-cell edit, not a single parameter.

**Neutral**

- The offset still appears in the deliverable, so the operational reading is not
  lost. It is computed, not optimised.
- Nothing prevents adding a reconfiguration constraint later. It would be an
  operational constraint on the feasible set, not a network-quality KPI, and it
  would supersede this record.

## Alternatives considered

**Optimise the tilt offset directly.** Matches how operators describe a change,
and biases naturally toward small adjustments. Rejected because the feasible
range then depends on the current configuration, so the action space moves with
the network — which breaks policy and surrogate transfer, and makes the BO and
MARL search spaces harder to prove identical.

**Optimise absolute tilt, but add a penalty on `|delta|`.** Keeps the fixed action
space while discouraging large reconfigurations. Rejected because it makes the
objective answer a different question from the one being researched: the
comparison between BO and MARL would then partly measure how each trades network
quality against reconfiguration effort, with a weight nobody has a principled
value for. PROJECT.md section 3.3 is explicit that the research objective is
network optimization, not minimising configuration change.

**Optimise `eTilt` alone, holding `mTilt` fixed.** Closest to what a remote
electrical tilt unit actually controls. Rejected because coverage depends on the
sum, so two cells with the same `eTilt` and different `mTilt` are different
networks — the optimizer would be searching a variable that does not determine
the outcome. The split is recoverable at reporting time from the known `mTilt`.
