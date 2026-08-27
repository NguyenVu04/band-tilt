# 3. Sionna-RT is ground truth; the surrogate only accelerates

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

Evaluating one tilt configuration means solving a radio map over a scene of 3,753
meshes and then computing five KPIs from it. That is expensive enough that the
cost, not the algorithm, determines what the project can do.

The two optimizers need very different evaluation counts. Bayesian Optimization is
designed for expensive objectives and might get by on a few hundred. MARL needs
orders of magnitude more — a training run is thousands of environment steps, each
of which would be a full ray-tracing solve. Training MARL directly against
Sionna-RT is not affordable, and PROJECT.md section 24 says so explicitly.

The obvious fix is a surrogate: train a model on `(state, tilt) -> KPI` pairs from
a few hundred simulated configurations, then let the optimizers query the model
instead of the simulator.

The obvious fix has an equally obvious failure mode, and it is not a general
worry about model error. An optimizer given an approximate objective will
systematically find the regions where the approximation is most *optimistic* —
that is what optimization does. Ordinary model error is symmetric; error under
optimization pressure is not. A surrogate that is accurate on average will still
be exploited exactly where it is wrong in the useful direction, and the resulting
configuration will look excellent and not be.

Compounding this, the BO loop contains a *second* approximation: the Gaussian
process Ax fits online over evaluated points. Running BO against the KPI
surrogate means fitting a GP to the predictions of another model — two error
sources stacked, with the optimizer pushing on both.

## Decision

**Sionna-RT is the source of truth. The surrogate is an acceleration mechanism
and never the source of a reported number.**

Concretely:

1. The surrogate may be used freely *inside* an optimization loop — as the BO
   objective, as the MARL environment's reward.
2. Every configuration that appears in a result is re-evaluated with Sionna-RT
   before it is reported. BO validates its top `validate_top_k` candidates; MARL
   validates the configuration its trained policy produces.
3. The reported KPI vector is the Sionna-RT one. Where a surrogate prediction
   appears in a report it is labelled as such and shown beside the ground truth.
4. The **gap** between prediction and ground truth is itself reported. It is the
   evidence that the surrogate was not exploited, and a gap larger than the
   improvement being claimed means the result is not supported.
5. Surrogate acceptance is decided per KPI, in each KPI's own units, and
   additionally over the best-performing decile — because global accuracy says
   nothing about accuracy where an optimizer actually looks.

Sionna-RT is also used for the initial dataset, for the baseline, and for
monitoring surrogate error over time.

## Consequences

**Positive**

- MARL becomes trainable at all, which is the precondition for the comparison
  this project exists to make.
- No reported number depends on a learned approximation, so a result survives the
  surrogate later turning out to be worse than believed.
- The prediction-versus-truth gap is a first-class output, so surrogate
  exploitation is visible rather than silent.
- Per-KPI and near-optimum error reporting catches the specific failure that a
  single averaged error metric hides.

**Negative**

- Every optimization run ends with a validation phase that costs real
  ray-tracing time, and the budget for it has to be planned.
- BO and MARL do not consume the surrogate equally — MARL leans on it far harder
  — so part of any observed difference between them is a difference in how much
  approximation each tolerated. That has to be stated when reporting, not
  explained away.
- Validating only the top *k* candidates means a genuinely good configuration
  the surrogate under-rated is never checked. The bias is toward false
  positives being caught and false negatives being missed.
- Two evaluation paths exist, so they can disagree for reasons other than
  surrogate error — a grid or scene mismatch between them would look like model
  error. `src.optim.objective` exists to keep them behind one interface.

**Neutral**

- The surrogate's own error report becomes a published result (PROJECT.md
  section 19 Step 6), not an internal diagnostic.
- Changing the ray-tracing settings changes the ground truth, so it invalidates
  every existing surrogate sample rather than adding to them. The DVC stage
  expresses this as a parameter dependency.

## Alternatives considered

**Optimise directly against Sionna-RT, no surrogate.** Every number is exact and
the whole class of surrogate-exploitation failures disappears. Rejected because
MARL is then untrainable, and the project reduces to a BO study — which removes
the comparison that is the research question.

**Trust the surrogate and skip Sionna-RT validation.** Much cheaper, and
defensible if the surrogate's held-out error is small. Rejected because held-out
error is measured on *sampled* configurations while the reported configuration is
*selected by an optimizer searching for the surrogate's optimistic regions*. Those
are different distributions, and the second is chosen adversarially with respect
to the first.

**Use the surrogate only to pre-screen, evaluating every shortlisted candidate.**
A middle position, and close to what is adopted. Rejected as a general rule
because MARL does not have a shortlist: the policy is shaped by millions of
surrogate rewards during training, not by a set of candidates that could be
screened. The adopted decision is the same idea applied where it works —
validation at the end — with the honest acknowledgement that MARL's training
signal was never validated.

**Retrain the surrogate on the optimizer's own proposals as it goes.** Directly
targets the exploitation problem: the model gets corrected exactly where the
optimizer probes. Rejected for now on cost and complexity — it needs a Sionna-RT
solve inside the loop, which is the expense the surrogate exists to avoid — but
it is the natural extension if the validation gaps turn out to be large, and it
would supersede this record.
