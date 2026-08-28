# 3. Sionna-RT is ground truth; the surrogate only accelerates

- **Status:** Accepted
- **Date:** 2026-08-28
- **Revised:** 2026-08-28 — revised in place to follow the PROJECT.md rewrite.
  The surrogate now predicts a radio map rather than a KPI vector; the core
  decision below is unchanged. See *Revision note*.
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
Sionna-RT is not affordable, and PROJECT.md section 11.4 says so explicitly.

The obvious fix is a surrogate: train a model on `(x, tilt) -> R` pairs from a few
hundred simulated configurations, then let the optimizers query the model instead
of the simulator and run the KPI evaluator over what it predicts.

The obvious fix has an equally obvious failure mode, and it is not a general
worry about model error. An optimizer given an approximate objective will
systematically find the regions where the approximation is most *optimistic* —
that is what optimization does. Ordinary model error is symmetric; error under
optimization pressure is not. A surrogate that is accurate on average will still
be exploited exactly where it is wrong in the useful direction, and the resulting
configuration will look excellent and not be.

Compounding this, the TuRBO loop contains a *second* approximation: the Gaussian
process it fits online over evaluated points, inside its trust region. Running
TuRBO against the surrogate means fitting a GP to KPIs derived from the
predictions of another model — two error sources stacked, with the optimizer
pushing on both.

## Decision

**Sionna-RT is the source of truth. The surrogate is an acceleration mechanism
and never the source of a reported number.**

Concretely:

1. The surrogate may be used freely *inside* an optimization loop — as the source
   of the TuRBO objective, as the MARL environment's reward.
2. **The surrogate predicts the radio map `R_hat`, not the KPIs** (PROJECT.md
   section 11, Decision 6). The five KPIs are always derived from a radio map by
   `src/kpi/`, whether that map came from Sionna-RT or from the model. There is
   one evaluator, and swapping the map underneath it is the only difference
   between a predicted score and a true one.
3. Every configuration that appears in a result is re-evaluated with Sionna-RT
   before it is reported. TuRBO validates its top `validate_top_k` candidates;
   MARL validates the configuration its trained policy produces.
4. The reported KPI vector is the Sionna-RT one. Where a surrogate prediction
   appears in a report it is labelled as such and shown beside the ground truth.
5. The **gap** between prediction and ground truth is itself reported, at both
   levels: radio-map error in dB, and the KPI error that error produces. It is
   the evidence that the surrogate was not exploited, and a gap larger than the
   improvement being claimed means the result is not supported.
6. Surrogate acceptance is decided on radio-map error (PROJECT.md section 11.3)
   **and** per derived KPI in each KPI's own units, additionally over the
   best-performing decile — because global accuracy says nothing about accuracy
   where an optimizer actually looks.

Sionna-RT is also used for the initial dataset, for the baseline, and for
monitoring surrogate error over time. Once the surrogate meets its acceptance
threshold it is **frozen** for the whole optimization phase (PROJECT.md section
11.4); it is not retrained on candidates the optimizer proposes.

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
- Predicting the map rather than the KPIs means the KPI definitions can change
  without retraining the surrogate, and a surrogate error is attributable to
  propagation prediction rather than hidden inside a KPI regression.

**Negative**

- Every optimization run ends with a validation phase that costs real
  ray-tracing time, and the budget for it has to be planned.
- TuRBO and MARL do not consume the surrogate equally — MARL leans on it far
  harder — so part of any observed difference between them is a difference in how
  much approximation each tolerated. That has to be stated when reporting, not
  explained away.
- Predicting a full `(location, cell, band)` tensor is a far larger output than
  five numbers, so the surrogate is more expensive to train and to query, and a
  small per-pixel error can still move a KPI if it lands near a threshold. Hole
  rate in particular is a hard threshold at -120 dBm: map error concentrated at
  the coverage edge costs more than the same error in the cell centre.
- Validating only the top *k* candidates means a genuinely good configuration
  the surrogate under-rated is never checked. The bias is toward false
  positives being caught and false negatives being missed.
- Two evaluation paths exist, so they can disagree for reasons other than
  surrogate error — a grid or scene mismatch between them would look like model
  error. `src.optim.objective` exists to keep them behind one interface.

**Neutral**

- The surrogate's own error report becomes a published result (PROJECT.md
  section 16 Phase 5), not an internal diagnostic. It now has two halves:
  radio-map error and derived-KPI error.
- Changing the ray-tracing settings changes the ground truth, so it invalidates
  every existing surrogate sample rather than adding to them. The DVC stage
  expresses this as a parameter dependency.

## Alternatives considered

**Optimise directly against Sionna-RT, no surrogate.** Every number is exact and
the whole class of surrogate-exploitation failures disappears. Rejected because
MARL is then untrainable, and the project reduces to a TuRBO study — which
removes the comparison that is the research question.

**Predict the five KPIs directly instead of the radio map.** This is what the
record originally specified: a much smaller output, cheaper to train, and no
intermediate tensor to store. Rejected in the revision because it forces two KPI
code paths — one computing KPIs from Sionna-RT maps, one regressing them — which
is exactly the failure PROJECT.md section 25.4 names, and because any change to a
KPI threshold would invalidate the trained model rather than just the labels.

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

## Revision note — 2026-08-28

Revised in place rather than superseded, alongside
[0002](0002-five-kpis-under-lexicographic-priority.md); the original text is in
Git history at `abcdf6c`.

The **core decision is unchanged**: Sionna-RT is ground truth, the surrogate only
accelerates, and no reported number comes from the model. What changed is what the
surrogate emits — a radio map `R_hat` instead of a KPI vector `K_hat` (PROJECT.md
section 11 and Decision 6) — which moves the KPI evaluator downstream of both the
simulator and the model instead of duplicating it. Acceptance and gap reporting
gained a radio-map layer as a result, and the freeze-before-optimization rule is
now stated explicitly.

`src/surrogate/` and `configs/surrogate.yaml` still describe the KPI-predicting
interface in places; that divergence is tracked in `CLAUDE.md` under *Known gaps*.
