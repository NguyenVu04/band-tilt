# Architecture decision records

An ADR captures one architecturally significant decision: what we chose, what
else we considered, and what it costs us. It is written when the decision is
made, while the alternatives are still fresh, and it is never rewritten
afterwards — a decision that turns out wrong gets a *new* record that supersedes
the old one.

The point is not process. It is that in two years someone will ask why a
coverage hole is defined at −120 dBm, why a hole outranks an overlap rather
than being traded off against it, or why the split falls between scenarios
rather than records — and the answer will otherwise have left with whoever made
the call.

## Relationship to the rest of the repository

There is no separate specification document. The code, the configs and
`README.md` state *what* the system does, and they are kept current.

These records are the history. They say *why*, including the alternative that was
rejected and what rejecting it cost — the part that does not belong in a document
that has to stay current, because a rejected alternative never stops being true.

## When to write one

Write an ADR when a decision is **costly to reverse** and **not obvious from the
code**:

- a change to the KPI definitions, their thresholds, or their priority order —
  every result produced before the change becomes incomparable;
- a change to the decision variable, the search space, or the tilt bounds;
- a change to the coordinate or angle conventions;
- a change to what a reported result is allowed to be computed from;
- a change to the split scheme, which decides what the final estimate measures;
- choosing or replacing a simulator, an optimization framework, or a third-party
  service;
- accepting a known trade-off — fidelity for speed, strictness for tractability —
  that a future reader would otherwise read as an accident.

Do not write one for a decision the code states plainly, a reversible choice, or
a matter of style the linter already settles. Hyperparameters are not ADRs;
`configs/` and MLflow record those.

## How

1. Copy [`0000-record-architecture-decisions.md`](0000-record-architecture-decisions.md)
   to `NNNN-short-title-in-kebab-case.md`, where `NNNN` is the next unused number.
   Numbers are never reused, even if a record is withdrawn.
2. Fill it in. Aim for one page. If it needs more, the decision probably contains
   two decisions.
3. Open it as a pull request with status **Proposed**, and let the discussion
   happen in review rather than in the record.
4. On approval, set the status to **Accepted** and add a row to the index below.

## Lifecycle

| Status | Meaning |
|---|---|
| **Proposed** | Under discussion; not yet binding |
| **Accepted** | In force. This is how the system works |
| **Superseded by NNNN** | Replaced. Kept as history — never delete or edit the reasoning |
| **Deprecated** | No longer applies, with nothing replacing it |

Superseding a record means editing exactly two lines: the old record's status,
and the new record's `Supersedes` line. The old record's Context and Decision
stay as they were written, because they are the historical account.

### Revision in place — the exception, not the practice

0002 was **revised in place** on 2026-09-09, at the maintainer's direction, when
the search moved onto a surrogate and the deliverable became the verified Pareto
front rather than a single configuration.

0001 was **revised in place** on 2026-08-28, at the maintainer's direction,
rather than superseded by a new record. The surviving records were revised
in place again on 2026-08-29, also at the maintainer's direction, to remove the
citations to a specification document that is no longer treated as a source of
truth. Both are departures from the rule above and are recorded as such: each
carries a `Revised` line in its header and a *Revision note* section stating
exactly what changed and why. The pre-revision text is in Git history.

Prefer superseding. Revision in place loses the shape of the original argument,
which is the thing these records exist to preserve.

## Index

| # | Title | Status | Date | Revised |
|---|---|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-28 | 2026-08-29 |
| [0001](0001-five-kpis-under-lexicographic-priority.md) | Four KPIs under lexicographic priority | Accepted | 2026-08-28 | 2026-09-12 |
| [0002](0002-bayesian-optimization-without-a-trust-region.md) | Bayesian optimization without a trust region | Accepted | 2026-09-07 | 2026-09-09 |
