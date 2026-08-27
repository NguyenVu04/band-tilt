# Architecture decision records

An ADR captures one architecturally significant decision: what we chose, what
else we considered, and what it costs us. It is written when the decision is
made, while the alternatives are still fresh, and it is never rewritten
afterwards — a decision that turns out wrong gets a *new* record that supersedes
the old one.

The point is not process. It is that in two years someone will ask why a
coverage hole is defined at −120 dBm, why a hole outranks an overlap rather
than being traded off against it, or why a reported KPI may not come from the
surrogate — and the answer will otherwise have left with whoever made the call.

## Relationship to PROJECT.md

`PROJECT.md` is the specification. It states the current formulation and is
revised as the research progresses.

These records are the history. Where they overlap, PROJECT.md says *what* and the
ADR says *why*, including the alternative that was rejected and what rejecting it
cost. PROJECT.md section 29 lists the confirmed decisions in a table; most rows
of that table have a record here.

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

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-28 |
| [0002](0002-five-kpis-under-lexicographic-priority.md) | Five KPIs under lexicographic priority | Accepted | 2026-08-28 |
| [0003](0003-sionna-rt-is-ground-truth.md) | Sionna-RT is ground truth; the surrogate only accelerates | Accepted | 2026-08-28 |
