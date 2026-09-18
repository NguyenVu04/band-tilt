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

### Rewriting in place — the exception, not the practice

On 2026-09-14, at the maintainer's direction, 0001 and 0003 were rewritten to
describe the current system and 0002 (superseded by 0003) was deleted. This
departs from the rule above; the earlier text, including every revision note,
is in Git history. Each rewritten record says so in its header.

On 2026-09-17, again at the maintainer's direction, 0004 and 0005 were deleted
when 0006 replaced the objective they defined, and the headers of 0001 and 0003
were amended to point at 0006.

Later on 2026-09-17, again at the maintainer's direction, 0006 was rewritten in
place and renamed from `0006-radio-load-cvar-objective.md` when its load term was
removed, and 0001 was amended to match.

Prefer superseding. Rewriting loses the shape of the original argument, which
is the thing these records exist to preserve.

## Index

| # | Title | Status | Date | Rewritten |
|---|---|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-28 | — |
| [0001](0001-four-kpis-and-weighted-score.md) | Four KPIs | Accepted | 2026-08-28 | 2026-09-14 |
| [0003](0003-turbo-on-a-weighted-kpi-score.md) | TuRBO on a weighted KPI score | Accepted | 2026-09-13 | 2026-09-14 |
| [0006](0006-radio-coverage-objective.md) | A radio coverage objective | Superseded by 0007 | 2026-09-17 | 2026-09-17 |
| [0007](0007-demand-weighted-objective.md) | A demand-weighted objective, a load ceiling, and the reported KPI set | Accepted | 2026-09-18 | — |

0002, 0004 and 0005 are deleted; the numbers are not reused.
