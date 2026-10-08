# Architecture decision records

An ADR captures one architecturally significant decision: what we chose, what
else we considered, and what it costs us. It is written when the decision is
made, while the alternatives are still fresh, and it is never rewritten
afterwards — a decision that turns out wrong gets a *new* record that supersedes
the old one.

The point is not process. It is that in two years someone will ask why a
coverage hole is defined at −120 dBm — and the answer will otherwise have left
with whoever made the call.

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

On 2026-09-19, again at the maintainer's direction, 0007 was rewritten in place
and retitled when the demand map moved from requested PRBs to MDT report counts,
the KDE was removed, and the objective was split into one term per band.

On 2026-09-22, again at the maintainer's direction, every superseded record was
deleted and the survivors renumbered. The objective records 0006, 0007, 0008,
0009 and 0010 were deleted, and so was 0011, which had superseded 0010's maximum
over bands with a contraharmonic mean. What was still in force from 0007 (the
admission ceiling, the reported KPI set, the capacity model) and from 0010 (the
per-band utility and `objective_version`) was folded, with 0011, into a new
0003. The former 0003 became 0002, and the headers of 0001 and 0002 were
amended to point at the new 0003. This reuses numbers, which departs from the
rule under *How*: a reference to "ADR 0003" written before this date means the
TuRBO record, now 0002.

On 2026-09-28, again at the maintainer's direction, 0001 (the four KPIs) was
deleted when the reported KPI set was replaced, and the survivors renumbered:
the TuRBO record 0002 became 0001, and the objective record 0003 became 0002,
rewritten in place with the new KPI set and without `objective_version`. A
reference to "ADR 0002" or "ADR 0003" written before this date means the record
now numbered one lower.

On 2026-09-30, again at the maintainer's direction, sections 2 to 4 of 0002 were
rewritten in place and the record retitled when the band-preference serving rule
and its admission gate were replaced by max-throughput sector selection over an
equal PRB share, and the estimated-throughput KPIs joined the set. Later the
same day, again at the maintainer's direction, section 1 of 0002 was rewritten
in place when the per-band utility `lambda e^(1 - lambda)` was replaced by the
strongest sector's share of the band's received power.

Prefer superseding. Rewriting loses the shape of the original argument, which
is the thing these records exist to preserve.

## Index

| # | Title | Status | Date | Rewritten |
|---|---|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-28 | — |
| [0001](0001-turbo-on-a-weighted-kpi-score.md) | TuRBO on a weighted KPI score | Superseded by 0003 | 2026-09-13 | 2026-09-14 |
| [0002](0002-contraharmonic-objective-and-kpi-set.md) | A contraharmonic, strength-aware objective, max-throughput sector selection, and the reported KPI set | Proposed; sections 1 and 3 superseded by 0003 | 2026-09-22 | 2026-10-03 |
| [0003](0003-three-objectives-and-morbo.md) | Coverage and separation searched by MORBO, recommended by hypervolume contribution | Accepted | 2026-10-07 | 2026-10-08 |

Every other record is deleted and lives in Git history. Numbers were reused
on 2026-09-22 and 2026-09-28; see above.
