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
cost. PROJECT.md section 21 lists the confirmed decisions in a table; most rows
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

### Revision in place — the exception, not the practice

0002 and 0003 were **revised in place** on 2026-08-28 to follow the PROJECT.md
rewrite, at the maintainer's direction, rather than superseded by new records.
That is a departure from the rule above and is recorded as such: each carries a
`Revised` line in its header and a *Revision note* section stating exactly what
changed and why. The pre-revision text is in Git history at `abcdf6c`.

Prefer superseding. Revision in place loses the shape of the original argument,
which is the thing these records exist to preserve.

## Index

| # | Title | Status | Date | Revised |
|---|---|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-28 | — |
| [0002](0002-five-kpis-under-lexicographic-priority.md) | Five KPIs under lexicographic priority | Accepted | 2026-08-28 | 2026-08-28 |
| [0003](0003-sionna-rt-is-ground-truth.md) | Sionna-RT is ground truth; the surrogate only accelerates | Accepted | 2026-08-28 | 2026-08-28 |

### Records referenced but never written

`configs/`, `src/`, `tests/` and `pyproject.toml` cited five records that were
never written. Those 28 citations now point at the PROJECT.md section that
actually carries each decision, so nothing dangles; the records themselves are
still owed and are listed in the README.md roadmap.

| Would-be # | Decision it was cited for | Cited as, now |
|---|---|---|
| 0001 | Absolute tilt is the decision variable; the offset is derived | PROJECT.md Decision 1 |
| 0004 | Coordinate and angle conventions | PROJECT.md section 22.2 |
| 0005 | Band-generic design and the multi-band cell configuration contract | PROJECT.md section 8 |
| 0006 | Choosing TorchRL over Ray/RLlib, and the optimizer frameworks generally | PROJECT.md sections 13 and 14 |
| 0007 | The split scheme — now scenario-level | PROJECT.md section 12.3 |

0004 is the one worth writing soonest. The PROJECT.md rewrite **dropped** the
coordinate-and-angle-conventions section entirely, so `src/radio/geometry.py` is
now the only place the convention is stated anywhere in the project. A sign error
there produces a plausible, entirely wrong radio map and no test that does not
already know the answer can catch it.
