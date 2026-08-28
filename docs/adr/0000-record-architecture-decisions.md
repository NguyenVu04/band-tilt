# 0. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

This project makes decisions that shape it for years: which quantity is
optimised, what "better coverage" means numerically, which approximations are
allowed to stand behind a reported result. Today that reasoning lives in
PROJECT.md, in chat history, and in the memory of whoever was in the room.

PROJECT.md is a specification: it states *what* the system does. It is also
revised as the research progresses, so it records the current position rather
than the history — read it in a year and you cannot tell which of its statements
were obvious, which were contested, and which were chosen against a real
alternative that has since become more attractive.

The code has the same problem in sharper form. `pitch = deg2rad(-tilt)` records
a convention and never the fact that the opposite sign produces a plausible,
entirely wrong radio map. A future change either repeats an experiment already
run, or removes a constraint that existed for a reason nobody can now state.

## Decision

We record architecturally significant decisions as ADRs in `docs/adr/`, one file
per decision, numbered sequentially and written at the time the decision is made.
Records are immutable once accepted: a decision that no longer holds is
superseded by a new record rather than edited.

PROJECT.md remains the specification and stays current. The ADRs are the
history, and where they overlap, an ADR explains *why* the specification says
what it says.

The criteria for significance, and the process, are in [README.md](README.md).

## Consequences

**Positive**

- The reasoning behind the formulation is discoverable from the repository.
- A change to the KPI definitions, the tilt bounds or the angle convention is
  visibly a change to a recorded decision, not a routine edit.
- New contributors can read the history rather than reconstruct it from
  PROJECT.md revisions.

**Negative**

- Every significant change costs an extra document and an extra review cycle.
- Judgement is required about what counts as significant; the boundary will be
  argued about.
- An index that nobody maintains rots, and a rotted index is worse than none.

**Neutral**

- Records accumulate and are never deleted. Superseded records stay as history.
- Some duplication with PROJECT.md is expected and accepted: the specification
  states the decision, the ADR states the alternatives and the cost.

## Alternatives considered

**Keep the reasoning in PROJECT.md alone.** One document, already the source of
truth, and section 21 already lists the confirmed decisions in a table. Rejected
because that table records conclusions without alternatives or costs — it says
absolute tilt is the optimization variable, not what using tilt offset would have
cost. It is also revised in place, so it cannot carry history.

**Keep the reasoning in commit messages and pull request threads.** Nothing extra
to maintain, and it is already where the discussion happens. Rejected because it
is not discoverable: finding why a decision was made requires knowing which
change made it, which is exactly what the reader does not know.

**Keep a design document in a wiki or a shared drive.** Better for long-form
design, and easier to write in. Rejected because it drifts from the code — it is
not reviewed with the change, so it is accurate only until the first thing that
contradicts it merges.

**Record nothing; rely on the code and PROJECT.md.** Zero cost, and works well
enough for a single-author project. Rejected because it fails precisely when it
matters most: at handover, when a result is disputed, and when someone revisits
the formulation after the multi-band data arrives.
