# 4. Angle convention fixed in one function

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

Two angle conventions meet in this project and they disagree on both axes.

*The radio convention*, used by the cell configuration and the MDT export.
Azimuth is a compass bearing: measured clockwise from north. Tilt is a downtilt:
a positive number points the beam at the ground.

*The simulation convention*, used by Sionna-RT. Yaw is measured
counter-clockwise from the x-axis. Pitch is a rotation in which a positive value
raises the beam.

The conversion, fixed by PROJECT.md section 5, is:

    yaw   = deg2rad(90.0 - azimuth)
    pitch = deg2rad(-tilt)

Both lines invert something. The azimuth conversion flips the direction of
rotation and shifts the origin by 90°; the tilt conversion flips the sign
outright, so a positive radio downtilt becomes a *negative* pitch.

The conversion is needed in scene construction, in every radio-map evaluation, in
any validation that compares simulated against measured RSRP, and in any
visualisation that draws antenna bearings. It is two lines of arithmetic, so the
natural thing is to write it wherever it is needed.

That is the trap, and the reason is specific rather than general tidiness. **A
sign error here does not raise.** It produces a complete, plausible radio map —
every beam pointing at the sky instead of the ground. RSRP values are finite and
in a believable range. The KPIs compute normally and are internally consistent.
The surrogate trains happily on them. The optimizer converges. Every number in
the final report is well-formed, and every one of them describes a network that
does not exist.

Nothing downstream can detect it, because there is no downstream check that
distinguishes "beams pointing down" from "beams pointing up" — coverage exists in
both cases, just in the wrong places. The only signal is a comparison against
measured MDT, which is exactly the check most likely to be deferred.

## Decision

The conversion exists **exactly once**, in `src/radio/geometry.py`:

- `absolute_tilt(etilt, mtilt)` — the definition `tilt = eTilt + mTilt`
- `azimuth_to_yaw(azimuth_deg)` — the bearing conversion
- `tilt_to_pitch(tilt_deg)` — the sign flip
- `orientations(cell_bands, theta)` — the assembled `(yaw, pitch, roll)` array

No other module may write `90.0 -` or `-tilt` against an angle. Every consumer —
`src/radio/scene.py`, `src/radio/radiomap.py`, the notebooks, any plotting code —
calls these functions.

The functions are unit-tested against hand-computed values in
`tests/test_geometry.py`, including cardinal bearings that pin down the direction
of rotation, and an explicit assertion that a positive downtilt yields a negative
pitch.

Notebook 02 asserts the same property on the real cell configuration before
spending a ray-tracing solve.

## Consequences

**Positive**

- There is one place to check, and one place a reviewer has to read carefully.
- The convention is testable in milliseconds without a simulator, a GPU, or a
  scene load — so the test actually gets run.
- Adding a consumer costs a function call rather than a re-derivation, which is
  where the second, subtly different copy would come from.
- The assertion in notebook 02 catches a configuration-level error before it
  costs simulation time.

**Negative**

- A trivial-looking function call in place of two lines of arithmetic invites a
  future contributor to inline it "for clarity", which reintroduces the problem.
  The module docstring exists to argue against exactly that.
- `src/radio/geometry.py` is imported almost everywhere in the radio path, so it
  is a wide dependency for a very small amount of code.

**Neutral**

- The functions deliberately do not wrap yaw into a fixed interval. Wrapping
  makes two mathematically equal orientations compare unequal and breaks the
  round-trip test; a caller that needs a wrapped angle wraps it.
- If Sionna-RT changes its convention in a future major version, the change is
  one module and one test file.

## Alternatives considered

**Convert at each call site.** Two lines, no indirection, and obvious in context.
Rejected because the cost of getting it wrong is uncapped and undetectable, and
the probability rises with every copy — a second call site written from memory
rather than from the spec is how the two copies come to differ.

**Convert once at load time, storing yaw and pitch on the cell-band table.** No
conversion in the hot path, and the table becomes self-describing. Rejected
because tilt is the optimization variable: it changes on every evaluation, so
pitch would have to be recomputed per configuration anyway, and a stale stored
pitch alongside a fresh theta is a worse failure than the one being avoided.

**Rely on a visual check of the radio map instead of a unit test.** A rendered
coverage map does look wrong when the beams point up. Rejected because it
requires a scene load and human judgement, so it happens once at the start and
never again in CI — and the failure it catches is one that could be introduced by
any later edit.

**Store the convention in `configs/` as a sign parameter.** Makes it explicit and
adjustable if the export convention ever changes. Rejected because it turns a
fact about two fixed conventions into a tunable, and a config with a wrong sign
produces the same silent failure with an extra layer of indirection in front of
it.
