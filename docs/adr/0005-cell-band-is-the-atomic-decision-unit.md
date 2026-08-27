# 5. Cell-band is the atomic decision unit

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

The project is named for multi-band tilt coordination. PROJECT.md optimises one
absolute tilt per `(cell, band)` pair, and the fifth KPI — the UE-weighted Band
Priority Score — exists specifically to reward high-priority bands dominating
where users are. Without more than one band per cell, that KPI is a constant and
the research question disappears.

The data on disk cannot express any of it. `data/raw/gcell_conf.csv` has:

    gcell_id, gnodeb_id, sim_x, sim_y, antenna_height, sync_date,
    digital_tilt, azimuth

26 cells across 10 gNodeBs, two or three sectors each — a single-band sectorised
deployment. There is no band column, no carrier frequency, no transmit power, and
no split between electrical and mechanical tilt, so `tilt = eTilt + mTilt`
(PROJECT.md section 4.2) cannot be evaluated from a single `digital_tilt`. There
are also no per-cell tilt bounds, and the bounds *are* the feasible set.

The multi-band export is expected but has not arrived. Three responses were
available: wait, fabricate a plausible multi-band overlay, or build for the target
shape now and connect the data when it lands.

A second question rides along with this one: whether the atomic unit is the cell
or the cell-band pair. Two carriers on one antenna share a position, a height and
an azimuth, and differ in frequency, power and tilt. Treating the cell as atomic
and the bands as attributes is one modelling choice; treating each pair as a
first-class row is another.

## Decision

**The `(cell, band)` pair is the atomic unit**, and the entire codebase is
written band-generic:

1. `src/radio/cell_band.build_table` produces an ordered table with one row per
   pair. **Its row order is the canonical ordering of every theta vector in the
   project** — the BO search space, a MARL action, a surrogate feature row, a
   reported result. The order is deterministic and explicitly sorted.
2. No module hardcodes the number of bands `B`, or assumes it is 1. Adding a band
   is an edit to `configs/radio.yaml`.
3. The band set, per-band carrier frequency, transmit power, priority weight
   `w_b` and tilt bounds `[theta_min, theta_max]` are **declared** in
   `configs/radio.yaml`, currently as `<placeholder>` values.
4. `configs/data.yaml` declares the multi-band columns the export will need —
   band, `eTilt`, `mTilt`, transmit power — marked as pending. `src/data/schema.py`
   skips any column whose declared name is still a placeholder, so the pipeline
   loads today's export without them.
5. The cell configuration is validated with `strict=False` until the multi-band
   export arrives. That is a temporary exception and is recorded as such.
6. **No synthetic band data is fabricated.** `cfg.radio.cells.from_cell_config`
   is `false` today, meaning bands come from the config declaration; it flips to
   `true` when the export carries a band column.

**The pipeline does not run end to end until the multi-band export lands.** That
is a stated, accepted consequence rather than a defect.

## Consequences

**Positive**

- The structure, configs, tests and documentation are all in place, so arrival of
  the data is a config edit plus a loader change, not a redesign.
- Nothing downstream needs to change when `B` grows: the search space, the
  surrogate features, the MARL action space and the report all derive their width
  from the table.
- Keeping the placeholders visible means the gap is documented in the repository
  rather than remembered.
- Refusing to fabricate bands means no result can be produced that silently rests
  on invented physics.

**Negative**

- **The pipeline cannot be run end to end today.** Notebooks 02 through 06 will
  not produce results, and the code past `src/data/` is therefore unexercised
  against real data.
- Placeholders in `configs/radio.yaml` and `configs/data.yaml` will reach numeric
  call sites if `src.config.validate_config` is not implemented and called. That
  check is the only thing standing between a placeholder and a KPI computed
  against the string `"<theta_min in degrees>"`.
- Validating the cell configuration non-strict weakens the contract for that
  input, and the weakening has to be remembered and removed.
- Carrying a band dimension of size one costs an axis in every array and a join
  in every table, for no benefit until the data arrives.
- With one band, the Band Priority Score is constant and the project's
  distinguishing KPI cannot be exercised or tested against real data.

**Neutral**

- The cell-band table is a small object rebuilt per run rather than persisted, so
  it always matches the current config.
- Its ordering has to be saved with any stored result. A theta vector is
  uninterpretable without it, and that is now a documented requirement in
  `src/surrogate/model.py` and `src/evaluation/report.py`.

## Alternatives considered

**Wait for the multi-band export before building anything.** No placeholders, no
unexercised code, no risk of designing for a shape the data does not have.
Rejected because the shape is already specified in PROJECT.md sections 3 and 4.2,
and the parts that do not depend on the band dimension — cleaning, splitting, the
KPI definitions, the angle convention, the search space — are buildable and
testable now.

**Synthesise a plausible multi-band overlay.** Assign each existing cell two or
three co-sited carriers with realistic frequencies and powers, so `B > 1`
experiments run today. Rejected because it would produce complete, publishable-
looking results grounded in invented data: the Band Priority Score would be
optimised against band placements nobody chose, and the number would look exactly
like a real one. If a synthetic overlay is later needed to exercise the code, it
belongs behind an explicit flag and every result from it must be labelled
synthetic — which supersedes this record.

**Treat the cell as atomic, with bands as attributes.** Fewer rows, and it matches
the current export shape directly. Rejected because the decision variable is one
tilt *per band*: a cell-atomic table would need a nested per-band tilt vector,
and every consumer would have to flatten it — reintroducing the ordering problem
in several places instead of solving it once.

**Hardcode a single band and generalise later.** Simplest possible thing that
works with today's data. Rejected because "generalise later" means touching the
search space, the surrogate features, the MARL action space, the KPI code and the
report — and the Band Priority Score would have to be written from scratch, since
it has no meaning at `B = 1`.
