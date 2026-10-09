# 4. UMa hexagon layout, the 3GPP calibration tilt as incumbent, and a −110 dBm hole threshold

- **Status:** Proposed
- **Date:** 2026-10-09
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —

## Context

The study ran on four SMa sites (ISD 1299 m, 35 m masts) on a triangle and its
centroid, every sector-band at a uniform 10° incumbent, with a hole at or below
−120 dBm. That setup could not show what the search is worth:

- At −120 dBm the 700 MHz layer alone covered nearly the whole grid, so the
  union of bands left little for tilt to fix. Uniform tilt from 0° to 20° moved
  the hole rate by about 3 percentage points, and every candidate the searches
  evaluated stayed within about 1.4 points of each other.
- The 10° incumbent had no source. It was neither a 3GPP value nor the best
  uniform tilt, so a gain over it could be read as a gain over a strawman.
- The holes that tilt did create lay at the scene periphery, never between
  sites: with four sites the inter-site gaps were too small to matter.

Sweeps of uniform tilt over 3GPP-anchored layouts and over the hole threshold
showed that the threshold, the mast height and the node count, not the layout
shape alone, decide how much hold tilt has on holes.

## Decision

1. **Layout: UMa on a hexagon.** ISD 500 m and BS height 25 m, the UMa
   evaluation parameters of 3GPP TR 38.901 (Rel-19) Table 7.2-1. Seven nodes,
   a centre site and its first tier, with `scenario.layout.node_spacing_m` the
   nearest-neighbour inter-site distance in every geometry
   (`src/scenario/layout.py`, `SHAPES`). Sector azimuth offset 0°.
2. **Incumbent: the UMa calibration tilt.** 12° on every sector-band: TR 38.901
   Table 7.8-1 sets the UMa electrical downtilt to 102°, with 90° the
   horizontal (Section 7.3.1). Bounds stay [0°, 20°].
3. **Hole threshold −110 dBm** (`kpi.hole_dbm`). 3GPP fixes no value: TS 37.320
   Annex A defines a coverage hole by the signal level needed for basic
   service without quantifying it, so the threshold is a project choice.
4. **Three demand hotspots** (`scenario.density.n_hotspots`).

## Consequences

**Positive**

- Every scenario parameter except the hole threshold and the traffic model
  cites a 3GPP table, and the incumbent is a 3GPP value rather than an
  arbitrary one.
- Tilt now has hold on both reported failure modes. Uniform tilt from 0° to 20°
  moves the hole rate by over ten percentage points and the overlap rate the
  other way, so a coordinated search has a real trade-off to resolve.

**Negative**

- Every result produced before this change is incomparable with those after
  it: the KPI threshold, the scenario and therefore the scenario id all
  changed.
- Seven three-sector nodes on three bands give 63 decision variables, against
  36 before, at an unchanged budget of 73 evaluations. Searches end while still
  improving.
- 3GPP specifies an electrical downtilt; `src/simulation/transmitter.py`
  applies tilt as the array's mechanical pitch, which also tilts the back and
  side lobes.
- The 8-row array with uniform weights has its first vertical null about 14.5°
  off boresight. At uniform tilts of roughly 14° to 18° that null sweeps the
  cell edge and every band's KPIs spike together; results in that range are an
  artefact of the antenna model, not of the network.
- Seven sites in a 6.2 × 6.5 km scene still leave tiles no tilt reaches, so the
  absolute hole rate overstates what tilt can fix.

**Neutral**

- `scenario.layout.shape` keeps the triangle, square and square-with-centre
  geometries available; they are not used by default.

## Alternatives considered

**Keep SMa and the triangle, raise only the threshold.** At −110 dBm the SMa
triangle also reached a ten-point hole swing. Rejected because the 3GPP SMa
calibration tilt (5°, TR 38.901 Table 7.8-1A) sits among the best uniform
tilts, so no sound incumbent left the search measurable headroom, while the
UMa calibration tilt does.

**Keep −120 dBm and change only the geometry.** Rejected: at −120 dBm the share
of tiles that tilt turns into holes is capped by how far the antenna sidelobes
reach, whatever the node arrangement.

**A square of four nodes at azimuth 0°.** It gave the largest uniform-tilt
swings, but with four nodes most holes lie beyond any site's reach, which makes
the hole rate a measure of reach rather than of tilt.
