# 4. Square-with-centre layout, the 3GPP calibration tilt as incumbent, and a −110 dBm hole threshold

- **Status:** Proposed
- **Date:** 2026-10-09
- **Deciders:** Nguyễn Duy Vũ
- **Supersedes:** —
- **Superseded by:** —
- **Rewritten:** 2026-10-10, at the maintainer's direction: decision 1 changed
  from the seven-node UMa hexagon (ISD 500 m) to five nodes on a square and its
  centre at 1732 m, and decisions 5 and 6 were added. The earlier text is in Git
  history.
- **Amended:** 2026-10-10, at the maintainer's direction: decision 5 changed from
  storing no radio map to storing the baseline map with its scene, because GPU
  ray tracing is not bit-reproducible.

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

The study then moved from one recommended configuration per method, read on
eleven KPIs, to a comparison an engineer can act on: the trade-off between
coverage, co-band interference and capacity, and the configurations that span it.

## Decision

1. **Layout: five nodes, a square and its centre.** Four corner nodes of an
   axis-aligned square and one at its centre, each corner 1732 m from the centre
   (`scenario.layout.node_spacing_m`, `src/scenario/layout.py`
   `NODE_OFFSETS`), chosen by the maintainer. Three sectors per node, azimuth
   offset 0°, masts 25 m, the UMa base-station height of 3GPP TR 38.901 (Rel-19)
   Table 7.2-1. The other geometries are removed.
2. **Incumbent: the UMa calibration tilt.** 12° on every sector-band: TR 38.901
   Table 7.8-1 sets the UMa electrical downtilt to 102°, with 90° the
   horizontal (Section 7.3.1). Bounds stay [0°, 20°].
3. **Hole threshold −110 dBm** (`kpi.hole_dbm`). 3GPP fixes no value: TS 37.320
   Annex A defines a coverage hole by the signal level needed for basic
   service without quantifying it, so the threshold is a project choice.
4. **Three demand hotspots** (`scenario.density.n_hotspots`).
5. **The baseline radio map is stored with its scene.** `task simulation:radio`
   traces the map at the sector table's tilts once and writes
   `simulation.output.radio_map_file`, `data/scenes/<scene_name>/radio_map.npz`
   (`src.simulation.radio.write`). Preprocessing, the notebooks, every search's
   incumbent and the evaluation read it (`src.simulation.radio.load`,
   `src.optim.evaluator.Evaluator`), which refuses a map solved under other
   settings, tilts or seed. Every other configuration is re-traced; no
   candidate's map is archived.
6. **The evaluation reads three KPIs.** Coverage rate, separation rate and
   median estimated throughput,
   with their per-band readings recorded for every evaluation. Each method's
   configuration is its largest hypervolume contribution on the three, and the
   combined Pareto front of both methods is published with its tilts
   (`reports/outputs/pareto_tilts.csv`). The search objectives of ADR 0003 are
   unchanged.

## Consequences

**Positive**

- Every scenario parameter except the hole threshold, the node spacing and the
  traffic model cites a 3GPP table, and the incumbent is a 3GPP value rather
  than an arbitrary one.
- Five three-sector nodes on three bands give 45 decision variables, against 63
  on the hexagon, at the same budget of 73 evaluations.
- An engineer chooses from a front instead of being handed one configuration,
  and the per-band KPIs say which layer a trade-off comes from.

**Negative**

- Every result produced before this change is incomparable with those after
  it: the scenario, and therefore the scenario id, changed.
- The 1732 m spacing has no 3GPP source; it is wider than the UMa ISD, so the
  absolute hole rate partly measures reach rather than tilt.
- The evaluation needs a CUDA GPU, and GPU ray tracing is not bit-reproducible,
  so a re-traced candidate can differ from the search's own reading of it in the
  trailing digits. The incumbent cannot: it is read, not re-traced.
- The search maximises the objectives of ADR 0003 while the evaluation reads
  three related but different KPIs, so a method can lead on one set and not
  the other.
- 3GPP specifies an electrical downtilt; `src/simulation/transmitter.py`
  applies tilt as the array's mechanical pitch, which also tilts the back and
  side lobes.
- The 8-row array with uniform weights has its first vertical null about 14.5°
  off boresight. At uniform tilts of roughly 14° to 18° that null sweeps the
  cell edge and every band's KPIs spike together; results in that range are an
  artefact of the antenna model, not of the network.

## Alternatives considered

**Keep SMa and the triangle, raise only the threshold.** At −110 dBm the SMa
triangle also reached a ten-point hole swing. Rejected because the 3GPP SMa
calibration tilt (5°, TR 38.901 Table 7.8-1A) sits among the best uniform
tilts, so no sound incumbent left the search measurable headroom, while the
UMa calibration tilt does.

**Keep −120 dBm and change only the geometry.** Rejected: at −120 dBm the share
of tiles that tilt turns into holes is capped by how far the antenna sidelobes
reach, whatever the node arrangement.

**The UMa hexagon, seven nodes at ISD 500 m.** The layout this record first
chose. Replaced at the maintainer's direction by the square with a centre node.

**Store no radio map and re-trace the baseline wherever it is needed.** The
first text of decision 5. Rejected on amendment: each re-trace of the incumbent
carried its own GPU noise, so the searches and the evaluation did not score the
same reference configuration.

**Archive every candidate's map.** Rejected: every consumer can re-trace a
candidate in seconds, and the front an engineer chooses from is in the run
histories.
