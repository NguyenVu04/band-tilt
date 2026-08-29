"""The five KPIs — the only definition of the objective in this project.

Modules
-------
- ``serving``       strongest signal, serving cell, dominant cell-band
- ``coverage``      hole rate, weak rate, overlap rate, mean overlap neighbours
- ``band_priority`` the UE-weighted Band Priority Score
- ``vector``        assemble K, normalise it, and order two candidates

Why this is a package and not a helper
--------------------------------------
Four different consumers score configurations: Bayesian Optimization, MARL, the
surrogate's training labels, and the final Sionna-RT validation. BO and MARL
must be compared under identical KPI definitions, and the surrogate must be
checked against ray-traced maps (docs/adr/0001). None
of that means anything if two of those consumers compute a KPI slightly
differently.

So the definitions live here once, they read their thresholds from
``configs/kpi.yaml``, and no call site re-derives one. A hard-coded ``-120`` in
an optimizer is a bug even when it happens to match the config.

The input contract
------------------
Every function here takes an RSRP array of shape ``(n_cell_bands, |G|)`` in
dBm, with ``-inf`` where there is no measurable signal. Nothing in this package
imports Sionna-RT, or ``src.radio`` at all — which is exactly what lets the same
code score a ray-traced map, a surrogate prediction, and a hand-built test
fixture with known answers.

Optimization directions are fixed by ``cfg.kpi.directions``: hole rate, overlap
rate, weak rate and mean overlap neighbours are minimised; the Band Priority
Score is maximised. ``vector`` is the only place that ordering is encoded.
"""
