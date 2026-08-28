"""The fast radio-map predictor that stands in for Sionna-RT during optimization.

Modules
-------
- ``dataset``  build and load D_sur, the (features, tilt, radio map) training set
- ``features`` turn scenario features and a tilt configuration into a tensor
- ``model``    the predictor itself, f_sur
- ``train``    fit it, and report radio-map and derived-KPI error

Why this exists
---------------
A Sionna-RT solve over this scene is too expensive to put inside an optimization
loop. Both optimizers need thousands of evaluations; the simulator can supply
hundreds. The surrogate closes that gap (PROJECT.md section 11).

What it is not
--------------
The surrogate is an acceleration mechanism and never the reported result. Every
candidate that matters is re-evaluated with Sionna-RT before it is believed, and
the final numbers in any report come from that evaluation — not from a
prediction. PROJECT.md section 11.4 and 16 Phase 7, and docs/adr/0003.

It predicts the radio map, not the KPIs
---------------------------------------
``f_sur(x, tilt) -> R_hat`` (PROJECT.md section 11, Decision 6). :mod:`src.kpi`
then derives the five KPIs from the predicted map exactly as it does from a
ray-traced one, so there is a single KPI implementation sitting downstream of
both.

Regressing the KPIs directly — which this module previously did — creates two
code paths for the same definition, which is the failure PROJECT.md section 25.4
names outright. It also welds the trained model to the current thresholds: move
the hole threshold off -120 dBm and a KPI regressor is invalid, while a map
predictor is untouched.

The cost is that the output is a ``(cell_band, grid_cell)`` tensor rather than
five numbers, so training is heavier and a small per-pixel error can still move
a KPI when it lands near a threshold.
"""
