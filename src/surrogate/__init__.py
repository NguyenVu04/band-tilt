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
hundreds. The surrogate closes that gap.

It predicts the radio map, not the KPIs
---------------------------------------
``f_sur(x, tilt) -> R_hat``. :mod:`src.kpi`
then derives the five KPIs from the predicted map exactly as it does from a
ray-traced one, so there is a single KPI implementation sitting downstream of
both.

Regressing the KPIs directly — which this module previously did — creates two
code paths for the same definition, which is the failure a single KPI
evaluator exists to prevent. It also welds the trained model to the current thresholds: move
the hole threshold off -120 dBm and a KPI regressor is invalid, while a map
predictor is untouched.

The cost is that the output is a ``(cell_band, grid_cell)`` tensor rather than
five numbers, so training is heavier and a small per-pixel error can still move
a KPI when it lands near a threshold.
"""
