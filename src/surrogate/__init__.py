"""The fast KPI predictor that stands in for Sionna-RT during optimization.

Modules
-------
- ``dataset``  build and load D_sur, the (state, tilt, KPI) training set
- ``features`` turn a state and a tilt configuration into a feature tensor
- ``model``    the predictor itself, f_sur
- ``train``    fit it, and report per-KPI error

Why this exists
---------------
A Sionna-RT solve over this scene is too expensive to put inside an optimization
loop. Both optimizers need thousands of evaluations; the simulator can supply
hundreds. The surrogate closes that gap (PROJECT.md section 18).

What it is not
--------------
The surrogate is an acceleration mechanism and never the reported result. Every
candidate that matters is re-evaluated with Sionna-RT before it is believed, and
the final numbers in any report come from that evaluation — not from a
prediction. PROJECT.md sections 24 and 26, and docs/adr/0003.

It predicts the KPI vector, not a scalar
----------------------------------------
Scalarization weights are an optimizer choice and change between experiments,
while re-deriving a scalar from predicted KPIs costs nothing. Predicting the
scalar directly would tie every trained surrogate to one weighting and force a
retrain to answer a different question.
"""
