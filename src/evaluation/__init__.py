"""Shared evaluation.

- ``metrics``  the single definition of each metric, used by every model
- ``analysis`` error analysis, calibration, feature importance, comparison plots

Metrics are defined here and nowhere else. A metric reimplemented inside a
model module will differ in some detail — averaging, handling of edge cases,
sample weighting — and the comparison table in notebook 03 will silently
compare measurement choices instead of models.
"""
