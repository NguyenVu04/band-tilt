"""Read the simulation artifacts, verify them, and write the processed tables.

The boundary this package holds is the one ``notebooks/01_eda.ipynb`` section 11
draws: everything here is deterministic and model-agnostic. Nothing is imputed,
scaled, encoded or fitted, and no row is dropped on a threshold derived from the
data. A bound is only enforced when its source can be named — the config, the
scenario manifest, or a standard.

The KPI thresholds are the standing example. ``-120`` dBm classifies a tile as a
coverage hole; it never removes a measurement. Holes are what the project
measures, not a defect in the data.
"""
