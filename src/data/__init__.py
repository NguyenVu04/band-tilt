"""Model-agnostic data handling — the code behind notebook 01.

Modules
-------
- ``load``   read raw data and the cleaned outputs of notebook 01
- ``schema`` enforce the contract declared in ``configs/data.yaml``
- ``clean``  deduplication, invalid-record filtering, column dropping
- ``split``  the single authoritative train/val/test splitter

Nothing in this package may learn from the data. Imputation, scaling,
encoding, feature engineering and statistical outlier removal all happen per
model in ``src.models``, fitted on the training split only.
"""
