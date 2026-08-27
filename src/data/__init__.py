"""Model-agnostic data handling — the code behind notebooks 00 and 01.

Modules
-------
- ``load``       read raw MDT and cell configuration, and the cleaned outputs
- ``schema``     enforce the contract declared in ``configs/data.yaml``
- ``clean``      deduplication, invalid-record filtering, coordinate validation
- ``split``      the single authoritative leakage-safe splitter
- ``ue_density`` grid the target area and count UE observations per grid cell

Nothing in this package may learn from the data. Any transform fitted on
observations — a scaler, an imputer, a learned encoder — belongs in
``src.surrogate.features``, fitted on the training split only.

Nothing in this package may import ``src.radio``, ``src.kpi``,
``src.surrogate`` or ``src.optim``. Data handling sits below all of them.
"""
