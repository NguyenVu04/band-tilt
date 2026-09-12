"""Importable, testable project logic.

Layout
------
- ``src.core``        the shared domain types: cell and per-band tilt
- ``src.simulation``  scene, UE population, ray-traced radio maps, synthetic MDT
- ``src.data``        verify the simulation output and write the typed tables
- ``src.kpi``         the four KPIs — the only definition of the objective
- ``src.optim``       multi-objective BO and the baselines, and the run it publishes
- ``src.evaluation``  compare finished runs, write tables and figures
- ``src.utils``       seeding and plotting
- ``src.config``      compose the Hydra config outside an entry point
- ``src.tracking``    log one stage as one MLflow run

Dependency direction
--------------------
::

    core        ->  nothing
    kpi         ->  core
    simulation  ->  core, kpi
    data        ->  simulation
    optim       ->  core, kpi, simulation
    evaluation  ->  kpi, optim, utils

There is no cycle. Add none.

``src.kpi`` deliberately does not import ``src.simulation``: it consumes an RSRP
array, not a simulator. That is what lets the same KPI code score a Sionna-RT
radio map and a hand-built test fixture alike.

``src.tracking`` is called only from ``@hydra.main`` entry points, never from
library code.

Rules
-----
- Notebooks import from ``src``; ``src`` never imports from notebooks.
- Anything reused by more than one notebook belongs here, not in a cell.

See README.md for the problem framing and docs/adr/ for the decisions.
"""
