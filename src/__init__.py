"""Importable, testable project logic.

Layout:
- ``src.core``        the shared domain types and file contracts: sector, UE table
- ``src.scenario``    the synthetic-data generator: UEs, sector layout, manifest
- ``src.simulation``  Sionna-RT scenes and ray-traced radio maps
- ``src.data``        verify the inputs and radio map, write the typed UE table
- ``src.kpi``         the reported KPIs and the serving rule
- ``src.optim``       the objectives, MORBO and random search, and the run it publishes
- ``src.evaluation``  compare finished runs, write tables and figures
- ``src.utils``       seeding and plotting
- ``src.config``      compose the Hydra config outside an entry point
- ``src.tracking``    log one stage as one MLflow run

Dependency direction:
::

    core        ->  nothing
    kpi         ->  core
    simulation  ->  core
    scenario    ->  core, simulation
    data        ->  core, simulation
    optim       ->  core, kpi, simulation
    evaluation  ->  core, kpi, optim, utils

There is no cycle. Add none.

Nothing imports ``src.scenario`` or reads ``cfg.scenario``: the generator only
writes the files ``simulation.input`` names, so real data can replace it
(``tests/test_architecture.py``).

``src.kpi`` deliberately does not import ``src.simulation``: it consumes an RSRP
array, not a simulator. That is what lets the same KPI code score a Sionna-RT
radio map and a hand-built test fixture alike.

``src.tracking`` is called only from ``@hydra.main`` entry points, never from
library code.

Rules:
- Notebooks import from ``src``; ``src`` never imports from notebooks.
- Anything reused by more than one notebook belongs here, not in a cell.

See README.md for the problem framing.
"""
