"""Importable, testable project logic.

Layout
------
- ``src.core``        the shared domain types: cell and per-band tilt
- ``src.simulation``  scene, UE population, ray-traced radio maps, synthetic MDT
- ``src.data``        verify the simulation output and write the typed tables
- ``src.kpi``         the four KPIs — the only definition of the objective
- ``src.surrogate``   the radio-map predictor that stands in for Sionna-RT
- ``src.optim``       multi-objective BO and the baselines, search and report
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
    optim       ->  core, kpi, simulation, surrogate
    surrogate   ->  core, optim, simulation
    evaluation  ->  kpi, optim, utils

``optim`` and ``surrogate`` import each other: the surrogate evaluator satisfies
the optimizer's evaluator protocol, and ``src.optim.run`` imports it lazily.
Add no new cycle.

``src.kpi`` deliberately does not import ``src.simulation``: it consumes an RSRP
array, not a simulator. That is what lets the same KPI code score a Sionna-RT
radio map, a surrogate prediction and a hand-built test fixture.

``src.tracking`` is called only from ``@hydra.main`` entry points, never from
library code.

Rules
-----
- Notebooks import from ``src``; ``src`` never imports from notebooks.
- Anything reused by more than one notebook belongs here, not in a cell.

See README.md for the problem framing and docs/adr/ for the decisions.
"""
