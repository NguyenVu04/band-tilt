"""Importable, testable project logic.

Layout
------
- ``src.data``       load, validate and split synthetic MDT and cell configuration
- ``src.radio``      scene construction, radio-map generation, tilt geometry
- ``src.kpi``        the five KPIs — the only definition of the objective
- ``src.surrogate``  the fast radio-map predictor that stands in for Sionna-RT
- ``src.optim``      TuRBO and MARL over the same search space
- ``src.evaluation`` Sionna-RT validation, method comparison, reporting
- ``src.utils``      seeding, artifact IO, experiment tracking, plotting

Dependency direction
--------------------
These are one-way. An import in the reverse direction is a bug, not a
shortcut::

    data, radio  ->  utils, config
    kpi          ->  utils
    surrogate    ->  kpi, radio, data
    optim        ->  surrogate, kpi, radio
    evaluation   ->  everything above

``src.kpi`` deliberately does not import ``src.radio``: it consumes an RSRP
array, not a simulator. That is what lets the same KPI code score a Sionna-RT
radio map, a surrogate prediction and a hand-built test fixture.

Rules
-----
- Notebooks import from ``src``; ``src`` never imports from notebooks.
- ``app`` imports from ``src``; ``src`` never imports from ``app``.
- Anything reused by more than one notebook belongs here, not in a cell.

Implementation status
---------------------
Almost every function in this package raises :class:`NotImplementedError` with
its own dotted path. That is the intended state, not a defect: the contracts,
docstrings and configs are written first so that the structure of the problem
is settled before any method body is. Fill them in deliberately, one module at
a time — do not treat a raise as a bug to be silenced.

See PROJECT.md for the problem formulation and docs/adr/ for the decisions.
"""
