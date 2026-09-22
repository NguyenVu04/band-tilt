"""Read optimization runs back, compare them, and visualize what changed.

Everything here works from artifacts on disk — the run directories under
``outputs/optim/`` and the baseline ``radio_map.npz``. Nothing re-solves, so
this package imports neither Sionna-RT nor ``src.optim.evaluator``, and a
comparison costs seconds on a machine with no GPU.

Every UE counts here, as it did in the search.

Modules, each with one reason to change:

``runs``
    Finding run directories, loading them, and refusing to compare ones that
    are not comparable.
``maps``
    Spatial reductions over a radio map: coverage classes, the demand raster,
    what changed between two maps.
``compare``
    The tables: before against after, method against method, how far antennas
    moved.
``plots``
    The figures, following the conventions ``notebooks/01_eda.ipynb``
    established.
``export``
    Writing tables to ``reports/``.
``run``
    Every table and figure notebook 04 presents; the ``task evaluate`` entry point.

Two views of the same cut:
This package reports coverage weighted by where UEs actually stand, alongside
the tile-weighted rates. The two can disagree sharply, because a hole need not
fall where anyone stands, and that difference is why both are printed.

Neither is the objective. The search maximises the tile-uniform effective
coverage of docs/adr/0010-monotone-strength-aware-objective.md, which reads no
demand at all, so the UE-weighted view here answers a question no score asks -
which is why it is printed rather than optimised. The thresholds
separating hole, weak and good are read from ``cfg.kpi``, never restated here,
so a diagnostic and its KPI always cut the map at the same dBm.
"""
