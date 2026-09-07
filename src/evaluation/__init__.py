"""Read optimization runs back, compare them, and visualize what changed.

Everything here works from artifacts on disk — the run directories under
``outputs/optim/`` and the baseline ``radio_map.npz``. Nothing re-solves, so
this package imports neither Sionna-RT nor ``src.optim.evaluator``, and a
comparison costs seconds on a machine with no GPU.

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
    Writing tables and figures to ``reports/``.

Demand-weighted numbers are diagnostics, not objectives
-------------------------------------------------------
This package reports coverage weighted by where UEs actually stand, alongside
the tile-weighted rates. On the committed scenario those disagree sharply — the
tile-based hole rate is several times the UE-weighted one, because most holes
fall where nobody is.

That is a reporting view and nothing more. The objective remains exactly the
five KPIs of docs/adr/0001-five-kpis-under-lexicographic-priority.md: no
optimizer sees a demand-weighted quantity, none of them joins or replaces a KPI,
and adding one would supersede that record and invalidate every comparison made
before it. The thresholds separating hole, weak and good are read from
``cfg.kpi``, never restated here, so a diagnostic and its KPI always cut the
map at the same dBm.
"""
