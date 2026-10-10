"""Read optimization runs back, compare them, and visualize what changed.

The comparison reads the run directories under ``outputs/optim/``. No radio map
is stored, so :mod:`run` re-traces the few configurations it maps through
:class:`src.optim.evaluator.Evaluator`, which needs a CUDA GPU; everything over
the candidates themselves reads ``history.parquet`` alone.

The evaluation's KPIs are coverage rate, separation rate and median estimated
throughput (:data:`compare.EVALUATION_KPIS`), not the objectives the search maximised.
Every UE counts here, as it did in the search.

Modules, each with one reason to change:

``runs``
    Finding run directories, loading them, and refusing to compare ones that
    are not comparable.
``maps``
    Spatial reductions over a radio map: coverage classes, the demand raster,
    what changed between two maps.
``compare``
    The tables: configuration against configuration, method against method,
    the Pareto front and its tilts.
``plots``
    The figures, following the conventions ``notebooks/01_eda.ipynb``
    established.
``export``
    Writing tables to ``reports/``.
``run``
    Every table and figure notebook 04 presents; the ``task evaluate`` entry point.

Two views of the same cut. This package reports coverage weighted by where UEs
actually stand, alongside the tile-weighted rates. The two can disagree sharply,
because a hole need not fall where anyone stands, and that difference is why both
are printed.

Neither is an objective. The coverage and separation objectives
(:mod:`src.optim.objective`) are tile-uniform and read no demand at all, and
the throughput objective counts UE reports, not holes under them, so the
UE-weighted view here answers a question they do not ask - which is why it is
printed rather than optimised. The thresholds
separating hole, weak and good are read from ``cfg.kpi``, never restated here,
so a diagnostic and its KPI always cut the map at the same dBm.
"""
