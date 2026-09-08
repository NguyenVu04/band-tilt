"""Optimization over the absolute-tilt space.

One decision vector, one objective, three ways of searching it. Every method
here proposes a tilt for each cell-band pair, has it scored by the five
functions in :mod:`src.kpi`, and writes the same artifacts, so a comparison
between methods is a comparison of search strategies and nothing else.

Modules, each with one reason to change:

``space``
    The box an optimizer may move in, and the only place a vector becomes
    :class:`src.core.cell.Cell` objects.
``objective``
    The KPI vector, the sign convention, and the rule that picks one
    configuration out of a Pareto front.
``evaluator``
    The expensive path: a tilt vector ray-traced into a radio map. The only
    module here that touches Sionna-RT.
``history``
    The evaluation log, the artifacts a run leaves behind, and the deliverable
    the tilt change is republished as.
``methods``
    One folder per method — Ax multi-objective BO, Ax random search, and a
    rule-based per-band sweep — behind the ``SearchMethod`` interface in
    ``methods/base.py``, with the registry that dispatches on the selected
    ``optim/method`` config group.
``run``
    The ``python -m`` entry point that wires the above together.

A method depends on the :class:`~src.optim.evaluator.ObjectiveEvaluator`
protocol rather than on the concrete evaluator, which is what lets a search be
tested without a GPU and lets a future surrogate stand in for the ray tracer
without the search noticing.
"""
