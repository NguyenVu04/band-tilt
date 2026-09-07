"""Domain types shared across the pipeline.

The vocabulary every other package agrees on. A *node* is a mast; a *cell* is
one antenna on it, carrying one tilt per band; a *tile* is one square of the
measurement grid, and lives in :mod:`src.simulation.grid`.

``core`` imports nothing from ``src.simulation``, ``src.kpi`` or ``src.data``.
They all import it, which is what stops the same dataclass being restated in
three places and drifting.
"""
