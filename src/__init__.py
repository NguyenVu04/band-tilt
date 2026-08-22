"""Importable, testable project logic.

Layout
------
- ``src.data``       model-agnostic loading, cleaning, schema checks, splitting
- ``src.models``     one module per model, each owning its own preprocessing
- ``src.evaluation`` shared metric definitions and error analysis
- ``src.utils``      seeding, artifact IO, experiment tracking

Rules
-----
- Notebooks import from ``src``; ``src`` never imports from notebooks.
- ``app`` imports from ``src``; ``src`` never imports from ``app``.
- Anything reused by more than one notebook belongs here, not in a cell.

See PROJECT.md for the full set of conventions.
"""
