"""Cross-cutting helpers.

- ``seed``     one place that sets every random source
- ``io``       artifact serialisation (always preprocessor + model together)
- ``tracking`` a thin MLflow wrapper, so the tracker can be swapped in one file

These modules are imported by ``src.data``, ``src.models`` and ``app``, so they
must not import from any of them — keep this package dependency-free within the
project.
"""
