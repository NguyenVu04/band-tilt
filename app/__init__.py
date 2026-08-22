"""Serving and demonstration layer.

Kept separate from ``src`` for two reasons: ``src`` stays pure ML logic, and the
web dependencies (FastAPI, Streamlit) stay out of the training environment.

Dependency direction is one-way: ``app`` imports from ``src``, never the
reverse. If serving needs something from the ML core, add it to ``src`` and
import it here.

Install with ``uv sync --extra app``.
"""
