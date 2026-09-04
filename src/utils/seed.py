"""Seed the interpreter's global random streams.

Distinct from :mod:`src.simulation.seeds`, which derives one independent stream
per stage from ``simulation.seed``. No simulation stage reads a global
generator, so this covers only the ad-hoc draws a notebook or a model makes.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed the ``random`` and NumPy global generators.

    This does not make the simulation reproducible. Every stage draws from its
    own generator, seeded through :func:`src.simulation.seeds.stream`; change
    ``seed`` in the config to move those.
    """
    random.seed(seed)
    np.random.seed(seed)
