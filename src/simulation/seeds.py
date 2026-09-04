"""Named random streams, all derived from the one configured seed.

Every stochastic step in the pipeline draws from its own stream so that
changing one leaves the others untouched: raising the UE count must not move
the buildings, and re-solving a band must not redraw the materials.

The streams are offsets of ``simulation.seed`` rather than independent seeds,
so the relationship between them is itself reproducible. NumPy runs an integer
seed through a ``SeedSequence``, which separates consecutive values well.

Naming them here rather than writing ``seed + 4`` at each call site is what
stops two stages sharing an offset by accident, and what makes adding a stream
a one-line change instead of an audit.
"""

from __future__ import annotations

from omegaconf import DictConfig

# Offsets are append-only. Changing one silently redraws that stream for every
# existing scenario while leaving the config -- and so the scenario id -- alone.
_OFFSET = {
    "scene": 0,  # the grid raster and the building perturbation
    "density": 1,  # hotspot centres and per-cell weights
    "sample": 2,  # UE positions
    "materials": 3,  # the radio-material draw, shared across bands by design
    "mdt": 4,  # measurement error and censoring
    "traffic": 5,  # per-interval counts and mixture masses
    "solver": 6,  # the ray tracer's own Monte-Carlo stream
}


def stream(cfg: DictConfig, name: str) -> int:
    """The seed for one named stream.

    Raises:
        KeyError: When ``name`` is not a declared stream, which means a caller
            invented one instead of adding it here.
    """
    if name not in _OFFSET:
        raise KeyError(f"unknown random stream {name!r}; declared streams are {sorted(_OFFSET)}")
    return int(cfg.simulation.seed) + _OFFSET[name]
