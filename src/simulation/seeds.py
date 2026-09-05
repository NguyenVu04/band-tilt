"""Named deterministic random streams derived from the configured seed."""

from __future__ import annotations

from omegaconf import DictConfig

# Append offsets only; changing one redraws existing scenarios.
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
