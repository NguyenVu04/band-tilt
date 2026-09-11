"""Named deterministic random streams derived from the configured seed."""

from __future__ import annotations

from omegaconf import DictConfig

# Append offsets only; changing one redraws existing scenarios.
_OFFSET = {
    "scene": 0,  # the grid raster
    "density": 1,  # hotspot centres and per-tile weights
    "sample": 2,  # UE positions
    "mdt": 4,  # measurement error
    "traffic": 5,  # per-interval counts and mixture masses
    "solver": 6,  # the ray tracer's own Monte-Carlo stream
    "mdt_sinr": 7,  # SINR measurement error
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
