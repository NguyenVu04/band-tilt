"""Named deterministic random streams derived from the configured seed."""

from __future__ import annotations

import hashlib

from omegaconf import DictConfig

# Adding a name is safe: each stream hashes its own name, so a new one cannot
# shift an existing one.
_STREAMS = frozenset(
    {
        "scene",  # the grid raster
        "density",  # hotspot centres and per-tile weights
        "sample",  # UE positions
        "traffic",  # per-interval counts and mixture masses
        "solver",  # the ray tracer's own Monte-Carlo stream
    }
)

# Sionna's solver takes a uint32 seed, so every stream stays in that range.
_MODULUS = 2**32


def stream(cfg: DictConfig, name: str) -> int:
    """Derive the seed for one named stream.

    Hashed from ``(seed, name)`` rather than offset from the seed: additive
    offsets alias across a seed sweep, so ``seed=1``'s density stream and
    ``seed=2``'s sample stream would draw the same numbers in runs the sweep
    treats as independent replicates.

    Returns:
        A seed in ``[0, 2**32)``, stable across processes and platforms.

    Raises:
        KeyError: When ``name`` is not a declared stream, which means a caller
            invented one instead of adding it here.
    """
    if name not in _STREAMS:
        raise KeyError(f"unknown random stream {name!r}; declared streams are {sorted(_STREAMS)}")
    digest = hashlib.sha256(f"{int(cfg.simulation.seed)}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _MODULUS
