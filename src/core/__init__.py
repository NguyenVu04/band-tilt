"""Domain types shared across the pipeline.

The vocabulary every other package agrees on. A *node* is a mast; a *sector* is
one antenna on it, carrying one tilt per band; a *tile* is one square of the
manifest's measurement grid. ``sector`` and ``ue`` also fix the columns of the
two tables every producer writes and every stage reads.

``core`` imports nothing from the rest of ``src``.
They all import it, which is what stops the same dataclass being restated in
three places and drifting.
"""
