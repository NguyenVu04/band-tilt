"""The UE table contract: one row per UE per interval.

Whoever produces the table, the synthetic generator (:mod:`src.scenario`) or a
measurement campaign, writes these columns, and everything downstream reads
only them.
"""

# Required columns. ``tile_col``/``tile_row`` index the manifest grid.
UE_COLUMNS = ("t_index", "t_s", "x", "y", "z", "tile_col", "tile_row")

# Allowed but not required: the synthetic mixture component a UE was drawn
# from (``-1`` background, ``0..`` hotspot), which real data does not have.
OPTIONAL_UE_COLUMNS = ("component",)
