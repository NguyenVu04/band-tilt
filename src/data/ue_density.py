"""UE spatial distribution over the evaluation grid — PROJECT.md section 9.

The target area is discretised into grid cells ``g``. For each one, ``rho(g)``
is the number of MDT observations that fall inside it. That count is the
spatial importance weight in the UE-weighted Band Priority Score (PROJECT.md
section 14): improving coverage where a thousand users were measured should
count for more than improving it where nobody was.

Why the grid is shared, not rebuilt
-----------------------------------
This module must produce a grid identical to the one the radio map uses — same
origin, same cell size, same bounds, same cell ordering. The KPI code indexes UE
density and RSRP with the same flat index, so a half-cell offset between the two
silently scores the wrong locations. Both take their geometry from
``cfg.radio.grid``, and the alignment is asserted rather than assumed.

Density is computed from the TRAINING split
-------------------------------------------
UE density derived from all records would let the test measurements influence
the objective the optimizer maximises. Build it from train, and treat the
density map as a fixed input to optimization thereafter.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def build_grid(cfg: DictConfig) -> pd.DataFrame:
    """Construct the evaluation grid G shared by UE density and every radio map.

    Args:
        cfg: Composed config; uses ``cfg.radio.grid`` (``cell_size_m``,
            ``bounds``, ``height_m``) and ``cfg.data.scene_file``.

    Returns:
        One row per grid cell, with its flat index and centre coordinates.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is the definition of ``G`` in PROJECT.md section 10. Every KPI is a
        sum over these cells, so the resolution changes every reported number: a
        coarse grid averages small holes away, a fine one multiplies ray-tracing
        cost. Fix it once, before generating the first surrogate sample.

    Example:
        >>> grid = build_grid(cfg)
        >>> len(grid)
    """
    # TODO(1): resolve bounds from cfg.radio.grid.bounds or the scene bounding box
    # TODO(2): build cell centres at cfg.radio.grid.cell_size_m spacing
    # TODO(3): assign a stable flat index, row-major, and never re-sort it downstream
    raise NotImplementedError("src.data.ue_density.build_grid")


def ue_density(df: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Count UE observations per grid cell, giving rho(g).

    Args:
        df: Cleaned MDT records, normally the training split.
        cfg: Composed config; uses ``cfg.radio.grid``.

    Returns:
        A vector of length ``|G|``, indexed by the flat grid index from
        :func:`build_grid`.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        All grid cells have equal area, so ``rho(g)`` is the raw count ``n_g``
        and needs no area normalisation (PROJECT.md section 9).

        Cells with zero observations are kept as zeros, not dropped. They are
        real parts of the area that simply had no measurements, and the Band
        Priority Score sums over the full grid.

    Example:
        >>> rho = ue_density(load_processed(cfg, "train"), cfg)
        >>> rho.sum()
    """
    # TODO(1): map each record (sim_x, sim_y) to a flat grid index
    # TODO(2): bincount into a length-|G| vector, minlength=|G|
    # TODO(3): drop records falling outside the grid, and report how many
    raise NotImplementedError("src.data.ue_density.ue_density")


def assert_grid_aligned(rho: np.ndarray, rsrp: np.ndarray) -> None:
    """Assert a density vector and a radio map index the same grid.

    Args:
        rho: UE density from :func:`ue_density`, shape ``(|G|,)``.
        rsrp: An RSRP array whose last axis is the grid, shape
            ``(n_cells, n_bands, |G|)``.

    Raises:
        NotImplementedError: Always — implement this module first.
        AssertionError: Once implemented, when the grid axes disagree.

    Notes:
        Call this wherever the two meet — in the Band Priority Score, and in the
        surrogate feature builder. A mismatch does not raise on its own: NumPy
        broadcasts, the score comes out plausible, and the optimizer happily
        maximises a quantity computed against shifted locations.

    Example:
        >>> assert_grid_aligned(rho, rsrp)
    """
    # TODO(1): compare rho.shape[0] against rsrp.shape[-1]
    # TODO(2): raise with both shapes in the message
    raise NotImplementedError("src.data.ue_density.assert_grid_aligned")
