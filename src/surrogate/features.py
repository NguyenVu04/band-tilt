"""Tilt-independent scene channels: surface height, distance field, shadow, angle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from src.core.cell import Cell
from src.simulation.grid import GridSpec, Raster
from src.simulation.scene import SceneBounds, surface_height

# A distance field has unit gradient, so a magnitude near zero is not a shallow
# wall but a place where no direction is downhill -- the medial axis of an open
# space, or the interior of a solid block. The angle there is meaningless.
_FLAT_GRADIENT = 1e-3

# A ray reaching within this of its target has not been blocked. Absorbs the
# floating-point error in normalising a direction over kilometre-scale spans.
_LOS_TOL_M = 1e-2


@dataclass(frozen=True)
class SceneFeatures:
    """What one perturbed scene contributes to the surrogate's input.

    Two resolutions. The first three arrays are built on the sub-tile grid,
    shaped ``[n_rows * sub, n_cols * sub]``, because a facade is narrower than a
    tile. The rest are on the tile grid the radio map is solved on, so they
    stack directly against ``rsrp_dbm``.

    Attributes:
        dsm_m: Height of the topmost surface, ``nan`` where the ray left the
            scene -- which is a point beyond the ground, not ground at zero.
        sdf_m: Distance to the nearest building wall, positive outside a
            building and negative inside. Horizontal only: it is measured on a
            top-down occupancy and says nothing about height.
        facade_azimuth_deg: Bearing of the outward wall normal, counter-clockwise
            from the x axis -- the convention
            :attr:`src.core.cell.Cell.azimuth_deg` uses, so a facade and a cell
            boresight are directly comparable. ``nan`` where the field is flat.
        mean_height_m: Surface height per tile.
        max_height_m: Tallest surface per tile. Kept beside the mean because a
            mean over a tile erases the lone tall blocker that casts the shadow.
        mean_sdf_m: Distance field per tile.
        los_fraction: Share of a tile's sub-tile points a mast can see, in
            ``[0, 1]``, shaped ``[n_tx, n_rows, n_cols]``.
        elevation_deg: Depression angle from the mast down to the UE plane,
            shaped ``[n_tx, n_rows, n_cols]``. Geometric and tilt-independent:
            subtract a cell-band's tilt to get the angle off its boresight.
    """

    dsm_m: np.ndarray
    sdf_m: np.ndarray
    facade_azimuth_deg: np.ndarray
    mean_height_m: np.ndarray
    max_height_m: np.ndarray
    mean_sdf_m: np.ndarray
    los_fraction: np.ndarray
    elevation_deg: np.ndarray


def build(
    mi_scene: Any,
    bounds: SceneBounds,
    raster: Raster,
    cells: tuple[Cell, ...],
    spec: GridSpec,
    ue_height_m: float,
) -> SceneFeatures:
    """Build every channel for one scene.

    Takes the :class:`~src.simulation.grid.Raster` rather than re-deriving the
    grid, so the channels carry the origin and shape the radio map was solved
    on. A locally recomputed grid could sit half a tile off and every channel
    would still look entirely plausible.

    ``mi_scene`` must be re-read from ``scene.mi_scene`` after any
    :func:`src.simulation.perturb.apply`, which rebuilds it.

    Args:
        mi_scene: The Mitsuba scene to cast against.
        bounds: The scene's extent; read for ``launch_z``.
        raster: The tile grid, from :func:`src.simulation.grid.build`.
        cells: The masts. Co-located cells are solved once and shared.
        spec: Read for ``subsamples_per_tile`` and ``free_height_tol_m``.
        ue_height_m: The UE plane, which must match the radio map's height.
    """
    sub = spec.subsamples_per_tile
    step_m = raster.tile_size_m / sub
    fine_x, fine_y = _fine_centres(raster, sub)

    dsm_m = surface_height(mi_scene, fine_x, fine_y, bounds.launch_z)
    sdf_m = signed_distance(dsm_m, spec.free_height_tol_m, step_m)
    # A tile whose ray left the scene pools as zero height, the convention
    # src.simulation.grid.build already uses for a tile holding no building.
    height_m = np.nan_to_num(dsm_m, nan=0.0)
    centre_x, centre_y = raster.tile_centres()

    return SceneFeatures(
        dsm_m=dsm_m,
        sdf_m=sdf_m,
        facade_azimuth_deg=facade_azimuth(sdf_m, step_m),
        mean_height_m=pool(height_m, sub),
        max_height_m=pool(height_m, sub, np.max),
        mean_sdf_m=pool(sdf_m, sub),
        los_fraction=los_fraction(mi_scene, cells, fine_x, fine_y, ue_height_m, sub),
        elevation_deg=elevation_deg(cells, centre_x, centre_y, ue_height_m),
    )


def signed_distance(dsm_m: np.ndarray, free_height_tol_m: float, step_m: float) -> np.ndarray:
    """Distance to the nearest wall, positive outside a building, negative inside.

    Thresholds the surface model the way :func:`src.simulation.grid.build` does,
    so "building" means the same thing in both. A ``nan`` is a ray that left the
    scene and counts as unoccupied, never as a structure.

    The zero level set is the facade, which is why no separate boundary mask is
    returned: a mask is one threshold away and carries strictly less.
    """
    occupied = np.nan_to_num(dsm_m, nan=0.0) > free_height_tol_m
    outside = ndimage.distance_transform_edt(~occupied, sampling=step_m)
    inside = ndimage.distance_transform_edt(occupied, sampling=step_m)
    return np.asarray(outside) - np.asarray(inside)


def facade_azimuth(sdf_m: np.ndarray, step_m: float, smooth_cells: float = 1.0) -> np.ndarray:
    """Bearing of the outward wall normal, counter-clockwise from the x axis.

    A distance field has unit gradient, so its gradient *is* the outward normal
    and needs no normalising. Which way a wall faces is what decides where
    specular energy goes, and a boundary mask alone cannot express it.

    ``smooth_cells`` is a Gaussian width in cells, applied before differencing.
    Without it the gradient of a rasterised field can only point along the
    directions the sampling lattice offers.

    Returns ``nan`` where the field is flat, rather than the spurious angle
    ``arctan2(0, 0)`` would give.
    """
    field = ndimage.gaussian_filter(sdf_m, smooth_cells) if smooth_cells > 0 else sdf_m
    gradient_y, gradient_x = np.gradient(field, step_m)
    azimuth = np.degrees(np.arctan2(gradient_y, gradient_x))
    return np.where(np.hypot(gradient_x, gradient_y) > _FLAT_GRADIENT, azimuth, np.nan)


def elevation_deg(
    cells: tuple[Cell, ...],
    centre_x: np.ndarray,
    centre_y: np.ndarray,
    ue_height_m: float,
) -> np.ndarray:
    """Depression angle from each mast down to each tile, ``[n_tx, n_rows, n_cols]``.

    Geometric, and deliberately independent of tilt: this is the angle the tilt
    is measured against, so it survives every configuration the optimizer tries.
    Positive below the mast horizon.
    """
    masts, inverse = _unique_masts(cells)
    angles = np.empty((len(masts), *np.shape(centre_x)), dtype=np.float64)
    for index, (x, y, z) in enumerate(masts):
        horizontal_m = np.hypot(centre_x - x, centre_y - y)
        angles[index] = np.degrees(np.arctan2(z - ue_height_m, horizontal_m))
    return angles[inverse]


def los_fraction(
    mi_scene: Any,
    cells: tuple[Cell, ...],
    fine_x: np.ndarray,
    fine_y: np.ndarray,
    ue_height_m: float,
    sub: int,
) -> np.ndarray:
    """Share of each tile's sub-tile points a mast can see, ``[n_tx, n_rows, n_cols]``.

    One ray per sub-tile point per mast, stopped short of its target by anything
    it meets. Averaging within the tile is what makes this a fraction rather
    than a flag: a tile is wider than a street, so its centre alone decides
    shadow by whichever side of a wall it happens to fall on.

    Raises:
        ImportError: When Mitsuba is not installed.
    """
    import mitsuba as mi

    flat_x = np.ascontiguousarray(np.asarray(fine_x, dtype=np.float64).ravel())
    flat_y = np.ascontiguousarray(np.asarray(fine_y, dtype=np.float64).ravel())
    masts, inverse = _unique_masts(cells)

    n_rows, n_cols = np.shape(fine_x)
    seen = np.empty((len(masts), n_rows // sub, n_cols // sub), dtype=np.float64)
    for index, (x, y, z) in enumerate(masts):
        delta_x = flat_x - x
        delta_y = flat_y - y
        delta_z = float(ue_height_m - z)
        distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        origin = mi.Point3f(
            np.full_like(flat_x, x), np.full_like(flat_x, y), np.full_like(flat_x, z)
        )
        direction = mi.Vector3f(
            delta_x / distance, delta_y / distance, np.full_like(flat_x, delta_z) / distance
        )
        interaction = mi_scene.ray_intersect(mi.Ray3f(origin, direction))

        blocked_at = np.where(
            np.asarray(interaction.is_valid(), dtype=bool),
            np.asarray(interaction.t, dtype=np.float64),
            np.inf,
        )
        clear = (blocked_at >= distance - _LOS_TOL_M).reshape(n_rows, n_cols)
        seen[index] = pool(clear.astype(np.float64), sub)
    return seen[inverse]


def pool(
    fine: np.ndarray,
    sub: int,
    reduce: Callable[..., np.ndarray] = np.mean,
) -> np.ndarray:
    """Reduce a sub-tile raster onto the tile grid it was sampled over."""
    n_rows, n_cols = fine.shape
    blocks = fine.reshape(n_rows // sub, sub, n_cols // sub, sub)
    return np.asarray(reduce(blocks, axis=(1, 3)))


def _fine_centres(raster: Raster, sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre of every sub-tile, each shaped ``[n_rows * sub, n_cols * sub]``."""
    step_m = raster.tile_size_m / sub
    xs = raster.origin_x + (np.arange(raster.n_cols * sub) + 0.5) * step_m
    ys = raster.origin_y + (np.arange(raster.n_rows * sub) + 0.5) * step_m
    return np.meshgrid(xs, ys, indexing="xy")


def _unique_masts(cells: tuple[Cell, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Distinct mast positions, and the index of each cell's own.

    Shadow and elevation are functions of where a mast stands, not of where its
    cells point, so a node's co-located cells are one ray cast rather than
    several.
    """
    positions = np.array([[cell.x, cell.y, cell.z] for cell in cells], dtype=np.float64)
    masts, inverse = np.unique(positions, axis=0, return_inverse=True)
    return masts, np.asarray(inverse).ravel()
