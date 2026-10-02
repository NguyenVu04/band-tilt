"""Build Sionna-RT transmitters from the cell table, and check their masts.

The layout itself is generated with the scenario (:mod:`src.scenario.run`)
and remains fixed per scenario; :func:`src.core.cell.read_cells` reads it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.core.cell import Cell
from src.simulation import scene as scene_module
from src.simulation.scene import SceneBounds


def build(scene: Any, cells: tuple[Cell, ...], band_name: str, power_dbm: float) -> None:
    """Add one transmitter per cell to the scene, at that cell's band tilt.

    Only the current carrier's transmitters are added: a scene carries one
    frequency, so the bands are solved in turn rather than together. Each
    cell contributes its own tilt for ``band_name``, so two bands of the same
    cell can point differently.

    Tilt is the pitch component of the orientation. A rotation about the y axis
    carries the boresight from ``+x`` toward ``-z``, so a *positive* pitch is a
    downtilt.
    """
    from sionna.rt import Transmitter

    for cell in cells:
        scene.add(
            Transmitter(
                name=cell.name,
                position=[cell.x, cell.y, cell.z],
                orientation=[
                    math.radians(cell.azimuth_deg),
                    math.radians(cell.tilt_for(band_name).baseline_deg),
                    0.0,
                ],
                power_dbm=power_dbm,
            )
        )


def validate(
    mi_scene: Any,
    bounds: SceneBounds,
    cells: tuple[Cell, ...],
    free_height_tol_m: float,
) -> tuple[str, ...]:
    """Report masts the current geometry no longer supports. Empty tuple is all clear.

    Holds the layout to the rule :func:`src.scenario.layout.generate_layout`
    places it under — a mast stands on open ground, at a measured height above
    it — against whatever the scene holds now. A changed scene can swallow a
    mast, leave it standing on a roof, or leave it on nothing. All three are
    reported for the run log; none is silently corrected, because moving a mast
    would defeat the point of holding the layout fixed.

    ``free_height_tol_m`` is ``scenario.grid.free_height_tol_m``, so ground and
    building mean the same thing here as everywhere else in the pipeline.
    """
    x = np.array([cell.x for cell in cells])
    y = np.array([cell.y for cell in cells])
    height = scene_module.surface_height(mi_scene, x, y, bounds.launch_z)

    problems = []
    for cell, surface in zip(cells, height, strict=True):
        if not np.isfinite(surface):
            problems.append(f"{cell.name}: mast at {cell.z:.1f} m stands over no surface at all")
        elif surface > cell.z:
            problems.append(
                f"{cell.name}: mast at {cell.z:.1f} m is inside geometry reaching {surface:.1f} m"
            )
        elif surface > free_height_tol_m:
            problems.append(
                f"{cell.name}: mast at {cell.z:.1f} m stands on a building "
                f"reaching {surface:.1f} m, not on open ground"
            )
    return tuple(problems)
