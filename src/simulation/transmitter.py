"""Build Sionna-RT transmitters from the cell table.

The table is ``simulation.input.cells_file``, read by
:func:`src.core.cell.read_cells`, and stays fixed while tilt moves.
"""

from __future__ import annotations

import math
from typing import Any

from src.core.cell import Cell


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
