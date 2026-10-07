"""Build Sionna-RT transmitters from the sector table.

The table is ``simulation.input.sectors_file``, read by
:func:`src.core.sector.read_sectors`, and stays fixed while tilt moves.
"""

from __future__ import annotations

import math
from typing import Any

from src.core.sector import Sector


def build(scene: Any, sectors: tuple[Sector, ...], band_name: str, power_dbm: float) -> None:
    """Add one transmitter per sector to the scene, at that sector's band tilt.

    Only the current carrier's transmitters are added: a scene carries one
    frequency, so the bands are solved in turn rather than together. Each
    sector contributes its own tilt for ``band_name``, so two bands of the same
    sector can point differently.

    Tilt is the pitch component of the orientation. A rotation about the y axis
    carries the boresight from ``+x`` toward ``-z``, so a *positive* pitch is a
    downtilt.
    """
    from sionna.rt import Transmitter

    for sector in sectors:
        scene.add(
            Transmitter(
                name=sector.name,
                position=[sector.x, sector.y, sector.z],
                orientation=[
                    math.radians(sector.azimuth_deg),
                    math.radians(sector.tilt_for(band_name).baseline_deg),
                    0.0,
                ],
                power_dbm=power_dbm,
            )
        )
