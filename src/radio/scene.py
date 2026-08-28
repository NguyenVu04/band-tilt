"""Sionna-RT scene construction — PROJECT.md section 16 Phase 1.

The scene is the propagation environment: terrain, building geometry, and the
electromagnetic material assigned to every surface. It is built once and reused
for every tilt configuration, because only the transmitter orientations change
between evaluations.

Build the scene once, not once per evaluation
---------------------------------------------
Loading 3,753 meshes and constructing the acceleration structure costs far more
than a single radio-map solve. The surrogate dataset needs hundreds of solves
over the same geometry, so a scene rebuilt inside the sampling loop turns a
tractable job into an intractable one. Load once, mutate orientations, solve
repeatedly.

The scene defines the coordinate frame
--------------------------------------
Cell positions, MDT positions and the evaluation grid are all expressed in the
scene local frame in metres. :func:`scene_bounds` is the single source of that
extent, so the cleaning stage, the UE density grid and the radio map all agree
on where the area is.
"""

from typing import Any

import pandas as pd
from omegaconf import DictConfig


def load_scene(cfg: DictConfig) -> Any:
    """Load the Sionna-RT scene from the 3D map on disk.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_file`` and
            ``cfg.data.scene_dir``.

    Returns:
        The Sionna-RT scene object.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the scene XML is missing.

    Notes:
        Import Sionna-RT inside the function, not at module scope. It is an
        optional extra (``uv sync --extra rt``) and pulls in a large rendering
        stack; a module-level import would make ``import src.radio`` fail for
        anyone doing data work without the simulator installed.

        The materials are declared in the scene XML. Overriding them in code
        would make the propagation environment depend on Python state that the
        scene file does not capture, so results would stop being reproducible
        from the scene alone.

    Example:
        >>> scene = load_scene(cfg)
    """
    # TODO(1): import sionna.rt inside the function body
    # TODO(2): load cfg.data.scene_file
    # TODO(3): raise a message naming the rt extra when the import fails
    raise NotImplementedError("src.radio.scene.load_scene")


def scene_bounds(cfg: DictConfig) -> tuple[float, float, float, float]:
    """Return the scene extent as ``(xmin, ymin, xmax, ymax)`` in metres.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_file``.

    Returns:
        The bounding box in the scene local coordinate frame.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        This is the authoritative extent. ``src.data.clean.drop_outside_scene``
        and ``src.data.ue_density.build_grid`` both call it rather than deriving
        bounds of their own — that is what keeps the UE density grid and the
        RSRP grid aligned.

        Read it from the scene metadata rather than by iterating meshes, which
        is slow enough to matter when a cleaning loop calls it.

    Example:
        >>> xmin, ymin, xmax, ymax = scene_bounds(cfg)
    """
    # TODO(1): read the bounding box from the scene XML or the loaded scene
    # TODO(2): return floats in the scene local frame, metres
    raise NotImplementedError("src.radio.scene.scene_bounds")


def add_transmitters(scene: Any, table: pd.DataFrame, cfg: DictConfig) -> Any:
    """Attach one transmitter per cell-band to the scene.

    Args:
        scene: The scene from :func:`load_scene`.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        cfg: Composed config; uses ``cfg.radio.antenna``.

    Returns:
        The scene, with transmitters attached in table order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Attach transmitters in the table order and give each one a name derived
        from ``(gcell_id, band)``. The radio map comes back indexed by
        transmitter, and that index has to map back to a row of the cell-band
        table without guessing.

        Orientation is not set here — it changes per evaluation and is applied
        by :func:`src.radio.radiomap.set_tilt`.

    Example:
        >>> scene = add_transmitters(load_scene(cfg), table, cfg)
    """
    # TODO(1): one Transmitter per row, positioned at (sim_x, sim_y, antenna_height)
    # TODO(2): name each transmitter from (gcell_id, band) so the index is recoverable
    # TODO(3): apply the antenna array from cfg.radio.antenna and the band carrier
    raise NotImplementedError("src.radio.scene.add_transmitters")
