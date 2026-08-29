"""Turn an absolute tilt configuration into an RSRP array.

This is the ground truth of the whole project::

    tilt -> Sionna-RT -> radio map -> RSRP

Everything downstream — the five KPIs, the surrogate labels, the final
validation — is computed from the array this module returns.

The array contract
------------------
Radio maps are returned as ``float`` arrays of shape
``(n_cell_bands, |G|)`` in dBm, where the first axis follows the row order of
the cell-band table and the second follows the flat grid index from
:func:`src.data.ue_density.build_grid`. Locations with no measurable signal hold
``-inf`` rather than a sentinel number, so the hole threshold comparison in
:mod:`src.kpi` needs no special case.

Returning a plain NumPy array, rather than a Sionna object, is deliberate: it is
what lets :mod:`src.kpi` score a ray-traced map, a surrogate prediction and a
hand-built test fixture with the same code.

Cost
----
One solve over this scene is the expensive operation the entire surrogate exists
to avoid. Reuse the scene across calls (see :mod:`src.radio.scene`), and cache
by tilt configuration when sampling: a repeated configuration is common in an
optimization loop and re-solving it wastes the project's scarcest resource.
"""

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def set_tilt(scene: Any, table: pd.DataFrame, tilt: np.ndarray) -> Any:
    """Apply an absolute tilt configuration to the scene transmitters.

    Args:
        scene: A scene with transmitters attached by
            :func:`src.radio.scene.add_transmitters`.
        table: The cell-band table, giving azimuth and row order.
        tilt: Absolute tilt per cell-band in degrees, ordered to match
            ``table``.

    Returns:
        The scene, with every transmitter reoriented.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when ``tilt`` does not match ``table``.

    Notes:
        Convert through :func:`src.radio.geometry.orientations` and nowhere
        else. Writing the ``90 - azimuth`` or ``-tilt`` conversion here would
        create a second copy of the convention, which is the failure
        :mod:`src.radio.geometry` exists to prevent.

        Mutate the existing transmitters rather than removing and re-adding
        them: re-adding invalidates the acceleration structure and costs a full
        rebuild per evaluation.

    Example:
        >>> scene = set_tilt(scene, table, tilt)
    """
    # TODO(1): orient = geometry.orientations(table, tilt)
    # TODO(2): assign orientation to each transmitter, in table order
    raise NotImplementedError("src.radio.radiomap.set_tilt")


def compute_radiomap(scene: Any, table: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Solve the radio map for the scene as currently oriented.

    Args:
        scene: A scene with transmitters attached and oriented.
        table: The cell-band table, giving the first-axis order.
        cfg: Composed config; uses ``cfg.radio.ray_tracing`` and
            ``cfg.radio.grid``.

    Returns:
        RSRP in dBm, shape ``(len(table), |G|)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The measurement plane sits at ``cfg.radio.grid.height_m``, which matches
        the MDT ``ue_height`` so that simulated and measured RSRP are
        comparable at all.

        Add each band transmit power to the path gain to get RSRP; the solver
        returns path gain, which is not the same quantity and differs by a
        per-band constant.

        The ray-tracing settings define the ground truth itself. Changing
        ``max_depth`` or ``samples_per_tx`` invalidates every surrogate sample
        generated before the change, because the labels no longer come from the
        same function.

    Example:
        >>> rsrp = compute_radiomap(scene, table, cfg)
        >>> rsrp.shape
        (26, 10000)
    """
    # TODO(1): build the radio map solver from cfg.radio.ray_tracing
    # TODO(2): solve on the plane at cfg.radio.grid.height_m over cfg.radio.grid bounds
    # TODO(3): convert path gain to RSRP using each band tx_power_dbm
    # TODO(4): flatten the spatial axes to the same row-major order as build_grid
    # TODO(5): return -inf where there is no measurable signal, never a sentinel number
    raise NotImplementedError("src.radio.radiomap.compute_radiomap")


def evaluate(tilt: np.ndarray, scene: Any, table: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Evaluate one tilt configuration end to end.

    Args:
        tilt: Absolute tilt per cell-band in degrees.
        scene: A scene with transmitters attached.
        table: The cell-band table.
        cfg: Composed config.

    Returns:
        RSRP in dBm, shape ``(len(table), |G|)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The one function an optimizer or a dataset builder should call. It
        validates the configuration before spending a solve on it: an
        out-of-bounds tilt produces a perfectly plausible radio map for a
        network that cannot exist.

    Example:
        >>> rsrp = evaluate(tilt, scene, table, cfg)
    """
    # TODO(1): sampling.assert_within_bounds(tilt, table)
    # TODO(2): set_tilt then compute_radiomap
    raise NotImplementedError("src.radio.radiomap.evaluate")
