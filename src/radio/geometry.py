"""Angle conventions — the one place the coordinate conversion is defined.

Two conventions meet in this project and they do not agree:

*The radio convention*, used by the cell configuration and the MDT export.
Azimuth is a compass bearing, measured clockwise from north. Tilt is a
downtilt: a positive number points the beam at the ground.

*The simulation convention*, used by Sionna-RT. Yaw is measured
counter-clockwise from the x-axis, and pitch is a rotation in which a positive
value lifts the beam.

The conversion fixed by PROJECT.md section 22.2 is::

    yaw = deg2rad(90.0 - azimuth)
    pitch = deg2rad(-tilt)

so a positive radio downtilt becomes a NEGATIVE pitch in the simulation frame.

Why this is a module and not two lines at the call site
-------------------------------------------------------
A sign error here does not raise. It produces a complete, plausible radio map
with every beam pointing at the sky, KPIs that are internally consistent, and
an optimizer that converges confidently on the wrong answer. The mistake is
invisible in every downstream number.

So the conversion exists exactly once, it is unit-tested against hand-computed
values, and no other module may write ``90.0 -`` or ``-tilt``. See
PROJECT.md section 22.2.
"""

import numpy as np
import pandas as pd


def absolute_tilt(etilt: np.ndarray, mtilt: np.ndarray) -> np.ndarray:
    """Combine electrical and mechanical tilt into the absolute tilt tilt.

    Args:
        etilt: Electrical tilt in degrees.
        mtilt: Mechanical tilt in degrees.

    Returns:
        Absolute tilt in degrees, ``etilt + mtilt``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        PROJECT.md section 8 defines ``tilt = eTilt + mTilt``, and section 3.1
        makes that sum — not either component — the optimization variable. The
        split back into a settable electrical tilt and a fixed mechanical tilt
        is a deployment concern, handled in reporting.

        Today's export carries a single ``digital_tilt`` and no mechanical tilt
        column, so this is not yet reachable from real data (PROJECT.md section 8).

    Example:
        >>> absolute_tilt(np.array([6.0]), np.array([2.0]))
        array([8.])
    """
    # TODO(1): validate the two arrays broadcast against each other
    # TODO(2): return etilt + mtilt
    raise NotImplementedError("src.radio.geometry.absolute_tilt")


def azimuth_to_yaw(azimuth_deg: np.ndarray) -> np.ndarray:
    """Convert a compass azimuth in degrees to a simulation yaw in radians.

    Args:
        azimuth_deg: Azimuth in degrees, clockwise from north.

    Returns:
        Yaw in radians, counter-clockwise from the x-axis.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``yaw = deg2rad(90.0 - azimuth)`` (PROJECT.md section 22.2). Do not wrap
        the result into a fixed interval unless a caller needs it: wrapping
        makes two mathematically equal orientations compare unequal, which
        breaks the round-trip test.

    Example:
        >>> azimuth_to_yaw(np.array([90.0]))
        array([0.])
    """
    # TODO(1): return np.deg2rad(90.0 - azimuth_deg)
    raise NotImplementedError("src.radio.geometry.azimuth_to_yaw")


def tilt_to_pitch(tilt_deg: np.ndarray) -> np.ndarray:
    """Convert an absolute radio downtilt in degrees to a simulation pitch in radians.

    Args:
        tilt_deg: Absolute tilt in degrees. Positive is downtilt.

    Returns:
        Pitch in radians. Negative for a downtilt.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``pitch = deg2rad(-tilt)`` (PROJECT.md section 22.2). The sign flip is the
        entire content of this function and the entire risk: read the assertion
        in the test rather than re-deriving it.

    Example:
        >>> tilt_to_pitch(np.array([6.0]))
        array([-0.10471976])
    """
    # TODO(1): return np.deg2rad(-tilt_deg)
    raise NotImplementedError("src.radio.geometry.tilt_to_pitch")


def orientations(cell_bands: pd.DataFrame, tilt: np.ndarray) -> np.ndarray:
    """Build the Sionna-RT orientation array for one tilt configuration.

    Args:
        cell_bands: The cell-band table from
            :func:`src.radio.cell_band.build_table`, carrying azimuth per row.
        tilt: Absolute tilt per cell-band, in degrees, ordered to match
            ``cell_bands``.

    Returns:
        An array of shape ``(len(cell_bands), 3)`` holding
        ``(yaw, pitch, roll)`` in radians, ready to hand to Sionna-RT.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when ``tilt`` and ``cell_bands`` differ
            in length.

    Notes:
        Roll is zero for every transmitter here; it is included so the array
        matches the shape Sionna-RT expects.

        Check the length rather than trusting it. A tilt vector that is out of
        order or the wrong size assigns tilts to the wrong cells, and every
        downstream number is then computed for a network that does not exist.

    Example:
        >>> orient = orientations(cell_bands, tilt)
        >>> orient.shape
        (26, 3)
    """
    # TODO(1): raise ValueError when len(tilt) != len(cell_bands)
    # TODO(2): yaw from azimuth_to_yaw(cell_bands.azimuth)
    # TODO(3): pitch from tilt_to_pitch(tilt)
    # TODO(4): stack with a zero roll column
    raise NotImplementedError("src.radio.geometry.orientations")
