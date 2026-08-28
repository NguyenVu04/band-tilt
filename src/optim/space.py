"""The search space X — the one definition both optimizers use.

PROJECT.md section 3.2 states the feasible set exactly::

    X = {tilt: tilt_min[i, b] <= tilt[i, b] <= tilt_max[i, b]}

A box, one dimension per cell-band, with per-dimension bounds. That is the whole
constraint set: there is no coupling between cells, no budget on total tilt
change, and no penalty term (PROJECT.md section 2.3).

TuRBO searches a *trust region* ``T_t`` inside this box rather than the box
itself (PROJECT.md section 13.2), and MARL actions are clipped to it every step
(section 14.3). Both still derive the box from here — the trust region is a
subset of X, never a redefinition of it.

Why this module exists at all
-----------------------------
It is three lines of arithmetic that could live in either optimizer. Putting it
in both is how the comparison in PROJECT.md section 17 quietly stops being
valid — one implementation clips, the other squashes; one works in degrees, the
other in normalised units; the bounds diverge by a config reload. The result is
then a comparison of two search spaces, reported as a comparison of two methods.

So: TuRBO and MARL both call this module, and neither reads
``configs/radio.yaml`` directly.

Absolute tilt, not offset
-------------------------
The coordinates of this space are absolute tilts. An offset parameterisation
would make the bounds depend on the current configuration —
``[tilt_min - current_tilt, tilt_max - current_tilt]`` — so the space would
change shape every time the network moved, and a policy or surrogate trained in
one would not transfer. PROJECT.md section 3.1 and section 25.1 are explicit
that absolute tilt is the decision variable and the offset is derived afterwards.

Normalisation
-------------
Optimizers generally want the unit cube; the simulator and the report want
degrees. :func:`to_unit` and :func:`from_unit` are the only sanctioned
conversion, so a candidate cannot be interpreted in the wrong units at one end
of the loop.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class TiltSpace:
    """The box-constrained absolute-tilt search space.

    Constructed once per experiment from the cell-band table, then shared by
    whichever optimizer is running. Holds the bounds and the column ordering
    together, because a bound vector without its ordering cannot be interpreted.
    """

    def __init__(self, table: pd.DataFrame, cfg: DictConfig) -> None:
        """Build the space from the cell-band table.

        Args:
            table: The cell-band table from
                :func:`src.radio.cell_band.build_table`.
            cfg: Composed config; uses ``cfg.radio.bands[*].tilt``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Store the table itself, not just the bounds. Every result has to be
            reported per cell-band (PROJECT.md section 19), and a bare vector
            of 26 numbers cannot be mapped back to cells afterwards.
        """
        # TODO(1): lower, upper = cell_band.tilt_bounds(table)
        # TODO(2): store table, lower, upper, and the dimension count
        # TODO(3): raise when any lower >= upper
        raise NotImplementedError("src.optim.space.TiltSpace.__init__")

    @property
    def n_dims(self) -> int:
        """Number of decision variables — cells times bands.

        Returns:
            The dimensionality of the search space.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            This is the number PROJECT.md section 17 asks to be varied for the
            scalability comparison. Standard GP-based BO degrades well before
            MARL does as it grows; record where, rather than avoiding the
            regime.
        """
        # TODO(1): return len(self.table)
        raise NotImplementedError("src.optim.space.TiltSpace.n_dims")

    def to_unit(self, tilt: np.ndarray) -> np.ndarray:
        """Map absolute tilts in degrees to the unit cube.

        Args:
            tilt: Shape ``(n_dims,)`` or ``(n, n_dims)``, in degrees.

        Returns:
            The same shape, each coordinate in ``[0, 1]``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Per-dimension, using each cell-band own bounds. A single global
            scaling would distort the space whenever two bands have different
            tilt ranges, which is the expected case.

        Example:
            >>> u = space.to_unit(tilt)
        """
        # TODO(1): (tilt - lower) / (upper - lower), broadcasting over a batch
        raise NotImplementedError("src.optim.space.TiltSpace.to_unit")

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates back to absolute tilts in degrees.

        Args:
            u: Shape ``(n_dims,)`` or ``(n, n_dims)``, each coordinate in
                ``[0, 1]``.

        Returns:
            Absolute tilts in degrees.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            The exact inverse of :meth:`to_unit`. Round-trip it in the tests:
            an asymmetry between the two is how a candidate ends up evaluated at
            a different configuration from the one the optimizer proposed.

        Example:
            >>> tilt = space.from_unit(u)
        """
        # TODO(1): lower + u * (upper - lower), broadcasting over a batch
        raise NotImplementedError("src.optim.space.TiltSpace.from_unit")

    def clip(self, tilt: np.ndarray) -> np.ndarray:
        """Project a configuration into the feasible box.

        Args:
            tilt: Shape ``(n_dims,)`` or ``(n, n_dims)``, in degrees.

        Returns:
            The same shape, with every coordinate inside its bounds.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            A last resort for floating-point drift at the boundary, not a way to
            handle an optimizer that proposes out-of-bounds candidates. Silently
            clipping a genuinely infeasible proposal hides the bug and reports a
            result for a configuration the optimizer never chose — prefer
            :func:`src.radio.sampling.assert_within_bounds` and fix the caller.

        Example:
            >>> tilt = space.clip(tilt)
        """
        # TODO(1): np.clip against the stored bounds
        raise NotImplementedError("src.optim.space.TiltSpace.clip")

    def baseline(self) -> np.ndarray:
        """The network as currently deployed, current_tilt.

        Returns:
            Absolute tilts in degrees, in table order.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            The reference for every reported improvement (PROJECT.md
            section 27.2) and the MARL ``baseline`` reset strategy.

        Example:
            >>> tilt_0 = space.baseline()
        """
        # TODO(1): return cell_band.current_tilt(self.table)
        raise NotImplementedError("src.optim.space.TiltSpace.baseline")
