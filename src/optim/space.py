"""The search space Theta — the one definition both optimizers use.

PROJECT.md section 16 states the feasible set exactly::

    Theta = {theta: theta_min[i, b] <= theta[i, b] <= theta_max[i, b]}

A box, one dimension per cell-band, with per-dimension bounds. That is the whole
constraint set: there is no coupling between cells, no budget on total tilt
change, and no penalty term (PROJECT.md section 3.3).

Why this module exists at all
-----------------------------
It is three lines of arithmetic that could live in either optimizer. Putting it
in both is how the comparison in PROJECT.md section 25 quietly stops being
valid — one implementation clips, the other squashes; one works in degrees, the
other in normalised units; the bounds diverge by a config reload. The result is
then a comparison of two search spaces, reported as a comparison of two methods.

So: BO and MARL both call this module, and neither reads ``configs/radio.yaml``
directly. See docs/adr/0006.

Absolute tilt, not offset
-------------------------
The coordinates of this space are absolute tilts. An offset parameterisation
would make the bounds depend on the current configuration —
``[theta_min - theta_current, theta_max - theta_current]`` — so the space would
change shape every time the network moved, and a policy or surrogate trained in
one would not transfer. See docs/adr/0001 and PROJECT.md section 3.1.

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
            reported per cell-band (PROJECT.md section 27.1), and a bare vector
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
            This is the number PROJECT.md section 25.4 asks to be varied for the
            scalability comparison. Standard GP-based BO degrades well before
            MARL does as it grows; record where, rather than avoiding the
            regime.
        """
        # TODO(1): return len(self.table)
        raise NotImplementedError("src.optim.space.TiltSpace.n_dims")

    def to_unit(self, theta: np.ndarray) -> np.ndarray:
        """Map absolute tilts in degrees to the unit cube.

        Args:
            theta: Shape ``(n_dims,)`` or ``(n, n_dims)``, in degrees.

        Returns:
            The same shape, each coordinate in ``[0, 1]``.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            Per-dimension, using each cell-band own bounds. A single global
            scaling would distort the space whenever two bands have different
            tilt ranges, which is the expected case.

        Example:
            >>> u = space.to_unit(theta)
        """
        # TODO(1): (theta - lower) / (upper - lower), broadcasting over a batch
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
            >>> theta = space.from_unit(u)
        """
        # TODO(1): lower + u * (upper - lower), broadcasting over a batch
        raise NotImplementedError("src.optim.space.TiltSpace.from_unit")

    def clip(self, theta: np.ndarray) -> np.ndarray:
        """Project a configuration into the feasible box.

        Args:
            theta: Shape ``(n_dims,)`` or ``(n, n_dims)``, in degrees.

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
            >>> theta = space.clip(theta)
        """
        # TODO(1): np.clip against the stored bounds
        raise NotImplementedError("src.optim.space.TiltSpace.clip")

    def baseline(self) -> np.ndarray:
        """The network as currently deployed, theta_current.

        Returns:
            Absolute tilts in degrees, in table order.

        Raises:
            NotImplementedError: Always — implement this module first.

        Notes:
            The reference for every reported improvement (PROJECT.md
            section 27.2) and the MARL ``baseline`` reset strategy.

        Example:
            >>> theta_0 = space.baseline()
        """
        # TODO(1): return cell_band.current_tilt(self.table)
        raise NotImplementedError("src.optim.space.TiltSpace.baseline")
