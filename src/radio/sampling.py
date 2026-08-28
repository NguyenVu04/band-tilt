"""Draw absolute tilt configurations for the surrogate dataset.

PROJECT.md section 16 Phase 3 needs a set of configurations whose radio maps
become the surrogate training data. The set has to cover the tilt space well
enough that the surrogate is accurate where the optimizer will actually look —
which is not where uniform random sampling concentrates.

Why a low-discrepancy sequence
------------------------------
The space has one dimension per cell-band, so 26 cells at one band is already
26-dimensional and grows with every band added. At the sample counts that are
affordable when each sample is a ray-tracing run, independent uniform draws
leave large regions untouched and cluster elsewhere. A Sobol or Latin hypercube
design spreads the same budget far more evenly.

Every sample must be feasible
-----------------------------
Bounds come from ``configs/radio.yaml`` via
:func:`src.radio.cell_band.tilt_bounds`, per cell-band — they are not global. A
configuration that violates them is not a hard case for the surrogate to learn,
it is a configuration the network cannot be put into, and training on it spends
budget teaching the surrogate about states no optimizer may propose.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def sample_configurations(table: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """Draw absolute tilt configurations covering the feasible set.

    Args:
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        cfg: Composed config; uses ``cfg.radio.sampling`` (``strategy``,
            ``n_samples``, ``step_deg``, ``seed``).

    Returns:
        An array of shape ``(n_samples, len(table))`` in degrees, each row a
        configuration within its per-cell-band bounds.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown sampling strategy.

    Notes:
        Include the current configuration as one of the samples. The baseline
        has to be in the dataset: it is the reference every reported improvement
        is measured against, and a surrogate that is inaccurate exactly there
        would undermine every comparison.

        Quantise to ``step_deg`` when it is set, after drawing and before
        clipping. A real remote electrical tilt unit moves in discrete steps, so
        a continuous optimum it cannot be set to is not an answer.

    Example:
        >>> tilts = sample_configurations(table, cfg)
        >>> tilts.shape
        (256, 26)
    """
    # TODO(1): dispatch on cfg.radio.sampling.strategy (sobol, lhs, uniform, grid)
    # TODO(2): draw in the unit cube, then scale to per-row tilt_bounds(table)
    # TODO(3): quantise to cfg.radio.sampling.step_deg when it is not null
    # TODO(4): prepend cell_band.current_tilt(table) as the first row
    # TODO(5): seed from cfg.radio.sampling.seed — never a literal
    raise NotImplementedError("src.radio.sampling.sample_configurations")


def assert_within_bounds(tilt: np.ndarray, table: pd.DataFrame) -> None:
    """Assert every tilt in a configuration lies within its own bounds.

    Args:
        tilt: One configuration of shape ``(len(table),)`` or a batch of shape
            ``(n, len(table))``, in degrees.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.

    Raises:
        NotImplementedError: Always — implement this module first.
        AssertionError: Once implemented, naming the cell-bands that violate
            their bounds and by how much.

    Notes:
        Call this on anything an optimizer proposes, not only on sampled
        designs. A MARL policy squashed into the wrong interval, or a BO
        candidate returned in normalised units, both produce out-of-bounds tilts
        that Sionna-RT will happily simulate.

    Example:
        >>> assert_within_bounds(optimized_tilt, table)
    """
    # TODO(1): fetch lower/upper from cell_band.tilt_bounds(table)
    # TODO(2): compare with broadcasting so a batch works too
    # TODO(3): raise naming the offending cell-bands and the amount of violation
    raise NotImplementedError("src.radio.sampling.assert_within_bounds")
