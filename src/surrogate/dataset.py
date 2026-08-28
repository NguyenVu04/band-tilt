"""Build and load D_sur, the surrogate training set — PROJECT.md section 11.

The dataset is::

    D_sur = {(x_k, tilt_k, R_k)} for k = 1..N

with ``x`` the scenario, network and MDT-derived features, ``tilt`` an absolute
tilt configuration, and ``R`` the reference Sionna-RT radio map (PROJECT.md
section 11.2). The target is the map itself, not the five KPIs — those are
derived from it by :mod:`src.kpi` at scoring time.

Building it is the expensive part of the project
------------------------------------------------
Each row costs one ray-tracing solve. That budget is the binding constraint on
everything downstream, so the sampling design in ``configs/radio.yaml`` matters
more than the surrogate architecture: a well-spread few hundred configurations
will beat a poorly-spread few thousand.

Make the build resumable. A run that dies at sample 400 of 500 and cannot
restart has thrown away days of compute.

Split on scenarios, never on configurations or grid cells
---------------------------------------------------------
Grid cells from one configuration are near-duplicates of each other — they share
the same tilts, the same geometry, and most of the same propagation paths. A
split that puts some cells of a configuration in train and others in test
reports an error far below the real one, and the surrogate then looks accurate
right up to the point where an optimizer relies on it.

Holding out whole *configurations* fixes that but is still not enough. PROJECT.md
section 12.3 and Decision 8 require the split to be at **scenario** level: one
scenario is one environment plus one UE mobility realisation, and every
configuration inside it shares the same buildings, materials and trajectories.
Splitting within a scenario leaks exactly the structure the sim-to-reality study
of section 12 exists to measure, and the reported generalisation is then a claim
about tilts dressed up as a claim about environments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def build(cfg: DictConfig) -> pd.DataFrame:
    """Generate D_sur by simulating sampled tilt configurations.

    Args:
        cfg: Composed config; uses ``cfg.radio.sampling``,
            ``cfg.surrogate.dataset`` and ``cfg.data``.

    Returns:
        One row per configuration, holding the tilt vector and its five KPIs.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Load the scene once, outside the loop
        (:func:`src.radio.scene.load_scene`), then reorient and re-solve per
        configuration. Rebuilding the scene per sample dominates the runtime.

        Write each result as it completes rather than accumulating in memory and
        saving at the end, and skip configurations already present on resume.

        Record the Sionna-RT settings alongside the data. The ray-tracing
        parameters define the labels, so a dataset whose provenance is unknown
        cannot be extended later — the new rows would come from a different
        function.

    Example:
        >>> d_sur = build(cfg)
        >>> len(d_sur)
    """
    # TODO(1): table = cell_band.build_table(load_cell_config(cfg), cfg)
    # TODO(2): tilts = sampling.sample_configurations(table, cfg)
    # TODO(3): scene = scene.load_scene(cfg); scene.add_transmitters(...) once
    # TODO(4): rho = ue_density(load_processed(cfg, "train"), cfg) — train split only
    # TODO(5): per tilt: radiomap.evaluate -> kpi.vector.kpi_vector -> append and flush
    # TODO(6): skip configurations already present, so the build is resumable
    raise NotImplementedError("src.surrogate.dataset.build")


def load(cfg: DictConfig) -> pd.DataFrame:
    """Read a previously built D_sur from disk.

    Args:
        cfg: Composed config; uses ``cfg.surrogate.dataset.path``.

    Returns:
        The dataset as written by :func:`build`.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the dataset has not been
            built.

    Notes:
        Point the error message at notebook 03, which is what builds it. A bare
        missing-file error on a parquet path sends the reader looking for a data
        problem rather than an unrun pipeline stage.

    Example:
        >>> d_sur = load(cfg)
    """
    # TODO(1): read cfg.surrogate.dataset.path
    # TODO(2): on FileNotFoundError, re-raise naming notebook 03
    raise NotImplementedError("src.surrogate.dataset.load")


def split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition D_sur into train, validation and test by configuration.

    Args:
        df: The dataset from :func:`load`.
        cfg: Composed config; uses ``cfg.surrogate.dataset`` (``split_on``,
            ``test_size``, ``val_size``, ``seed``).

    Returns:
        ``(train_df, val_df, test_df)``.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when ``split_on`` is not
            ``"configuration"`` and the caller has not justified it.

    Notes:
        This is a different split from the MDT one in :mod:`src.data.split`, and
        deliberately so: that partition governs which measurements inform UE
        density, this one governs which simulated configurations the surrogate
        is scored on. They are independent.

        Keep the baseline configuration in train. The surrogate has to be
        accurate at the reference point every improvement is measured against.

    Example:
        >>> train, val, test = split(load(cfg), cfg)
    """
    # TODO(1): reject split_on values other than "configuration"
    # TODO(2): partition whole configurations, seeded from cfg.surrogate.dataset.seed
    # TODO(3): force the baseline configuration into the train partition
    raise NotImplementedError("src.surrogate.dataset.split")


def save(df: pd.DataFrame, path: str | Path, metadata: dict | None = None) -> None:
    """Persist D_sur together with the settings that produced it.

    Args:
        df: The dataset to write.
        path: Destination path, from ``cfg.surrogate.dataset.path``.
        metadata: Provenance — ray-tracing settings, grid geometry, band table,
            seed, and the code revision.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        The metadata is what makes the dataset extendable. Without it, nobody
        can tell whether new samples came from the same simulator configuration,
        and mixing two is worse than having fewer rows.

    Example:
        >>> save(d_sur, cfg.surrogate.dataset.path, metadata=provenance)
    """
    # TODO(1): mkdir the parent directory
    # TODO(2): write parquet, storing metadata in the file schema, not a sidecar
    raise NotImplementedError("src.surrogate.dataset.save")


def tilt_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the tilt configurations from D_sur as a matrix.

    Args:
        df: The dataset from :func:`load`.

    Returns:
        Shape ``(n_configurations, n_cell_bands)`` in degrees, in cell-band
        table order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Column order must match the cell-band table exactly. A dataset whose
        column order was inferred from a dict, a set, or an unsorted groupby
        will train a surrogate that maps tilts to the wrong cells and gives no
        indication that it has.

    Example:
        >>> tilt_matrix(d_sur).shape
        (256, 26)
    """
    # TODO(1): select the tilt columns in cell-band table order
    # TODO(2): raise when a column is missing, rather than silently reordering
    raise NotImplementedError("src.surrogate.dataset.tilt_matrix")
