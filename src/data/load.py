"""Read the project's inputs and the processed outputs of the data pipeline.

Every path comes from ``configs/data.yaml``. No module hardcodes a filename:
the raw MDT export, the cell configuration and the scene are all pointed at by
config, so switching to a different area or a different measurement window is a
config edit rather than a code change.

Why the two raw loaders are separate
------------------------------------
MDT records and cell configuration are different shapes with different keys and
different lifetimes. The cell configuration is a small table describing the
network as deployed; the MDT export is millions of measurements against it. They
are joined explicitly, in cleaning, where the join can be audited — not silently
at load time, where a bad join is invisible.
"""

from pathlib import Path
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

#: Which side of the frozen train/test partition to read.
Split = Literal["train", "test"]


def load_mdt(cfg: DictConfig) -> pd.DataFrame:
    """Read the raw MDT export.

    Args:
        cfg: Composed config; uses ``cfg.data.mdt_path``.

    Returns:
        One row per measurement, with the columns declared in
        ``cfg.data.schema.mdt.columns``.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the export is missing.

    Notes:
        Parse ``date`` as a timezone-aware timestamp here rather than leaving it
        as a string. The temporal split depends on ordering, and lexicographic
        ordering of ISO strings happens to work until a timezone offset differs.

        Do not filter, deduplicate or repair anything here. Loading and cleaning
        are separate so that notebook 00 can report on exactly what arrived.

    Example:
        >>> mdt = load_mdt(cfg)
        >>> len(mdt)
        41481
    """
    # TODO(1): read cfg.data.mdt_path with pandas, dtype from cfg.data.schema.mdt
    # TODO(2): parse the date column as UTC-aware datetime
    # TODO(3): return the frame unmodified — no cleaning at load time
    raise NotImplementedError("src.data.load.load_mdt")


def load_cell_config(cfg: DictConfig) -> pd.DataFrame:
    """Read the cell (and, once available, cell-band) configuration.

    Args:
        cfg: Composed config; uses ``cfg.data.cell_config_path``.

    Returns:
        One row per cell today; one row per cell-band once the multi-band export
        lands.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the export is missing.

    Notes:
        Today's export has one ``digital_tilt`` per cell and no band, carrier
        frequency or transmit power. PROJECT.md section 8 requires all of
        them, and section 3 defines ``tilt = eTilt + mTilt``, which a single
        tilt column cannot express. Until the multi-band export arrives, the
        columns marked pending in ``configs/data.yaml`` will be absent — so
        validate this frame with ``strict=False``. See PROJECT.md section 8.

        This function returns the configuration as exported. Turning it into the
        cell-band table the rest of the project uses is
        :func:`src.radio.cell_band.build_table`, which is where the band
        declaration in ``configs/radio.yaml`` is applied.

    Example:
        >>> cells = load_cell_config(cfg)
        >>> cells.gcell_id.nunique()
        26
    """
    # TODO(1): read cfg.data.cell_config_path with pandas
    # TODO(2): parse sync_date as a date
    # TODO(3): return as exported — the band overlay is applied in src.radio.cell_band
    raise NotImplementedError("src.data.load.load_cell_config")


def load_processed(cfg: DictConfig, split: Split) -> pd.DataFrame:
    """Read one side of the frozen train/test partition.

    Args:
        cfg: Composed config; uses ``cfg.data.train_path`` and
            ``cfg.data.test_path``.
        split: Which side to read.

    Returns:
        The cleaned MDT records for that split.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown ``split`` value.
        FileNotFoundError: Once implemented, when the split has not been written.

    Notes:
        Raise a message that names notebook 03 when the file is missing. The
        default ``FileNotFoundError`` on a parquet path sends the reader looking
        for a data problem rather than an unrun pipeline stage.

    Example:
        >>> train = load_processed(cfg, "train")
    """
    # TODO(1): map split -> cfg.data.train_path / cfg.data.test_path
    # TODO(2): raise ValueError on any other value, naming the two valid ones
    # TODO(3): on FileNotFoundError, re-raise pointing at notebook 03
    raise NotImplementedError("src.data.load.load_processed")


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Write a processed frame to parquet, creating parent directories.

    Args:
        df: The frame to persist.
        path: Destination path, from ``configs/data.yaml``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Parquet rather than CSV: it round-trips dtypes. A CSV turns the
        timezone-aware ``date`` column back into a string and the categorical
        band identifier back into an object, so the schema check that passed
        before writing fails after reading.

    Example:
        >>> save_processed(train_df, cfg.data.train_path)
    """
    # TODO(1): mkdir the parent directory
    # TODO(2): write parquet with index=False
    raise NotImplementedError("src.data.load.save_processed")
