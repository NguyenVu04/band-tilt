"""Reading data from disk.

Two kinds of reads live here:

- :func:`load_raw` — the immutable source data, used by notebook 00 (EDA) and
  by the cleaning stage.
- :func:`load_processed` — the cleaned, split outputs of notebook 01, used by
  every ``02x`` model notebook and by notebook 03.

What must NOT go here
---------------------
Transformation of any kind. This module reads files and returns frames; the
only thing it may do beyond reading is enforce dtypes already declared in
``configs/data.yaml``.

``data/raw/`` is never written to. Every step writes new files elsewhere.
"""

from pathlib import Path
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

#: Splits produced by notebook 01 and readable by :func:`load_processed`.
Split = Literal["train", "test"]


def load_raw(cfg: DictConfig) -> pd.DataFrame:
    """Read the immutable source dataset.

    Args:
        cfg: Composed config; uses ``cfg.raw_path``.

    Returns:
        The raw data, unmodified apart from declared dtypes.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the raw file is absent —
            usually because ``dvc pull`` has not been run.

    Notes:
        Keep the failure message actionable: if the path is missing, say so and
        mention ``task dvc:pull``.
    """
    # TODO(1): resolve Path(cfg.raw_path) and raise a clear error if missing
    # TODO(2): read with the format-appropriate reader (parquet by default)
    # TODO(3): apply declared dtypes from cfg.schema.columns
    raise NotImplementedError("src.data.load.load_raw")


def load_processed(cfg: DictConfig, split: Split) -> pd.DataFrame:
    """Read one split written by the cleaning stage (notebook 01).

    Args:
        cfg: Composed config; uses ``cfg.train_path`` / ``cfg.test_path``.
        split: Which split to read.

    Returns:
        The requested split.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown split name.

    Notes:
        A ``02x`` notebook must only ever ask for ``"train"``. The test split is
        frozen from the moment notebook 01 writes it until notebook 03 runs.
        Consider logging a warning when ``"test"`` is requested, so an
        accidental read is visible in the notebook output.

    Example:
        >>> train = load_processed(cfg, "train")
    """
    # TODO(1): map the split name to cfg.train_path / cfg.test_path
    # TODO(2): read the file and return it
    raise NotImplementedError("src.data.load.load_processed")


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Write a cleaned split to ``data/processed/``.

    Args:
        df: The frame to write.
        path: Destination path; parent directories are created as needed.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Write a columnar format with dtypes preserved (parquet) so that a
        reload cannot silently change types. After writing, track the file with
        ``dvc add`` so the contents are versioned against the Git commit.
    """
    # TODO(1): mkdir the parent directory
    # TODO(2): write the frame without the index
    raise NotImplementedError("src.data.load.save_processed")
