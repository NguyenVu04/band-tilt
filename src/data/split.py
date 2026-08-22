"""Shared train/val/test splitting — the single authoritative implementation.

Every notebook, script and test obtains its splits from this module, using the
scheme and seed declared in ``configs/data.yaml``. Nothing else in the project
may split data.

Why this is centralised
-----------------------
Models validated on different folds cannot be compared. If each ``02x``
notebook drew its own validation split, the results table in notebook 03 would
be measuring the folds as much as the models.

Why splits are group-aware
--------------------------
Records that belong to the same entity — the same trajectory, device, patient
or session — are correlated. A plain random or stratified split puts siblings
on both sides of the boundary, so the model is scored on data it effectively
saw during training and the estimate comes out optimistic. Splitting on the
group keeps every entity wholly on one side.
"""

import pandas as pd
from omegaconf import DictConfig


def train_test_split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned data into the train and test sets. Called once, in notebook 01.

    Args:
        df: Cleaned frame.
        cfg: Composed config; uses ``cfg.split`` (``method``, ``group_col``,
            ``stratify_col``, ``seed``, ``test_size``).

    Returns:
        ``(train_df, test_df)``.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown ``split.method`` or a
            missing grouping column.

    Notes:
        The test split produced here is frozen until notebook 03. Re-running
        this function with the same config and seed must reproduce the same
        partition exactly — that property is what makes the final estimate
        meaningful.

    Example:
        >>> train_df, test_df = train_test_split(clean_df, cfg)
    """
    # TODO(1): dispatch on cfg.split.method
    # TODO(2): for group methods, use sklearn GroupShuffleSplit on cfg.split.group_col
    # TODO(3): pass cfg.split.seed as random_state — never a literal
    # TODO(4): assert_no_group_leakage(train_df, test_df, cfg) before returning
    raise NotImplementedError("src.data.split.train_test_split")


def train_val_split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into train and validation. Called by every ``02x`` notebook.

    Args:
        df: The training split from :func:`train_test_split`.
        cfg: Composed config; uses ``cfg.split`` (``val_size`` in particular).

    Returns:
        ``(train_df, val_df)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``val_size`` is read from ``configs/data.yaml``, never from a model
        config, so every model tunes and early-stops against the same fold.
        ``val_size`` is a fraction of the training split, not of the full
        dataset. Use the same grouping rule as the outer split.

    Example:
        >>> train_df, val_df = train_val_split(load_processed(cfg, "train"), cfg)
    """
    # TODO(1): reuse the same method/group logic as train_test_split
    # TODO(2): apply cfg.split.val_size to the frame passed in
    raise NotImplementedError("src.data.split.train_val_split")


def cv_folds(df: pd.DataFrame, cfg: DictConfig, n_splits: int = 5) -> list[tuple]:
    """Yield group-aware cross-validation folds for hyperparameter search.

    Args:
        df: The training split.
        cfg: Composed config; uses ``cfg.split.group_col`` and ``cfg.split.seed``.
        n_splits: Number of folds.

    Returns:
        A list of ``(train_idx, val_idx)`` index pairs.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Optional, but if any model tunes with cross-validation then all of them
        should use these folds — same reasoning as the single validation split.
    """
    # TODO(1): build GroupKFold (or StratifiedGroupKFold) over cfg.split.group_col
    # TODO(2): return the index pairs so callers stay framework-agnostic
    raise NotImplementedError("src.data.split.cv_folds")


def assert_no_group_leakage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    """Assert that no group appears on both sides of a split.

    Args:
        left: One side of the split.
        right: The other side.
        cfg: Composed config; uses ``cfg.split.group_col``.

    Raises:
        NotImplementedError: Always — implement this module first.
        AssertionError: Once implemented, when the group sets intersect.

    Notes:
        Cheap to run and worth calling after every split, in notebooks as well
        as in code. Group leakage produces results that look good and are
        wrong, which is the most expensive kind of bug in a modelling project.

    Example:
        >>> assert_no_group_leakage(train_df, test_df, cfg)
    """
    # TODO(1): compare the two group sets and raise with the offending values
    raise NotImplementedError("src.data.split.assert_no_group_leakage")
