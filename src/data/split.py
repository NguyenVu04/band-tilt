"""Shared train/test splitting — the single authoritative implementation.

Every notebook, script and test obtains its splits from this module, using the
scheme and seed declared in ``configs/data.yaml``. Nothing else in the project
may split data.

Why MDT cannot be split randomly
--------------------------------
MDT records are correlated along two axes at once, and a random split ignores
both (PROJECT.md section 19 Step 3):

*Spatially.* Consecutive points on one trajectory are metres apart. Two
neighbouring points see almost the same propagation environment, so a random
split puts a record in test whose near-twin is in train. The model is then
scored on something it has effectively already seen.

*Temporally.* One UE contributes a burst of measurements over one session. The
network configuration is constant within that burst, so the split leaks the
configuration as much as the position.

The available methods trade off which generalisation is being measured:
``spatial_block`` holds out whole regions and answers whether results transfer
to places with no measurements; ``temporal`` holds out the tail of the
observation window and answers whether they still hold next week. They are
different questions and they give different numbers — say which one a reported
result used.

``random`` is provided only so that the size of the leakage can be demonstrated
against a defensible baseline. It must never back a reported result.
"""

import pandas as pd
from omegaconf import DictConfig


def train_test_split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned MDT into the train and test sets. Called once, in notebook 01.

    Args:
        df: Cleaned MDT frame.
        cfg: Composed config; uses ``cfg.data.split`` (``method``, ``group_col``,
            ``block_size_m``, ``cutoff_date``, ``seed``, ``test_size``).

    Returns:
        ``(train_df, test_df)``.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, for an unknown ``split.method`` or a
            missing grouping column.

    Notes:
        The test split produced here is frozen until notebook 06. Re-running
        this function with the same config and seed must reproduce the same
        partition exactly — that property is what makes the final estimate
        meaningful.

    Example:
        >>> train_df, test_df = train_test_split(clean_df, cfg)
    """
    # TODO(1): dispatch on cfg.data.split.method
    # TODO(2): spatial_block -> bucket records into block_size_m squares, hold out buckets
    # TODO(3): temporal      -> order by date, hold out the tail (or cutoff_date)
    # TODO(4): group_shuffle -> sklearn GroupShuffleSplit on cfg.data.split.group_col
    # TODO(5): pass cfg.data.split.seed as random_state — never a literal
    # TODO(6): assert_no_leakage(train_df, test_df, cfg) before returning
    raise NotImplementedError("src.data.split.train_test_split")


def train_val_split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into train and validation.

    Args:
        df: The training split from :func:`train_test_split`.
        cfg: Composed config; uses ``cfg.data.split`` (``val_size`` in
            particular).

    Returns:
        ``(train_df, val_df)``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        ``val_size`` is read from ``configs/data.yaml``, never from a surrogate
        or optimizer config, so every experiment tunes against the same fold.
        ``val_size`` is a fraction of the training split, not of the full
        dataset. Use the same method as the outer split — validating on a random
        fold of a spatially split training set measures the wrong thing.

    Example:
        >>> train_df, val_df = train_val_split(load_processed(cfg, "train"), cfg)
    """
    # TODO(1): reuse the same method/group logic as train_test_split
    # TODO(2): apply cfg.data.split.val_size to the frame passed in
    raise NotImplementedError("src.data.split.train_val_split")


def assert_no_leakage(left: pd.DataFrame, right: pd.DataFrame, cfg: DictConfig) -> None:
    """Assert that the two sides of a split share no group, block or time window.

    Args:
        left: One side of the split.
        right: The other side.
        cfg: Composed config; uses ``cfg.data.split``.

    Raises:
        NotImplementedError: Always — implement this module first.
        AssertionError: Once implemented, when the two sides overlap under the
            active split method.

    Notes:
        What counts as leakage depends on the method, so check the one that is
        configured: shared ``group_col`` values for ``group_shuffle``, shared
        spatial blocks for ``spatial_block``, an overlapping time range for
        ``temporal``.

        Cheap to run and worth calling after every split, in notebooks as well
        as in code. Leakage produces results that look good and are wrong, which
        is the most expensive kind of bug in a modelling project.

    Example:
        >>> assert_no_leakage(train_df, test_df, cfg)
    """
    # TODO(1): dispatch on cfg.data.split.method
    # TODO(2): compute the two key sets (group ids, block ids, or date ranges)
    # TODO(3): raise naming the offending values, not just the count
    raise NotImplementedError("src.data.split.assert_no_leakage")
