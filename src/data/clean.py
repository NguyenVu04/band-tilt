"""Model-agnostic cleaning — the code behind notebook 01.

This module is the most rule-bound part of the project, because everything it
does happens *before* the train/test split and therefore affects both sides.

Allowed here (deterministic, model-agnostic)
--------------------------------------------
- Schema and dtype validation
- Duplicate record removal
- Dropping non-predictive columns (record IDs and the like)
- Dropping records with a missing or invalid label
- Dropping records that violate a **hard, externally known constraint** — a
  physical limit, a specification range, a bound fixed by a standard

Forbidden here (learns from the data — belongs in a ``02x`` notebook)
---------------------------------------------------------------------
- Imputation of missing values
- Statistical outlier detection (IQR, z-score, isolation forest)
- Scaling, encoding, any fitted transformation
- Feature engineering and selection

The distinction is the *origin of the threshold*, not its effect: a bound known
before seeing the data belongs here; a bound computed from the data leaks
test-set information into the filtering decision and belongs in ``02x``.

Running as a script
-------------------
``python -m src.data.clean`` executes the full stage — load, validate, clean,
split, write — and is the command wired into the ``clean_split`` stage of
``dvc.yaml``.
"""

import hydra
import pandas as pd
from omegaconf import DictConfig


def drop_duplicates(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Remove duplicate records.

    Args:
        df: Frame to deduplicate.
        cfg: Composed config, for the subset of columns that defines identity.

    Returns:
        The frame without duplicates.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Decide explicitly what "duplicate" means for this dataset: fully
        identical rows, or rows identical on a natural key. Document the choice
        here — it is a modelling decision disguised as cleaning.
    """
    # TODO(1): define the identity subset (full row, or a natural key)
    # TODO(2): drop and report how many records were removed
    raise NotImplementedError("src.data.clean.drop_duplicates")


def drop_non_predictive(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Drop columns that must never reach a model.

    Args:
        df: Frame to prune.
        cfg: Composed config; uses ``cfg.drop_columns``.

    Returns:
        The frame without the dropped columns.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Record identifiers are the usual case: they let a model memorise rows
        and often correlate with collection order, which is leakage in
        disguise. The grouping column is a special case — it is needed by the
        splitter, so drop it after splitting, not here.
    """
    # TODO(1): drop cfg.drop_columns, tolerating already-absent columns
    raise NotImplementedError("src.data.clean.drop_non_predictive")


def drop_invalid_labels(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Drop records whose label is missing or invalid.

    Args:
        df: Frame to filter.
        cfg: Composed config; uses ``cfg.target``.

    Returns:
        The frame with only usable labels.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Unlabelled records cannot be trained or scored on, so dropping them is
        model-agnostic and safe before the split. Report the count: a large
        share of missing labels is a data-collection problem worth surfacing,
        not something to silently discard.
    """
    # TODO(1): drop rows where cfg.target is null
    # TODO(2): drop rows whose label falls outside the declared valid range/set
    raise NotImplementedError("src.data.clean.drop_invalid_labels")


def drop_constraint_violations(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Drop records violating hard, externally defined constraints.

    Args:
        df: Frame to filter.
        cfg: Composed config; uses the ``min`` / ``max`` / ``allowed`` entries
            of ``cfg.schema.columns``.

    Returns:
        The frame containing only physically valid records.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Every bound applied here must have a ``source`` recorded next to it in
        ``configs/data.yaml``. If you cannot name the standard, specification
        or physical law behind a threshold, it is a statistical threshold and
        belongs in a ``02x`` notebook instead.
    """
    # TODO(1): build the mask from schema bounds via src.data.schema
    # TODO(2): log the dropped count per constraint for the cleaning audit
    raise NotImplementedError("src.data.clean.drop_constraint_violations")


def clean(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run the full cleaning sequence in the required order.

    Args:
        df: Raw frame.
        cfg: Composed config.

    Returns:
        The cleaned frame, ready to split.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Order matters: validate, then deduplicate, then drop invalid labels,
        then apply hard constraints, and only then drop non-predictive
        columns — the identifier columns are often what deduplication needs.

    Example:
        >>> clean_df = clean(load_raw(cfg), cfg)
    """
    # TODO(1): schema.validate(df, cfg)
    # TODO(2): drop_duplicates -> drop_invalid_labels -> drop_constraint_violations
    # TODO(3): drop_non_predictive
    # TODO(4): schema.validate again, to prove the cleaning worked
    raise NotImplementedError("src.data.clean.clean")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the cleaning and splitting stage end to end.

    This is the command behind the ``clean_split`` stage in ``dvc.yaml`` and
    behind ``task clean:data``. It is the scripted equivalent of notebook 01;
    keep the two in step, with this module as the source of truth.

    Args:
        cfg: Config composed by Hydra from ``configs/config.yaml``.

    Raises:
        NotImplementedError: Always — implement this module first.
    """
    # TODO(1): set_seed(cfg.seed)
    # TODO(2): df = load_raw(cfg); df = clean(df, cfg)
    # TODO(3): train_df, test_df = split.train_test_split(df, cfg)
    # TODO(4): save_processed(train_df, cfg.train_path) and the test split
    # TODO(5): print the row counts so the DVC stage log is self-documenting
    raise NotImplementedError("src.data.clean.main")


if __name__ == "__main__":
    main()
