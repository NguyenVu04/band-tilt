"""Schema and constraint validation.

``configs/data.yaml`` declares *what* is true about the data: columns, dtypes,
nullability, allowed categories, and hard bounds. This module holds the logic
that *enforces* those declarations.

Keep the split of responsibilities: when a constraint needs cross-column or
conditional logic ("``altitude`` may exceed X only when ``mode == 'flight'``"),
express it here in Python — do not grow the YAML into a rule engine.

What must NOT go here
---------------------
Any bound derived from the data itself. Every threshold checked in this module
must come from an external source — a standard, a specification, a physical
limit — and that source must be recorded in ``configs/data.yaml``. A threshold
computed from the dataset leaks test-set information into notebook 01.
"""

import pandas as pd
from omegaconf import DictConfig


class SchemaError(ValueError):
    """Raised when the data violates the contract in ``configs/data.yaml``."""


def validate(df: pd.DataFrame, cfg: DictConfig, *, strict: bool = True) -> pd.DataFrame:
    """Check a frame against the declared schema.

    Args:
        df: Frame to validate.
        cfg: Composed config; uses ``cfg.schema.columns``.
        strict: Raise on the first violation. When ``False``, collect every
            violation and report them together — friendlier during EDA.

    Returns:
        The same frame, with declared dtypes applied.

    Raises:
        NotImplementedError: Always — implement this module first.
        SchemaError: Once implemented, when a declared column is missing, has
            the wrong dtype, contains nulls where ``nullable: false``, or holds
            a value outside its declared bounds or allowed set.

    Notes:
        Validation runs twice in the pipeline: on the raw frame at the start of
        notebook 01, and on the cleaned frame just before the split. The second
        pass is what proves the cleaning code actually did its job.

    Example:
        >>> df = validate(load_raw(cfg), cfg)
    """
    # TODO(1): check every declared column is present
    # TODO(2): coerce/verify dtypes
    # TODO(3): check nullability
    # TODO(4): check min/max bounds and allowed category sets
    # TODO(5): add project-specific cross-column rules here
    raise NotImplementedError("src.data.schema.validate")


def find_violations(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Report which records violate which hard constraints.

    Args:
        df: Frame to inspect.
        cfg: Composed config; uses ``cfg.schema.columns``.

    Returns:
        One row per violating record, with the column and rule that failed.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Used by notebook 00 to size the problem before any filtering, and by
        notebook 01's cleaning audit to document exactly what was dropped and
        why. Returning a frame rather than raising keeps it usable in analysis.
    """
    # TODO(1): evaluate each declared constraint into a boolean mask
    # TODO(2): return the violating rows annotated with column + rule
    raise NotImplementedError("src.data.schema.find_violations")
