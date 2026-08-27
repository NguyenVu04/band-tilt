"""Schema and constraint validation.

``configs/data.yaml`` declares *what* is true about the data: columns, dtypes,
nullability, allowed categories, and hard bounds. This module holds the logic
that *enforces* those declarations.

Keep the split of responsibilities: when a constraint needs cross-column or
conditional logic ("every ``gcell_id`` in the MDT export has a row in the cell
configuration"), express it here in Python — do not grow the YAML into a rule
engine.

Two schemas, one validator
--------------------------
``configs/data.yaml`` declares two contracts, ``schema.mdt`` and
``schema.cell_config``, because the two inputs are different shapes. Both are
checked by the same functions; the caller says which contract applies. The cell
configuration must currently be validated with ``strict=False`` — the
multi-band columns PROJECT.md section 4.2 requires are declared but not yet
present in the export (docs/adr/0005).

What must NOT go here
---------------------
Any bound derived from the data itself. Every threshold checked in this module
must come from an external source — a standard, a specification, a physical
limit — and that source must be recorded in ``configs/data.yaml``. A threshold
computed from the dataset leaks test-set information into notebook 01.

The RSRP bound is the worked example: ``[-156, -31]`` dBm is the 3GPP TS 38.133
reporting range, not a quantile of the observed values. The raw export contains
``rsrp = 0.0``, which that bound rejects because no receiver reports it, not
because it looked unusual.
"""

from typing import Literal

import pandas as pd
from omegaconf import DictConfig

#: Which of the two declarations in ``configs/data.yaml`` to validate against.
Contract = Literal["mdt", "cell_config"]


class SchemaError(ValueError):
    """Raised when the data violates the contract in ``configs/data.yaml``."""


def validate(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    contract: Contract = "mdt",
    strict: bool = True,
) -> pd.DataFrame:
    """Check a frame against one of the declared schemas.

    Args:
        df: Frame to validate.
        cfg: Composed config; uses ``cfg.data.schema[contract].columns``.
        contract: Which declaration to check against — ``"mdt"`` or
            ``"cell_config"``.
        strict: Raise on the first violation. When ``False``, collect every
            violation and report them together — friendlier during EDA, and
            currently required for ``"cell_config"``.

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

        Skip any column whose declared name is still a ``<placeholder>``. Those
        entries document a column that has not arrived yet; treating them as
        required would fail every load until the multi-band export lands.

    Example:
        >>> mdt = validate(load_mdt(cfg), cfg, contract="mdt")
        >>> cells = validate(load_cell_config(cfg), cfg, contract="cell_config", strict=False)
    """
    # TODO(1): select cfg.data.schema[contract].columns, skipping "<placeholder>" names
    # TODO(2): check every declared column is present
    # TODO(3): coerce/verify dtypes
    # TODO(4): check nullability
    # TODO(5): check min/max bounds and allowed category sets
    # TODO(6): cross-column rule — every mdt.gcell_id appears in the cell config
    raise NotImplementedError("src.data.schema.validate")


def find_violations(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    contract: Contract = "mdt",
) -> pd.DataFrame:
    """Report which records violate which hard constraints.

    Args:
        df: Frame to inspect.
        cfg: Composed config; uses ``cfg.data.schema[contract].columns``.
        contract: Which declaration to check against.

    Returns:
        One row per violating record, with the column and rule that failed.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Used by notebook 00 to size the problem before any filtering, and by
        notebook 01's cleaning audit to document exactly what was dropped and
        why. Returning a frame rather than raising keeps it usable in analysis.

    Example:
        >>> bad = find_violations(mdt, cfg)
        >>> bad.groupby("rule").size()
    """
    # TODO(1): evaluate each declared constraint into a boolean mask
    # TODO(2): return the violating rows annotated with column + rule
    raise NotImplementedError("src.data.schema.find_violations")
