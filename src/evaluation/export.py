"""Write result tables to disk, so a number can be read without rerunning a notebook.

The counterpart to :func:`src.utils.plotting.save_fig`, with the same signature
and the same Colab rule, so a notebook binds both the same way::

    save_table = partial(save_table, in_colab=IN_COLAB, directory=...)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Beside reports/figures, and ignored by git the same way.
TABLES_DIR = Path("reports/tables")


def save_table(
    frame: pd.DataFrame,
    name: str,
    in_colab: bool,
    directory: str | Path = TABLES_DIR,
) -> Path | None:
    """Write ``frame`` to ``directory/name.csv``. Returns the path, or None on Colab.

    Skipped on Colab for the same reason :func:`src.utils.plotting.save_fig`
    skips there: ``/content`` is a fresh clone that does not survive a runtime
    reset, so the file would be written and then silently lost.

    The index is dropped, matching :func:`src.data.load.save` — every table here
    carries its keys in columns.
    """
    if in_colab:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path
