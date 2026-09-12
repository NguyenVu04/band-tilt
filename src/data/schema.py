"""Check the simulation artifacts against the contract before anything reads them.

Every check names the source of the bound it enforces: ``configs/simulation.yaml``,
the scenario manifest, or the radio map itself. A bound with no nameable source is
a statistical threshold and belongs in a ``02x`` notebook, fitted on train only
(``notebooks/01_eda.ipynb`` section 11).

The KPI thresholds are deliberately absent. ``kpi.hole_dbm`` and ``kpi.weak_dbm``
classify a tile; they never disqualify a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.load import Artifacts
from src.simulation import transmitter
from src.simulation.mdt import POSITION_COLUMNS

_CONFIG = "configs/simulation.yaml"
_MANIFEST = "scenario.json"
_MAP = "radio_map.npz"

# The MDT is written with float_format="%.3f", so t_s round-trips to the
# millisecond and no closer.
_T_S_TOL = 1e-3


class SchemaError(ValueError):
    """Raised when an artifact violates the contract."""


def verify(artifacts: Artifacts, cfg: DictConfig) -> pd.DataFrame:
    """Run every check and report the outcome, one row each.

    Args:
        artifacts: The loaded artifacts, as read.
        cfg: Composed config; reads ``simulation.ue.height_m``,
            ``simulation.antenna.power_rs`` and the cell table.

    Returns:
        A frame of ``check``, ``source``, ``holds`` and ``violations``. Never
        raises on a failed check — :func:`require` does that, so a notebook can
        show the whole table before it stops.
    """
    checks: list[tuple[str, str, bool, int]] = []

    def record(name: str, source: str, holds: bool, violations: int = 0) -> None:
        checks.append((name, source, bool(holds), int(violations)))

    mdt = artifacts.mdt
    grid = artifacts.manifest["grid"]
    n_rows, n_cols = artifacts.shape
    max_x = grid["origin_x"] + grid["n_cols"] * grid["tile_size_m"]
    max_y = grid["origin_y"] + grid["n_rows"] * grid["tile_size_m"]
    position = [column for column in mdt.columns if not column.startswith("rsrp_")]
    measurement = [column for column in mdt.columns if column.startswith("rsrp_")]

    # --- structure: do the three files describe the same run? ---------------
    cells = transmitter.load(cfg)
    record(
        "npz tx_name matches the configured cells",
        _CONFIG,
        artifacts.tx_names == [cell.name for cell in cells],
    )
    record(
        "npz band_label matches the configured bands",
        _CONFIG,
        artifacts.band_labels == [str(entry.name) for entry in cfg.simulation.radio_map.bands],
    )
    record(
        "npz scenario_id matches the manifest",
        _MANIFEST,
        str(artifacts.radio["scenario_id"]) == artifacts.scenario_id,
    )
    record(
        "npz grid matches the manifest grid",
        _MANIFEST,
        all(
            float(artifacts.radio[key]) == float(grid[key])
            for key in ("origin_x", "origin_y", "tile_size_m", "n_cols", "n_rows")
        ),
    )
    record(
        "npz ue_height_m matches the config",
        _CONFIG,
        float(artifacts.radio["ue_height_m"]) == float(cfg.simulation.ue.height_m),
    )
    record("mdt position columns are the declared set", _CONFIG, position == list(POSITION_COLUMNS))
    record(
        "mdt measurement columns are cell x band, in order",
        _MAP,
        measurement == artifacts.measurement_columns,
    )

    # --- rows: bounds whose source is the config or the manifest ------------
    values = mdt[measurement].to_numpy()
    finite = np.isfinite(values)

    record(
        "z equals the configured UE height",
        _CONFIG,
        *_count(mdt["z"].to_numpy() != float(cfg.simulation.ue.height_m)),
    )
    record(
        "0 <= tile_row < n_rows",
        _MANIFEST,
        *_count((mdt["tile_row"] < 0) | (mdt["tile_row"] >= n_rows)),
    )
    record(
        "0 <= tile_col < n_cols",
        _MANIFEST,
        *_count((mdt["tile_col"] < 0) | (mdt["tile_col"] >= n_cols)),
    )
    record(
        "x, y inside the grid extent",
        _MANIFEST,
        *_count(
            (mdt["x"] < grid["origin_x"])
            | (mdt["x"] > max_x)
            | (mdt["y"] < grid["origin_y"])
            | (mdt["y"] > max_y)
        ),
    )
    record(
        "reported RSRP does not exceed the transmit RS power",
        _CONFIG,
        *_count(finite & (values > float(cfg.simulation.antenna.power_rs))),
    )

    schedule = np.asarray(artifacts.manifest["time"]["t_s"], dtype=np.float64)
    t_index = mdt["t_index"].to_numpy()
    in_range = (t_index >= 0) & (t_index < len(schedule))
    record("t_index lies inside the schedule", _MANIFEST, *_count(~in_range))
    record(
        "t_s matches the schedule for its t_index",
        _MANIFEST,
        *_count(
            in_range
            & ~np.isclose(
                mdt["t_s"].to_numpy(),
                schedule[np.clip(t_index, 0, len(schedule) - 1)],
                atol=_T_S_TOL,
            )
        ),
    )

    # Section 11 names (t_index, x, y) as the join key back to the UE table, so
    # a duplicate there would make the join ambiguous, not merely redundant.
    record("no duplicate rows", _MAP, *_count(mdt.duplicated().to_numpy()))
    record(
        "no duplicate (t_index, x, y) join keys",
        _MAP,
        *_count(mdt.duplicated(subset=["t_index", "x", "y"]).to_numpy()),
    )
    record("every row reports at least one measurement", _MAP, *_count(~finite.any(axis=1)))

    # --- the cell table the map was solved at -------------------------------
    tilt_deg = np.asarray(artifacts.radio["tilt_deg"], dtype=np.float64)
    configured = np.array(
        [[cell.tilt_for(band).baseline_deg for cell in cells] for band in artifacts.band_labels]
    )
    record(
        "npz tilt_deg equals the configured baseline tilts",
        _CONFIG,
        tilt_deg.shape == configured.shape and bool(np.allclose(tilt_deg, configured)),
    )
    record(
        "every cell carries a tilt for every band",
        _CONFIG,
        all(band in cell.tilt for cell in cells for band in artifacts.band_labels),
    )

    return pd.DataFrame(checks, columns=["check", "source", "holds", "violations"])


def require(checks: pd.DataFrame) -> None:
    """Raise unless every check held.

    Raises:
        SchemaError: Listing each failed check and its violation count.
    """
    failed = checks[~checks["holds"]]
    if failed.empty:
        return
    # A structural check is boolean and carries no count, so reporting "0
    # violations" against it would read as a contradiction.
    lines = "\n".join(
        f"  - {row.check} ({row.source})" + (f": {row.violations} rows" if row.violations else "")
        for row in failed.itertuples()
    )
    raise SchemaError(
        f"{len(failed)} of {len(checks)} schema checks failed:\n{lines}\n"
        "The artifacts and the config disagree. Regenerate with `task simulation`."
    )


def _count(mask: object) -> tuple[bool, int]:
    """Turn a violation mask into a ``(holds, violations)`` pair."""
    violations = int(np.asarray(mask).sum())
    return violations == 0, violations
