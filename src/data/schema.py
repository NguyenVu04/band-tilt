"""Check the simulation artifacts against the contract before anything reads them.

Every check names the source of the bound it enforces: ``configs/simulation.yaml``,
the scenario manifest, or the radio map itself. A bound with no nameable source is
a statistical threshold and does not belong in this contract.

The KPI thresholds are deliberately absent. ``kpi.hole_dbm`` and ``kpi.weak_dbm``
classify a tile; they never disqualify a measurement. The one exception is the MDT:
the serving rule never admits a UE at or below ``kpi.hole_dbm``, so an MDT row there
is a writer fault, not a hole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.load import Artifacts
from src.simulation import transmitter
from src.simulation.mdt import MDT_COLUMNS
from src.simulation.sample import CSV_COLUMNS

_CONFIG = "configs/simulation.yaml"
_MANIFEST = "scenario.json"
_MAP = "radio_map.npz"
_UE = "ue_positions.csv"
_MDT = "mdt.csv"
_KPI = "configs/kpi.yaml"

# The UE table writes t_s with three decimals, so a t_s drawn inside its
# interval round-trips to the millisecond and no closer.
_T_S_TOL = 1e-3


class SchemaError(ValueError):
    """Raised when an artifact violates the contract."""


def verify(artifacts: Artifacts, cfg: DictConfig) -> pd.DataFrame:
    """Run every check and report the outcome, one row each.

    Args:
        artifacts: The loaded artifacts, as read.
        cfg: Composed config; reads ``simulation.ue.height_m``,
            ``simulation.antenna.power_rs``, ``kpi.hole_dbm`` and the cell table.

    Returns:
        A frame of ``check``, ``source``, ``holds`` and ``violations``. Never
        raises on a failed check — :func:`require` does that, so a notebook can
        show the whole table before it stops.
    """
    checks: list[tuple[str, str, bool, int]] = []

    def record(name: str, source: str, holds: bool, violations: int = 0) -> None:
        checks.append((name, source, bool(holds), int(violations)))

    ue = artifacts.ue
    grid = artifacts.manifest["grid"]
    n_rows, n_cols = artifacts.shape
    max_x = grid["origin_x"] + grid["n_cols"] * grid["tile_size_m"]
    max_y = grid["origin_y"] + grid["n_rows"] * grid["tile_size_m"]

    # Structure: do the three files describe the same run?
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
    record("ue columns are the declared set", _CONFIG, list(ue.columns) == list(CSV_COLUMNS))
    record(
        "npz sinr_db has the shape of rsrp_dbm",
        _MAP,
        "sinr_db" in artifacts.radio
        and artifacts.radio["sinr_db"].shape == artifacts.radio["rsrp_dbm"].shape,
    )

    # Rows: bounds whose source is the config or the manifest.
    record(
        "z equals the configured UE height",
        _CONFIG,
        *_count(ue["z"].to_numpy() != float(cfg.simulation.ue.height_m)),
    )
    record(
        "0 <= tile_row < n_rows",
        _MANIFEST,
        *_count((ue["tile_row"] < 0) | (ue["tile_row"] >= n_rows)),
    )
    record(
        "0 <= tile_col < n_cols",
        _MANIFEST,
        *_count((ue["tile_col"] < 0) | (ue["tile_col"] >= n_cols)),
    )
    record(
        "x, y inside the grid extent",
        _MANIFEST,
        *_count(
            (ue["x"] < grid["origin_x"])
            | (ue["x"] > max_x)
            | (ue["y"] < grid["origin_y"])
            | (ue["y"] > max_y)
        ),
    )
    record(
        "radio-map RSRP does not exceed the transmit RS power",
        _CONFIG,
        *_count(
            np.nan_to_num(artifacts.radio["rsrp_dbm"], nan=-np.inf)
            > float(cfg.simulation.antenna.power_rs)
        ),
    )

    schedule = np.asarray(artifacts.manifest["time"]["t_s"], dtype=np.float64)
    t_index = ue["t_index"].to_numpy()
    in_range = (t_index >= 0) & (t_index < len(schedule))
    record("t_index lies inside the schedule", _MANIFEST, *_count(~in_range))
    interval_s = float(artifacts.manifest["time"]["spec"]["interval_s"])
    offset = ue["t_s"].to_numpy() - schedule[np.clip(t_index, 0, len(schedule) - 1)]
    record(
        "t_s lies inside the interval its t_index names",
        _MANIFEST,
        *_count(in_range & ((offset < -_T_S_TOL) | (offset > interval_s + _T_S_TOL))),
    )

    # Positions are continuous draws, so a repeated row is a writer fault.
    record("no duplicate rows", _UE, *_count(ue.duplicated().to_numpy()))

    # The MDT: a served subset of the UE table.
    mdt = artifacts.mdt
    has_columns = list(mdt.columns) == list(MDT_COLUMNS)
    record("mdt columns are the declared set", _CONFIG, has_columns)
    if has_columns:
        rsrp = mdt["rsrp_dbm"].to_numpy(dtype=float)
        # Both files write millimetre precision, so an exact join is the subset test.
        matched = mdt.merge(ue.drop_duplicates(), on=list(CSV_COLUMNS), how="left", indicator=True)
        record("every mdt row is a UE row", _UE, *_count(matched["_merge"] != "both"))
        record(
            "mdt rsrp_dbm is above the hole threshold",
            _KPI,
            *_count(~(rsrp > float(cfg.kpi.hole_dbm))),
        )
        record(
            "mdt rsrp_dbm does not exceed the transmit RS power",
            _CONFIG,
            *_count(rsrp > float(cfg.simulation.antenna.power_rs)),
        )
        # A report is in the MDT because a cell-band scheduled PRBs for it, so
        # a non-positive or non-finite requirement is a writer fault.
        prb = mdt["prb_per_ue"].to_numpy(dtype=float)
        record(
            "mdt prb_per_ue is finite and positive",
            _MDT,
            *_count(~(np.isfinite(prb) & (prb > 0.0))),
        )
        record("no duplicate mdt rows", _MDT, *_count(mdt.duplicated().to_numpy()))

    # The cell table the map was solved at.
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
