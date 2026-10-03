"""Check the UE table, manifest and radio map against the contract before anything reads them.

Every check names the source of the bound it enforces: ``configs/simulation.yaml``,
the scenario manifest, the cell table, the UE table's contract
(:mod:`src.core.ue`), or the radio map itself. A bound with no nameable source
is a statistical threshold and does not belong in this contract.

The KPI thresholds are deliberately absent. ``kpi.hole_dbm`` and ``kpi.weak_dbm``
classify a tile; they never disqualify a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import read_cells
from src.core.ue import OPTIONAL_UE_COLUMNS, UE_COLUMNS
from src.data.load import Artifacts
from src.simulation.radio import baseline_tilts

_CONFIG = "configs/simulation.yaml"
_MANIFEST = "scenario.json"
_MAP = "radio_map.npz"
_UE = "ue_positions.csv"
_CELLS = "cells.csv"

# The manifest keys any producer must supply, synthetic or measured.
_MANIFEST_KEYS = (
    ("scenario_id",),
    *(("grid", key) for key in ("origin_x", "origin_y", "tile_size_m", "n_cols", "n_rows")),
    ("time", "t_s"),
    ("time", "interval_s"),
)

_INTEGER_COLUMNS = ("t_index", "tile_col", "tile_row", "component")

# The UE table is written to the millimetre, so x, y, t_s and z round-trip to
# 5e-4 and no closer.
_CSV_TOL = 5e-4


class SchemaError(ValueError):
    """Raised when an artifact violates the contract."""


def verify(artifacts: Artifacts, cfg: DictConfig) -> pd.DataFrame:
    """Run every check and report the outcome, one row each.

    Structural checks (manifest keys, UE columns and dtypes, a non-empty UE
    table, a readable cell table) run first. When one fails, the checks that
    would index into the broken structure are skipped rather than crashing, so
    the table always lists what is wrong.

    Args:
        artifacts: The loaded artifacts, as read.
        cfg: Composed config; reads ``simulation.ue.height_m``,
            ``simulation.antenna.power_rs``, ``simulation.radio_map.bands`` and
            ``simulation.input.cells_file``.

    Returns:
        A frame of ``check``, ``source``, ``holds`` and ``violations``. Never
        raises on a failed check — :func:`require` does that, so a notebook can
        show the whole table before it stops.
    """
    checks: list[tuple[str, str, bool, int]] = []

    def record(name: str, source: str, holds: bool, violations: int = 0) -> None:
        checks.append((name, source, bool(holds), int(violations)))

    def table() -> pd.DataFrame:
        return pd.DataFrame(checks, columns=["check", "source", "holds", "violations"])

    # Structure: everything below indexes into these.
    manifest = artifacts.manifest
    missing_keys = [".".join(path) for path in _MANIFEST_KEYS if not _has(manifest, path)]
    record(
        "manifest carries scenario_id, grid and time",
        _MANIFEST,
        not missing_keys,
        len(missing_keys),
    )
    ue = artifacts.ue
    allowed = {*UE_COLUMNS, *OPTIONAL_UE_COLUMNS}
    columns_hold = set(UE_COLUMNS) <= set(ue.columns) <= allowed
    record("ue columns are the contract's", _UE, columns_hold)
    if columns_hold:
        present = [column for column in _INTEGER_COLUMNS if column in ue.columns]
        record(
            "integer ue columns hold integers",
            _UE,
            *_count([not pd.api.types.is_integer_dtype(ue[column]) for column in present]),
        )
        real = [column for column in UE_COLUMNS if column not in _INTEGER_COLUMNS]
        numeric = all(pd.api.types.is_numeric_dtype(ue[column]) for column in real)
        record("real ue columns are numeric", _UE, numeric)
        if numeric:
            record("no missing ue value", _UE, *_count(ue[real].isna().to_numpy()))
    record("ue table holds at least one row", _UE, len(ue) > 0)
    try:
        cells = read_cells(cfg.simulation.input.cells_file)
    except (FileNotFoundError, ValueError):
        cells = ()
    record("cell table reads", _CELLS, bool(cells))
    if not all(holds for _, _, holds, _ in checks):
        return table()

    grid = manifest["grid"]
    n_rows, n_cols = artifacts.shape
    tile = float(grid["tile_size_m"])
    max_x = grid["origin_x"] + grid["n_cols"] * tile
    max_y = grid["origin_y"] + grid["n_rows"] * tile
    bands = list(cfg.simulation.radio_map.bands)
    rsrp, sinr = artifacts.radio["rsrp_dbm"], artifacts.radio.get("sinr_db")

    # Do the files describe the same run?
    record(
        "npz tx_name matches the cell table",
        _CELLS,
        artifacts.tx_names == [cell.name for cell in cells],
    )
    record(
        "npz band_label matches the configured bands",
        _CONFIG,
        artifacts.band_labels == [str(entry.name) for entry in bands],
    )
    record(
        "npz band_hz matches the configured frequencies",
        _CONFIG,
        "band_hz" in artifacts.radio
        and np.array_equal(
            np.asarray(artifacts.radio["band_hz"], dtype=float),
            [float(entry.frequency) for entry in bands],
        ),
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
    sinr_shape = sinr is not None and sinr.shape == rsrp.shape
    record("npz sinr_db has the shape of rsrp_dbm", _MAP, sinr_shape)
    if sinr_shape:
        # The solver defines SINR exactly where a path reached the tile.
        record(
            "npz sinr_db is NaN exactly where rsrp_dbm is",
            _MAP,
            *_count(np.isnan(sinr) != np.isnan(rsrp)),
        )

    # Rows: bounds whose source is the config, the manifest or the contract.
    record(
        "z equals the configured UE height",
        _CONFIG,
        *_count(~np.isclose(ue["z"].to_numpy(), float(cfg.simulation.ue.height_m), atol=_CSV_TOL)),
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
        "tile_col, tile_row are the tile holding x, y",
        _MANIFEST,
        *_count(
            ~_on_tile(ue["x"].to_numpy(float), ue["tile_col"], grid["origin_x"], tile, n_cols)
            | ~_on_tile(ue["y"].to_numpy(float), ue["tile_row"], grid["origin_y"], tile, n_rows)
        ),
    )
    record(
        "radio-map RSRP does not exceed the transmit RS power",
        _CONFIG,
        *_count(np.nan_to_num(rsrp, nan=-np.inf) > float(cfg.simulation.antenna.power_rs)),
    )

    schedule = np.asarray(manifest["time"]["t_s"], dtype=np.float64)
    t_index = ue["t_index"].to_numpy()
    in_range = (t_index >= 0) & (t_index < len(schedule))
    record("t_index lies inside the schedule", _MANIFEST, *_count(~in_range))
    interval_s = float(manifest["time"]["interval_s"])
    offset = ue["t_s"].to_numpy() - schedule[np.clip(t_index, 0, len(schedule) - 1)]
    record(
        "t_s lies inside the interval its t_index names",
        _MANIFEST,
        *_count(in_range & ((offset < -_CSV_TOL) | (offset > interval_s + _CSV_TOL))),
    )

    # Positions are continuous draws, so a repeated row is a writer fault.
    record("no duplicate rows", _UE, *_count(ue.duplicated().to_numpy()))

    # The cell table the map was solved at.
    tilts_match = False
    if "tilt_deg" in artifacts.radio and artifacts.tx_names == [cell.name for cell in cells]:
        tilt_deg = np.asarray(artifacts.radio["tilt_deg"], dtype=np.float64)
        try:
            configured = baseline_tilts(cells, artifacts.band_labels)
        except KeyError:
            configured = np.empty(0)
        tilts_match = tilt_deg.shape == configured.shape and bool(np.allclose(tilt_deg, configured))
    record("npz tilt_deg equals the cell table's baseline tilts", _CELLS, tilts_match)

    return table()


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
        "The artifacts and the config disagree. Regenerate or re-supply them."
    )


def _has(mapping: dict, path: tuple[str, ...]) -> bool:
    """Whether the nested ``path`` exists in ``mapping``."""
    node = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _on_tile(
    coordinate: np.ndarray, index: pd.Series, origin: float, tile: float, n_tiles: int
) -> np.ndarray:
    """Whether ``index`` is the tile holding ``coordinate`` to the UE table's precision.

    A coordinate rounded onto a tile edge may name either neighbour, so both
    are accepted; the edge of the grid clips inward, as the generator does.
    """
    low = np.clip(np.floor((coordinate - _CSV_TOL - origin) / tile), 0, n_tiles - 1)
    high = np.clip(np.floor((coordinate + _CSV_TOL - origin) / tile), 0, n_tiles - 1)
    index = index.to_numpy()
    return (low <= index) & (index <= high)


def _count(mask: object) -> tuple[bool, int]:
    """Turn a violation mask into a ``(holds, violations)`` pair."""
    violations = int(np.asarray(mask).sum())
    return violations == 0, violations
