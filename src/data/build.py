"""Build the three processed tables from the verified artifacts.

``cell.parquet`` is the configuration the radio map was solved at — the
pre-optimization tilt every ``DeltaTilt`` is reported against.
``ue.parquet`` is the UE population, typed, with every drawn UE kept; evaluation
scores on it, as the search does. ``mdt.parquet`` is the served subset, kept for
reference and plots; no score reads it.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data import schema
from src.data.load import Artifacts, load_artifacts, save
from src.simulation import transmitter
from src.simulation.mdt import MDT_COLUMNS
from src.simulation.radio import Band
from src.simulation.sample import CSV_COLUMNS
from src.tracking import log_stage

# int16 covers the grid with room to spare; float32 holds the millimetre
# coordinates the UE table is written with. t_s stays float64: over a week-long
# horizon float32's spacing grows coarser than the milliseconds the UE table is
# written with, so late intervals would quantise.
_DTYPES = {
    "t_index": "int32",
    "t_s": "float64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "tile_col": "int16",
    "tile_row": "int16",
    "component": "int16",
}


def build_cells(cfg: DictConfig, artifacts: Artifacts) -> pd.DataFrame:
    """One row per cell-band pair: the decision variable, at its baseline.

    Long rather than wide because the decision variable is per cell-band,
    so one row is one tilt an optimizer may move.

    Returns:
        ``n_cell * n_band`` rows carrying the cell's geometry, the band, the
        baseline tilt and its bounds, the ``max_prb`` limit and the scenario.
    """
    cells = transmitter.load(cfg)
    bands = {str(entry.name): Band.from_config(entry) for entry in cfg.simulation.radio_map.bands}

    rows = []
    for cell in cells:
        for label in artifacts.band_labels:
            tilt = cell.tilt_for(label)
            low, high = tilt.bounds_deg
            rows.append(
                {
                    "cell": cell.name,
                    # Generated names are n<node>c<cell>; the node is the mast
                    # the cell stands on, and co-located cells share it.
                    "node": cell.name.rsplit("c", 1)[0],
                    "x": cell.x,
                    "y": cell.y,
                    "z": cell.z,
                    "azimuth_deg": cell.azimuth_deg,
                    "band": label,
                    "frequency_hz": bands[label].frequency_hz,
                    "bandwidth_hz": bands[label].bandwidth_hz,
                    "scs_hz": bands[label].scs_hz,
                    "tilt_baseline_deg": tilt.baseline_deg,
                    "tilt_min_deg": low,
                    "tilt_max_deg": high,
                    "max_prb": cell.max_prb_for(label),
                    "scenario_id": artifacts.scenario_id,
                }
            )

    frame = pd.DataFrame(rows)
    for column in ("cell", "node", "band", "scenario_id"):
        frame[column] = frame[column].astype("category")
    return frame


def build_ue(artifacts: Artifacts) -> pd.DataFrame:
    """The UE population, typed, with no row dropped.

    A UE no transmitter reaches stays in: the serving rule counts it as not
    served, which is what the served rate has to see.

    Returns:
        One row per UE per interval: the UE table's columns and
        ``scenario_id``. Sorted so the output does not depend on the order the
        simulator emitted rows.
    """
    frame = artifacts.ue[list(CSV_COLUMNS)].astype(_DTYPES)
    frame["scenario_id"] = pd.Categorical([artifacts.scenario_id] * len(frame))
    return frame.sort_values(
        ["t_index", "tile_row", "tile_col", "x", "y"], kind="stable"
    ).reset_index(drop=True)


def build_mdt(artifacts: Artifacts) -> pd.DataFrame:
    """The MDT, typed and ordered as :func:`build_ue`, with no row dropped.

    Returns:
        One row per served UE per interval: :data:`src.simulation.mdt.MDT_COLUMNS`
        and ``scenario_id``.
    """
    frame = artifacts.mdt[list(MDT_COLUMNS)].astype(_DTYPES | {"rsrp_dbm": "float32"})
    frame["scenario_id"] = pd.Categorical([artifacts.scenario_id] * len(frame))
    return frame.sort_values(
        ["t_index", "tile_row", "tile_col", "x", "y"], kind="stable"
    ).reset_index(drop=True)


def run(cfg: DictConfig) -> tuple[Path, Path, Path]:
    """Load, verify and write every table. Returns the three paths written.

    Raises:
        FileNotFoundError: When a simulation stage has not been run.
        SchemaError: When the artifacts violate the contract. Nothing is
            written in that case — a bad artifact must not reach ``processed``.
    """
    artifacts = load_artifacts(cfg)
    checks = schema.verify(artifacts, cfg)
    schema.require(checks)

    ue = build_ue(artifacts)
    mdt = build_mdt(artifacts)
    cells = build_cells(cfg, artifacts)
    ue_path = save(ue, cfg.data.output.ue_file)
    mdt_path = save(mdt, cfg.data.output.mdt_file)
    cell_path = save(cells, cfg.data.output.cell_file)

    print(f"scenario:  {artifacts.scenario_id}")
    print(f"checks:    {len(checks)} passed")
    print(f"ue:        {len(ue):,} rows x {ue.shape[1]} columns  ->  {ue_path}")
    print(f"mdt:       {len(mdt):,} rows x {mdt.shape[1]} columns  ->  {mdt_path}")
    print(f"cells:     {len(cells)} cell-band pairs  ->  {cell_path}")
    return ue_path, mdt_path, cell_path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the processed tables. The script form of ``02_preprocessing.ipynb``.

    Example:
        $ uv run python -m src.data.build data.output.ue_file=/tmp/ue.parquet
    """
    log_stage(cfg, "preprocessing", groups=["data"], outputs=run(cfg))


if __name__ == "__main__":
    main()
