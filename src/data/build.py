"""Build the processed UE and cell tables from the verified artifacts.

``ue.parquet`` is the UE population, typed, with every drawn UE kept; the search
and the evaluation both score on it. ``cells.parquet`` is the cell table, typed;
the search, the capacity model and the evaluation read the cells from it.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import CELL_COLUMNS
from src.data import schema
from src.data.load import Artifacts, load_artifacts, save
from src.scenario.sample import CSV_COLUMNS
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


# Names stay categorical; the PRB limit is a count.
_CELL_DTYPES = {
    "node": "category",
    "node_x": "float64",
    "node_y": "float64",
    "node_z": "float64",
    "cell": "category",
    "azimuth_deg": "float64",
    "band": "category",
    "tilt_deg": "float64",
    "tilt_min_deg": "float64",
    "tilt_max_deg": "float64",
    "max_prb": "int32",
}


def build_cells(artifacts: Artifacts) -> pd.DataFrame:
    """The cell table, typed, in the order the scenario wrote it: the radio map's tx order."""
    return artifacts.cells[list(CELL_COLUMNS)].astype(_CELL_DTYPES)


def build_ue(artifacts: Artifacts) -> pd.DataFrame:
    """The UE population, typed, with no row dropped.

    A UE no transmitter reaches stays in: the serving rule counts it as not
    served, which is what the service failure rate has to see.

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


def run(cfg: DictConfig) -> tuple[Path, Path]:
    """Load, verify and write the UE and cell tables. Returns ``(ue_path, cells_path)``.

    Raises:
        FileNotFoundError: When a simulation stage has not been run.
        SchemaError: When the artifacts violate the contract. Nothing is
            written in that case — a bad artifact must not reach ``processed``.
    """
    artifacts = load_artifacts(cfg)
    checks = schema.verify(artifacts, cfg)
    schema.require(checks)

    ue = build_ue(artifacts)
    ue_path = save(ue, cfg.data.output.ue_file)
    cells = build_cells(artifacts)
    cells_path = save(cells, cfg.data.output.cells_file)

    print(f"scenario:  {artifacts.scenario_id}")
    print(f"checks:    {len(checks)} passed")
    print(f"ue:        {len(ue):,} rows x {ue.shape[1]} columns  ->  {ue_path}")
    print(f"cells:     {len(cells):,} cell-band rows  ->  {cells_path}")
    return ue_path, cells_path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the processed tables. The script form of ``02_preprocessing.ipynb``.

    Example:
        $ uv run python -m src.data.build data.output.ue_file=/tmp/ue.parquet
    """
    log_stage(cfg, "preprocessing", groups=["data"], outputs=list(run(cfg)))


if __name__ == "__main__":
    main()
