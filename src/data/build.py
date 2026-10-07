"""Build the processed UE table from the verified artifacts.

``ue.parquet`` is the UE population, typed, with every UE kept; the search and
the evaluation both score on it. The sector table is not copied: every consumer
reads ``simulation.input.sectors_file`` itself.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.core.ue import OPTIONAL_UE_COLUMNS, UE_COLUMNS
from src.data import schema
from src.data.load import Artifacts, load_artifacts, save
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


def build_ue(artifacts: Artifacts) -> pd.DataFrame:
    """The UE population, typed, with no row dropped.

    A UE no transmitter reaches stays in: the serving rule counts it as not
    served, which is what the service failure rate has to see.

    Returns:
        One row per UE per interval: the contract columns, any optional ones
        present, and ``scenario_id``. Sorted so the output does not depend on
        the order the producer emitted rows.
    """
    columns = [*UE_COLUMNS, *(c for c in OPTIONAL_UE_COLUMNS if c in artifacts.ue.columns)]
    frame = artifacts.ue[columns].astype({column: _DTYPES[column] for column in columns})
    frame["scenario_id"] = pd.Categorical([artifacts.scenario_id] * len(frame))
    return frame.sort_values(
        ["t_index", "tile_row", "tile_col", "x", "y"], kind="stable"
    ).reset_index(drop=True)


def run(cfg: DictConfig) -> Path:
    """Load, verify and write the UE table. Returns its path.

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

    print(f"scenario:  {artifacts.scenario_id}")
    print(f"checks:    {len(checks)} passed")
    print(f"ue:        {len(ue):,} rows x {ue.shape[1]} columns  ->  {ue_path}")
    return ue_path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the processed UE table. The script form of ``02_preprocessing.ipynb``.

    Example:
        $ uv run python -m src.data.build data.output.ue_file=/tmp/ue.parquet
    """
    log_stage(cfg, "preprocessing", groups=["data"], outputs=[run(cfg)])


if __name__ == "__main__":
    main()
