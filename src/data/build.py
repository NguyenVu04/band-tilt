"""Build the two processed tables from the verified artifacts.

``cell.parquet`` is the configuration the radio map was solved at — the
pre-optimization tilt every ``DeltaTilt`` is reported against.
``mdt.parquet`` is the synthetic MDT, typed, with an explicit reported
indicator beside every cell-band's RSRP and SINR.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data import schema
from src.data.load import Artifacts, load_artifacts, save
from src.simulation import transmitter
from src.simulation.mdt import POSITION_COLUMNS
from src.simulation.radio import Band

# int16 covers a 74 x 61 grid with room to spare; float32 holds the 3-decimal
# dBm the MDT was written with. t_s stays float64: at a 604800 s horizon,
# float32 cannot represent whole seconds exactly.
_DTYPES = {
    "t_index": "int32",
    "t_s": "float64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "tile_col": "int16",
    "tile_row": "int16",
}


def build_cells(cfg: DictConfig, artifacts: Artifacts) -> pd.DataFrame:
    """One row per cell-band pair: the decision variable, at its baseline.

    Long rather than wide because the decision variable is per cell-band,
    so one row is one tilt an optimizer may move.

    Returns:
        ``n_cell * n_band`` rows carrying the cell's geometry, the band, the
        baseline tilt and its bounds, the ``max_prb`` limit, and
        ``rsrp_column``/``sinr_column`` — the names of the matching columns in
        ``mdt.parquet``, which is what joins the two files.
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
                    "tilt_baseline_deg": tilt.baseline_deg,
                    "tilt_min_deg": low,
                    "tilt_max_deg": high,
                    "max_prb": cell.max_prb_for(label),
                    "rsrp_column": f"rsrp_{cell.name}_{label}",
                    "sinr_column": f"sinr_{cell.name}_{label}",
                    "scenario_id": artifacts.scenario_id,
                }
            )

    frame = pd.DataFrame(rows)
    for column in ("cell", "node", "band", "scenario_id"):
        frame[column] = frame[column].astype("category")
    return frame


def build_mdt(artifacts: Artifacts) -> pd.DataFrame:
    """The MDT, typed, with a reported indicator beside every measurement.

    An empty ``rsrp_*`` in the interim CSV means the ray tracer found *no
    path*. Carrying the indicator explicitly stops a later reader mistaking the
    gap for a zero.

    Returns:
        One row per UE report: the position columns, ``scenario_id``, the
        ``rsrp_*`` and ``sinr_*`` measurements and the ``reported_*`` flags,
        which cover both since SINR has a path exactly where RSRP does. Sorted
        so the output does not depend on the order the simulator emitted rows.
    """
    measurement = artifacts.measurement_columns
    sinr = artifacts.sinr_columns
    frame = artifacts.mdt.copy()

    reported = frame[measurement].notna()
    reported.columns = [column.replace("rsrp_", "reported_", 1) for column in measurement]

    frame = frame.astype(_DTYPES)
    frame[measurement + sinr] = frame[measurement + sinr].astype("float32")
    frame["scenario_id"] = pd.Categorical([artifacts.scenario_id] * len(frame))

    frame = pd.concat(
        [frame[[*POSITION_COLUMNS, "scenario_id"]], frame[measurement], frame[sinr], reported],
        axis=1,
    )
    return frame.sort_values(
        ["t_index", "tile_row", "tile_col", "x", "y"], kind="stable"
    ).reset_index(drop=True)


def run(cfg: DictConfig) -> tuple[Path, Path]:
    """Load, verify and write both tables. Returns ``(mdt_path, cell_path)``.

    Raises:
        FileNotFoundError: When a simulation stage has not been run.
        SchemaError: When the artifacts violate the contract. Nothing is
            written in that case — a bad artifact must not reach ``processed``.
    """
    artifacts = load_artifacts(cfg)
    checks = schema.verify(artifacts, cfg)
    schema.require(checks)

    mdt = build_mdt(artifacts)
    cells = build_cells(cfg, artifacts)
    mdt_path = save(mdt, cfg.data.output.mdt_file)
    cell_path = save(cells, cfg.data.output.cell_file)

    flags = [column for column in mdt.columns if column.startswith("reported_")]
    print(f"scenario:  {artifacts.scenario_id}")
    print(f"checks:    {len(checks)} passed")
    print(f"mdt:       {len(mdt):,} rows x {mdt.shape[1]} columns  ->  {mdt_path}")
    print(f"reported:  {mdt[flags].to_numpy().mean():.1%} of cell-band pairs")
    print(f"cells:     {len(cells)} cell-band pairs  ->  {cell_path}")
    return mdt_path, cell_path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the processed tables. The script form of ``02_preprocessing.ipynb``.

    Example:
        $ uv run python -m src.data.build data.output.mdt_file=/tmp/mdt.parquet
    """
    run(cfg)


if __name__ == "__main__":
    main()
