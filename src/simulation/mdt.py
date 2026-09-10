"""Create synthetic UE reports from radio-map samples.

Adds independent RSRP measurement error. Every cell-band with a path is
reported; NaN means only that the ray tracer found no path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.simulation import seeds

# Each interval is independent, so MDT rows have no UE identifier.
POSITION_COLUMNS = ("t_index", "t_s", "x", "y", "z", "tile_col", "tile_row")


@dataclass(frozen=True)
class MdtSpec:
    """How far a report may stray from the truth.

    Attributes:
        rsrp_noise_sigma_db: Standard deviation of the receiver measurement
            error.
    """

    rsrp_noise_sigma_db: float

    def __post_init__(self) -> None:
        """Reject a negative noise level.

        Raises:
            ValueError: When ``rsrp_noise_sigma_db`` is negative.
        """
        if self.rsrp_noise_sigma_db < 0:
            raise ValueError("simulation.mdt.rsrp_noise_sigma_db must not be negative")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> MdtSpec:
        """Read ``simulation.mdt``."""
        return cls(rsrp_noise_sigma_db=float(cfg.simulation.mdt.rsrp_noise_sigma_db))


def measure(clean: np.ndarray, spec: MdtSpec, seed: int) -> np.ndarray:
    """Add measurement error, returning reported RSRP.

    ``clean`` is ``[n_ue, n_measurement]`` in dBm, NaN where the ray tracer
    found no path; NaN stays NaN.
    """
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, spec.rsrp_noise_sigma_db, size=clean.shape)


def build(cfg: DictConfig) -> Path:
    """Read the radio map and UE table, write the MDT. Returns the output path.

    Raises:
        FileNotFoundError: When an earlier stage has not been run.
        ValueError: When the radio map and the UE table are from different
            scenarios, or the UEs fall outside the map's grid.
    """
    ue_path = Path(cfg.simulation.output.ue_file)
    map_path = Path(cfg.simulation.output.radio_map_file)
    for path, stage in ((ue_path, "scenario"), (map_path, "radio")):
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{stage}` first.")

    ues = pd.read_csv(ue_path)
    with np.load(map_path, allow_pickle=False) as data:
        rsrp = data["rsrp_dbm"]
        tx_names = [str(name) for name in data["tx_name"]]
        band_labels = [str(label) for label in data["band_label"]]
        n_cols = int(data["n_cols"])
        n_rows = int(data["n_rows"])

    col = ues["tile_col"].to_numpy()
    row = ues["tile_row"].to_numpy()
    if col.min() < 0 or col.max() >= n_cols or row.min() < 0 or row.max() >= n_rows:
        raise ValueError(
            f"UE grid tiles span cols {col.min()}..{col.max()} rows {row.min()}..{row.max()}, "
            f"outside the radio map's {n_cols} x {n_rows} grid. The two stages used "
            "different grids."
        )

    # [band, tx, row, col] sampled at each UE's tile -> [ue, band, tx], then
    # flattened so a cell's bands sit next to each other.
    sampled = rsrp[:, :, row, col].transpose(2, 1, 0)
    clean = sampled.reshape(len(ues), -1).astype(np.float64)
    columns = [f"rsrp_{tx}_{band}" for tx in tx_names for band in band_labels]

    reported = measure(clean, MdtSpec.from_config(cfg), seeds.stream(cfg, "mdt"))

    heard = np.isfinite(clean)
    covered = heard.any(axis=1)
    n_no_signal = int((~covered).sum())
    ues = ues.loc[covered].reset_index(drop=True)
    heard = heard[covered]
    reported = reported[covered]

    frame = pd.concat(
        [ues[list(POSITION_COLUMNS)], pd.DataFrame(reported, columns=columns, index=ues.index)],
        axis=1,
    )
    path = Path(cfg.simulation.output.mdt_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.3f", na_rep="")

    n_intervals = int(ues["t_index"].nunique())
    print(
        f"ue:        {len(ues)} rows x {len(columns)} measurements "
        f"({n_no_signal} no-signal UEs dropped)"
    )
    print(
        f"intervals: {n_intervals}, "
        f"{len(ues) / max(n_intervals, 1):.0f} UEs per interval on average"
    )
    print(f"reachable: {heard.mean():6.1%} of measurements had a path")
    print(f"per ue:    {heard.sum(axis=1).min()} to {heard.sum(axis=1).max()} cells reported")
    print(f"mdt:       {path}")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the synthetic MDT. Entry point for ``task simulation:mdt``.

    Example:
        $ task simulation:mdt -- simulation.mdt.rsrp_noise_sigma_db=3.0
    """
    build(cfg)


if __name__ == "__main__":
    main()
