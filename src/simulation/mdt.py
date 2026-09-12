"""Create synthetic UE reports and the PRB demand map from radio-map samples.

Adds RSRP measurement error. Every cell-band with a path is reported; NaN means
only that the ray tracer found no path. Each UE is then served from what it
reported, by :mod:`src.kpi.capacity`, and the PRBs it needs there - at its first
choice when blocked - make up the demand map.

SINR is not reported. The serving rule recomputes it from the reported RSRP with
:func:`src.kpi.capacity.sinr_db`, the one definition in the project, so the
measurement error reaches PRB demand once - through RSRP - rather than twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi import capacity
from src.simulation import seeds
from src.tracking import log_stage

# Each interval is independent, so MDT rows have no UE identifier.
POSITION_COLUMNS = ("t_index", "t_s", "x", "y", "z", "tile_col", "tile_row")


@dataclass(frozen=True)
class MdtSpec:
    """How far a report may stray from the truth.

    Attributes:
        rsrp_noise_sigma_db: Standard deviation of the RSRP measurement error.
    """

    rsrp_noise_sigma_db: float

    def __post_init__(self) -> None:
        """Reject a negative noise level.

        Raises:
            ValueError: When the sigma is negative.
        """
        if self.rsrp_noise_sigma_db < 0:
            raise ValueError("simulation.mdt.rsrp_noise_sigma_db must not be negative")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> MdtSpec:
        """Read ``simulation.mdt``."""
        return cls(rsrp_noise_sigma_db=float(cfg.simulation.mdt.rsrp_noise_sigma_db))


def measure(clean: np.ndarray, sigma_db: float, seed: int) -> np.ndarray:
    """Add Gaussian measurement error in dB, returning the reported values.

    ``clean`` is ``[n_ue, n_measurement]``, NaN where the ray tracer found no
    path; NaN stays NaN.
    """
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, sigma_db, size=clean.shape)


def build(cfg: DictConfig) -> Path:
    """Read the radio map and UE table, write the MDT and the demand map.

    Returns the MDT path. Also writes ``simulation.output.demand_map_file``:
    ``prb_required`` ``[n_t, n_rows, n_cols]``, aligned to ``t_index``.

    Raises:
        FileNotFoundError: When an earlier stage has not been run.
        ValueError: When the radio map and the UE table are from different
            scenarios, the UEs fall outside the map's grid, or the config
            does not cover the map's bands and cells.
    """
    ue_path = Path(cfg.simulation.output.ue_file)
    map_path = Path(cfg.simulation.output.radio_map_file)
    for path, stage in ((ue_path, "scenario"), (map_path, "radio")):
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{stage}` first.")

    ues = pd.read_csv(ue_path)
    with np.load(map_path, allow_pickle=False) as data:
        rsrp = data["rsrp_dbm"]
        scenario = str(data["scenario_id"])
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

    # [band, tx, row, col] sampled at each UE's tile -> [ue, tx, band], then
    # flattened so a cell's bands sit next to each other.
    n_band, n_tx = rsrp.shape[:2]
    clean = rsrp[:, :, row, col].transpose(2, 1, 0).reshape(len(ues), -1).astype(np.float64)
    pairs = [f"{tx}_{band}" for tx in tx_names for band in band_labels]

    spec = MdtSpec.from_config(cfg)
    reported = measure(clean, spec.rsrp_noise_sigma_db, seeds.stream(cfg, "mdt"))

    heard = np.isfinite(clean)
    covered = heard.any(axis=1)
    n_no_signal = int((~covered).sum())
    ues = ues.loc[covered].reset_index(drop=True)
    heard = heard[covered]
    reported = reported[covered]

    frame = pd.concat(
        [
            ues[list(POSITION_COLUMNS)],
            pd.DataFrame(reported, columns=[f"rsrp_{p}" for p in pairs], index=ues.index),
        ],
        axis=1,
    )
    path = Path(cfg.simulation.output.mdt_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.3f", na_rep="")

    # Serve from the reports, back in [ue, band, tx] for the capacity rule.
    t_index = ues["t_index"].to_numpy()
    spec_capacity = capacity.CapacitySpec.from_config(cfg, band_labels, n_tx)
    rsrp_ue = reported.reshape(-1, n_tx, n_band).transpose(0, 2, 1)
    # sinr_db sums interference over the tx axis, so it takes [band, tx, ue].
    sinr_map = capacity.sinr_db(rsrp_ue.transpose(1, 2, 0), spec_capacity.noise_dbm)
    band, _tx, prb_per_ue = capacity.serve_rows(
        rsrp_ue, sinr_map.transpose(2, 0, 1), t_index, spec_capacity
    )
    t_values, prb = capacity.prb_by_interval(
        t_index,
        ues["tile_row"].to_numpy(),
        ues["tile_col"].to_numpy(),
        prb_per_ue,
        (n_rows, n_cols),
    )
    demand_path = Path(cfg.simulation.output.demand_map_file)
    demand_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        demand_path,
        prb_required=prb.astype(np.float32),
        t_index=t_values,
        scenario_id=scenario,
    )

    n_intervals = int(ues["t_index"].nunique())
    print(
        f"ue:        {len(ues)} rows x {len(pairs)} cell-bands of RSRP "
        f"({n_no_signal} no-signal UEs dropped)"
    )
    print(
        f"intervals: {n_intervals}, "
        f"{len(ues) / max(n_intervals, 1):.0f} UEs per interval on average"
    )
    print(f"reachable: {heard.mean():6.1%} of measurements had a path")
    print(f"per ue:    {heard.sum(axis=1).min()} to {heard.sum(axis=1).max()} cells reported")
    print(f"admitted:  {(band >= 0).mean():6.1%} of UEs fit a cell-band's PRBs")
    print(f"demand:    peak tile {prb.max():.1f} PRBs in one interval")
    print(f"mdt:       {path}")
    print(f"demand:    {demand_path}  shape {prb.shape} [t, row, col]")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the synthetic MDT. Entry point for ``task simulation:mdt``.

    Example:
        $ task simulation:mdt -- simulation.mdt.rsrp_noise_sigma_db=3.0
    """
    mdt = build(cfg)
    outputs = [mdt, cfg.simulation.output.demand_map_file]
    log_stage(cfg, "simulation_mdt", groups=["simulation", "kpi"], outputs=outputs)


if __name__ == "__main__":
    main()
