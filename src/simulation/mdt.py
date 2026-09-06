"""Create synthetic UE reports from radio-map samples.

Adds independent RSRP noise, then censors eligible reports while preserving
the strongest measurements.
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
_POSITION_COLUMNS = ("t_index", "t_s", "x", "y", "z", "cell_col", "cell_row")


@dataclass(frozen=True)
class MdtSpec:
    """How far a report may stray from the truth, and how much goes unreported.

    Attributes:
        rsrp_noise_sigma_db: Standard deviation of the receiver measurement
            error.
        drop_fraction: Share of the eligible measurements left unreported.
        protect_within_db: Measurements within this margin of the strongest are
            never dropped. Defaults to the KPI's own overlap margin, which is
            what makes the censoring KPI-safe.
        protect_strongest_n: Additional floor on how many of the strongest are
            protected, regardless of the margin.
    """

    rsrp_noise_sigma_db: float
    drop_fraction: float
    protect_within_db: float
    protect_strongest_n: int

    def __post_init__(self) -> None:
        """Reject a reporting model that cannot be applied.

        Raises:
            ValueError: When the drop fraction is outside ``[0, 1]`` or a
                parameter is negative.
        """
        if not 0.0 <= self.drop_fraction <= 1.0:
            raise ValueError(
                f"simulation.mdt.drop_fraction must be in [0, 1], got {self.drop_fraction}"
            )
        if self.rsrp_noise_sigma_db < 0 or self.protect_within_db < 0:
            raise ValueError("simulation.mdt sigma and margin must not be negative")
        if self.protect_strongest_n < 1:
            raise ValueError(
                "simulation.mdt.protect_strongest_n must be at least 1, so every UE keeps "
                f"its serving cell; got {self.protect_strongest_n}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> MdtSpec:
        """Read ``simulation.mdt``."""
        mdt = cfg.simulation.mdt
        return cls(
            rsrp_noise_sigma_db=float(mdt.rsrp_noise_sigma_db),
            drop_fraction=float(mdt.drop_fraction),
            protect_within_db=float(mdt.protect_within_db),
            protect_strongest_n=int(mdt.protect_strongest_n),
        )


def measure(clean: np.ndarray, spec: MdtSpec, seed: int) -> np.ndarray:
    """Add measurement error and censor, returning reported RSRP.

    ``clean`` is ``[n_ue, n_measurement]`` in dBm, already NaN where the ray
    tracer found no path. Returns an array of the same shape with the reported
    values, NaN where nothing is reported.
    """
    rng = np.random.default_rng(seed)
    reported = clean + rng.normal(0.0, spec.rsrp_noise_sigma_db, size=clean.shape)

    heard = np.isfinite(reported)
    # Unheard measurements sit at -inf rather than NaN so a UE that hears
    # nothing reduces to -inf instead of warning about an all-NaN slice. Such a
    # UE is a genuine coverage hole, not an error: it has nothing to protect and
    # nothing to drop, and the comparison below leaves it untouched.
    strongest = np.where(heard, reported, -np.inf).max(axis=1)
    protected = heard & (reported >= strongest[:, None] - spec.protect_within_db)

    # Rank descending so the strongest few are protected outright, whatever the
    # margin says. Unheard measurements sort last and are never selected.
    order = np.argsort(np.where(heard, -reported, np.inf), axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")
    protected |= heard & (rank < spec.protect_strongest_n)

    eligible = heard & ~protected
    key = np.where(eligible, rng.random(clean.shape), np.inf)
    drop_rank = np.argsort(np.argsort(key, axis=1, kind="stable"), axis=1, kind="stable")
    n_drop = np.rint(spec.drop_fraction * eligible.sum(axis=1)).astype(np.int64)

    return np.where(eligible & (drop_rank < n_drop[:, None]), np.nan, reported)


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

    col = ues["cell_col"].to_numpy()
    row = ues["cell_row"].to_numpy()
    if col.min() < 0 or col.max() >= n_cols or row.min() < 0 or row.max() >= n_rows:
        raise ValueError(
            f"UE grid cells span cols {col.min()}..{col.max()} rows {row.min()}..{row.max()}, "
            f"outside the radio map's {n_cols} x {n_rows} grid. The two stages used "
            "different grids."
        )

    # [band, tx, row, col] sampled at each UE's cell -> [ue, band, tx], then
    # flattened so a cell's bands sit next to each other.
    sampled = rsrp[:, :, row, col].transpose(2, 1, 0)
    clean = sampled.reshape(len(ues), -1).astype(np.float64)
    columns = [f"rsrp_{tx}_{band}" for tx in tx_names for band in band_labels]

    spec = MdtSpec.from_config(cfg)
    reported = measure(clean, spec, seeds.stream(cfg, "mdt"))

    heard = np.isfinite(clean)
    covered = heard.any(axis=1)
    n_no_signal = int((~covered).sum())
    ues = ues.loc[covered].reset_index(drop=True)
    heard = heard[covered]
    reported = reported[covered]

    frame = pd.concat(
        [ues[list(_POSITION_COLUMNS)], pd.DataFrame(reported, columns=columns, index=ues.index)],
        axis=1,
    )
    path = Path(cfg.simulation.output.mdt_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.3f", na_rep="")

    kept = np.isfinite(reported)
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
    print(f"reported:  {kept.sum() / max(heard.sum(), 1):6.1%} of those survived censoring")
    print(f"per ue:    {kept.sum(axis=1).min()} to {kept.sum(axis=1).max()} cells reported")
    print(f"mdt:       {path}")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the synthetic MDT. Entry point for ``task simulation:mdt``.

    Example:
        $ task simulation:mdt -- simulation.mdt.drop_fraction=0.5
    """
    build(cfg)


if __name__ == "__main__":
    main()
