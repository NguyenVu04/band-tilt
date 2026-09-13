"""Read the four simulation artifacts, and write processed tables as Parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# The stage that writes each artifact, named in the error when one is missing.
_STAGES = {
    "ue_file": "scenario",
    "manifest_file": "scenario",
    "radio_map_file": "radio",
    "mdt_file": "mdt",
}


@dataclass(frozen=True)
class Artifacts:
    """One scenario's simulation output, read but not yet verified.

    Attributes:
        mdt: ``data/interim/mdt.csv`` as read, before any typing.
        ue: The UE population the MDT was sampled from. Longer than ``mdt``,
            which drops UEs no transmitter reaches.
        radio: Every array in ``radio_map.npz``, keyed as written.
        manifest: The parsed ``scenario.json``.
    """

    mdt: pd.DataFrame
    ue: pd.DataFrame
    radio: dict[str, np.ndarray]
    manifest: dict[str, Any]

    @property
    def tx_names(self) -> list[str]:
        """Cell names in the radio map's transmitter-axis order."""
        return [str(name) for name in self.radio["tx_name"]]

    @property
    def band_labels(self) -> list[str]:
        """Band names in the radio map's band-axis order."""
        return [str(label) for label in self.radio["band_label"]]

    @property
    def measurement_columns(self) -> list[str]:
        """The ``rsrp_*`` column names the MDT is expected to carry, in order.

        Cell-major, band-minor, matching how ``src.simulation.mdt.build``
        flattens the sampled array.
        """
        return [f"rsrp_{tx}_{band}" for tx in self.tx_names for band in self.band_labels]

    @property
    def sinr_columns(self) -> list[str]:
        """The ``sinr_*`` column names, in :attr:`measurement_columns` order."""
        return [column.replace("rsrp_", "sinr_", 1) for column in self.measurement_columns]

    @property
    def shape(self) -> tuple[int, int]:
        """The grid's ``(n_rows, n_cols)``."""
        return int(self.radio["n_rows"]), int(self.radio["n_cols"])

    @property
    def scenario_id(self) -> str:
        """The manifest's scenario identifier, the intended split key."""
        return str(self.manifest["scenario_id"])


def load_artifacts(cfg: DictConfig) -> Artifacts:
    """Read every artifact named in ``simulation.output``.

    Read-only: the artifacts are regenerated with ``task simulation``, never
    edited in place.

    Raises:
        FileNotFoundError: When a stage has not been run, naming which one.
    """
    output = cfg.simulation.output
    paths = {key: Path(output[key]) for key in _STAGES}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{_STAGES[key]}` first.")

    with np.load(paths["radio_map_file"], allow_pickle=False) as archive:
        radio = {key: archive[key] for key in archive.files}

    return Artifacts(
        mdt=pd.read_csv(paths["mdt_file"]),
        ue=pd.read_csv(paths["ue_file"]),
        radio=radio,
        manifest=json.loads(paths["manifest_file"].read_text(encoding="utf-8")),
    )


def save(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to Parquet, creating the directory. Returns the path.

    Parquet rather than CSV because it round-trips dtypes. The reported
    indicators stay boolean and the RSRP columns stay float32 with a real NaN,
    which a CSV reader would have to guess at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
