"""Read the simulation inputs and the radio map, and write processed tables as Parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Each artifact's ``simulation`` config block, and what to run when it is
# missing. The inputs come from the generator or from real data.
_SOURCES = {
    "ue_file": ("input", "Run `task simulation:scenario`, or supply it"),
    "manifest_file": ("input", "Run `task simulation:scenario`, or supply it"),
    "radio_map_file": ("output", "Run `task simulation:radio`"),
}


@dataclass(frozen=True)
class Artifacts:
    """One scenario's UE table, manifest and radio map, read but not yet verified.

    Attributes:
        ue: ``simulation.input.ue_file`` as read, before any typing.
            Every UE, including those no transmitter reaches.
        radio: Every array in ``simulation.output.radio_map_file``, keyed as written.
        manifest: The parsed ``simulation.input.manifest_file``.
    """

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
    def shape(self) -> tuple[int, int]:
        """The grid's ``(n_rows, n_cols)``."""
        return int(self.radio["n_rows"]), int(self.radio["n_cols"])

    @property
    def scenario_id(self) -> str:
        """The manifest's scenario identifier, the intended split key."""
        return str(self.manifest["scenario_id"])


def load_artifacts(cfg: DictConfig) -> Artifacts:
    """Read the UE table and manifest of ``simulation.input`` and the radio map.

    Read-only: the artifacts are regenerated or re-supplied, never edited in place.

    Raises:
        FileNotFoundError: When a file is missing, naming what produces it.
    """
    paths = {key: Path(cfg.simulation[block][key]) for key, (block, _) in _SOURCES.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. {_SOURCES[key][1]}.")

    with np.load(paths["radio_map_file"], allow_pickle=False) as archive:
        radio = {key: archive[key] for key in archive.files}

    return Artifacts(
        ue=pd.read_csv(paths["ue_file"]),
        radio=radio,
        manifest=json.loads(paths["manifest_file"].read_text(encoding="utf-8")),
    )


def save(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to Parquet, creating the directory. Returns the path.

    Parquet rather than CSV because it round-trips dtypes: the int16 tiles and
    the categorical columns, which a CSV reader would have to guess at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
