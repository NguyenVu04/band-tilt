"""Read the four simulation artifacts, and write processed tables as Parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Each artifact's config section and the stage that writes it, named in the
# error when one is missing.
_STAGES = {
    "ue_file": ("scenario", "scenario"),
    "cells_file": ("scenario", "scenario"),
    "manifest_file": ("scenario", "scenario"),
    "radio_map_file": ("simulation", "radio"),
}


@dataclass(frozen=True)
class Artifacts:
    """One scenario's simulation output, read but not yet verified.

    Attributes:
        ue: ``scenario.output.ue_file`` as read, before any typing.
            Every drawn UE, including those no transmitter reaches.
        cells: ``scenario.output.cells_file`` as read, one row per cell-band.
        radio: Every array in ``simulation.output.radio_map_file``, keyed as written.
        manifest: The parsed ``scenario.output.manifest_file``.
    """

    ue: pd.DataFrame
    cells: pd.DataFrame
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
    """Read every artifact named in ``scenario.output`` and ``simulation.output``.

    Read-only: the artifacts are regenerated with ``task simulation``, never
    edited in place.

    Raises:
        FileNotFoundError: When a stage has not been run, naming which one.
    """
    paths = {key: Path(cfg[section].output[key]) for key, (section, _) in _STAGES.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{_STAGES[key][1]}` first.")

    with np.load(paths["radio_map_file"], allow_pickle=False) as archive:
        radio = {key: archive[key] for key in archive.files}

    return Artifacts(
        ue=pd.read_csv(paths["ue_file"]),
        cells=pd.read_csv(paths["cells_file"]),
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
