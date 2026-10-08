"""Read the simulation inputs and the radio map, and write processed tables as Parquet."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.simulation.radio import read_manifest

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
        """Sector names in the radio map's transmitter-axis order."""
        return [str(name) for name in self.radio["tx_name"]]

    @property
    def band_labels(self) -> list[str]:
        """Band names in the radio map's band-axis order."""
        return [str(label) for label in self.radio["band_label"]]

    @property
    def shape(self) -> tuple[int, int]:
        """The grid's ``(n_rows, n_cols)``."""
        return grid_shape(self.radio)

    @property
    def scenario_id(self) -> str:
        """The manifest's scenario identifier."""
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

    return Artifacts(
        ue=pd.read_csv(paths["ue_file"]),
        radio=load_npz(paths["radio_map_file"]),
        manifest=read_manifest(cfg),
    )


def grid_shape(radio: dict[str, Any]) -> tuple[int, int]:
    """A radio map's grid ``(n_rows, n_cols)``."""
    return int(radio["n_rows"]), int(radio["n_cols"])


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Every array of an ``.npz`` archive, read eagerly so the file is closed, pickles refused."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to Parquet, creating the directory. Returns the path.

    Parquet rather than CSV because it round-trips dtypes: the int16 tiles and
    the categorical columns, which a CSV reader would have to guess at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` as strict, indented JSON, creating the directory. Returns the path.

    A non-finite float is written as the string ``"inf"``, ``"-inf"`` or
    ``"nan"``, which ``float`` reads back: ``null`` would not say which, and
    the percentile KPIs are ``-inf`` where nothing is covered.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_finite_or_text(payload), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _finite_or_text(value: Any) -> Any:
    """``value`` with every non-finite float, at any depth, replaced by its text."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_or_text(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_text(item) for item in value]
    return value
