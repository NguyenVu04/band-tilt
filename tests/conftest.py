"""Shared fixtures: the sector table the stages read from disk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from src.core.sector import Sector, sectors_to_frame


def write_sectors(directory: Path, sectors: Sequence[dict], name: str = "sectors.csv") -> str:
    """Write sectors given as mappings to a sector table under ``directory``; returns its path."""
    built = [Sector.from_config(OmegaConf.create(sector)) for sector in sectors]
    path = Path(directory) / name
    sectors_to_frame(built).to_csv(path, index=False)
    return str(path)
