"""Shared fixtures: the sector table the stages read from disk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from src.core.sector import Sector, Tilt, sectors_to_frame


def sector_from_mapping(entry: Mapping) -> Sector:
    """One sector written as a mapping; without ``node`` it stands on a mast of its own."""
    entry = OmegaConf.create(dict(entry))
    return Sector(
        name=str(entry.name),
        node=str(entry.get("node", entry.name)),
        x=float(entry.x),
        y=float(entry.y),
        z=float(entry.z),
        azimuth_deg=float(entry.azimuth_deg),
        tilt={str(band): Tilt.from_config(value) for band, value in entry.tilt.items()},
        max_prb={str(band): int(value) for band, value in entry.get("max_prb", {}).items()},
    )


def write_sectors(directory: Path, sectors: Sequence[dict], name: str = "sectors.csv") -> str:
    """Write sectors given as mappings to a sector table under ``directory``; returns its path."""
    built = [sector_from_mapping(sector) for sector in sectors]
    path = Path(directory) / name
    sectors_to_frame(built).to_csv(path, index=False)
    return str(path)
