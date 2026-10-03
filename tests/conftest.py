"""Shared fixtures: the cell table the stages read from disk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from src.core.cell import Cell, cells_to_frame


def write_cells(directory: Path, cells: Sequence[dict], name: str = "cells.csv") -> str:
    """Write cells given as mappings to a cell table under ``directory``; returns its path."""
    built = [Cell.from_config(OmegaConf.create(cell)) for cell in cells]
    path = Path(directory) / name
    cells_to_frame(built).to_csv(path, index=False)
    return str(path)
