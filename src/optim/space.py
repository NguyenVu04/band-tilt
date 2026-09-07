"""The tilt search space: the box an optimizer may move in."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import Cell, Tilt
from src.simulation import transmitter


@dataclass(frozen=True)
class TiltSpace:
    """The box of absolute tilts, one dimension per cell-band pair.

    Dimensions are ordered cell-major, band-minor, matching
    :attr:`src.data.load.Artifacts.measurement_columns` and the row order of
    ``data/processed/cell.parquet``, so a vector here joins those tables
    without a re-sort.

    Attributes:
        cells: The layout in the config's order. Position and azimuth are
            fixed; only tilt moves.
        band_names: Band names in the order ``simulation.radio_map.bands``
            declares them, which is also the radio map's band-axis order.
        lower: Per-dimension lower bound, degrees.
        upper: Per-dimension upper bound, degrees.
        baseline: The committed tilt, the incumbent every result is measured
            against.
    """

    cells: tuple[Cell, ...]
    band_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    baseline: np.ndarray

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TiltSpace:
        """Read the cell table and the band list.

        Raises:
            ValueError: When a cell carries no tilt for a configured band, so
                the space would hold a dimension with no bounds to move in.
        """
        cells = transmitter.load(cfg)
        band_names = tuple(str(entry.name) for entry in cfg.simulation.radio_map.bands)

        missing = [
            f"{cell.name}/{band}" for cell in cells for band in band_names if band not in cell.tilt
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} cell-band pairs have no tilt: {', '.join(missing[:8])}"
                f"{' ...' if len(missing) > 8 else ''}. Every cell in "
                "simulation.transmitters.cells needs one entry per band in "
                "simulation.radio_map.bands; re-run `task simulation:layout` if the bands changed."
            )

        tilts = [cell.tilt_for(band) for cell in cells for band in band_names]
        return cls(
            cells=cells,
            band_names=band_names,
            lower=np.array([tilt.bounds_deg[0] for tilt in tilts], dtype=float),
            upper=np.array([tilt.bounds_deg[1] for tilt in tilts], dtype=float),
            baseline=np.array([tilt.baseline_deg for tilt in tilts], dtype=float),
        )

    @property
    def n_dim(self) -> int:
        """Number of decision variables: one per cell-band pair."""
        return len(self.cells) * len(self.band_names)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The ``(cell, band)`` behind each dimension, in dimension order."""
        return tuple((cell.name, band) for cell in self.cells for band in self.band_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Column and Ax-parameter name for each dimension.

        Deliberately the same string in both roles, so an Ax parameterisation
        and a history row need no renaming between them.
        """
        return tuple(f"tilt_{cell}_{band}" for cell, band in self.pairs)

    def to_cells(self, tilt_deg: np.ndarray) -> tuple[Cell, ...]:
        """The layout with every cell-band tilt set to this vector.

        Bands the space does not cover keep the tilt the cell already carried,
        so a vector never silently drops a carrier the config declared.

        Raises:
            ValueError: When the vector is the wrong length, holds a
                non-finite value, or leaves the box. The last is checked here,
                naming the cell and band, rather than left to
                :class:`src.core.cell.Tilt`, whose message cannot say which
                dimension was at fault.
        """
        values = np.asarray(tilt_deg, dtype=float).reshape(-1)
        if values.size != self.n_dim:
            raise ValueError(f"expected {self.n_dim} tilts, got {values.size}")
        if not np.isfinite(values).all():
            raise ValueError("tilt vector holds a non-finite value")

        outside = np.flatnonzero((values < self.lower) | (values > self.upper))
        if outside.size:
            first = int(outside[0])
            cell, band = self.pairs[first]
            raise ValueError(
                f"{outside.size} tilts lie outside their bounds, first {cell}/{band} at "
                f"{values[first]:.3f} deg, bounds [{self.lower[first]}, {self.upper[first]}]. "
                "Clip a proposal to the box before evaluating it."
            )

        n_band = len(self.band_names)
        return tuple(
            replace(
                cell,
                tilt={
                    **cell.tilt,
                    **{
                        band: Tilt(
                            baseline_deg=float(values[index * n_band + offset]),
                            bounds_deg=cell.tilt_for(band).bounds_deg,
                        )
                        for offset, band in enumerate(self.band_names)
                    },
                },
            )
            for index, cell in enumerate(self.cells)
        )

    def clip(self, tilt_deg: np.ndarray) -> np.ndarray:
        """The vector moved to the nearest point inside the box."""
        return np.clip(np.asarray(tilt_deg, dtype=float).reshape(-1), self.lower, self.upper)

    def as_frame(self, tilt_deg: np.ndarray) -> pd.DataFrame:
        """One row per cell-band pair, carrying the tilt and its bounds."""
        values = np.asarray(tilt_deg, dtype=float).reshape(-1)
        cell_names, band_names = zip(*self.pairs, strict=True)
        return pd.DataFrame(
            {
                "cell": list(cell_names),
                "band": list(band_names),
                "tilt_deg": values,
                "tilt_min_deg": self.lower,
                "tilt_max_deg": self.upper,
            }
        )
