"""The tilt search space: the box an optimizer may move in."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.sector import Sector, Tilt, read_sectors


@dataclass(frozen=True)
class TiltSpace:
    """The box of absolute tilts, one dimension per sector-band pair.

    Dimensions are ordered sector-major, band-minor: sectors in
    ``simulation.input.sectors_file`` order, bands in ``simulation.radio_map.bands``
    order.

    Attributes:
        sectors: The layout in the config's order. Position and azimuth are
            fixed; only tilt moves.
        band_names: Band names in the order ``simulation.radio_map.bands``
            declares them, which is also the radio map's band-axis order.
        lower: Per-dimension lower bound, degrees.
        upper: Per-dimension upper bound, degrees.
        baseline: The committed tilt, the incumbent every result is measured
            against.
        resolution_deg: The tilt step an antenna can be set to; every point a
            search proposes is snapped to ``lower + k * resolution_deg``.
    """

    sectors: tuple[Sector, ...]
    band_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    baseline: np.ndarray
    resolution_deg: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TiltSpace:
        """Read the sector table, the band list and ``optim.tilt_resolution_deg``.

        Raises:
            ValueError: When a sector carries no tilt for a configured band, so
                the space would hold a dimension with no bounds to move in.
        """
        sectors = read_sectors(cfg.simulation.input.sectors_file)
        band_names = tuple(str(entry.name) for entry in cfg.simulation.radio_map.bands)

        missing = [
            f"{sector.name}/{band}"
            for sector in sectors
            for band in band_names
            if band not in sector.tilt
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} sector-band pairs have no tilt: {', '.join(missing[:8])}"
                f"{' ...' if len(missing) > 8 else ''}. Every sector in "
                f"{cfg.simulation.input.sectors_file} needs one row per band in "
                "simulation.radio_map.bands; regenerate or fix the sector table if the bands "
                "changed."
            )

        tilts = [sector.tilt_for(band) for sector in sectors for band in band_names]
        return cls(
            sectors=sectors,
            band_names=band_names,
            lower=np.array([tilt.bounds_deg[0] for tilt in tilts], dtype=float),
            upper=np.array([tilt.bounds_deg[1] for tilt in tilts], dtype=float),
            baseline=np.array([tilt.baseline_deg for tilt in tilts], dtype=float),
            resolution_deg=float(cfg.optim.tilt_resolution_deg),
        )

    @property
    def n_dim(self) -> int:
        """Number of decision variables: one per sector-band pair."""
        return len(self.sectors) * len(self.band_names)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The ``(sector, band)`` behind each dimension, in dimension order."""
        return tuple((sector.name, band) for sector in self.sectors for band in self.band_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """History column name for each dimension, in dimension order."""
        return tuple(f"tilt_{sector}_{band}" for sector, band in self.pairs)

    def to_sectors(self, tilt_deg: np.ndarray) -> tuple[Sector, ...]:
        """The layout with every sector-band tilt set to this vector.

        Bands the space does not cover keep the tilt the sector already carried,
        so a vector never silently drops a carrier the config declared.

        Raises:
            ValueError: When the vector is the wrong length, holds a
                non-finite value, or leaves the box. The last is checked here,
                naming the sector and band, rather than left to
                :class:`src.core.sector.Tilt`, whose message cannot say which
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
            sector, band = self.pairs[first]
            raise ValueError(
                f"{outside.size} tilts lie outside their bounds, first {sector}/{band} at "
                f"{values[first]:.3f} deg, bounds [{self.lower[first]}, {self.upper[first]}]. "
                "Clip a proposal to the box before evaluating it."
            )

        n_band = len(self.band_names)
        return tuple(
            replace(
                sector,
                tilt={
                    **sector.tilt,
                    **{
                        band: Tilt(
                            baseline_deg=float(values[index * n_band + offset]),
                            bounds_deg=sector.tilt_for(band).bounds_deg,
                        )
                        for offset, band in enumerate(self.band_names)
                    },
                },
            )
            for index, sector in enumerate(self.sectors)
        )

    @property
    def unit_span(self) -> np.ndarray:
        """Per-dimension width the unit cube is scaled by.

        A dimension whose bounds coincide has nowhere to move, so its width is
        one: any unit value maps back to the same tilt after :meth:`clip`.
        """
        return np.where(self.upper > self.lower, self.upper - self.lower, 1.0)

    def from_unit(self, point: np.ndarray) -> np.ndarray:
        """The tilt vector for a point of the unit cube, on the resolution lattice, in the box."""
        offset = np.asarray(point, dtype=float) * self.unit_span
        return self.clip(self.lower + np.round(offset / self.resolution_deg) * self.resolution_deg)

    def to_unit(self, tilt_deg: np.ndarray) -> np.ndarray:
        """The unit-cube point of a tilt vector; the inverse of :meth:`from_unit` inside the box."""
        return (np.asarray(tilt_deg, dtype=float).reshape(-1) - self.lower) / self.unit_span

    def clip(self, tilt_deg: np.ndarray) -> np.ndarray:
        """The vector moved to the nearest point inside the box."""
        return np.clip(np.asarray(tilt_deg, dtype=float).reshape(-1), self.lower, self.upper)

    def as_frame(self, tilt_deg: np.ndarray) -> pd.DataFrame:
        """One row per sector-band pair, carrying the tilt and its bounds."""
        values = np.asarray(tilt_deg, dtype=float).reshape(-1)
        sector_names, band_names = zip(*self.pairs, strict=True)
        return pd.DataFrame(
            {
                "sector": list(sector_names),
                "band": list(band_names),
                "tilt_deg": values,
                "tilt_min_deg": self.lower,
                "tilt_max_deg": self.upper,
            }
        )
