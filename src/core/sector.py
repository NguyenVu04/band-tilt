"""The sector-band decision variable: one antenna, one tilt per band."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# The sector table's columns, one row per sector-band: the mast (node), then the
# sector on it, then that sector's tilt and PRB limit on one band.
SECTOR_COLUMNS = (
    "node",
    "node_x",
    "node_y",
    "node_z",
    "sector",
    "azimuth_deg",
    "band",
    "tilt_deg",
    "tilt_min_deg",
    "tilt_max_deg",
    "max_prb",
)

# What every row of one sector must agree on: its mast, then where that sector stands and points.
_SITE_COLUMNS = ("node", "node_x", "node_y", "node_z", "azimuth_deg")


@dataclass(frozen=True)
class Tilt:
    """The downtilt of one sector-band pair, and the range it may move in.

    Attributes:
        baseline_deg: The downtilt this sector-band starts at.
        bounds_deg: Inclusive range an optimizer may move it within.
    """

    baseline_deg: float
    bounds_deg: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject a tilt outside the range it is allowed to move in.

        Raises:
            ValueError: When the bounds are inverted or exclude the baseline.
                A baseline outside its own bounds means the run starts from an
                infeasible configuration, which is forbidden throughout.
        """
        low, high = self.bounds_deg
        if low > high:
            raise ValueError(f"tilt bounds_deg {self.bounds_deg} is inverted")
        if not low <= self.baseline_deg <= high:
            raise ValueError(
                f"tilt baseline_deg {self.baseline_deg} lies outside its bounds "
                f"{self.bounds_deg}, so the run would start infeasible"
            )

    @classmethod
    def from_config(cls, entry: DictConfig) -> Tilt:
        """Read one ``baseline_deg``/``bounds_deg`` pair."""
        low, high = (float(value) for value in entry.bounds_deg)
        return cls(baseline_deg=float(entry.baseline_deg), bounds_deg=(low, high))


@dataclass(frozen=True)
class Sector:
    """One sector: a mast, an azimuth, and a tilt for each band it carries.

    Tilt is held per band rather than per sector because the decision variable
    is one absolute tilt per *sector-band* pair. A single tilt
    shared across a sector's bands would remove the very thing the project
    optimizes: the freedom to point frequency layers differently.

    Attributes:
        name: Unique across the layout, and ``n<node>s<sector>`` for a generated
            one; becomes the transmitter name.
        node: The mast the sector is mounted on, shared by co-located sectors.
        x: Position east, in scene metres.
        y: Position north, in scene metres.
        z: Height in scene metres, absolute and not above local ground: the
            generator sets it to the measured ground height plus the mast.
        azimuth_deg: Boresight bearing, counter-clockwise from the x axis.
        tilt: One :class:`Tilt` per band name.
        max_prb: PRBs each band of this sector can schedule at most, per band name.
    """

    name: str
    node: str
    x: float
    y: float
    z: float
    azimuth_deg: float
    tilt: dict[str, Tilt]
    # Defaulted so tilt-only callers need not invent limits;
    # max_prb_for raises for a band that was never given one.
    max_prb: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a PRB limit no UE could be scheduled under.

        Raises:
            ValueError: When any ``max_prb`` entry is not positive.
        """
        bad = {band: value for band, value in self.max_prb.items() if value <= 0}
        if bad:
            raise ValueError(f"sector {self.name!r} has non-positive max_prb {bad}")

    def max_prb_for(self, band_name: str) -> int:
        """The PRB limit this sector carries on one band.

        Raises:
            KeyError: When the sector has no entry for that band.
        """
        if band_name not in self.max_prb:
            raise KeyError(
                f"sector {self.name!r} has no max_prb for band {band_name!r}. Every sector "
                f"needs one per band; this one has {sorted(self.max_prb)}."
            )
        return self.max_prb[band_name]

    def tilt_for(self, band_name: str) -> Tilt:
        """The tilt this sector carries on one band.

        Raises:
            KeyError: When the sector has no entry for that band, which means
                the sector table and the band table disagree.
        """
        if band_name not in self.tilt:
            raise KeyError(
                f"sector {self.name!r} has no tilt for band {band_name!r}. Every sector "
                f"needs one per band; this one has {sorted(self.tilt)}."
            )
        return self.tilt[band_name]

    @classmethod
    def from_config(cls, entry: DictConfig) -> Sector:
        """Read one sector written as a mapping, the form test fixtures use.

        A mapping without ``node`` stands on a mast of its own, named after it.
        """
        return cls(
            name=str(entry.name),
            node=str(entry.get("node", entry.name)),
            x=float(entry.x),
            y=float(entry.y),
            z=float(entry.z),
            azimuth_deg=float(entry.azimuth_deg),
            tilt={str(band): Tilt.from_config(value) for band, value in entry.tilt.items()},
            max_prb={str(band): int(value) for band, value in entry.get("max_prb", {}).items()},
        )


def sectors_to_frame(sectors: Sequence[Sector]) -> pd.DataFrame:
    """The sector table, :data:`SECTOR_COLUMNS`, one row per sector-band.

    A band a sector has a tilt or a PRB limit for, but not both, leaves the other
    columns NaN.
    """
    rows = []
    for sector in sectors:
        for band in dict.fromkeys([*sector.tilt, *sector.max_prb]):
            tilt = sector.tilt.get(band)
            rows.append(
                {
                    "node": sector.node,
                    "node_x": sector.x,
                    "node_y": sector.y,
                    "node_z": sector.z,
                    "sector": sector.name,
                    "azimuth_deg": sector.azimuth_deg,
                    "band": band,
                    "tilt_deg": tilt.baseline_deg if tilt else np.nan,
                    "tilt_min_deg": tilt.bounds_deg[0] if tilt else np.nan,
                    "tilt_max_deg": tilt.bounds_deg[1] if tilt else np.nan,
                    "max_prb": sector.max_prb.get(band, np.nan),
                }
            )
    return pd.DataFrame(rows, columns=list(SECTOR_COLUMNS))


def sectors_from_frame(frame: pd.DataFrame) -> tuple[Sector, ...]:
    """Rebuild the sectors from a :func:`sectors_to_frame` table, in first-row order.

    Raises:
        ValueError: When a column is missing, a ``(sector, band)`` pair has more
            than one row, a sector's rows disagree on where it stands or points,
            a position is not finite, a ``max_prb`` is not a whole number, or
            as :class:`Tilt` and :class:`Sector` validate.
    """
    missing = [column for column in SECTOR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"sector table has no {', '.join(missing)} column")
    repeated = frame[frame.duplicated(["sector", "band"], keep=False)]
    if not repeated.empty:
        pairs = sorted({f"{row.sector}/{row.band}" for row in repeated.itertuples()})
        raise ValueError(f"sector table has more than one row for {', '.join(pairs)}")
    sectors = []
    for name, rows in frame.groupby("sector", sort=False, observed=True):
        if len(rows[list(_SITE_COLUMNS)].drop_duplicates()) > 1:
            raise ValueError(f"sector {name!r} has rows that disagree on its node or position")
        first = rows.iloc[0]
        if not np.isfinite(first[list(_SITE_COLUMNS[1:])].to_numpy(dtype=float)).all():
            raise ValueError(f"sector {name!r} has a position or azimuth that is not finite")
        tilt, max_prb = {}, {}
        for row in rows.itertuples():
            if pd.notna(row.tilt_deg):
                tilt[str(row.band)] = Tilt(
                    float(row.tilt_deg), (float(row.tilt_min_deg), float(row.tilt_max_deg))
                )
            if pd.notna(row.max_prb):
                if row.max_prb != int(row.max_prb):
                    raise ValueError(
                        f"sector {name!r} band {row.band!r} has max_prb {row.max_prb}, "
                        "which is not a whole number"
                    )
                max_prb[str(row.band)] = int(row.max_prb)
        sectors.append(
            Sector(
                name=str(name),
                node=str(first.node),
                x=float(first.node_x),
                y=float(first.node_y),
                z=float(first.node_z),
                azimuth_deg=float(first.azimuth_deg),
                tilt=tilt,
                max_prb=max_prb,
            )
        )
    return tuple(sectors)


def read_sectors(path: str | Path) -> tuple[Sector, ...]:
    """Read a sector table, CSV or Parquet by suffix, into sectors in table order.

    Raises:
        FileNotFoundError: When the file does not exist, which means the stage
            that writes it has not been run.
        ValueError: As :func:`src.core.sector.sectors_from_frame`, or when the
            table is empty.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No sector table at {path}. Run `task simulation:scenario`, or supply one."
        )
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} holds no sector.")
    return sectors_from_frame(frame)


def site_frame(sectors: Sequence[Sector]) -> pd.DataFrame:
    """Where each sector stands and points, one row per sector, in ``sectors`` order.

    Returns:
        Columns ``sector``, ``node``, ``x``, ``y``, ``z`` and ``azimuth_deg``: what
        maps mark and per-sector tables label, without the per-band tilts.
    """
    return pd.DataFrame(
        [
            {
                "sector": sector.name,
                "node": sector.node,
                "x": sector.x,
                "y": sector.y,
                "z": sector.z,
                "azimuth_deg": sector.azimuth_deg,
            }
            for sector in sectors
        ],
        columns=["sector", "node", "x", "y", "z", "azimuth_deg"],
    )
