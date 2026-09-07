"""The cell-band decision variable: one antenna, one tilt per band."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class Tilt:
    """The downtilt of one cell-band pair, and the range it may move in.

    Attributes:
        baseline_deg: The downtilt this cell-band starts at.
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
class Cell:
    """One cell: a mast, an azimuth, and a tilt for each band it carries.

    Tilt is held per band rather than per cell because the decision variable
    is one absolute tilt per *cell-band* pair. A single tilt
    shared across a cell's bands would remove the very thing the project
    optimizes: the freedom to point frequency layers differently.

    Attributes:
        name: Unique across the layout, and ``n<node>c<cell>`` for a generated
            one; becomes the transmitter name and the stem of the MDT column
            names.
        x: Position east, in scene metres.
        y: Position north, in scene metres.
        z: Mast height above the scene's ground plane.
        azimuth_deg: Boresight bearing, counter-clockwise from the x axis.
        tilt: One :class:`Tilt` per band name.
    """

    name: str
    x: float
    y: float
    z: float
    azimuth_deg: float
    tilt: dict[str, Tilt]

    def tilt_for(self, band_name: str) -> Tilt:
        """The tilt this cell carries on one band.

        Raises:
            KeyError: When the cell has no entry for that band, which means
                the cell table and the band table disagree.
        """
        if band_name not in self.tilt:
            raise KeyError(
                f"cell {self.name!r} has no tilt for band {band_name!r}. Every cell "
                f"needs one per band; this one has {sorted(self.tilt)}."
            )
        return self.tilt[band_name]

    @classmethod
    def from_config(cls, entry: DictConfig) -> Cell:
        """Read one entry of ``simulation.transmitters.cells``."""
        return cls(
            name=str(entry.name),
            x=float(entry.x),
            y=float(entry.y),
            z=float(entry.z),
            azimuth_deg=float(entry.azimuth_deg),
            tilt={str(band): Tilt.from_config(value) for band, value in entry.tilt.items()},
        )
