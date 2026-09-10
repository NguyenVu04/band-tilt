"""The tilt sweep, the antenna pattern fitted from it, and the training pairs.

One transmitter's orientation cannot alter another transmitter's field, so
moving ``tilt[cell i, band b]`` moves only the slice ``rsrp[b, i]`` and the
36-dimensional problem is 36 independent one-dimensional ones. That is what
:func:`check_decomposition` verifies and what makes :func:`sweep` cheap: set
every cell of a band to the same tilt, solve once, and harvest all twelve
transmitters at that tilt.

The sweep is therefore not a sample of the objective landscape but the whole of
it on its tilt grid, which is why the surrogate can be scored against exact
ground truth without any further ray tracing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.core.cell import Cell
from src.optim.evaluator import Evaluator
from src.simulation.grid import GridSpec, Raster
from src.surrogate import features
from src.surrogate.features import SceneFeatures

# The encoding contract, shared with src.surrogate.train and read back by
# src.surrogate.evaluator: these are what the weights are fitted against, so a
# model and an encoding that disagree on them are silently incompatible.
#
# Where the model's dB channels are centred and how they are scaled. Taken from
# the KPI thresholds in configs/kpi.yaml -- a hole is -120 dBm and weak is -90 --
# rather than from run statistics, so the normalisation is a property of the
# problem and not of whichever sweep was on disk when training started.
DB_CENTRE = -100.0
DB_SCALE = 20.0

# The same idea for the two angle channels: a pattern's main lobe is a few
# degrees wide and its correction a few dB, so these put both near unit scale.
GAIN_SCALE = 10.0
ANGLE_SCALE = 45.0

# Tiles no tilt in the sweep ever reached have no propagation term to recover,
# so the reference level falls back to this. Below the hole threshold by a wide
# margin: every KPI already treats such a tile as uncovered.
_FLOOR_DBM = -160.0

# The SceneFeatures fields, which are also the keys of the cached npz.
_FEATURE_FIELDS = tuple(SceneFeatures.__dataclass_fields__)


@dataclass(frozen=True)
class DecompositionReport:
    """Whether one cell's tilt leaves every other cell's map untouched.

    Attributes:
        moved_cell: The cell whose tilt was changed.
        band: The band solved.
        max_abs_delta_db: Largest change over the transmitters that did *not*
            move. Zero means the per-slice surrogate is sound.
        moved_max_abs_delta_db: Largest change on the transmitter that did,
            reported so a null result cannot be mistaken for success.
    """

    moved_cell: str
    band: str
    max_abs_delta_db: float
    moved_max_abs_delta_db: float

    @property
    def holds(self) -> bool:
        """True when untouched transmitters were bit-identical and the moved one moved."""
        return self.max_abs_delta_db == 0.0 and self.moved_max_abs_delta_db > 0.0


def check_decomposition(
    cfg: DictConfig, band_index: int = 0, delta_deg: float = 4.0
) -> DecompositionReport:
    """Solve one band twice, moving a single cell, and compare the rest.

    The whole surrogate design rests on the answer, so it is measured rather
    than argued: physics says transmitters do not couple, but a solver sharing
    one Monte-Carlo stream across transmitters could still let one cell's
    orientation shift another cell's samples.

    Costs two band-solves. Needs a CUDA GPU and the ``rt`` extra.
    """
    with Evaluator(cfg) as evaluator:
        space = evaluator.space
        band = evaluator.bands[band_index]
        dimension = space.parameter_names.index(f"tilt_{space.cells[0].name}_{band.name}")

        before = space.baseline.copy()
        after = before.copy()
        after[dimension] = min(before[dimension] + delta_deg, space.upper[dimension])

        first = evaluator.solve(space.to_cells(before), band_index)
        second = evaluator.solve(space.to_cells(after), band_index)

    delta = np.abs(np.nan_to_num(second, nan=_FLOOR_DBM) - np.nan_to_num(first, nan=_FLOOR_DBM))
    others = np.ones(delta.shape[0], dtype=bool)
    others[0] = False
    return DecompositionReport(
        moved_cell=space.cells[0].name,
        band=band.name,
        max_abs_delta_db=float(delta[others].max()),
        moved_max_abs_delta_db=float(delta[0].max()),
    )


@dataclass(frozen=True)
class Sweep:
    """Every band's radio map at every tilt on its grid.

    Attributes:
        band_names: Bands in the radio map's band-axis order.
        tilts: Per band, the tilt grid solved, in degrees.
        rsrp: Per band, ``[n_tilt, n_tx, n_rows, n_cols]`` in dBm, ``nan``
            where no path reached the tile.
        tx_names: Transmitters in the radio map's tx-axis order.
        origin_x: The x of the grid's lower corner.
        origin_y: The y of the grid's lower corner.
        tile_size_m: Side of a square tile.
        scenario_id: The world these maps were solved in. Carried so a model
            trained on one scenario cannot be pointed at another: the search
            and the report phase must share a world, and nothing downstream
            could tell that they did not.
    """

    band_names: tuple[str, ...]
    tilts: tuple[np.ndarray, ...]
    rsrp: tuple[np.ndarray, ...]
    tx_names: tuple[str, ...]
    origin_x: float
    origin_y: float
    tile_size_m: float
    scenario_id: str

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns of the solved grid."""
        return self.rsrp[0].shape[-2:]

    def tile_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Centre coordinates of every tile, each shaped ``[n_rows, n_cols]``.

        Derived from the grid this sweep was solved on rather than rebuilt, for
        the reason :func:`src.surrogate.features.build` gives: a locally
        recomputed grid can sit half a tile off and still look plausible.
        """
        n_rows, n_cols = self.shape
        xs = self.origin_x + (np.arange(n_cols) + 0.5) * self.tile_size_m
        ys = self.origin_y + (np.arange(n_rows) + 0.5) * self.tile_size_m
        return np.meshgrid(xs, ys, indexing="xy")

    def index_of(self, band: int, tilt_deg: float) -> int:
        """Position of ``tilt_deg`` on this band's grid.

        Raises:
            ValueError: When the tilt is not on the grid. The sweep is an exact
                table, never an interpolant, so a near miss is a caller bug.
        """
        matches = np.flatnonzero(np.isclose(self.tilts[band], tilt_deg))
        if matches.size != 1:
            raise ValueError(
                f"tilt {tilt_deg} deg is not on the {self.band_names[band]} sweep grid "
                f"[{self.tilts[band][0]}, {self.tilts[band][-1]}] "
                f"step {self.tilts[band][1] - self.tilts[band][0]}"
            )
        return int(matches[0])

    def at(self, tilt_deg: np.ndarray) -> np.ndarray:
        """The exact radio map for one tilt vector, ``[n_band, n_tx, n_rows, n_cols]``.

        The ground truth every surrogate metric is measured against, assembled
        from the sweep rather than re-solved: because tilts decompose, slice
        ``[b, i]`` of the answer is the slice this sweep already holds for cell
        ``i`` at that dimension's tilt.

        Args:
            tilt_deg: A vector in :class:`~src.optim.space.TiltSpace` dimension
                order, cell-major and band-minor. Every value must be on its
                band's sweep grid.
        """
        n_band, n_tx = len(self.band_names), len(self.tx_names)
        values = np.asarray(tilt_deg, dtype=float).reshape(n_tx, n_band)
        out = np.empty((n_band, n_tx, *self.shape), dtype=np.float32)
        for band in range(n_band):
            for tx in range(n_tx):
                out[band, tx] = self.rsrp[band][self.index_of(band, values[tx, band]), tx]
        return out

    @classmethod
    def load(cls, path: str | Path) -> Sweep:
        """Read a sweep written by :func:`sweep`."""
        with np.load(Path(path), allow_pickle=False) as data:
            names = tuple(str(name) for name in data["band_label"])
            return cls(
                band_names=names,
                tilts=tuple(data[f"tilt_{name}"] for name in names),
                rsrp=tuple(data[f"rsrp_{name}"] for name in names),
                tx_names=tuple(str(name) for name in data["tx_name"]),
                origin_x=float(data["origin_x"]),
                origin_y=float(data["origin_y"]),
                tile_size_m=float(data["tile_size_m"]),
                scenario_id=str(data["scenario_id"]),
            )


def sweep(cfg: DictConfig, step_deg: float = 0.5, path: str | Path | None = None) -> Path:
    """Solve every band over its whole tilt range and write the tensor.

    All twelve cells of a band are set to the same tilt, so one solve yields
    every transmitter at that tilt -- sound only because tilts decompose, which
    :func:`check_decomposition` establishes.

    Times the first solve and prints the projected total before committing to
    the rest, so an unexpectedly slow machine is visible in seconds rather than
    after an hour.

    Args:
        cfg: The composed config.
        step_deg: Tilt resolution. Integer degrees train the model and the
            intermediate values test it, so this must divide 1.0.
        path: Where to write; defaults to ``data/interim/tilt_sweep.npz``.

    Raises:
        ValueError: When ``step_deg`` does not divide one degree.
    """
    if not np.isclose(round(1.0 / step_deg) * step_deg, 1.0):
        raise ValueError(
            f"step_deg must divide 1.0 so integer tilts land on the grid, got {step_deg}"
        )

    path = Path(path or "data/interim/tilt_sweep.npz")
    arrays: dict[str, np.ndarray] = {}

    with Evaluator(cfg) as evaluator:
        space = evaluator.space
        n_band = len(evaluator.bands)
        grids = [
            np.arange(
                space.lower[index::n_band].max(),
                space.upper[index::n_band].min() + 0.5 * step_deg,
                step_deg,
            )
            for index in range(n_band)
        ]
        total = sum(len(grid) for grid in grids)
        print(f"tilt sweep: {total} band-solves over {n_band} bands")

        done = 0
        started = time.perf_counter()
        for band_index, band in enumerate(evaluator.bands):
            maps = []
            for tilt in grids[band_index]:
                vector = space.baseline.copy()
                vector[band_index::n_band] = tilt
                maps.append(evaluator.solve(space.to_cells(vector), band_index))
                done += 1
                if done == 1:
                    each = time.perf_counter() - started
                    print(f"  first solve {each:.1f}s -> projected {each * total / 60:.1f} min")
            arrays[f"rsrp_{band.name}"] = np.stack(maps).astype(np.float32)
            arrays[f"tilt_{band.name}"] = grids[band_index]
            print(f"  {band.name}: {arrays[f'rsrp_{band.name}'].shape} [tilt, tx, row, col]")

        meta = evaluator.grid_meta
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            band_label=np.array(evaluator.band_labels),
            band_hz=np.array([band.frequency_hz for band in evaluator.bands]),
            tx_name=np.array([cell.name for cell in space.cells]),
            scenario_id=evaluator.scenario_id,
            origin_x=meta["origin_x"],
            origin_y=meta["origin_y"],
            tile_size_m=meta["tile_size_m"],
            n_rows=meta["n_rows"],
            n_cols=meta["n_cols"],
            ue_height_m=float(cfg.simulation.ue.height_m),
            **arrays,
        )
    print(f"tilt sweep: {path}  ({path.stat().st_size / 1e6:.1f} MB, {total} solves)")
    return path


def build_features(cfg: DictConfig, path: str | Path | None = None) -> Path:
    """Build the scene channels and write them beside the sweep.

    Cast here and cached because training must not need Mitsuba: the channels
    are a function of the perturbed scene and the mast positions alone, so they
    are the same for every tilt the optimizer will ever try.

    The grid is taken from the scenario manifest rather than rebuilt by
    :func:`src.simulation.grid.build`, so it cannot land half a tile away from
    the radio map the channels will be stacked against.

    Args:
        cfg: The composed config.
        path: Where to write; defaults to ``data/interim/scene_features.npz``.
    """
    path = Path(path or "data/interim/scene_features.npz")

    with Evaluator(cfg) as evaluator:
        meta = evaluator.grid_meta
        shape = (int(meta["n_rows"]), int(meta["n_cols"]))
        # free_fraction and mean_built_height carry the grid's shape and nothing
        # else here: features.build reads only the extent and the tile centres.
        raster = Raster(
            origin_x=float(meta["origin_x"]),
            origin_y=float(meta["origin_y"]),
            tile_size_m=float(meta["tile_size_m"]),
            free_fraction=np.zeros(shape),
            mean_built_height=np.zeros(shape),
        )
        built = features.build(
            evaluator.scene.mi_scene,
            evaluator.bounds,
            raster,
            evaluator.space.cells,
            GridSpec.from_config(cfg),
            float(cfg.simulation.ue.height_m),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{field: getattr(built, field) for field in _FEATURE_FIELDS})
    print(f"scene features: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_features(path: str | Path | None = None) -> SceneFeatures:
    """Read the channels written by :func:`build_features`."""
    with np.load(Path(path or "data/interim/scene_features.npz"), allow_pickle=False) as data:
        return SceneFeatures(**{field: data[field] for field in _FEATURE_FIELDS})


@dataclass(frozen=True)
class Pattern:
    """The effective vertical pattern, and the propagation term it multiplies.

    For a path set that tilt only re-weights, RSRP separates as
    ``rsrp[t, p] = base[p] + gain(elev[p] - t)``: everything about the
    geometry, the material and the horizontal pattern is tilt-independent and
    lands in ``base``, while tilt enters only through the angle off boresight.

    Both terms are fitted from the sweep rather than assumed. The 3GPP TR 38.901
    element pattern is not what an 8x8 cross-polarised array synthesises under
    ``RadioMapSolver``, and the difference is exactly what a surrogate built on
    the assumption would have to unlearn.

    Attributes:
        band_names: Bands, in the sweep's band order.
        offset_deg: Bin centres of the angle off boresight, ``elev - tilt``.
        gain_db: Fitted pattern per band, ``[n_band, n_bin]``, peak at 0 dB.
        base_db: Tilt-independent propagation per band and transmitter,
            ``[n_band, n_tx, n_rows, n_cols]``, ``nan`` where no tilt in the
            sweep ever reached the tile.
    """

    band_names: tuple[str, ...]
    offset_deg: np.ndarray
    gain_db: np.ndarray
    base_db: np.ndarray

    def gain(self, band: int, offset_deg: np.ndarray) -> np.ndarray:
        """The pattern at these angles off boresight, clamped outside the fitted range."""
        return np.interp(offset_deg, self.offset_deg, self.gain_db[band])

    def delta(
        self, band: int, elevation_deg: np.ndarray, before: float, after: float
    ) -> np.ndarray:
        """The analytic correction for a tilt change: how much gain each tile gains.

        This alone is the zero-parameter baseline the learned model has to beat.

        :meth:`Encoder.encode_batch` does not call this, and not by oversight:
        it needs the before-gain separately for the no-path fill, so it takes
        both terms from one grouped interpolation rather than paying for the
        before-gain twice.
        """
        return self.gain(band, elevation_deg - after) - self.gain(band, elevation_deg - before)

    def reference(self, band: int, elevation_deg: np.ndarray, tilt_deg: float) -> np.ndarray:
        """A finite RSRP level for every tile, ``[n_tx, n_rows, n_cols]``.

        What the map would read with no obstruction beyond what ``base_db``
        already carries. Used to fill tiles the ray tracer found no path to:
        two thirds of every slice has none, and filling those with a constant
        would make ``rsrp + delta`` meaningless across most of the map.
        """
        level = self.base_db[band] + self.gain(band, elevation_deg - tilt_deg)
        return np.where(np.isfinite(level), level, _FLOOR_DBM)


def _bin_of(elevation_deg: np.ndarray, tilt_deg: float, edges: np.ndarray) -> np.ndarray:
    """Which angle bin each tile falls in at one tilt, clipped to the grid."""
    angle = elevation_deg - tilt_deg
    return np.clip(np.digitize(angle, edges) - 1, 0, len(edges) - 2)


def fit_pattern(sweep_data: Sweep, elevation_deg: np.ndarray) -> Pattern:
    """Recover the pattern by differencing along tilt, then the level it sits on.

    Differencing is what makes this exact and non-iterative. A tile's
    propagation term is the same at every tilt, so subtracting the map at one
    tilt from the map at the next cancels it outright and leaves only

        rsrp[k] - rsrp[k+1] = gain(theta) - gain(theta - step),

    the pattern's own first difference -- and with the bins set to the tilt
    step, that difference is exactly what a bin-to-bin step of the fitted curve
    is. So the pattern is a cumulative sum of binned differences, and the
    propagation term follows by subtraction.

    Backfitting the two terms alternately is the textbook route and does not
    work here: each tile only ever sees a window of angles as wide as the tilt
    range, so the tile factor and the angle factor are near-collinear and the
    iteration collapses the pattern towards flat.

    The gauge is fixed by putting the pattern's peak at 0 dB, which makes
    ``base_db`` read as the level a tile sees at boresight.

    Args:
        sweep_data: The solved tilt sweep.
        elevation_deg: Depression angle per transmitter and tile, from
            :attr:`src.surrogate.features.SceneFeatures.elevation_deg`.

    Raises:
        ValueError: When a band's sweep holds fewer than two tilts, which
            leaves nothing to difference.
    """
    gains, bases, grids = [], [], []

    for band, tilts in enumerate(sweep_data.tilts):
        if len(tilts) < 2:
            raise ValueError(
                f"band {sweep_data.band_names[band]} has {len(tilts)} tilts; "
                "the pattern is recovered by differencing and needs at least two"
            )
        step = float(tilts[1] - tilts[0])
        edges = np.arange(-30.0, 90.0 + step, step)
        centres = 0.5 * (edges[:-1] + edges[1:])
        rsrp = sweep_data.rsrp[band]

        # Walked one tilt at a time rather than broadcast over the whole sweep.
        # The angle and its bin index are the same size as the maps, so holding
        # them for every tilt at once cost twenty times the sweep's own float32
        # data to produce two curves and a per-tile level.
        total = np.zeros(len(centres))
        count = np.zeros(len(centres))
        for index in range(len(tilts) - 1):
            difference = rsrp[index].astype(np.float64) - rsrp[index + 1]
            known = np.isfinite(difference)
            binned = _bin_of(elevation_deg, tilts[index], edges)[known]
            total += np.bincount(binned, weights=difference[known], minlength=len(centres))
            count += np.bincount(binned, minlength=len(centres))

        # A bin no pair of tilts reached contributes no step, which holds the
        # curve flat there rather than inventing a slope for it.
        rise = np.divide(total, count, out=np.zeros(len(centres)), where=count > 0)
        gain = np.cumsum(rise)
        gain -= gain.max()

        # The level each tile sits at, averaged over the tilts that reached it.
        # Second pass because it needs the gain the first one produced.
        level = np.zeros(rsrp.shape[1:])
        seen = np.zeros(rsrp.shape[1:], dtype=np.int64)
        for index in range(len(tilts)):
            bins = _bin_of(elevation_deg, tilts[index], edges)
            without_gain = rsrp[index].astype(np.float64) - gain[bins]
            finite = np.isfinite(without_gain)
            level += np.where(finite, without_gain, 0.0)
            seen += finite
        bases.append(np.divide(level, seen, out=np.full(seen.shape, np.nan), where=seen > 0))

        gains.append(gain)
        grids.append(centres)

    # Every band is re-sampled onto the finest grid so the pattern is one array.
    offset_deg = max(grids, key=len)
    return Pattern(
        band_names=sweep_data.band_names,
        offset_deg=offset_deg,
        gain_db=np.stack(
            [np.interp(offset_deg, grid, gain) for grid, gain in zip(grids, gains, strict=True)]
        ),
        base_db=np.stack(bases),
    )


@dataclass(frozen=True)
class Encoder:
    """Turns one radio map and one tilt change into what the model reads.

    Separate from :class:`TiltPairs` because two callers need it and only one
    of them is a dataset: training reads pairs out of the sweep, while
    :class:`src.surrogate.evaluator.SurrogateEvaluator` encodes an anchor map
    against whatever tilt the optimizer proposed. The channel order, the
    scaling and the no-path fill are what the weights were fitted to, so a
    second copy of them is a second thing to keep in step.

    Attributes:
        sweep: The sweep the tilt grids and the tile centres come from.
        pattern: The fitted pattern, for the correction and the no-path fill.
        elevation: Depression angle per transmitter, ``[n_tx, n_rows, n_cols]``.
        static: The tilt-independent channels, ``[n_tx, 7, n_rows, n_cols]``.
    """

    sweep: Sweep
    pattern: Pattern
    elevation: np.ndarray
    static: np.ndarray

    @classmethod
    def build(
        cls, sweep_data: Sweep, features: SceneFeatures, pattern: Pattern, cells: tuple[Cell, ...]
    ) -> Encoder:
        """Cache the channels that depend only on the scene and the masts."""
        return cls(
            sweep=sweep_data,
            pattern=pattern,
            elevation=features.elevation_deg,
            static=_static_channels(features, cells, sweep_data.tile_centres()),
        )

    def assemble(
        self,
        band: int,
        tx: int,
        source_db: np.ndarray,
        tilt_before: float,
        tilt_after: float,
    ) -> dict[str, np.ndarray]:
        """Everything the model reads for one transmitter, band and tilt change.

        Shared with :class:`src.surrogate.evaluator.SurrogateEvaluator` rather
        than reimplemented there: the channel order, the scaling and the
        no-path fill are what the weights were fitted to, and a second copy of
        them is a second thing to keep in step.

        Args:
            band: Index into the sweep's band axis.
            tx: Index into the sweep's transmitter axis.
            source_db: The map to move from, ``[n_rows, n_cols]`` in dBm, with
                ``nan`` where no path reached the tile.
            tilt_before: The tilt ``source_db`` was solved at.
            tilt_after: The tilt to predict.

        Returns:
            The input stack, the scalar conditioning, the angle-off-boresight
            maps, the source coverage, and the analytic baseline prediction.
        """
        batch = self.encode_batch(
            np.array([band]),
            np.array([tx]),
            np.asarray(source_db)[None],
            np.array([tilt_before], dtype=float),
            np.array([tilt_after], dtype=float),
        )
        return {key: value[0] for key, value in batch.items()}

    def encode_batch(
        self,
        bands: np.ndarray,
        txs: np.ndarray,
        source_db: np.ndarray,
        tilt_before: np.ndarray,
        tilt_after: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """The same encoding for many transmitter-band-tilt triples at once.

        The single implementation; :meth:`assemble` is one row of it. A search
        encodes all 36 dimensions per candidate, and doing that a row at a time
        cost more than the model did: the per-row path reached
        :meth:`Pattern.reference`, which builds a level for *every* transmitter
        and keeps one, so thirty-six rows built four hundred and thirty-two
        slices. Indexing ``base_db`` by the pairs removes that, and grouping the
        interpolation by band turns seventy-two ``np.interp`` calls into three.

        Args:
            bands: Band index per row, ``[n]``.
            txs: Transmitter index per row, ``[n]``.
            source_db: Maps to move from, ``[n, n_rows, n_cols]`` in dBm, with
                ``nan`` where no path reached the tile.
            tilt_before: The tilt each source was solved at, ``[n]``.
            tilt_after: The tilt to predict, ``[n]``.

        Returns:
            The five arrays :meth:`assemble` returns, each with a leading row
            axis.
        """
        elevation = self.elevation[txs]
        source = np.asarray(source_db, dtype=np.float64)
        has_path = np.isfinite(source)

        angle_before = elevation - tilt_before[:, None, None]
        angle_after = elevation - tilt_after[:, None, None]
        gain_before = self._gain(bands, angle_before)
        delta_gain = self._gain(bands, angle_after) - gain_before

        # Paired indexing, so only the rows asked for are built.
        level = self.pattern.base_db[bands, txs] + gain_before
        reference = np.where(np.isfinite(level), level, _FLOOR_DBM)
        filled = np.where(has_path, np.nan_to_num(source), reference)

        span = np.array([self.sweep.tilts[band][-1] for band in bands], dtype=float)
        return {
            "x": np.concatenate(
                [
                    ((filled - DB_CENTRE) / DB_SCALE)[:, None],
                    has_path[:, None].astype(np.float64),
                    (delta_gain / GAIN_SCALE)[:, None],
                    self.static[txs],
                ],
                axis=1,
            ),
            "cond": np.concatenate(
                [
                    (tilt_before / span)[:, None],
                    (tilt_after / span)[:, None],
                    ((tilt_after - tilt_before) / span)[:, None],
                    (bands[:, None] == np.arange(len(self.sweep.band_names))).astype(float),
                ],
                axis=1,
            ),
            "cond_map": np.stack([angle_before, angle_after], axis=1) / ANGLE_SCALE,
            "has_path": has_path[:, None].astype(np.float64),
            "baseline": (filled + delta_gain)[:, None],
        }

    def _gain(self, bands: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
        """The pattern at each row's angles, one interpolation per distinct band.

        ``np.interp`` takes a single curve, so the rows are grouped by band
        rather than walked. Within a group each transmitter still carries its
        own tilt, which is already in ``angle_deg``.
        """
        out = np.empty_like(angle_deg)
        for band in np.unique(bands):
            rows = bands == band
            out[rows] = self.pattern.gain(int(band), angle_deg[rows])
        return out


class TiltPairs(Dataset):
    """Ordered ``(tilt before, tilt after)`` pairs for one transmitter and band.

    A pair is one training example: the map at the first tilt, the scene, the
    two tilts, and the map at the second as the target. The model predicts a
    residual over ``rsrp_before + pattern.delta(...)``, so the target it is
    actually fitted to is what the analytic re-embedding gets wrong.

    The split is on the tilt axis, not the scene: the scene is fixed by design,
    so the only generalisation that means anything here is to tilts never
    solved. ``train`` pairs move between whole degrees; ``test`` pairs start on
    a whole degree -- as deployment always does, anchored on the committed
    tilts -- and land between them.

    Attributes:
        channels: Names of the input stack's channels, in order.
    """

    channels = (
        "rsrp_before",
        "has_path",
        "delta_gain",
        "elevation",
        "los_fraction",
        "sdf",
        "max_height",
        "cos_bearing",
        "sin_bearing",
        "log_distance",
    )

    def __init__(
        self,
        sweep_data: Sweep,
        features: SceneFeatures,
        pattern: Pattern,
        cells: tuple[Cell, ...],
        split: str = "train",
    ) -> None:
        """Enumerate the pairs of one split and cache the tilt-independent channels.

        Raises:
            ValueError: When ``split`` is neither ``"train"`` nor ``"test"``.
        """
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.encoder = Encoder.build(sweep_data, features, pattern, cells)
        self.sweep = sweep_data
        self.pattern = pattern
        self.elevation = features.elevation_deg
        self.index = _enumerate_pairs(sweep_data, split)

    def __len__(self) -> int:
        """Number of pairs in this split."""
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """Assemble one pair.

        Returns the input stack, the scalar conditioning, the two
        angle-off-boresight maps the spatial modulation reads, the coverage of
        the input map, the analytic baseline prediction, and the target map
        with its own coverage.
        """
        band, tx, before, after = self.index[item]
        inputs = self.encoder.assemble(
            band,
            tx,
            self.sweep.rsrp[band][before, tx],
            float(self.sweep.tilts[band][before]),
            float(self.sweep.tilts[band][after]),
        )

        target = self.sweep.rsrp[band][after, tx].astype(np.float64)
        return {
            **{key: _tensor(value) for key, value in inputs.items()},
            "target": _tensor(np.nan_to_num(target, nan=_FLOOR_DBM)[None]),
            "target_mask": _tensor(np.isfinite(target)[None].astype(np.float64)),
        }


def _tensor(array: np.ndarray) -> torch.Tensor:
    """One array as the float32 tensor the model runs on."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _static_channels(
    features: SceneFeatures, cells: tuple[Cell, ...], centres: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """The seven tilt-independent channels per transmitter, ``[n_tx, 7, n_rows, n_cols]``.

    Cached because they are a function of the scene and the mast alone, so
    every pair of a given transmitter reads the same arrays. Scales are round
    numbers of the units involved -- a 45 degree depression, a 100 m distance
    field, a 30 m mast -- not fitted statistics, so a rebuilt sweep cannot
    silently shift them.

    Bearing is carried as its cosine and sine rather than as an angle: the
    horizontal pattern is symmetric about boresight and wraps at 180 degrees,
    and a raw difference in degrees would put a discontinuity behind the mast.
    """
    centre_x, centre_y = centres
    stacked = []

    for index, cell in enumerate(cells):
        delta_x, delta_y = centre_x - cell.x, centre_y - cell.y
        # arctan2(dy, dx) is the convention Cell.azimuth_deg uses, so this
        # difference is the angle off this cell's boresight.
        off = np.arctan2(delta_y, delta_x) - np.radians(cell.azimuth_deg)
        # Floored at a metre, so the tile a mast stands on is not log10(0).
        distance = np.maximum(np.hypot(delta_x, delta_y), 1.0)
        stacked.append(
            np.stack(
                [
                    features.elevation_deg[index] / ANGLE_SCALE,
                    features.los_fraction[index],
                    features.mean_sdf_m / 100.0,
                    features.max_height_m / 30.0,
                    np.cos(off),
                    np.sin(off),
                    np.log10(distance) / 3.0,
                ]
            )
        )
    return np.stack(stacked)


def _enumerate_pairs(sweep_data: Sweep, split: str) -> np.ndarray:
    """Every ``(band, tx, before, after)`` of one split, as an index table."""
    rows = []
    for band, tilts in enumerate(sweep_data.tilts):
        whole = np.isclose(tilts, np.round(tilts))
        before_pool = np.flatnonzero(whole)
        after_pool = np.flatnonzero(whole if split == "train" else ~whole)
        for tx in range(len(sweep_data.tx_names)):
            for before in before_pool:
                for after in after_pool:
                    if before != after:
                        rows.append((band, tx, before, after))
    return np.array(rows, dtype=np.int64)


def build(cfg: DictConfig) -> tuple[Path, Path]:
    """Check the decomposition, cast the scene channels, solve the sweep.

    The script form of notebook 03a, in the order the notebook runs it: the
    decomposition check first, because a sweep that assumed it wrongly would be
    an hour spent on an unusable table.

    Raises:
        RuntimeError: When one cell's tilt moved another cell's map, which
            makes the per-slice surrogate unsound.
    """
    report = check_decomposition(cfg)
    print(
        f"decomposition: moving {report.moved_cell}/{report.band} changed the other "
        f"transmitters by at most {report.max_abs_delta_db:g} dB "
        f"(its own map moved {report.moved_max_abs_delta_db:.1f} dB)"
    )
    if not report.holds:
        raise RuntimeError(
            "tilts do not decompose per transmitter on this solver, so one sweep cannot "
            "stand in for every cell. The per-slice surrogate design does not survive this; "
            "re-plan before spending the sweep."
        )

    feature_path = build_features(cfg, cfg.surrogate.output.feature_file)
    sweep_path = sweep(cfg, float(cfg.surrogate.sweep.step_deg), cfg.surrogate.output.sweep_file)
    return sweep_path, feature_path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``task surrogate:dataset``."""
    build(cfg)


if __name__ == "__main__":
    main()
