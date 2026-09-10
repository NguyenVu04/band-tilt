"""Ray-trace one clean radio map per band.

Rebuilds the scenario and writes RSRP on its UE grid.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from src.core.cell import Cell
from src.simulation import materials, seeds, transmitter
from src.simulation import scenario as scenario_module
from src.simulation import scene as scene_module
from src.simulation.grid import GridSpec
from src.simulation.scene import SceneSpec

# Use NaN for tiles no ray reached; weak paths retain finite values.
_NO_PATH = np.nan


@dataclass(frozen=True)
class Band:
    """One carrier.

    Tilt is deliberately absent: it belongs to the cell-band pair, so it lives
    on the cell (:class:`src.core.cell.Cell`). A band-level
    tilt would force every cell of a band to point alike.

    Attributes:
        name: Identifies the band in the cell tilt table and in the MDT
            column names. The one place the band's identity is spelled.
        frequency_hz: Carrier frequency.
        bandwidth_hz: Transmission bandwidth; with temperature it fixes the
            thermal noise power.
    """

    name: str
    frequency_hz: float
    bandwidth_hz: float

    @classmethod
    def from_config(cls, entry: DictConfig) -> Band:
        """Read one entry of ``simulation.radio_map.bands``."""
        return cls(
            name=str(entry.name),
            frequency_hz=float(entry.frequency),
            bandwidth_hz=float(entry.bandwidth),
        )


@dataclass(frozen=True)
class SolverSpec:
    """Ray-tracing settings. These define the ground truth, not merely its cost.

    Every propagation mechanism the solver can model is named here rather than
    left to its library default. A default that changes between sionna-rt
    releases would silently change what "ground truth" means, and a flag that
    is never written down cannot be recorded alongside the map it produced.

    Attributes:
        samples_per_tx: Rays shot per transmitter.
        max_depth: Maximum number of interactions along a path.
        los: Include the direct path.
        specular_reflection: Include mirror-like reflection.
        diffuse_reflection: Include diffuse scattering. Without it the
            materials' scattering coefficient is inert.
        refraction: Include transmission through surfaces.
        diffraction: Include diffracted paths. Matters more as frequency falls.
        edge_diffraction: Also diffract at the edges of finite surfaces, not
            only at wedges.
        diffraction_lit_region: Keep diffracted paths where a direct path
            already exists, instead of only in shadow.
        rr_depth: Depth at which Russian-roulette path termination starts;
            ``-1`` disables it. The cost lever to reach for before ``max_depth``,
            since it shortens weak paths rather than truncating every path.
        rr_prob: Survival probability once Russian roulette is active.
        temperature_k: Scene temperature, for the thermal noise power.
    """

    samples_per_tx: int
    max_depth: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    edge_diffraction: bool
    diffraction_lit_region: bool
    rr_depth: int
    rr_prob: float
    temperature_k: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SolverSpec:
        """Read ``simulation.radio_map``."""
        radio_map = cfg.simulation.radio_map
        return cls(
            samples_per_tx=int(radio_map.samples_per_tx),
            max_depth=int(radio_map.max_depth),
            los=bool(radio_map.los),
            specular_reflection=bool(radio_map.specular_reflection),
            diffuse_reflection=bool(radio_map.diffuse_reflection),
            refraction=bool(radio_map.refraction),
            diffraction=bool(radio_map.diffraction),
            edge_diffraction=bool(radio_map.edge_diffraction),
            diffraction_lit_region=bool(radio_map.diffraction_lit_region),
            rr_depth=int(radio_map.rr_depth),
            rr_prob=float(radio_map.rr_prob),
            temperature_k=float(radio_map.temperature),
        )


def solve(cfg: DictConfig) -> Path:
    """Solve every band's radio map and write them. Returns the output path."""
    manifest = _read_manifest(cfg)
    grid_meta = manifest["grid"]
    cells = transmitter.load(cfg)
    bands = tuple(Band.from_config(entry) for entry in cfg.simulation.radio_map.bands)
    solver_spec = SolverSpec.from_config(cfg)
    power_dbm = float(cfg.simulation.antenna.power_rs)
    height_m = float(cfg.simulation.ue.height_m)

    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))

    _check_tilt_table(cells, bands)

    problems = transmitter.validate(
        scene.mi_scene, bounds, cells, GridSpec.from_config(cfg).free_height_tol_m
    )
    for problem in problems:
        print(f"WARNING transmitter {problem}")

    configure_arrays(scene, cfg)
    maps = []
    centres = None
    for band in bands:
        rsrp, elapsed, centres, _radio_map = solve_band(
            scene,
            cells,
            band,
            solver_spec,
            seeds.stream(cfg, "solver"),
            grid_meta,
            height_m,
            power_dbm,
        )
        maps.append(rsrp)
        # Reduce over the reached tiles only: a tile no ray found is all-NaN,
        # and nanmax over one warns rather than simply meaning "no coverage".
        served = np.isfinite(rsrp).any(axis=0)
        best = np.nanmax(rsrp[:, served], axis=0)
        tilts = [cell.tilt_for(band.name).baseline_deg for cell in cells]
        print(
            f"{band.name:>8s}  tilt {min(tilts):4.1f}-{max(tilts):4.1f} deg  {elapsed:6.1f}s  "
            f"tiles reached {served.mean():6.1%}  "
            f"best server {best.min():6.1f} to {best.max():6.1f} dBm"
        )

    path = Path(cfg.simulation.output.radio_map_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rsrp_dbm=np.stack(maps).astype(np.float32),
        band_hz=np.array([band.frequency_hz for band in bands]),
        band_label=np.array([band.name for band in bands]),
        # One tilt per cell-band pair, [band, tx], matching rsrp_dbm's leading
        # two axes. This is the configuration the map was solved at, so a stored
        # map carries the decision vector that produced it.
        tilt_deg=np.array(
            [[cell.tilt_for(band.name).baseline_deg for cell in cells] for band in bands]
        ),
        tx_name=np.array([cell.name for cell in cells]),
        origin_x=grid_meta["origin_x"],
        origin_y=grid_meta["origin_y"],
        tile_size_m=grid_meta["tile_size_m"],
        n_cols=grid_meta["n_cols"],
        n_rows=grid_meta["n_rows"],
        ue_height_m=height_m,
        scenario_id=manifest["scenario_id"],
        # The ray-tracing settings define what this ground truth IS, not merely
        # what it cost, so they travel with it. Without them a stray archive
        # cannot be told apart from one solved at a different fidelity, and two
        # such files must never be mixed into one dataset.
        samples_per_tx=solver_spec.samples_per_tx,
        max_depth=solver_spec.max_depth,
        los=solver_spec.los,
        specular_reflection=solver_spec.specular_reflection,
        diffuse_reflection=solver_spec.diffuse_reflection,
        refraction=solver_spec.refraction,
        diffraction=solver_spec.diffraction,
        edge_diffraction=solver_spec.edge_diffraction,
        diffraction_lit_region=solver_spec.diffraction_lit_region,
        rr_depth=solver_spec.rr_depth,
        rr_prob=solver_spec.rr_prob,
        solver_seed=seeds.stream(cfg, "solver"),
        temperature_k=solver_spec.temperature_k,
        bandwidth_hz=np.array([band.bandwidth_hz for band in bands]),
        power_dbm=power_dbm,
        # The solver's own tile centres, so alignment against the UE grid can be
        # checked rather than assumed. A silent half-tile offset would corrupt
        # every RSRP lookup while leaving the file entirely plausible.
        tile_centre=centres,
    )
    print(f"radio map: {path}  shape {np.stack(maps).shape} [band, tx, row, col]")
    return path


def solve_band(
    scene: Any,
    cells: tuple[Cell, ...],
    band: Band,
    spec: SolverSpec,
    solver_seed: int,
    grid_meta: dict[str, Any],
    height_m: float,
    power_dbm: float,
) -> tuple[np.ndarray, float, np.ndarray, Any]:
    """Solve one band.

    Public so that a renderer can re-solve a band against the same scene and
    keep the live :class:`sionna.rt.RadioMap` the solver returns — the array
    written by :func:`solve` is a numpy copy that cannot be rendered with
    :meth:`sionna.rt.Scene.render`.

    Returns RSRP ``[n_tx, n_rows, n_cols]`` in dBm, the elapsed seconds, the
    solver's own tile centres for the alignment check, and the solver's
    :class:`sionna.rt.RadioMap`.
    """
    import mitsuba as mi
    from sionna.rt import RadioMapSolver

    # Before the frequency, never after. Setting the frequency runs every
    # registered material's update callback, which would both raise on a carrier
    # outside the material's published ITU range and overwrite this scenario's
    # draw. Installing first switches those callbacks off.
    materials.install(scene, band.frequency_hz)
    scene.frequency = band.frequency_hz
    scene.bandwidth = band.bandwidth_hz
    scene.temperature = spec.temperature_k

    for cell in cells:
        if scene.get(cell.name) is not None:
            scene.remove(cell.name)
    transmitter.build(scene, cells, band.name, power_dbm)

    size_x = grid_meta["n_cols"] * grid_meta["tile_size_m"]
    size_y = grid_meta["n_rows"] * grid_meta["tile_size_m"]

    started = time.perf_counter()
    radio_map = RadioMapSolver()(
        scene,
        # Given explicitly so the map's tiles coincide with the grid the UEs
        # were binned into. Letting the solver default its own extent would
        # make every RSRP lookup silently wrong.
        center=mi.Point3f(
            grid_meta["origin_x"] + 0.5 * size_x,
            grid_meta["origin_y"] + 0.5 * size_y,
            height_m,
        ),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(size_x, size_y),
        # sionna-rt calls a map square a "cell"; it is our tile.
        cell_size=mi.Point2f(grid_meta["tile_size_m"], grid_meta["tile_size_m"]),
        samples_per_tx=spec.samples_per_tx,
        max_depth=spec.max_depth,
        los=spec.los,
        specular_reflection=spec.specular_reflection,
        diffuse_reflection=spec.diffuse_reflection,
        refraction=spec.refraction,
        diffraction=spec.diffraction,
        edge_diffraction=spec.edge_diffraction,
        diffraction_lit_region=spec.diffraction_lit_region,
        rr_depth=spec.rr_depth,
        rr_prob=spec.rr_prob,
        # Passed explicitly: the solver seeds its own Monte-Carlo stream and
        # otherwise runs at a fixed library default, so without this the map
        # would ignore simulation.seed entirely.
        seed=solver_seed,
    )
    # rss is path gain times transmit power, in watts, so with power_dbm set to
    # the per-resource-element reference power this reads directly as RSRP.
    rss = np.asarray(radio_map.rss, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsrp = 10.0 * np.log10(rss) + 30.0
    centres = np.asarray(radio_map.cell_centers, dtype=np.float64)
    # Materialising the lazy arrays must be inside the solver timer.
    elapsed = time.perf_counter() - started
    return np.where(np.isfinite(rsrp), rsrp, _NO_PATH), elapsed, centres, radio_map


def _check_tilt_table(cells: tuple[Cell, ...], bands: tuple[Band, ...]) -> None:
    """Check every cell carries a tilt for every band.

    Checked once, up front, so a mismatched table names every gap rather than
    failing on whichever band happens to be solved first.

    Raises:
        ValueError: When any cell-band pair has no tilt.
    """
    missing = [
        f"{cell.name}/{band.name}" for cell in cells for band in bands if band.name not in cell.tilt
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} cell-band pairs have no tilt: {', '.join(missing[:8])}"
            f"{' ...' if len(missing) > 8 else ''}. Every cell needs one entry per band in "
            "simulation.radio_map.bands; re-run `task simulation:layout` if the bands changed."
        )


def configure_arrays(scene: Any, cfg: DictConfig) -> None:
    """Attach the transmit and receive arrays described by ``simulation.antenna``."""
    from sionna.rt import PlanarArray

    for attribute, entry in (
        ("tx_array", cfg.simulation.antenna.transmitter),
        ("rx_array", cfg.simulation.antenna.receiver),
    ):
        setattr(
            scene,
            attribute,
            PlanarArray(
                num_rows=int(entry.num_rows),
                num_cols=int(entry.num_cols),
                vertical_spacing=float(entry.vertical_spacing),
                horizontal_spacing=float(entry.horizontal_spacing),
                pattern=str(entry.pattern),
                polarization=str(entry.polarization),
            ),
        )


def _read_manifest(cfg: DictConfig) -> dict[str, Any]:
    """Read the scenario manifest and check it matches this config.

    Raises:
        FileNotFoundError: When the scenario stage has not been run.
        ValueError: When the manifest describes a different scenario, which
            means the config changed after the UEs were drawn and the radio map
            would not correspond to them.
    """
    path = Path(cfg.simulation.output.manifest_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"No scenario manifest at {path}. Run `task simulation:scenario` first."
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = scenario_module.scenario_id(cfg)
    if manifest["scenario_id"] != expected:
        raise ValueError(
            f"{path} describes scenario {manifest['scenario_id']}, but this config is "
            f"{expected}. The scenario changed after the UEs were drawn; re-run "
            "`task simulation:scenario`."
        )
    return manifest


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Solve the radio maps. Entry point for ``task simulation:radio``.

    Example:
        $ task simulation:radio -- simulation.radio_map.samples_per_tx=100000
    """
    solve(cfg)


if __name__ == "__main__":
    main()
