"""Ray-trace one clean radio map per band.

Reads the stored scenario manifest and writes RSRP and SINR on its UE grid,
both per resource element, as the solver's :class:`sionna.rt.RadioMap` reports
them. Nothing here
redraws the scenario: a map must describe the population already on disk.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
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
from src.tracking import log_stage

# A tile no ray reached is unknown, not weak: NaN keeps it out of every
# reduction instead of competing with the finite values a weak path leaves.
_NO_PATH = np.nan


@dataclass(frozen=True)
class Band:
    """One carrier.

    Tilt is deliberately absent: it belongs to the cell-band pair, so it lives
    on the cell (:class:`src.core.cell.Cell`). A band-level
    tilt would force every cell of a band to point alike.

    Attributes:
        name: Identifies the band in the cell tilt table and the radio map.
            The one place the band's identity is spelled.
        frequency_hz: Carrier frequency.
        bandwidth_hz: Channel bandwidth. Fixes N_RB and so ``max_prb``; the
            solver does not read it.
        scs_hz: Subcarrier spacing: the bandwidth of one resource element,
            and so the solver's thermal-noise bandwidth.
    """

    name: str
    frequency_hz: float
    bandwidth_hz: float
    scs_hz: float

    @classmethod
    def from_config(cls, entry: DictConfig) -> Band:
        """Read one entry of ``simulation.radio_map.bands``."""
        return cls(
            name=str(entry.name),
            frequency_hz=float(entry.frequency),
            bandwidth_hz=float(entry.bandwidth),
            scs_hz=float(entry.scs_hz),
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


@dataclass(frozen=True)
class RadioSetup:
    """Everything a solve reads from the config and the manifest; none of it depends on tilt.

    Shared by :func:`solve` and :class:`src.optim.evaluator.Evaluator`, so the
    baseline map and every candidate are solved under the same settings.

    Attributes:
        scenario_id: The scenario the manifest describes.
        grid_meta: The manifest's ``grid`` block: origin, tile size, shape.
        bands: ``simulation.radio_map.bands``, in radio-map band-axis order.
        solver: The ray-tracing settings.
        height_m: UE height, where the map is measured.
        power_dbm: Per-resource-element reference power.
    """

    scenario_id: str
    grid_meta: dict[str, Any]
    bands: tuple[Band, ...]
    solver: SolverSpec
    height_m: float
    power_dbm: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RadioSetup:
        """Read the manifest and ``simulation``; raises as :func:`read_manifest`."""
        manifest = read_manifest(cfg)
        return cls(
            scenario_id=str(manifest["scenario_id"]),
            grid_meta=manifest["grid"],
            bands=tuple(Band.from_config(entry) for entry in cfg.simulation.radio_map.bands),
            solver=SolverSpec.from_config(cfg),
            height_m=float(cfg.simulation.ue.height_m),
            power_dbm=float(cfg.simulation.antenna.power_rs),
        )


def load_scene(cfg: DictConfig, cells: tuple[Cell, ...]) -> Any:
    """Load the scene, warn about masts off open ground, and attach the antenna arrays."""
    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))
    for problem in transmitter.validate(
        scene.mi_scene, bounds, cells, GridSpec.from_config(cfg).free_height_tol_m
    ):
        print(f"WARNING transmitter {problem}")
    configure_arrays(scene, cfg)
    return scene


def solve_bands(
    scene: Any, cells: tuple[Cell, ...], setup: RadioSetup, solver_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Solve every band with :func:`solve_band`.

    Returns RSRP and SINR stacked to ``[n_band, n_tx, n_rows, n_cols]``, the
    solver's tile centres, and each band's elapsed seconds.
    """
    rsrp_maps, sinr_maps, elapsed = [], [], []
    centres = np.empty(0)
    for band in setup.bands:
        rsrp, sinr, seconds, centres = solve_band(
            scene,
            cells,
            band,
            setup.solver,
            solver_seed,
            setup.grid_meta,
            setup.height_m,
            setup.power_dbm,
        )
        rsrp_maps.append(rsrp)
        sinr_maps.append(sinr)
        elapsed.append(seconds)
    return np.stack(rsrp_maps), np.stack(sinr_maps), centres, elapsed


def solve(cfg: DictConfig) -> Path:
    """Solve every band's radio map and write them. Returns the output path."""
    setup = RadioSetup.from_config(cfg)
    cells = transmitter.load(cfg)
    _check_tilt_table(cells, setup.bands)
    solver_seed = seeds.stream(cfg, "solver")

    scene = load_scene(cfg, cells)
    rsrp, sinr, centres, elapsed = solve_bands(scene, cells, setup, solver_seed)
    for band, band_rsrp, seconds in zip(setup.bands, rsrp, elapsed, strict=True):
        tilts = [cell.tilt_for(band.name).baseline_deg for cell in cells]
        # Reduce over the reached tiles only: a tile no ray found is all-NaN,
        # and nanmax over one warns rather than simply meaning "no coverage".
        served = np.isfinite(band_rsrp).any(axis=0)
        best = np.nanmax(band_rsrp[:, served], axis=0)
        span = f"{best.min():6.1f} to {best.max():6.1f} dBm" if best.size else "none"
        print(
            f"{band.name:>8s}  tilt {min(tilts):4.1f}-{max(tilts):4.1f} deg  {seconds:6.1f}s  "
            f"tiles reached {served.mean():6.1%}  best server {span}"
        )

    path = write_radio_map(
        cfg.simulation.output.radio_map_file,
        rsrp=rsrp,
        sinr=sinr,
        bands=setup.bands,
        cells=cells,
        grid_meta=setup.grid_meta,
        solver_spec=setup.solver,
        solver_seed=solver_seed,
        height_m=setup.height_m,
        power_dbm=setup.power_dbm,
        scenario_id=setup.scenario_id,
        centres=centres,
    )
    print(f"radio map: {path}  shape {rsrp.shape} [band, tx, row, col]")
    return path


def baseline_tilts(cells: tuple[Cell, ...], band_names: Sequence[str]) -> np.ndarray:
    """Each cell's baseline tilt, ``[n_band, n_tx]``: a radio map's ``tilt_deg`` array.

    Raises:
        KeyError: When a cell carries no tilt for one of the bands.
    """
    return np.array([[cell.tilt_for(band).baseline_deg for cell in cells] for band in band_names])


def write_radio_map(
    path: str | Path,
    *,
    rsrp: np.ndarray,
    sinr: np.ndarray,
    bands: tuple[Band, ...],
    cells: tuple[Cell, ...],
    grid_meta: dict[str, Any],
    solver_spec: SolverSpec,
    solver_seed: int,
    height_m: float,
    power_dbm: float,
    scenario_id: str,
    centres: np.ndarray,
) -> Path:
    """Write one radio map archive; the one schema every map in the project uses.

    ``rsrp`` and ``sinr`` are ``[n_band, n_tx, n_rows, n_cols]`` in dBm and dB.
    Creates the parent directory. Returns the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rsrp_dbm=rsrp.astype(np.float32),
        sinr_db=sinr.astype(np.float32),
        band_hz=np.array([band.frequency_hz for band in bands]),
        band_label=np.array([band.name for band in bands]),
        # One tilt per cell-band pair, [band, tx], matching rsrp_dbm's leading
        # two axes. This is the configuration the map was solved at, so a stored
        # map carries the decision vector that produced it.
        tilt_deg=baseline_tilts(cells, [band.name for band in bands]),
        tx_name=np.array([cell.name for cell in cells]),
        origin_x=grid_meta["origin_x"],
        origin_y=grid_meta["origin_y"],
        tile_size_m=grid_meta["tile_size_m"],
        n_cols=grid_meta["n_cols"],
        n_rows=grid_meta["n_rows"],
        ue_height_m=height_m,
        scenario_id=scenario_id,
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
        solver_seed=solver_seed,
        power_dbm=power_dbm,
        # The solver's own tile centres, so alignment against the UE grid can be
        # checked rather than assumed. A silent half-tile offset would corrupt
        # every RSRP lookup while leaving the file entirely plausible.
        tile_centre=centres,
    )
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
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Solve one band.

    Returns RSRP ``[n_tx, n_rows, n_cols]`` in dBm, SINR of the same shape in
    dB, the elapsed seconds, and the solver's own tile centres. Both maps are
    NaN where no path reached the tile.

    SINR is :attr:`sionna.rt.RadioMap.sinr`: every other transmitter in the
    scene is interference at full power, plus ``k * T * scs_hz`` noise, all
    per resource element. The scene holds only this band's transmitters, so the
    interference is co-band.
    """
    import mitsuba as mi
    from sionna.rt import RadioMapSolver

    # Before the frequency, never after. Setting the frequency runs every
    # registered material's update callback, which would both raise on a carrier
    # outside the material's published ITU range and overwrite this scenario's
    # draw. Installing first switches those callbacks off.
    materials.install(scene, band.frequency_hz)
    scene.frequency = band.frequency_hz
    # Sionna-RT's noise is temperature * k * scene.bandwidth. Every transmitter
    # radiates power_rs per RE, so the noise must be one RE's too: the SCS, not
    # the channel bandwidth, which would add 10*log10(N_RB * 12) dB of noise.
    # Per-RE signal over per-RE noise plus interference is SS-SINR's form
    # (TS 38.215 5.1.5).
    scene.bandwidth = band.scs_hz
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
    # the per-resource-element reference power this reads directly as RSRP, the
    # per-RE power of TS 38.215 5.1.1.
    rss = np.asarray(radio_map.rss, dtype=np.float64)
    sinr_linear = np.asarray(radio_map.sinr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsrp = 10.0 * np.log10(rss) + 30.0
        sinr = 10.0 * np.log10(sinr_linear)
    centres = np.asarray(radio_map.cell_centers, dtype=np.float64)
    # Materialising the lazy arrays must be inside the solver timer.
    elapsed = time.perf_counter() - started
    reached = np.isfinite(rsrp)
    return (
        np.where(reached, rsrp, _NO_PATH),
        np.where(reached, sinr, _NO_PATH),
        elapsed,
        centres,
    )


def _check_tilt_table(cells: tuple[Cell, ...], bands: tuple[Band, ...]) -> None:
    """Check every cell carries a tilt and a PRB limit for every band.

    Checked once, up front, so a mismatched table names every gap rather than
    failing on whichever band happens to be solved first.

    Raises:
        ValueError: When any cell-band pair has no tilt or no ``max_prb``.
    """
    missing = [
        f"{cell.name}/{band.name}"
        for cell in cells
        for band in bands
        if band.name not in cell.tilt or band.name not in cell.max_prb
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} cell-band pairs have no tilt or max_prb: {', '.join(missing[:8])}"
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


def read_manifest(cfg: DictConfig) -> dict[str, Any]:
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
    log_stage(cfg, "simulation_radio", groups=["simulation"], outputs=[solve(cfg)])


if __name__ == "__main__":
    main()
