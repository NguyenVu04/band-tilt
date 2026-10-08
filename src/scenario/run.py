"""Generate a scenario's synthetic data: UEs over time, the sector layout, the manifest.

Writes the files ``simulation.input`` names (the UE table, the sector table and
the manifest), which every later stage reads and real data can replace, plus
``scenario.output.record_file``, the generator's own record of what it drew.
Layouts use open ground in the loaded scene and remain fixed per scenario.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.load import write_json
from src.scenario import density, grid, sample, traffic
from src.scenario.density import DensitySpec
from src.scenario.grid import GridSpec
from src.scenario.layout import LayoutSpec, default_max_prb, default_tilts, generate_layout
from src.scenario.sample import UeSpec
from src.scenario.traffic import TrafficSpec
from src.simulation import scene as scene_module
from src.simulation import seeds
from src.simulation.scene import SceneSpec
from src.tracking import log_stage

# Output paths do not affect scenario identity.
_IDENTITY_KEYS = ("grid", "ue", "time", "density", "layout", "seed")


def scenario_id(cfg: DictConfig) -> str:
    """A short, stable id for the scenario this config describes.

    Derived from the settings that determine the world and its population, so
    two runs share an id exactly when they are the same scenario.
    """
    identity = {key: _resolved(cfg.scenario[key]) for key in _IDENTITY_KEYS}
    identity["scene_file"] = str(cfg.simulation.input.scene_file)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "scn_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _resolved(node: Any) -> Any:
    """Plain Python for a config value, whether it is a container or a scalar."""
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def generate(cfg: DictConfig) -> tuple[Path, Path, Path, Path]:
    """Run the scenario stage. Returns ``(ue_file, sectors_file, manifest_file, record_file)``.

    Raises:
        ValueError: As :func:`src.scenario.layout.generate_layout`.
    """
    grid_spec = GridSpec.from_config(cfg)
    ue = UeSpec.from_config(cfg)
    density_spec = DensitySpec.from_config(cfg)
    traffic_spec = TrafficSpec.from_config(cfg)
    layout = LayoutSpec.from_config(cfg)

    scene, bounds = scene_module.load(SceneSpec.from_config(cfg))

    raster = grid.build(scene.mi_scene, bounds, grid_spec, seeds.stream(cfg.scenario.seed, "scene"))
    field = density.field(raster, density_spec, seeds.stream(cfg.scenario.seed, "density"))
    schedule = traffic.build(
        traffic_spec,
        ue.count_range,
        len(field.hotspots),
        density_spec.hotspot_mass_fraction,
        seeds.stream(cfg.scenario.seed, "traffic"),
    )
    interval, t_s, x, y, component = sample.sample_positions(
        scene.mi_scene,
        bounds,
        raster,
        field,
        schedule,
        grid_spec,
        seeds.stream(cfg.scenario.seed, "sample"),
    )

    # The layout is the step that can fail on config; draw everything before writing
    # so a failure leaves the previous scenario's files intact, and write the
    # manifest last because it is what marks the set complete.
    sectors = generate_layout(
        scene.mi_scene,
        bounds,
        raster,
        layout,
        grid_spec,
        default_tilts(cfg),
        default_max_prb(cfg),
    )

    ue_file = sample.write_csv(
        Path(cfg.simulation.input.ue_file), interval, t_s, x, y, component, raster, ue
    )
    sectors_file = Path(cfg.simulation.input.sectors_file)
    sectors_file.parent.mkdir(parents=True, exist_ok=True)
    sectors.to_csv(sectors_file, index=False, float_format="%.3f")
    record_file = write_json(
        _record(cfg, bounds, field, schedule, x), cfg.scenario.output.record_file
    )
    manifest_file = write_json(_manifest(cfg, raster, schedule), cfg.simulation.input.manifest_file)

    eligible = density.eligible_tiles(raster)
    tile_col, tile_row = raster.tile_indices(x, y)
    hotspot_mass = schedule.component_mass[:, 1:].sum(axis=1)
    print(f"scenario: {scenario_id(cfg)}")
    print(
        f"grid:     {raster.n_cols} x {raster.n_rows} tiles, "
        f"{int(np.count_nonzero(eligible))} eligible"
    )
    print(f"hotspots: {len(field.hotspots)}")
    print(
        f"time:     {schedule.n_intervals} intervals of {schedule.interval_s:.0f} s, "
        f"{int(schedule.count.min())}-{int(schedule.count.max())} UEs each"
    )
    print(
        f"demand:   hotspots hold {hotspot_mass.min():.0%} to {hotspot_mass.max():.0%} "
        "of the population across intervals"
    )
    print(f"ue:       {x.size} rows at z={ue.height_m} m")
    print(
        "densest decile holds "
        f"{sample.densest_decile_share(tile_col, tile_row, raster, eligible):.1%} of UEs"
    )
    print(
        f"sectors:    {sectors['sector'].nunique()} over {sectors['node'].nunique()} nodes, "
        f"masts {layout.mast_height_m} m on tiles at least {layout.min_free_fraction:.0%} open"
    )
    print(f"csv:      {ue_file}, {sectors_file}")
    print(f"manifest: {manifest_file}")
    print(f"record:   {record_file}")
    return ue_file, sectors_file, manifest_file, record_file


def _manifest(cfg: DictConfig, raster: grid.Raster, schedule: traffic.Schedule) -> dict[str, Any]:
    """The consumer contract: the scenario id, the grid and the time schedule.

    Every key here is read downstream (:mod:`src.simulation.radio`,
    :mod:`src.data`); a real dataset supplies the same keys.
    """
    return {
        "scenario_id": scenario_id(cfg),
        "grid": {
            "origin_x": raster.origin_x,
            "origin_y": raster.origin_y,
            "tile_size_m": raster.tile_size_m,
            "n_cols": raster.n_cols,
            "n_rows": raster.n_rows,
        },
        "time": {
            "interval_s": schedule.interval_s,
            "t_s": schedule.t_s.tolist(),
        },
    }


def _record(
    cfg: DictConfig,
    bounds: scene_module.SceneBounds,
    field: density.DensityField,
    schedule: traffic.Schedule,
    x: np.ndarray,
) -> dict[str, Any]:
    """What the generator drew and from which settings; no stage outside it reads this."""
    return {
        "scenario_id": scenario_id(cfg),
        "seed": int(cfg.scenario.seed),
        "scene_file": str(cfg.simulation.input.scene_file),
        "layout": OmegaConf.to_container(cfg.scenario.layout, resolve=True),
        "grid": {"launch_z": bounds.launch_z},
        # Store inputs to the density, not its deterministic tile weights.
        "density": {
            "spec": OmegaConf.to_container(cfg.scenario.density, resolve=True),
            "hotspots": [dataclasses.asdict(hotspot) for hotspot in field.hotspots],
        },
        "time": {
            "spec": OmegaConf.to_container(cfg.scenario.time, resolve=True),
            "n_intervals": schedule.n_intervals,
            "count": schedule.count.tolist(),
            "component_mass": schedule.component_mass.tolist(),
            "phase_rad": schedule.phase_rad.tolist(),
        },
        "ue": {
            "rows": int(x.size),
            "count_range": list(cfg.scenario.ue.count_range),
            "height_m": float(cfg.scenario.ue.height_m),
        },
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the scenario. Entry point for ``task simulation:scenario``."""
    log_stage(cfg, "simulation_scenario", groups=["simulation"], outputs=generate(cfg))


if __name__ == "__main__":
    main()
