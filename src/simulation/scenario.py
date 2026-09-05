"""Build a perturbed scenario, draw UEs, and write its manifest."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.simulation import density, grid, perturb, sample, seeds, traffic
from src.simulation import scene as scene_module
from src.simulation.density import DensitySpec
from src.simulation.grid import GridSpec
from src.simulation.perturb import PerturbSpec
from src.simulation.sample import UeSpec
from src.simulation.scene import SceneSpec
from src.simulation.traffic import TrafficSpec

# Output paths do not affect scenario identity.
_IDENTITY_KEYS = (
    "scene",
    "area",
    "grid",
    "ue",
    "time",
    "density",
    "perturbation",
    "materials",
    "seed",
)


def scenario_id(cfg: DictConfig) -> str:
    """A short, stable id for the scenario this config describes.

    Derived from the settings that determine the world and its population, so
    two runs share an id exactly when they are the same scenario. Train,
    validation and test split between whole scenarios, and this is the key they
    split on.
    """
    identity = {key: _resolved(cfg.simulation[key]) for key in _IDENTITY_KEYS}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "scn_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _resolved(node: Any) -> Any:
    """Plain Python for a config value, whether it is a container or a scalar."""
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def generate(cfg: DictConfig) -> tuple[Path, Path]:
    """Run the scenario stage. Returns ``(ue_file, manifest_file)``."""
    grid_spec = GridSpec.from_config(cfg)
    ue = UeSpec.from_config(cfg)
    density_spec = DensitySpec.from_config(cfg)
    traffic_spec = TrafficSpec.from_config(cfg)

    scene, delivered = scene_module.load(SceneSpec.from_config(cfg))
    report = perturb.apply(scene, PerturbSpec.from_config(cfg), seeds.stream(cfg, "scene"))

    # Preserve the delivered grid frame; only raise the ray launch height.
    perturbed = scene_module.bounds_of(scene)
    bounds = dataclasses.replace(delivered, max_z=max(delivered.max_z, perturbed.max_z))
    roi = bounds.inset(float(cfg.simulation.area.margin_m))

    raster = grid.build(scene.mi_scene, bounds, grid_spec, seeds.stream(cfg, "scene"))
    mask = grid.roi_mask(raster, roi)
    field = density.field(raster, density_spec, seeds.stream(cfg, "density"), mask)
    schedule = traffic.build(
        traffic_spec,
        ue.count_range,
        len(field.hotspots),
        density_spec.hotspot_mass_fraction,
        seeds.stream(cfg, "traffic"),
    )
    interval, x, y, component = sample.sample_positions(
        scene.mi_scene,
        bounds,
        roi,
        raster,
        field,
        schedule,
        grid_spec,
        seeds.stream(cfg, "sample"),
    )

    ue_file = sample.write_csv(
        Path(cfg.simulation.output.ue_file), interval, schedule, x, y, component, raster, ue
    )
    manifest_file = _write_manifest(cfg, report, raster, bounds, roi, field, schedule, x)

    eligible = density.eligible_cells(raster, mask)
    cell_col, cell_row = raster.cell_indices(x, y)
    hotspot_mass = schedule.component_mass[:, 1:].sum(axis=1)
    print(f"scenario: {scenario_id(cfg)}")
    print(f"removed:  {len(report.removed)} buildings   jittered: {len(report.jittered)}")
    print(
        f"grid:     {raster.n_cols} x {raster.n_rows} cells, "
        f"{int(np.count_nonzero(eligible))} eligible inside a "
        f"{cfg.simulation.area.margin_m} m margin"
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
        f"{sample.densest_decile_share(cell_col, cell_row, raster, eligible):.1%} of UEs"
    )
    print(f"csv:      {ue_file}")
    print(f"manifest: {manifest_file}")
    return ue_file, manifest_file


def _write_manifest(
    cfg: DictConfig,
    report: perturb.PerturbReport,
    raster: grid.Raster,
    bounds: scene_module.SceneBounds,
    roi: scene_module.SceneBounds,
    field: density.DensityField,
    schedule: traffic.Schedule,
    x: np.ndarray,
) -> Path:
    """Record what this scenario is, so it can be regenerated and split on."""
    path = Path(cfg.simulation.output.manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": scenario_id(cfg),
        "seed": int(cfg.simulation.seed),
        "scene": OmegaConf.to_container(cfg.simulation.scene, resolve=True),
        "area": {
            "margin_m": float(cfg.simulation.area.margin_m),
            "min_x": roi.min_x,
            "max_x": roi.max_x,
            "min_y": roi.min_y,
            "max_y": roi.max_y,
        },
        "perturbation": {
            "removed": list(report.removed),
            "jittered": list(report.jittered),
            "spec": OmegaConf.to_container(cfg.simulation.perturbation, resolve=True),
        },
        "materials": OmegaConf.to_container(cfg.simulation.materials, resolve=True),
        "grid": {
            "origin_x": raster.origin_x,
            "origin_y": raster.origin_y,
            "cell_size_m": raster.cell_size_m,
            "n_cols": raster.n_cols,
            "n_rows": raster.n_rows,
            "launch_z": bounds.launch_z,
        },
        # Store inputs to the density, not its deterministic cell weights.
        "density": {
            "spec": OmegaConf.to_container(cfg.simulation.density, resolve=True),
            "hotspots": [dataclasses.asdict(hotspot) for hotspot in field.hotspots],
        },
        "time": {
            "spec": OmegaConf.to_container(cfg.simulation.time, resolve=True),
            "n_intervals": schedule.n_intervals,
            "t_s": schedule.t_s.tolist(),
            "count": schedule.count.tolist(),
            "component_mass": schedule.component_mass.tolist(),
            "phase_rad": schedule.phase_rad.tolist(),
        },
        "ue": {
            "rows": int(x.size),
            "count_range": list(cfg.simulation.ue.count_range),
            "height_m": float(cfg.simulation.ue.height_m),
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the scenario. Entry point for ``task simulation:scenario``.

    Example:
        $ task simulation:scenario -- simulation.time.horizon_s=3600 seed=7
    """
    generate(cfg)


if __name__ == "__main__":
    main()
