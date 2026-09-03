"""Stage 1: build one perturbed scenario and draw its UE population.

``python -m src.simulation.scenario`` perturbs the delivered scene, rasters it,
and draws UEs over the result, writing the UE table and a manifest.

This stage is deliberately separate from the radio map. The map is a function
of tilt and must be re-solved for every tilt configuration, while the geometry
and the UE positions must *not* move when tilt does — otherwise the KPIs stop
being a function of tilt, which is the property the whole optimization rests on.
A scenario is drawn once and reused across every tilt.

Radio materials are not installed here. They have no effect on geometry, and
nothing in this stage propagates anything; the radio stage installs them per
band. They remain a pure function of the seed, so the scenario stays
regenerable from the manifest.
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

from src.simulation import density, grid, perturb, sample
from src.simulation import scene as scene_module
from src.simulation.density import DensitySpec
from src.simulation.grid import GridSpec
from src.simulation.perturb import PerturbSpec
from src.simulation.sample import UeSpec
from src.simulation.scene import SceneSpec

# Config sections that define what a scenario IS. Output paths are excluded on
# purpose: writing the same population somewhere else is not a new scenario.
_IDENTITY_KEYS = ("scene", "grid", "ue", "density", "perturbation", "materials", "seed")


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
    """Run the scenario stage. Returns ``(ue_file, manifest_file)``.

    Offsets, not independent draws, so the relationship between the streams is
    itself reproducible: NumPy runs an integer seed through a SeedSequence, so
    consecutive values give well-separated streams.
    """
    seed = int(cfg.simulation.seed)
    grid_spec = GridSpec.from_config(cfg)
    ue = UeSpec.from_config(cfg)

    scene, delivered = scene_module.load(SceneSpec.from_config(cfg))
    report = perturb.apply(scene, PerturbSpec.from_config(cfg), seed)

    # The grid comes from the DELIVERED extent, not the perturbed one. Jittering
    # a building a couple of metres would otherwise shift the grid origin, and
    # radio maps from two scenarios would no longer be cell-for-cell
    # comparable. The ground is never perturbed, so its footprint is the stable
    # frame. Only the ray launch height is taken from the perturbed scene, since
    # a heightened building can now stand above the delivered maximum.
    perturbed = scene_module.bounds_of(scene)
    bounds = dataclasses.replace(delivered, max_z=max(delivered.max_z, perturbed.max_z))

    raster = grid.build(scene.mi_scene, bounds, grid_spec, seed)
    field = density.field(raster, DensitySpec.from_config(cfg), seed + 1)
    x, y, component = sample.sample_positions(
        scene.mi_scene, bounds, raster, field, ue, grid_spec, seed + 2
    )

    ue_file = sample.write_csv(Path(cfg.simulation.output.ue_file), x, y, component, raster, ue)
    manifest_file = _write_manifest(cfg, report, raster, bounds, x, y)

    cell_col, cell_row = raster.cell_indices(x, y)
    open_cells = int(np.count_nonzero(raster.free_fraction > 0.0))
    print(f"scenario: {scenario_id(cfg)}")
    print(f"removed:  {len(report.removed)} buildings   jittered: {len(report.jittered)}")
    print(f"grid:     {raster.n_cols} x {raster.n_rows} cells, {open_cells} with open ground")
    print(f"hotspots: {len(field.hotspots)}")
    print(f"ue:       {x.size} at z={ue.height_m} m")
    print(
        f"densest decile holds {sample.densest_decile_share(cell_col, cell_row, raster):.1%} of UEs"
    )
    print(f"csv:      {ue_file}")
    print(f"manifest: {manifest_file}")
    return ue_file, manifest_file


def _write_manifest(
    cfg: DictConfig,
    report: perturb.PerturbReport,
    raster: grid.Raster,
    bounds: scene_module.SceneBounds,
    x: np.ndarray,
    y: np.ndarray,
) -> Path:
    """Record what this scenario is, so it can be regenerated and split on."""
    path = Path(cfg.simulation.output.manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": scenario_id(cfg),
        "seed": int(cfg.simulation.seed),
        "scene": OmegaConf.to_container(cfg.simulation.scene, resolve=True),
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
        "ue": {
            "count": int(x.size),
            "height_m": float(cfg.simulation.ue.height_m),
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the scenario. Entry point for ``task simulation:scenario``.

    Example:
        $ task simulation:scenario -- simulation.ue.count=1000 seed=7
    """
    generate(cfg)


if __name__ == "__main__":
    main()
