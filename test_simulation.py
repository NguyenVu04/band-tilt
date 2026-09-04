"""Plot a solved scenario: where everything is, and what the radio map says.

``python test_simulation.py`` writes to ``reports/figures/``:

``ue_transmitters.png``
    UE positions and masts over the scene's building footprint, coloured by the
    interval each UE was drawn in.
``rsrp_map_{band}.png``
    One file per configured band: best-server RSRP over that band, rendered
    with sionna-rt's own renderer (:meth:`sionna.rt.Scene.render`) from a
    top-down :class:`sionna.rt.Camera`, overlaid on the rendered scene geometry.

The scene is rebuilt exactly as :mod:`src.simulation.radio` rebuilds it —
loaded, then perturbed with the same seed — so the footprint is the one the map
is solved against, not the delivered city. The RSRP figures re-run
:func:`src.simulation.radio.solve_band` rather than read
``simulation.output.radio_map_file``: sionna-rt's renderer needs the live
:class:`sionna.rt.RadioMap` the solver returns, and that object cannot be
recovered from the numpy array the radio stage writes to disk. This repeats
the radio stage's ray tracing at its full configured fidelity, so it costs
the same as ``task simulation:radio``.

Requires the scenario stage to have run, and a CUDA GPU, since both the
footprint and the radio map come from casting rays at the scene.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import pandas as pd
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.simulation import grid, perturb, radio, seeds, transmitter  # noqa: E402
from src.simulation import scene as scene_module  # noqa: E402
from src.simulation.grid import GridSpec  # noqa: E402
from src.simulation.materials import MaterialSpec  # noqa: E402
from src.simulation.perturb import PerturbSpec  # noqa: E402
from src.simulation.radio import Band, SolverSpec  # noqa: E402
from src.simulation.scene import SceneSpec  # noqa: E402

# reports/figures is gitignored; reports/ itself is not, so generated plots
# do not land in a commit by accident.
_REPORT_DIR = Path("reports/figures")

# Long enough to read on the plot, short enough not to cross a cell.
_BORESIGHT_M = 90.0

# Horizontal field of view of the top-down render camera. Narrow enough that
# the perspective stays close to orthographic, wide enough to keep the camera
# (and the clearance it needs above the tallest building) at a sane height.
_CAMERA_FOV_DEG = 60.0

# Fraction of extra width the camera is backed off by, so the grid's own edge
# cells are not clipped by the render's field of view.
_CAMERA_MARGIN = 1.1


def _build_scenario(cfg: DictConfig) -> tuple[Any, Any, grid.Raster]:
    """Rebuild the perturbed scene, as the radio stage does.

    Returns the scene, bounds, and raster.
    """
    scene, delivered = scene_module.load(SceneSpec.from_config(cfg))
    perturb.apply(scene, PerturbSpec.from_config(cfg), seeds.stream(cfg, "scene"))

    bounds = dataclasses.replace(
        delivered, max_z=max(delivered.max_z, scene_module.bounds_of(scene).max_z)
    )
    raster = grid.build(
        scene.mi_scene, bounds, GridSpec.from_config(cfg), seeds.stream(cfg, "scene")
    )
    return scene, bounds, raster


def _extent(raster: grid.Raster) -> list[float]:
    """The grid's outer edges, for ``imshow``."""
    return [
        raster.origin_x,
        raster.origin_x + raster.n_cols * raster.cell_size_m,
        raster.origin_y,
        raster.origin_y + raster.n_rows * raster.cell_size_m,
    ]


def _draw_sites(axes: plt.Axes, sectors: tuple[transmitter.Sector, ...]) -> None:
    """Mark each mast and point a stub along every sector's boresight."""
    for sector in sectors:
        angle = math.radians(sector.azimuth_deg)
        axes.plot(
            [sector.x, sector.x + _BORESIGHT_M * math.cos(angle)],
            [sector.y, sector.y + _BORESIGHT_M * math.sin(angle)],
            color="red",
            linewidth=1.2,
            zorder=5,
        )
    axes.scatter(
        [sector.x for sector in sectors],
        [sector.y for sector in sectors],
        marker="^",
        s=70,
        color="red",
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
        label="transmitters",
    )


def plot_positions(cfg: DictConfig, raster: grid.Raster) -> Path:
    """Draw UEs and masts over the building footprint. Returns the file written."""
    ues = pd.read_csv(Path(cfg.simulation.output.ue_file))
    sectors = transmitter.load(cfg)

    figure, axes = plt.subplots(figsize=(11, 9))
    axes.imshow(
        1.0 - raster.free_fraction,
        extent=_extent(raster),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.5,
        interpolation="nearest",
    )

    scatter = axes.scatter(
        ues["x"],
        ues["y"],
        c=ues["t_index"],
        cmap="viridis",
        s=3,
        alpha=0.55,
        linewidths=0,
        label="UEs",
    )
    figure.colorbar(scatter, ax=axes, label="interval index", shrink=0.8)
    _draw_sites(axes, sectors)

    intervals = int(ues["t_index"].nunique())
    axes.set_title(
        f"UE positions and sites — {len(ues)} UEs over {intervals} intervals, "
        f"{len(sectors)} sectors"
    )
    return _finish(figure, axes, raster, "ue_transmitters.png")


def _camera(raster: grid.Raster, bounds: Any) -> Any:
    """A top-down camera framing the grid, high enough to clear the tallest building.

    ``sionna.rt.Camera.look_at`` is degenerate for a viewpoint directly above
    its target — the up vector and the view direction are then parallel — and
    handles it internally with a small epsilon offset, so a true top-down view
    is a supported, if edge-case, camera pose rather than something to avoid.
    """
    from sionna.rt import Camera

    extent_x = raster.n_cols * raster.cell_size_m
    extent_y = raster.n_rows * raster.cell_size_m
    centre_x = raster.origin_x + 0.5 * extent_x
    centre_y = raster.origin_y + 0.5 * extent_y

    half_fov = math.radians(_CAMERA_FOV_DEG / 2.0)
    height = max(
        _CAMERA_MARGIN * 0.5 * extent_x / math.tan(half_fov),
        bounds.max_z + 50.0,
    )
    return Camera(position=(centre_x, centre_y, height), look_at=(centre_x, centre_y, 0.0))


def _resolution(raster: grid.Raster, long_edge: int = 1000) -> tuple[int, int]:
    """Render resolution matching the grid's aspect ratio.

    Keeps the camera's fixed horizontal field of view covering the same
    ground extent on both axes.
    """
    extent_x = raster.n_cols * raster.cell_size_m
    extent_y = raster.n_rows * raster.cell_size_m
    if extent_x >= extent_y:
        return (long_edge, max(1, round(long_edge * extent_y / extent_x)))
    return (max(1, round(long_edge * extent_x / extent_y)), long_edge)


def plot_rsrp(cfg: DictConfig, scene: Any, bounds: Any, raster: grid.Raster) -> tuple[Path, ...]:
    """Render best-server RSRP per band with sionna-rt's own renderer.

    Re-solves each band against ``scene`` rather than reading
    ``simulation.output.radio_map_file``: :meth:`sionna.rt.Scene.render` needs
    the live :class:`sionna.rt.RadioMap` the solver returns, which the numpy
    archive the radio stage writes does not carry. Returns the files written,
    one per band.
    """
    sectors = transmitter.load(cfg)
    bands = tuple(Band.from_config(entry) for entry in cfg.simulation.radio_map.bands)
    solver_spec = SolverSpec.from_config(cfg)
    material_spec = MaterialSpec.from_config(cfg)
    power_dbm = float(cfg.simulation.antenna.power_rs)
    height_m = float(cfg.simulation.ue.height_m)
    grid_meta = {
        "origin_x": raster.origin_x,
        "origin_y": raster.origin_y,
        "cell_size_m": raster.cell_size_m,
        "n_cols": raster.n_cols,
        "n_rows": raster.n_rows,
    }

    radio.configure_arrays(scene, cfg)
    camera = _camera(raster, bounds)
    resolution = _resolution(raster)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for band in bands:
        _, elapsed, _, radio_map = radio.solve_band(
            scene,
            sectors,
            band,
            solver_spec,
            material_spec,
            seeds.stream(cfg, "materials"),
            seeds.stream(cfg, "solver"),
            grid_meta,
            height_m,
            power_dbm,
        )
        # rm_metric="rss" with rm_db_scale=True renders 10*log10(rss_mW), i.e.
        # dBm, matching the RSRP the radio stage writes to its archive.
        figure = scene.render(
            camera=camera,
            radio_map=radio_map,
            rm_metric="rss",
            rm_db_scale=True,
            rm_show_color_bar=True,
            resolution=resolution,
        )
        figure.suptitle(f"{band.name} best-server RSRP — {elapsed:.1f}s solve")
        path = _REPORT_DIR / f"rsrp_map_{band.name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def _finish(figure: plt.Figure, axes: plt.Axes, raster: grid.Raster, name: str) -> Path:
    """Label the axes, save under ``reports/figures/`` and close. Returns the path."""
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal")
    axes.legend(loc="upper right", framealpha=0.9)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / name
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Write the UE-position figure and one rendered RSRP figure per band.

    Example:
        $ python test_simulation.py
    """
    ue_file = Path(cfg.simulation.output.ue_file)
    if not ue_file.is_file():
        raise FileNotFoundError(f"No {ue_file}. Run `task simulation:scenario` first.")

    scene, bounds, raster = _build_scenario(cfg)
    print(f"positions: {plot_positions(cfg, raster)}")
    for path in plot_rsrp(cfg, scene, bounds, raster):
        print(f"rsrp:      {path}")


if __name__ == "__main__":
    main()
