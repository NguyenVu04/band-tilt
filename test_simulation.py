"""Plot a solved scenario: where everything is, and what the radio map says.

``python test_simulation.py`` writes to ``reports/figures/``:

``ue_transmitters.png``
    Every UE and every sector, rendered by sionna-rt itself as receivers and
    transmitters over the scene's building footprint — the whole population,
    never a sample. Receivers are coloured by the interval their UE was drawn
    in.
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
import numpy as np
import pandas as pd
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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

# sionna-rt's Camera renderer hard-codes a 10,000 m far clip (see
# make_render_sensor in sionna.rt.utils.render), so the camera cannot simply be
# placed arbitrarily high for a straight-down shot. This sits close to that
# ceiling, with the field of view then solved from the grid extent so the
# ground is fully framed. At this height-to-extent ratio the camera's
# perspective is close enough to orthographic that a UE's pixel position and
# its ground (x, y) agree to a small fraction of a cell, so positions are
# placed with the same world-metre extent used to frame the render, rather
# than by simulating the camera's true (very slightly non-linear) projection.
_CAMERA_HEIGHT_M = 9000.0
_CAMERA_CLEARANCE_M = 500.0

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


def _framed_extent(raster: grid.Raster) -> tuple[float, float, float, float]:
    """The ground extent the top-down camera actually frames, in world metres.

    ``_CAMERA_MARGIN`` widens the render past the raster's own footprint so
    the grid's edge cells aren't clipped by the camera's field of view
    (:func:`_camera`). Anything overlaid on that render — an ``imshow``
    extent, a scatter of points at their true coordinates — must be placed
    against this same padded box, not the raster's bare extent, or it
    silently disagrees with what the camera actually captured: the rendered
    geometry would be squeezed into a smaller box than it was framed at,
    shifting it inward relative to anything plotted at its true position.
    """
    extent_x = raster.n_cols * raster.cell_size_m
    extent_y = raster.n_rows * raster.cell_size_m
    centre_x = raster.origin_x + 0.5 * extent_x
    centre_y = raster.origin_y + 0.5 * extent_y
    half_x = 0.5 * _CAMERA_MARGIN * extent_x
    half_y = 0.5 * _CAMERA_MARGIN * extent_y
    return (centre_x - half_x, centre_x + half_x, centre_y - half_y, centre_y + half_y)


def _draw_boresights(axes: plt.Axes, sectors: tuple[transmitter.Sector, ...]) -> None:
    """Point a stub along every sector's boresight.

    Mast positions themselves are already in the backdrop — sionna-rt draws
    its own transmitter marker there. This only adds the azimuth a top-down
    device icon cannot show.
    """
    for sector in sectors:
        angle = math.radians(sector.azimuth_deg)
        axes.plot(
            [sector.x, sector.x + _BORESIGHT_M * math.cos(angle)],
            [sector.y, sector.y + _BORESIGHT_M * math.sin(angle)],
            color="red",
            linewidth=1.2,
            zorder=5,
        )


def _render_geometry(
    scene: Any, camera: Any, fov_deg: float, resolution: tuple[int, int]
) -> np.ndarray:
    """Render the scene from the top-down camera, as an RGBA array.

    ``show_devices`` draws every :class:`sionna.rt.Transmitter` and
    :class:`sionna.rt.Receiver` currently added to ``scene`` natively; the
    caller is responsible for having added one per sector and per UE first.
    """
    import mitsuba as mi

    bitmap = scene.render(
        camera=camera,
        fov=fov_deg,
        resolution=resolution,
        show_devices=True,
        return_bitmap=True,
    )
    image = bitmap.convert(component_format=mi.Struct.Type.UInt8, srgb_gamma=True)
    return np.array(image)


def plot_positions(
    cfg: DictConfig,
    scene: Any,
    camera: Any,
    fov_deg: float,
    resolution: tuple[int, int],
    raster: grid.Raster,
) -> Path:
    """Render every UE as a receiver and every sector as a transmitter.

    Every row of the UE table becomes a live :class:`sionna.rt.Receiver` and
    every sector a :class:`sionna.rt.Transmitter`, added to ``scene`` so
    sionna-rt's own renderer draws both over the building geometry — the
    whole population and every sector, never a sample. Receivers are removed
    again once the render is captured, so they do not linger into
    :func:`plot_rsrp`'s renders. Which band's tilt the transmitters are built
    at does not matter here: this is a top-down view, and tilt is pitch about
    the boresight, invisible from directly above.

    Returns the file written.
    """
    from sionna.rt import Receiver

    ues = pd.read_csv(Path(cfg.simulation.output.ue_file))
    sectors = transmitter.load(cfg)
    band_name = str(cfg.simulation.radio_map.bands[0].name)
    power_dbm = float(cfg.simulation.antenna.power_rs)

    radio.configure_arrays(scene, cfg)
    transmitter.build(scene, sectors, band_name, power_dbm)

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=ues["t_index"].min(), vmax=ues["t_index"].max())
    names = tuple(f"ue{index}" for index in range(len(ues)))
    for name, row in zip(names, ues.itertuples(index=False), strict=True):
        scene.add(
            Receiver(
                name=name,
                position=[row.x, row.y, row.z],
                color=tuple(float(channel) for channel in cmap(norm(row.t_index))[:3]),
            )
        )
    try:
        backdrop = _render_geometry(scene, camera, fov_deg, resolution)
    finally:
        for name in names:
            scene.remove(name)

    figure, axes = plt.subplots(figsize=(11, 9))
    axes.imshow(backdrop, extent=_framed_extent(raster), origin="upper")
    _draw_boresights(axes, sectors)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    figure.colorbar(mappable, ax=axes, label="interval index (receiver colour)", shrink=0.8)
    axes.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=(1.0, 0.0, 0.0),
                markeredgecolor="white",
                markersize=9,
                label="sectors",
            )
        ],
        loc="upper right",
        framealpha=0.9,
    )

    intervals = int(ues["t_index"].nunique())
    axes.set_title(
        f"UE positions and sites — {len(ues)} UEs over {intervals} intervals, "
        f"{len(sectors)} sectors"
    )
    return _finish(figure, axes, raster, "ue_transmitters.png")


def _camera(raster: grid.Raster, bounds: Any) -> tuple[Any, float]:
    """A near-orthographic top-down camera framing the grid, and its field of view.

    ``sionna.rt.Camera.look_at`` is degenerate for a viewpoint directly above
    its target — the up vector and the view direction are then parallel — and
    handles it internally with a small epsilon offset, so a true top-down view
    is a supported, if edge-case, camera pose rather than something to avoid.

    Returns the camera and the field of view [deg] that frames the grid at
    its height; ``scene.render`` does not read a camera's own field of view,
    so this must be passed to every render alongside it.
    """
    from sionna.rt import Camera

    x0, x1, y0, y1 = _framed_extent(raster)
    centre_x = 0.5 * (x0 + x1)
    centre_y = 0.5 * (y0 + y1)

    height = max(_CAMERA_HEIGHT_M, bounds.max_z + _CAMERA_CLEARANCE_M)
    fov_deg = math.degrees(2.0 * math.atan(0.5 * (x1 - x0) / height))
    camera = Camera(position=(centre_x, centre_y, height), look_at=(centre_x, centre_y, 0.0))
    return camera, fov_deg


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


def plot_rsrp(
    cfg: DictConfig,
    scene: Any,
    camera: Any,
    fov_deg: float,
    resolution: tuple[int, int],
    raster: grid.Raster,
) -> tuple[Path, ...]:
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
            fov=fov_deg,
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
    """Label the axes, save under ``reports/figures/`` and close. Returns the path.

    The legend itself is the caller's: it is the one that knows what it drew.
    """
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal")

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
    camera, fov_deg = _camera(raster, bounds)
    resolution = _resolution(raster)

    print(f"positions: {plot_positions(cfg, scene, camera, fov_deg, resolution, raster)}")
    for path in plot_rsrp(cfg, scene, camera, fov_deg, resolution, raster):
        print(f"rsrp:      {path}")


if __name__ == "__main__":
    main()
