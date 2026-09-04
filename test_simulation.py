"""Plot a solved scenario: where everything is, and what the radio map says.

``python test_simulation.py`` writes two figures to ``reports/figures/``:

``ue_transmitters.png``
    UE positions and masts over the scene's building footprint, coloured by the
    interval each UE was drawn in.
``rsrp_map.png``
    Best-server RSRP over the same grid, from the ray-traced radio map.

Both are drawn in scene metres on the same axes, so a hole in the second figure
can be read directly against the geometry and the sites in the first.

The backdrop is the building raster rather than a 3D render, for that reason:
a perspective view of the scene would not line up with a top-down radio map.
The scene is rebuilt exactly as :mod:`src.simulation.radio` rebuilds it —
loaded, then perturbed with the same seed — so the footprint is the one the map
was solved against, not the delivered city.

Requires the scenario and radio stages to have run, and a CUDA GPU, since the
footprint comes from casting rays at the scene.
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
import matplotlib.pyplot as plt  # noqa: E402

from src.simulation import grid, perturb, seeds, transmitter  # noqa: E402
from src.simulation import scene as scene_module  # noqa: E402
from src.simulation.grid import GridSpec  # noqa: E402
from src.simulation.perturb import PerturbSpec  # noqa: E402
from src.simulation.scene import SceneSpec  # noqa: E402

# reports/figures is gitignored; reports/ itself is not, so generated plots
# do not land in a commit by accident.
_REPORT_DIR = Path("reports/figures")

# Long enough to read on the plot, short enough not to cross a cell.
_BORESIGHT_M = 90.0


def _load_scene(cfg: DictConfig) -> tuple[Any, Any]:
    """Rebuild the perturbed scene, as the radio stage does. Returns the raster and bounds."""
    scene, delivered = scene_module.load(SceneSpec.from_config(cfg))
    perturb.apply(scene, PerturbSpec.from_config(cfg), seeds.stream(cfg, "scene"))

    bounds = dataclasses.replace(
        delivered, max_z=max(delivered.max_z, scene_module.bounds_of(scene).max_z)
    )
    raster = grid.build(
        scene.mi_scene, bounds, GridSpec.from_config(cfg), seeds.stream(cfg, "scene")
    )
    return raster, bounds


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


def plot_rsrp(cfg: DictConfig, raster: grid.Raster) -> Path:
    """Draw best-server RSRP over the grid. Returns the file written."""
    map_path = Path(cfg.simulation.output.radio_map_file)
    with np.load(map_path, allow_pickle=False) as data:
        rsrp = data["rsrp_dbm"]
        bands = [str(label) for label in data["band_label"]]

    best = _best_server(rsrp)
    sectors = transmitter.load(cfg)

    figure, axes = plt.subplots(figsize=(11, 9))
    colours = plt.get_cmap("turbo").copy()
    colours.set_bad("black")  # a cell no ray reached, which is not a weak cell
    image = axes.imshow(
        np.ma.masked_invalid(best),
        extent=_extent(raster),
        origin="lower",
        cmap=colours,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axes, label="best-server RSRP (dBm)", shrink=0.8)
    _draw_sites(axes, sectors)

    reached = float(np.isfinite(best).mean())
    axes.set_title(
        f"Best-server RSRP over {len(bands)} bands ({', '.join(bands)}) — "
        f"{reached:.1%} of cells reached, black is no path"
    )
    return _finish(figure, axes, raster, "rsrp_map.png")


def _best_server(rsrp: np.ndarray) -> np.ndarray:
    """Strongest RSRP over every cell-band layer, NaN where no ray arrived.

    ``rsrp`` is ``[band, tx, row, col]``. Reduced over the reached cells only:
    a cell no ray found is all-NaN, and ``nanmax`` over one warns rather than
    simply meaning "no coverage".
    """
    layers = rsrp.reshape(-1, *rsrp.shape[2:])
    served = np.isfinite(layers).any(axis=0)
    best = np.full(rsrp.shape[2:], np.nan, dtype=np.float64)
    best[served] = np.nanmax(layers[:, served], axis=0)
    return best


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
    """Write both figures.

    Example:
        $ python test_simulation.py
    """
    for path, stage in (
        (Path(cfg.simulation.output.ue_file), "scenario"),
        (Path(cfg.simulation.output.radio_map_file), "radio"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{stage}` first.")

    raster, _ = _load_scene(cfg)
    print(f"positions: {plot_positions(cfg, raster)}")
    print(f"rsrp:      {plot_rsrp(cfg, raster)}")


if __name__ == "__main__":
    main()
