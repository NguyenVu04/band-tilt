"""The figures notebook 04 and ``task evaluate`` present.

Every map is drawn ``origin="lower"`` on the grid's metric extent, because row 0
of the raster is the lowest y; matplotlib's default would mirror it. Every name
a reader sees goes through :func:`src.utils.plotting.label`.

Each function returns a :class:`~matplotlib.figure.Figure` and shows nothing, so
the caller decides whether to display it, save it, or both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.evaluation import compare, maps
from src.kpi.capacity import max_rsrp
from src.utils.plotting import label

# One colour per configuration, identical in every figure so a reader learns
# them once.
COLOURS = {
    "incumbent": "tab:red",
    "morbo": "tab:blue",
    "random": "tab:green",
}

# Hexagon width of the UE-report maps. Wide enough that a background hexagon
# pools several reports instead of showing one report as one pixel, narrow
# enough to separate the demand hotspots.
HEX_SIZE_M = 150.0
# Fewest reports a throughput hexagon needs to be drawn: below it the median is
# one or two reports, not a reading of the area.
MIN_REPORTS = 5


def _overlay(axis: plt.Axes, sectors: pd.DataFrame | None, hotspots: pd.DataFrame | None) -> None:
    """Mark the nodes (crimson triangles) and demand hotspot centres (black crosses)."""
    if sectors is not None and len(sectors):
        axis.scatter(sectors["x"], sectors["y"], marker="^", s=45, color="crimson", zorder=3)
    if hotspots is not None and len(hotspots):
        axis.scatter(hotspots["x"], hotspots["y"], marker="x", s=70, color="black", zorder=4)


def class_map(
    axis: plt.Axes,
    values: np.ndarray,
    names: Sequence[str],
    colours: Sequence[str],
    title: str,
    extent: list[float],
    sectors: pd.DataFrame | None = None,
) -> None:
    """Draw a categorical raster with a legend; ``values`` index ``names``, ``-1`` for none."""
    axis.imshow(
        np.where(values >= 0, values, np.nan),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(list(colours)),
        vmin=-0.5,
        vmax=len(names) - 0.5,
        interpolation="nearest",
    )
    _overlay(axis, sectors, None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in colours]
    axis.legend(handles, names, loc="lower left", fontsize=8, frameon=True)
    axis.set(title=title, xlabel="x [m]", ylabel="y [m]")
    axis.grid(False)


def _map_axes(axis: plt.Axes, extent: list[float], title: str) -> None:
    """Label one map panel."""
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])


def coverage_maps(
    before: np.ndarray,
    after: np.ndarray,
    radio: dict[str, Any],
    cfg: DictConfig,
    *,
    sectors: pd.DataFrame | None = None,
    name: str = "optimized",
    band: str | None = None,
) -> Figure:
    """Best-server RSRP before and after, and which tiles crossed the hole threshold.

    The third panel is the threshold crossing rather than a signed difference:
    a tile gaining 3 dB while staying a hole has changed nothing a KPI can see.

    Args:
        before: Best-server RSRP of the incumbent, ``[n_rows, n_cols]``.
        after: Best-server RSRP of the other configuration.
        radio: Any radio map of the scenario, for the grid extent.
        cfg: Composed config; reads ``kpi.hole_dbm``.
        sectors: Optional sector table with ``x`` and ``y``.
        name: Key of the second configuration, for its title.
        band: The band both rasters were read on, for the title.
    """
    extent = maps.extent_of(radio)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    for axis, values, key in ((axes[0], before, "incumbent"), (axes[1], after, name)):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=maps.RSRP_LIMITS[0],
            vmax=maps.RSRP_LIMITS[1],
        )
        figure.colorbar(image, ax=axis, label="Best-server RSRP [dBm]")
        _overlay(axis, sectors, None)
        _map_axes(axis, extent, label(key))

    change = maps.change_mask(before, after, cfg)
    image = axes[2].imshow(
        change, origin="lower", extent=extent, aspect="equal", cmap="coolwarm_r", vmin=-1, vmax=1
    )
    bar = figure.colorbar(image, ax=axes[2], ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["Hole opened", "Unchanged", "Hole closed"])
    _overlay(axes[2], sectors, None)
    _map_axes(
        axes[2],
        extent,
        f"Coverage-hole changes\n{int((change > 0).sum())} closed, "
        f"{int((change < 0).sum())} opened",
    )
    on = f", {label(band).lower() if band == 'all' else label(band)}" if band else ""
    figure.suptitle(f"Best-server RSRP before and after optimization ({label(name)}){on}")
    return figure


def _hexbin(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    extent: list[float],
    values: np.ndarray | None = None,
    min_reports: int = 1,
    cmap: Any = None,
) -> Any:
    """Draw UE reports on :data:`HEX_SIZE_M` hexagons over ``extent``.

    Counts the reports per hexagon when ``values`` is None, else takes the
    median of ``values``. The grid depends only on ``extent``, so two calls on
    one scenario share their hexagons. Returns the hexagon collection.
    """
    return axis.hexbin(
        x,
        y,
        C=values,
        reduce_C_function=np.median,
        gridsize=max(1, round((extent[1] - extent[0]) / HEX_SIZE_M)),
        extent=extent,
        mincnt=min_reports,
        cmap=cmap,
        linewidths=0.0,
    )


def demand_signal_maps(
    rsrp: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radio: dict[str, Any],
    *,
    sectors: pd.DataFrame | None = None,
    hotspots: pd.DataFrame | None = None,
    name: str | None = None,
) -> Figure:
    """Where the demand is and where the signal is.

    Args:
        rsrp: Radio map in dBm, ``[n_band, n_tx, n_rows, n_cols]``.
        x: Every UE report's x [m].
        y: Every UE report's y [m].
        radio: Any radio map of the scenario, for the grid extent.
        sectors: Optional sector table with ``x`` and ``y``.
        hotspots: Optional hotspot table with ``x`` and ``y``.
        name: Key of the configuration ``rsrp`` belongs to, for the title.
    """
    extent = maps.extent_of(radio)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    # Hexagons with no report stay blank: "nobody here" is not "one person here".
    image = _hexbin(axes[0], np.asarray(x), np.asarray(y), extent, cmap="YlOrRd")
    figure.colorbar(image, ax=axes[0], label="UE reports per hexagon")
    _overlay(axes[0], sectors, hotspots)
    _map_axes(axes[0], extent, f"Traffic demand [UE reports per {HEX_SIZE_M:g} m hexagon]")

    image = axes[1].imshow(
        max_rsrp(rsrp),
        origin="lower",
        extent=extent,
        vmin=maps.RSRP_LIMITS[0],
        vmax=maps.RSRP_LIMITS[1],
    )
    figure.colorbar(image, ax=axes[1], label="Best-server RSRP [dBm]")
    _overlay(axes[1], sectors, hotspots)
    _map_axes(axes[1], extent, "Best-server RSRP")
    of = f" ({label(name)})" if name else ""
    figure.suptitle(f"Traffic demand and best-server RSRP{of}")
    return figure


def throughput_hexbins(
    panels: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    radio: dict[str, Any],
    *,
    colorbar_label: str,
    title: str,
    symmetric: bool = False,
    sectors: pd.DataFrame | None = None,
) -> Figure:
    """Median of a per-report value on :data:`HEX_SIZE_M` hexagons, one panel each, one scale.

    Hexagons with fewer than :data:`MIN_REPORTS` reports are blank.

    Args:
        panels: Title to one value per UE report, aligned with ``x`` and ``y``.
        x: Every UE report's x [m].
        y: Every UE report's y [m].
        radio: Any radio map of the scenario, for the grid extent.
        colorbar_label: Colour bar label, with units.
        title: The figure title.
        symmetric: Centre the scale on zero with a diverging colormap, for
            difference maps.
        sectors: Optional sector table with ``x`` and ``y``.
    """
    extent = maps.extent_of(radio)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.8 * len(panels) + 1.0, 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = "RdBu_r" if symmetric else "viridis"
    images = []
    for axis, (panel, values) in zip(axes[0], panels.items(), strict=True):
        images.append(
            _hexbin(
                axis, np.asarray(x), np.asarray(y), extent, np.asarray(values), MIN_REPORTS, cmap
            )
        )
        _overlay(axis, sectors, None)
        _map_axes(axis, extent, panel)
    present = np.concatenate([np.asarray(image.get_array()) for image in images])
    if present.size == 0:
        present = np.zeros(1)
    if symmetric:
        # 99th percentile, as in map_row: hexagons nearest a mast swing hardest.
        limit = float(np.quantile(np.abs(present), 0.99)) or 1.0
        low, high = -limit, limit
        colorbar_label = f"{colorbar_label}, clipped at ±{limit:.1f}"
    else:
        low, high = float(present.min()), float(present.max())
    for image in images:
        image.set_clim(low, high)
    figure.colorbar(images[-1], ax=axes[0].tolist(), label=colorbar_label)
    figure.suptitle(title)
    return figure


def map_row(
    panels: dict[str, np.ndarray],
    radio: dict[str, Any],
    *,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    cmap: Any = None,
    sectors: pd.DataFrame | None = None,
    hotspots: pd.DataFrame | None = None,
    title: str | None = None,
) -> Figure:
    """One raster per panel, side by side, on one colour scale.

    Args:
        panels: Title to ``[n_rows, n_cols]`` raster, drawn in order.
        radio: Any radio map of the scenario, for the grid extent.
        colorbar_label: Colour bar label, with units.
        vmin: Lower limit; the data minimum when None.
        vmax: Upper limit; the data maximum when None.
        symmetric: Centre the scale on zero with a diverging colormap, for
            difference maps.
        cmap: Colormap; viridis, or ``RdBu_r`` when ``symmetric``.
        sectors: Optional sector table with ``x`` and ``y``.
        hotspots: Optional hotspot table with ``x`` and ``y``.
        title: The figure title; none when None.
    """
    extent = maps.extent_of(radio)
    values = [np.where(np.isfinite(panel), panel, np.nan) for panel in panels.values()]
    present = np.concatenate([value[np.isfinite(value)] for value in values])
    if present.size == 0:
        present = np.zeros(1)
    if symmetric:
        # 99th percentile, not the maximum: tiles nearest a mast swing hardest
        # under tilt, and letting them set the scale flattens everything else.
        limit = float(np.quantile(np.abs(present), 0.99)) or 1.0
        vmin, vmax, cmap = -limit, limit, cmap or "RdBu_r"
        colorbar_label = f"{colorbar_label}, clipped at ±{limit:.1f}"
    vmin = float(present.min()) if vmin is None else vmin
    vmax = float(present.max()) if vmax is None else vmax

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.4 * len(panels) + 1.0, 4.4),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for axis, title, value in zip(axes[0], panels, values, strict=True):
        image = axis.imshow(
            value, origin="lower", extent=extent, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap
        )
        _overlay(axis, sectors, hotspots)
        _map_axes(axis, extent, title)
    figure.colorbar(image, ax=axes[0].tolist(), label=colorbar_label)
    if title:
        figure.suptitle(title)
    return figure


def sector_band_heatmaps(tables: dict[str, pd.DataFrame], value: str) -> Figure:
    """One value per sector-band as a sector-by-band grid, one panel per configuration.

    Args:
        tables: Configuration key to :func:`src.evaluation.compare.sector_band_load` output.
        value: The column to draw; the colour scale is shared across panels.
    """
    first = next(iter(tables.values()))
    sector_order = list(dict.fromkeys(first["sector"]))
    band_order = list(dict.fromkeys(first["band"]))
    grids = {
        key: table.pivot(index="sector", columns="band", values=value)
        .reindex(index=sector_order, columns=band_order)
        .to_numpy(float)
        for key, table in tables.items()
    }
    top = max((np.nanmax(g) for g in grids.values() if np.isfinite(g).any()), default=1.0)
    figure, axes = plt.subplots(
        1,
        len(tables),
        figsize=(1.3 * len(band_order) * len(tables) + 2.5, 0.3 * len(sector_order) + 1.8),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for index, (axis, (key, grid)) in enumerate(zip(axes[0], grids.items(), strict=True)):
        image = axis.imshow(grid, vmin=0.0, vmax=top, cmap="YlGnBu", aspect="auto")
        axis.grid(False)
        for (row, col), sector_value in np.ndenumerate(grid):
            axis.text(col, row, f"{sector_value:.3g}", ha="center", va="center", fontsize=7)
        axis.set_xticks(range(len(band_order)), [label(band) for band in band_order])
        axis.set_yticks(range(len(sector_order)), sector_order if index == 0 else [])
        axis.set_title(label(key))
    figure.colorbar(image, ax=axes[0].tolist(), label=label(value))
    figure.suptitle(f"{label(value)} per sector and frequency band")
    return figure


def band_share_bars(summaries: dict[str, dict[str, float]], band_labels: Sequence[str]) -> Figure:
    """Share of UE reports per serving band, per configuration; the rest are unserved.

    Args:
        summaries: Configuration key to :func:`src.evaluation.compare.service_summary` output.
        band_labels: Band keys, as in the summaries' ``share_<band>`` entries.
    """
    keys = list(summaries)
    figure, axis = plt.subplots(figsize=(1.8 * len(keys) + 3.0, 4.5), constrained_layout=True)
    bottom = np.zeros(len(keys))
    for band in band_labels:
        heights = np.array([summaries[config][f"share_{band}"] for config in keys])
        axis.bar([label(config) for config in keys], heights, bottom=bottom, label=label(band))
        for x, (height, base) in enumerate(zip(heights, bottom, strict=True)):
            if height >= 0.04:
                axis.text(
                    x, base + height / 2, f"{height:.0%}", ha="center", va="center", fontsize=8
                )
        bottom += heights
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of UE reports")
    axis.set_title("Share of UE reports by serving band")
    axis.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    return figure


def convergence_plot(frame: pd.DataFrame) -> Figure:
    """Hypervolume of the search objectives over every evaluation so far.

    Args:
        frame: :func:`src.evaluation.compare.convergence` output.
    """
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for method, group in frame.groupby("method", sort=False):
        axis.plot(
            group["iteration"],
            group["value"],
            lw=1.8,
            color=COLOURS.get(str(method)),
            label=label(method),
        )
    axis.set_xlabel("Number of evaluations")
    axis.set_ylabel("Hypervolume (higher is better)")
    axis.set_title("Hypervolume of the search objectives versus number of evaluations")
    axis.legend()
    return figure


def tradeoff_scatter(
    frame: pd.DataFrame, x: str, y: str, picks: dict[str, int] | None = None
) -> Figure:
    """Every evaluated configuration on two KPIs, each method's front and its pick.

    Points on a method's three-KPI front (``on_front``) are outlined; the
    dashed line joins the configurations no other beats on this pair alone.

    Args:
        frame: :func:`src.evaluation.compare.candidates` output.
        x: Measure on the horizontal axis.
        y: Measure on the vertical axis.
        picks: Method to the ``frame`` row of its largest hypervolume
            contribution, drawn as a star.
    """
    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for method, group in frame.groupby("method", sort=False):
        colour = COLOURS.get(str(method))
        searched = group[group["phase"] != "incumbent"]
        axis.scatter(searched[x], searched[y], s=14, alpha=0.4, color=colour, label=label(method))
        front = searched[searched["on_front"]]
        axis.scatter(front[x], front[y], s=30, facecolor="none", edgecolor=colour, lw=1.2)
    for method, row in (picks or {}).items():
        axis.scatter(
            frame.loc[row, x],
            frame.loc[row, y],
            marker="*",
            s=260,
            color=COLOURS.get(method),
            edgecolor="black",
            zorder=4,
            label=f"{label(method)}: largest hypervolume contribution",
        )
    incumbent = frame[frame["phase"] == "incumbent"].iloc[0]
    axis.scatter(
        incumbent[x],
        incumbent[y],
        marker="X",
        s=140,
        color=COLOURS["incumbent"],
        edgecolor="black",
        zorder=5,
        label=label("incumbent"),
    )
    front = frame[compare.pareto_front(frame, [x, y])].sort_values(x)
    axis.plot(front[x], front[y], color="0.2", ls="--", lw=1.0, label="Pareto front of this pair")
    axis.scatter([], [], s=30, facecolor="none", edgecolor="0.3", label="On the three-KPI front")
    axis.set_xlabel(f"{label(x)} ({compare.direction(x)})")
    axis.set_ylabel(f"{label(y)} ({compare.direction(y)})")
    axis.set_title(f"{label(x)} versus {label(y).lower()} of all evaluated configurations")
    axis.legend(fontsize=8)
    return figure


def tilt_delta_heatmap(table: pd.DataFrame, name: str) -> Figure:
    """Tilt change per sector and band on one diverging scale.

    Args:
        table: :func:`src.evaluation.compare.tilt_table` output.
        name: Key of the configuration, for the title.
    """
    table = table.astype({"sector": str, "band": str})
    sectors = list(dict.fromkeys(table["sector"]))
    bands = list(dict.fromkeys(table["band"]))
    grid = table.pivot(index="sector", columns="band", values="delta_tilt_deg")
    grid = grid.reindex(index=sectors, columns=bands).to_numpy()
    limit = float(np.nanmax(np.abs(grid))) or 1.0

    figure, axis = plt.subplots(
        figsize=(1.6 * len(bands) + 2.5, 0.35 * len(sectors) + 1.5), constrained_layout=True
    )
    image = axis.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.grid(False)
    for (row, col), value in np.ndenumerate(grid):
        axis.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(bands)), [label(band) for band in bands])
    axis.set_yticks(range(len(sectors)), sectors)
    figure.colorbar(image, ax=axis, label="Tilt change [°] (negative: uptilt)")
    axis.set_title(f"Tilt change per sector and band ({label(name)} versus current configuration)")
    return figure


def tilt_change_bars(table: pd.DataFrame, name: str) -> Figure:
    """Tilt change per sector-band as grouped bars: one group per sector, one bar per band.

    Args:
        table: :func:`src.evaluation.compare.tilt_table` output.
        name: Key of the configuration, for the title.
    """
    table = table.astype({"sector": str, "band": str})
    sectors = list(dict.fromkeys(table["sector"]))
    bands = list(dict.fromkeys(table["band"]))
    grid = table.pivot(index="sector", columns="band", values="delta_tilt_deg")
    grid = grid.reindex(index=sectors, columns=bands)
    width = 0.8 / len(bands)
    positions = np.arange(len(sectors))

    figure, axis = plt.subplots(figsize=(0.6 * len(sectors) + 3.0, 4.6), constrained_layout=True)
    for offset, band in enumerate(bands):
        axis.bar(
            positions + (offset - (len(bands) - 1) / 2) * width,
            grid[band].to_numpy(),
            width=width,
            label=label(band),
        )
    axis.axhline(0.0, color="0.2", lw=0.8)
    axis.set_xticks(positions, sectors, rotation=90)
    axis.set_xlabel(label("sector"))
    axis.set_ylabel("Tilt change [°] (negative: uptilt)")
    axis.set_title(f"Tilt change per sector-band ({label(name)} versus current configuration)")
    axis.legend(fontsize=8)
    return figure


def coverage_class_maps(
    rasters: dict[str, np.ndarray],
    radio: dict[str, Any],
    cfg: DictConfig,
    *,
    sectors: pd.DataFrame | None = None,
) -> Figure:
    """Hole, weak and good coverage per configuration, on identical classes.

    Args:
        rasters: Configuration key to its radio map's ``rsrp_dbm``.
        radio: Any radio map of the scenario, for the grid extent.
        cfg: Composed config; reads ``kpi.hole_dbm`` and ``kpi.weak_dbm``.
        sectors: Optional sector table with ``x`` and ``y``.
    """
    extent = maps.extent_of(radio)
    colours = ListedColormap(["black", "tab:orange", "tab:green"])
    figure, axes = plt.subplots(
        1,
        len(rasters),
        figsize=(4.8 * len(rasters) + 1.0, 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for axis, (key, rsrp) in zip(axes[0], rasters.items(), strict=True):
        classes = maps.coverage_class(rsrp, cfg)
        image = axis.imshow(
            classes,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=colours,
            vmin=-0.5,
            vmax=2.5,
        )
        _overlay(axis, sectors, None)
        _map_axes(
            axis,
            extent,
            f"{label(key)}: hole {(classes == maps.HOLE).mean():.1%}, "
            f"weak {(classes == maps.WEAK).mean():.1%}",
        )
    bar = figure.colorbar(image, ax=axes[0].tolist(), ticks=range(len(maps.COVERAGE_CLASSES)))
    bar.ax.set_yticklabels([name.capitalize() for name in maps.COVERAGE_CLASSES])
    figure.suptitle("Spatial distribution of coverage classes (hole, weak, good)")
    return figure


def band_kpi_panels(table: pd.DataFrame, kpis: Sequence[str]) -> Figure:
    """One panel per KPI, grouped bars over bands, one bar per configuration.

    Args:
        table: :func:`src.evaluation.compare.band_table` output.
        kpis: Which measures to draw, in panel order.
    """
    keys = list(dict.fromkeys(table["configuration"]))
    width = 0.8 / max(len(keys), 1)

    columns = min(len(kpis), 3)
    rows = -(-len(kpis) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 3.4 * rows), constrained_layout=True, squeeze=False
    )
    flat = axes.ravel()
    for axis in flat[len(kpis) :]:
        axis.set_visible(False)
    for axis, name in zip(flat, kpis, strict=False):
        bands = list(dict.fromkeys(table["band"]))
        positions = np.arange(len(bands))
        for offset, key in enumerate(keys):
            mine = table[table["configuration"] == key].set_index("band")[name]
            axis.bar(
                positions + (offset - (len(keys) - 1) / 2) * width,
                mine.reindex(bands).to_numpy(),
                width=width,
                color=COLOURS.get(key),
                label=label(key),
            )
        axis.set_xticks(positions, [label(band) for band in bands], fontsize=8)
        axis.set_title(label(name), fontsize=9)
    # Outside the panels: inside one, the key covers that panel's bars.
    figure.legend(
        *flat[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(keys),
        fontsize=9,
    )
    figure.suptitle("KPIs per band: current and selected configurations")
    return figure


def band_tradeoff(frame: pd.DataFrame, band_labels: Sequence[str]) -> Figure:
    """Coverage rate against separation rate on each band alone, every candidate, with its front.

    The dashed line joins the configurations no other beats on that band's pair.

    Args:
        frame: :func:`src.evaluation.compare.candidates` output, with the
            ``<kpi>_<band>`` columns the search recorded.
        band_labels: The bands, one panel each.
    """
    figure, axes = plt.subplots(
        1,
        len(band_labels),
        figsize=(4.6 * len(band_labels), 4.4),
        constrained_layout=True,
        squeeze=False,
    )
    incumbent = frame[frame["phase"] == "incumbent"].iloc[0]
    for axis, band in zip(axes[0], band_labels, strict=True):
        x, y = f"coverage_rate_{band}", f"separation_rate_{band}"
        for method, group in frame.groupby("method", sort=False):
            searched = group[group["phase"] != "incumbent"]
            axis.scatter(
                searched[x],
                searched[y],
                s=12,
                alpha=0.4,
                color=COLOURS.get(str(method)),
                label=label(method),
            )
        axis.scatter(
            incumbent[x],
            incumbent[y],
            marker="X",
            s=120,
            color=COLOURS["incumbent"],
            edgecolor="black",
            zorder=5,
            label=label("incumbent"),
        )
        front = frame[compare.pareto_front(frame, [x, y])].sort_values(x)
        axis.plot(front[x], front[y], color="0.2", ls="--", lw=1.0, label="Pareto front")
        axis.set_xlabel(label("coverage_rate"))
        axis.set_ylabel(label("separation_rate"))
        axis.set_title(label(band))
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Coverage rate versus separation rate per band, with Pareto fronts")
    return figure


def throughput_by_time_of_day(served: dict[str, pd.DataFrame], interval_s: float) -> Figure:
    """Quantiles of every UE report's throughput by time of day, with the UEs per interval.

    Every report is placed on its time-of-day slot, the days pooled, and each
    slot's quantiles are taken over those reports directly: the line is the
    median, the band the interquartile range, the dotted line the 5th percentile.

    Args:
        served: Configuration key to its :func:`src.kpi.capacity.serve_intervals`
            output; every frame holds the same UE reports.
        interval_s: Interval length, from the scenario manifest.
    """
    slots_per_day = round(86400.0 / interval_s)
    figure, axis = plt.subplots(figsize=(12.0, 5.0), constrained_layout=True)
    handles, names = [], []
    hour = pd.Series(dtype=float)
    for key, frame in served.items():
        hour = (frame["t_index"] % slots_per_day) * interval_s / 3600.0
        quantiles = (
            frame["estimated_throughput_mbps"]
            .groupby(hour)
            .quantile([0.05, 0.25, 0.5, 0.75])
            .unstack()
        )
        colour = COLOURS.get(str(key))
        (line,) = axis.plot(quantiles.index, quantiles[0.5], lw=1.6, color=colour)
        axis.fill_between(
            quantiles.index, quantiles[0.25], quantiles[0.75], color=colour, alpha=0.15, lw=0
        )
        axis.plot(quantiles.index, quantiles[0.05], lw=0.9, ls=":", color=colour)
        handles.append(line)
        names.append(label(key))
    axis.set_xlabel("Time of day [h]")
    axis.set_xticks(range(0, 25, 3))
    axis.set_ylabel("Estimated throughput [Mbps]")

    # Every frame holds the same reports, so the last one's slots give the load.
    t_index = next(reversed(served.values()))["t_index"]
    ues = t_index.groupby(hour).size() / t_index.groupby(hour).nunique()
    density = axis.twinx()
    density.plot(ues.index, ues.to_numpy(), color="0.45", lw=1.0, ls="--")
    density.set_ylabel("Mean UEs per interval", color="0.4")
    density.grid(False)

    handles += [
        plt.Line2D([], [], color="0.3", ls=":"),
        plt.Line2D([], [], color="0.45", ls="--"),
    ]
    names += ["5th percentile", "Mean UEs per interval (right axis)"]
    axis.legend(handles, names, fontsize=8, loc="upper left")
    axis.set_title("Estimated UE throughput by time of day")
    return figure


def throughput_vs_load(table: pd.DataFrame) -> Figure:
    """Each interval's median throughput against the UEs in it, per configuration.

    Args:
        table: :func:`src.evaluation.compare.interval_throughput` output.
    """
    statistic = "throughput_p50_mbps"
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for key, group in table.groupby("configuration", sort=False):
        means = group.groupby("ues")[statistic].mean()
        colour = COLOURS.get(str(key))
        axis.scatter(group["ues"], group[statistic], s=8, alpha=0.25, color=colour)
        axis.plot(
            means.index, means.to_numpy(), color=colour, lw=1.8, marker="o", ms=3, label=label(key)
        )
    axis.set_xlabel(label("ues"))
    axis.set_ylabel(label(statistic))
    axis.set_title("Median estimated throughput per interval versus number of UEs")
    axis.legend(fontsize=8)
    return figure


def cdf_plot(samples: dict[str, np.ndarray], xlabel: str, title: str) -> Figure:
    """Empirical CDF of each configuration's samples on one axis.

    Args:
        samples: Configuration key to its values; NaN values are dropped.
        xlabel: Horizontal axis label, with units.
        title: The figure title.
    """
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for key, values in samples.items():
        values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
        axis.step(
            values,
            np.arange(1, values.size + 1) / max(values.size, 1),
            where="post",
            color=COLOURS.get(key),
            lw=1.6,
            label=label(key),
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Share of UE reports at or below")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.legend(fontsize=8)
    return figure
