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
from matplotlib.colors import ListedColormap, LogNorm
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
        f"Coverage holes: {int((change > 0).sum())} closed, {int((change < 0).sum())} opened",
    )
    on = f", {label(band)}" if band else ""
    figure.suptitle(f"Coverage before and after — {label(name)}{on}")
    return figure


def demand_signal_maps(
    rsrp: np.ndarray,
    counts: np.ndarray,
    radio: dict[str, Any],
    cfg: DictConfig,
    *,
    sectors: pd.DataFrame | None = None,
    hotspots: pd.DataFrame | None = None,
    quantile: float = 0.75,
) -> Figure:
    """Where the demand is, where the signal is, and where the two disagree.

    Tiles with no report are blank in the demand panel: "nobody here" is a
    different statement from "one person here".
    """
    extent = maps.extent_of(radio)
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    occupied = np.where(counts > 0, counts.astype(float), np.nan)
    # Log scale: a handful of hotspot tiles would otherwise flatten every other tile to one colour.
    image = axes[0].imshow(
        occupied,
        origin="lower",
        extent=extent,
        norm=LogNorm(vmin=np.nanmin(occupied), vmax=np.nanmax(occupied))
        if np.isfinite(occupied).any()
        else None,
    )
    figure.colorbar(image, ax=axes[0], label="UE reports (log scale)")
    _overlay(axes[0], sectors, hotspots)
    _map_axes(axes[0], extent, "Traffic demand")

    image = axes[1].imshow(
        max_rsrp(rsrp),
        origin="lower",
        extent=extent,
        vmin=maps.RSRP_LIMITS[0],
        vmax=maps.RSRP_LIMITS[1],
    )
    figure.colorbar(image, ax=axes[1], label="Best-server RSRP [dBm]")
    _overlay(axes[1], sectors, hotspots)
    _map_axes(axes[1], extent, "Signal strength")

    flagged = maps.underserved(rsrp, counts, cfg, quantile)
    axes[2].imshow(
        np.where(flagged, 1.0, np.nan),
        origin="lower",
        extent=extent,
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    _overlay(axes[2], sectors, hotspots)
    _map_axes(axes[2], extent, f"High demand without good coverage: {int(flagged.sum())} tiles")
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
    axis.set_title("Serving frequency band mix")
    axis.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    return figure


def convergence_plot(frame: pd.DataFrame) -> Figure:
    """Hypervolume of every evaluation so far: on the search objectives, and on the KPIs.

    Args:
        frame: :func:`src.evaluation.compare.convergence` output.
    """
    titles = {
        "objectives": "Search objectives",
        "kpis": "Coverage, separation and median throughput",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for axis, (measures, title) in zip(axes, titles.items(), strict=True):
        part = frame[frame["measures"] == measures]
        for method, group in part.groupby("method", sort=False):
            axis.plot(
                group["iteration"],
                group["value"],
                lw=1.8,
                color=COLOURS.get(str(method)),
                label=label(method),
            )
        axis.set_xlabel("Evaluations")
        axis.set_ylabel("Hypervolume so far (higher is better)")
        axis.set_title(title)
        axis.legend()
    figure.suptitle("Search progress")
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
    axis.set_title(f"{label(x)} against {label(y)}")
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
    axis.set_title(f"Tilt change per sector and band — {label(name)}")
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
    figure.suptitle("Coverage classes")
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
    flat[0].legend(fontsize=8)
    figure.suptitle("KPIs per frequency layer")
    return figure


def band_tradeoff(frame: pd.DataFrame, band_labels: Sequence[str]) -> Figure:
    """Coverage rate against separation rate on each band alone, every candidate.

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
        axis.set_xlabel(label("coverage_rate"))
        axis.set_ylabel(label("separation_rate"))
        axis.set_title(label(band))
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Coverage against separation per frequency layer, every candidate")
    return figure


def interval_throughput_plot(table: pd.DataFrame, interval_s: float) -> Figure:
    """p05, median and mean throughput by time of day, with the UEs per interval on a second axis.

    Intervals are folded onto one day: the line is the mean over days of each
    time-of-day slot and the band its interquartile range, so the diurnal load
    cycle reads through the interval-to-interval noise.

    Args:
        table: :func:`src.evaluation.compare.interval_throughput` output.
        interval_s: Interval length, from the scenario manifest.
    """
    statistics = ("throughput_p05_mbps", "throughput_p50_mbps", "throughput_mean_mbps")
    hours = (table["t_index"] % round(86400.0 / interval_s)) * interval_s / 3600.0
    figure, axes = plt.subplots(
        len(statistics), 1, figsize=(12.0, 8.5), sharex=True, constrained_layout=True
    )
    ues = table.assign(hour=hours).drop_duplicates("t_index").groupby("hour")["ues"].mean()
    for axis, name in zip(axes, statistics, strict=True):
        for key, group in table.assign(hour=hours).groupby("configuration", sort=False):
            by_slot = group.groupby("hour")[name]
            colour = COLOURS.get(str(key))
            axis.plot(by_slot.mean().index, by_slot.mean(), lw=1.6, color=colour, label=label(key))
            axis.fill_between(
                by_slot.mean().index,
                by_slot.quantile(0.25),
                by_slot.quantile(0.75),
                color=colour,
                alpha=0.15,
                lw=0,
            )
        density = axis.twinx()
        density.plot(ues.index, ues.to_numpy(), color="0.45", lw=1.0, ls="--", label=label("ues"))
        density.set_ylabel(label("ues"), color="0.4")
        density.grid(False)
        axis.set_ylabel(label(name))
    axes[0].legend(fontsize=8, loc="upper left")
    figure.legend(
        [plt.Line2D([], [], color="0.45", ls="--")],
        [f"{label('ues')} (right axis)"],
        loc="upper right",
        fontsize=8,
    )
    axes[-1].set_xlabel("Time of day [h]")
    axes[-1].set_xticks(range(0, 25, 3))
    figure.suptitle("Estimated throughput by time of day: mean over days, interquartile band")
    return figure


def throughput_vs_load(table: pd.DataFrame, statistic: str = "throughput_p50_mbps") -> Figure:
    """One interval's throughput statistic against the UEs in it, per configuration.

    Args:
        table: :func:`src.evaluation.compare.interval_throughput` output.
        statistic: The column to draw.
    """
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
    axis.set_title(f"{label(statistic)} against interval load (line: mean per UE count)")
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
