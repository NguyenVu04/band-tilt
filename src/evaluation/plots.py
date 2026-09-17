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
from src.optim.objective import MEASURE_NAMES
from src.utils.plotting import label

# One colour per configuration, identical in every figure so a reader learns
# them once.
COLOURS = {
    "incumbent": "tab:red",
    "turbo": "tab:blue",
    "random": "tab:green",
    "rule": "tab:purple",
}


def _overlay(axis: plt.Axes, cells: pd.DataFrame | None, hotspots: pd.DataFrame | None) -> None:
    """Mark the nodes (crimson triangles) and demand hotspot centres (black crosses)."""
    if cells is not None and len(cells):
        axis.scatter(cells["x"], cells["y"], marker="^", s=45, color="crimson", zorder=3)
    if hotspots is not None and len(hotspots):
        axis.scatter(hotspots["x"], hotspots["y"], marker="x", s=70, color="black", zorder=4)


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
    cells: pd.DataFrame | None = None,
    name: str = "optimized",
) -> Figure:
    """Best-server RSRP before and after, and which tiles crossed the hole threshold.

    The third panel is the threshold crossing rather than a signed difference:
    a tile gaining 3 dB while staying a hole has changed nothing a KPI can see.

    Args:
        before: Best-server RSRP of the incumbent, ``[n_rows, n_cols]``.
        after: Best-server RSRP of the other configuration.
        radio: Any radio-map archive, for the grid extent.
        cfg: Composed config; reads ``kpi.hole_dbm``.
        cells: Optional cell table with ``x`` and ``y``.
        name: Key of the second configuration, for its title.
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
        _overlay(axis, cells, None)
        _map_axes(axis, extent, label(key))

    change = maps.change_mask(before, after, cfg)
    image = axes[2].imshow(
        change, origin="lower", extent=extent, aspect="equal", cmap="coolwarm_r", vmin=-1, vmax=1
    )
    bar = figure.colorbar(image, ax=axes[2], ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["Hole opened", "Unchanged", "Hole closed"])
    _overlay(axes[2], cells, None)
    _map_axes(
        axes[2],
        extent,
        f"Coverage holes: {int((change > 0).sum())} closed, {int((change < 0).sum())} opened",
    )
    figure.suptitle(f"Coverage before and after — {label(name)}")
    return figure


def demand_signal_maps(
    rsrp: np.ndarray,
    counts: np.ndarray,
    radio: dict[str, Any],
    cfg: DictConfig,
    *,
    cells: pd.DataFrame | None = None,
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
    figure.colorbar(image, ax=axes[0], label="PRBs required, busiest interval (log scale)")
    _overlay(axes[0], cells, hotspots)
    _map_axes(axes[0], extent, "Traffic demand")

    image = axes[1].imshow(
        max_rsrp(rsrp),
        origin="lower",
        extent=extent,
        vmin=maps.RSRP_LIMITS[0],
        vmax=maps.RSRP_LIMITS[1],
    )
    figure.colorbar(image, ax=axes[1], label="Best-server RSRP [dBm]")
    _overlay(axes[1], cells, hotspots)
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
    _overlay(axes[2], cells, hotspots)
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
    cells: pd.DataFrame | None = None,
    hotspots: pd.DataFrame | None = None,
) -> Figure:
    """One raster per panel, side by side, on one colour scale.

    Args:
        panels: Title to ``[n_rows, n_cols]`` raster, drawn in order.
        radio: Any radio-map archive, for the grid extent.
        colorbar_label: Colour bar label, with units.
        vmin: Lower limit; the data minimum when None.
        vmax: Upper limit; the data maximum when None.
        symmetric: Centre the scale on zero with a diverging colormap, for
            difference maps.
        cmap: Colormap; viridis, or ``RdBu_r`` when ``symmetric``.
        cells: Optional cell table with ``x`` and ``y``.
        hotspots: Optional hotspot table with ``x`` and ``y``.
    """
    extent = maps.extent_of(radio)
    values = [np.where(np.isfinite(panel), panel, np.nan) for panel in panels.values()]
    present = np.concatenate([value[np.isfinite(value)] for value in values] + [np.zeros(1)])
    if symmetric:
        # 99th percentile, not the maximum: a few tiles beside the masts change by tens of dB
        # and would otherwise wash out every other change.
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
        _overlay(axis, cells, hotspots)
        _map_axes(axis, extent, title)
    figure.colorbar(image, ax=axes[0].tolist(), label=colorbar_label)
    return figure


def utilisation_heatmaps(tables: dict[str, pd.DataFrame]) -> Figure:
    """Peak PRB utilisation per cell-band, one panel per configuration.

    Args:
        tables: Configuration key to :func:`src.evaluation.compare.cell_band_load` output.
    """
    first = next(iter(tables.values()))
    cell_order = list(dict.fromkeys(first["cell"]))
    band_order = list(dict.fromkeys(first["band"]))
    figure, axes = plt.subplots(
        1,
        len(tables),
        figsize=(1.3 * len(band_order) * len(tables) + 2.5, 0.3 * len(cell_order) + 1.8),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for index, (axis, (key, table)) in enumerate(zip(axes[0], tables.items(), strict=True)):
        grid = table.pivot(index="cell", columns="band", values="peak_utilisation")
        grid = grid.reindex(index=cell_order, columns=band_order).to_numpy()
        image = axis.imshow(grid, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
        axis.grid(False)
        for (row, col), value in np.ndenumerate(grid):
            axis.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=7)
        axis.set_xticks(range(len(band_order)), [label(band) for band in band_order])
        axis.set_yticks(range(len(cell_order)), cell_order if index == 0 else [])
        axis.set_title(label(key))
    figure.colorbar(image, ax=axes[0].tolist(), label="Peak PRB utilisation, busiest interval")
    figure.suptitle("Cell load per frequency band")
    return figure


def band_share_bars(summaries: dict[str, dict[str, float]], band_labels: Sequence[str]) -> Figure:
    """Share of UE reports per serving band, and not served, per configuration.

    Args:
        summaries: Configuration key to :func:`src.evaluation.compare.service_summary` output.
        band_labels: Band keys, as in the summaries' ``share_<band>`` entries.
    """
    keys = list(summaries)
    figure, axis = plt.subplots(figsize=(1.8 * len(keys) + 3.0, 4.5), constrained_layout=True)
    bottom = np.zeros(len(keys))
    parts = [(f"share_{band}", label(band)) for band in band_labels]
    parts.append(("not_served_share", "Not served"))
    for key, name in parts:
        heights = np.array([summaries[config][key] for config in keys])
        colour = "0.35" if key == "not_served_share" else None
        axis.bar(
            [label(config) for config in keys], heights, bottom=bottom, label=name, color=colour
        )
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


def kpi_comparison(summary: pd.DataFrame) -> Figure:
    """Mean relative improvement over the incumbent per measure, one panel each.

    Relative, in percent of the incumbent's value, so a rate and the cell-edge
    RSRP read on a comparable scale; see
    :func:`src.evaluation.compare.relative_improvement`. Each panel is signed so
    positive is better whichever direction its KPI runs. No error bars: the
    seed interval is in the scoreboard table.

    Args:
        summary: :func:`src.evaluation.compare.seed_summary` output.
    """
    improvement = compare.relative_improvement(summary).set_index("method")
    methods = list(improvement.index)
    positions = np.arange(len(methods))

    figure, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)
    for axis in axes.ravel()[len(MEASURE_NAMES) :]:
        axis.set_visible(False)
    for axis, name in zip(axes.ravel(), MEASURE_NAMES, strict=False):
        values = improvement[name].to_numpy()
        axis.axhline(0, color="0.4", lw=1, zorder=1)
        bars = axis.bar(
            positions,
            values,
            width=0.6,
            color=[COLOURS.get(method) for method in methods],
            zorder=2,
        )
        axis.bar_label(bars, fmt="%+.1f%%", fontsize=7, padding=2)
        axis.margins(y=0.15)
        axis.set_xticks(positions, [label(method) for method in methods], fontsize=8)
        axis.set_ylabel("Improvement [%]", fontsize=8)
        axis.set_title(label(name), fontsize=9)
    figure.suptitle("Relative KPI improvement over the current configuration (higher is better)")
    return figure


def convergence_plot(frame: pd.DataFrame) -> Figure:
    """Best objective so far per evaluation: mean over seeds, with the min–max range.

    Args:
        frame: :func:`src.evaluation.compare.convergence` output.
    """
    part = frame[frame["kpi"] == "objective"]
    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    for method, group in part.groupby("method", sort=False):
        stats = group.groupby("iteration")["value"].agg(["mean", "min", "max"])
        colour = COLOURS.get(str(method))
        n_seeds = group["seed"].nunique()
        axis.plot(
            stats.index, stats["mean"], lw=1.8, color=colour, label=f"{label(method)} (n={n_seeds})"
        )
        axis.fill_between(stats.index, stats["min"], stats["max"], color=colour, alpha=0.2)
    axis.set_xlabel("Evaluations")
    axis.set_ylabel("Best objective so far (higher is better)")
    axis.set_title("Search progress")
    axis.legend()
    return figure


def tilt_movement_plot(best_tilt: pd.DataFrame, name: str) -> Figure:
    """Where every cell-band ended up, and how far it moved.

    Args:
        best_tilt: A run's ``best_tilt`` table.
        name: Key of the configuration, for the title.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    for band, group in best_tilt.groupby("band", observed=True):
        axes[0].scatter(
            group["current_tilt_deg"], group["optimized_tilt_deg"], s=45, label=label(band)
        )
    values = best_tilt[["current_tilt_deg", "optimized_tilt_deg"]].to_numpy()
    limits = [values.min() - 1.0, values.max() + 1.0]
    axes[0].plot(limits, limits, color="0.7", ls="--", lw=1, label="No change")
    axes[0].set_xlabel("Current tilt [°]")
    axes[0].set_ylabel("Proposed tilt [°]")
    axes[0].set_title("Proposed against current tilt")
    axes[0].legend(fontsize=8, title="Band")

    order = best_tilt.sort_values("delta_tilt_deg")
    axes[1].barh(
        range(len(order)),
        order["delta_tilt_deg"],
        color=["tab:blue" if value < 0 else "tab:orange" for value in order["delta_tilt_deg"]],
    )
    axes[1].set_yticks(
        range(len(order)),
        [f"{cell} {label(band)}" for cell, band in zip(order["cell"], order["band"], strict=True)],
        fontsize=6,
    )
    axes[1].axvline(0, color="0.4", lw=1)
    axes[1].set_xlabel("Tilt change [°] (negative: uptilt, positive: downtilt)")
    axes[1].set_title("Tilt change per cell and band")
    figure.suptitle(f"Recommended antenna tilt changes — {label(name)}")
    return figure


def tradeoff_scatter(frame: pd.DataFrame, x: str, y: str) -> Figure:
    """Every evaluated configuration on two measures, with each method's pick and the Pareto front.

    Args:
        frame: :func:`src.evaluation.compare.candidates` output.
        x: Measure on the horizontal axis.
        y: Measure on the vertical axis.
    """
    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    searched = frame[frame["iteration"] > 0]
    for method, group in searched.groupby("method", sort=False):
        colour = COLOURS.get(str(method))
        axis.scatter(group[x], group[y], s=14, alpha=0.4, color=colour, label=label(method))
        pick = group.loc[group["objective"].idxmax()]
        axis.scatter(pick[x], pick[y], marker="*", s=260, color=colour, edgecolor="black", zorder=4)
    axis.scatter(
        [],
        [],
        marker="*",
        s=260,
        color="white",
        edgecolor="black",
        label="Best objective per method",
    )
    incumbent = frame[frame["iteration"] == 0].iloc[0]
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
    axis.plot(
        front[x], front[y], color="0.2", ls="--", lw=1.2, marker="o", ms=4, label="Pareto front"
    )
    axis.set_xlabel(f"{label(x)} ({compare.direction(x)})")
    axis.set_ylabel(f"{label(y)} ({compare.direction(y)})")
    axis.set_title(f"{label(x)} against {label(y)}")
    axis.legend(fontsize=8)
    return figure


def tilt_delta_heatmap(best_tilt: pd.DataFrame, name: str) -> Figure:
    """Tilt change per cell and band on one diverging scale.

    Args:
        best_tilt: A run's ``best_tilt`` table.
        name: Key of the configuration, for the title.
    """
    table = best_tilt.astype({"cell": str, "band": str})
    cells = list(dict.fromkeys(table["cell"]))
    bands = list(dict.fromkeys(table["band"]))
    grid = table.pivot(index="cell", columns="band", values="delta_tilt_deg")
    grid = grid.reindex(index=cells, columns=bands).to_numpy()
    limit = float(np.nanmax(np.abs(grid))) or 1.0

    figure, axis = plt.subplots(
        figsize=(1.6 * len(bands) + 2.5, 0.35 * len(cells) + 1.5), constrained_layout=True
    )
    image = axis.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.grid(False)
    for (row, col), value in np.ndenumerate(grid):
        axis.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(bands)), [label(band) for band in bands])
    axis.set_yticks(range(len(cells)), cells)
    figure.colorbar(image, ax=axis, label="Tilt change [°] (negative: uptilt)")
    axis.set_title(f"Tilt change per cell and band — {label(name)}")
    return figure


def coverage_class_maps(
    rasters: dict[str, np.ndarray],
    radio: dict[str, Any],
    cfg: DictConfig,
    *,
    cells: pd.DataFrame | None = None,
) -> Figure:
    """Hole, weak and good coverage per configuration, on identical classes.

    Args:
        rasters: Configuration key to its radio map's ``rsrp_dbm``.
        radio: Any radio-map archive, for the grid extent.
        cfg: Composed config; reads ``kpi.hole_dbm`` and ``kpi.weak_dbm``.
        cells: Optional cell table with ``x`` and ``y``.
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
        _overlay(axis, cells, None)
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
