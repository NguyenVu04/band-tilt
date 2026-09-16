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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.evaluation import maps
from src.kpi.capacity import max_rsrp
from src.optim.objective import KPI_NAMES, MAXIMISED
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
    """Mean improvement over the incumbent per KPI, in units of that KPI's tolerance.

    Dividing by the tolerance puts incomparable KPIs on one axis: height 1 is
    one noise floor. Positive is better whichever direction the KPI runs. Error
    bars are the 95 % interval over seeds; the shaded band is a tie.

    Args:
        summary: :func:`src.evaluation.compare.seed_summary` output.
    """
    rows = summary[summary["kpi"].isin(KPI_NAMES)]
    methods = list(dict.fromkeys(rows["method"]))
    positions = np.arange(len(KPI_NAMES))
    width = 0.8 / max(len(methods), 1)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    axis.axhspan(-1, 1, color="0.88", zorder=0, label="Within solver noise")
    axis.axhline(0, color="0.4", lw=1, zorder=1)
    for index, method in enumerate(methods):
        mine = rows[rows["method"] == method].set_index("kpi").loc[list(KPI_NAMES)]
        sign = np.array([1.0 if name in MAXIMISED else -1.0 for name in KPI_NAMES])
        height = sign * mine["mean_delta"].to_numpy() / mine["tolerance"].to_numpy()
        half = (mine["ci95_high"] - mine["mean"]).to_numpy() / mine["tolerance"].to_numpy()
        axis.bar(
            positions + index * width - 0.4 + width / 2,
            height,
            width=width,
            yerr=np.where(np.isfinite(half), half, 0.0),
            capsize=3,
            label=label(method),
            color=COLOURS.get(method),
            zorder=2,
        )
    axis.set_xticks(positions, [label(name) for name in KPI_NAMES])
    axis.set_ylabel("Improvement over current configuration\n[multiples of tolerance]")
    axis.set_title("KPI improvement against solver noise")
    axis.legend()
    return figure


def convergence_plot(frame: pd.DataFrame) -> Figure:
    """Best weighted score so far per evaluation: mean over seeds, with the min–max range.

    Args:
        frame: :func:`src.evaluation.compare.convergence` output.
    """
    part = frame[frame["kpi"] == "score"]
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
    axis.set_ylabel("Best weighted score so far (higher is better)")
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
