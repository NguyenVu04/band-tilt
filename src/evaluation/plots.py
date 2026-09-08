"""The figures, following the conventions ``notebooks/01_eda.ipynb`` established.

Every map is drawn ``origin="lower"`` on the grid's metric extent, because row 0
of the raster is the lowest y; matplotlib's default would mirror it. Colours come
from :func:`src.utils.plotting.setup_plotting` — viridis for images, the default
cycle for lines — so nothing here sets a colormap except where the data is
diverging and needs one.

Each function returns a :class:`~matplotlib.figure.Figure` and shows nothing, so
a notebook can display it and :mod:`src.evaluation.export` can save it without
either knowing about the other.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.evaluation import maps
from src.evaluation.compare import TIE
from src.optim.objective import KPI_NAMES, MAXIMISED, tolerances

# One colour per method, kept identical across every figure so a reader learns
# them once. The incumbent is red everywhere, and is never a method.
METHOD_COLOURS = {
    "mobo": "tab:blue",
    "random": "tab:green",
    "rule": "tab:purple",
}
INCUMBENT_COLOUR = "tab:red"


def _overlay(axis: plt.Axes, cells: pd.DataFrame | None, hotspots: pd.DataFrame | None) -> None:
    """Mark the transmitters and the demand hotspots on a map.

    The marker vocabulary is 01's: crimson triangles are cells, black crosses
    are hotspot centres.
    """
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
    label: str = "optimized",
) -> Figure:
    """Best-server RSRP before and after, and which tiles crossed the threshold.

    Args:
        before: Best-server RSRP of the incumbent, ``[n_rows, n_cols]``.
        after: Best-server RSRP of the optimized configuration.
        radio: Any radio-map archive, for the grid extent.
        cfg: Composed config; reads ``cfg.kpi.hole_dbm``.
        cells: Optional cell table with ``x`` and ``y``, to mark the masts.
        label: What to call the second configuration.

    The third panel is deliberately the threshold crossing rather than a signed
    RSRP difference: a tile gaining 3 dB while staying a hole has changed
    nothing any KPI can see.
    """
    extent = maps.extent_of(radio)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    for axis, values, title in (
        (axes[0], before, "Best-server RSRP — incumbent"),
        (axes[1], after, f"Best-server RSRP — {label}"),
    ):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=maps.RSRP_LIMITS[0],
            vmax=maps.RSRP_LIMITS[1],
        )
        figure.colorbar(image, ax=axis, label="RSRP [dBm]")
        _overlay(axis, cells, None)
        _map_axes(axis, extent, title)

    change = maps.change_mask(before, after, cfg)
    image = axes[2].imshow(
        change, origin="lower", extent=extent, aspect="equal", cmap="coolwarm_r", vmin=-1, vmax=1
    )
    figure.colorbar(
        image, ax=axes[2], label="-1 opened   0 unchanged   +1 filled", ticks=[-1, 0, 1]
    )
    _overlay(axes[2], cells, None)
    _map_axes(
        axes[2],
        extent,
        f"Holes filled {int((change > 0).sum())}, opened {int((change < 0).sum())}",
    )
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
    """Where the demand is, where the signal is, and where they disagree.

    Empty tiles are left blank in the demand panel rather than drawn as the
    darkest colour: about half the grid holds no report at all, and "nobody
    here" is a different statement from "one person here".

    The third panel is the intersection worth acting on — tiles that are busy
    and not well covered — as distinct from tiles that are merely dark.
    """
    extent = maps.extent_of(radio)
    best = maps.best_server(rsrp)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    occupied = np.where(counts > 0, counts.astype(float), np.nan)
    image = axes[0].imshow(occupied, origin="lower", extent=extent, aspect="equal")
    figure.colorbar(image, ax=axes[0], label="UE reports, all intervals")
    _overlay(axes[0], cells, hotspots)
    _map_axes(axes[0], extent, f"Demand — {int((counts > 0).mean() * 100)}% of tiles occupied")

    image = axes[1].imshow(
        best,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=maps.RSRP_LIMITS[0],
        vmax=maps.RSRP_LIMITS[1],
    )
    figure.colorbar(image, ax=axes[1], label="RSRP [dBm]")
    _overlay(axes[1], cells, hotspots)
    _map_axes(axes[1], extent, "Signal — best-server RSRP")

    flagged = maps.underserved(rsrp, counts, cfg, quantile)
    image = axes[2].imshow(
        flagged.astype(float),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(image, ax=axes[2], label="busy and not well covered", ticks=[0, 1])
    _overlay(axes[2], cells, hotspots)
    _map_axes(axes[2], extent, f"Under-served — {int(flagged.sum())} tiles")
    return figure


def coverage_cdf(best: np.ndarray, counts: np.ndarray, cfg: DictConfig) -> Figure:
    """Coverage weighted by area against coverage weighted by users.

    The horizontal gap between the two curves is the whole point: at the hole
    threshold it is the difference between the hole rate the optimizer
    minimises and the share of users actually in a hole.
    """
    levels, tile_share, demand_share = maps.coverage_cdf(best, counts)
    hole_dbm = float(cfg.kpi.hole_dbm)
    weak_dbm = float(cfg.kpi.weak_dbm)

    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    axis.plot(levels, tile_share, lw=1.8, label="weighted by tiles (the KPI)")
    axis.plot(levels, demand_share, lw=1.8, label="weighted by UE reports")

    for threshold, name in ((hole_dbm, "hole"), (weak_dbm, "weak")):
        axis.axvline(threshold, color="0.5", ls="--", lw=1)
        index = int(np.searchsorted(levels, threshold, side="right")) - 1
        if 0 <= index < levels.size:
            axis.annotate(
                f"{name} @ {threshold:.0f} dBm\ntiles {tile_share[index]:.3f}"
                f"\nusers {demand_share[index]:.3f}",
                (threshold, tile_share[index]),
                xytext=(8, -4),
                textcoords="offset points",
                fontsize=8,
                color="0.3",
            )

    axis.set_xlim(levels.min(), maps.RSRP_LIMITS[1])
    axis.set_xlabel("best-server RSRP [dBm]")
    axis.set_ylabel("share at or below")
    axis.set_title("Coverage by area and by demand")
    axis.legend(loc="upper left")
    return figure


def kpi_comparison(deltas: dict[str, pd.DataFrame], cfg: DictConfig) -> Figure:
    """Improvement over the incumbent, per KPI, in units of that KPI's tolerance.

    The five KPIs are on incomparable scales — a share of the grid beside a
    UE-weighted score — so raw deltas cannot share an axis. Dividing by each
    KPI's tolerance
    puts them on one: a bar of height 1 is exactly one noise floor. Positive is
    always better, whichever direction the KPI runs.

    The shaded band is +/-1 tolerance. A bar inside it is a tie, and the
    configuration behind it has not been shown to differ from the incumbent.
    """
    tolerance = tolerances(cfg)
    labels = list(deltas)
    positions = np.arange(len(KPI_NAMES))
    width = 0.8 / max(len(labels), 1)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    axis.axhspan(-1, 1, color="0.85", zorder=0)
    axis.axhline(0, color="0.4", lw=1, zorder=1)

    for index, label in enumerate(labels):
        table = deltas[label].set_index("kpi")
        heights = []
        for kpi_index, name in enumerate(KPI_NAMES):
            delta = float(table.loc[name, "delta"])
            improvement = delta if name in MAXIMISED else -delta
            heights.append(improvement / tolerance[kpi_index])
        axis.bar(
            positions + index * width - 0.4 + width / 2,
            heights,
            width=width,
            label=label,
            color=METHOD_COLOURS.get(label),
            zorder=2,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([name.replace("_", "\n") for name in KPI_NAMES], fontsize=8)
    axis.set_ylabel("improvement over incumbent\n[tolerances; shaded band is a tie]")
    axis.set_title("What each method gained, against the noise floor")
    axis.legend()
    return figure


def convergence_plot(frame: pd.DataFrame) -> Figure:
    """Best value so far, per KPI, with the methods overlaid.

    One panel per KPI because they run in different directions and on different
    scales. A flat line is a method that never improved on its own first point.
    """
    figure, axes = plt.subplots(1, len(KPI_NAMES), figsize=(17.0, 3.4), constrained_layout=True)
    for axis, name in zip(np.ravel(axes), KPI_NAMES, strict=True):
        part = frame[frame["kpi"] == name]
        for method, group in part.groupby("method", observed=True):
            axis.plot(
                group["iteration"],
                group["value"],
                lw=1.4,
                label=method,
                color=METHOD_COLOURS.get(str(method)),
            )
        arrow = "higher is better" if name in MAXIMISED else "lower is better"
        axis.set_title(f"{name}\n({arrow})", fontsize=9)
        axis.set_xlabel("evaluation")
    np.ravel(axes)[0].set_ylabel("best so far")
    np.ravel(axes)[-1].legend(fontsize=8)
    return figure


def pareto_plot(runs: list[Any], cfg: DictConfig) -> Figure:
    """Hole rate against band priority score, every evaluation, every method.

    The trade this problem is made of: the low band reaches furthest, so it
    serves most users and holds the score down, and buying the score means
    tilting it off the ground it was covering.
    """
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    for run in runs:
        history = run.history
        colour = METHOD_COLOURS.get(run.method)
        axis.scatter(
            history["hole_rate"],
            history["band_priority_score"],
            s=18,
            alpha=0.45,
            color=colour,
            label=f"{run.method} ({len(history)} evals)",
        )
        front = history[history["on_pareto"]]
        axis.scatter(
            front["hole_rate"], front["band_priority_score"], s=48, color=colour, edgecolor="white"
        )

    if runs:
        incumbent = runs[0].incumbent_kpi
        axis.scatter(
            [incumbent.hole_rate],
            [incumbent.band_priority_score],
            s=180,
            marker="*",
            color=INCUMBENT_COLOUR,
            zorder=5,
            label="incumbent",
        )
    axis.set_xlabel("hole rate (minimise)")
    axis.set_ylabel("band priority score (maximise)")
    axis.set_title("Filled markers are non-dominated within their run")
    axis.legend(fontsize=8)
    return figure


def tilt_movement_plot(run: Any) -> Figure:
    """Where every cell-band ended up, and how far it moved.

    Reported, never optimized: a movement penalty is an explicit non-goal, so
    nothing in the objective has seen this.
    """
    table = run.best_tilt
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)

    for band, group in table.groupby("band", observed=True):
        axes[0].scatter(group["current_tilt_deg"], group["optimized_tilt_deg"], s=45, label=band)
    values = table[["current_tilt_deg", "optimized_tilt_deg"]].to_numpy()
    limits = [values.min() - 1.0, values.max() + 1.0]
    axes[0].plot(limits, limits, color="0.7", ls="--", lw=1)
    axes[0].set_xlabel("current tilt [deg]")
    axes[0].set_ylabel("optimized tilt [deg]")
    axes[0].set_title(f"{run.label} — unchanged cells sit on the line")
    axes[0].legend(fontsize=8, title="band")

    order = table.sort_values("delta_tilt_deg")
    axes[1].barh(
        range(len(order)),
        order["delta_tilt_deg"],
        color=["tab:blue" if value < 0 else "tab:orange" for value in order["delta_tilt_deg"]],
    )
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels(order["cell"] + " " + order["band"], fontsize=6)
    axes[1].axvline(0, color="0.4", lw=1)
    axes[1].set_xlabel("delta tilt [deg]")
    axes[1].set_title("Down is negative, up is positive")
    return figure


def verdict_counts(deltas: dict[str, pd.DataFrame]) -> Figure:
    """How many KPIs each method actually moved, past the noise floor.

    A blunt summary, and often the honest headline: on a small budget every bar
    is a tie, and no amount of KPI table detail changes that.
    """
    from src.evaluation.compare import BETTER, WORSE

    labels = list(deltas)
    counted = {
        verdict: [int((deltas[label]["verdict"] == verdict).sum()) for label in labels]
        for verdict in (BETTER, TIE, WORSE)
    }
    figure, axis = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    bottom = np.zeros(len(labels))
    for verdict, colour in ((BETTER, "tab:green"), (TIE, "0.75"), (WORSE, "tab:red")):
        axis.bar(labels, counted[verdict], bottom=bottom, label=verdict, color=colour)
        bottom += np.array(counted[verdict])
    axis.set_ylabel("KPIs")
    axis.set_ylim(0, len(KPI_NAMES))
    axis.set_title("Verdict per KPI, against each method's own incumbent")
    axis.legend(fontsize=8)
    return figure
