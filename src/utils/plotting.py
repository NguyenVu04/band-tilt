"""Figure defaults and display names, so every notebook's plots and tables read alike."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt

_RC_PARAMS = {
    "figure.figsize": (9.0, 5.0),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 10,
    "legend.frameon": False,
    # Perceptually uniform, and readable in greyscale once a figure is printed.
    "image.cmap": "viridis",
}

# The one spelling of every name a reader sees: KPIs, bands, methods and table
# columns. Code keeps the short keys; only presentation goes through `label`.
LABELS = {
    "hole_rate": "Coverage hole rate",
    "overlap_rate": "Co-band overlap rate",
    "served_ratio": "Served UE ratio",
    "weak_rate": "Weak coverage rate",
    "edge_rsrp_dbm": "Cell-edge RSRP",
    "j_radio": "Radio utility J_radio",
    "j_load": "Load utility J_load",
    "score": "Objective J",
    "b700": "700 MHz",
    "b1800": "1800 MHz",
    "b2600": "2600 MHz",
    "incumbent": "Current configuration",
    "turbo": "TuRBO",
    "random": "Random search",
    "rule": "Rule-based sweep",
    "kpi": "KPI",
    "method": "Method",
    "seed": "Seed",
    "run": "Run",
    "band": "Band",
    "cell": "Cell",
    "node": "Node",
    "configuration": "Configuration",
    "coverage": "Coverage class",
    "direction": "Direction",
    "verdict": "Verdict",
    "check": "Check",
    "holds": "Holds",
    "offenders": "Offending runs",
    "azimuth_deg": "Azimuth [°]",
    "current_tilt_deg": "Current tilt [°]",
    "optimized_tilt_deg": "Proposed tilt [°]",
    "delta_tilt_deg": "Tilt change [°]",
    "tilt_min_deg": "Minimum tilt [°]",
    "tilt_max_deg": "Maximum tilt [°]",
    "n_seeds": "Seeds",
    "n_solver_seeds": "Solver seeds",
    "mean": "Mean",
    "std": "Standard deviation",
    "ci95_low": "95% CI lower",
    "ci95_high": "95% CI upper",
    "mean_delta": "Mean change",
    "init_median": "Initial design, median score",
    "candidate_median": "All candidates, median score",
    "candidate_p90": "All candidates, 90th percentile score",
    "winner": "Best score found",
    "reference": "Reference",
    "n_pairs": "Seed pairs",
    "mean_gain": "Mean score gain",
    "method_better": "Seeds won",
    "wilcoxon_p": "Wilcoxon p-value",
    "scheme": "Weighting",
    "mean_winner_score": "Mean best score",
    "rank": "Rank",
    "same_winner_share": "Share with unchanged best configuration",
    "pooled_std": "Solver noise, standard deviation",
    "positive_share": "Share of solver seeds improved",
    "recorded": "Recorded",
    "recomputed": "Recomputed",
    "abs_gap": "Absolute gap",
    "tiles": "Tiles",
    "tile_share": "Share of area",
    "prb": "Peak PRB demand",
    "demand_share": "Share of demand",
    "evaluations": "Evaluations",
    "best_iteration": "Best evaluation",
    "ray_tracing_min": "Ray tracing [min]",
    "wall_clock_min": "Wall clock [min]",
    "kpis_improved": "KPIs improved",
    "kpis_worsened": "KPIs worsened",
    "iteration": "Evaluation",
    "value": "Best so far",
    "reports": "UE reports",
    "not_served_share": "Share not served",
    "sinr_p10_db": "Served SINR, 10th percentile [dB]",
    "sinr_median_db": "Served SINR, median [dB]",
    "prb_per_served_ue_median": "PRBs per served UE, median",
    "share_b700": "Share served on 700 MHz",
    "share_b1800": "Share served on 1800 MHz",
    "share_b2600": "Share served on 2600 MHz",
    "served_reports_before": "Served reports, before",
    "served_reports_after": "Served reports, after",
    "served_reports_change": "Served reports, change",
    "peak_utilisation_before": "Peak PRB utilisation, before",
    "peak_utilisation_after": "Peak PRB utilisation, after",
    "peak_utilisation_change": "Peak PRB utilisation, change",
    "median_sinr_db_before": "Median SINR before [dB]",
    "median_sinr_db_after": "Median SINR after [dB]",
    "median_sinr_db_change": "Median SINR change [dB]",
    "n_cells": "Cells",
    "n_moved": "Cells moved",
    "mean_abs_delta_deg": "Mean absolute tilt change [°]",
    "max_abs_delta_deg": "Largest tilt change [°]",
    "mean_delta_deg": "Mean tilt change [°]",
    "parameter": "Parameter",
    "setting": "Setting",
    "phase": "Phase",
    "budget": "Evaluations",
    "gamma": "gamma",
    "same_best": "Same pick as the run",
    "mean_neighbours_covered": "Mean overlap neighbours, covered tiles",
    "mean_neighbours_all": "Mean overlap neighbours, all tiles",
    "share_0_neighbours": "Share with 0 neighbours",
    "share_1_neighbours": "Share with 1 neighbour",
    "share_2_neighbours": "Share with 2 neighbours",
    "share_3plus_neighbours": "Share with 3+ neighbours",
    "coverage_share": "Share of area covered by the band",
    "mean_band_rsrp_dbm": "Mean band RSRP where covered [dBm]",
    "serving_tile_share": "Share of area served on the band",
    "served_share": "Share of UE reports served on the band",
    "served_sinr_median_db": "Served SINR on the band, median [dB]",
}


def label(name: object) -> str:
    """The display name of a key in :data:`LABELS`, or the key itself when it has none."""
    return LABELS.get(str(name), str(name))


def setup_plotting() -> None:
    """Apply the project's figure defaults.

    Side effect: mutates the global matplotlib ``rcParams``.
    """
    plt.rcParams.update(_RC_PARAMS)


def save_fig(
    figure: matplotlib.figure.Figure,
    name: str,
    in_colab: bool,
    directory: str | Path = "reports/figures",
) -> None:
    """Write ``figure`` to ``directory/name.png`` so it can be viewed without rerunning it.

    Skipped on Colab: ``/content`` does not survive a runtime reset and
    ``reports/`` there is a fresh clone, so saving would silently discard the
    file.
    """
    if in_colab:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png")
