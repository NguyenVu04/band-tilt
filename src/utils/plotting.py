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
    "weak_rate": "Weak coverage rate",
    "overlap_rate": "Co-band overlap rate",
    "overlap_neighbor_mean": "Overlap neighbours per covered tile",
    "rsrp_p50_dbm": "Median RSRP, p50 [dBm]",
    "rsrp_p05_dbm": "Cell-edge RSRP, p05 [dBm]",
    "sinr_p50_db": "Median SINR, p50 [dB]",
    "sinr_p05_db": "Cell-edge SINR, p05 [dB]",
    "estimated_throughput_p05_mbps": "Cell-edge estimated throughput, p05 [Mbps]",
    "estimated_throughput_p50_mbps": "Median estimated throughput, p50 [Mbps]",
    "estimated_throughput_mean_mbps": "Mean estimated throughput [Mbps]",
    "coverage_rate": "Coverage rate",
    "separation_rate": "Separation rate",
    "delta_coverage_rate": "Coverage rate change",
    "delta_separation_rate": "Separation rate change",
    "delta_estimated_throughput_p50_mbps": "Median estimated throughput change [Mbps]",
    "hv_contribution": "Hypervolume contribution",
    "measures": "Measures",
    "objectives": "Search objectives",
    "kpis": "Coverage, separation, median throughput",
    "n_method": "Candidates, method",
    "n_reference": "Candidates, reference",
    "median_method": "Median, method",
    "median_reference": "Median, reference",
    "mann_whitney_p": "Mann–Whitney U p-value (one-sided)",
    "cliffs_delta": "Cliff's delta",
    "front_points": "Front points",
    "c_metric": "C-metric (share of the reference front dominated)",
    "dominate_incumbent": "Candidates dominating the current configuration",
    "dominate_incumbent_share": "Share dominating the current configuration",
    "initial_design": "Initial design",
    "restarts": "Restarts",
    "search": "Search",
    "global_seed": "Global seed (solver stream)",
    "shared_design_identical": "Shared design identical",
    "shared_design_max_gap": "Largest measure gap on the shared design",
    "overhead_min": "Overhead [min]",
    "ues": "UEs in the interval",
    "throughput_p05_mbps": "Throughput p05 [Mbps]",
    "throughput_p50_mbps": "Throughput median [Mbps]",
    "throughput_mean_mbps": "Throughput mean [Mbps]",
    "coverage_objective": "Coverage objective",
    "separation_objective": "Separation objective",
    "throughput_objective": "Throughput objective, mean of ln(1 + R)",
    "hypervolume": "Hypervolume",
    "all": "All bands",
    "b700": "700 MHz",
    "b1800": "1800 MHz",
    "b2600": "2600 MHz",
    "incumbent": "Current configuration",
    "morbo": "MORBO",
    "random": "Random search",
    "kpi": "KPI",
    "method": "Method",
    "seed": "Seed",
    "run": "Run",
    "band": "Band",
    "sector": "Sector",
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
    "incumbent_hv": "Hypervolume, current configuration",
    "initial_design_hv": "Hypervolume, initial design",
    "final_hv": "Hypervolume, every evaluation",
    "pareto_points": "Pareto points",
    "reference": "Reference",
    "rank": "Rank",
    "recorded": "Recorded",
    "recomputed": "Recomputed",
    "abs_gap": "Absolute gap",
    "tiles": "Tiles",
    "tile_share": "Share of area",
    "demand_share": "Share of demand",
    "evaluations": "Evaluations",
    "ray_tracing_min": "Ray tracing [min]",
    "wall_clock_min": "Wall clock [min]",
    "iteration": "Evaluation",
    "value": "Best so far",
    "reports": "UE reports",
    "share_b700": "Share served on 700 MHz",
    "share_b1800": "Share served on 1800 MHz",
    "share_b2600": "Share served on 2600 MHz",
    "parameter": "Parameter",
    "setting": "Setting",
    "phase": "Phase",
    "budget": "Evaluations",
    "mean_neighbours_covered": "Mean overlap neighbours, covered tiles",
    "mean_neighbours_all": "Mean overlap neighbours, all tiles",
    "share_0_neighbours": "Share with 0 neighbours",
    "share_1_neighbours": "Share with 1 neighbour",
    "share_2_neighbours": "Share with 2 neighbours",
    "share_3plus_neighbours": "Share with 3+ neighbours",
    "t_index": "Interval",
    "served_reports": "Served reports",
    "peak_ues": "Most UEs in one interval",
    "median_throughput_mbps": "Median estimated throughput [Mbps]",
    "median_sinr_db": "Median SINR [dB]",
    "served_share": "Share of UE reports served on the band",
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
    directory: str | Path,
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
