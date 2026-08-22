"""Matplotlib and seaborn configuration for report-ready figures.

Every notebook calls :func:`setup_plotting` once, in its Setup cell, right
after :func:`src.utils.seed.set_seed`. Figures then share one set of fonts,
colors and resolution across all four notebooks, so a plot pasted into
``reports/`` doesn't look like it came from a different document than the one
next to it.

Colors are a validated, colorblind-safe palette: fixed-order categorical hues,
a single-hue sequential ramp for magnitude, and a blue/red diverging ramp for
signed values. Never call ``sns.set_palette`` with something else mid-notebook
— a report where the same category gets different hues in different figures
is a report that misleads.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

#: Fixed-order categorical palette. Colorblind-safe by construction — do not
#: reorder or cycle past the eighth color; fold additional series into
#: "other" or facet instead.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

#: Single-hue blue ramp, light -> dark, for magnitude (heatmaps, ordered bars).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: Blue <-> red through a neutral gray midpoint, for signed values
#: (residuals, correlations, deltas). Center it on zero with
#: ``matplotlib.colors.TwoSlopeNorm(0)`` — otherwise the gray midpoint lands
#: on the data's midpoint instead of on zero.
DIVERGING = ["#0d366b", "#256abf", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#8a1f1e"]

# Chart chrome — off-white surface and muted ink, tuned for a printed page
# rather than a screen.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"


def setup_plotting(context: str = "notebook") -> None:
    """Configure matplotlib and seaborn for report-ready figures.

    Args:
        context: Seaborn scaling context. Use the default ``"notebook"``
            while exploring; switch to ``"paper"`` right before generating
            the final figures for ``reports/figures/`` — it shrinks fonts and
            line weights to the size a printed report actually needs.

    Example:
        >>> setup_plotting()
        >>> sns.barplot(data=df, x="segment", y=target)
    """
    sns.set_theme(
        context=context,
        style="ticks",
        palette=CATEGORICAL,
        rc={
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "axes.edgecolor": _AXIS,
            "axes.labelcolor": _INK,
            "axes.titlecolor": _INK,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": _GRIDLINE,
            "grid.linewidth": 0.6,
            "text.color": _INK,
            "xtick.color": _INK_SECONDARY,
            "ytick.color": _INK_SECONDARY,
            "legend.frameon": False,
        },
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CATEGORICAL)


def sequential_cmap() -> LinearSegmentedColormap:
    """Build the single-hue colormap for magnitude encodings.

    Returns:
        A light-to-dark blue colormap for heatmaps and ordered bars.
    """
    return LinearSegmentedColormap.from_list("report_sequential", SEQUENTIAL)


def diverging_cmap() -> LinearSegmentedColormap:
    """Build the blue/red colormap for signed encodings.

    Returns:
        A colormap for residuals, correlations and deltas. Pass
        ``norm=matplotlib.colors.TwoSlopeNorm(0)`` when plotting so the gray
        midpoint sits on zero rather than on the data's midpoint.
    """
    return LinearSegmentedColormap.from_list("report_diverging", DIVERGING)
