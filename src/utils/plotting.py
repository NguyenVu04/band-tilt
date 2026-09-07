"""Figure defaults, so every notebook's plots look alike."""

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
