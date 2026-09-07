"""The five KPIs - the only definition of the objective in this project.

TuRBO, MARL, the surrogate's labels and the final Sionna-RT validation all score
through these functions, so none of them can be measuring a slightly different
thing. Thresholds come from ``configs/kpi.yaml``; a literal ``-120`` at a call
site is a bug even when it happens to match.

Every function takes an RSRP array of shape ``[n_band, n_tx, n_rows, n_cols]``
in dBm, NaN where the ray tracer found no path - the array
:func:`src.simulation.radio.solve` writes and the surrogate will predict. The
package imports neither ``src.simulation`` nor Sionna-RT, which is what lets one
implementation score a ray-traced map, a surrogate prediction and a hand-built
fixture.

Four KPIs are rates in ``[0, 1]``;
:func:`~src.kpi.mean_overlap_neighbors.mean_overlap_neighbors` is a count. All
five are minimised except :func:`~src.kpi.bps.band_priority_score`, which is
maximised. Priority order is hole, overlap, mean overlap neighbours, band
priority, weak - see
docs/adr/0001-five-kpis-under-lexicographic-priority.md.
"""

from src.kpi.bps import band_priority_score
from src.kpi.hole import hole_rate
from src.kpi.mean_overlap_neighbors import mean_overlap_neighbors
from src.kpi.overlap import overlap_rate
from src.kpi.weak import weak_rate

__all__ = [
    "band_priority_score",
    "hole_rate",
    "mean_overlap_neighbors",
    "overlap_rate",
    "weak_rate",
]
