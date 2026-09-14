"""The four KPIs - the only definition of the objective in this project.

Every optimizer and every evaluation scores through these functions, so none of
them can be measuring a slightly different thing. Thresholds come from
``configs/kpi.yaml``; a literal ``-120`` at a call site is a bug even when it
happens to match.

Every function takes an RSRP array of shape ``[n_band, n_tx, n_rows, n_cols]``
in dBm, NaN where the ray tracer found no path - the array
:func:`src.simulation.radio.solve` writes. The served ratio also takes that
file's ``sinr_db``. The package imports neither ``src.simulation`` nor
Sionna-RT, which is what lets one implementation score a ray-traced map and a
hand-built fixture alike.

All four land in ``[0, 1]``. Three are grid shares and minimised;
:func:`~src.kpi.served.served_ratio` is a UE share and maximised. See
docs/adr/0001-four-kpis-and-weighted-score.md.

:mod:`src.kpi.capacity` sits beside them: the serving-cell rule the served ratio
counts by, and the PRB demand map, a diagnostic outside the objective.
"""

from src.kpi.hole import hole_rate
from src.kpi.overlap import overlap_rate
from src.kpi.served import served_ratio
from src.kpi.weak import weak_rate

__all__ = [
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
]
