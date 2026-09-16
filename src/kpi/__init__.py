"""The KPIs - the only definition of the objective in this project.

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

The four rates land in ``[0, 1]``: three are grid shares and minimised,
:func:`~src.kpi.served.served_ratio` is a UE share and maximised.
:func:`~src.kpi.quality.edge_rsrp_dbm` is in dBm and reported only.

Each of the three objective KPIs also has a ``*_desirability`` twin that
applies the same threshold **softly, per tile** (:mod:`src.kpi.soft`) before
averaging. Those twins are what the objective reads; the rates above are
reported and read by the hard score. All of the softening happens here, at
measure time, because a per-tile transform cannot be recovered from a share.
Which of them each score reads is ``WEIGHTED_NAMES`` and ``OBJECTIVE_NAMES``
in :mod:`src.optim.objective`;
see docs/adr/0001-four-kpis-and-weighted-score.md and
docs/adr/0004-soft-threshold-desirability-objective.md.

:mod:`src.kpi.capacity` sits beside them: the serving-cell rule the served ratio
counts by, and the PRB demand map, a diagnostic outside the objective.
"""

from src.kpi.hole import hole_desirability, hole_rate
from src.kpi.overlap import overlap_desirability, overlap_rate
from src.kpi.quality import edge_rsrp_dbm
from src.kpi.served import served_desirability, served_ratio
from src.kpi.weak import weak_rate

__all__ = [
    "edge_rsrp_dbm",
    "hole_desirability",
    "hole_rate",
    "overlap_desirability",
    "overlap_rate",
    "served_desirability",
    "served_ratio",
    "weak_rate",
]
