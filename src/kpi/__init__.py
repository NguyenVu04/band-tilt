"""The KPIs - the only definition of each reported measure in this project.

Every optimizer and every evaluation measures through these functions, so none
of them can be measuring a slightly different thing. Thresholds come from
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
:func:`~src.kpi.quality.edge_rsrp_dbm` is in dBm. None of them is the search
objective; that is :mod:`src.optim.objective`, docs/adr/0006-radio-load-cvar-objective.md.

:mod:`src.kpi.capacity` sits beside them: the serving-cell rule the served ratio
and the objective's load term count by, and the PRB demand map.
"""

from src.kpi.hole import hole_rate
from src.kpi.overlap import overlap_rate
from src.kpi.quality import edge_rsrp_dbm
from src.kpi.served import served_ratio
from src.kpi.weak import weak_rate

__all__ = [
    "edge_rsrp_dbm",
    "hole_rate",
    "overlap_rate",
    "served_ratio",
    "weak_rate",
]
