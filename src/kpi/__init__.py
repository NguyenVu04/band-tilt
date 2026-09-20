"""The KPIs - the only definition of each reported measure in this project.

Every optimizer and every evaluation measures through these functions, so none
of them can be measuring a slightly different thing. Thresholds come from
``configs/kpi.yaml``; a literal ``-120`` at a call site is a bug even when it
happens to match.

Most functions take an RSRP array of shape ``[n_band, n_tx, n_rows, n_cols]``
in dBm, NaN where the ray tracer found no path - the array
:func:`src.simulation.radio.solve` writes. The percentile measures also take
that file's ``sinr_db``. Slicing the band axis to one band is how every one of
them is read per band: the definition does not change, only what it is given.
The package imports neither ``src.simulation`` nor Sionna-RT, which is what lets
one implementation score a ray-traced map and a hand-built fixture alike.

The served rate and the two load measures take a
:func:`src.kpi.capacity.serve_intervals` assignment instead, because serving the
UE table is the expensive half of a measurement and all three read the same
output.

Shares land in ``[0, 1]``: the three grid rates are minimised,
:func:`~src.kpi.served.served_rate` is a UE share and maximised. The percentiles
are in dBm and dB, :func:`~src.kpi.overlap.overlap_neighbor_mean` is a count and
:func:`~src.kpi.load.load_imbalance` a coefficient of variation. None of them is
the search objective; that is :mod:`src.optim.objective`,
docs/adr/0009-effective-coverage-objective.md, built on
:func:`~src.kpi.overlap.serving_multiplicity`, which this package exports beside
the KPIs because it is the same co-band count read on one band.

:mod:`src.kpi.capacity` sits beside them: the serving-cell rule the served rate
counts by, and the PRB demand map.
"""

from src.kpi.hole import hole_rate
from src.kpi.load import load_imbalance, prb_by_cell_interval, prb_utilisation_max, utilisation
from src.kpi.overlap import overlap_neighbor_mean, overlap_rate, serving_multiplicity
from src.kpi.quality import rsrp_percentile_dbm, sinr_percentile_db
from src.kpi.served import served_rate
from src.kpi.weak import weak_rate

__all__ = [
    "hole_rate",
    "load_imbalance",
    "overlap_neighbor_mean",
    "overlap_rate",
    "prb_by_cell_interval",
    "prb_utilisation_max",
    "rsrp_percentile_dbm",
    "served_rate",
    "serving_multiplicity",
    "sinr_percentile_db",
    "utilisation",
    "weak_rate",
]
