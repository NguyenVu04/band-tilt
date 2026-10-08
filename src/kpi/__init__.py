"""The KPIs - the only definition of each reported measure in this project.

Every optimizer and every evaluation measures through these functions.
Thresholds come from ``configs/kpi.yaml``; a literal ``-120`` at a call site is a
bug even when it happens to match.

The map measures start from an RSRP array of shape
``[n_band, n_tx, n_rows, n_cols]`` in dBm, NaN where the ray tracer found no
path - the array :func:`src.simulation.radio.solve` writes - reduced once by
:func:`src.kpi.capacity.max_rsrp` and :func:`src.kpi.overlap.overlap_neighbors`
and then shared (:func:`src.optim.objective.map_kpis`). Slicing the band axis to
one band is how every one of them is read per band. The package imports neither
``src.simulation`` nor Sionna-RT, so a ray-traced map and a hand-built fixture
are scored by the same code.

:mod:`src.kpi.served`'s throughput statistics take a
:func:`src.kpi.capacity.serve_intervals` assignment instead, because serving
the UE table is the expensive half of a measurement.

None of them is a search objective; those are :mod:`src.optim.objective`.
"""

from src.kpi.served import throughput_mean_mbps, throughput_percentile_mbps

__all__ = ["throughput_mean_mbps", "throughput_percentile_mbps"]
