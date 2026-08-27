"""Scene construction, radio-map generation, and the geometry that feeds them.

Modules
-------
- ``geometry``  angle conventions and the tilt definition
- ``cell_band`` the cell-band table: the atomic unit of the decision variable
- ``scene``     build the Sionna-RT scene from the 3D map and materials
- ``radiomap``  turn an absolute tilt configuration into an RSRP array
- ``sampling``  draw valid tilt configurations for the surrogate dataset

This package owns the only dependency on Sionna-RT. Everything downstream
consumes plain NumPy arrays, which is what lets the KPI code score a surrogate
prediction and a ray-traced map with the same functions, and lets the tests run
without a simulator.

Nothing here may import ``src.kpi``, ``src.surrogate`` or ``src.optim``.
"""
