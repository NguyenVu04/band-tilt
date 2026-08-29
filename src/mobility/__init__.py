"""SUMO UE mobility — pipeline Phase 2.

Modules
-------
- ``frame``     the SUMO-network to Sionna-scene coordinate transform
- ``network``   load the road network and report what is in it
- ``demand``    UE population, arrival process, origin/destination pairs, routes
- ``simulate``  run SUMO and collect one tidy trajectory frame
- ``checks``    speed, step distance and arrival-profile invariants
- ``persist``   write trajectories and their manifest under ``data/interim/``

The mobility model is an input to the objective, not scenery
--------------------------------------------------------------
Two of the five KPIs are weighted by where UEs actually are. The UE-weighted
Band Priority Score (:mod:`src.kpi.band_priority`) asks whether *users* are on
the right frequency layer, not whether *area* is — so a mobility model that
puts every UE on one arterial road produces a score about that road. The
perturbed scenarios vary exactly these parameters, and notebook 06 has to report whether the
optimized configuration survived it.

The frame transform is the silent failure
-------------------------------------------
SUMO and Sionna-RT describe the same place in two different metric frames. An
offset error does not raise: it produces trajectories that look fine in
isolation and sit in the wrong streets, and every downstream number is then
computed for a network that does not exist. :mod:`src.mobility.frame` derives
the transform at run time from the two files themselves and is the only place
in the project that may do so.

Where this package sits
------------------------
Below the KPIs and everything that consumes them, beside ``src.data`` and
``src.radio``. It may import ``src.utils``, ``src.config``, ``src.data`` and
``src.radio.scene`` — the last only for :func:`src.radio.scene.scene_bounds`
and :func:`src.radio.scene.scene_metadata`, which own the scene extent and
metadata this package clips and derives the frame against. Importing Sionna-RT
here is a bug: nothing in mobility needs a propagation simulator, and the
``rt`` extra must not become a prerequisite for generating trajectories.

Nothing here may import ``src.kpi``, ``src.surrogate``, ``src.optim`` or
``src.evaluation``.
"""
