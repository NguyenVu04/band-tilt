"""Optimization over the absolute-tilt space: Bayesian Optimization and MARL.

Modules
-------
- ``space``     the search space Theta, shared by both methods
- ``objective`` turn a KPI vector into whatever the optimizer consumes
- ``bo``        Ax / BoTorch
- ``marl``      TorchRL

The comparison is the point
---------------------------
PROJECT.md section 25 requires BO and MARL to be evaluated under identical
inputs, action space, tilt constraints, KPI definitions, surrogate and
validation procedure. Anything else measures the two implementations rather than
the two methods.

``space`` and ``objective`` are what enforce that. Both optimizers derive their
bounds from :mod:`src.optim.space` and score through :mod:`src.kpi`; neither
reads ``configs/radio.yaml`` or a KPI threshold directly. A second definition of
the search space anywhere in this package silently invalidates the comparison
the project exists to make.

What legitimately differs
-------------------------
How each method searches: the acquisition function, the policy architecture, the
number of evaluations, the wall-clock cost. Those are the results
(PROJECT.md sections 25.1 to 25.4), and they belong in
``configs/optim/bo.yaml`` and ``configs/optim/marl.yaml``.

See docs/adr/0006.
"""
