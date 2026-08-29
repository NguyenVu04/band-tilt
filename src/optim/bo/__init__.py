"""TuRBO over the absolute-tilt space.

Trust-region Bayesian Optimization, built on Ax and BoTorch. Treats the problem
as expensive black-box optimization: propose a configuration, evaluate it,
update the model, repeat — but fits the model and maximises the acquisition
inside a *trust region* ``T_t`` rather than over the whole feasible box.

The trust region is what makes this the named method rather than plain BO. One
dimension per cell-band puts the problem in the tens of dimensions, where a
single global GP fits poorly and the acquisition flattens; restricting to a
local region keeps the model meaningful, and the region expands on success and
contracts on failure. ``T_t`` is always a subset of ``X`` — it narrows the
search, never the constraint.

Reads its search space from :mod:`src.optim.space` and scores through
:mod:`src.optim.objective`, so it and MARL are provably solving the same
problem.
"""
