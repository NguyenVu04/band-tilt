"""Bayesian Optimization over the absolute-tilt space — PROJECT.md section 20.

Built on Ax and BoTorch. Treats the problem as expensive black-box
optimization: propose a configuration, evaluate it, update the model, repeat.

Reads its search space from :mod:`src.optim.space` and scores through
:mod:`src.optim.objective`, so it and MARL are provably solving the same
problem (docs/adr/0006).
"""
