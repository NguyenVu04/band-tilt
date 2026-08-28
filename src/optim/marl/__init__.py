"""Multi-Agent RL over the absolute-tilt space — PROJECT.md section 14.

Built on TorchRL. Models tilt coordination as a multi-agent control problem:
one agent per cell (or per site), each emitting the absolute tilts of its own
bands, trained against a shared reward derived from the five KPIs.

Modules
-------
- ``env``    the TorchRL environment wrapping the tilt problem
- ``reward`` the scalar reward, and the priority it has to preserve
- ``train``  the training loop

Reads its action bounds from :mod:`src.optim.space` and scores through
:mod:`src.optim.objective`, so it and BO are provably solving the same problem
(PROJECT.md sections 13 and 14).
"""
