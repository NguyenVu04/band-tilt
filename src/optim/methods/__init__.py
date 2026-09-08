"""The search methods, one folder each, behind one interface.

Every method scores its proposals with the same five functions in
:mod:`src.kpi` and returns the same :class:`~src.optim.history.History`, so a
comparison between methods is a comparison of search strategies and nothing
else. Adding one is a folder and a registry entry, not an edit to a dispatch
chain.

``mobo`` and ``random`` each carry their own Ax ask-tell loop. They are kept
matched on budget, on thresholds and on what they record; ADR 0002 is why the
generation strategy must remain the only difference between them.
"""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig

from src.optim.evaluator import ObjectiveEvaluator
from src.optim.history import History
from src.optim.methods import mobo, random, rule
from src.optim.methods.base import SearchMethod

SEARCHES: Mapping[str, SearchMethod] = {
    "mobo": mobo.search,
    "random": random.search,
    "rule": rule.search,
}


def run_search(evaluator: ObjectiveEvaluator, cfg: DictConfig) -> History:
    """Run the method ``cfg.optim.method.name`` names.

    The name and the parameters arrive together: ``optim/method`` is a config
    group, so selecting a method selects its block. There is no way to ask for
    one method while holding another's settings.

    Raises:
        KeyError: When the method is not registered, naming those that are.
    """
    method = str(cfg.optim.method.name)
    if method not in SEARCHES:
        raise KeyError(
            f"unknown optim/method {method!r}; registered methods are {sorted(SEARCHES)}"
        )
    return SEARCHES[method](evaluator, cfg)
