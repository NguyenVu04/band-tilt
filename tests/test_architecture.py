"""The scenario generator is an independent workflow: nothing outside it may reach into it.

Downstream stages read the files ``simulation.input`` names, so real data can
replace the synthetic generator without a code change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONSUMERS = sorted(
    path
    for path in (_ROOT / "src").rglob("*.py")
    if (_ROOT / "src" / "scenario") not in path.parents
)


def _violations(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.scenario"):
            found.append(f"line {node.lineno}: from {node.module} import")
        elif isinstance(node, ast.ImportFrom) and node.module == "src":
            names = [alias.name for alias in node.names]
            found += (
                [f"line {node.lineno}: from src import scenario"] if "scenario" in names else []
            )
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names if alias.name.startswith("src.scenario")]
            found += [f"line {node.lineno}: import {name}" for name in names]
        elif isinstance(node, ast.Attribute) and node.attr == "scenario":
            found.append(f"line {node.lineno}: reads the scenario config")
    return found


@pytest.mark.parametrize("path", _CONSUMERS, ids=lambda path: str(path.relative_to(_ROOT)))
def test_no_module_outside_the_generator_reaches_into_it(path: Path) -> None:
    """No import of ``src.scenario`` and no read of ``cfg.scenario`` outside ``src/scenario``."""
    assert _violations(path) == []
