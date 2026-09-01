"""Locate and run the installed SUMO toolchain.

Every subprocess this package launches goes through here, and nothing else in
it touches ``subprocess``, ``SUMO_HOME`` or ``sumolib``. A SUMO installation is
resolved once, the same way, whether it came from the ambient environment or
from the ``eclipse-sumo`` wheel — so a missing installation fails with one
message rather than differently at each call site.

Named ``toolchain`` rather than ``sumo``: the ``eclipse-sumo`` wheel already
owns the top-level module name ``sumo``, which :func:`home` imports.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def home() -> Path:
    """Return ``SUMO_HOME`` when it names a directory, else the eclipse-sumo root.

    The installed ``eclipse-sumo`` wheel ships the same ``bin/`` and ``tools/``
    trees; a ``FileNotFoundError`` means neither is available.
    """
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        import sumo
    except ImportError as exc:
        raise FileNotFoundError(
            "SUMO_HOME is not set and the eclipse-sumo package is not installed. "
            "Install it with `uv sync --extra sumo`, or set SUMO_HOME."
        ) from exc
    return Path(sumo.SUMO_HOME)


def binary(name: str) -> Path:
    """Resolve a SUMO command-line binary such as ``netconvert``.

    Goes through ``sumolib.checkBinary``, which honours ``SUMO_HOME`` when it is
    set and otherwise falls back to the binaries shipped in the ``eclipse-sumo``
    wheel.

    Raises:
        FileNotFoundError: When ``sumolib`` is not installed, or when the
            resolved path does not exist.
    """
    try:
        import sumolib
    except ImportError as exc:
        raise FileNotFoundError(
            f"sumolib is required to locate {name}. Install it with `uv sync --extra sumo`."
        ) from exc
    found = Path(sumolib.checkBinary(name))
    if not found.is_file():
        raise FileNotFoundError(
            f"{name} was resolved to {found}, which does not exist. Check the SUMO installation."
        )
    return found


def tool(name: str) -> Path:
    """Resolve a script under ``$SUMO_HOME/tools``, such as ``randomTrips.py``.

    These are Python scripts, not binaries: run them with ``sys.executable``.

    Raises:
        FileNotFoundError: When the script is not present in the installation.
    """
    script = home() / "tools" / name
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found; check the SUMO installation.")
    return script


def run(command: Sequence[str]) -> None:
    """Echo an argument vector and run it without a shell, raising if it fails.

    The ``RuntimeError`` carries the captured stderr, which is where netconvert,
    ``osmGet.py`` and ``randomTrips.py`` report what they objected to.
    """
    print("+ " + " ".join(command), flush=True)
    # SUMO writes UTF-8; decoding with the Windows default codec raises on the
    # non-ASCII street names netconvert echoes back in its warnings.
    result = subprocess.run(
        command, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(command[0]).name} exited {result.returncode}\n{result.stderr.strip()}"
        )
