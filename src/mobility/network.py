"""Load the SUMO road network, and resolve SUMO binaries reproducibly.

``sumolib``/``traci`` are not ordinary importable packages. A system SUMO
install exposes them only via ``$SUMO_HOME/tools`` on ``PYTHONPATH`` — not
recorded anywhere, not something ``uv.lock`` can pin. The ``eclipse-sumo``
wheel (the ``sumo`` extra) ships the same files under
``site-packages/sumo/tools/``, importable the same way once
``$SUMO_HOME/tools`` is on ``sys.path`` — ``import sumo`` sets ``SUMO_HOME``
to the wheel's own install root as a side effect, which is what makes the
wheel self-contained without a system install or an ambient environment
variable.

:func:`sumo_binary` resolves through ``sumolib.checkBinary``, which prefers
``SUMO_HOME`` (a system install, if one is configured) and falls back to
whatever ``sumolib`` itself was imported from. Having both a system SUMO and
the wheel installed at different versions is a real hazard — the network could
be built with one and the trajectories generated with another without either
process complaining — so it also checks that the *binary* resolved and the
*module* imported come from the same installation root, and refuses to
proceed silently when they do not.

``sumolib`` is imported lazily, on first use
-----------------------------------------------
Importing this module must not require SUMO to be resolvable — the same rule
``src.radio.scene.load_scene`` follows for Sionna-RT, and for the same reason:
``src.mobility.demand``, ``.checks`` and ``.simulate`` import this module for
functions (``run_command``, pure config arithmetic) that have nothing to do
with SUMO, and forcing every one of those imports to fail without the
``sumo`` extra would make even a test suite run with ``task sync`` alone
uncollectable. Only :func:`sumo_binary` and :func:`load_net` actually touch
``sumolib``, and both call :func:`_sumolib` to get it, which is where the
bootstrap and the potential ``ImportError`` happen.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

#: Set on first successful _sumolib() call; avoids re-running the path
#: bootstrap and re-importing on every call in the same process.
_sumolib_module = None


def _bootstrap_sumo_tools() -> None:
    """Put ``$SUMO_HOME/tools`` on ``sys.path`` so ``sumolib``/``traci`` import.

    Prefers an already-set ``SUMO_HOME`` (a system install the user
    configured deliberately) over the ``eclipse-sumo`` wheel, so a developer
    with a system SUMO keeps using it. Falls back to ``import sumo`` — the
    wheel installed by the ``sumo`` extra — only when ``SUMO_HOME`` is unset,
    which is what makes a Colab runtime or a machine without a system SUMO
    work with nothing but ``uv sync --extra sumo``.

    Raises:
        ImportError: When ``SUMO_HOME`` is unset and the ``eclipse-sumo``
            wheel is not installed either.
    """
    if "SUMO_HOME" not in os.environ:
        try:
            import sumo  # noqa: F401  (side effect: sets os.environ["SUMO_HOME"])
        except ImportError as exc:
            raise ImportError(
                "SUMO_HOME is not set and the `eclipse-sumo` wheel is not "
                "installed. Install it with `uv sync --extra sumo`, or install "
                "a system SUMO and export SUMO_HOME to its installation root."
            ) from exc
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def _sumolib() -> Any:
    """Return the ``sumolib`` module, importing and bootstrapping it on first use.

    Raises:
        ImportError: When SUMO cannot be resolved — see
            :func:`_bootstrap_sumo_tools`.
    """
    global _sumolib_module
    if _sumolib_module is None:
        _bootstrap_sumo_tools()
        import sumolib

        _sumolib_module = sumolib
    return _sumolib_module


def sumo_binary(name: str) -> str:
    """Resolve a SUMO binary, checking it matches the imported ``sumolib``.

    Args:
        name: A SUMO command name, e.g. ``"sumo"``, ``"netconvert"``,
            ``"duarouter"``.

    Returns:
        The resolved binary's absolute path.

    Raises:
        RuntimeError: When the resolved binary's installation root does not
            match the root ``sumolib`` was imported from — a version-skew
            hazard between a system SUMO and the ``eclipse-sumo`` wheel.

    Notes:
        ``sumolib.checkBinary`` itself looks for a ``<NAME>_BINARY``
        environment variable, then ``$SUMO_HOME/bin``, then a binary on
        ``PATH``. It can therefore resolve a *different* installation than
        the one ``import sumolib`` came from — which is exactly the skew this
        function exists to catch, not silently propagate into a manifest.

    Example:
        >>> sumo_binary("sumo")
        'C:/.../bin/sumo.exe'
    """
    sumolib = _sumolib()
    binary = sumolib.checkBinary(name)
    # sumolib.__file__ is <SUMO_HOME>/tools/sumolib/__init__.py; the binary is
    # <SUMO_HOME>/bin/<name>. Two parents from each side lands on SUMO_HOME.
    sumolib_home = Path(sumolib.__file__).resolve().parents[2]
    binary_home = Path(binary).resolve().parents[1]
    if sumolib_home != binary_home:
        raise RuntimeError(
            f"Resolved '{name}' at {binary} (under {binary_home}), but "
            f"`import sumolib` came from {sumolib_home} — a different SUMO "
            "installation. Reinstall from one source only: either "
            "`uv sync --extra sumo` with SUMO_HOME unset, or a system SUMO "
            "with SUMO_HOME pointed at it."
        )
    return binary


def sumo_version(binary: str) -> str:
    """Return the first line of ``<binary> --version``.

    Args:
        binary: A resolved binary path, from :func:`sumo_binary`.

    Returns:
        The version banner's first line, e.g. ``"Eclipse SUMO sumo 1.27.1"``.

    Example:
        >>> sumo_version(sumo_binary("sumo"))
        'Eclipse SUMO sumo 1.27.1'
    """
    result = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0].strip()


def load_net(cfg: DictConfig) -> Any:
    """Load the SUMO road network named by ``cfg.mobility.network.net_file``.

    Args:
        cfg: Composed config; uses ``cfg.mobility.network.net_file``.

    Returns:
        A ``sumolib.net.Net``, in SUMO network-local coordinates (not yet
        translated to the Sionna-RT scene frame — see
        :mod:`src.mobility.frame`).

    Notes:
        Parses the whole file, edges and lanes included, so it is the
        expensive call in this module (on the order of a second or two for
        the delivered network) — everywhere else in this package that only
        needs the coordinate frame or the extent reads the small
        ``<location>`` header directly instead (see
        :func:`src.data.scenario.read_net_location`).

        Loaded with ``withInternal=True``. SUMO's own FCD output reports a
        vehicle's lane as a junction-internal lane (``":<junction>_<index>"``)
        while it crosses an intersection, and ``src.mobility.simulate.tidy``
        carries that through as ``edge_id`` unchanged — without
        ``withInternal``, ``net.getEdge()`` raises ``KeyError`` for every one
        of those rows, which is not a genuine speed-limit violation and would
        otherwise be indistinguishable from one in
        :func:`src.mobility.checks.speed_within_limits`.

    Example:
        >>> net = load_net(cfg)
    """
    return _sumolib().net.readNet(str(cfg.mobility.network.net_file), withInternal=True)


def network_summary(net: Any) -> dict[str, Any]:
    """Summarise a loaded network: edge count, total lane length, bbox.

    Args:
        net: A network from :func:`load_net`.

    Returns:
        A mapping with ``edge_count`` (int, ordinary road edges only —
        see Notes), ``internal_edge_count`` (int, junction-internal
        connectors), ``total_lane_length_m`` (float, ordinary edges' lanes
        only), and ``bbox`` — the ``(xmin, ymin, xmax, ymax)`` convex
        boundary in SUMO network-local coordinates (unaffected by the
        internal/ordinary distinction; ``net.getBoundary()`` already covers
        the whole network).

    Example:
        >>> network_summary(load_net(cfg))["edge_count"]
        20799

    Notes:
        :func:`load_net` loads the network with ``withInternal=True``, so
        ``net.getEdges()`` includes the short connector edges SUMO generates
        inside every junction — on the delivered network there are roughly
        four of these for every one ordinary road edge, and their length is
        typically metres, not the tens or hundreds of metres an ordinary edge
        spans. Reporting them together with ordinary edges would inflate
        both counts in a way that misrepresents the size of the road network
        to a reader of a scenario report, so ``edge_count`` and
        ``total_lane_length_m`` here are ordinary (``not edge.isSpecial()``)
        edges only. This is a summary-only choice — the check that actually
        needs the internal edges resolvable,
        :func:`src.mobility.checks.speed_within_limits`, looks them up
        directly by id and does not go through this function.
    """
    edges = net.getEdges()
    ordinary = [edge for edge in edges if not edge.isSpecial()]
    total_lane_length_m = sum(lane.getLength() for edge in ordinary for lane in edge.getLanes())
    xmin, ymin, xmax, ymax = net.getBoundary()
    return {
        "edge_count": len(ordinary),
        "internal_edge_count": len(edges) - len(ordinary),
        "total_lane_length_m": total_lane_length_m,
        "bbox": (xmin, ymin, xmax, ymax),
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    """Echo and run a subprocess, raising on a nonzero exit.

    Args:
        command: The argv to run — a resolved binary or ``sys.executable``
            followed by a ``.py`` tool script, plus its arguments.

    Returns:
        The completed process (stdout/stderr are not captured; they pass
        through to this process's own, matching ``scripts/build_scene.py``).

    Raises:
        subprocess.CalledProcessError: When the command exits nonzero.

    Notes:
        Shared by :mod:`src.mobility.demand` (``randomTrips.py``,
        ``duarouter``) and :mod:`src.mobility.simulate` (``sumo``) so the
        three subprocess invocations that build one scenario are echoed and
        checked identically, and so the exact argv used is available to be
        recorded verbatim in the scenario manifest by ``src.mobility.persist``.

    Example:
        >>> run_command(["sumo", "--version"])
    """
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True)
