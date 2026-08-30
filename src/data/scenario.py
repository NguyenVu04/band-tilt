"""Scenario identity — the single producer of ``scenario_id``.

A scenario is one environment (buildings, dimensions, materials) together
with one UE mobility realisation. The scenario-level split
(:mod:`src.data.split`), the perturbed scenarios, and notebook 06's held-out
validation all index by ``scenario_id``. Before this module, nothing in the
pipeline produced one — this is that producer, and the only one.

Deterministic, from the definition, not the artifacts
-------------------------------------------------------
``scenario_id`` is a hash of the scenario's *definition* — the mobility config
that would regenerate it, plus the fingerprint of the environment it was
generated against — not a hash of the trajectory file bytes. Two runs with the
same definition and the same seed must produce the same id even before either
one has run, which is what lets ``src.mobility.persist`` decide where to write
without generating the trajectories first, and what lets a manifest be
compared against a fresh run's definition without re-reading the parquet.

What is excluded, and why
--------------------------
``cfg.mobility.output.*`` is excluded: a path is not part of what a scenario
*is,* and including it would mean writing the same scenario twice to two
different directories produces two different identities. The tilt
configuration is excluded too — ``src.data.split`` is explicit that many
tilt configurations live inside one
scenario; a scenario is the environment and the mobility realisation, not a
network configuration.

``version`` exists so that changing what goes into the hash is a visible,
deliberate act — bump it and every previously stored scenario re-identifies
under a new id rather than silently colliding or silently diverging.

Where this sits
----------------
Below ``src.radio`` and ``src.mobility``, not inside either of them:
``src.radio`` will need scenario identity for the environment
perturbations, and may not import ``src.mobility``; ``src.mobility`` produces
trajectories and calls this module rather than defining identity itself, so
the same id is available to a future radio-side perturbation without a second
implementation. This module imports nothing from either package except
``src.radio.scene.scene_metadata`` — one function, at the same level as the
``scene_bounds`` exception ``src/data/__init__.py`` already documents.

Nothing here reads a trajectory or a radio map. It reads two small XML
headers and one Hydra config.
"""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.radio import scene

#: Bumped only when the scenario_id payload definition itself changes — never
#: for an ordinary config value. See the module docstring.
PAYLOAD_VERSION = 1

#: Prefix on every generated id, so a scenario_id is recognisable in a path,
#: a log line, or a partition name without a length check.
_ID_PREFIX = "scn_"

#: Hex digits in the id after the prefix (blake2b digest_size=8 -> 16 hex chars).
_DIGEST_SIZE = 8


def read_net_location(net_file: str | Path) -> dict[str, str]:
    """Read the ``<location .../>`` element's attributes from a SUMO net file.

    Args:
        net_file: Path to a ``.net.xml`` file written by ``netconvert``.

    Returns:
        The element's attributes verbatim, as strings — ``netOffset``,
        ``convBoundary``, ``origBoundary`` and ``projParameter``.

    Raises:
        FileNotFoundError: When ``net_file`` does not exist. ``data/`` is
            DVC-tracked and not part of a fresh clone — run ``task dvc:pull``,
            or ``task scene:build`` if the network itself has not been built.
        ValueError: When the file has no ``<location>`` element.

    Notes:
        Public rather than module-private because
        ``src.mobility.frame.derive_frame`` reads the same element — the
        ``projParameter`` proj4 string is what it projects the scene centre
        through, and ``netOffset`` is the other half of the translation. One
        parser, two consumers.
    """
    path = Path(net_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run `task dvc:pull`, or `task scene:build` "
            "if the SUMO network has not been built yet."
        )
    # A <net> file's <location> is one of the first elements; stop as soon as
    # it is found rather than parsing the (potentially tens-of-megabytes) edge
    # and lane data that follows.
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "location":
            return dict(elem.attrib)
        elem.clear()
    raise ValueError(f"{path}: no <location> element found")


def scene_fingerprint(cfg: DictConfig) -> dict[str, Any]:
    """Fingerprint the environment a scenario is generated against.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_file`` (via
            :func:`src.radio.scene.scene_metadata`) and
            ``cfg.mobility.network.net_file``.

    Returns:
        A mapping with ``scene_file``, ``scenegen`` (the scene XML's
        ``scenegen_*`` metadata), ``net_file`` and ``net_location`` (the SUMO
        net's ``<location>`` attributes).

    Raises:
        FileNotFoundError: When the scene XML or the SUMO net file is missing.

    Notes:
        A rebuilt network, a re-projected scene, or a different study area
        each change one of these four values, and all three would otherwise
        produce trajectories that silently mean something different while
        keeping the same identity. Hashing this metadata rather than the mesh
        or network file bytes is deliberate: the mesh directory holds 4,737
        files and the net is tens of megabytes, and a content hash would cost
        seconds on every call.

    Example:
        >>> fp = scene_fingerprint(cfg)
        >>> fp["scenegen"]["scenegen_UTM_zone"]
        'EPSG:32648'
    """
    return {
        "scene_file": str(cfg.data.scene_file),
        "scenegen": scene.scene_metadata(cfg),
        "net_file": str(cfg.mobility.network.net_file),
        "net_location": read_net_location(cfg.mobility.network.net_file),
    }


def scenario_id(cfg: DictConfig, fingerprint: dict[str, Any]) -> str:
    """Compute the deterministic id of the scenario ``cfg`` and ``fingerprint`` define.

    Args:
        cfg: Composed config; uses ``cfg.mobility`` (every key except
            ``output``, which is a destination, not part of the definition).
        fingerprint: The environment fingerprint from :func:`scene_fingerprint`.

    Returns:
        ``"scn_" + <16 hex chars>`` — a `blake2b` digest of the scenario
        definition. See the module docstring for what is hashed and why.

    Raises:
        AttributeError: When ``cfg`` has no ``mobility`` section.

    Notes:
        The payload is serialised with ``sort_keys=True`` so that dict
        iteration order — not guaranteed stable across a config reload — can
        never change the id. Two composed configs that differ only in key
        order, or only in the output directory, must produce the same id;
        two that differ in UE count, seed, or the network location must not.

    Example:
        >>> scenario_id(cfg, scene_fingerprint(cfg))
        'scn_a1b2c3d4e5f6a7b8'
    """
    mobility = OmegaConf.to_container(cfg.mobility, resolve=True)
    mobility.pop("output", None)
    payload = {
        "version": PAYLOAD_VERSION,
        "mobility": mobility,
        "seed": int(cfg.mobility.seed),
        "scene": fingerprint,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.blake2b(blob, digest_size=_DIGEST_SIZE).hexdigest()
    return f"{_ID_PREFIX}{digest}"


def _git_commit() -> str | None:
    """Best-effort current commit hash, or ``None`` outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def build_manifest(cfg: DictConfig, sid: str, **sections: Any) -> dict[str, Any]:
    """Assemble the sidecar manifest written beside a scenario's trajectories.

    Args:
        cfg: Composed config.
        sid: The scenario's id, from :func:`scenario_id`.
        **sections: Named blocks to merge in after the header — for example
            ``sumo=...``, ``seeds=...``, ``mobility=...``, ``frame=...``,
            ``network=...``, ``clipping=...``, ``counts=...``. Each caller
            names its own sections; this function does not know their shape.

    Returns:
        A JSON-serialisable mapping: ``scenario_id``, ``version``,
        ``created_utc``, ``git_commit`` (``None`` outside a Git checkout),
        then every section passed in, in the order given.

    Notes:
        ``created_utc`` and ``git_commit`` are provenance for a human
        reading the manifest later, not part of the scenario's identity —
        they are deliberately absent from the :func:`scenario_id` payload,
        because a scenario regenerated a year later on a different commit is
        still the same scenario.

    Example:
        >>> manifest = build_manifest(cfg, "scn_a1b2c3d4e5f6a7b8", counts={"n_rows": 720})
        >>> manifest["scenario_id"]
        'scn_a1b2c3d4e5f6a7b8'
    """
    manifest: dict[str, Any] = {
        "scenario_id": sid,
        "version": PAYLOAD_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
    }
    manifest.update(sections)
    return manifest
