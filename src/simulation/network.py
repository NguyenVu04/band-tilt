"""Build the SUMO road network for the study area, in the scene's frame.

Three steps, in order: fetch the OpenStreetMap extract, convert it to
netconvert's plain XML, then assemble that plain XML into the ``.net.xml`` the
routing and simulation tools require.

The network and the scene describe the same place, in one coordinate frame —
see :mod:`src.simulation.frame` for how the two are tied together, and
:func:`build_net` for why the assemble step must not re-apply the offset.

Every path and parameter comes from ``configs/simulation.yaml``, through
:class:`NetworkPaths` and :class:`NetworkSpec`; no other function in this module
reads a config key.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from src.simulation import toolchain
from src.simulation.frame import SceneFrame


@dataclass(frozen=True)
class NetworkPaths:
    """Where the network's inputs and outputs live.

    Attributes:
        osm: The OpenStreetMap extract, downloaded or reused.
        plain_prefix: netconvert's ``--plain-output-prefix``; the five plain
            files are this plus ``.nod.xml``, ``.edg.xml`` and so on.
        net: The compiled network :func:`build_net` writes.
    """

    osm: Path
    plain_prefix: Path
    net: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> NetworkPaths:
        """Read ``simulation.osm.file``, ``simulation.network`` and ``simulation.prefix``."""
        return cls(
            osm=Path(cfg.simulation.osm.file),
            plain_prefix=Path(cfg.simulation.network.out_dir) / str(cfg.simulation.prefix),
            net=Path(cfg.simulation.network.net_file),
        )

    def _plain(self, kind: str) -> Path:
        return self.plain_prefix.with_name(f"{self.plain_prefix.name}.{kind}.xml")

    @property
    def nod(self) -> Path:
        """The plain node file."""
        return self._plain("nod")

    @property
    def edg(self) -> Path:
        """The plain edge file."""
        return self._plain("edg")

    @property
    def con(self) -> Path:
        """The plain connection file."""
        return self._plain("con")

    @property
    def tll(self) -> Path:
        """The plain traffic-light-logic file."""
        return self._plain("tll")

    @property
    def typ(self) -> Path:
        """The plain edge-type file."""
        return self._plain("typ")


@dataclass(frozen=True)
class NetworkSpec:
    """What to build: the study area, and how netconvert should treat it.

    Attributes:
        prefix: Basename ``osmGet.py`` composes its output filename from.
        bbox: The study area as ``(west, south, east, north)`` — the order both
            ``osmGet.py --bbox`` and ``--keep-edges.in-geo-boundary`` take.
        refresh: Re-download the OSM extract even when it already exists.
        clip_to_scene: Restrict the network to ``bbox``.
        netconvert_options: Cleanup flags passed through verbatim.
    """

    prefix: str
    bbox: tuple[float, float, float, float]
    refresh: bool
    clip_to_scene: bool
    netconvert_options: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: DictConfig) -> NetworkSpec:
        """Read ``simulation.prefix``, ``simulation.scene.bbox`` and ``simulation.osm/network``."""
        bbox = cfg.simulation.scene.bbox
        return cls(
            prefix=str(cfg.simulation.prefix),
            bbox=(
                float(bbox.min_lon),
                float(bbox.min_lat),
                float(bbox.max_lon),
                float(bbox.max_lat),
            ),
            refresh=bool(cfg.simulation.osm.refresh),
            clip_to_scene=bool(cfg.simulation.network.clip_to_scene),
            netconvert_options=tuple(
                str(option) for option in cfg.simulation.network.netconvert_options
            ),
        )

    @property
    def bbox_arg(self) -> str:
        """The bounding box as the comma-separated string both SUMO tools take."""
        return ",".join(repr(value) for value in self.bbox)


def fetch_osm(paths: NetworkPaths, spec: NetworkSpec) -> Path:
    """Download the OpenStreetMap extract for the study area, or reuse it.

    Reaches the network, unless the extract already exists and ``spec.refresh``
    is false. Returns ``paths.osm``.

    ``osmGet.py`` derives its output filename from ``--prefix`` and the query
    mode, so the expected name is configuration (``simulation.osm.file``); a SUMO
    version that names it differently fails loudly here rather than being
    silently re-downloaded forever.
    """
    if paths.osm.is_file() and not spec.refresh:
        return paths.osm

    osm_dir = paths.osm.parent
    osm_dir.mkdir(parents=True, exist_ok=True)

    toolchain.run(
        [
            sys.executable,
            str(toolchain.tool("osmGet.py")),
            "--bbox",
            spec.bbox_arg,
            "--prefix",
            spec.prefix,
            "-d",
            str(osm_dir),
        ]
    )

    if not paths.osm.is_file():
        produced = sorted(p.name for p in osm_dir.glob("*.osm.xml"))
        raise RuntimeError(
            f"osmGet.py did not write {paths.osm}. Files in {osm_dir}: "
            f"{produced or 'none'}. Set simulation.osm.file to the name this "
            "SUMO version writes."
        )
    return paths.osm


def build_plain(paths: NetworkPaths, spec: NetworkSpec, frame: SceneFrame) -> tuple[Path, Path]:
    """Convert the OSM extract into netconvert's plain XML, in the scene frame.

    Obtains the extract through :func:`fetch_osm`. Returns
    ``(paths.nod, paths.edg)``.

    ``--plain-output-prefix`` also writes the connection, tll and typ files.
    netconvert offers no way to suppress them; they are left on disk, and are
    what makes the plain files assemblable by :func:`build_net`.

    ``spec.clip_to_scene`` bounds the network approximately, not exactly:
    ``--keep-edges.in-geo-boundary`` keeps a crossing edge *whole*, so some
    nodes overhang the bbox. netconvert cannot split an edge at the boundary; a
    caller needing every position inside the scene must clip it itself.
    """
    fetch_osm(paths, spec)
    paths.plain_prefix.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(toolchain.binary("netconvert")),
        "--osm-files",
        str(paths.osm),
        # Bound to the configured zone; --proj.utm would have netconvert guess
        # one from the OSM data instead of matching the scene.
        "--proj",
        frame.proj4,
        # Without this, netconvert re-centres the network on its own minimum
        # corner and discards the offsets below.
        "--offset.disable-normalization",
        "--offset.x",
        repr(frame.offset_x),
        "--offset.y",
        repr(frame.offset_y),
    ]
    if spec.clip_to_scene:
        command += ["--keep-edges.in-geo-boundary", spec.bbox_arg]
    command += list(spec.netconvert_options)
    command += ["--plain-output-prefix", str(paths.plain_prefix)]

    toolchain.run(command)

    missing = [str(path) for path in (paths.nod, paths.edg) if not path.is_file()]
    if missing:
        raise RuntimeError("netconvert exited cleanly but did not write: " + ", ".join(missing))
    return paths.nod, paths.edg


def build_net(paths: NetworkPaths) -> Path:
    """Assemble the plain XML into the ``.net.xml`` the SUMO tools require.

    ``randomTrips.py``, ``duarouter`` and ``sumo`` all need a compiled network;
    :func:`build_plain` stops one step short of one. Returns ``paths.net``.

    The scene frame survives this step by *omission*, which is worth stating
    because both omissions look like oversights:

    - No ``--offset.x`` / ``--offset.y``. netconvert's ``--offset.x`` **adds**
      to the positions it loads, and the plain node coordinates already carry
      the scene offset :func:`build_plain` applied. Passing it again would shift
      the whole network a second time.
    - No ``--proj``. The plain files carry a ``<location>`` element holding the
      ``netOffset`` and ``projParameter`` from the OSM build, and netconvert
      copies both into the assembled net — so the net stays geo-referenced, and
      ``sumolib`` can still convert its coordinates to lon/lat, without the
      projection being restated here.

    ``--offset.disable-normalization`` is not optional: without it netconvert
    re-centres the network on its own minimum corner and silently discards the
    scene frame.
    """
    paths.net.parent.mkdir(parents=True, exist_ok=True)

    toolchain.run(
        [
            str(toolchain.binary("netconvert")),
            "--node-files",
            str(paths.nod),
            "--edge-files",
            str(paths.edg),
            "--connection-files",
            str(paths.con),
            "--tllogic-files",
            str(paths.tll),
            "--type-files",
            str(paths.typ),
            "--offset.disable-normalization",
            "--output-file",
            str(paths.net),
        ]
    )

    if not paths.net.is_file():
        raise RuntimeError(f"netconvert exited cleanly but did not write: {paths.net}")
    return paths.net


def build(cfg: DictConfig) -> NetworkPaths:
    """Ensure the compiled network exists, building only what is missing.

    A step whose output is already on disk is skipped. The plain XML is
    delivered, DVC-tracked data: regenerating it on every trajectory run would
    cost a full netconvert pass over the OSM extract and would overwrite files
    the caller may be holding deliberately. Delete an output to force it to be
    rebuilt.
    """
    paths = NetworkPaths.from_config(cfg)
    if not (paths.nod.is_file() and paths.edg.is_file()):
        build_plain(paths, NetworkSpec.from_config(cfg), SceneFrame.from_config(cfg))
    if not paths.net.is_file():
        build_net(paths)
    return paths
