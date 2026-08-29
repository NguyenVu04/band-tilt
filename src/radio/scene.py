"""Sionna-RT scene construction — PROJECT.md section 16 Phase 1.

The scene is the propagation environment: terrain, building geometry, and the
electromagnetic material assigned to every surface. It is built once and reused
for every tilt configuration, because only the transmitter orientations change
between evaluations.

Build the scene once, not once per evaluation
---------------------------------------------
Loading 3,753 meshes and constructing the acceleration structure costs far more
than a single radio-map solve. The surrogate dataset needs hundreds of solves
over the same geometry, so a scene rebuilt inside the sampling loop turns a
tractable job into an intractable one. Load once, mutate orientations, solve
repeatedly.

The scene defines the coordinate frame
--------------------------------------
Cell positions, MDT positions and the evaluation grid are all expressed in the
scene local frame in metres. :func:`scene_bounds` is the single source of that
extent, so the cleaning stage, the UE density grid and the radio map all agree
on where the area is.

``scene_bounds`` and ``scene_metadata`` are implemented; only Sionna-RT itself
(``load_scene``, ``add_transmitters``) is not. Neither reads more than the scene
XML header and the ground mesh, so both stay usable without the ``rt`` extra —
``src.mobility.frame`` and ``src.data.clean``/``src.data.ue_density`` call them
without pulling in a propagation simulator. Keep it that way: importing
``sionna`` anywhere at module scope in this file would make the ``rt`` extra a
prerequisite for generating UE trajectories, which it must never be.
"""

import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

#: <default name="scenegen_*" value="..."/> entries the scenegen tool writes
#: into the scene XML — the environment's identity (study area, materials,
#: UTM zone, scene centre). scene_metadata() returns exactly these, verbatim.
_SCENEGEN_PREFIX = "scenegen_"


def load_scene(cfg: DictConfig) -> Any:
    """Load the Sionna-RT scene from the 3D map on disk.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_file`` and
            ``cfg.data.scene_dir``.

    Returns:
        The Sionna-RT scene object.

    Raises:
        NotImplementedError: Always — implement this module first.
        FileNotFoundError: Once implemented, when the scene XML is missing.

    Notes:
        Import Sionna-RT inside the function, not at module scope. It is an
        optional extra (``uv sync --extra rt``) and pulls in a large rendering
        stack; a module-level import would make ``import src.radio`` fail for
        anyone doing data work without the simulator installed.

        The materials are declared in the scene XML. Overriding them in code
        would make the propagation environment depend on Python state that the
        scene file does not capture, so results would stop being reproducible
        from the scene alone.

    Example:
        >>> scene = load_scene(cfg)
    """
    # TODO(1): import sionna.rt inside the function body
    # TODO(2): load cfg.data.scene_file
    # TODO(3): raise a message naming the rt extra when the import fails
    raise NotImplementedError("src.radio.scene.load_scene")


def scene_metadata(cfg: DictConfig) -> dict[str, str]:
    """Read the ``scenegen_*`` defaults block from the scene XML.

    The scene-generation tool that built ``cfg.data.scene_file`` records the
    study area and projection it was built from as a run of
    ``<default name="scenegen_..." value="..."/>`` elements at the top of the
    file — center latitude/longitude, the lat/lon bounding box, the UTM zone,
    and the default materials. Nothing else in the scene identifies the
    environment; the mesh geometry alone does not.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_file``.

    Returns:
        A mapping from each ``scenegen_*`` name (with the prefix kept, exactly
        as it appears in the XML) to its value, both as strings — this
        function does not know which entries are numeric and which are not,
        so it does not guess.

    Raises:
        FileNotFoundError: When ``cfg.data.scene_file`` does not exist. Run
            ``task dvc:pull`` — ``data/`` is DVC-tracked and not part of a
            fresh clone.

    Notes:
        This is the single parser of the scene XML's metadata block.
        ``src.mobility.frame.derive_frame`` reads ``scenegen_center_lat/lon``
        and ``scenegen_UTM_zone`` from it to derive the network-to-scene
        translation, and ``src.data.scenario.scene_fingerprint`` hashes it
        whole. One parser, two consumers — see CLAUDE.md's single-source
        table. :func:`scene_bounds` does not use it; it reads the ground mesh
        directly, because the ``scenegen_*`` extent is one of the three
        candidate bounding boxes :func:`scene_bounds` deliberately does not
        use (see its docstring).

        The ``<default name="scenegen_...">`` elements are not all declared
        before the first ``<shape>`` — two of them (``scenegen_bbox_width``
        and ``scenegen_bbox_length``) sit between the first and second shape
        on the delivered scene, so this reads the whole file rather than
        stopping at the first shape. At just over a megabyte that is still
        cheap; it is the 4,737-mesh iteration this docstring warns other
        callers off, not this one.

    Example:
        >>> meta = scene_metadata(cfg)
        >>> meta["scenegen_UTM_zone"]
        'EPSG:32648'
    """
    scene_file = Path(cfg.data.scene_file)
    if not scene_file.is_file():
        raise FileNotFoundError(
            f"{scene_file} does not exist. data/ is DVC-tracked and not part of a "
            "fresh clone — run `task dvc:pull`."
        )
    metadata: dict[str, str] = {}
    # iterparse + clear(): a streaming read of the whole file's <default> tags,
    # without ElementTree ever holding the 4,737 <shape> subtrees in memory at
    # once.
    for _, elem in ET.iterparse(scene_file, events=("end",)):
        if elem.tag == "default" and elem.get("name", "").startswith(_SCENEGEN_PREFIX):
            metadata[elem.get("name")] = elem.get("value", "")
        elem.clear()
    return metadata


def scene_bounds(cfg: DictConfig) -> tuple[float, float, float, float]:
    """Return the scene extent as ``(xmin, ymin, xmax, ymax)`` in metres.

    Args:
        cfg: Composed config; uses ``cfg.data.scene_dir``.

    Returns:
        The bounding box in the scene local coordinate frame.

    Raises:
        FileNotFoundError: When the ground mesh does not exist. Run
            ``task dvc:pull`` — ``data/`` is DVC-tracked and not part of a
            fresh clone.

    Notes:
        Read from the ground mesh (``<scene_dir>/mesh/ground.ply``), which is
        a single flat quad — a 122-byte file, four vertices — not by iterating
        the 4,737 building meshes, which is slow enough to matter when a
        cleaning loop calls it.

        The ground quad is not axis-aligned: on the delivered scene it is
        rotated by about 0.276 degrees, which is exactly the UTM grid
        convergence at the scene's longitude in zone 48N. Its axis-aligned
        bounding box therefore overhangs the true ground footprint by up to
        ~30 m at the corners — a location in that overhang has no reflecting
        surface under it and is not a coverage measurement. This function
        returns the **inscribed** axis-aligned rectangle instead: for each of
        the two vertices left of the mesh centroid, ``xmin`` takes the more
        restrictive (larger) of the two x-coordinates, and symmetrically for
        ``xmax``, ``ymin`` and ``ymax``. That guarantees every location the
        rectangle contains has ground under it, at some cost in coverage near
        the corners.

        Two other extents are recorded in the scene file and neither is used
        here: ``scenegen_bbox_width``/``length`` are the tool's own bounding
        box, and the four ``scenegen_min/max_lat/lon`` corners project to a
        third rectangle. All three differ from each other and from the ground
        mesh by tens of metres, because they are metadata approximations of a
        shape the mesh states exactly.

    Example:
        >>> xmin, ymin, xmax, ymax = scene_bounds(cfg)
    """
    ground_path = Path(cfg.data.scene_dir) / "mesh" / "ground.ply"
    if not ground_path.is_file():
        raise FileNotFoundError(
            f"{ground_path} does not exist. data/ is DVC-tracked and not part of a "
            "fresh clone — run `task dvc:pull`."
        )
    vertices = _read_ply_vertices_xy(ground_path)
    cx = sum(x for x, _ in vertices) / len(vertices)
    cy = sum(y for _, y in vertices) / len(vertices)
    left = [x for x, y in vertices if x < cx]
    right = [x for x, y in vertices if x > cx]
    below = [y for x, y in vertices if y < cy]
    above = [y for x, y in vertices if y > cy]
    xmin, xmax = max(left), min(right)
    ymin, ymax = max(below), min(above)
    return (xmin, ymin, xmax, ymax)


def _read_ply_vertices_xy(path: Path) -> list[tuple[float, float]]:
    r"""Read the ``(x, y)`` of every vertex from a binary little-endian PLY.

    Matches the format the scene-generation tool writes (Open3D,
    ``format binary_little_endian 1.0``, ``double`` or ``float`` x/y/z
    properties): a plain-text header terminated by ``end_header\\n``, then the
    vertex records packed back to back. Face data, if present, is not read.

    Args:
        path: Path to the ``.ply`` file.

    Returns:
        One ``(x, y)`` tuple per vertex, in file order.

    Raises:
        ValueError: When the header does not declare a ``vertex`` element, or
            declares ``x``/``y`` in an unsupported scalar type.
    """
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end == -1:
        raise ValueError(f"{path}: no end_header — not a PLY file")
    header = data[:header_end].decode("ascii")
    body = data[header_end + len(b"end_header\n") :]

    n_vertices: int | None = None
    props: list[tuple[str, str]] = []  # (type, name), vertex element only
    in_vertex_element = False
    for line in header.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "element":
            in_vertex_element = tokens[1] == "vertex"
            if in_vertex_element:
                n_vertices = int(tokens[2])
        elif tokens[0] == "property" and in_vertex_element:
            if tokens[1] == "list":
                break  # a list property (e.g. face indices) ends the vertex block
            props.append((tokens[1], tokens[2]))
    if n_vertices is None:
        raise ValueError(f"{path}: no 'element vertex' in the header")

    _FMT = {"float": "f", "double": "d", "float32": "f", "float64": "d"}
    try:
        record_fmt = "<" + "".join(_FMT[ptype] for ptype, _ in props)
    except KeyError as exc:
        raise ValueError(f"{path}: unsupported PLY scalar type {exc}") from exc
    record_size = struct.calcsize(record_fmt)
    names = [name for _, name in props]
    xi, yi = names.index("x"), names.index("y")

    vertices = []
    for i in range(n_vertices):
        record = struct.unpack_from(record_fmt, body, i * record_size)
        vertices.append((record[xi], record[yi]))
    return vertices


def add_transmitters(scene: Any, table: pd.DataFrame, cfg: DictConfig) -> Any:
    """Attach one transmitter per cell-band to the scene.

    Args:
        scene: The scene from :func:`load_scene`.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        cfg: Composed config; uses ``cfg.radio.antenna``.

    Returns:
        The scene, with transmitters attached in table order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Attach transmitters in the table order and give each one a name derived
        from ``(gcell_id, band)``. The radio map comes back indexed by
        transmitter, and that index has to map back to a row of the cell-band
        table without guessing.

        Orientation is not set here — it changes per evaluation and is applied
        by :func:`src.radio.radiomap.set_tilt`.

    Example:
        >>> scene = add_transmitters(load_scene(cfg), table, cfg)
    """
    # TODO(1): one Transmitter per row, positioned at (sim_x, sim_y, antenna_height)
    # TODO(2): name each transmitter from (gcell_id, band) so the index is recoverable
    # TODO(3): apply the antenna array from cfg.radio.antenna and the band carrier
    raise NotImplementedError("src.radio.scene.add_transmitters")
