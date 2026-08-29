"""Download the OSM extract and build the SUMO network and polygon layer.

``src/mobility/`` needs a SUMO road network before UE trajectories can be
generated. This module is the three-command sequence that produces one:
``osmGet.py`` fetches the OpenStreetMap extract for the study area,
``netconvert`` turns it into a SUMO network, and ``polyconvert`` extracts the
buildings and land-use polygons that go with it.

Rationale
---------
The offsets applied by ``netconvert`` are the reason this is a module rather
than a note in a README. ``--proj.utm`` puts the network in UTM metres, and
``--offset.x`` / ``--offset.y`` then shift that origin onto the local frame of
the Sionna-RT scene declared as ``data.scene_file`` in ``configs/data.yaml``.
SUMO coordinates must land in that frame; getting the shift wrong does not
raise, it silently places every UE in the wrong part of the scene. The two
numbers therefore belong somewhere they can be reviewed and changed once, not
retyped per invocation. ``src/mobility/frame.py`` reads the resulting shift
back out of the net file rather than assuming it.

This module shells out to SUMO's own binaries and imports nothing from ``src``:
it produces an input to the pipeline, it is not a stage of it. It lives beside
the modules that consume its output, and — unlike them — takes plain
command-line arguments rather than Hydra configuration, because the study area
and the scene offsets are properties of the delivered scene rather than of a
scenario.

Requires a SUMO installation: ``SUMO_HOME`` must be set, and ``netconvert`` and
``polyconvert`` must be on ``PATH``. Neither is declared in ``pyproject.toml``
-- SUMO is a system package, not a Python dependency.

Example:
    Build into the default directory with the defaults below::

        task scene:build

    Override the study area and the scene offsets::

        task scene:build -- --bbox 105.7,20.9,105.8,21.0 --offset-x -580296.07
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Defaults --------------------------------------------------------------
# The study area, in OSM order: min-longitude, min-latitude, max-longitude,
# max-latitude (WGS84 degrees).
DEFAULT_BBOX = "105.74628353118896,20.94207644462585,105.7985544204712,20.991815328598022"

# Shift applied to the UTM coordinates produced by --proj.utm, in metres, to
# land on the local frame of the Sionna-RT scene. See the module docstring.
DEFAULT_OFFSET_X = -580296.07
DEFAULT_OFFSET_Y = -2318683.15

# Everything under data/ is DVC-tracked and never committed; this is the
# directory configs/data.yaml already points at for the Sionna-RT scene.
DEFAULT_OUT_DIR = Path("data/external/simulation_map")
DEFAULT_PREFIX = "scene"

# Name osmGet.py writes inside <out-dir>/osm/. Queried with --bbox it composes
# "<prefix>_bbox" + the suffix; other query modes and other SUMO versions name
# it differently, so --osm-name stays an override rather than a hardcoded path.
OSM_NAME_TEMPLATE = "{prefix}_bbox.osm.xml"


def sumo_home() -> Path:
    """Return the SUMO installation root.

    Returns:
        The directory named by the ``SUMO_HOME`` environment variable.

    Raises:
        SystemExit: If ``SUMO_HOME`` is unset or does not exist.
    """
    raw = os.environ.get("SUMO_HOME")
    if not raw:
        raise SystemExit(
            "SUMO_HOME is not set. Install Eclipse SUMO and export SUMO_HOME to "
            "its installation root before running this script."
        )
    root = Path(raw)
    if not root.is_dir():
        raise SystemExit(f"SUMO_HOME points at {root}, which is not a directory.")
    return root


def require_binary(name: str) -> str:
    """Resolve a SUMO command-line binary on PATH.

    Args:
        name: Binary to look for, e.g. ``netconvert``.

    Returns:
        The absolute path to the binary.

    Raises:
        SystemExit: If the binary is not on PATH.
    """
    found = shutil.which(name)
    if found is None:
        raise SystemExit(
            f"{name} is not on PATH. It ships with Eclipse SUMO; add $SUMO_HOME/bin to PATH."
        )
    return found


def run(command: list[str], *, dry_run: bool) -> None:
    """Echo a command and run it, failing the script if it fails.

    Args:
        command: Argument vector, passed without a shell.
        dry_run: When true, print the command and return without running it.

    Raises:
        SystemExit: If the command exits non-zero.
    """
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{Path(command[0]).name} exited {result.returncode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build the SUMO network and polygon layer for the Sionna-RT scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="OSM bounding box, W,S,E,N in degrees")
    parser.add_argument("--offset-x", type=float, default=DEFAULT_OFFSET_X, help="UTM x shift, m")
    parser.add_argument("--offset-y", type=float, default=DEFAULT_OFFSET_Y, help="UTM y shift, m")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory to write the artifacts to"
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="basename for the outputs")
    parser.add_argument(
        "--osm-name",
        default=None,
        help=(
            "filename osmGet.py writes inside <out-dir>/osm/ "
            f"(default: {OSM_NAME_TEMPLATE.format(prefix='<prefix>')})"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the three commands without running them"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run osmGet.py, netconvert and polyconvert in order.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv[1:]``.

    Raises:
        SystemExit: If SUMO is missing, if any of the three steps fails, or if
            osmGet.py did not produce the expected OSM file.
    """
    args = parse_args(argv)
    home = sumo_home()

    osm_get = home / "tools" / "osmGet.py"
    if not osm_get.is_file():
        raise SystemExit(f"{osm_get} not found; check the SUMO installation.")
    type_file = home / "data" / "typemap" / "osmPolyconvert.typ.xml"
    if not type_file.is_file():
        raise SystemExit(f"{type_file} not found; check the SUMO installation.")

    netconvert = require_binary("netconvert")
    polyconvert = require_binary("polyconvert")

    out_dir = args.out_dir
    osm_dir = out_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)

    osm_name = args.osm_name or OSM_NAME_TEMPLATE.format(prefix=args.prefix)
    osm_file = osm_dir / osm_name
    net_file = out_dir / f"{args.prefix}.net.xml"
    poly_file = out_dir / f"{args.prefix}.poly.xml"

    # 1. Fetch the OSM extract. Run it with this interpreter; SUMO's tools put
    #    $SUMO_HOME/tools on sys.path themselves, so sumolib resolves.
    run(
        [
            sys.executable,
            str(osm_get),
            "--bbox",
            args.bbox,
            "--prefix",
            args.prefix,
            "-d",
            str(osm_dir),
        ],
        dry_run=args.dry_run,
    )

    if not args.dry_run and not osm_file.is_file():
        produced = sorted(p.name for p in osm_dir.glob("*.osm.xml"))
        raise SystemExit(
            f"osmGet.py did not write {osm_file}. Files in {osm_dir}: "
            f"{produced or 'none'}. Re-run with --osm-name <name> if your SUMO "
            "version writes a different one."
        )

    # 2. Build the SUMO network, projected to UTM and shifted onto the scene
    #    frame. --offset.disable-normalization stops netconvert re-centering
    #    the network, which would discard the shift.
    run(
        [
            netconvert,
            "--osm-files",
            str(osm_file),
            "-o",
            str(net_file),
            "--proj.utm",
            "--offset.disable-normalization",
            "--offset.x",
            repr(args.offset_x),
            "--offset.y",
            repr(args.offset_y),
            "--geometry.remove",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--tls.discard-simple",
        ],
        dry_run=args.dry_run,
    )

    # 3. Extract buildings and land use against the same network, so the
    #    polygons carry the same offset as the edges.
    run(
        [
            polyconvert,
            "--net-file",
            str(net_file),
            "--osm-files",
            str(osm_file),
            "--type-file",
            str(type_file),
            "-o",
            str(poly_file),
        ],
        dry_run=args.dry_run,
    )

    print(f"\nWrote:\n  {osm_file}\n  {net_file}\n  {poly_file}")


if __name__ == "__main__":
    main()
