"""The SUMO-network to Sionna-RT scene coordinate transform — single owner.

SUMO and Sionna-RT describe the same place in two different local metric
frames. A wrong or a hardcoded transform does not raise: it silently places
every UE in the wrong street, and every KPI computed downstream is then about
a network that does not exist. This module is the one place that transform may
be derived, applied or checked.

Derived at run time, never a literal
-------------------------------------
Both frames on the delivered scene turn out to be EPSG:32648 (UTM zone 48N)
metres, so the relation reduces to a pure translation — but *that* fact is a
property of this data, not a guarantee. The translation itself is computed
from two small XML headers every time it is needed:

- the SUMO net's ``<location netOffset="..." projParameter="..."/>``
  (``src.data.scenario.read_net_location``), and
- the Sionna scene's ``scenegen_center_lat``/``scenegen_center_lon``
  (``src.radio.scene.scene_metadata``).

A network rebuilt with different ``--offset.x``/``--offset.y``
(``src/mobility/scene_build.py``), or a scene regenerated over a different study
area, changes one of these files and this module picks it up automatically.
A hardcoded ``(dx, dy)`` would keep applying the stale shift instead.
:data:`configs/mobility.yaml`'s ``mobility.frame.expected_offset`` is a
*guard*, not a source: :func:`derive_frame` raises when the freshly derived
offset has moved beyond ``tolerance_m`` from it, which is what catches a
rebuild nobody meant to be silent.

The network is bigger than the scene
---------------------------------------
On the delivered data the SUMO road network extends beyond the Sionna-RT
ground plane on all four sides — the network was built from a wider OSM
extract than the ray-tracing scene covers. Positions outside the scene are
therefore the ordinary case, not an exceptional one, and asserting every
mapped position lies inside the scene bounds (notebook 01's original section 3
check) is simply false against this data. :func:`clip_to_scene` clips and
reports the clipped fraction instead of asserting it away.

Importable without the ``sumo`` extra
----------------------------------------
``pyproj`` is imported inside :func:`derive_frame`, not at module scope, so
that :class:`SceneFrame`, :func:`to_scene_frame` and :func:`clip_to_scene`
stay usable — and this module stays importable — without ``uv sync
--extra sumo`` installed. ``src.mobility.simulate`` imports this module for
those pure functions; only actually deriving a transform needs ``pyproj``.
"""

from dataclasses import dataclass

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data import scenario
from src.radio import scene


@dataclass(frozen=True)
class SceneFrame:
    """The derived SUMO-network to Sionna-scene transform, with its provenance.

    Carrying the source values alongside ``dx``/``dy`` — rather than a bare
    offset tuple — means the manifest written by
    :func:`src.mobility.persist.save_trajectories` and the clip performed by
    :func:`clip_to_scene` can never disagree about where the two numbers came
    from: they read the same frozen object.

    Attributes:
        dx: Added to a SUMO-local x to obtain the scene-local x.
        dy: Added to a SUMO-local y to obtain the scene-local y.
        bounds: The Sionna-RT scene extent, ``(xmin, ymin, xmax, ymax)`` in
            the scene local frame — from :func:`src.radio.scene.scene_bounds`.
        net_offset: The SUMO net's ``<location netOffset>``, as written by
            ``netconvert --offset.x``/``--offset.y``.
        proj_parameter: The SUMO net's ``<location projParameter>`` — the
            proj4 string the network coordinates are projected through. This
            is what the scene centre is projected with, not an assumed EPSG
            code, so a net built under a different projection is handled
            correctly rather than silently mismatched.
        scene_centre_lonlat: ``(lon, lat)`` of the Sionna scene's local
            origin, WGS84 degrees — ``scenegen_center_lon``/``_lat``.
        utm_zone: The scene tool's own record of the UTM zone
            (``scenegen_UTM_zone``), kept for provenance; not used in the
            projection itself, since ``proj_parameter`` is authoritative.
    """

    dx: float
    dy: float
    bounds: tuple[float, float, float, float]
    net_offset: tuple[float, float]
    proj_parameter: str
    scene_centre_lonlat: tuple[float, float]
    utm_zone: str


def derive_frame(cfg: DictConfig) -> SceneFrame:
    """Derive the SUMO-network to Sionna-scene translation, at run time.

    Args:
        cfg: Composed config; uses ``cfg.mobility.network.net_file``,
            ``cfg.data.scene_file``/``cfg.data.scene_dir`` (via
            :func:`src.radio.scene.scene_metadata` and
            :func:`src.radio.scene.scene_bounds`), and
            ``cfg.mobility.frame.expected_offset``/``tolerance_m``.

    Returns:
        The derived :class:`SceneFrame`.

    Raises:
        FileNotFoundError: When the net file or the scene XML is missing.
        ImportError: When ``pyproj`` is not installed — install with
            ``uv sync --extra sumo``. Not imported at module scope: this
            module's other exports (in particular the :class:`SceneFrame`
            dataclass) must stay usable without the ``sumo`` extra, the same
            rule ``src.radio.scene.load_scene`` follows for Sionna-RT.
        ValueError: When the derived offset has moved more than
            ``cfg.mobility.frame.tolerance_m`` from
            ``cfg.mobility.frame.expected_offset``. See the module docstring.

    Notes:
        Project the scene's local-origin coordinates (WGS84 lon/lat) through
        the NET's own projection (``proj_parameter``) to land in the same UTM
        frame the net uses; add ``net_offset`` to land in SUMO-local
        coordinates; negate. That vector, added to a SUMO-local position,
        lands in the scene-local frame.

        This reads only the two files' small metadata headers — never the
        road network's edges and lanes (tens of megabytes) or the scene's
        4,737 mesh references — so it is cheap enough to call before every
        scenario is generated, not just once by hand.

    Example:
        >>> frame = derive_frame(cfg)
        >>> round(frame.dx, 4), round(frame.dy, 4)
        (244.7237, 129.4592)
    """
    import pyproj  # deferred — see the module docstring and the Raises entry above

    location = scenario.read_net_location(cfg.mobility.network.net_file)
    net_offset = tuple(float(v) for v in location["netOffset"].split(","))
    proj_parameter = location["projParameter"]

    meta = scene.scene_metadata(cfg)
    centre_lon = float(meta["scenegen_center_lon"])
    centre_lat = float(meta["scenegen_center_lat"])
    utm_zone = meta["scenegen_UTM_zone"]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_parameter, always_xy=True)
    centre_x_utm, centre_y_utm = transformer.transform(centre_lon, centre_lat)
    centre_x_net = centre_x_utm + net_offset[0]
    centre_y_net = centre_y_utm + net_offset[1]
    dx, dy = -centre_x_net, -centre_y_net

    bounds = scene.scene_bounds(cfg)

    frame = SceneFrame(
        dx=dx,
        dy=dy,
        bounds=bounds,
        net_offset=net_offset,
        proj_parameter=proj_parameter,
        scene_centre_lonlat=(centre_lon, centre_lat),
        utm_zone=utm_zone,
    )

    expected = cfg.mobility.frame.get("expected_offset")
    if expected is not None:
        assert_offset_within_tolerance(
            dx, dy, float(expected[0]), float(expected[1]), float(cfg.mobility.frame.tolerance_m)
        )

    return frame


def assert_offset_within_tolerance(
    dx: float, dy: float, expected_dx: float, expected_dy: float, tolerance_m: float
) -> None:
    """Raise when a derived offset has moved beyond tolerance from the expected one.

    Args:
        dx: The freshly derived x offset.
        dy: The freshly derived y offset.
        expected_dx: ``cfg.mobility.frame.expected_offset[0]``.
        expected_dy: ``cfg.mobility.frame.expected_offset[1]``.
        tolerance_m: ``cfg.mobility.frame.tolerance_m``.

    Raises:
        ValueError: When the Euclidean distance between ``(dx, dy)`` and
            ``(expected_dx, expected_dy)`` exceeds ``tolerance_m``.

    Notes:
        Factored out of :func:`derive_frame` so the tolerance check itself is
        testable without a real SUMO net file or scene XML — every other line
        of :func:`derive_frame` depends on one of the two.

    Example:
        >>> assert_offset_within_tolerance(244.7, 129.5, 244.7237, 129.4592, 1.0)
    """
    moved = ((dx - expected_dx) ** 2 + (dy - expected_dy) ** 2) ** 0.5
    if moved > tolerance_m:
        raise ValueError(
            f"The derived SUMO-to-scene offset ({dx:.4f}, {dy:.4f}) has "
            f"moved {moved:.4f} m from mobility.frame.expected_offset "
            f"({expected_dx}, {expected_dy}), beyond tolerance_m="
            f"{tolerance_m}. The road network or the scene was likely "
            "rebuilt against a different study area. Update "
            "expected_offset once the new value has been checked by hand "
            "(`task mobility:frame`); do not raise tolerance_m to make "
            "this pass."
        )


def to_scene_frame(df: pd.DataFrame, frame: SceneFrame) -> pd.DataFrame:
    """Translate ``sim_x``/``sim_y`` from SUMO-network to Sionna-scene coordinates.

    Args:
        df: A frame with ``sim_x``, ``sim_y`` columns in SUMO network
            coordinates (SUMO's raw ``x``/``y``, not yet translated).
        frame: The transform from :func:`derive_frame`.

    Returns:
        A copy of ``df`` with ``sim_x``/``sim_y`` translated into the scene
        local frame. Every other column is unchanged.

    Notes:
        A pure translation (``sim_x += frame.dx``, ``sim_y += frame.dy``),
        because both frames share one UTM projection — see
        :func:`derive_frame`. Clipping to the scene extent is a deliberately
        separate step, :func:`clip_to_scene`, so a caller can inspect the
        untranslated and translated positions side by side before anything is
        dropped.

    Example:
        >>> to_scene_frame(df, frame)[["sim_x", "sim_y"]]
    """
    out = df.copy()
    out["sim_x"] = out["sim_x"] + frame.dx
    out["sim_y"] = out["sim_y"] + frame.dy
    return out


def clip_to_scene(
    df: pd.DataFrame, frame: SceneFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Restrict trajectory samples to the Sionna-RT scene extent.

    Args:
        df: A frame with ``sim_x``/``sim_y`` already in the scene local frame
            (i.e. already passed through :func:`to_scene_frame`) and a
            ``ue_id`` column.
        frame: Carries the scene ``bounds`` to clip against.
        cfg: Composed config; uses ``cfg.mobility.clip``
            (``policy``, ``max_clipped_fraction``).

    Returns:
        ``(clipped_df, stats)`` — the retained rows, sorted by ``(ue_id, t)``
        when both columns are present, and a stats mapping with
        ``rows_in``, ``rows_out``, ``clipped_fraction`` and ``ues_dropped``.

    Raises:
        ValueError: When ``cfg.mobility.clip.policy`` is not one of
            ``drop_samples``/``drop_ues``/``error``; when ``policy="error"``
            and at least one position falls outside the bounds; or when more
            than ``cfg.mobility.clip.max_clipped_fraction`` of rows fall
            outside the bounds under any policy.

    Notes:
        The delivered road network is larger than the ray-tracing scene on
        all four sides (see ``configs/mobility.yaml``'s ``clip`` block), so
        clipping is the ordinary outcome here, not a failure. ``max_clipped_
        fraction`` is the guard that turns a genuinely mismatched network and
        scene into a loud failure rather than a quietly small trajectory set.

    Example:
        >>> clipped, stats = clip_to_scene(df, frame, cfg)
        >>> 0.0 <= stats["clipped_fraction"] <= 1.0
        True
    """
    policy = cfg.mobility.clip.policy
    xmin, ymin, xmax, ymax = frame.bounds
    inside = df["sim_x"].between(xmin, xmax) & df["sim_y"].between(ymin, ymax)
    rows_in = len(df)

    if policy == "error":
        if not inside.all():
            n_out = int((~inside).sum())
            raise ValueError(
                f"{n_out}/{rows_in} positions fall outside the scene bounds "
                f"{frame.bounds} under clip.policy='error'. Use "
                "'drop_samples' or 'drop_ues' instead, or rebuild the network "
                "to match the scene."
            )
        clipped = df.copy()
        ues_dropped = 0
    elif policy == "drop_samples":
        clipped = df.loc[inside].copy()
        ues_dropped = 0
    elif policy == "drop_ues":
        offending = set(df.loc[~inside, "ue_id"])
        clipped = df.loc[~df["ue_id"].isin(offending)].copy()
        ues_dropped = len(offending)
    else:
        raise ValueError(
            f"Unknown mobility.clip.policy={policy!r}; expected one of "
            "'drop_samples', 'drop_ues', 'error'."
        )

    rows_out = len(clipped)
    clipped_fraction = 0.0 if rows_in == 0 else 1.0 - rows_out / rows_in
    max_fraction = float(cfg.mobility.clip.max_clipped_fraction)
    if clipped_fraction > max_fraction:
        raise ValueError(
            f"{clipped_fraction:.1%} of positions fell outside the scene "
            f"bounds {frame.bounds}, above mobility.clip.max_clipped_fraction="
            f"{max_fraction:.1%}. The road network and the scene probably do "
            "not describe the same place — check `task mobility:frame` "
            "before loosening this guard."
        )

    sort_cols = [c for c in ("ue_id", "t") if c in clipped.columns]
    if sort_cols:
        clipped = clipped.sort_values(sort_cols).reset_index(drop=True)

    stats = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "clipped_fraction": clipped_fraction,
        "ues_dropped": ues_dropped,
    }
    return clipped, stats


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Print the derived transform and the network/scene extents.

    Entry point for ``task mobility:frame`` — checks the single highest-risk
    number in this package (the offset) in about a second, without generating
    a single trajectory.

    Args:
        cfg: Composed config (Hydra).

    Example:
        $ task mobility:frame
    """
    frame = derive_frame(cfg)
    print(f"dx = {frame.dx:.4f}  dy = {frame.dy:.4f}")
    print(f"net_offset           = {frame.net_offset}")
    print(f"proj_parameter       = {frame.proj_parameter}")
    print(f"scene centre lon/lat = {frame.scene_centre_lonlat}")
    print(f"utm_zone             = {frame.utm_zone}")
    print(f"scene bounds (scene frame) = {frame.bounds}")

    location = scenario.read_net_location(cfg.mobility.network.net_file)
    conv = tuple(float(v) for v in location["convBoundary"].split(","))
    net_bbox_scene = (
        conv[0] + frame.dx,
        conv[1] + frame.dy,
        conv[2] + frame.dx,
        conv[3] + frame.dy,
    )
    print(f"net bbox   (net frame)     = {conv}")
    print(f"net bbox   (scene frame)   = {net_bbox_scene}")

    sx0, sy0, sx1, sy1 = frame.bounds
    nx0, ny0, nx1, ny1 = net_bbox_scene
    if nx0 < sx0 or ny0 < sy0 or nx1 > sx1 or ny1 > sy1:
        print(
            "The road network extends beyond the scene on at least one side "
            "- trajectories will be clipped, not merely bounded. See "
            "configs/mobility.yaml:clip."
        )
    else:
        print("The road network lies entirely inside the scene bounds.")


if __name__ == "__main__":
    main()
