"""The Sionna-RT scene's coordinate frame, as netconvert options.

The scene's frame is UTM (``simulation.scene.utm_zone``) metres shifted so
``simulation.scene.center`` sits at the origin. Passing netconvert the negated
projected centre as ``--offset.x`` / ``--offset.y`` therefore makes a SUMO
coordinate *equal* a scene coordinate.

The offset is re-derived from ``configs/simulation.yaml`` on every call: a stale
literal does not raise, it silently puts every node in the wrong street.
``scene.xml``'s ``scenegen_*`` block records the same values but is not read
here — the config is the authority, and the two are synced by hand.

This is the only module that imports ``pyproj``.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class SceneFrame:
    """The projection and translation that map WGS84 onto the scene local frame.

    Attributes:
        proj4: The proj4 string netconvert's ``--proj`` takes.
        offset_x: Metres to add to a projected x to obtain a scene x.
        offset_y: Metres to add to a projected y to obtain a scene y.
    """

    proj4: str
    offset_x: float
    offset_y: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SceneFrame:
        """Derive the frame from ``simulation.scene`` (``utm_zone`` and ``center``).

        Raises:
            ImportError: When pyproj is not installed.
            ValueError: When ``utm_zone`` is not a UTM EPSG code.
        """
        try:
            import pyproj
        except ImportError as exc:  # pragma: no cover - depends on the installed extras
            raise ImportError(
                "pyproj is required to derive the scene offset. "
                "Install it with `uv sync --extra sumo`."
            ) from exc

        proj4 = utm_proj4(str(cfg.simulation.scene.utm_zone))
        center = cfg.simulation.scene.center
        transformer = pyproj.Transformer.from_crs("EPSG:4326", proj4, always_xy=True)
        x, y = transformer.transform(float(center.lon), float(center.lat))
        return cls(proj4=proj4, offset_x=-x, offset_y=-y)


def utm_proj4(epsg: str) -> str:
    """Translate a UTM EPSG code into the proj4 string netconvert takes.

    Accepts ``EPSG:326zz`` (north) and ``EPSG:327zz`` (south), where ``zz`` is the
    zone number. Anything else raises: a non-UTM projection would need a
    different network-to-scene relation than the pure translation applied here.

    Example:
        >>> utm_proj4("EPSG:32648")
        '+proj=utm +zone=48 +north +datum=WGS84 +units=m +no_defs'
    """
    code = epsg.strip().upper().removeprefix("EPSG:")
    invalid = ValueError(f"{epsg!r} is not a UTM EPSG code (expected EPSG:326zz or EPSG:327zz).")
    if not code.isdigit() or len(code) != 5:
        raise invalid
    band, zone = code[:3], int(code[3:])
    if band not in {"326", "327"} or not 1 <= zone <= 60:
        raise invalid
    hemisphere = "+north" if band == "326" else "+south"
    return f"+proj=utm +zone={zone} {hemisphere} +datum=WGS84 +units=m +no_defs"
