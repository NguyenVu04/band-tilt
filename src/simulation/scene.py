"""Load Sionna-RT scenes and query their bounds and surface heights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class SceneSpec:
    """Which scene to load, and what to compute it with.

    Attributes:
        name: A scene bundled with sionna-rt, resolved against
            ``sionna.rt.scene``.
        mitsuba_variant: Mitsuba compute variant, selected before sionna-rt is
            imported. Must be a ``mono_polarized`` one.
        merge_shapes: Let sionna-rt merge shapes sharing a radio material.
            Merging is a solver optimisation, but it collapses a city into a
            handful of per-material blobs.
    """

    name: str
    mitsuba_variant: str
    merge_shapes: bool

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SceneSpec:
        """Read ``simulation.scene``."""
        scene = cfg.simulation.scene
        return cls(
            name=str(scene.name),
            mitsuba_variant=str(scene.mitsuba_variant),
            merge_shapes=bool(scene.merge_shapes),
        )


@dataclass(frozen=True)
class SceneBounds:
    """The scene's axis-aligned extent, in scene metres.

    Attributes:
        min_x: Lower x bound.
        max_x: Upper x bound.
        min_y: Lower y bound.
        max_y: Upper y bound.
        min_z: Lower z bound.
        max_z: Upper z bound; the tallest geometry in the scene.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width_m(self) -> float:
        """Extent along x."""
        return self.max_x - self.min_x

    @property
    def depth_m(self) -> float:
        """Extent along y."""
        return self.max_y - self.min_y

    @property
    def launch_z(self) -> float:
        """A z above every surface, to cast downward rays from.

        Ten metres clear of the tallest geometry, so a ray launched here starts
        outside the scene regardless of what stands below it.
        """
        return self.max_z + 10.0

    def inset(self, margin_m: float) -> SceneBounds:
        """This extent pulled in by ``margin_m`` on all four horizontal sides.

        The region of interest. Near the scene boundary there is no geometry
        beyond the edge to reflect or block anything, so power leaks outward
        and the radio map reads optimistically there. Everything the scenario
        places on purpose -- UEs, hotspot centres, masts -- is confined to this
        interior; the grid and the radio map still span the full extent, so
        rays arriving from the margin are not lost.

        The z bounds are carried through unchanged: the margin is horizontal,
        and ``launch_z`` must still clear the tallest geometry in the *whole*
        scene, not merely the part inside the region.

        Raises:
            ValueError: When the margin is negative, or so large that it leaves
                no interior.
        """
        if margin_m < 0:
            raise ValueError(f"simulation.area.margin_m must not be negative, got {margin_m}")
        if 2.0 * margin_m >= min(self.width_m, self.depth_m):
            raise ValueError(
                f"simulation.area.margin_m of {margin_m} m leaves no interior in a scene "
                f"{self.width_m:.1f} x {self.depth_m:.1f} m"
            )
        return SceneBounds(
            min_x=self.min_x + margin_m,
            max_x=self.max_x - margin_m,
            min_y=self.min_y + margin_m,
            max_y=self.max_y - margin_m,
            min_z=self.min_z,
            max_z=self.max_z,
        )


def load(spec: SceneSpec) -> tuple[Any, SceneBounds]:
    """Load the scene and read its extent.

    Returns the sionna-rt ``Scene`` and its bounds. Callers reach the Mitsuba
    scene through ``scene.mi_scene`` and should read it *afresh* each time
    rather than holding it: ``Scene.edit`` builds a new Mitsuba scene, so a
    cached reference silently goes stale the moment a building is removed.

    Raises:
        ImportError: When sionna-rt or Mitsuba is not installed.
        ValueError: When ``spec.name`` is not a scene bundled with sionna-rt,
            or when the variant is not a ``mono_polarized`` one.
        RuntimeError: When the loaded scene carries no Mitsuba scene to measure.
    """
    try:
        import mitsuba as mi
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "mitsuba is required to load the scene. Install it with `uv sync --extra rt`."
        ) from exc

    if "mono_polarized" not in spec.mitsuba_variant:
        raise ValueError(
            f"simulation.scene.mitsuba_variant is {spec.mitsuba_variant!r}, but sionna-rt "
            "needs a 'mono_polarized' variant: it packs a 2x2 Jones matrix into mi.Spectrum, "
            "which holds four components only there. Another variant rasters the scene "
            "correctly and then fails inside the radio solvers."
        )
    mi.set_variant(spec.mitsuba_variant)

    try:
        import sionna.rt as rt
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "sionna-rt is required to load the scene. Install it with `uv sync --extra rt`."
        ) from exc

    scene = rt.load_scene(_scene_file(rt, spec.name), merge_shapes=spec.merge_shapes)
    return scene, bounds_of(scene)


def bounds_of(scene: Any) -> SceneBounds:
    """Read a scene's axis-aligned extent.

    Separate from :func:`load` so an edited scene can be re-measured: the grid
    must be laid out over what the scene is now.

    Raises:
        RuntimeError: When the scene carries no Mitsuba scene to measure.
    """
    mi_scene: Any = scene.mi_scene
    try:
        bbox = mi_scene.bbox()
    except AttributeError as exc:
        raise RuntimeError(
            "scene loaded without a Mitsuba scene: its `mi_scene` is a "
            f"{type(mi_scene).__name__}, which has no `bbox()`."
        ) from exc
    lo, hi = bbox.min, bbox.max
    return SceneBounds(
        min_x=float(lo.x),
        max_x=float(hi.x),
        min_y=float(lo.y),
        max_y=float(hi.y),
        min_z=float(lo.z),
        max_z=float(hi.z),
    )


def surface_height(mi_scene: Any, x: np.ndarray, y: np.ndarray, launch_z: float) -> np.ndarray:
    """Height of the topmost surface at each ``(x, y)``, by downward ray cast.

    Casts one ray straight down per point, from ``launch_z``. Returns an array
    of the same shape carrying the hit height, and ``nan`` where the ray left
    the scene without hitting anything — which is a point beyond the ground,
    not a point at height zero, and must not be read as open ground.

    The scene's ground is a single flat surface, so a height at or near zero is
    open ground and anything above it is a structure standing on that ground.
    This one measurement therefore serves both the free-space mask and the
    built-volume proxy the UE density is weighted by.
    """
    import mitsuba as mi

    flat_x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    flat_y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())

    origin = mi.Point3f(flat_x, flat_y, np.full(flat_x.shape, launch_z))
    direction = mi.Vector3f(0.0, 0.0, -1.0)
    interaction = mi_scene.ray_intersect(mi.Ray3f(origin, direction))

    distance = np.asarray(interaction.t, dtype=np.float64)
    hit = np.asarray(interaction.is_valid(), dtype=bool)

    height = np.full(flat_x.shape, np.nan, dtype=np.float64)
    height[hit] = launch_z - distance[hit]
    return height.reshape(np.shape(x))


def _scene_file(rt: Any, name: str) -> str:
    """Resolve a bundled scene name to the scene file sionna-rt ships.

    Raises:
        ValueError: When ``name`` is not a bundled scene.
    """
    path = getattr(rt.scene, name, None)
    if not isinstance(path, str):
        available = sorted(
            attribute
            for attribute in dir(rt.scene)
            if not attribute.startswith("_") and isinstance(getattr(rt.scene, attribute), str)
        )
        raise ValueError(
            f"{name!r} is not a scene bundled with sionna-rt. Available: {', '.join(available)}."
        )
    return path
