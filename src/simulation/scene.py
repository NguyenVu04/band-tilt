"""The Sionna-RT scene, its extent, and the surface height above it.

The only module that imports ``sionna.rt`` or ``mitsuba``. Everything
downstream works in NumPy against :func:`surface_height`, which reduces the
scene to the one measurement the UE population needs: how high the geometry
stands at a given ``(x, y)``.

The scene's extent is read from the loaded scene rather than from
``configs/simulation.yaml``. A restated bound does not raise when it drifts
from the geometry; it silently places UEs off the scene.

Mitsuba's compute variant is selected here, before ``sionna.rt`` is imported.
The order matters twice over. The variant fixes the types Mitsuba builds its
bindings from, and sionna-rt binds against whatever is current when it loads —
but sionna-rt also selects its own variant if none is set yet
(``sionna/rt/__init__.py``), so setting one first *overrides* that choice.

Only the ``mono_polarized`` variants work. sionna-rt packs a 2x2 Jones matrix
into ``mi.Spectrum``, which holds four components there and three in an ``rgb``
variant. An ``rgb`` variant therefore loads, ray-casts and rasters perfectly
well, and then fails deep inside the radio solvers with a bare
``Color3f.__init__(): Input has the wrong size``. :func:`load` rejects it up
front rather than letting that surface an hour later.
"""

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
            handful of per-material blobs, and a building that is no longer a
            distinct object cannot be perturbed.
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

    Separate from :func:`load` because perturbation changes it: removing or
    raising buildings moves the bounding box, and the grid must be laid out
    over what the scene is now.

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
