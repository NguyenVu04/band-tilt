"""Geometry perturbation: what this scenario's city gets wrong about the real one.

Three perturbations, applied once per scenario: buildings removed, buildings
made taller or shorter, and buildings nudged and turned.

The last of those is **survey error, not urban change**. Buildings do not
slide sideways or swivel; the footprints a scene is built from carry position
and orientation error, and this reproduces that. Framing it as "the building
moved" would not survive review.

Two traps this module exists to get right, both verified against the geometry
rather than assumed:

Scaling is about the object's centroid, not its base
    Sionna scales a mesh about its own centre, so scaling height by ``s``
    sinks a building's base below the ground by as much as it raises the roof,
    and delivers only half the intended change above ground. Scaling about the
    ground plane instead is exactly ``position.z *= s``: the base stays at
    zero and the roof rises by the full factor. This assumes buildings stand on
    a ground plane at z = 0, which is what the bundled scenes provide.

A building is several objects
    Walls and roof arrive as separate meshes sharing a name prefix, with
    *different* centroids. Rotating each about its own centre pulls a building
    apart. Every part is therefore transformed about one common centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig

# Scene objects are named "<building>-itu_<material>"; the ground is a lone
# object with no such suffix and must never be removed or moved.
_PART_SEPARATOR = "-itu_"
_GROUND = "ground"


@dataclass(frozen=True)
class PerturbSpec:
    """How far this scenario's geometry may stray from the delivered scene.

    Attributes:
        n_removed: Buildings to delete outright.
        n_jittered: Buildings to resize, nudge and turn.
        height_scale: Multiplier range for building height, about the ground.
        position_m: Maximum horizontal offset, per axis.
        orientation_deg: Maximum rotation about the vertical axis.
    """

    n_removed: int
    n_jittered: int
    height_scale: tuple[float, float]
    position_m: float
    orientation_deg: float

    def __post_init__(self) -> None:
        """Reject a perturbation that cannot be applied.

        Raises:
            ValueError: When a count is negative or a height scale is not
                positive.
        """
        if self.n_removed < 0 or self.n_jittered < 0:
            raise ValueError(
                "simulation.perturbation counts must not be negative, got "
                f"n_removed={self.n_removed}, n_jittered={self.n_jittered}"
            )
        if min(self.height_scale) <= 0:
            raise ValueError(
                f"simulation.perturbation.jitter.height_scale must be positive, "
                f"got {self.height_scale}"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> PerturbSpec:
        """Read ``simulation.perturbation``."""
        perturbation = cfg.simulation.perturbation
        jitter = perturbation.jitter
        low, high = (float(value) for value in jitter.height_scale)
        return cls(
            n_removed=int(perturbation.remove.n_buildings),
            n_jittered=int(jitter.n_buildings),
            height_scale=(low, high),
            position_m=float(jitter.position_m),
            orientation_deg=float(jitter.orientation_deg),
        )


@dataclass(frozen=True)
class PerturbReport:
    """What was actually done to the scene, for the scenario manifest.

    Attributes:
        removed: Names of the buildings deleted.
        jittered: Names of the buildings resized, nudged and turned.
    """

    removed: tuple[str, ...]
    jittered: tuple[str, ...]


def buildings(scene: Any) -> dict[str, list[Any]]:
    """Group scene objects into buildings by name, excluding the ground.

    Walls and roof share a name prefix and are returned together, so a
    perturbation reaches every part of a building or none of it.
    """
    grouped: dict[str, list[Any]] = {}
    for obj in scene.objects.values():
        name = obj.name.split(_PART_SEPARATOR)[0]
        if name == _GROUND:
            continue
        grouped.setdefault(name, []).append(obj)
    return grouped


def apply(scene: Any, spec: PerturbSpec, seed: int) -> PerturbReport:
    """Remove, then resize and displace buildings. Returns what was touched.

    Removal happens first so no effort is spent jittering a building that is
    about to be deleted, and so the two draws cannot select the same building.

    ``scene.edit`` rebuilds the Mitsuba scene, so callers must re-read
    ``scene.mi_scene`` afterwards rather than holding a reference across this
    call.

    Raises:
        ValueError: When the scene holds fewer buildings than the spec asks to
            perturb.
    """
    rng = np.random.default_rng(seed)
    grouped = buildings(scene)
    names = sorted(grouped)

    wanted = spec.n_removed + spec.n_jittered
    if wanted > len(names):
        raise ValueError(
            f"simulation.perturbation asks to touch {wanted} buildings but the scene "
            f"holds {len(names)}."
        )

    chosen = rng.choice(len(names), size=wanted, replace=False)
    removed = tuple(names[index] for index in chosen[: spec.n_removed])
    jittered = tuple(names[index] for index in chosen[spec.n_removed :])

    if removed:
        scene.edit(remove=[obj.name for name in removed for obj in grouped[name]])

    for name in jittered:
        _jitter_building(grouped[name], spec, rng)

    return PerturbReport(removed=removed, jittered=jittered)


def _jitter_building(parts: list[Any], spec: PerturbSpec, rng: np.random.Generator) -> None:
    """Resize, nudge and turn one building, keeping its parts together.

    Height scales about the ground plane, and the horizontal offset and
    rotation are applied about the building's own centre, shared by every part.
    """
    import mitsuba as mi

    scale = float(rng.uniform(*spec.height_scale))
    offset_x = float(rng.uniform(-spec.position_m, spec.position_m))
    offset_y = float(rng.uniform(-spec.position_m, spec.position_m))
    angle = math.radians(float(rng.uniform(-spec.orientation_deg, spec.orientation_deg)))

    centroids = [np.asarray(part.position).ravel()[:3] for part in parts]
    centre = np.mean(centroids, axis=0)
    cos, sin = math.cos(angle), math.sin(angle)

    for part, centroid in zip(parts, centroids, strict=True):
        # Rotating about the shared centre rather than each part's own keeps
        # walls and roof aligned; scaling z about the ground keeps the base
        # planted. See the module docstring for the derivation.
        delta_x = float(centroid[0] - centre[0])
        delta_y = float(centroid[1] - centre[1])
        part.scaling = mi.Vector3f(1.0, 1.0, scale)
        part.orientation = mi.Point3f(angle, 0.0, 0.0)
        part.position = mi.Point3f(
            float(centre[0]) + offset_x + cos * delta_x - sin * delta_y,
            float(centre[1]) + offset_y + sin * delta_x + cos * delta_y,
            float(centroid[2]) * scale,
        )
