"""Scene geometry reduced to the raster channels a radio-map surrogate reads.

Ray tracing is too slow to sit inside a search loop, so a learned model is meant
to predict the RSRP map during optimization and ``src/kpi/`` is meant to score
that prediction with the same functions that score a ray-traced map. This
package builds the geometric half of that model's input.

``features``
    A digital surface model, a signed distance field, the facade azimuth that
    field's gradient carries, the depression angle from each mast to each tile,
    and a line-of-sight fraction measured by casting to every sub-tile point.

**None of these channels depend on tilt.** They are a function of the perturbed
scene and the mast positions alone, so a scenario builds them once and every
tilt configuration the optimizer tries reads the same arrays. That is the whole
reason they are separated from anything the solver produces: a channel that had
to be rebuilt per evaluation would cost what it is meant to save.

Two resolutions come back together. Facades are narrower than a grid tile, so
the surface model, the distance field and the azimuth are built on the
sub-tile grid ``simulation.grid.subsamples_per_tile`` already defines, while
the per-mast channels land on the tile grid the radio map is solved on. Both
are pinned to a :class:`src.simulation.grid.Raster`, so a channel cannot drift
half a tile away from the RSRP it will be trained against.

Materials are deliberately absent. They are available -- a scene object's name
carries its ITU material and :func:`src.simulation.materials.evaluate` turns
that into permittivity and conductivity -- but the scene's palette is small,
its conductivities are separated mostly by whether a surface is metal, and
``simulation.materials.perturbation`` moves them only by a modest multiplier.
Add the channel when a measured residual asks for it, not before.
"""

from src.surrogate.features import (
    SceneFeatures,
    build,
    elevation_deg,
    facade_azimuth,
    los_fraction,
    pool,
    signed_distance,
)

__all__ = [
    "SceneFeatures",
    "build",
    "elevation_deg",
    "facade_azimuth",
    "los_fraction",
    "pool",
    "signed_distance",
]
