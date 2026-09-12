"""Predict the radio map after a tilt change, instead of ray tracing it.

**Parked, and not part of the pipeline.** It was built when ray tracing was
believed to cost 30-40 s per candidate. Measured against a warm kernel cache it
costs 8.3 s, and ``src/kpi/`` needs 0.66 s of that to score whatever map it is
given -- a floor no predictor can go under, which caps any surrogate at a 12.6x
speedup over simply ray tracing. The optimizer therefore ray-traces every
candidate. See ``outputs/fidelity_bench/surrogate_need_results.md``.

Kept because the measurement holds for one scene and one tilt-only search. A
second scene, a geometry that moves, or a search wide enough that 8.3 s a point
stops being affordable would all reopen the question.

What it does: learns the map a tilt change produces from the map before it, so
``src/kpi/`` can score that prediction with the same functions that score a
ray-traced map.

The design rests on one structural fact, verified rather than assumed by
:func:`src.surrogate.dataset.check_decomposition`: one transmitter's
orientation cannot alter another transmitter's field, so moving
``tilt[cell i, band b]`` moves only the slice ``rsrp[b, i]``. The 36-dimensional
problem is therefore 36 independent one-dimensional ones, the surrogate is a
per-slice operator applied in one batch, and a full tilt sweep costs *tilts*
solves rather than *tilts x cells*.

``features``
    A digital surface model, a signed distance field, the facade azimuth that
    field's gradient carries, the depression angle from each mast to each tile,
    and a line-of-sight fraction measured by casting to every sub-tile point.
``dataset``
    The tilt sweep, the vertical antenna pattern recovered from it by
    differencing, and the training pairs. Scope is one scene, so the split is
    on the tilt axis: whole degrees train, the values between them test.
``model``
    A residual over the analytic pattern re-embedding -- a factorized Fourier
    layer, a multi-level Haar wavelet layer and FiLM conditioning, per block.
    A zero residual is the analytic answer, so an untrained forward pass
    degrades to physics rather than to noise.
``train``
    Level, coverage and cycle-consistency losses; writes the checkpoint.
``evaluator``
    ``SurrogateEvaluator``, scoring a tilt vector into its own
    ``Prediction``. It no longer shares the optimizer's result type; reviving
    it in a search means reconciling the two deliberately.

**None of the scene channels depend on tilt.** They are a function of the
scene and the mast positions alone, so a scenario builds them once and
every tilt configuration the optimizer tries reads the same arrays. That is the
whole reason they are separated from anything the solver produces: a channel
that had to be rebuilt per evaluation would cost what it is meant to save. It is
also what keeps training free of Mitsuba, since ``dataset.build_features``
caches them to disk.

Two resolutions come back together. Facades are narrower than a grid tile, so
the surface model, the distance field and the azimuth are built on the
sub-tile grid ``simulation.grid.subsamples_per_tile`` already defines, while
the per-mast channels land on the tile grid the radio map is solved on. Both
are pinned to a :class:`src.simulation.grid.Raster`, so a channel cannot drift
half a tile away from the RSRP it will be trained against.

Materials are deliberately absent. They are available -- a scene object's name
carries its ITU material and :func:`src.simulation.materials.evaluate` turns
that into permittivity and conductivity -- but the scene's palette is small,
and its conductivities are separated mostly by whether a surface is metal.
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
