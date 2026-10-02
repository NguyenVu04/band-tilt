"""Sionna-RT side of the pipeline: loading scenes, and the radio maps.

One stage, a ``python -m`` entry point, follows the scenario stage
(:mod:`src.scenario`):

``radio``
    Reads the scenario's manifest and cell table, places the transmitters, and
    ray-traces one clean radio map per band.

The population moves over time; the map does not, and does not need to. Tilt
and geometry are fixed for the whole scenario, so an interval changes only
where the UEs stand, and every interval reads the same map. That is what makes
a hundred snapshots cost what one costs.

The two stages are separate because the radio map is a function of tilt and
must be re-solved for every tilt configuration, while the geometry and the UE
positions must *not* move when tilt does. Fused into one run, every tilt change
would redraw the UEs and the KPIs would stop being a function of tilt — the
property the whole optimization rests on.

Supporting modules, each with one reason to change:

``scene``
    Loads the scene, reads its extent, and reduces the geometry to the surface
    height above any ``(x, y)``. The only module that touches ``sionna.rt`` or
    ``mitsuba`` directly for geometry.
``materials``
    Frequency-static radio materials, replacing the ITU ones that forbid
    sub-GHz carriers.
``seeds``
    The named random streams, each hashed from the one configured seed and its name.
``transmitter``
    The transmitters built from the cell table, and the check that their masts
    still stand on open ground.

Settings cross the config boundary as frozen dataclasses with ``from_config``
constructors — the only places the key names of ``configs/simulation.yaml``
(Sionna-RT) and ``configs/scenario.yaml`` are spelled.

The scene's extent is deliberately absent from the config: it is read from the
loaded scene, because a restated bound does not raise when it drifts from the
geometry, it silently samples UEs off the scene. UEs, hotspot centres and masts
may be placed anywhere on the scene's open ground, edges included.
"""
