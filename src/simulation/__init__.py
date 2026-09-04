"""Simulation environment: a perturbed scene, its UEs, its radio maps, its MDT.

Three stages, run in order, each a ``python -m`` entry point:

``scenario``
    Perturbs the delivered scene, rasters it, and draws the UE population over
    the result, once per interval across the horizon. Writes the UE table and a
    manifest.
``radio``
    Rebuilds that scenario, places the transmitters, and ray-traces one clean
    radio map per band. This artifact is the surrogate's label.
``mdt``
    Samples the radio map at the UE positions, adds measurement error and
    censors, producing what a UE would actually report.

The population moves over time; the map does not, and does not need to. Tilt
and geometry are fixed for the whole scenario, so an interval changes only
where the UEs stand, and every interval reads the same map. That is what makes
a hundred snapshots cost what one costs.

They are separate because the radio map is a function of tilt and must be
re-solved for every tilt configuration, while the geometry and the UE positions
must *not* move when tilt does. Fused into one run, every tilt change would
redraw the UEs and the KPIs would stop being a function of tilt — the property
the whole optimization rests on.

Supporting modules, each with one reason to change:

``scene``
    Loads the scene, reads its extent, and reduces the geometry to the surface
    height above any ``(x, y)``. The only module that touches ``sionna.rt`` or
    ``mitsuba`` directly for geometry.
``materials``
    Frequency-static radio materials and their perturbation, replacing the ITU
    ones that forbid sub-GHz carriers and silently discard perturbations.
``perturb``
    Buildings removed, resized, nudged and turned — this scenario's errors
    about the real city.
``grid``
    Square cells over the scene, and the open-ground and building rasters that
    one ray-cast pass yields.
``density``
    Where the UE density puts its mass: a uniform background plus hotspots
    drawn where the surrounding building volume is greatest. Time-invariant.
``traffic``
    How much mass each component holds, interval by interval — a diurnal
    profile with an AR(1) term, so demand is correlated in time rather than
    redrawn from nothing at every snapshot.
``sample``
    Drawing UE positions from that density, and writing them.
``seeds``
    The named random streams, all offsets of the one configured seed.
``transmitter``
    The site layout and the transmitters built from it. Its own entry point
    generates the layout; that is a one-off, not part of the chain.

Settings cross the config boundary as frozen dataclasses with ``from_config``
constructors — the only places ``configs/simulation.yaml``'s key names are
spelled. Every other function takes just the settings it uses.

The scene's extent is deliberately absent from the config: it is read from the
loaded scene, because a restated bound does not raise when it drifts from the
geometry, it silently samples UEs off the scene. What the config does state is
``area.margin_m``, the inset from that extent. Everything placed on purpose —
UEs, hotspot centres, masts — stays inside it, because near the boundary there
is no geometry beyond the edge to block or reflect anything and the radio map
reads optimistically there. The grid and the map still span the full extent, so
energy arriving from the margin is not lost.
"""
