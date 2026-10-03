"""The synthetic-data generator: an independent workflow nothing else imports.

One ``python -m src.scenario.run`` entry point (``task simulation:scenario``)
rasters the scene named by ``scenario.scene_file``, draws the UE population over
it once per interval across the horizon, and lays the nodes and cells out on its
open ground. It writes the UE table, the cell table and the manifest to the
paths ``simulation.input`` names, where real data can stand in for them, and
its own record to ``scenario.output.record_file``. Its settings are
``configs/scenario.yaml``.

Modules, each with one reason to change:

``run``
    The stage, the scenario id, the manifest and the generator record.
``grid``
    Square tiles over the scene, and the open-ground and building rasters that
    one ray-cast pass yields. The radio maps are solved on the same tiles.
``density``
    Where the UE density puts its mass: a uniform background plus hotspots
    drawn where the surrounding building volume is greatest. Time-invariant.
``traffic``
    How much mass each component holds, interval by interval — a diurnal
    profile with an AR(1) term, so demand is correlated in time rather than
    redrawn from nothing at every snapshot.
``sample``
    Drawing UE positions from that density, and writing them.
``layout``
    The nodes and their cells, mounted on open ground, as the cell table.

Scene loading and the named random streams stay in :mod:`src.simulation`
(``scene``, ``seeds``), which the radio stage shares; the dependency runs from
here to there only.
"""
