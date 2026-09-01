"""Simulation environment for UE mobility generation.

Four modules, each with one reason to change:

``toolchain``
    Locates and runs the installed SUMO. The only module that touches
    ``subprocess``, ``SUMO_HOME`` or ``sumolib``.
``frame``
    The Sionna-RT scene's coordinate frame, as netconvert options. The only
    module that imports ``pyproj``.
``network``
    Builds the SUMO road network for the study area from OpenStreetMap, offset
    onto the scene's local frame so network and scene coordinates coincide:
    OSM extract, then plain XML, then the compiled ``.net.xml``.
``trip``
    Samples UE demand over that network and runs SUMO to record trajectories.
    ``python -m src.simulation.trip`` runs the whole stage.

Settings cross the config boundary as frozen dataclasses with ``from_config``
constructors — the only places ``configs/simulation.yaml``'s key names are
spelled. Every other function takes just the settings it uses.
"""
