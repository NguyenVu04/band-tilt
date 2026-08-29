"""UE demand: population, arrival process, and routes over the road network.

``randomTrips.py`` samples origin/destination edge pairs, weighted by lane
count and edge length; ``duarouter`` turns them into legal routes. Both run as
subprocesses of the resolved SUMO installation (:mod:`src.mobility.network`)
rather than being reimplemented against ``sumolib.net`` directly:
reimplementing route generation against a 47.7 MB network would duplicate
roughly 900 lines SUMO already owns, and it would produce demand that is not
comparable to anything else in the SUMO ecosystem — which matters because
the perturbed scenarios vary this demand and a report has to defend the
perturbation, not a bespoke generator's behaviour.

Three seeds, not one
-----------------------
``src.utils.seed.set_seed`` cannot reach a subprocess, so seeding here is
explicit. :func:`seeds` derives three distinct values from
``cfg.mobility.seed`` — one each for trip generation, routing, and the
simulation run — so that changing one stage's randomness does not silently
reshuffle another's. All three are recorded verbatim in the scenario manifest
by :mod:`src.mobility.persist`.

``randomTrips.py``'s own ``--validate`` defaults to ``True`` in SUMO 1.27 and,
if left on, runs an *internal* ``duarouter`` pass with its own seed handling —
which would silently bypass the explicit two-seed separation above.
:func:`build_trips` always passes ``--no-validate``; routing happens exactly
once, in :func:`build_routes`.
"""

import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from src.mobility import network


def seeds(cfg: DictConfig) -> dict[str, int]:
    """Derive the three subprocess seeds from ``cfg.mobility.seed``.

    Args:
        cfg: Composed config; uses ``cfg.mobility.seed``.

    Returns:
        ``{"trips": seed, "route": seed + 1, "sumo": seed + 2}``.

    Notes:
        Offsets rather than independent draws, so the relationship between the
        three is itself reproducible and legible in the manifest — changing
        the base seed shifts all three together rather than requiring three
        separately-tracked values.

    Example:
        >>> seeds(cfg)
        {'trips': 42, 'route': 43, 'sumo': 44}
    """
    base = int(cfg.mobility.seed)
    return {"trips": base, "route": base + 1, "sumo": base + 2}


def departure_period_s(cfg: DictConfig) -> float:
    """The randomTrips.py insertion period implied by the UE count and window.

    Args:
        cfg: Composed config; uses ``cfg.mobility.run.begin_s``/``end_s`` and
            ``cfg.mobility.ue.count``.

    Returns:
        ``(end_s - begin_s) / count``, in seconds — SUMO's demand tools take
        an insertion *period*, not a UE count, so this is the conversion.

    Raises:
        ValueError: When ``ue.count`` is not positive, or the run window is
            not positive.

    Example:
        >>> departure_period_s(cfg)
        18.0
    """
    run_cfg = cfg.mobility.run
    count = int(cfg.mobility.ue.count)
    if count <= 0:
        raise ValueError(f"mobility.ue.count must be positive, got {count}")
    span = float(run_cfg.end_s) - float(run_cfg.begin_s)
    if span <= 0:
        raise ValueError(
            f"mobility.run.end_s ({run_cfg.end_s}) must be greater than "
            f"mobility.run.begin_s ({run_cfg.begin_s})"
        )
    return span / count


def build_trips(cfg: DictConfig, out_dir: Path) -> Path:
    """Run ``randomTrips.py``, producing a ``.trips.xml`` file.

    Args:
        cfg: Composed config; uses ``cfg.mobility.network.net_file``,
            ``cfg.mobility.run``, ``cfg.mobility.ue``, ``cfg.mobility.routing``.
        out_dir: Directory to write ``trips.trips.xml`` into; created if
            missing.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: When ``randomTrips.py`` is not found under
            ``$SUMO_HOME/tools`` — check the SUMO installation.
        subprocess.CalledProcessError: When ``randomTrips.py`` exits nonzero.

    Notes:
        ``cfg.mobility.arrival.process`` (``uniform``/``poisson``/``binomial``/
        ``random``) is applied here, as a ``randomTrips.py`` flag — it shapes
        *when* trips depart, same as ``departure_period_s`` shapes *how
        many*. ``depart_offset_s`` is the one arrival parameter that is not:
        it is SUMO's own ``--random-depart-offset``, a run-time option
        applied by :func:`src.mobility.simulate.run_sumo` rather than at trip
        generation.

    Example:
        >>> build_trips(cfg, Path("data/interim/trajectories/scn_.../sumo"))
        PosixPath('.../trips.trips.xml')
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trips_path = out_dir / "trips.trips.xml"

    # network.py's import already ran _bootstrap_sumo_tools(), so SUMO_HOME is
    # set regardless of whether it came from the ambient environment or the
    # `sumo` extra wheel.
    script = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found; check the SUMO installation.")

    run_cfg = cfg.mobility.run
    routing = cfg.mobility.routing
    ue = cfg.mobility.ue
    arrival = cfg.mobility.arrival

    command = [
        sys.executable,
        str(script),
        "-n",
        str(cfg.mobility.network.net_file),
        "-o",
        str(trips_path),
        "-b",
        str(run_cfg.begin_s),
        "-e",
        str(run_cfg.end_s),
        "-p",
        str(departure_period_s(cfg)),
        "-s",
        str(seeds(cfg)["trips"]),
        "--vehicle-class",
        str(ue.vehicle_class),
        "--fringe-factor",
        str(routing.fringe_factor),
        "--min-distance",
        str(routing.min_distance_m),
        "--max-distance",
        str(routing.max_distance_m),
        "--prefix",
        "ue",
        "--threads",
        str(routing.threads),
        "--no-validate",  # routing happens once, explicitly, in build_routes()
    ]
    if routing.remove_loops:
        command.append("--remove-loops")
    if routing.weight_by_length:
        command.append("--length")
    if routing.weight_by_lanes:
        command.append("--lanes")

    if arrival.process == "uniform":
        pass  # the fixed -p period above is already a uniform arrival process
    elif arrival.process == "poisson":
        command.append("--poisson")
    elif arrival.process == "binomial":
        command += ["--binomial", str(arrival.binomial_n)]
    elif arrival.process == "random":
        command.append("--random-depart")
    else:
        raise ValueError(
            f"Unknown mobility.arrival.process={arrival.process!r}; expected "
            "one of 'uniform', 'poisson', 'binomial', 'random'."
        )

    network.run_command(command)
    if not trips_path.is_file():
        raise RuntimeError(f"randomTrips.py did not write {trips_path}")
    return trips_path


def build_routes(cfg: DictConfig, trips: Path, out: Path) -> Path:
    """Run ``duarouter``, turning trips into routes over the road network.

    Args:
        cfg: Composed config; uses ``cfg.mobility.network.net_file`` and
            ``cfg.mobility.routing``.
        trips: The ``.trips.xml`` from :func:`build_trips`.
        out: Destination for the ``.rou.xml`` file.

    Returns:
        The path written.

    Raises:
        subprocess.CalledProcessError: When ``duarouter`` exits nonzero — in
            particular, with ``routing.validate: true`` (the default), when
            any trip has no legal route. That is the correct failure: a
            sampled origin/destination pair that cannot be connected is a
            defect in the sampled demand, not something to drop silently.

    Notes:
        ``routing.validate`` controls strictness here, not
        ``randomTrips.py``'s own (disabled) ``--validate`` — see the module
        docstring. ``validate: false`` passes ``--ignore-errors`` so
        unroutable trips are dropped rather than failing the whole run.

    Example:
        >>> build_routes(cfg, trips_path, out_dir / "routes.rou.xml")
        PosixPath('.../routes.rou.xml')
    """
    out = Path(out)
    routing = cfg.mobility.routing
    binary = network.sumo_binary("duarouter")

    command = [
        binary,
        "--net-file",
        str(cfg.mobility.network.net_file),
        "--route-files",
        str(trips),
        "--output-file",
        str(out),
        "--seed",
        str(seeds(cfg)["route"]),
        "--routing-threads",
        str(routing.threads),
        "--no-step-log",
    ]
    if routing.remove_loops:
        command.append("--remove-loops")
    if not routing.validate:
        command.append("--ignore-errors")

    network.run_command(command)
    if not out.is_file():
        raise RuntimeError(f"duarouter did not write {out}")
    return out
