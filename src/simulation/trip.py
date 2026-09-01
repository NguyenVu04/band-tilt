"""Sample UE demand over the road network and run SUMO to record trajectories.

Two subprocesses of the resolved SUMO installation, in order: ``randomTrips.py``
samples origin/destination edge pairs and — through ``-r`` — has ``duarouter``
turn them into routes; then ``sumo`` replays those routes and writes a
floating-car-data trace.

Nothing here steps the simulation from Python. The trajectory is an open-loop
input to the rest of the pipeline, so a ``traci`` round trip per step would buy
nothing and cost a great deal.

Two choices that look arbitrary and are not:

FCD is written as CSV, not Parquet
    SUMO 1.27's ``--output.format parquet`` writer is undocumented-experimental
    and was recorded in this repository's history as crashing on every run on
    this platform (``IOError: Appending to file not implemented``), regardless
    of ``--device.fcd.period`` or ``--fcd-output.skip-empty``. CSV with the same
    attributes exits cleanly. Re-check against a newer SUMO release before
    switching; do not assume the fix landed.

The run configuration is written by SUMO itself
    :func:`write_sumocfg` hands the option vector to ``sumo -C`` rather than
    composing XML by hand, so ``scene.sumocfg`` is always valid for the SUMO
    version that will execute it, and the saved configuration cannot drift from
    what :func:`run_sumo` actually runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.simulation import network, toolchain


@dataclass(frozen=True)
class TripPaths:
    """Where the demand and trajectory artifacts are written.

    Attributes:
        trips: ``randomTrips.py``'s origin/destination pairs.
        routes: The routed demand ``sumo`` replays.
        sumocfg: The saved run configuration.
        fcd: The trajectory CSV, the one artifact downstream stages read.
    """

    trips: Path
    routes: Path
    sumocfg: Path
    fcd: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TripPaths:
        """Read the four paths under ``simulation.output``."""
        output = cfg.simulation.output
        return cls(
            trips=Path(output.trips_file),
            routes=Path(output.route_file),
            sumocfg=Path(output.sumocfg_file),
            fcd=Path(output.fcd_file),
        )

    def mkdirs(self) -> None:
        """Create the parent directory of every artifact."""
        for path in (self.trips, self.routes, self.sumocfg, self.fcd):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DemandSpec:
    """What demand to sample, and how to route it.

    ``weight_by_lanes`` and ``weight_by_length`` make busy roads likelier
    origins. That is a crude stand-in for an origin/destination matrix and shapes
    *where* the UEs are, so it is a modelling choice a report has to state rather
    than a nuisance parameter.
    """

    count: int
    vehicle_class: str
    fringe_factor: float
    min_distance_m: float
    max_distance_m: float
    weight_by_lanes: bool
    weight_by_length: bool
    remove_loops: bool
    threads: int
    seed: int

    def __post_init__(self) -> None:
        """Reject a demand nothing could be sampled from.

        Raises:
            ValueError: When ``count`` or ``threads`` is not positive.
        """
        if self.count <= 0:
            raise ValueError(f"simulation.ue.count must be positive, got {self.count}")
        if self.threads <= 0:
            raise ValueError(f"simulation.routing.threads must be positive, got {self.threads}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> DemandSpec:
        """Read ``simulation.ue``, ``simulation.routing`` and ``simulation.seed``."""
        routing = cfg.simulation.routing
        return cls(
            count=int(cfg.simulation.ue.count),
            vehicle_class=str(cfg.simulation.ue.vehicle_class),
            fringe_factor=float(routing.fringe_factor),
            min_distance_m=float(routing.min_distance_m),
            max_distance_m=float(routing.max_distance_m),
            weight_by_lanes=bool(routing.weight_by_lanes),
            weight_by_length=bool(routing.weight_by_length),
            remove_loops=bool(routing.remove_loops),
            threads=int(routing.threads),
            seed=int(cfg.simulation.seed),
        )


@dataclass(frozen=True)
class RunSpec:
    """How to run the simulation, and how often to sample it."""

    begin_s: float
    end_s: float
    step_length_s: float
    teleport_s: float
    interval_s: float
    attributes: tuple[str, ...]
    skip_empty_timesteps: bool
    seed: int

    def __post_init__(self) -> None:
        """Reject a run window nothing could happen in.

        Raises:
            ValueError: When the window or the sampling interval is not
                positive.
        """
        if self.end_s <= self.begin_s:
            raise ValueError(
                f"simulation.run.end_s ({self.end_s}) must be greater than "
                f"simulation.run.begin_s ({self.begin_s})"
            )
        if self.interval_s <= 0:
            raise ValueError(
                f"simulation.sampling.interval_s must be positive, got {self.interval_s}"
            )

    @property
    def duration_s(self) -> float:
        """The length of the run window, in seconds."""
        return self.end_s - self.begin_s

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RunSpec:
        """Read ``simulation.run``, ``simulation.sampling`` and ``simulation.seed``."""
        run = cfg.simulation.run
        sampling = cfg.simulation.sampling
        return cls(
            begin_s=float(run.begin_s),
            end_s=float(run.end_s),
            step_length_s=float(run.step_length_s),
            teleport_s=float(run.teleport_s),
            interval_s=float(sampling.interval_s),
            attributes=tuple(str(attribute) for attribute in sampling.attributes),
            skip_empty_timesteps=bool(sampling.skip_empty_timesteps),
            # An offset, not an independent draw, so the relationship between the
            # two subprocess seeds is itself reproducible. --random is never passed.
            seed=int(cfg.simulation.seed) + 1,
        )


def departure_period_s(demand: DemandSpec, run: RunSpec) -> float:
    """The insertion period implied by the UE count and the run window.

    SUMO's demand tools take a period between departures, not a population size;
    this is the conversion.
    """
    return run.duration_s / demand.count


def build_routes(net: Path, paths: TripPaths, demand: DemandSpec, run: RunSpec) -> Path:
    """Sample trips over ``net`` and route them, writing ``paths.routes``.

    One ``randomTrips.py`` subprocess: ``-r`` makes it invoke ``duarouter``
    itself, so the trips file and the route file come out of a single sampling
    pass. Returns ``paths.routes``.

    Raises:
        FileNotFoundError: When ``randomTrips.py`` is missing from the SUMO
            installation.
        RuntimeError: When ``randomTrips.py`` exits nonzero, or exits cleanly
            without writing the route file.

    Notes:
        ``--validate`` is what makes the route file hold the full
        ``demand.count``. Sampling checks the distance and loop constraints but
        never connectivity, so a fraction of the pairs have no route and
        ``duarouter`` drops them. Validation adds a pass that identifies the
        failures, resamples replacements, and rewrites the route file from a
        pool of pairs already known to connect
        (``randomTrips.py:1069-1078``) — so it costs several extra
        ``duarouter`` passes over the whole network, and this step is the slow
        part of the stage.

        The resulting UE population is drawn only from origin/destination pairs
        a car can actually drive between, which on this network means the main
        road component rather than its cul-de-sacs and service pockets.

        ``--threads`` must be passed explicitly even to request the default
        behaviour: ``randomTrips.py`` fills it in with half the CPU count when it
        is unset (``randomTrips.py:317``) and forwards anything above 1 to
        ``duarouter`` as ``--routing-threads``. Multithreaded ``duarouter`` is
        not bit-reproducible across runs, and this stage runs once per scenario,
        so ``demand.threads`` is 1 and reproducibility wins over wall-clock.

        Unroutable trips are dropped rather than fatal because
        ``randomTrips.py`` always passes ``duarouter --ignore-errors``
        (``randomTrips.py:982``); that is not something this function can turn
        off.
    """
    paths.mkdirs()

    command = [
        sys.executable,
        str(toolchain.tool("randomTrips.py")),
        "-n",
        str(net),
        "-o",
        str(paths.trips),
        "-r",
        str(paths.routes),
        "-b",
        repr(run.begin_s),
        "-e",
        repr(run.end_s),
        "-p",
        repr(departure_period_s(demand, run)),
        "-s",
        str(demand.seed),
        "--vehicle-class",
        demand.vehicle_class,
        "--prefix",
        "ue",
        "--fringe-factor",
        repr(demand.fringe_factor),
        "--min-distance",
        repr(demand.min_distance_m),
        "--max-distance",
        repr(demand.max_distance_m),
        "--threads",
        str(demand.threads),
        # Stated rather than left to randomTrips.py's own default, which is on:
        # whether every UE has a route is a property of the dataset, not
        # something to inherit from an upstream default that could change.
        "--validate",
    ]
    if demand.weight_by_lanes:
        command.append("--lanes")
    if demand.weight_by_length:
        command.append("--length")
    if demand.remove_loops:
        command.append("--remove-loops")

    toolchain.run(command)

    if not paths.routes.is_file():
        raise RuntimeError(f"randomTrips.py exited cleanly but did not write: {paths.routes}")
    return paths.routes


def write_sumocfg(net: Path, paths: TripPaths, run: RunSpec) -> Path:
    """Have SUMO save the run configuration to ``paths.sumocfg``.

    ``sumo --save-configuration`` writes the configuration and exits without
    simulating. ``--save-configuration.relative`` rewrites every path relative to
    the configuration's own directory, which is what lets a trajectory
    destination outside the scene directory be reached from inside it. Returns
    ``paths.sumocfg``.

    Raises:
        RuntimeError: When ``sumo`` exits nonzero, or exits cleanly without
            writing the configuration.
    """
    paths.mkdirs()

    toolchain.run(
        [
            str(toolchain.binary("sumo")),
            *_sumo_options(net, paths, run),
            "--save-configuration",
            str(paths.sumocfg),
            "--save-configuration.relative",
        ]
    )

    if not paths.sumocfg.is_file():
        raise RuntimeError(f"sumo exited cleanly but did not write: {paths.sumocfg}")
    return paths.sumocfg


def run_sumo(paths: TripPaths) -> Path:
    """Run the saved configuration, writing the trajectory CSV.

    Returns ``paths.fcd``. The options come from ``paths.sumocfg`` alone, so the
    run cannot diverge from the configuration on disk.

    Raises:
        RuntimeError: When ``sumo`` exits nonzero, or exits cleanly without
            writing the trajectory.
    """
    toolchain.run([str(toolchain.binary("sumo")), "-c", str(paths.sumocfg)])

    if not paths.fcd.is_file():
        raise RuntimeError(f"sumo exited cleanly but did not write: {paths.fcd}")
    return paths.fcd


def _sumo_options(net: Path, paths: TripPaths, run: RunSpec) -> list[str]:
    """The sumo option vector, built once so the saved config and the run agree.

    ``--time-to-teleport -1`` disables teleporting: a teleport moves a UE
    discontinuously and is an artefact in the trajectory, not traffic.
    """
    options = [
        "--net-file",
        str(net),
        "--route-files",
        str(paths.routes),
        "--begin",
        repr(run.begin_s),
        "--end",
        repr(run.end_s),
        "--step-length",
        repr(run.step_length_s),
        "--seed",
        str(run.seed),
        "--time-to-teleport",
        repr(run.teleport_s),
        "--fcd-output",
        str(paths.fcd),
        "--output.format",
        "csv",
        "--fcd-output.attributes",
        ",".join(run.attributes),
        "--device.fcd.period",
        repr(run.interval_s),
        "--no-step-log",
        "--duration-log.disable",
    ]
    if run.skip_empty_timesteps:
        options.append("--fcd-output.skip-empty")
    return options


def generate(cfg: DictConfig) -> TripPaths:
    """Run the whole mobility stage: network, demand, configuration, simulation.

    Builds whatever part of the road network is missing
    (:func:`src.simulation.network.build`), then samples demand, saves the run
    configuration and runs it. Returns the paths written.
    """
    net = network.build(cfg).net
    paths = TripPaths.from_config(cfg)
    demand = DemandSpec.from_config(cfg)
    run = RunSpec.from_config(cfg)

    build_routes(net, paths, demand, run)
    write_sumocfg(net, paths, run)
    run_sumo(paths)
    return paths


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the UE trajectories. Entry point for ``task simulation:trips``.

    Example:
        $ task simulation:trips -- simulation.ue.count=20 simulation.run.end_s=300
    """
    paths = generate(cfg)
    vehicles = paths.routes.read_text(encoding="utf-8").count("<vehicle ")
    with paths.fcd.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1

    print(f"routes:  {paths.routes}  ({vehicles} vehicles)")
    print(f"sumocfg: {paths.sumocfg}")
    print(f"fcd:     {paths.fcd}  ({rows} rows)")


if __name__ == "__main__":
    main()
