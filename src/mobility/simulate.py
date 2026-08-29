"""Run SUMO and collect one tidy UE trajectory frame.

The generation pipeline is three subprocesses of the resolved SUMO
installation, in order: ``randomTrips.py`` then ``duarouter``
(:mod:`src.mobility.demand`), then ``sumo`` itself with ``--fcd-output``
(this module). Nothing here steps the simulation from Python: the trajectory
is an open-loop input to the rest of the pipeline, nothing needs to influence
the simulation while it runs, and a Python round-trip per ``traci`` step would
buy nothing but cost. ``cfg.mobility.run.collector`` names the branch that
would (``"traci"``); it is a documented, unimplemented option, not a silent
gap.

``--fcd-output`` is written as CSV, not Parquet
--------------------------------------------------
SUMO 1.27.1's ``--output.format parquet`` writer is undocumented-experimental
and was verified here (on this platform) to crash on every run —
``IOError: Appending to file not implemented``, nonzero exit — independent of
``--device.fcd.period`` or ``--fcd-output.skip-empty``. CSV with the same
``--fcd-output.attributes`` was verified to exit cleanly and parse. Re-check
this against a newer SUMO release before switching back to Parquet; do not
assume the fix has landed.

Deliberately the MDT record, minus three columns
-------------------------------------------------------
The tidy frame this module produces is the minimal MDT record declared by
``configs/data.yaml``'s ``schema.mdt``, with ``gcell_id``, ``band_id`` and
``rsrp`` missing — exactly the columns the blocked ``src.radio.mdt`` stage
would add once the multi-band cell configuration arrives. Two names are resolved in favour of
``configs/data.yaml``'s schema, which is the contract ``src.data.schema``
actually enforces: ``date`` (not
``timestamp``) and ``gcell_id`` (not ``cell_id``, added by the MDT stage).
"""

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import scenario
from src.mobility import checks, demand, network, persist
from src.mobility import frame as frame_module

#: The SUMO FCD attributes :func:`tidy` requires, in the default
#: ``configs/mobility.yaml`` ``sampling.attributes`` order. This is a fixed
#: contract, not a generic mapping over whatever the config asks for:
#: reconfiguring ``sampling.attributes`` to drop one of these makes
#: :func:`tidy` raise a ``KeyError`` naming the missing SUMO column, and
#: extending :func:`tidy` to a new attribute means adding it here too — see
#: ``configs/mobility.yaml``'s comment on ``sampling.attributes``.
_REQUIRED_FCD_ATTRIBUTES = ("id", "x", "y", "speed", "angle", "type", "lane")


def run_sumo(cfg: DictConfig, routes: Path, out_dir: Path) -> Path:
    """Run ``sumo --fcd-output``, producing an FCD CSV file.

    Args:
        cfg: Composed config; uses ``cfg.mobility.run``, ``cfg.mobility.sampling``.
        routes: The ``.rou.xml`` from :func:`src.mobility.demand.build_routes`.
        out_dir: Directory to write ``fcd.csv`` into; created if missing.

    Returns:
        The path written.

    Raises:
        NotImplementedError: When ``cfg.mobility.run.collector`` is
            ``"traci"`` — see the module docstring.
        ValueError: For any other unrecognised ``collector`` value.
        subprocess.CalledProcessError: When ``sumo`` exits nonzero.

    Notes:
        ``--fcd-output.attributes`` is taken directly from
        ``cfg.mobility.sampling.attributes``. :func:`tidy` requires exactly
        :data:`_REQUIRED_FCD_ATTRIBUTES` to be among them; reconfiguring the
        list to drop one is a config error, not a silent narrowing — see
        :func:`read_fcd`.

    Example:
        >>> run_sumo(cfg, routes_path, out_dir)
        PosixPath('.../fcd.csv')
    """
    collector = cfg.mobility.run.collector
    if collector == "traci":
        raise NotImplementedError(
            "mobility.run.collector='traci' is not implemented. Nothing in "
            "this project currently needs to influence the SUMO simulation "
            "while it runs, so the 'fcd' collector (sumo --fcd-output) is the "
            "only one built. Implement this branch only once a scenario "
            "needs closed-loop control — see the module docstring."
        )
    if collector != "fcd":
        raise ValueError(
            f"Unknown mobility.run.collector={collector!r}; expected 'fcd' or 'traci'."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fcd_path = out_dir / "fcd.csv"

    run_cfg = cfg.mobility.run
    sampling = cfg.mobility.sampling
    arrival = cfg.mobility.arrival
    binary = network.sumo_binary("sumo")

    command = [
        binary,
        "--net-file",
        str(cfg.mobility.network.net_file),
        "--route-files",
        str(routes),
        "--begin",
        str(run_cfg.begin_s),
        "--end",
        str(run_cfg.end_s),
        "--step-length",
        str(run_cfg.step_length_s),
        "--seed",
        str(demand.seeds(cfg)["sumo"]),
        "--time-to-teleport",
        str(run_cfg.teleport_s),
        "--random-depart-offset",
        str(arrival.depart_offset_s),
        "--fcd-output",
        str(fcd_path),
        "--output.format",
        "csv",
        "--fcd-output.attributes",
        ",".join(sampling.attributes),
        "--device.fcd.period",
        str(sampling.interval_s),
        "--no-step-log",
        "--duration-log.disable",
    ]
    if sampling.skip_empty_timesteps:
        command.append("--fcd-output.skip-empty")

    network.run_command(command)
    if not fcd_path.is_file():
        raise RuntimeError(f"sumo did not write {fcd_path}")
    return fcd_path


def read_fcd(path: Path, cfg: DictConfig) -> pd.DataFrame:
    """Read the raw ``sumo --fcd-output`` CSV.

    Args:
        path: The file from :func:`run_sumo`.
        cfg: Composed config. Unused directly — kept for a uniform signature
            with the rest of this module's I/O functions and for a future
            check against ``cfg.mobility.sampling.attributes``.

    Returns:
        The raw frame, columns named ``<tag>_<attribute>`` (SUMO's own
        naming, e.g. ``timestep_time``, ``vehicle_id``, ``vehicle_x``) —
        untransformed and not yet in the scene coordinate frame. See
        :func:`tidy`.

    Raises:
        ValueError: When any of :data:`_REQUIRED_FCD_ATTRIBUTES` has no
            corresponding ``vehicle_<attribute>`` column in the file — either
            ``cfg.mobility.sampling.attributes`` was reconfigured to drop one,
            or SUMO's FCD column naming has changed.

    Notes:
        SUMO's CSV FCD output is semicolon-delimited, not comma-delimited.

    Example:
        >>> read_fcd(fcd_path, cfg).columns.tolist()
        ['timestep_time', 'vehicle_id', 'vehicle_x', ...]
    """
    raw = pd.read_csv(path, sep=";")
    missing = [attr for attr in _REQUIRED_FCD_ATTRIBUTES if f"vehicle_{attr}" not in raw.columns]
    if missing:
        raise ValueError(
            f"{path}: missing {[f'vehicle_{a}' for a in missing]}, required by "
            f"tidy(). Found columns: {sorted(raw.columns)}. Check that "
            "mobility.sampling.attributes still lists "
            f"{_REQUIRED_FCD_ATTRIBUTES}, or that SUMO's FCD column naming "
            "has not changed."
        )
    return raw


def tidy(
    raw: pd.DataFrame, cfg: DictConfig, sid: str, frame: frame_module.SceneFrame
) -> pd.DataFrame:
    """Turn the raw FCD frame into the project's trajectory schema.

    Args:
        raw: The frame from :func:`read_fcd`, in SUMO network-local
            coordinates.
        cfg: Composed config; uses ``cfg.mobility.ue.height_m`` and
            ``cfg.mobility.output.epoch``.
        sid: The scenario id every row is tagged with.
        frame: The transform from :func:`src.mobility.frame.derive_frame`,
            applied here so every column downstream of this function is
            already in the scene local frame. Clipping (also against
            ``frame``) happens separately, in
            :func:`src.mobility.frame.clip_to_scene`, on this function's
            output.

    Returns:
        A frame with columns ``scenario_id``, ``ue_id``, ``t``, ``date``,
        ``sim_x``, ``sim_y`` (scene local frame), ``ue_height``,
        ``speed_mps``, ``heading_deg``, ``edge_id``, ``vehicle_type`` —
        sorted by ``(ue_id, t)``.

    Notes:
        ``edge_id`` is derived from SUMO's lane id by dropping the trailing
        ``_<lane index>`` — kept (rather than dropped entirely) because
        :mod:`src.mobility.checks` needs it to look up a speed limit, and
        because a perturbed scenario varies speed via the vehicle type,
        which would otherwise have to be re-derived from the network later,
        at which point it could disagree with what SUMO actually simulated.

    Example:
        >>> tidy(raw, cfg, "scn_...", frame).columns.tolist()
        ['scenario_id', 'ue_id', 't', 'date', 'sim_x', 'sim_y', 'ue_height',
         'speed_mps', 'heading_deg', 'edge_id', 'vehicle_type']
    """
    out = pd.DataFrame(
        {
            "scenario_id": pd.Series([sid] * len(raw), dtype="string"),
            "ue_id": raw["vehicle_id"].astype("string"),
            "t": raw["timestep_time"].astype("float64"),
            "sim_x": raw["vehicle_x"].astype("float64"),
            "sim_y": raw["vehicle_y"].astype("float64"),
            "ue_height": float(cfg.mobility.ue.height_m),
            "speed_mps": raw["vehicle_speed"].astype("float64"),
            "heading_deg": raw["vehicle_angle"].astype("float64"),
            "edge_id": (
                raw["vehicle_lane"].astype("string").str.rsplit("_", n=1).str[0].astype("string")
            ),
            "vehicle_type": raw["vehicle_type"].astype("string"),
        }
    )
    epoch = pd.Timestamp(str(cfg.mobility.output.epoch))
    out["date"] = epoch + pd.to_timedelta(out["t"], unit="s")

    out = frame_module.to_scene_frame(out, frame)
    out = out.sort_values(["ue_id", "t"]).reset_index(drop=True)
    return out[
        [
            "scenario_id",
            "ue_id",
            "t",
            "date",
            "sim_x",
            "sim_y",
            "ue_height",
            "speed_mps",
            "heading_deg",
            "edge_id",
            "vehicle_type",
        ]
    ]


def generate(cfg: DictConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full mobility pipeline for one scenario: trips, routes, SUMO, clip.

    Args:
        cfg: Composed config; the single source of every parameter every
            stage below reads.

    Returns:
        ``(trajectories, manifest)`` — the clipped, scene-frame trajectory
        frame and the manifest :func:`src.mobility.persist.save_trajectories`
        writes beside it.

    Raises:
        ValueError: When ``cfg.mobility.ue.height_m`` does not equal
            ``cfg.radio.grid.height_m`` — the radio map is solved on a plane
            at that height, and a mismatch would be silently wrong (see
            ``configs/mobility.yaml``).

    Notes:
        Everything under ``out_dir`` except ``fcd.csv``'s tidy output and the
        manifest is intermediate SUMO input (trips, routes, the raw FCD file)
        and is only kept when ``cfg.mobility.output.keep_sumo_inputs`` is set.

    Example:
        >>> trajectories, manifest = generate(cfg)
    """
    ue_height = float(cfg.mobility.ue.height_m)
    grid_height = float(cfg.radio.grid.height_m)
    if ue_height != grid_height:
        raise ValueError(
            f"mobility.ue.height_m ({ue_height}) must equal radio.grid.height_m "
            f"({grid_height}) — a measurement taken at a different height than "
            "the radio map is solved at is not comparable to it."
        )

    frame = frame_module.derive_frame(cfg)
    fingerprint = scenario.scene_fingerprint(cfg)
    sid = scenario.scenario_id(cfg, fingerprint)

    sumo_dir = Path(cfg.mobility.output.dir) / f"scenario_id={sid}" / "sumo"
    trips = demand.build_trips(cfg, sumo_dir)
    routes = demand.build_routes(cfg, trips, sumo_dir / "routes.rou.xml")
    fcd_path = run_sumo(cfg, routes, sumo_dir)

    raw = read_fcd(fcd_path, cfg)
    tidied = tidy(raw, cfg, sid, frame)
    clipped, clip_stats = frame_module.clip_to_scene(tidied, frame, cfg)

    net = network.load_net(cfg)  # loaded once; both the summary and the checks need it
    net_summary = network.network_summary(net)
    check_results = checks.run_all(clipped, net, cfg)

    sumo_binary_path = network.sumo_binary("sumo")

    manifest = scenario.build_manifest(
        cfg,
        sid,
        sumo={
            "version": network.sumo_version(sumo_binary_path),
            "binary": sumo_binary_path,
        },
        seeds=demand.seeds(cfg),
        mobility=OmegaConf.to_container(cfg.mobility, resolve=True),
        frame={
            "dx": frame.dx,
            "dy": frame.dy,
            "net_offset": frame.net_offset,
            "proj_parameter": frame.proj_parameter,
            "scene_centre_lonlat": frame.scene_centre_lonlat,
            "utm_zone": frame.utm_zone,
            "scene_bounds": frame.bounds,
        },
        network={**net_summary, "net_file": str(cfg.mobility.network.net_file)},
        clipping=clip_stats,
        counts={
            "n_ue_requested": int(cfg.mobility.ue.count),
            "n_ue_observed": int(clipped["ue_id"].nunique()),
            "n_rows": int(len(clipped)),
            "departure_period_s": demand.departure_period_s(cfg),
            "sampling_interval_s": float(cfg.mobility.sampling.interval_s),
            "t_begin": float(cfg.mobility.run.begin_s),
            "t_end": float(cfg.mobility.run.end_s),
        },
        checks={name: len(violations) for name, violations in check_results.items()},
    )

    persist.save_trajectories(clipped, manifest, cfg)
    if not cfg.mobility.output.keep_sumo_inputs:
        persist.discard_sumo_inputs(sumo_dir)

    return clipped, manifest


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate one scenario's trajectories. Entry point for ``task mobility:generate``.

    Args:
        cfg: Composed config (Hydra).

    Example:
        $ task mobility:generate -- mobility.ue.count=20 mobility.run.end_s=300
    """
    trajectories, manifest = generate(cfg)
    print(f"scenario_id: {manifest['scenario_id']}")
    print(f"rows: {len(trajectories)}  UEs: {trajectories['ue_id'].nunique()}")
    print(f"clipped_fraction: {manifest['clipping']['clipped_fraction']:.1%}")


if __name__ == "__main__":
    main()
