"""Trajectory invariants: speed limits, step continuity, arrival profile.

Every check here returns the violating rows, not a boolean. "Passed" or
"failed" throws away exactly the information a scenario report needs — how
many, how bad, and where — and that is precisely what a perturbed scenario's
mobility model has to be reported on. An empty frame is a
clean pass; a nonempty one is not automatically a defect in this module, since
a real road network has UEs that legitimately exceed a posted limit briefly or
arrive in bursts — it is something for a scenario report to look at and
explain, which is why :func:`src.mobility.simulate.generate` records only the
violation *counts* in the manifest rather than treating a nonempty result as
fatal.
"""

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.mobility import demand

#: A generous upper bound on any road vehicle's speed, used only to flag a
#: physically implausible jump between two consecutive samples of one UE
#: (a "teleport" — SUMO relocating a stuck vehicle, or a clipping artefact).
#: Not a KPI threshold and not scenario-tunable, so it stays a constant here
#: rather than a configs/mobility.yaml key: it is a sanity bound on the
#: simulation output, not a modelling choice a perturbed scenario varies.
_MAX_PLAUSIBLE_SPEED_MPS = 55.0  # ~198 km/h


def speed_within_limits(df: pd.DataFrame, net: Any, cfg: DictConfig) -> pd.DataFrame:
    """Flag samples whose recorded speed exceeds their edge's speed limit.

    Args:
        df: A trajectory frame with ``edge_id`` and ``speed_mps`` columns —
            :func:`src.mobility.simulate.tidy`'s output, or a subset of it.
        net: The network from :func:`src.mobility.network.load_net`, used to
            look up each edge's speed limit.
        cfg: Composed config. Unused directly — kept for a uniform signature
            with the other checks in this module.

    Returns:
        The rows of ``df`` where ``speed_mps`` exceeds the edge's speed limit
        (``net.getEdge(edge_id).getSpeed()``), with a ``speed_limit_mps``
        column appended. Empty when nothing violates it.

    Notes:
        An edge id absent from ``net`` is itself a defect worth surfacing —
        every ``edge_id`` in ``df`` was derived from this same network's lane
        ids in :func:`src.mobility.simulate.tidy` — so such rows are included
        in the result with ``speed_limit_mps`` as ``NaN`` rather than dropped.

    Example:
        >>> speed_within_limits(df, net, cfg).empty
        True
    """
    limits = {}
    for edge_id in df["edge_id"].unique():
        try:
            limits[edge_id] = net.getEdge(edge_id).getSpeed()
        except KeyError:
            limits[edge_id] = float("nan")
    speed_limit_mps = df["edge_id"].map(limits)
    violates = (df["speed_mps"] > speed_limit_mps) | speed_limit_mps.isna()
    out = df.loc[violates].copy()
    out["speed_limit_mps"] = speed_limit_mps.loc[violates]
    return out


def step_distance_is_consistent(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Flag consecutive samples of one UE implying an implausible speed.

    Args:
        df: A trajectory frame with ``ue_id``, ``t``, ``sim_x``, ``sim_y``
            columns, sorted by ``(ue_id, t)`` — :func:`src.mobility.
            simulate.tidy`'s output.
        cfg: Composed config. Unused directly — kept for a uniform signature
            with the other checks in this module.

    Returns:
        One row per consecutive-sample pair whose implied speed
        (straight-line distance / elapsed time) exceeds
        :data:`_MAX_PLAUSIBLE_SPEED_MPS`, with ``dt_s``, ``distance_m`` and
        ``implied_speed_mps`` columns appended. Empty when nothing violates
        it.

    Notes:
        This is a coarse sanity bound, not a precise reconstruction of SUMO's
        car-following model — its purpose is to catch discontinuities
        (:mod:`src.mobility.frame`'s clipping leaving a gap in one UE's
        trajectory, or SUMO relocating a stuck vehicle), not to validate
        physics.

    Example:
        >>> step_distance_is_consistent(df, cfg).empty
        True
    """
    ordered = df.sort_values(["ue_id", "t"])
    same_ue = ordered["ue_id"] == ordered["ue_id"].shift(1)
    dt = ordered["t"] - ordered["t"].shift(1)
    distance = (
        (ordered["sim_x"] - ordered["sim_x"].shift(1)) ** 2
        + (ordered["sim_y"] - ordered["sim_y"].shift(1)) ** 2
    ) ** 0.5
    implied_speed = distance / dt.replace(0.0, float("nan"))

    out = ordered.copy()
    out["dt_s"] = dt
    out["distance_m"] = distance
    out["implied_speed_mps"] = implied_speed
    violates = same_ue & (implied_speed > _MAX_PLAUSIBLE_SPEED_MPS)
    return out.loc[violates]


def arrivals_match_configuration(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Compare the observed UE arrival profile against the configured demand.

    Args:
        df: A trajectory frame with ``ue_id`` and ``t`` columns.
        cfg: Composed config; uses ``cfg.mobility.run``,
            ``cfg.mobility.sampling.interval_s`` and
            ``cfg.mobility.ue.count`` (via
            :func:`src.mobility.demand.departure_period_s`).

    Returns:
        One row per sampling-interval-sized time bin over the run window,
        with ``bin_start``, ``bin_end``, ``observed`` (UEs first seen in that
        bin) and ``expected`` (``interval_s / departure_period_s``) —
        restricted to bins where ``observed`` is more than 3x or less than
        1/3 of ``expected``. Empty when every bin is within that band.

    Notes:
        The tolerance band is wide on purpose: ``arrival.process`` in
        ``configs/mobility.yaml`` includes ``poisson`` and ``binomial``,
        which produce genuine bin-to-bin variance around the mean by design.
        This check is for a gross mismatch — an arrival process that is not
        behaving like the one configured at all — not a statistical test of
        the distribution.

    Example:
        >>> arrivals_match_configuration(df, cfg).empty
        True
    """
    run_cfg = cfg.mobility.run
    interval_s = float(cfg.mobility.sampling.interval_s)
    period_s = demand.departure_period_s(cfg)
    expected = interval_s / period_s if period_s > 0 else 0.0

    begin, end = float(run_cfg.begin_s), float(run_cfg.end_s)
    arrival_times = df.groupby("ue_id")["t"].min()

    n_bins = int((end - begin) // interval_s)
    bin_starts = pd.Series(range(n_bins)) * interval_s + begin
    rows = []
    for bin_start in bin_starts:
        bin_end = bin_start + interval_s
        observed = int(((arrival_times >= bin_start) & (arrival_times < bin_end)).sum())
        rows.append(
            {"bin_start": bin_start, "bin_end": bin_end, "observed": observed, "expected": expected}
        )
    result = pd.DataFrame(rows)
    if expected <= 0:
        return result.iloc[0:0]
    out_of_band = (result["observed"] > 3 * expected) | (result["observed"] < expected / 3)
    return result.loc[out_of_band].reset_index(drop=True)


def run_all(df: pd.DataFrame, net: Any, cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Run every check in this module and collect the results.

    Args:
        df: The trajectory frame to check —
            :func:`src.mobility.simulate.tidy`'s output, typically after
            :func:`src.mobility.frame.clip_to_scene`.
        net: The network from :func:`src.mobility.network.load_net`.
        cfg: Composed config.

    Returns:
        A mapping from each check's name to its violations frame.
        :func:`src.mobility.simulate.generate` records only the row counts in
        the scenario manifest; the frames themselves are for interactive use
        (notebook 01) rather than storage.

    Example:
        >>> {name: len(v) for name, v in run_all(df, net, cfg).items()}
        {'speed_within_limits': 0, 'step_distance_is_consistent': 0,
         'arrivals_match_configuration': 0}
    """
    return {
        "speed_within_limits": speed_within_limits(df, net, cfg),
        "step_distance_is_consistent": step_distance_is_consistent(df, cfg),
        "arrivals_match_configuration": arrivals_match_configuration(df, cfg),
    }
