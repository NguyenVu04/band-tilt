"""The tables: before against after, method against method, and how far to trust either."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats

from src.evaluation import maps
from src.evaluation.runs import Run
from src.kpi.capacity import CapacitySpec, covered, covered_best, finite, serve_intervals
from src.kpi.overlap import overlap_neighbors
from src.optim.methods.base import INIT
from src.optim.objective import (
    MAXIMISED,
    KpiVector,
    best_by_objective,
    evaluate_kpis,
    map_kpis,
    ue_kpis,
)
from src.utils.plotting import label as display_name

BETTER = "better"
WORSE = "worse"
UNCHANGED = "unchanged"
UNDEFINED = "undefined"

# What is reported over all bands at once. Best-server RSRP and SINR are left
# out: the strongest layer across bands is not a layer any UE is measured on.
NETWORK_KPIS = (
    "hole_rate",
    "weak_rate",
    "overlap_rate",
    "overlap_neighbor_mean",
    "ue_service_failure_rate",
    "estimated_throughput_p05_mbps",
    "estimated_throughput_p50_mbps",
    "estimated_throughput_mean_mbps",
)

# What is reported per band, each read off that band's layers alone.
BAND_KPIS = (
    "hole_rate",
    "weak_rate",
    "overlap_rate",
    "rsrp_p50_dbm",
    "rsrp_p05_dbm",
    "sinr_p50_db",
    "sinr_p05_db",
)

# The search traces: the network KPIs and the objective every method ranked by.
SEARCH_MEASURES = (*NETWORK_KPIS, "objective")


def direction(name: str) -> str:
    """Whether a measure is maximised or minimised."""
    return "maximise" if name in MAXIMISED else "minimise"


def _verdict(name: str, delta: float) -> str:
    """Better or worse by the sign of the delta, in that KPI's direction.

    An exactly zero delta is ``unchanged``: it means the same measurement, not a
    small one. Anything else is reported at face value, so a reader judges the
    size of a move from the delta itself. A NaN delta is ``undefined``: a
    percentile over no covered tile is infinite on both sides, and
    ``inf - inf`` carries no direction.
    """
    if np.isnan(delta):
        return UNDEFINED
    if delta == 0.0:
        return UNCHANGED
    improved = delta > 0 if direction(name) == "maximise" else delta < 0
    return BETTER if improved else WORSE


def _interval(values: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, sample standard deviation and the 95 % Student-t interval of the mean.

    The spread and the interval are NaN below two samples.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan,) * 4
    mean = float(values.mean())
    if values.size < 2:
        return mean, np.nan, np.nan, np.nan
    std = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, values.size - 1)) * std / np.sqrt(values.size)
    return mean, std, mean - half, mean + half


def delta_table(before: KpiVector, after: KpiVector) -> pd.DataFrame:
    """Before, after and the verdict for each measure, in reporting order.

    Returns:
        Columns ``kpi``, ``direction``, ``before``, ``after``, ``delta``,
        ``verdict``, in :data:`NETWORK_KPIS` order.
    """
    rows = []
    for name in NETWORK_KPIS:
        start, end = getattr(before, name), getattr(after, name)
        rows.append(
            {
                "kpi": name,
                "direction": direction(name),
                "before": start,
                "after": end,
                "delta": end - start,
                "verdict": _verdict(name, end - start),
            }
        )
    return pd.DataFrame(rows)


def improvement_table(before: KpiVector, after: KpiVector) -> pd.DataFrame:
    """:func:`delta_table` plus ``improvement_pct``, the change relative to ``before``.

    ``improvement_pct`` is positive when the measure moved in its better
    direction, and NaN where ``before`` is zero, as in :func:`relative_improvement`.
    """
    table = delta_table(before, after)
    sign = np.where(table["kpi"].isin(MAXIMISED), 1.0, -1.0)
    base = table["before"].abs().replace(0.0, np.nan)
    table["improvement_pct"] = 100.0 * sign * table["delta"] / base
    return table


def seed_summary(runs: list[Run]) -> pd.DataFrame:
    """Each method's winners, summarised over its seeds, against the incumbent.

    Every run measures the same incumbent under the same solver seed, so the
    first run's stands for all. The verdict reads the sign of the mean delta,
    and the spread beside it says how far to trust one.

    Returns:
        One row per method and :data:`SEARCH_MEASURES` entry (the network KPIs,
        then the objective): ``method``, ``kpi``,
        ``direction``, ``n_seeds``, ``incumbent``, ``mean``, ``std``,
        ``ci95_low``, ``ci95_high``, ``mean_delta``, ``verdict``.

    Raises:
        ValueError: When there are no runs.
    """
    if not runs:
        raise ValueError("no runs to summarise")
    incumbent = runs[0].incumbent_kpi
    before = incumbent.as_dict()

    rows = []
    for method in dict.fromkeys(run.method for run in runs):
        mine = [run for run in runs if run.method == method]
        values = {name: [getattr(run.best_kpi, name) for run in mine] for name in SEARCH_MEASURES}
        for name, series in values.items():
            mean, std, low, high = _interval(np.asarray(series))
            delta = mean - before[name]
            rows.append(
                {
                    "method": method,
                    "kpi": name,
                    "direction": direction(name),
                    "n_seeds": len(mine),
                    "incumbent": before[name],
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_delta": delta,
                    "verdict": _verdict(name, delta),
                }
            )
    return pd.DataFrame(rows)


def winner_vs_candidates(runs: list[Run]) -> pd.DataFrame:
    """How far each winner stands above what its own search measured.

    A search earns credit for the gap between its winner and a typical
    candidate, not for the gap to the incumbent: when the median candidate
    already beats the incumbent by nearly as much, the incumbent was weak.

    Returns:
        One row per run, all objectives: ``method``, ``seed``,
        ``incumbent``, ``init_median`` (every Sobol design point, so for TuRBO
        it includes the designs its restarts draw; NaN when the run has none),
        ``candidate_median``, ``candidate_p90``
        and ``winner``, row 0 excluded from the candidates.
    """
    rows = []
    for run in runs:
        scores = run.history["objective"].to_numpy()
        candidates = scores[1:] if scores.size > 1 else scores
        init = scores[(run.history["phase"] == INIT).to_numpy()]
        rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "incumbent": float(scores[0]),
                "init_median": float(np.median(init)) if init.size else np.nan,
                "candidate_median": float(np.median(candidates)),
                "candidate_p90": float(np.quantile(candidates, 0.9)),
                "winner": float(scores.max()),
            }
        )
    return pd.DataFrame(rows)


def paired_method_gain(
    runs: list[Run], method: str = "turbo", reference: str = "random"
) -> pd.DataFrame:
    """Winner objective of ``method`` minus ``reference``, paired by seed.

    Paired because both methods share each seed's Sobol design. The Wilcoxon
    signed-rank test (``scipy.stats.wilcoxon``) assumes no normality; with few
    seeds its smallest attainable p-value is coarse, so the count of seeds
    ``method`` won sits beside it.

    Returns:
        One row: ``method``, ``reference``, ``n_pairs``, ``mean_gain``,
        ``ci95_low``, ``ci95_high``, ``method_better``, ``wilcoxon_p``.
    """
    best = {(run.method, run.seed): run.best_kpi.objective for run in runs}
    paired = sorted(seed for name, seed in best if name == method and (reference, seed) in best)
    gains = np.array([best[(method, seed)] - best[(reference, seed)] for seed in paired])
    mean, _, low, high = _interval(gains)
    p_value = float(stats.wilcoxon(gains).pvalue) if gains.size >= 2 and gains.any() else np.nan
    return pd.DataFrame(
        [
            {
                "method": method,
                "reference": reference,
                "n_pairs": int(gains.size),
                "mean_gain": mean,
                "ci95_low": low,
                "ci95_high": high,
                "method_better": int((gains > 0).sum()),
                "wilcoxon_p": p_value,
            }
        ]
    )


def method_table(runs: list[Run]) -> pd.DataFrame:
    """One row per run: what it found, and what it cost to find it.

    The methods are matched on evaluations, not on time, so both halves are shown.

    Returns:
        Columns for the run's identity and seed, its budget, its cost, the
        :data:`NETWORK_KPIS`, and how many of them moved each way against the
        incumbent.
    """
    rows = []
    for run in runs:
        deltas = delta_table(run.incumbent_kpi, run.best_kpi)
        wall = run.wall_clock_seconds
        rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "run": run.run_id,
                "evaluations": run.n_evaluations,
                "best_iteration": run.best_index,
                "ray_tracing_min": run.ray_tracing_seconds / 60.0,
                "wall_clock_min": wall / 60.0 if wall is not None else np.nan,
                **{name: getattr(run.best_kpi, name) for name in NETWORK_KPIS},
                "kpis_improved": int((deltas["verdict"] == BETTER).sum()),
                "kpis_worsened": int((deltas["verdict"] == WORSE).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_method(runs: list[Run]) -> Run:
    """The run whose winner has the highest objective; a tie keeps the earlier run.

    Raises:
        ValueError: When there are no runs.
    """
    if not runs:
        raise ValueError("no runs to choose between")
    return runs[best_by_objective([run.best_kpi for run in runs])]


def best_run_per_method(runs: list[Run]) -> dict[str, Run]:
    """Each method's highest-scoring run over its seeds, keyed by method."""
    methods = dict.fromkeys(run.method for run in runs)
    return {
        method: best_method([run for run in runs if run.method == method]) for method in methods
    }


def convergence(runs: list[Run]) -> pd.DataFrame:
    """Best value seen so far, per :data:`SEARCH_MEASURES` entry, per evaluation and run.

    Long form: ``method``, ``seed``, ``iteration``, ``kpi``, ``value``. Each
    measure accumulates in its own direction.
    """
    frames = []
    for run in runs:
        history = run.history
        for name in SEARCH_MEASURES:
            values = history[name]
            running = values.cummax() if direction(name) == "maximise" else values.cummin()
            frames.append(
                pd.DataFrame(
                    {
                        "method": run.method,
                        "seed": run.seed,
                        "iteration": history["iteration"].to_numpy(),
                        "kpi": name,
                        "value": running.to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tilt_movement(run: Run) -> pd.DataFrame:
    """How far the antennas moved, summarised per band.

    Reported only: nothing in the objective has seen these numbers.

    Returns:
        Columns ``band``, ``n_sectors``, ``n_moved``, ``mean_abs_delta_deg``,
        ``max_abs_delta_deg``, ``mean_delta_deg``.
    """
    table = run.best_tilt
    delta = table["delta_tilt_deg"]
    grouped = table.assign(abs_delta=delta.abs(), moved=delta.abs() > 1e-9).groupby(
        "band", observed=True
    )
    summary = grouped.agg(
        n_sectors=("sector", "size"),
        n_moved=("moved", "sum"),
        mean_abs_delta_deg=("abs_delta", "mean"),
        max_abs_delta_deg=("abs_delta", "max"),
        mean_delta_deg=("delta_tilt_deg", "mean"),
    )
    return summary.reset_index()


@dataclass(frozen=True)
class Configuration:
    """One configuration's radio map, and how the UEs are served on it.

    Attributes:
        rsrp: ``[n_band, n_tx, n_rows, n_cols]`` in dBm.
        sinr: The solver's SINR in dB, same shape.
        served: :func:`src.kpi.capacity.serve_intervals` output.
        demand: UE reports per tile over every interval, served or not: this
            is where the traffic is, not what was served.
    """

    rsrp: np.ndarray
    sinr: np.ndarray
    served: pd.DataFrame
    demand: np.ndarray


def configuration(
    archive: Mapping[str, np.ndarray], ue: pd.DataFrame, cfg: DictConfig
) -> Configuration:
    """Serve the UEs on one archived radio map, and count the reports per tile."""
    rsrp = archive["rsrp_dbm"].astype(float)
    sinr = archive["sinr_db"].astype(float)
    served = serve_intervals(rsrp, sinr, [str(b) for b in archive["band_label"]], ue, cfg)
    n_rows, n_cols = rsrp.shape[-2:]
    # int64 first: the processed UE table stores tiles as int16, and row * n_cols overflows it.
    flat = served["tile_row"].to_numpy(np.int64) * n_cols + served["tile_col"].to_numpy(np.int64)
    demand = np.bincount(flat, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    return Configuration(rsrp=rsrp, sinr=sinr, served=served, demand=demand)


def reproducibility(
    recorded: Mapping[str, KpiVector],
    configurations: Mapping[str, Configuration],
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """The KPIs recomputed from each archived map, against what the run recorded.

    A run's archived winner map is a second solve of the winning tilt at the
    same solver seed (:func:`src.optim.run.run`), stored as float32, so
    ``abs_gap`` is GPU ray-tracing non-determinism plus float32 rounding. A gap
    far beyond that means the archive is not the configuration that was scored.

    Returns:
        One row per configuration and :data:`NETWORK_KPIS` entry:
        ``configuration``, ``kpi``, ``recorded``, ``recomputed``, ``abs_gap``.
    """
    rows = []
    for name, kpi in recorded.items():
        config = configurations[name]
        again = evaluate_kpis(config.rsrp, config.sinr, band_labels, ue, cfg, served=config.served)
        for kpi_name in NETWORK_KPIS:
            gap = abs(getattr(again, kpi_name) - getattr(kpi, kpi_name))
            rows.append(
                {
                    "configuration": name,
                    "kpi": kpi_name,
                    "recorded": getattr(kpi, kpi_name),
                    "recomputed": getattr(again, kpi_name),
                    "abs_gap": gap,
                }
            )
    return pd.DataFrame(rows)


def sector_band_load(
    served: pd.DataFrame, band_labels: Sequence[str], tx_names: Sequence[str]
) -> pd.DataFrame:
    """UEs and their estimated throughput per sector-band for one configuration.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        band_labels: Band names, the ``band`` index order.
        tx_names: Sector names, the ``tx`` index order.

    Returns:
        One row per sector-band: ``sector``, ``band``, ``served_reports``,
        ``peak_ues`` (the most UEs sharing it in one interval), and the
        ``median_throughput_mbps`` and ``median_sinr_db`` of the UEs it served.
    """
    rows = []
    for b, band in enumerate(band_labels):
        for t, sector in enumerate(tx_names):
            mine = served[(served["band"] == b) & (served["tx"] == t)]
            rows.append(
                {
                    "sector": sector,
                    "band": band,
                    "served_reports": len(mine),
                    "peak_ues": int(mine.groupby("t_index").size().max()) if len(mine) else 0,
                    "median_throughput_mbps": float(mine["estimated_throughput_mbps"].median())
                    if len(mine)
                    else np.nan,
                    "median_sinr_db": float(mine["sinr_db"].median()) if len(mine) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def sector_impact(
    best_tilt: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    """Per sector-band of a recommended configuration: its tilt change and its load change.

    Args:
        best_tilt: The run's ``best_tilt`` table.
        before: :func:`sector_band_load` of the incumbent.
        after: :func:`sector_band_load` of the recommended configuration.
        sectors: One row per sector with ``sector``, ``node`` and ``azimuth_deg``.

    Returns:
        Sorted by the size of the traffic shift, largest first, so the sectors to
        watch after rollout lead.
    """
    columns = ["sector", "band", "served_reports", "median_throughput_mbps", "median_sinr_db"]
    impact = (
        best_tilt[["sector", "band", "current_tilt_deg", "optimized_tilt_deg", "delta_tilt_deg"]]
        .astype({"sector": str, "band": str})
        .merge(before[columns], on=["sector", "band"])
        .merge(after[columns], on=["sector", "band"], suffixes=("_before", "_after"))
        .merge(sectors[["sector", "node", "azimuth_deg"]].astype({"sector": str}), on="sector")
    )
    for name in ("served_reports", "median_throughput_mbps", "median_sinr_db"):
        impact[f"{name}_change"] = impact[f"{name}_after"] - impact[f"{name}_before"]
    leading = ["node", "sector", "azimuth_deg", "band"]
    impact = impact[leading + [c for c in impact.columns if c not in leading]]
    return impact.sort_values(
        "served_reports_change", key=np.abs, ascending=False, ignore_index=True
    )


def service_summary(served: pd.DataFrame, band_labels: Sequence[str]) -> dict[str, float]:
    """How one configuration serves the UE reports.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        band_labels: Band names, the ``band`` index order.

    Returns:
        ``reports`` (how many rows the shares are taken over),
        ``not_served_share`` (no layer above ``kpi.hole_dbm``), and
        ``share_<band>`` of all reports served on each band.
    """
    band = served["band"].to_numpy()
    summary = {
        "reports": float(len(served)),
        "not_served_share": float((band < 0).mean()),
    }
    for index, label in enumerate(band_labels):
        summary[f"share_{label}"] = float((band == index).mean())
    return summary


def coverage_comparison(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Several coverage tables side by side, keyed by label.

    Args:
        tables: Label to the frame :func:`src.evaluation.maps.coverage_table`
            returns, e.g. ``{"incumbent": ..., "turbo": ...}``.

    Returns:
        One row per coverage class, with a ``tile_share`` and a
        ``demand_share`` column per label.
    """
    merged = None
    for label, table in tables.items():
        part = table[["coverage", "tile_share", "demand_share"]].rename(
            columns={"tile_share": f"{label}_tile", "demand_share": f"{label}_demand"}
        )
        merged = part if merged is None else merged.merge(part, on="coverage")
    return merged if merged is not None else pd.DataFrame()


def coverage_by_area_and_demand(
    configurations: Mapping[str, Configuration], cfg: DictConfig
) -> pd.DataFrame:
    """:func:`coverage_comparison` of every configuration, with display column names.

    Every configuration is weighed by the ``incumbent`` entry's demand, so the
    weights do not move with the tilts.

    Raises:
        KeyError: When ``configurations`` has no ``incumbent``.
    """
    demand = configurations["incumbent"].demand
    coverage = coverage_comparison(
        {name: maps.coverage_table(c.rsrp, demand, cfg) for name, c in configurations.items()}
    )
    coverage.columns = ["coverage"] + [
        f"{display_name(key)}: {display_name(f'{share}_share')}"
        for key, share in (column.rsplit("_", 1) for column in coverage.columns[1:])
    ]
    return coverage


def experiment_setup(
    baseline: Mapping[str, np.ndarray], ue: pd.DataFrame, runs: list[Run], cfg: DictConfig
) -> pd.DataFrame:
    """The network, search space and budget the comparison ran on.

    Returns:
        Columns ``parameter`` and ``setting``, the setting as text.
    """
    n_rows, n_cols = maps.grid_shape(baseline)
    tile = float(baseline["tile_size_m"])
    tilt = runs[0].best_tilt
    current = sorted(tilt["current_tilt_deg"].unique())
    rows = [
        ("Scenario", baseline["scenario_id"]),
        ("Sectors", len(baseline["tx_name"])),
        ("Frequency bands", ", ".join(display_name(str(band)) for band in baseline["band_label"])),
        ("Decision variables (sector-band tilts)", len(tilt)),
        ("Evaluation area [m]", f"{n_cols * tile:g} x {n_rows * tile:g}"),
        ("Grid resolution [m]", f"{tile:g}"),
        ("Grid tiles", n_rows * n_cols),
        ("UE reports", len(ue)),
        ("Measurement intervals", ue["t_index"].nunique()),
        ("Tilt bounds [°]", f"{tilt['tilt_min_deg'].min():g} to {tilt['tilt_max_deg'].max():g}"),
        ("Current tilts [°]", ", ".join(f"{value:g}" for value in current)),
        ("Hole threshold [dBm]", f"{float(cfg.kpi.hole_dbm):g}"),
        ("Weak coverage upper bound [dBm]", f"{float(cfg.kpi.weak_dbm):g}"),
        ("Overlap margin [dB]", f"{float(cfg.kpi.overlap_margin_db):g}"),
        (
            "Usable PRB share [of max_prb]",
            f"{float(cfg.kpi.capacity.max_admission_utilisation):g}",
        ),
    ]
    for run in runs:
        settings = {k: v for k, v in run.meta["config"]["optim"]["method"].items() if k != "name"}
        rows.append(
            (
                f"{display_name(run.method)}, seed {run.seed}",
                f"{run.n_evaluations} evaluations; {settings}",
            )
        )
    return pd.DataFrame(
        [(name, str(value)) for name, value in rows], columns=["parameter", "setting"]
    )


def relative_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    """Percentage change of each method's mean winner against the incumbent, positive is better.

    Args:
        summary: :func:`seed_summary` output.

    Returns:
        One row per method, one column per KPI in :data:`NETWORK_KPIS`. NaN
        where the incumbent is zero.
    """
    summary = summary[summary["kpi"].isin(NETWORK_KPIS)]
    sign = np.where(summary["direction"] == "maximise", 1.0, -1.0)
    base = summary["incumbent"].abs().replace(0.0, np.nan)
    frame = summary.assign(improvement=sign * (summary["mean"] - summary["incumbent"]) / base * 100)
    wide = frame.pivot(index="method", columns="kpi", values="improvement")
    wide = wide.reindex(
        index=list(dict.fromkeys(summary["method"])), columns=list(dict.fromkeys(summary["kpi"]))
    )
    wide.columns.name = None
    return wide.reset_index()


def sample_efficiency(
    trace: pd.DataFrame,
    kpis: Sequence[str] = ("objective", "hole_rate", "overlap_rate"),
    budgets: Sequence[int] = (10, 25, 50, 100),
) -> pd.DataFrame:
    """Best value each method had reached after a fixed number of evaluations, mean over seeds.

    The incumbent is the first evaluation. The longest run's length is added to
    ``budgets``; a budget beyond a run's length is NaN for that run.

    Args:
        trace: :func:`convergence` output.
        kpis: Measures to report.
        budgets: Evaluation counts to read the running best at.

    Returns:
        Columns ``kpi``, ``budget``, then one per method.
    """
    lengths = trace.groupby(["method", "seed"])["iteration"].max() + 1
    budgets = sorted({*budgets, int(lengths.max())})
    rows = []
    for (method, seed, kpi), group in trace[trace["kpi"].isin(kpis)].groupby(
        ["method", "seed", "kpi"], sort=False
    ):
        values = group.set_index("iteration")["value"]
        for budget in budgets:
            reached = budget <= lengths[(method, seed)]
            rows.append(
                {
                    "kpi": kpi,
                    "budget": budget,
                    "method": method,
                    "value": float(values.loc[budget - 1]) if reached else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    wide = frame.groupby(["kpi", "budget", "method"], sort=False)["value"].mean().unstack("method")
    wide = wide.reindex(columns=list(dict.fromkeys(frame["method"])))
    wide = wide.reindex(pd.MultiIndex.from_product([list(kpis), budgets], names=["kpi", "budget"]))
    wide.columns.name = None
    return wide.reset_index()


def pareto_front(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Which rows no other row dominates, each column read in its own direction.

    A row dominates another when it is no worse on every column and strictly
    better on at least one, so identical rows do not dominate each other.

    Returns:
        Boolean mask over the rows.
    """
    # ponytail: O(n^2) pairwise comparison; fine for a few thousand candidates, sort-based if more.
    values = np.column_stack(
        [
            frame[column].to_numpy(float) * (-1.0 if direction(column) == "maximise" else 1.0)
            for column in columns
        ]
    )
    no_worse = (values[:, None, :] <= values[None, :, :]).all(axis=2)
    better = (values[:, None, :] < values[None, :, :]).any(axis=2)
    return ~(no_worse & better).any(axis=0)


def candidates(runs: list[Run]) -> pd.DataFrame:
    """Every configuration each run evaluated.

    Returns:
        Columns ``method``, ``seed``, ``iteration``, ``phase`` and :data:`SEARCH_MEASURES`.
    """
    frames = [
        run.history[["iteration", "phase", *SEARCH_MEASURES]].assign(
            method=run.method, seed=run.seed
        )
        for run in runs
    ]
    frame = pd.concat(frames, ignore_index=True)
    leading = ["method", "seed"]
    return frame[leading + [column for column in frame.columns if column not in leading]]


def overlap_neighbour_summary(
    configurations: Mapping[str, Configuration], cfg: DictConfig
) -> pd.DataFrame:
    """How many co-band neighbours overlap the serving sector, per configuration.

    Covered tiles are those whose strongest layer is above ``kpi.hole_dbm``. An
    uncovered tile has no neighbours by definition, so the covered mean is the
    one to compare.

    Returns:
        One row per configuration: ``mean_neighbours_covered``,
        ``mean_neighbours_all``, and the share of covered tiles with 0, 1, 2,
        and 3 or more overlapping neighbours.
    """
    rows = []
    for name, config in configurations.items():
        counts = overlap_neighbors(config.rsrp, cfg)
        on_covered = counts[covered(config.rsrp, cfg)]
        if on_covered.size == 0:
            on_covered = np.array([np.nan])
        rows.append(
            {
                "configuration": name,
                "mean_neighbours_covered": float(on_covered.mean()),
                "mean_neighbours_all": float(counts.mean()),
                "share_0_neighbours": float((on_covered == 0).mean()),
                "share_1_neighbours": float((on_covered == 1).mean()),
                "share_2_neighbours": float((on_covered == 2).mean()),
                "share_3plus_neighbours": float((on_covered >= 3).mean()),
            }
        )
    return pd.DataFrame(rows)


def band_layer_summary(
    configurations: Mapping[str, Configuration], band_labels: Sequence[str], cfg: DictConfig
) -> pd.DataFrame:
    """What each frequency layer covers and carries, per configuration.

    Returns:
        One row per configuration and band: ``coverage_share`` (tiles where the
        band's strongest sector is above ``kpi.hole_dbm``), ``mean_band_rsrp_dbm``
        over those tiles, ``serving_tile_share`` (tiles where a lone UE would be
        served on the band, :func:`src.evaluation.maps.serving_band`),
        ``served_share`` (UE reports served on the band, with every UE
        connected) and ``served_sinr_median_db`` of those reports.
    """
    rows = []
    for name, config in configurations.items():
        spec = CapacitySpec.from_config(cfg, band_labels, config.rsrp.shape[1])
        tile_band = maps.serving_band(config.rsrp, config.sinr, spec)
        strongest = finite(config.rsrp).max(axis=1)
        served_band = config.served["band"].to_numpy()
        for index, band in enumerate(band_labels):
            is_covered = covered_best(strongest[index], cfg)
            mine = served_band == index
            rows.append(
                {
                    "configuration": name,
                    "band": band,
                    "coverage_share": float(is_covered.mean()),
                    "mean_band_rsrp_dbm": float(strongest[index][is_covered].mean())
                    if is_covered.any()
                    else np.nan,
                    "serving_tile_share": float((tile_band == index).mean()),
                    "served_share": float(mine.mean()),
                    "served_sinr_median_db": float(config.served.loc[mine, "sinr_db"].median())
                    if mine.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# The whole-network row of :func:`band_kpis`, beside the per-band ones.
ALL_BANDS = "all"


def _band_view(config: Configuration, band: int) -> tuple[np.ndarray, np.ndarray]:
    """One band's slice of a configuration's maps.

    Slicing the band axis rather than reparameterising the KPIs is what keeps
    one definition of each measure: a per-band rate is the same function given
    one band's layers.
    """
    layer = slice(band, band + 1)
    return config.rsrp[layer], config.sinr[layer]


def band_kpis(
    configurations: Mapping[str, Configuration],
    band_labels: Sequence[str],
    cfg: DictConfig,
) -> pd.DataFrame:
    """The reported KPIs per configuration, over all bands and per band.

    The ``all`` row carries :data:`NETWORK_KPIS` and equals the run's own
    :class:`~src.optim.objective.KpiVector` on them. A band row carries
    :data:`BAND_KPIS`, each the same measure given only that band's layers, so
    a coverage hole on 700 MHz is a hole in the 700 MHz row whatever the other
    layers do. The rates over tiles do not sum across bands, because a tile can
    be a hole on two bands at once.

    The UE KPIs are NaN on band rows: a UE fails when no band reaches it and
    takes its throughput from whichever band it chose, so both belong to the
    network, not to one band. RSRP and SINR are NaN on the ``all`` row, as
    :data:`NETWORK_KPIS` says why.

    Returns:
        One row per configuration and band: ``configuration``, ``band``, then
        the union of :data:`NETWORK_KPIS` and :data:`BAND_KPIS`.
    """
    columns = list(dict.fromkeys([*NETWORK_KPIS, *BAND_KPIS]))
    rows = []
    for name, config in configurations.items():
        network = {**map_kpis(config.rsrp, config.sinr, cfg), **ue_kpis(config.served)}
        rows.append(
            {"configuration": name, "band": ALL_BANDS}
            | {key: network[key] if key in NETWORK_KPIS else np.nan for key in columns}
        )
        for index, band in enumerate(band_labels):
            layer = map_kpis(*_band_view(config, index), cfg)
            rows.append(
                {"configuration": name, "band": band}
                | {key: layer[key] if key in BAND_KPIS else np.nan for key in columns}
            )
    return pd.DataFrame(rows, columns=["configuration", "band", *columns])
