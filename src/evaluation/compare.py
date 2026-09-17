"""The tables: before against after, method against method, and how far to trust either."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats

from src.evaluation.runs import Run
from src.kpi.capacity import prb_by_interval, serve_intervals
from src.optim.objective import (
    MAXIMISED,
    MEASURE_NAMES,
    KpiVector,
    best_by_score,
    evaluate_kpis,
    score,
    score_frame,
)

BETTER = "better"
WORSE = "worse"
UNCHANGED = "unchanged"

SCORE = "score"


def direction(name: str) -> str:
    """Whether a measure, or the objective score, is maximised or minimised."""
    return "maximise" if name in MAXIMISED or name == SCORE else "minimise"


def _verdict(name: str, delta: float) -> str:
    """Better or worse by the sign of the delta, in that KPI's direction.

    An exactly zero delta is ``unchanged``: it means the same measurement, not a
    small one. Anything else is reported at face value, so a reader judges the
    size of a move from the delta itself.
    """
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
    """Before, after and the verdict for each measure, in priority order.

    Returns:
        Columns ``kpi``, ``direction``, ``before``, ``after``, ``delta``,
        ``verdict``, in :data:`MEASURE_NAMES` order.
    """
    rows = []
    for name in MEASURE_NAMES:
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


def seed_summary(runs: list[Run], cfg: DictConfig) -> pd.DataFrame:
    """Each method's winners, summarised over its seeds, against the incumbent.

    Every run measures the same incumbent under the same solver seed, so the
    first run's stands for all. The verdict reads the sign of the mean delta,
    and the spread beside it says how far to trust one.

    Returns:
        One row per method and KPI, then ``score``: ``method``, ``kpi``,
        ``direction``, ``n_seeds``, ``incumbent``, ``mean``, ``std``,
        ``ci95_low``, ``ci95_high``, ``mean_delta``, ``verdict``.

    Raises:
        ValueError: When there are no runs.
    """
    if not runs:
        raise ValueError("no runs to summarise")
    incumbent = runs[0].incumbent_kpi
    before = {**incumbent.as_dict(), SCORE: float(score([incumbent], cfg)[0])}

    rows = []
    for method in dict.fromkeys(run.method for run in runs):
        mine = [run for run in runs if run.method == method]
        values = {name: [getattr(run.best_kpi, name) for run in mine] for name in MEASURE_NAMES}
        values[SCORE] = score([run.best_kpi for run in mine], cfg)
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


def winner_vs_candidates(runs: list[Run], cfg: DictConfig) -> pd.DataFrame:
    """How far each winner stands above what its own search measured.

    Read from the search history, so the UE-counted terms are over the MDT.

    A search earns credit for the gap between its winner and a typical
    candidate, not for the gap to the incumbent: when the median candidate
    already beats the incumbent by nearly as much, the incumbent was weak.

    Returns:
        One row per run, all selection scores: ``method``, ``seed``,
        ``incumbent``, ``init_median`` (the Sobol design random search and TuRBO
        share; NaN for the rule sweep), ``candidate_median``, ``candidate_p90``
        and ``winner``, row 0 excluded from the candidates.
    """
    rows = []
    for run in runs:
        scores = score_frame(run.history, cfg)
        candidates = scores[1:] if scores.size > 1 else scores
        init = scores[(run.history["phase"] == "init").to_numpy()]
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
    runs: list[Run], cfg: DictConfig, method: str = "turbo", reference: str = "random"
) -> pd.DataFrame:
    """Winner score of ``method`` minus ``reference``, paired by seed.

    Paired because both methods share each seed's Sobol design. The Wilcoxon
    signed-rank test (``scipy.stats.wilcoxon``) assumes no normality; with few
    seeds its smallest attainable p-value is coarse, so the count of seeds
    ``method`` won sits beside it.

    Returns:
        One row: ``method``, ``reference``, ``n_pairs``, ``mean_gain``,
        ``ci95_low``, ``ci95_high``, ``method_better``, ``wilcoxon_p``.
    """
    best = {(run.method, run.seed): float(score([run.best_kpi], cfg)[0]) for run in runs}
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


def method_table(runs: list[Run], cfg: DictConfig) -> pd.DataFrame:
    """One row per run: what it found, and what it cost to find it.

    The methods are matched on evaluations, not on time, so both halves are shown.

    Returns:
        Columns for the run's identity and seed, its budget, its cost, every
        KPI and the score, and how many KPIs moved each way against the
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
                **{name: getattr(run.best_kpi, name) for name in MEASURE_NAMES},
                SCORE: float(score([run.best_kpi], cfg)[0]),
                "kpis_improved": int((deltas["verdict"] == BETTER).sum()),
                "kpis_worsened": int((deltas["verdict"] == WORSE).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_method(runs: list[Run], cfg: DictConfig) -> Run:
    """The run whose winner has the highest objective score; a tie keeps the earlier run.

    Raises:
        ValueError: When there are no runs.
    """
    if not runs:
        raise ValueError("no runs to choose between")
    return runs[best_by_score([run.best_kpi for run in runs], cfg)]


def best_run_per_method(runs: list[Run], cfg: DictConfig) -> dict[str, Run]:
    """Each method's highest-scoring run over its seeds, keyed by method."""
    methods = dict.fromkeys(run.method for run in runs)
    return {
        method: best_method([run for run in runs if run.method == method], cfg)
        for method in methods
    }


def convergence(runs: list[Run], cfg: DictConfig) -> pd.DataFrame:
    """Best value seen so far, per measure and for the objective score, per evaluation and run.

    Long form: ``method``, ``seed``, ``iteration``, ``kpi``, ``value``. Each
    KPI accumulates in its own direction. A search trace: the UE-counted
    measures are over the MDT.
    """
    frames = []
    for run in runs:
        history = run.history
        series = {name: history[name] for name in MEASURE_NAMES}
        series[SCORE] = pd.Series(score_frame(history, cfg))
        for name, values in series.items():
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
        Columns ``band``, ``n_cells``, ``n_moved``, ``mean_abs_delta_deg``,
        ``max_abs_delta_deg``, ``mean_delta_deg``.
    """
    table = run.best_tilt
    delta = table["delta_tilt_deg"]
    grouped = table.assign(abs_delta=delta.abs(), moved=delta.abs() > 1e-9).groupby(
        "band", observed=True
    )
    summary = grouped.agg(
        n_cells=("cell", "size"),
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
        demand: PRBs required per tile in its busiest interval.
    """

    rsrp: np.ndarray
    sinr: np.ndarray
    served: pd.DataFrame
    demand: np.ndarray


def configuration(
    archive: Mapping[str, np.ndarray], ue: pd.DataFrame, cfg: DictConfig
) -> Configuration:
    """Serve the UEs on one archived radio map."""
    rsrp = archive["rsrp_dbm"].astype(float)
    sinr = archive["sinr_db"].astype(float)
    served = serve_intervals(rsrp, sinr, [str(b) for b in archive["band_label"]], ue, cfg)
    _, prb = prb_by_interval(
        served["t_index"].to_numpy(),
        served["tile_row"].to_numpy(),
        served["tile_col"].to_numpy(),
        served["prb_per_ue"].to_numpy(),
        rsrp.shape[-2:],
    )
    return Configuration(rsrp=rsrp, sinr=sinr, served=served, demand=prb.max(axis=0))


def reproducibility(
    recorded: Mapping[str, KpiVector],
    configurations: Mapping[str, Configuration],
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """The KPIs recomputed from each archived map, against what the run recorded.

    The map is the one the run scored, so the two readings are the same
    measurement twice and ``abs_gap`` is float round-off. Anything a reader can
    see at the printed precision means the archived map is not the map that was
    scored.

    Returns:
        One row per configuration and KPI: ``configuration``, ``kpi``,
        ``recorded``, ``recomputed``, ``abs_gap``.
    """
    rows = []
    for name, kpi in recorded.items():
        maps = configurations[name]
        again = evaluate_kpis(maps.rsrp, maps.sinr, band_labels, ue, cfg)
        for kpi_name in MEASURE_NAMES:
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


def cell_band_load(
    served: pd.DataFrame,
    band_labels: Sequence[str],
    tx_names: Sequence[str],
    max_prb: np.ndarray,
) -> pd.DataFrame:
    """PRB load and service per cell-band for one configuration.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        band_labels: Band names, the ``band`` index order.
        tx_names: Cell names, the ``tx`` index order.
        max_prb: ``[n_band, n_tx]`` limits, as ``CapacitySpec.max_prb``.

    Returns:
        One row per cell-band: ``cell``, ``band``, ``served_reports``,
        ``mean_prb`` (averaged over every interval, idle ones as zero),
        ``peak_prb``, ``max_prb``, ``peak_utilisation`` (``peak_prb`` over
        ``max_prb``) and ``median_sinr_db`` of the UEs it served. Only admitted
        UEs load a cell-band; a blocked UE's demand is on no row.
    """
    n_intervals = max(served["t_index"].nunique(), 1)
    admitted = served[served["band"] >= 0]
    rows = []
    for b, band in enumerate(band_labels):
        for t, cell in enumerate(tx_names):
            mine = admitted[(admitted["band"] == b) & (admitted["tx"] == t)]
            load = mine.groupby("t_index")["prb_per_ue"].sum()
            rows.append(
                {
                    "cell": cell,
                    "band": band,
                    "served_reports": len(mine),
                    "mean_prb": float(load.sum()) / n_intervals,
                    "peak_prb": float(load.max()) if len(load) else 0.0,
                    "max_prb": float(max_prb[b, t]),
                    "median_sinr_db": float(mine["sinr_db"].median()) if len(mine) else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    frame["peak_utilisation"] = frame["peak_prb"] / frame["max_prb"]
    return frame


def cell_impact(
    best_tilt: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Per cell-band of a recommended configuration: its tilt change and its load change.

    Args:
        best_tilt: The run's ``best_tilt`` table.
        before: :func:`cell_band_load` of the incumbent.
        after: :func:`cell_band_load` of the recommended configuration.
        cells: One row per cell with ``cell``, ``node`` and ``azimuth_deg``.

    Returns:
        Sorted by the size of the traffic shift, largest first, so the cells to
        watch after rollout lead.
    """
    columns = ["cell", "band", "served_reports", "peak_utilisation", "median_sinr_db"]
    impact = (
        best_tilt[["cell", "band", "current_tilt_deg", "optimized_tilt_deg", "delta_tilt_deg"]]
        .astype({"cell": str, "band": str})
        .merge(before[columns], on=["cell", "band"])
        .merge(after[columns], on=["cell", "band"], suffixes=("_before", "_after"))
        .merge(cells[["cell", "node", "azimuth_deg"]].astype({"cell": str}), on="cell")
    )
    for name in ("served_reports", "peak_utilisation", "median_sinr_db"):
        impact[f"{name}_change"] = impact[f"{name}_after"] - impact[f"{name}_before"]
    leading = ["node", "cell", "azimuth_deg", "band"]
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
        ``not_served_share`` (blocked by PRB limits, or no path), the 10th
        percentile and median SINR of served reports, their median PRBs per
        UE, and ``share_<band>`` of all reports served on each band.
    """
    band = served["band"].to_numpy()
    admitted = band >= 0
    sinr = served.loc[admitted, "sinr_db"]
    summary = {
        "reports": float(len(served)),
        "not_served_share": float((~admitted).mean()),
        "sinr_p10_db": float(sinr.quantile(0.1)),
        "sinr_median_db": float(sinr.median()),
        "prb_per_served_ue_median": float(served.loc[admitted, "prb_per_ue"].median()),
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


def summarise(runs: list[Run], cfg: DictConfig) -> str:
    """One line per run a reader can act on, including when not to act."""
    if not runs:
        return "No runs found."

    lines = []
    for run in runs:
        deltas = delta_table(run.incumbent_kpi, run.best_kpi)
        improved = int((deltas["verdict"] == BETTER).sum())
        worsened = int((deltas["verdict"] == WORSE).sum())
        if run.best_index == 0:
            verdict = "found nothing that beat the incumbent"
        elif improved == 0 and worsened == 0:
            verdict = "moved no KPI at all"
        else:
            verdict = f"improved {improved} KPI(s), worsened {worsened}"
        lines.append(f"  {run.label} (seed {run.seed}): {run.n_evaluations} evaluations, {verdict}")
    return "\n".join(lines)
