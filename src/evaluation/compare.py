"""The tables: before against after, method against method, how far tilts moved."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.evaluation.runs import Run
from src.optim.objective import KPI_NAMES, MAXIMISED, KpiVector, lexicographic_best, tolerances

# What a delta smaller than its KPI's tolerance is called. Naming it in the
# table rather than showing a bare signed number is the point: on this problem
# most deltas are noise, and a reader should not have to know the tolerances to
# see that.
TIE = "tie (within solver noise)"
BETTER = "better"
WORSE = "worse"


def direction(name: str) -> str:
    """Whether a KPI is maximised or minimised."""
    return "maximise" if name in MAXIMISED else "minimise"


def delta_table(before: KpiVector, after: KpiVector, cfg: DictConfig) -> pd.DataFrame:
    """Before, after and the verdict for each KPI, in priority order.

    The verdict compares the delta against that KPI's tolerance from
    ``configs/kpi.yaml``. Below it, the ray tracer cannot tell the two
    configurations apart and the honest answer is :data:`TIE` rather than a
    sign.

    Returns:
        Columns ``kpi``, ``direction``, ``before``, ``after``, ``delta``,
        ``tolerance``, ``verdict``. Row order is ADR 0001's priority order, so
        reading top to bottom is reading the order the winner was decided in.
    """
    tolerance = tolerances(cfg)
    rows = []
    for index, name in enumerate(KPI_NAMES):
        start, end = getattr(before, name), getattr(after, name)
        delta = end - start
        if abs(delta) <= tolerance[index]:
            verdict = TIE
        else:
            improved = delta > 0 if name in MAXIMISED else delta < 0
            verdict = BETTER if improved else WORSE
        rows.append(
            {
                "kpi": name,
                "direction": direction(name),
                "before": start,
                "after": end,
                "delta": delta,
                "tolerance": tolerance[index],
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def method_table(runs: list[Run], cfg: DictConfig) -> pd.DataFrame:
    """One row per run: what it found, and what it cost to find it.

    Both halves matter. The methods are matched on evaluations and not on time,
    so a table reporting only the KPIs would credit a method for spending
    longer, and one reporting only the time would miss that it spent it well.

    Returns:
        Columns for the run's identity, its budget, its cost, its five KPIs,
        and how many of them beat the incumbent by more than tolerance.
    """
    rows = []
    for run in runs:
        deltas = delta_table(run.incumbent_kpi, run.best_kpi, cfg)
        wall = run.wall_clock_seconds
        rows.append(
            {
                "method": run.method,
                "run": run.run_id,
                "evaluations": run.n_evaluations,
                "best_iteration": run.best_index,
                "on_pareto": int(run.history["on_pareto"].sum()),
                "ray_tracing_min": run.ray_tracing_seconds / 60.0,
                "wall_clock_min": wall / 60.0 if wall is not None else float("nan"),
                **{name: getattr(run.best_kpi, name) for name in KPI_NAMES},
                "kpis_improved": int((deltas["verdict"] == BETTER).sum()),
                "kpis_worsened": int((deltas["verdict"] == WORSE).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_method(runs: list[Run], cfg: DictConfig) -> Run:
    """The run whose winner the priority order prefers.

    One pass over the runs' winners, the same rule applied inside a run. A tie
    resolves to the earlier run, so a method only displaces another by actually
    beating it.

    Raises:
        ValueError: When there are no runs.
    """
    if not runs:
        raise ValueError("no runs to choose between")
    return runs[lexicographic_best([run.best_kpi for run in runs], cfg)]


def convergence(runs: list[Run]) -> pd.DataFrame:
    """Best value seen so far, per KPI, per evaluation, for every run.

    Long form — ``method``, ``run``, ``iteration``, ``kpi``, ``value`` — so one
    call feeds a faceted plot without reshaping. Each KPI runs in its own
    direction, so the cumulative reduction differs per column.
    """
    frames = []
    for run in runs:
        history = run.history
        for name in KPI_NAMES:
            series = history[name]
            running = series.cummax() if name in MAXIMISED else series.cummin()
            frames.append(
                pd.DataFrame(
                    {
                        "method": run.method,
                        "run": run.run_id,
                        "iteration": history["iteration"].to_numpy(),
                        "kpi": name,
                        "value": running.to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tilt_movement(run: Run) -> pd.DataFrame:
    """How far the antennas moved, summarised per band.

    Reported only. A penalty on movement is an explicit non-goal, so nothing in
    the objective has seen these numbers.

    Returns:
        Columns ``band``, ``n_cells``, ``n_moved``, ``mean_abs_delta_deg``,
        ``max_abs_delta_deg``, ``mean_delta_deg``. The signed mean says whether
        a band went up or down as a whole; the absolute mean says how much it
        moved regardless of direction.
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


def coverage_comparison(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Several coverage tables side by side, keyed by label.

    Args:
        tables: Label to the frame :func:`src.evaluation.maps.coverage_table`
            returns, e.g. ``{"incumbent": ..., "mobo": ...}``.

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
    """One paragraph a reader can act on, including when not to act.

    A run whose winner is its own incumbent, or whose every delta is a tie, has
    found nothing — and on a budget too small to cover 36 dimensions that is the
    expected outcome, not a bug. Saying so is more useful than a table of zeros.
    """
    if not runs:
        return "No runs found."

    lines = []
    for run in runs:
        deltas = delta_table(run.incumbent_kpi, run.best_kpi, cfg)
        improved = int((deltas["verdict"] == BETTER).sum())
        worsened = int((deltas["verdict"] == WORSE).sum())
        if run.best_index == 0:
            verdict = "found nothing that beat the incumbent"
        elif improved == 0 and worsened == 0:
            verdict = "moved every KPI by less than its tolerance"
        else:
            verdict = f"improved {improved} KPI(s), worsened {worsened}"
        lines.append(f"  {run.label}: {run.n_evaluations} evaluations, {verdict}")

    thin = [run.label for run in runs if run.n_evaluations < 40]
    note = ""
    if thin:
        note = (
            "\nNote: "
            + ", ".join(thin)
            + " evaluated fewer than 40 configurations over a 36-dimensional space. "
            "Read these as a check that the machinery runs, not as a comparison of methods."
        )
    return "\n".join(lines) + note
