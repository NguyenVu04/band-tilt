"""The tables: configuration against configuration, method against method, the Pareto front."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats

from src.data.load import grid_shape
from src.evaluation import maps
from src.evaluation.runs import Run
from src.kpi.capacity import covered, serve_intervals
from src.kpi.overlap import overlap_neighbor_mean_of, overlap_neighbors
from src.kpi.quality import LOW_PERCENTILE, MEDIAN_PERCENTILE
from src.optim.history import History
from src.optim.methods.base import INCUMBENT, INIT, SEARCH, SOBOL
from src.optim.objective import (
    BAND_KPI_NAMES,
    MAXIMISED,
    MEASURE_NAMES,
    OBJECTIVE_NAMES,
    KpiVector,
    evaluate_kpis,
    hypervolume,
    hypervolume_contributions,
    pareto_mask,
)
from src.optim.space import TiltSpace
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
    "estimated_throughput_p05_mbps",
    "estimated_throughput_p50_mbps",
    "estimated_throughput_mean_mbps",
)

# The evaluation's three KPIs, one each for coverage, co-band interference and
# capacity. Every front, pick and test below reads these, not the objectives.
EVALUATION_KPIS = ("coverage_rate", "separation_rate", "estimated_throughput_p50_mbps")

# Maximised besides MAXIMISED; a ``<name>_<band>`` column takes its name's direction.
_MAXIMISED = MAXIMISED | {"coverage_rate", "separation_rate", "served_share"}


def direction(name: str) -> str:
    """Whether a measure, or its ``<name>_<band>`` column, is maximised or minimised."""
    maximised = name in _MAXIMISED or any(name.startswith(f"{m}_b") for m in _MAXIMISED)
    return "maximise" if maximised else "minimise"


def _verdict(name: str, delta: float) -> str:
    """Better or worse by the sign of the delta, in that KPI's direction.

    An exactly zero delta is ``unchanged``: it means the same measurement, not a
    small one. A NaN delta is ``undefined``: a percentile over no covered tile
    is infinite on both sides, and ``inf - inf`` carries no direction.
    """
    if np.isnan(delta):
        return UNDEFINED
    if delta == 0.0:
        return UNCHANGED
    improved = delta > 0 if direction(name) == "maximise" else delta < 0
    return BETTER if improved else WORSE


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


def with_rates(history: pd.DataFrame) -> pd.DataFrame:
    """A copy with ``coverage_rate`` (1 - hole rate) and ``separation_rate`` (1 - overlap rate)."""
    return history.assign(
        coverage_rate=1.0 - history["hole_rate"], separation_rate=1.0 - history["overlap_rate"]
    )


def kpi_vector(row: pd.Series) -> KpiVector:
    """The measures of one history row."""
    return KpiVector.from_mapping(row[list(MEASURE_NAMES)].to_dict())


def shared_design(history: pd.DataFrame) -> np.ndarray:
    """Mask of the rows every method evaluates alike: the incumbent and the Sobol initial design."""
    phase, node = history["phase"], history["generation_node"]
    return ((phase == INCUMBENT) | ((phase == INIT) & (node == SOBOL))).to_numpy()


def pick(frame: pd.DataFrame, columns: Sequence[str] = EVALUATION_KPIS) -> int:
    """Row label of the largest hypervolume contribution on ``columns``; a tie keeps the earlier.

    Every column must be maximised: the hypervolume is against the origin
    (:func:`src.optim.objective.hypervolume`).
    """
    contributions = hypervolume_contributions(frame[list(columns)].to_numpy(float))
    return frame.index[int(np.argmax(contributions))]


def method_pick(frame: pd.DataFrame, method: str) -> int:
    """:func:`pick` over the configurations ``method`` proposed, the incumbent left out.

    The incumbent is every run's evaluation 0, not something a search found;
    left in, it wins whenever it alone holds the top of one KPI.

    Args:
        frame: :func:`candidates` output.
        method: The method whose evaluations to choose from.
    """
    return pick(frame[(frame["method"] == method) & (frame["phase"] != INCUMBENT)])


def hypervolume_table(runs: list[Run]) -> pd.DataFrame:
    """Hypervolume of each run on the search objectives and on :data:`EVALUATION_KPIS`.

    A search earns credit for what it adds to the shared initial design, not
    for the design itself, so the design's hypervolume sits beside the final one.

    Returns:
        One row per run and measure set (``objectives`` or ``kpis``):
        ``incumbent_hv``, ``initial_design_hv``, ``final_hv`` and ``pareto_points``.
    """
    rows = []
    for run in runs:
        history = with_rates(run.history)
        design = shared_design(history)
        for measures, columns in (("objectives", OBJECTIVE_NAMES), ("kpis", EVALUATION_KPIS)):
            values = history[list(columns)].to_numpy(float)
            rows.append(
                {
                    "method": run.method,
                    "measures": measures,
                    "incumbent_hv": hypervolume(values[:1]),
                    "initial_design_hv": hypervolume(values[design]),
                    "final_hv": hypervolume(values),
                    "pareto_points": int(pareto_mask(values).sum()),
                }
            )
    return pd.DataFrame(rows)


def convergence(runs: list[Run]) -> pd.DataFrame:
    """Hypervolume of every evaluation so far, on the search objectives.

    Long form: ``method``, ``iteration``, ``value``.
    """
    frames = []
    for run in runs:
        values = run.history[list(OBJECTIVE_NAMES)].to_numpy(float)
        frames.append(
            pd.DataFrame(
                {
                    "method": run.method,
                    "iteration": run.history["iteration"].to_numpy(),
                    "value": [hypervolume(values[: k + 1]) for k in range(len(values))],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def search_budget(runs: list[Run]) -> pd.DataFrame:
    """Whether the methods spent the same budget, from the same start, under the same noise.

    ``shared_design_identical`` compares the tilt vectors of the incumbent and
    the Sobol initial design across runs; ``shared_design_max_gap`` is the
    largest difference in any measure on them, which is GPU ray-tracing
    non-determinism when the tilts agree.

    Returns:
        One row per run: evaluations by phase, the search seed and the global
        seed the solver stream hashes from, the shared-design checks, and
        ray-tracing, wall-clock and overhead minutes.
    """
    if not runs:
        return pd.DataFrame()
    reference = runs[0].history[shared_design(runs[0].history)]
    tilts = [column for column in reference.columns if column.startswith("tilt_")]
    rows = []
    for run in runs:
        history = run.history
        phase, node = history["phase"], history["generation_node"]
        design = history[shared_design(history)]
        identical = design.shape == reference.shape and np.array_equal(
            design[tilts].to_numpy(), reference[tilts].to_numpy()
        )
        gap = np.abs(
            design[list(MEASURE_NAMES)].to_numpy(float)
            - reference[list(MEASURE_NAMES)].to_numpy(float)
        )
        rows.append(
            {
                "method": run.method,
                "run": run.run_id,
                "evaluations": run.n_evaluations,
                "incumbent": int((phase == INCUMBENT).sum()),
                "initial_design": int(((phase == INIT) & (node == SOBOL)).sum()),
                "restarts": int(((phase == INIT) & (node != SOBOL)).sum()),
                "search": int((phase == SEARCH).sum()),
                "seed": run.seed,
                "global_seed": int(run.meta["config"]["seed"]),
                "shared_design_identical": bool(identical),
                "shared_design_max_gap": float(np.nanmax(gap)) if identical else np.nan,
                "ray_tracing_min": run.ray_tracing_seconds / 60.0,
                "wall_clock_min": run.wall_clock_seconds / 60.0,
                "overhead_min": (run.wall_clock_seconds - run.ray_tracing_seconds) / 60.0,
            }
        )
    return pd.DataFrame(rows)


def candidates(runs: list[Run]) -> pd.DataFrame:
    """Every configuration each run evaluated, with the rates and ``shared`` marked.

    Returns:
        The history columns plus ``method``, ``coverage_rate``,
        ``separation_rate``, ``shared`` (:func:`shared_design`) and
        ``on_front``, whether the row is on its own run's
        :data:`EVALUATION_KPIS` Pareto front.
    """
    frames = []
    for run in runs:
        history = with_rates(run.history)
        values = history[list(EVALUATION_KPIS)].to_numpy(float)
        frames.append(
            history.assign(
                method=run.method, shared=shared_design(history), on_front=pareto_mask(values)
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame[["method", *(column for column in frame.columns if column != "method")]]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """``P(X > Y) - P(X < Y)`` over every pair, in ``[-1, 1]`` (Cliff, 1993)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sign(x[:, None] - y[None, :]).mean())


def method_tests(
    frame: pd.DataFrame, method: str = "morbo", reference: str = "random"
) -> pd.DataFrame:
    """Whether ``method``'s candidates score better than ``reference``'s, per KPI.

    The shared design is left out: it is the same configurations in both runs.
    The one-sided Mann-Whitney U test (``scipy.stats.mannwhitneyu``) asks
    whether a ``method`` candidate tends to score better, without assuming
    normality; Cliff's delta is its effect size, positive when ``method`` is
    better whichever way the KPI is read. ``frame`` is :func:`candidates` output.

    Returns:
        One row per :data:`EVALUATION_KPIS` entry: ``n_method``,
        ``n_reference``, both medians, ``mann_whitney_p`` and ``cliffs_delta``.
    """
    searched = frame[~frame["shared"]]
    rows = []
    for name in EVALUATION_KPIS:
        sign = 1.0 if direction(name) == "maximise" else -1.0
        x = searched.loc[searched["method"] == method, name].to_numpy(float)
        y = searched.loc[searched["method"] == reference, name].to_numpy(float)
        rows.append(
            {
                "kpi": name,
                "n_method": x.size,
                "n_reference": y.size,
                "median_method": float(np.median(x)) if x.size else np.nan,
                "median_reference": float(np.median(y)) if y.size else np.nan,
                "mann_whitney_p": float(
                    stats.mannwhitneyu(sign * x, sign * y, alternative="greater").pvalue
                )
                if x.size and y.size
                else np.nan,
                "cliffs_delta": cliffs_delta(sign * x, sign * y) if x.size and y.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def set_coverage(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the rows of ``b`` that some row of ``a`` weakly dominates, every column maximised.

    Zitzler and Thiele's C-metric (IEEE TEC 3(4), 1999). It is not symmetric,
    so a comparison reports both ``C(a, b)`` and ``C(b, a)``.
    """
    if len(b) == 0:
        return np.nan
    return float(
        (np.asarray(a)[:, None, :] >= np.asarray(b)[None, :, :]).all(axis=2).any(axis=0).mean()
    )


def front_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Each method's :data:`EVALUATION_KPIS` front against every other's, and against today.

    Args:
        frame: :func:`candidates` output.

    Returns:
        One row per ordered pair of methods: ``front_points``,
        ``c_metric`` (the share of the other front this front weakly dominates),
        and ``dominate_incumbent``, the candidates that beat the incumbent on
        every KPI, with their ``share`` of the method's candidates.
    """
    methods = list(dict.fromkeys(frame["method"]))
    fronts = {
        m: frame.loc[(frame["method"] == m) & frame["on_front"], list(EVALUATION_KPIS)].to_numpy(
            float
        )
        for m in methods
    }
    rows = []
    for method in methods:
        mine = frame[frame["method"] == method]
        incumbent = mine.loc[mine["phase"] == INCUMBENT, list(EVALUATION_KPIS)].to_numpy(float)[0]
        others = mine[mine["phase"] != INCUMBENT][list(EVALUATION_KPIS)].to_numpy(float)
        beats = ((others >= incumbent).all(axis=1) & (others > incumbent).any(axis=1)).sum()
        for other in methods:
            if other == method:
                continue
            rows.append(
                {
                    "method": method,
                    "reference": other,
                    "front_points": len(fronts[method]),
                    "c_metric": set_coverage(fronts[method], fronts[other]),
                    "dominate_incumbent": int(beats),
                    "dominate_incumbent_share": float(beats / len(others))
                    if len(others)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def band_columns(band_labels: Sequence[str]) -> list[str]:
    """The ``<name>_<band>`` history columns of :data:`src.optim.objective.BAND_KPI_NAMES`."""
    return [f"{name}_{band}" for name in BAND_KPI_NAMES for band in band_labels]


def pareto_tilts(runs: list[Run], band_labels: Sequence[str]) -> pd.DataFrame:
    """The combined :data:`EVALUATION_KPIS` Pareto front of every run, with its tilts.

    The shared design is taken from the first run only, so a configuration both
    runs evaluated appears once. Rows are ranked by hypervolume contribution on
    the front, largest first; ``delta_<kpi>`` is against the incumbent.

    Returns:
        One row per front point: ``rank``, ``method``, ``iteration``, ``phase``,
        the KPIs and their deltas, ``hv_contribution``, the per-band KPIs, then
        one ``tilt_<sector>_<band>`` column per decision variable.
    """
    frames = []
    for index, run in enumerate(runs):
        history = with_rates(run.history).assign(method=run.method)
        frames.append(history if index == 0 else history[~shared_design(history)])
    pool = pd.concat(frames, ignore_index=True)
    incumbent = pool.loc[pool["phase"] == INCUMBENT].iloc[0]
    front = pool[pareto_front(pool, EVALUATION_KPIS)].copy()
    front["hv_contribution"] = hypervolume_contributions(
        front[list(EVALUATION_KPIS)].to_numpy(float)
    )
    front = front.sort_values("hv_contribution", ascending=False, kind="stable")
    front.insert(0, "rank", np.arange(1, len(front) + 1))
    for name in EVALUATION_KPIS:
        front[f"delta_{name}"] = front[name] - incumbent[name]
    tilts = [column for column in front.columns if column.startswith("tilt_")]
    columns = [
        "rank",
        "method",
        "iteration",
        "phase",
        *EVALUATION_KPIS,
        *(f"delta_{name}" for name in EVALUATION_KPIS),
        "hv_contribution",
        *band_columns(band_labels),
        *tilts,
    ]
    return front[columns].reset_index(drop=True)


def tilt_table(row: pd.Series, space: TiltSpace) -> pd.DataFrame:
    """Current, proposed and delta tilt per sector-band of one history row."""
    return History(space).tilt_table(row[list(space.parameter_names)].to_numpy(float))


def band_table(rows: Mapping[str, pd.Series], band_labels: Sequence[str]) -> pd.DataFrame:
    """The per-band KPIs of the named history rows, one row per configuration and band.

    Returns:
        ``configuration``, ``band`` and one column per
        :data:`src.optim.objective.BAND_KPI_NAMES` entry.
    """
    records = [
        {"configuration": name, "band": band}
        | {kpi: float(row[f"{kpi}_{band}"]) for kpi in BAND_KPI_NAMES}
        for name, row in rows.items()
        for band in band_labels
    ]
    return pd.DataFrame(records, columns=["configuration", "band", *BAND_KPI_NAMES])


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
    radio: Mapping[str, np.ndarray], ue: pd.DataFrame, cfg: DictConfig
) -> Configuration:
    """Serve the UEs on one radio map, and count the reports per tile."""
    rsrp = radio["rsrp_dbm"].astype(float)
    sinr = radio["sinr_db"].astype(float)
    served = serve_intervals(rsrp, sinr, [str(b) for b in radio["band_label"]], ue, cfg)
    n_rows, n_cols = rsrp.shape[-2:]
    # int64 first: the processed UE table stores tiles as int16, and row * n_cols overflows it.
    flat = served["tile_row"].to_numpy(np.int64) * n_cols + served["tile_col"].to_numpy(np.int64)
    demand = np.bincount(flat, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    return Configuration(rsrp=rsrp, sinr=sinr, served=served, demand=demand)


def reproducibility(
    recorded: Mapping[str, pd.Series],
    configurations: Mapping[str, Configuration],
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """The measures of each re-traced configuration, against what its run recorded.

    The re-trace uses the search's solver seed, but GPU ray tracing is not
    bit-reproducible, so ``abs_gap`` is that non-determinism plus float32
    rounding. A gap far beyond it means the re-trace is not the configuration
    that was scored. ``recorded`` maps each configuration name to its history row.

    Returns:
        One row per configuration and measure: ``configuration``, ``kpi``,
        ``recorded``, ``recomputed``, ``abs_gap``.
    """
    rows = []
    for name, row in recorded.items():
        config = configurations[name]
        again = evaluate_kpis(config.rsrp, config.sinr, band_labels, ue, cfg, served=config.served)
        for kpi in MEASURE_NAMES:
            rows.append(
                {
                    "configuration": name,
                    "kpi": kpi,
                    "recorded": float(row[kpi]),
                    "recomputed": getattr(again, kpi),
                    "abs_gap": abs(getattr(again, kpi) - float(row[kpi])),
                }
            )
    return pd.DataFrame(rows)


def interval_throughput(configurations: Mapping[str, Configuration]) -> pd.DataFrame:
    """Estimated throughput per measurement interval, with the UEs in it.

    Every report counts, an unserved one at 0 Mbps, as in the UE KPIs.

    Returns:
        One row per configuration and interval: ``t_index``, ``ues``,
        ``throughput_p05_mbps``, ``throughput_p50_mbps``, ``throughput_mean_mbps``.
    """
    frames = []
    for name, config in configurations.items():
        grouped = config.served.groupby("t_index")["estimated_throughput_mbps"]
        frames.append(
            pd.DataFrame(
                {
                    "configuration": name,
                    "ues": grouped.size(),
                    "throughput_p05_mbps": grouped.quantile(LOW_PERCENTILE / 100.0),
                    "throughput_p50_mbps": grouped.quantile(MEDIAN_PERCENTILE / 100.0),
                    "throughput_mean_mbps": grouped.mean(),
                }
            ).reset_index()
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame[["configuration", *(c for c in frame.columns if c != "configuration")]]


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


def service_summary(served: pd.DataFrame, band_labels: Sequence[str]) -> dict[str, float]:
    """How one configuration serves the UE reports.

    Args:
        served: :func:`src.kpi.capacity.serve_intervals` output.
        band_labels: Band names, the ``band`` index order.

    Returns:
        ``reports`` (how many rows the shares are taken over) and
        ``share_<band>`` of all reports served on each band.
    """
    band = served["band"].to_numpy()
    summary = {"reports": float(len(served))}
    for index, label in enumerate(band_labels):
        summary[f"share_{label}"] = float((band == index).mean())
    return summary


def coverage_comparison(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Several coverage tables side by side, keyed by label.

    Args:
        tables: Label to the frame :func:`src.evaluation.maps.coverage_table`
            returns, e.g. ``{"incumbent": ..., "morbo": ...}``.

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
    radio: Mapping[str, np.ndarray],
    ue: pd.DataFrame,
    runs: list[Run],
    space: TiltSpace,
    cfg: DictConfig,
) -> pd.DataFrame:
    """The network, search space and budget the comparison ran on; ``radio`` supplies the grid.

    Returns:
        Columns ``parameter`` and ``setting``, the setting as text.
    """
    n_rows, n_cols = grid_shape(radio)
    tile = float(radio["tile_size_m"])
    rows = [
        ("Scenario", radio["scenario_id"]),
        ("Sectors", len(space.sectors)),
        ("Frequency bands", ", ".join(display_name(band) for band in space.band_names)),
        ("Decision variables (sector-band tilts)", space.n_dim),
        ("Evaluation area [m]", f"{n_cols * tile:g} x {n_rows * tile:g}"),
        ("Grid resolution [m]", f"{tile:g}"),
        ("Grid tiles", n_rows * n_cols),
        ("UE reports", len(ue)),
        ("Measurement intervals", ue["t_index"].nunique()),
        ("Tilt bounds [°]", f"{space.lower.min():g} to {space.upper.max():g}"),
        ("Current tilts [°]", ", ".join(f"{value:g}" for value in np.unique(space.baseline))),
        ("Tilt resolution [°]", f"{space.resolution_deg:g}"),
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


def pareto_front(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Which rows no other row dominates, each column read in its own direction.

    A row dominates another when it is no worse on every column and strictly
    better on at least one, so identical rows do not dominate each other.

    Returns:
        Boolean mask over the rows.
    """
    return pareto_mask(
        np.column_stack(
            [
                frame[column].to_numpy(float) * (1.0 if direction(column) == "maximise" else -1.0)
                for column in columns
            ]
        )
    )


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
        is_covered = covered(config.rsrp, cfg)
        on_covered = counts[is_covered] if is_covered.any() else np.array([np.nan])
        rows.append(
            {
                "configuration": name,
                "mean_neighbours_covered": overlap_neighbor_mean_of(counts, is_covered),
                "mean_neighbours_all": float(counts.mean()),
                "share_0_neighbours": float((on_covered == 0).mean()),
                "share_1_neighbours": float((on_covered == 1).mean()),
                "share_2_neighbours": float((on_covered == 2).mean()),
                "share_3plus_neighbours": float((on_covered >= 3).mean()),
            }
        )
    return pd.DataFrame(rows)
