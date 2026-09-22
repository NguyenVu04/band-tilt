"""Build every evaluation table and figure: the one implementation notebook 04 presents.

Entry point for ``task evaluate``. Reads run directories, the baseline radio
map and the processed tables only, so like the rest of :mod:`src.evaluation` it
needs no GPU. UEs are served from ``data.output.ue_file``, every UE.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.evaluation import compare, maps, plots
from src.evaluation import runs as run_store
from src.evaluation.export import readable, save_table
from src.kpi.capacity import CapacitySpec, max_rsrp
from src.kpi.overlap import overlap_neighbors
from src.optim.objective import MEASURE_NAMES
from src.tracking import log_stage
from src.utils.plotting import label, save_fig, setup_plotting
from src.utils.seed import set_seed

FIGURES_DIR = Path("reports/figures/04_evaluation")
TABLES_DIR = Path("reports/tables/04_evaluation")


def load_runs(
    cfg: DictConfig, baseline: dict[str, np.ndarray] | None = None
) -> tuple[list[run_store.Run], pd.DataFrame]:
    """The newest run of each method and seed, verified comparable with the baseline.

    Returns the runs and the comparability checks they passed. ``baseline`` is
    read from ``cfg`` when None.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When the runs are not comparable with each other or the baseline.
    """
    runs = run_store.latest_per_method_and_seed(run_store.discover(cfg.optim.output.dir))
    if not runs:
        raise FileNotFoundError(f"No runs under {cfg.optim.output.dir}. Run `task optim` first.")
    if baseline is None:
        baseline = run_store.baseline_map(cfg)
    checks = run_store.verify(runs, baseline, cfg)
    run_store.require(checks)
    return runs, checks


def evaluate(cfg: DictConfig, *, in_colab: bool = False) -> dict[str, pd.DataFrame | Figure]:
    """Build the evaluation tables and figures and write them under ``reports/``.

    Tables are returned and written with display names (:func:`readable`).
    Figures are closed after saving; displaying a closed figure still renders it.

    Returns:
        Every table and figure keyed by file name, in presentation order.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When a run is unfinished or the runs are not comparable.
    """
    set_seed(cfg.seed)
    setup_plotting()
    results: dict[str, pd.DataFrame | Figure] = {}

    def add(name: str, item: pd.DataFrame | Figure) -> None:
        if isinstance(item, pd.DataFrame):
            item = readable(item)
            save_table(item, name, in_colab=in_colab, directory=TABLES_DIR)
        else:
            save_fig(item, name, in_colab=in_colab, directory=FIGURES_DIR)
            plt.close(item)
        results[name] = item

    baseline = run_store.baseline_map(cfg)
    runs, checks = load_runs(cfg, baseline)
    add("comparability_checks", checks)

    ue = pd.read_parquet(cfg.data.output.ue_file)
    cells = pd.read_parquet(cfg.data.output.cell_file).drop_duplicates("cell")
    band_labels = [str(band) for band in baseline["band_label"]]
    tx_names = [str(name) for name in baseline["tx_name"]]
    add("experiment_setup", compare.experiment_setup(baseline, ue, runs, cfg))

    summary = compare.seed_summary(runs)
    add("kpi_scoreboard", summary)
    add("kpi_relative_improvement", compare.relative_improvement(summary))
    add("kpi_improvement", plots.kpi_comparison(summary))
    add("winner_vs_candidates", compare.winner_vs_candidates(runs))
    add("paired_gain_turbo_vs_random", compare.paired_method_gain(runs))

    searched = compare.candidates(runs)
    add("candidates", searched)
    for x, y in (
        ("hole_rate", "overlap_rate"),
        ("hole_rate", "served_rate"),
        ("overlap_rate", "served_rate"),
    ):
        add(f"tradeoff_{x}_vs_{y}", plots.tradeoff_scatter(searched, x, y))

    best = compare.best_run_per_method(runs)
    winner = compare.best_method(runs)
    configurations = {
        "incumbent": compare.configuration(baseline, ue, cfg),
        **{method: compare.configuration(run.radio_map, ue, cfg) for method, run in best.items()},
    }
    add(
        "kpi_reproducibility",
        compare.reproducibility(
            {"incumbent": runs[0].incumbent_kpi, **{m: run.best_kpi for m, run in best.items()}},
            configurations,
            band_labels,
            ue,
            cfg,
        ),
    )

    best_rsrp = {name: max_rsrp(config.rsrp) for name, config in configurations.items()}
    # No path in either map makes the change undefined; it stays blank on the map.
    with np.errstate(invalid="ignore"):
        rsrp_change = {label(m): best_rsrp[m] - best_rsrp["incumbent"] for m in best}
    add(
        "rsrp_change_maps",
        plots.map_row(
            rsrp_change,
            baseline,
            colorbar_label="Change in best-server RSRP [dB]",
            symmetric=True,
            cells=cells,
        ),
    )
    add(
        "coverage_before_after",
        plots.coverage_maps(
            best_rsrp["incumbent"],
            best_rsrp[winner.method],
            baseline,
            cfg,
            cells=cells,
            name=winner.method,
        ),
    )
    # Every configuration weighed by the incumbent's demand, so the weights do not move.
    demand = configurations["incumbent"].demand
    coverage = compare.coverage_comparison(
        {name: maps.coverage_table(c.rsrp, demand, cfg) for name, c in configurations.items()}
    )
    coverage.columns = ["coverage"] + [
        f"{label(key)}: {label(f'{share}_share')}"
        for key, share in (column.rsplit("_", 1) for column in coverage.columns[1:])
    ]
    add("coverage_by_area_and_demand", coverage)
    add(
        "coverage_class_maps",
        plots.coverage_class_maps(
            {key: configurations[key].rsrp for key in ("incumbent", winner.method)},
            baseline,
            cfg,
            cells=cells,
        ),
    )
    add("overlap_neighbour_summary", compare.overlap_neighbour_summary(configurations, cfg))
    add(
        "overlap_neighbour_maps",
        plots.map_row(
            {
                label(key): overlap_neighbors(configurations[key].rsrp, cfg).astype(float)
                for key in ("incumbent", winner.method)
            },
            baseline,
            colorbar_label="Overlapping co-band neighbours",
            vmin=0.0,
            cmap="magma",
            cells=cells,
        ),
    )
    add("band_layer_summary", compare.band_layer_summary(configurations, band_labels, cfg))
    per_band = compare.band_kpis(configurations, band_labels, cfg)
    add("band_kpis", per_band)
    add(
        "band_kpi_panels",
        plots.band_kpi_panels(
            per_band,
            (
                "hole_rate",
                "overlap_rate",
                "weak_rate",
                "rsrp_p05_dbm",
                "sinr_p05_db",
                "served_rate",
            ),
        ),
    )

    service = {
        name: compare.service_summary(c.served, band_labels) for name, c in configurations.items()
    }
    add("ue_service_summary", pd.DataFrame(service).T.rename_axis("configuration").reset_index())
    add("serving_band_mix", plots.band_share_bars(service, band_labels))

    max_prb = CapacitySpec.from_config(cfg, band_labels, len(tx_names)).max_prb
    load = {
        name: compare.cell_band_load(configurations[name].served, band_labels, tx_names, max_prb)
        for name in ("incumbent", winner.method)
    }
    add("cell_band_utilisation", plots.utilisation_heatmaps(load))
    add("cell_band_load", pd.concat([f.assign(configuration=k) for k, f in load.items()]))
    usage = compare.prb_usage_by_time(
        {key: configurations[key] for key in ("incumbent", winner.method)},
        band_labels,
        tx_names,
        max_prb,
    )
    add("prb_usage_by_time", usage)
    add(
        "prb_usage_heatmaps",
        plots.prb_usage_heatmaps(usage, float(cfg.kpi.capacity.max_admission_utilisation)),
    )
    add(
        "cell_impact",
        compare.cell_impact(winner.best_tilt, load["incumbent"], load[winner.method], cells),
    )
    add("recommended_tilt", winner.best_tilt)
    add("tilt_movement_summary", compare.tilt_movement(winner))
    add("tilt_movement", plots.tilt_movement_plot(winner.best_tilt, winner.method))
    add("tilt_delta_heatmap", plots.tilt_delta_heatmap(winner.best_tilt, winner.method))

    add("method_cost", compare.method_table(runs))
    trace = compare.convergence(runs)
    add("convergence", trace)
    add("sample_efficiency", compare.sample_efficiency(trace))
    add("search_progress", plots.convergence_plot(trace))
    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare the runs. Entry point for ``task evaluate``."""
    matplotlib.use("Agg")
    evaluate(cfg)
    summary = compare.seed_summary(load_runs(cfg)[0])
    log_stage(
        cfg,
        "evaluation",
        groups=["kpi"],
        metrics={
            f"{row.method}_{row.kpi}_mean": row.mean
            for row in summary.itertuples()
            if row.kpi in MEASURE_NAMES
        },
        artifacts=[FIGURES_DIR, TABLES_DIR],
    )


if __name__ == "__main__":
    main()
