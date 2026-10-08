"""Build every evaluation table and figure: the one implementation notebook 04 presents.

Entry point for ``task evaluate``. Reads run directories, the baseline radio
map and the processed UE table only, so like the rest of :mod:`src.evaluation`
it needs no GPU. UEs are served from ``data.output.ue_file``, every UE; the
sectors are read from ``simulation.input.sectors_file``.
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

from src.core.sector import read_sectors, site_frame
from src.data.load import grid_shape
from src.evaluation import compare, maps, plots
from src.evaluation import runs as run_store
from src.evaluation.export import readable, save_table
from src.kpi.capacity import max_rsrp
from src.kpi.overlap import overlap_neighbors
from src.optim.objective import OBJECTIVE_NAMES
from src.tracking import log_stage
from src.utils.plotting import label, save_fig, setup_plotting

# Subdirectory of cfg.reports.figures_dir and cfg.reports.tables_dir.
_STAGE = "04_evaluation"


def output_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """Where :func:`evaluate` writes, ``(figures, tables)``, from ``cfg.reports``."""
    return Path(cfg.reports.figures_dir) / _STAGE, Path(cfg.reports.tables_dir) / _STAGE


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
    checks = run_store.verify(runs, baseline)
    run_store.require(checks)
    return runs, checks


def evaluate(cfg: DictConfig, *, in_colab: bool = False) -> dict[str, pd.DataFrame | Figure]:
    """Build the evaluation tables and figures and write them to :func:`output_dirs`.

    Tables are returned and written with display names (:func:`readable`).
    Figures are closed after saving; displaying a closed figure still renders it.

    Returns:
        Every table and figure keyed by file name, in presentation order.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When a run is unfinished or the runs are not comparable.
    """
    return _evaluate(cfg, in_colab=in_colab)[0]


def _evaluate(
    cfg: DictConfig, *, in_colab: bool
) -> tuple[dict[str, pd.DataFrame | Figure], pd.DataFrame]:
    """:func:`evaluate`, also returning :func:`compare.seed_summary` before relabelling."""
    figures_dir, tables_dir = output_dirs(cfg)
    setup_plotting()
    results: dict[str, pd.DataFrame | Figure] = {}

    def add(name: str, item: pd.DataFrame | Figure) -> None:
        if isinstance(item, pd.DataFrame):
            item = readable(item)
            save_table(item, name, in_colab=in_colab, directory=tables_dir)
        else:
            save_fig(item, name, in_colab=in_colab, directory=figures_dir)
            plt.close(item)
        results[name] = item

    baseline = run_store.baseline_map(cfg)
    runs, checks = load_runs(cfg, baseline)
    add("comparability_checks", checks)

    ue = pd.read_parquet(cfg.data.output.ue_file)
    sectors = site_frame(read_sectors(cfg.simulation.input.sectors_file))
    band_labels = [str(band) for band in baseline["band_label"]]
    tx_names = [str(name) for name in baseline["tx_name"]]
    add("experiment_setup", compare.experiment_setup(baseline, ue, runs, cfg))

    summary = compare.seed_summary(runs)
    add("kpi_scoreboard", summary)
    add(
        "kpi_comparison",
        plots.kpi_comparison(
            summary,
            compare.NETWORK_KPIS,
            "KPIs: current configuration against each method's pick",
        ),
    )
    add(
        "objective_comparison",
        plots.kpi_comparison(
            summary,
            OBJECTIVE_NAMES,
            "Objectives: current configuration against each method's pick",
        ),
    )
    add("hypervolume", compare.hypervolume_table(runs))
    add("paired_gain_morbo_vs_random", compare.paired_method_gain(runs))

    searched = compare.candidates(runs)
    add("candidates", searched)
    for x, y in (
        ("coverage_objective", "separation_objective"),
        ("hole_rate", "overlap_rate"),
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

    # Per band only: the strongest layer across bands is not one a UE measures.
    for index, band in enumerate(band_labels):
        band_rsrp = {
            name: max_rsrp(config.rsrp[index : index + 1])
            for name, config in configurations.items()
        }
        # No path in either map makes the change undefined; it stays blank on the map.
        with np.errstate(invalid="ignore"):
            rsrp_change = {label(m): band_rsrp[m] - band_rsrp["incumbent"] for m in best}
        add(
            f"rsrp_change_maps_{band}",
            plots.map_row(
                rsrp_change,
                baseline,
                colorbar_label=f"Change in {label(band)} best-server RSRP [dB]",
                symmetric=True,
                sectors=sectors,
            ),
        )
        add(
            f"coverage_before_after_{band}",
            plots.coverage_maps(
                band_rsrp["incumbent"],
                band_rsrp[winner.method],
                baseline,
                cfg,
                sectors=sectors,
                name=winner.method,
                band=band,
            ),
        )
    add("coverage_by_area_and_demand", compare.coverage_by_area_and_demand(configurations, cfg))
    add(
        "coverage_class_maps",
        plots.coverage_class_maps(
            {key: configurations[key].rsrp for key in ("incumbent", winner.method)},
            baseline,
            cfg,
            sectors=sectors,
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
            sectors=sectors,
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
                "weak_rate",
                "overlap_rate",
                "rsrp_p05_dbm",
                "sinr_p05_db",
            ),
        ),
    )

    service = {
        name: compare.service_summary(c.served, band_labels) for name, c in configurations.items()
    }
    add("ue_service_summary", pd.DataFrame(service).T.rename_axis("configuration").reset_index())
    add("serving_band_mix", plots.band_share_bars(service, band_labels))
    add(
        "ue_throughput_maps",
        plots.map_row(
            {
                label(key): maps.tile_median(
                    configurations[key].served,
                    "estimated_throughput_mbps",
                    grid_shape(baseline),
                )
                for key in ("incumbent", winner.method)
            },
            baseline,
            colorbar_label="Median estimated throughput [Mbit/s]",
            vmin=0.0,
            cmap="viridis",
            sectors=sectors,
        ),
    )

    load = {
        name: compare.sector_band_load(configurations[name].served, band_labels, tx_names)
        for name in ("incumbent", winner.method)
    }
    add("sector_band_throughput", plots.sector_band_heatmaps(load, "median_throughput_mbps"))
    add("sector_band_load", pd.concat([f.assign(configuration=k) for k, f in load.items()]))
    add(
        "sector_impact",
        compare.sector_impact(winner.best_tilt, load["incumbent"], load[winner.method], sectors),
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
    return results, summary


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare the runs. Entry point for ``task evaluate``."""
    matplotlib.use("Agg")
    _, summary = _evaluate(cfg, in_colab=False)
    log_stage(
        cfg,
        "evaluation",
        groups=["kpi"],
        metrics={f"{row.method}_{row.kpi}_mean": row.mean for row in summary.itertuples()},
        artifacts=list(output_dirs(cfg)),
    )


if __name__ == "__main__":
    main()
