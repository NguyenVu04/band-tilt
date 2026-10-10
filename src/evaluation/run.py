"""Build every evaluation table and figure: the one implementation notebook 04 presents.

Entry point for ``task evaluate``. Reads the newest run of each method, the
processed UE table and the stored baseline map (the incumbent), and re-traces each
method's largest hypervolume contribution on
:data:`src.evaluation.compare.EVALUATION_KPIS`, so it needs a CUDA GPU. UEs are
served from ``data.output.ue_file``, every UE; the sectors are read from
``simulation.input.sectors_file``.
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
from src.evaluation import compare, plots
from src.evaluation import runs as run_store
from src.evaluation.export import readable, save_table
from src.kpi.capacity import max_rsrp
from src.kpi.overlap import overlap_neighbors
from src.optim.evaluator import Evaluator
from src.optim.objective import BAND_KPI_NAMES
from src.optim.space import TiltSpace
from src.simulation.radio import read_manifest
from src.tracking import log_stage
from src.utils.plotting import label, save_fig, setup_plotting

# Subdirectory of cfg.reports.figures_dir and cfg.reports.tables_dir.
_STAGE = "04_evaluation"

# The method whose pick the location and time sections analyse.
PROPOSED = "morbo"


def output_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """Where :func:`evaluate` writes, ``(figures, tables)``, from ``cfg.reports``."""
    return Path(cfg.reports.figures_dir) / _STAGE, Path(cfg.reports.tables_dir) / _STAGE


def load_runs(cfg: DictConfig) -> tuple[list[run_store.Run], pd.DataFrame]:
    """The newest run of each method, verified comparable with each other and with ``cfg``.

    Returns the runs and the comparability checks they passed.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When the runs are not comparable.
    """
    runs = run_store.latest_per_method(run_store.discover(cfg.optim.output.dir))
    if not runs:
        raise FileNotFoundError(f"No runs under {cfg.optim.output.dir}. Run `task optim` first.")
    checks = run_store.verify(runs, cfg, str(read_manifest(cfg)["scenario_id"]))
    run_store.require(checks)
    return runs, checks


def retrace(cfg: DictConfig, tilts: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Each named tilt vector's radio map, in :func:`src.simulation.radio.radio_map`'s schema.

    The baseline tilt is read from the stored map; see :class:`Evaluator`.
    Side effect: ray-traces every other vector on the GPU.
    """
    with Evaluator(cfg, keep_rsrp=True) as evaluator:
        return {name: evaluator.radio_map(evaluator.evaluate(tilt)) for name, tilt in tilts.items()}


def evaluate(cfg: DictConfig, *, in_colab: bool = False) -> dict[str, pd.DataFrame | Figure]:
    """Build the evaluation tables and figures and write them to :func:`output_dirs`.

    Tables are returned and written with display names (:func:`readable`); the
    Pareto tilt table is also written to ``optim.output.deliverable_dir``.
    Figures are closed after saving; displaying a closed figure still renders it.

    Returns:
        Every table and figure keyed by file name, in presentation order.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When a run is unfinished or the runs are not comparable.
    """
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

    runs, checks = load_runs(cfg)
    add("comparability_checks", checks)
    add("search_budget", compare.search_budget(runs))

    space = TiltSpace.from_config(cfg)
    band_labels = list(space.band_names)
    searched = compare.candidates(runs)
    picks = {
        method: compare.method_pick(searched, method)
        for method in dict.fromkeys(searched["method"])
    }
    incumbent_row = searched.index[searched["phase"] == "incumbent"][0]
    rows = {"incumbent": searched.loc[incumbent_row]} | {
        m: searched.loc[i] for m, i in picks.items()
    }
    tilts = {name: row[list(space.parameter_names)].to_numpy(float) for name, row in rows.items()}

    radio = retrace(cfg, tilts)
    ue = pd.read_parquet(cfg.data.output.ue_file)
    configurations = {name: compare.configuration(radio[name], ue, cfg) for name in rows}
    sectors = site_frame(read_sectors(cfg.simulation.input.sectors_file))
    grid = radio["incumbent"]
    add("experiment_setup", compare.experiment_setup(grid, ue, runs, space, cfg))
    add(
        "kpi_reproducibility",
        compare.reproducibility(rows, configurations, band_labels, ue, cfg),
    )

    # Search effectiveness.
    add("hypervolume", compare.hypervolume_table(runs))
    add("search_progress", plots.convergence_plot(compare.convergence(runs)))

    # Trade-offs between the three KPIs.
    add("candidates", searched)
    kpis = compare.EVALUATION_KPIS
    for x, y in ((kpis[0], kpis[1]), (kpis[0], kpis[2]), (kpis[1], kpis[2])):
        add(f"tradeoff_{x}_vs_{y}", plots.tradeoff_scatter(searched, x, y, picks))
    chosen = searched.loc[[incumbent_row, *picks.values()], ["method", "iteration", "phase", *kpis]]
    add("picks", chosen.assign(configuration=list(rows)))

    # Method-level statistics.
    add("method_tests", compare.method_tests(searched))
    add("front_comparison", compare.front_comparison(searched))

    # Per band.
    per_band = compare.band_table(rows, band_labels)
    add("band_kpis", per_band)
    add(
        "band_kpi_panels",
        plots.band_kpi_panels(per_band, [k for k in BAND_KPI_NAMES if k != "served_share"]),
    )
    add("band_tradeoff", plots.band_tradeoff(searched, band_labels))
    service = {
        name: compare.service_summary(c.served, band_labels) for name, c in configurations.items()
    }
    add("serving_band_mix", plots.band_share_bars(service, band_labels))

    # By location: the incumbent against the proposed method's pick.
    shown = ("incumbent", PROPOSED)
    add("coverage_by_area_and_demand", compare.coverage_by_area_and_demand(configurations, cfg))
    add(
        "coverage_class_maps",
        plots.coverage_class_maps(
            {k: configurations[k].rsrp for k in shown}, grid, cfg, sectors=sectors
        ),
    )
    add(
        "coverage_before_after_all",
        plots.coverage_maps(
            *(max_rsrp(configurations[k].rsrp) for k in shown),
            grid,
            cfg,
            sectors=sectors,
            name=PROPOSED,
            band="all",
        ),
    )
    for index, band in enumerate(band_labels):
        add(
            f"coverage_before_after_{band}",
            plots.coverage_maps(
                *(max_rsrp(configurations[k].rsrp[index : index + 1]) for k in shown),
                grid,
                cfg,
                sectors=sectors,
                name=PROPOSED,
                band=band,
            ),
        )
    add(
        "overlap_neighbour_maps",
        plots.map_row(
            {label(k): overlap_neighbors(configurations[k].rsrp, cfg).astype(float) for k in shown},
            grid,
            colorbar_label="Overlapping co-band neighbours",
            vmin=0.0,
            cmap="magma",
            sectors=sectors,
            title="Number of overlapping co-band neighbours per tile",
        ),
    )
    # serve_intervals keeps the UE table's row order, so its positions align.
    x, y = ue["x"].to_numpy(), ue["y"].to_numpy()
    throughput = {
        k: configurations[k].served["estimated_throughput_mbps"].to_numpy() for k in shown
    }
    hexagon = f"{plots.HEX_SIZE_M:g} m hexagon"
    add(
        "ue_throughput_maps",
        plots.throughput_hexbins(
            {label(k): throughput[k] for k in shown},
            x,
            y,
            grid,
            colorbar_label="Median estimated throughput [Mbps]",
            title=f"Median estimated UE throughput per {hexagon}",
            sectors=sectors,
        ),
    )
    add(
        "ue_throughput_change_map",
        plots.throughput_hexbins(
            {
                f"Change: {label(PROPOSED)} vs {label('incumbent').lower()}": throughput[PROPOSED]
                - throughput["incumbent"]
            },
            x,
            y,
            grid,
            colorbar_label="Median change in estimated throughput [Mbps]",
            title=f"Change in estimated UE throughput per {hexagon}",
            symmetric=True,
            sectors=sectors,
        ),
    )
    add(
        "demand_signal_maps",
        plots.demand_signal_maps(
            configurations[PROPOSED].rsrp, x, y, grid, sectors=sectors, name=PROPOSED
        ),
    )

    # Over time, with UE density.
    by_interval = compare.interval_throughput({k: configurations[k] for k in shown})
    add("interval_throughput", by_interval)
    interval_s = float(read_manifest(cfg)["time"]["interval_s"])
    add(
        "interval_throughput_plot",
        plots.throughput_by_time_of_day({k: configurations[k].served for k in shown}, interval_s),
    )
    add("throughput_vs_load", plots.throughput_vs_load(by_interval))
    add(
        "throughput_cdf",
        plots.cdf_plot(
            {
                k: c.served["estimated_throughput_mbps"].to_numpy()
                for k, c in configurations.items()
            },
            "Estimated throughput [Mbps]",
            "Empirical CDF of estimated UE throughput (unserved UEs at 0 Mbps)",
        ),
    )

    # What the engineers choose from.
    front = compare.pareto_tilts(runs, band_labels)
    add("pareto_tilts", front)
    deliverable = Path(cfg.optim.output.deliverable_dir)
    if not in_colab:
        deliverable.mkdir(parents=True, exist_ok=True)
        front.to_csv(deliverable / "pareto_tilts.csv", index=False)
    add(
        "tilt_delta_heatmap",
        plots.tilt_delta_heatmap(compare.tilt_table(rows[PROPOSED], space), PROPOSED),
    )
    add(
        "tilt_change_bars",
        plots.tilt_change_bars(compare.tilt_table(rows[PROPOSED], space), PROPOSED),
    )
    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare the runs. Entry point for ``task evaluate``."""
    matplotlib.use("Agg")
    results = evaluate(cfg, in_colab=False)
    picks = results["picks"]
    log_stage(
        cfg,
        "evaluation",
        groups=["kpi"],
        metrics={
            f"{row[label('configuration')]}_{kpi}": float(row[label(kpi)])
            for _, row in picks.iterrows()
            for kpi in compare.EVALUATION_KPIS
        },
        artifacts=list(output_dirs(cfg)),
    )


if __name__ == "__main__":
    main()
