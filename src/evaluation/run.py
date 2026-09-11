"""Compare the newest verified run of every method. The script form of notebook 05.

Entry point for ``task evaluate``. Reads run directories and the baseline radio
map only, so like the rest of :mod:`src.evaluation` it needs no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig

from src.evaluation import compare, maps, plots
from src.evaluation import runs as run_store
from src.evaluation.export import save_table
from src.optim.objective import KPI_NAMES
from src.tracking import log_stage
from src.utils.plotting import save_fig, setup_plotting
from src.utils.seed import set_seed

FIGURES_DIR = Path("reports/figures/05_evaluation")
TABLES_DIR = Path("reports/tables/05_evaluation")


def evaluate(cfg: DictConfig) -> pd.DataFrame:
    """Write notebook 05's tables and figures under ``reports/``.

    Returns the method comparison table.

    Raises:
        FileNotFoundError: When ``optim.output.dir`` holds no finished run.
        RunError: When a run is unfinished or unverified, or the newest runs are
            not comparable with each other or with the baseline map.
    """
    set_seed(cfg.seed)
    setup_plotting()

    def figure(fig: plt.Figure, name: str) -> None:
        save_fig(fig, name, in_colab=False, directory=FIGURES_DIR)
        plt.close(fig)

    def table(frame: pd.DataFrame, name: str) -> None:
        save_table(frame, name, in_colab=False, directory=TABLES_DIR)

    latest = run_store.latest_per_method(run_store.discover(cfg.optim.output.dir))
    if not latest:
        raise FileNotFoundError(f"No runs under {cfg.optim.output.dir}. Run `task optim` first.")
    runs = [latest[method] for method in sorted(latest)]
    baseline = run_store.baseline_map(cfg)
    run_store.require(run_store.verify(runs, baseline))
    print(compare.summarise(runs, cfg))

    mdt = pd.read_parquet(cfg.data.output.mdt_file)
    cells = pd.read_parquet(cfg.data.output.cell_file).drop_duplicates("cell")[["cell", "x", "y"]]
    manifest = json.loads(Path(cfg.simulation.output.manifest_file).read_text(encoding="utf-8"))
    hotspots = pd.DataFrame(manifest["density"]["hotspots"])

    baseline_rsrp = baseline["rsrp_dbm"].astype(float)
    # PRB demand is the incumbent's: SINR, and so PRBs per UE, depend on the map.
    counts = maps.demand(baseline_rsrp, [str(b) for b in baseline["band_label"]], mdt, cfg)

    winner = compare.best_method(runs, cfg)
    deltas = {run.method: compare.delta_table(run.incumbent_kpi, run.best_kpi, cfg) for run in runs}
    table(deltas[winner.method], f"delta_{winner.method}")

    before = maps.best_server(baseline_rsrp)
    after = maps.best_server(winner.radio_map["rsrp_dbm"].astype(float))
    figure(
        plots.coverage_maps(before, after, baseline, cfg, cells=cells, label=winner.method),
        "coverage_before_after",
    )
    figure(
        plots.demand_signal_maps(
            baseline_rsrp, counts, baseline, cfg, cells=cells, hotspots=hotspots
        ),
        "demand_vs_signal",
    )
    figure(plots.coverage_cdf(before, counts, cfg), "coverage_cdf")

    coverage = {"incumbent": maps.coverage_table(baseline_rsrp, counts, cfg)}
    for run in runs:
        coverage[run.method] = maps.coverage_table(
            run.radio_map["rsrp_dbm"].astype(float), counts, cfg
        )
    table(compare.coverage_comparison(coverage), "coverage_by_area_and_demand")

    methods = compare.method_table(runs, cfg)
    table(methods, "method_comparison")
    figure(plots.kpi_comparison(deltas, cfg), "kpi_improvement_vs_tolerance")
    figure(plots.verdict_counts(deltas), "verdict_per_kpi")

    trace = compare.convergence(runs)
    table(trace, "convergence")
    figure(plots.convergence_plot(trace), "convergence")
    figure(plots.pareto_plot(runs, cfg), "pareto_front")

    table(compare.tilt_movement(winner), f"tilt_movement_{winner.method}")
    figure(plots.tilt_movement_plot(winner), "tilt_movement")
    table(winner.best_tilt, f"best_tilt_{winner.method}")

    print(f"the priority order prefers {winner.label}; wrote {FIGURES_DIR} and {TABLES_DIR}")
    return methods


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare the runs. Entry point for ``task evaluate``."""
    matplotlib.use("Agg")
    methods = evaluate(cfg)
    log_stage(
        cfg,
        "evaluation",
        groups=["kpi"],
        metrics={
            f"{row['method']}_{name}": row[name]
            for _, row in methods.iterrows()
            for name in KPI_NAMES
        },
        artifacts=[FIGURES_DIR, TABLES_DIR],
    )


if __name__ == "__main__":
    main()
