"""Rerun every stage and notebook, then zip what the report is written from.

Entry point for ``task reproduce``. Notebooks 00-02 build the scenario, radio
map and UE table; each method and seed is searched by ``src.optim.run``;
notebooks 03a, 03b and 04 then find those runs on disk instead of searching
again. Needs a CUDA GPU and every extra (``task setup``).

Usage::

    uv run python scripts/reproduce.py --seeds 42,43,44 optim.budget.n_iter=200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config import load_config  # noqa: E402
from src.evaluation import runs as run_store  # noqa: E402

HEARTBEAT_SECONDS = 60


def run_stage(name: str, command: list[str], env: dict[str, str]) -> None:
    """Run one stage, printing a heartbeat while it runs.

    Raises:
        subprocess.CalledProcessError: When the stage exits non-zero.
    """
    print(f"\n=== {name}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, env=env)
    while True:
        try:
            code = process.wait(timeout=HEARTBEAT_SECONDS)
            break
        except subprocess.TimeoutExpired:
            print(f"... {name} running, {(time.monotonic() - started) / 60:.1f} min", flush=True)
    minutes = (time.monotonic() - started) / 60
    print(f"=== {name}: exit {code} after {minutes:.1f} min", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def notebook(name: str, out_dir: Path, env: dict[str, str]) -> None:
    """Execute ``notebooks/<name>.ipynb`` into ``out_dir``, leaving the tracked file as is."""
    run_stage(
        name,
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output-dir",
            str(out_dir),
            str(ROOT / "notebooks" / f"{name}.ipynb"),
        ],
        env,
    )


def bundle(path: Path, sources: list[Path]) -> None:
    """Zip every existing file under ``sources``, at its path relative to the project root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            files = sorted(source.rglob("*")) if source.is_dir() else [source]
            for file in files:
                if file.is_file():
                    archive.write(file, file.resolve().relative_to(ROOT))
    print(f"\nbundle: {path}", flush=True)


def main() -> None:
    """Parse the arguments, run every stage in order, and always write the bundle."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", help="comma-separated search seeds; default optim.seed")
    parser.add_argument("--methods", default="morbo,random", help="comma-separated methods")
    parser.add_argument("--bundle", type=Path, help="zip to write; default under outputs/")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for every stage")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    seeds = [int(s) for s in (args.seeds or str(cfg.optim.seed)).split(",")]
    methods = args.methods.split(",")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "outputs" / "reproduce" / stamp
    bundle_path = args.bundle or out_dir.with_suffix(".zip")
    runs_dir = Path(cfg.optim.output.dir)

    # The notebooks read their overrides and seeds from these (cell 1 of each).
    env = {
        **os.environ,
        "BAND_TILT_OVERRIDES": " ".join(args.overrides),
        "BAND_TILT_SEEDS": " ".join(map(str, seeds)),
    }
    if run_store.discover(runs_dir):
        # Notebook 03a reuses a run with the same method and seed whatever its budget.
        print(f"WARNING {runs_dir} already holds runs; matching (method, seed) runs are reused.")

    try:
        for name in ("00_simulation", "01_eda", "02_preprocessing"):
            notebook(name, out_dir / "notebooks", env)
        for seed in seeds:
            for method in methods:
                done = {(r.method, r.seed) for r in run_store.discover(runs_dir)}
                if (method, seed) in done:
                    print(f"\n=== skip {method} seed {seed}: a finished run is on disk")
                    continue
                run_stage(
                    f"{method} seed {seed}",
                    [
                        sys.executable,
                        "-m",
                        "src.optim.run",
                        f"optim/method={method}",
                        f"optim.seed={seed}",
                        *args.overrides,
                    ],
                    env,
                )
        for name in ("03a_baseline", "03b_morbo", "04_evaluation"):
            notebook(name, out_dir / "notebooks", env)
    finally:
        bundle(
            bundle_path,
            [
                out_dir,
                Path(cfg.reports.figures_dir),
                Path(cfg.reports.tables_dir),
                Path(cfg.optim.output.deliverable_dir),
                runs_dir,
                Path(cfg.simulation.input.ue_file),
                Path(cfg.simulation.input.sectors_file),
                Path(cfg.simulation.input.manifest_file),
                Path(cfg.scenario.output.record_file),
                Path(cfg.simulation.output.radio_map_file),
                Path(cfg.data.output.ue_file),
            ],
        )


if __name__ == "__main__":
    main()
