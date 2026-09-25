#!/usr/bin/env bash
# TuRBO hyperparameter grid at the committed budget: every variant on every
# search seed, then a summary. Needs a CUDA GPU and the rt extra.
#
#   bash scripts/tune_turbo.sh            run what is missing, then summarise
#   bash scripts/tune_turbo.sh summary    summarise only
#
# PYTHON overrides the interpreter, e.g. PYTHON=python on a device without uv.
# A (variant, seed) whose run.json exists is skipped, so a crashed grid resumes.
set -euo pipefail

PYTHON=${PYTHON:-"uv run python"}
ROOT=outputs/optim_tuning
SEEDS=(42 1 2)
declare -A VARIANTS=(
  [baseline]=""
  [improve1e-4]="optim.method.trust_region.improvement=0.0001"
  [local5]="optim.method.trust_region.perturbed_dimensions=5"
  [combined]="optim.method.trust_region.improvement=0.0001 optim.method.trust_region.perturbed_dimensions=5 optim.method.trust_region.length_init=0.4"
)

if [[ "${1:-}" != summary ]]; then
  for variant in baseline improve1e-4 local5 combined; do
    for seed in "${SEEDS[@]}"; do
      dir="$ROOT/$variant/seed$seed"
      if compgen -G "$dir/turbo/*/run.json" > /dev/null; then
        echo "skip $variant seed $seed: done"
        continue
      fi
      echo "run $variant seed $seed"
      # optim.seed, not seed: only the search seed moves, the solver's stays.
      # word splitting of the overrides is intended
      # shellcheck disable=SC2086
      $PYTHON -m src.optim.run optim/method=turbo optim.seed="$seed" \
        optim.output.dir="$dir" \
        optim.output.deliverable_dir="$dir/deliverables" \
        optim.output.save_radio_map=false \
        ${VARIANTS[$variant]}
    done
  done
fi

$PYTHON - "$ROOT" <<'EOF'
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.compare import seed_summary
from src.evaluation.runs import load
from src.optim.objective import KPI_NAMES

root = Path(sys.argv[1])
runs = {}
for meta in sorted(root.glob("*/seed*/turbo/*/run.json")):
    run = load(meta.parent)
    runs.setdefault(meta.parents[3].name, []).append(run)
if not runs:
    sys.exit(f"no runs under {root}")

rows = [
    {
        "variant": variant,
        "seed": run.seed,
        "incumbent_J": run.incumbent_kpi.objective,
        "best_J": run.best_kpi.objective,
        "gain_J": run.best_kpi.objective - run.incumbent_kpi.objective,
        "best_iteration": run.best_index,
        "n_evaluations": run.n_evaluations,
        **{name: getattr(run.best_kpi, name) for name in KPI_NAMES},
    }
    for variant, mine in runs.items()
    for run in mine
]
per_run = pd.DataFrame(rows).sort_values(["variant", "seed"])
baseline = per_run[per_run["variant"] == "baseline"].set_index("seed")["best_J"]
per_run["paired_dJ_vs_baseline"] = per_run["best_J"] - per_run["seed"].map(baseline)
per_run.to_csv(root / "summary.csv", index=False)

kpis = pd.concat(
    [seed_summary(mine).assign(variant=variant) for variant, mine in runs.items()]
)
kpis.to_csv(root / "variant_summary.csv", index=False)

pd.set_option("display.width", 200)
print(per_run[["variant", "seed", "best_J", "gain_J", "best_iteration", "paired_dJ_vs_baseline"]])
print(
    per_run.groupby("variant")[["best_J", "paired_dJ_vs_baseline"]].agg(["mean", "min", "count"])
)
print(f"wrote {root / 'summary.csv'} and {root / 'variant_summary.csv'}")
EOF
