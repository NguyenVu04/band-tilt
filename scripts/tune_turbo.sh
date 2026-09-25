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
SEEDS=(42 1 2 3 4)
# Round 2: variations on local5. Round 1's improve1e-4 and combined are no
# longer run, since both let the region grow; their runs stay in the summary.
TR=optim.method.trust_region
ORDER=(baseline local5 local3 local5-shrink local5-cap)
declare -A VARIANTS=(
  [baseline]=""
  [local5]="$TR.perturbed_dimensions=5"
  [local3]="$TR.perturbed_dimensions=3"
  [local5-shrink]="$TR.perturbed_dimensions=5 $TR.length_init=0.4"
  [local5-cap]="$TR.perturbed_dimensions=5 $TR.length_max=0.8"
)

if [[ "${1:-}" != summary ]]; then
  total=$(( ${#ORDER[@]} * ${#SEEDS[@]} ))
  index=0 ran=0
  for variant in "${ORDER[@]}"; do
    for seed in "${SEEDS[@]}"; do
      index=$(( index + 1 ))
      dir="$ROOT/$variant/seed$seed"
      if compgen -G "$dir/turbo/*/run.json" > /dev/null; then
        echo "[$index/$total] skip $variant seed $seed: already done"
        continue
      fi
      echo "[$index/$total] $(date +%H:%M:%S) start $variant seed $seed"
      started=$SECONDS
      # optim.seed, not seed: only the search seed moves, the solver's stays.
      # word splitting of the overrides is intended
      # shellcheck disable=SC2086
      $PYTHON -m src.optim.run optim/method=turbo optim.seed="$seed" \
        optim.output.dir="$dir" \
        optim.output.deliverable_dir="$dir/deliverables" \
        optim.output.save_radio_map=false \
        ${VARIANTS[$variant]}
      ran=$(( ran + 1 ))
      # ETA from this session's mean run time; skipped runs cost nothing.
      left=$(( (total - index) * SECONDS / ran ))
      echo "[$index/$total] $(date +%H:%M:%S) done $variant seed $seed" \
        "in $(( (SECONDS - started) / 60 )) min; about $(( left / 60 )) min left"
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
