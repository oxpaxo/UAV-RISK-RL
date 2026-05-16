#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG="${WATCH_LOG:-runs/logs/p0_p1_p0_5_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] p0_p1_p0_5_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  status="$(python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(".")
missing = []

required = [
    "results/P0_P1_COMPLETE.flag",
    "P1_THREE_SEED_REPLICATION_REPORT.md",
    "results/p0_5_distance_sanity/distance_cost_sanity.csv",
    "results/p0_5_distance_sanity/DISTANCE_COST_SANITY_REPORT.md",
    "results/p0_5_distance_sanity/distance_trigger_stats.csv",
    "results/p0_5_distance_sanity/DISTANCE_TRIGGER_STATS_REPORT.md",
    "results/p0_5_distance_wide/distance_wide_by_step_scenario.csv",
    "results/p0_5_distance_wide/DISTANCE_WIDE_REPORT.md",
    "results/p0_5_distance_wide/P0_5_DISTANCE_WIDE_COMPLETE.flag",
]
for item in required:
    path = root / item
    if not path.exists() or path.stat().st_size == 0:
        missing.append(item)

sanity_rows = 0
trigger_rows = 0
wide_rows = 0
wide_eval_csvs = len(list((root / "results/p0_5_distance_wide/eval").glob("*.csv"))) if (root / "results/p0_5_distance_wide/eval").exists() else 0

try:
    sanity = pd.read_csv(root / "results/p0_5_distance_sanity/distance_cost_sanity.csv")
    sanity_rows = len(sanity)
    required_cols = {
        "method",
        "episode",
        "min_min_distance",
        "max_distance_warning_cost_trace",
        "max_recomputed_distance_warning_cost",
        "max_abs_diff_between_trace_and_recomputed",
        "judgment",
    }
    if sanity_rows < 30 or not required_cols.issubset(set(sanity.columns)):
        missing.append("distance_cost_sanity.csv:30_rows_required_columns")
except Exception:
    if (root / "results/p0_5_distance_sanity/distance_cost_sanity.csv").exists():
        missing.append("distance_cost_sanity.csv:readable")

try:
    trigger = pd.read_csv(root / "results/p0_5_distance_sanity/distance_trigger_stats.csv")
    trigger_rows = len(trigger)
    required_cols = {
        "method",
        "step",
        "scenario",
        "distance_warning_cost_nonzero_rate",
        "min_distance_mean",
        "success_rate",
        "reaction_time_eval_style",
    }
    if trigger_rows != 18 or not required_cols.issubset(set(trigger.columns)):
        missing.append("distance_trigger_stats.csv:18_rows_required_columns")
except Exception:
    if (root / "results/p0_5_distance_sanity/distance_trigger_stats.csv").exists():
        missing.append("distance_trigger_stats.csv:readable")

try:
    wide = pd.read_csv(root / "results/p0_5_distance_wide/distance_wide_by_step_scenario.csv")
    wide_rows = len(wide)
    required_cols = {
        "method",
        "step",
        "scenario",
        "d_warning_eval",
        "distance_warning_cost_nonzero_rate",
        "success_rate",
        "reaction_time_eval_style",
    }
    if wide_rows != 64 or not required_cols.issubset(set(wide.columns)):
        missing.append("distance_wide_by_step_scenario.csv:64_rows_required_columns")
    elif "attention_full_distance_penalty_wide_d2" not in set(wide["method"].astype(str)):
        missing.append("distance_wide_by_step_scenario.csv:wide_method_present")
except Exception:
    if (root / "results/p0_5_distance_wide/distance_wide_by_step_scenario.csv").exists():
        missing.append("distance_wide_by_step_scenario.csv:readable")

if wide_eval_csvs < 16:
    missing.append("results/p0_5_distance_wide/eval:>=16_csv")

try:
    text = (root / "P1_THREE_SEED_REPLICATION_REPORT.md").read_text(encoding="utf-8")
    if "## Interaction with P0.5 Distance Sanity / Wide Distance Ablation" not in text:
        missing.append("P1 report:P0.5 interaction section")
    if "d_warning=2.0" not in text and "wide_d2" not in text:
        missing.append("P1 report:wide distance conclusion")
except Exception:
    if (root / "P1_THREE_SEED_REPLICATION_REPORT.md").exists():
        missing.append("P1 report:readable")

payload = {
    "complete": len(missing) == 0,
    "missing": missing[:30],
    "missing_count": len(missing),
    "sanity_rows": sanity_rows,
    "trigger_rows": trigger_rows,
    "wide_rows": wide_rows,
    "wide_eval_csvs": wide_eval_csvs,
}
print(json.dumps(payload, separators=(",", ":")))
PY
)"

  complete="$(python - <<PY
import json
data=json.loads('''${status}''')
print('1' if data.get('complete') else '0')
PY
)"
  echo "[$(date '+%F %T')] status=${status}" | tee -a "${LOG}"
  if [[ "${complete}" == "1" ]]; then
    echo "[$(date '+%F %T')] COMPLETE P0/P1/P0.5 artifacts verified" | tee -a "${LOG}"
    exit 0
  fi
  sleep "${INTERVAL}"
done
