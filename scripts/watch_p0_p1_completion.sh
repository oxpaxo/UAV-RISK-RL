#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG="${WATCH_LOG:-runs/logs/p0_p1_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] p0_p1_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  status="$(python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(".")
missing = []
required = [
    "P0_TRACE_PREDICTIVE_ANALYSIS_REPORT.md",
    "P1_THREE_SEED_REPLICATION_REPORT.md",
    "results/p0_trace/p0_trace_summary.csv",
    "results/p1_three_seed/p1_by_seed_step_scenario.csv",
    "results/p1_three_seed/p1_summary_by_method_step_scenario.csv",
    "results/P0_P1_COMPLETE.flag",
]
for item in required:
    path = root / item
    if not path.exists() or path.stat().st_size == 0:
        missing.append(item)

p0_rows = 0
p1_rows = 0
p1_summary_rows = 0
p0_png = len(list((root / "results/p0_trace/plots").glob("*.png"))) if (root / "results/p0_trace/plots").exists() else 0
p1_png = len(list((root / "results/p1_three_seed/plots").glob("*.png"))) if (root / "results/p1_three_seed/plots").exists() else 0
train_logs = len(list((root / "runs/logs").glob("train_p1_*.log"))) if (root / "runs/logs").exists() else 0
eval_logs = len(list((root / "runs/logs").glob("eval_p1_*.log"))) if (root / "runs/logs").exists() else 0

try:
    p0 = pd.read_csv(root / "results/p0_trace/p0_trace_summary.csv")
    p0_rows = len(p0)
    needed_cols = {"episode", "lead_time_sum", "reaction_time", "source_trace_path"}
    if p0_rows < 10 or not needed_cols.issubset(set(p0.columns)):
        missing.append("results/p0_trace/p0_trace_summary.csv:10_rows_and_required_columns")
    elif pd.to_numeric(p0["reaction_time"], errors="coerce").isna().all():
        missing.append("results/p0_trace/p0_trace_summary.csv:reaction_time_not_all_nan")
except Exception:
    if (root / "results/p0_trace/p0_trace_summary.csv").exists():
        missing.append("results/p0_trace/p0_trace_summary.csv:readable")

try:
    by_seed = pd.read_csv(root / "results/p1_three_seed/p1_by_seed_step_scenario.csv")
    p1_rows = len(by_seed)
    required_cols = {"method", "seed", "step", "scenario", "success_rate", "collision_rate", "reaction_time_eval_style", "checkpoint_path", "eval_csv_path"}
    if p1_rows != 144 or not required_cols.issubset(set(by_seed.columns)):
        missing.append("results/p1_three_seed/p1_by_seed_step_scenario.csv:144_rows_required_columns")
    else:
        counts = by_seed.groupby(["method", "seed", "step"]).size()
        if counts.min() != 4 or counts.max() != 4 or len(counts) != 36:
            missing.append("results/p1_three_seed/p1_by_seed_step_scenario.csv:complete_method_seed_step_scenario_grid")
        for path in by_seed["eval_csv_path"].astype(str):
            fp = root / path
            if not fp.exists() or fp.stat().st_size == 0:
                missing.append(f"missing_eval_csv:{path}")
                break
except Exception:
    if (root / "results/p1_three_seed/p1_by_seed_step_scenario.csv").exists():
        missing.append("results/p1_three_seed/p1_by_seed_step_scenario.csv:readable")

try:
    summary = pd.read_csv(root / "results/p1_three_seed/p1_summary_by_method_step_scenario.csv")
    p1_summary_rows = len(summary)
    required_cols = {"method", "step", "scenario", "mean_reaction", "std_reaction", "mean_success", "std_success"}
    if p1_summary_rows != 48 or not required_cols.issubset(set(summary.columns)):
        missing.append("results/p1_three_seed/p1_summary_by_method_step_scenario.csv:48_rows_required_columns")
except Exception:
    if (root / "results/p1_three_seed/p1_summary_by_method_step_scenario.csv").exists():
        missing.append("results/p1_three_seed/p1_summary_by_method_step_scenario.csv:readable")

if p0_png < 11:
    missing.append("results/p0_trace/plots:>=11_png")
if p1_png < 4:
    missing.append("results/p1_three_seed/plots:>=4_png")

try:
    flag = json.loads((root / "results/P0_P1_COMPLETE.flag").read_text())
    if int(flag.get("required_p1_rows", 0)) != 144:
        missing.append("results/P0_P1_COMPLETE.flag:manifest")
except Exception:
    if (root / "results/P0_P1_COMPLETE.flag").exists():
        missing.append("results/P0_P1_COMPLETE.flag:valid_json")

payload = {
    "complete": len(missing) == 0,
    "missing": missing[:20],
    "missing_count": len(missing),
    "p0_rows": p0_rows,
    "p1_rows": p1_rows,
    "p1_summary_rows": p1_summary_rows,
    "p0_png": p0_png,
    "p1_png": p1_png,
    "train_logs": train_logs,
    "eval_logs": eval_logs,
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
    echo "[$(date '+%F %T')] COMPLETE P0/P1 artifacts verified" | tee -a "${LOG}"
    exit 0
  fi
  sleep "${INTERVAL}"
done
