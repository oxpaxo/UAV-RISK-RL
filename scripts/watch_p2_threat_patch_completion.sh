#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG_PATH="${WATCH_LOG:-$ROOT/runs/logs/p2_threat_patch_watcher.log}"
mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] p2_threat_patch_watcher_start interval=${INTERVAL}" | tee -a "$LOG_PATH"

while true; do
  STATUS_JSON="$(python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
out = root / "results/p2_rich_motion"
missing: list[str] = []
rows: dict[str, int] = {}
terminal = ""

def require_file(rel: str) -> None:
    if not (root / rel).exists():
        missing.append(rel)

def read_rows(rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        missing.append(rel)
        rows[rel] = 0
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        rows[rel] = len(df)
        return df
    except Exception:
        missing.append(rel + ":readable")
        rows[rel] = 0
        return pd.DataFrame()

flag_path = out / "P2_PATCHED_STAGE1_COMPLETE.flag"
if not flag_path.exists():
    missing.append("results/p2_rich_motion/P2_PATCHED_STAGE1_COMPLETE.flag")
else:
    try:
        flag = json.loads(flag_path.read_text())
        terminal = str(flag.get("terminal_decision", ""))
    except Exception:
        missing.append("results/p2_rich_motion/P2_PATCHED_STAGE1_COMPLETE.flag:valid_json")

require_file("P2_ENVIRONMENT_SANITY_REPORT_PATCHED.md")
require_file("P2_STAGE1_OOD_EVAL_REPORT.md")

random_df = read_rows("results/p2_rich_motion/env_sanity_random_policy_patched.csv")
short_df = read_rows("results/p2_rich_motion/env_sanity_short_ppo_patched.csv")
invalid_df = read_rows("results/p2_rich_motion/threat_generation_invalid_reasons.csv")
stage1_df = read_rows("results/p2_rich_motion/p2_stage1_ood_eval.csv")

if len(random_df) != 2:
    missing.append("env_sanity_random_policy_patched.csv:2_rows")
if len(short_df) != 2:
    missing.append("env_sanity_short_ppo_patched.csv:2_rows")
if len(invalid_df) < 2:
    missing.append("threat_generation_invalid_reasons.csv:>=2_rows")
if len(stage1_df) != 40:
    missing.append("p2_stage1_ood_eval.csv:40_rows")

for label, df in [("random", random_df), ("short", short_df)]:
    if not df.empty:
        if (pd.to_numeric(df.get("scenario_valid_rate"), errors="coerce") < 1.0).any():
            missing.append(f"{label}:scenario_valid_rate_1_required")
        if (pd.to_numeric(df.get("planned_threat_valid_rate"), errors="coerce") < 0.8).any():
            missing.append(f"{label}:planned_threat_valid_rate_ge_0p8_required")
        if (pd.to_numeric(df.get("init_collision_rate"), errors="coerce") > 0.0).any():
            missing.append(f"{label}:init_collision_rate_0_required")
        if (pd.to_numeric(df.get("collision_rate"), errors="coerce") >= 1.0).any():
            missing.append(f"{label}:not_all_collision_required")
        if (pd.to_numeric(df.get("realized_near_miss_rate"), errors="coerce") <= 0.0).any():
            missing.append(f"{label}:not_no_threat_required")

plot_count = len(list((out / "plots").glob("stage1_*.png")))
if plot_count < 5:
    missing.append("plots/stage1_*.png:>=5")

if terminal != "patched_stage1_complete":
    missing.append("P2_PATCHED_STAGE1_COMPLETE.flag:terminal_patched_stage1_complete")

complete = not missing
print(json.dumps({
    "complete": complete,
    "terminal_decision": terminal,
    "missing": missing[:25],
    "missing_count": len(missing),
    "row_counts": rows,
    "stage1_plot_count": plot_count,
}, sort_keys=True))
PY
)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] status=${STATUS_JSON}" | tee -a "$LOG_PATH"
  if python - "$STATUS_JSON" <<'PY'
import json, sys
sys.exit(0 if json.loads(sys.argv[1]).get("complete") else 1)
PY
  then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE P2 threat patch Stage0+Stage1 artifacts verified" | tee -a "$LOG_PATH"
    exit 0
  fi
  sleep "$INTERVAL"
done
