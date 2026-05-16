#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG_PATH="${WATCH_LOG:-$ROOT/runs/logs/p2_watcher.log}"
mkdir -p "$(dirname "$LOG_PATH")"

cd "$ROOT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] p2_watcher_start interval=${INTERVAL}" | tee -a "$LOG_PATH"

while true; do
  STATUS_JSON="$(python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
out = root / "results/p2_rich_motion"
missing: list[str] = []
row_counts: dict[str, int] = {}
terminal = ""

def rows(rel: str) -> int:
    path = root / rel
    if not path.exists():
        missing.append(rel)
        row_counts[rel] = 0
        return 0
    try:
        count = len(pd.read_csv(path))
    except Exception:
        missing.append(rel + ":readable")
        count = 0
    row_counts[rel] = count
    return count

def require_file(rel: str) -> None:
    if not (root / rel).exists():
        missing.append(rel)

def require_log(marker_rel: str, marker: str) -> None:
    path = root / marker_rel
    if not path.exists():
        missing.append(marker_rel)
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    if marker not in text:
        missing.append(marker_rel + f":missing_{marker}")

flag_path = out / "P2_COMPLETE.flag"
flag = {}
if not flag_path.exists():
    missing.append("results/p2_rich_motion/P2_COMPLETE.flag")
else:
    try:
        flag = json.loads(flag_path.read_text())
        terminal = str(flag.get("terminal_decision", ""))
    except Exception:
        missing.append("results/p2_rich_motion/P2_COMPLETE.flag:valid_json")

require_file("P2_RICH_MOTION_FINAL_REPORT.md")
require_file("P2_ENVIRONMENT_SANITY_REPORT.md")
random_rows = rows("results/p2_rich_motion/env_sanity_random_policy.csv")
short_rows = rows("results/p2_rich_motion/env_sanity_short_ppo.csv")
if random_rows != 5:
    missing.append("env_sanity_random_policy.csv:5_rows")
if short_rows != 5:
    missing.append("env_sanity_short_ppo.csv:5_rows")
require_log("runs/logs/train_p2_stage0_short_p2_short_attention_full_s0.log", "TRAIN_END")

valid_terminals = {
    "stage0_no_go_environment",
    "stage1_no_go_ood",
    "stage3_no_go_seed0",
    "stage4_complete",
}
if terminal and terminal not in valid_terminals:
    missing.append("P2_COMPLETE.flag:known_terminal_decision")

if terminal in {"stage1_no_go_ood", "stage3_no_go_seed0", "stage4_complete"}:
    require_file("P2_STAGE1_OOD_EVAL_REPORT.md")
    stage1_rows = rows("results/p2_rich_motion/p2_stage1_ood_eval.csv")
    if stage1_rows != 40:
        missing.append("p2_stage1_ood_eval.csv:40_rows")
    plot_count = len(list((out / "plots").glob("stage1_*.png")))
    if plot_count < 5:
        missing.append("plots/stage1_*.png:>=5")

if terminal in {"stage3_no_go_seed0", "stage4_complete"}:
    require_file("P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md")
    require_file("P2_STAGE3_SEED0_PARETO_REPORT.md")
    seed0_rows = rows("results/p2_rich_motion/p2_seed0_by_step_scenario.csv")
    main_rows = rows("results/p2_rich_motion/p2_seed0_main_750k_1000k_table.csv")
    adapt_rows = rows("results/p2_rich_motion/p2_risk_adaptation_summary.csv")
    gate_rows = rows("results/p2_rich_motion/p2_stage3_gate_by_scenario.csv")
    if seed0_rows != 96:
        missing.append("p2_seed0_by_step_scenario.csv:96_rows")
    if main_rows != 48:
        missing.append("p2_seed0_main_750k_1000k_table.csv:48_rows")
    if adapt_rows != 24:
        missing.append("p2_risk_adaptation_summary.csv:24_rows")
    if gate_rows < 6:
        missing.append("p2_stage3_gate_by_scenario.csv:>=6_rows")
    for method in ["attention_full", "attention_full_distance_penalty_wide_d2", "attention_full_risk_penalty"]:
        require_log(f"runs/logs/train_p2_stage2_{method}_s0.log", "TRAIN_END")
    pareto_count = len(list((out / "plots").glob("seed0_pareto_*.png")))
    if pareto_count < 30:
        missing.append("plots/seed0_pareto_*.png:>=30")

if terminal == "stage4_complete":
    require_file("P2_THREE_SEED_CONFIRMATION_REPORT.md")
    three_rows = rows("results/p2_rich_motion/p2_three_seed_summary.csv")
    if three_rows != 192:
        missing.append("p2_three_seed_summary.csv:192_rows")
    for method in ["attention_full_distance_penalty_wide_d2", "attention_full_risk_penalty"]:
        for seed in [1, 2]:
            require_log(f"runs/logs/train_p2_stage4_{method}_s{seed}.log", "TRAIN_END")

complete = bool(terminal) and not missing
print(json.dumps({
    "complete": complete,
    "terminal_decision": terminal,
    "missing": missing[:20],
    "missing_count": len(missing),
    "row_counts": row_counts,
}, sort_keys=True))
PY
)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] status=${STATUS_JSON}" | tee -a "$LOG_PATH"
  if python - "$STATUS_JSON" <<'PY'
import json, sys
sys.exit(0 if json.loads(sys.argv[1]).get("complete") else 1)
PY
  then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE P2 artifacts verified" | tee -a "$LOG_PATH"
    exit 0
  fi
  sleep "$INTERVAL"
done
