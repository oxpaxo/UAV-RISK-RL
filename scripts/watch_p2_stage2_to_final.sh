#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG_PATH="${WATCH_LOG:-$ROOT/runs/logs/p2_stage2_to_final_watcher.log}"
RUN_LOG="$ROOT/runs/logs/p2_stage2_to_final_runner.log"
COMPLETE_FLAG="$ROOT/results/p2_rich_motion/P2_STAGE2_TO_FINAL_COMPLETE.flag"
NO_GO_FLAG="$ROOT/results/p2_rich_motion/P2_STAGE2_TO_FINAL_NO_GO.flag"

mkdir -p "$(dirname "$LOG_PATH")"
cd "$ROOT" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] p2_stage2_to_final_watcher_start interval=${INTERVAL}" | tee -a "$LOG_PATH"

find_runner() {
  pgrep -f "python scripts/run_p2_stage2_to_final.py" | head -n 1 || true
}

start_or_attach_runner() {
  RUNNER_PID="$(find_runner)"
  RUNNER_IS_CHILD="false"
  if [[ -n "$RUNNER_PID" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] existing_runner_detected pid=${RUNNER_PID}; monitoring only" | tee -a "$LOG_PATH"
    return
  fi
  python scripts/run_p2_stage2_to_final.py >"$RUN_LOG" 2>&1 &
  RUNNER_PID="$!"
  RUNNER_IS_CHILD="true"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] runner_started pid=${RUNNER_PID} log=${RUN_LOG}" | tee -a "$LOG_PATH"
}

status_json() {
  python - <<'PY'
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
out = root / "results/p2_rich_motion"
logs = root / "runs/logs"
methods = [
    "attention_full",
    "attention_full_distance_penalty_wide_d2",
    "attention_full_risk_penalty",
]
stage4_methods = [
    "attention_full_distance_penalty_wide_d2",
    "attention_full_risk_penalty",
]
steps = [250000, 500000, 750000, 1000000]
scenarios = [
    "eval_random_switch",
    "eval_sudden_turn",
    "eval_random_switch_hard",
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_mixed_v2",
    "eval_threat_validated_sudden",
]
suffix = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "eval_sinusoidal": "sinusoidal",
    "eval_accel_decel": "accel",
    "eval_ar1": "ar1",
    "eval_mixed_v2": "mixed_v2",
    "eval_threat_validated_sudden": "threat_sudden",
}

def rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0

def marker(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8", errors="ignore")

def latest_checkpoint(method: str, seed: int) -> int:
    latest = 0
    for step in steps:
        if (root / f"checkpoints/p2_rich_motion/{method}_s{seed}_step{step}.zip").exists():
            latest = step
    return latest

stage2_train_done = sum(marker(logs / f"train_p2_stage2_{m}_s0.log", "TRAIN_END") for m in methods)
latest = {m: latest_checkpoint(m, 0) for m in methods}
expected_stage2_eval = len(methods) * len(steps) * len(scenarios)
completed_stage2_eval = 0
for method in methods:
    for step in steps:
        for scenario in scenarios:
            if rows(out / "stage2_eval" / f"{method}_s0_step{step}_{suffix[scenario]}.csv") >= 50:
                completed_stage2_eval += 1

expected_stage4_eval = len(stage4_methods) * 2 * len(steps) * len(scenarios)
completed_stage4_eval = 0
stage4_started = rows(out / "p2_three_seed_summary.csv") > 0
for method in stage4_methods:
    for seed in [1, 2]:
        if marker(logs / f"train_p2_stage4_{method}_s{seed}.log", "TRAIN_START"):
            stage4_started = True
        for step in steps:
            for scenario in scenarios:
                if rows(out / "stage4_eval" / f"{method}_s{seed}_step{step}_{suffix[scenario]}.csv") >= 50:
                    completed_stage4_eval += 1

complete_flag = out / "P2_STAGE2_TO_FINAL_COMPLETE.flag"
no_go_flag = out / "P2_STAGE2_TO_FINAL_NO_GO.flag"
terminal_flag = complete_flag if complete_flag.exists() else no_go_flag if no_go_flag.exists() else None
terminal_decision = ""
terminal_name = ""
if terminal_flag:
    terminal_name = terminal_flag.name
    try:
        terminal_decision = json.loads(terminal_flag.read_text()).get("terminal_decision", "")
    except Exception:
        terminal_decision = "invalid_flag_json"

current_stage = "Stage 2"
if rows(out / "p2_seed0_by_step_scenario.csv") >= expected_stage2_eval:
    current_stage = "Stage 3"
if stage4_started:
    current_stage = "Stage 4"
if terminal_flag:
    current_stage = "Terminal"

current_run = ""
for method in methods:
    if not marker(logs / f"train_p2_stage2_{method}_s0.log", "TRAIN_END"):
        current_run = f"stage2:{method}:seed0"
        break
if stage4_started and not terminal_flag:
    for method in stage4_methods:
        for seed in [1, 2]:
            if not marker(logs / f"train_p2_stage4_{method}_s{seed}.log", "TRAIN_END"):
                current_run = f"stage4:{method}:seed{seed}"
                break
        if current_run.startswith("stage4:"):
            break

required = [
    "P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md",
    "P2_RICH_MOTION_FINAL_REPORT.md",
    "results/p2_rich_motion/p2_seed0_by_step_scenario.csv",
    "results/p2_rich_motion/p2_seed0_main_750k_1000k_table.csv",
    "results/p2_rich_motion/p2_risk_adaptation_summary.csv",
]
missing = []
for rel in required:
    path = root / rel
    if not path.exists() or path.stat().st_size == 0:
        missing.append(rel)
if len(glob.glob(str(out / "plots/seed0_pareto_*.png"))) < 35:
    missing.append("results/p2_rich_motion/plots/seed0_pareto_*.png")
if terminal_decision == "stage4_complete":
    for rel in ["P2_THREE_SEED_CONFIRMATION_REPORT.md", "results/p2_rich_motion/p2_three_seed_summary.csv"]:
        path = root / rel
        if not path.exists() or path.stat().st_size == 0:
            missing.append(rel)

if stage4_started:
    completed_eval = completed_stage4_eval
    pending_eval = max(expected_stage4_eval - completed_stage4_eval, 0)
else:
    completed_eval = completed_stage2_eval
    pending_eval = max(expected_stage2_eval - completed_stage2_eval, 0)

print(json.dumps({
    "current_stage": current_stage,
    "current_run": current_run,
    "latest_checkpoint": latest,
    "completed_eval_count": completed_eval,
    "pending_eval_count": pending_eval,
    "stage2_train_done": int(stage2_train_done),
    "stage2_train_total": len(methods),
    "watcher_status": "terminal_flag_detected" if terminal_flag else "running",
    "terminal_flag": terminal_name,
    "terminal_decision": terminal_decision,
    "missing_required_count": len(missing),
    "missing_required": missing[:10],
}, sort_keys=True))
PY
}

verify_terminal() {
  python - <<'PY'
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
out = root / "results/p2_rich_motion"
complete = out / "P2_STAGE2_TO_FINAL_COMPLETE.flag"
no_go = out / "P2_STAGE2_TO_FINAL_NO_GO.flag"
flag = complete if complete.exists() else no_go if no_go.exists() else None
if flag is None:
    sys.exit(1)
payload = json.loads(flag.read_text())
terminal = payload.get("terminal_decision", "")
stage4 = terminal == "stage4_complete"

def require_file(rel: str) -> None:
    path = root / rel
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing {rel}")

def require_rows(rel: str, expected: int) -> None:
    require_file(rel)
    count = len(pd.read_csv(root / rel))
    if count != expected:
        raise SystemExit(f"bad row count {rel}: {count} != {expected}")

require_file("P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md")
require_file("P2_RICH_MOTION_FINAL_REPORT.md")
require_rows("results/p2_rich_motion/p2_seed0_by_step_scenario.csv", 96)
require_rows("results/p2_rich_motion/p2_seed0_main_750k_1000k_table.csv", 48)
require_rows("results/p2_rich_motion/p2_risk_adaptation_summary.csv", 24)
if len(glob.glob(str(out / "plots/seed0_pareto_*.png"))) < 35:
    raise SystemExit("missing seed0 pareto plots")
if stage4:
    require_file("P2_THREE_SEED_CONFIRMATION_REPORT.md")
    require_rows("results/p2_rich_motion/p2_three_seed_summary.csv", 192)
print(terminal)
PY
}

if [[ ! -f "$COMPLETE_FLAG" && ! -f "$NO_GO_FLAG" ]]; then
  start_or_attach_runner
else
  RUNNER_PID=""
  RUNNER_IS_CHILD="false"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] terminal_flag_already_present" | tee -a "$LOG_PATH"
fi

while true; do
  STATUS_JSON="$(status_json)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] status=${STATUS_JSON}" | tee -a "$LOG_PATH"

  if [[ -f "$COMPLETE_FLAG" || -f "$NO_GO_FLAG" ]]; then
    if TERMINAL_DECISION="$(verify_terminal 2>>"$LOG_PATH")"; then
      if [[ -f "$COMPLETE_FLAG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE P2 Stage2-to-final artifacts verified" | tee -a "$LOG_PATH"
        echo "terminal_decision = ${TERMINAL_DECISION}" | tee -a "$LOG_PATH"
      else
        REASON="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("results/p2_rich_motion/P2_STAGE2_TO_FINAL_NO_GO.flag").read_text()).get("no_go_reason", "unspecified"))
PY
)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] NO-GO triggered: ${REASON}" | tee -a "$LOG_PATH"
        echo "terminal_decision = ${TERMINAL_DECISION}" | tee -a "$LOG_PATH"
      fi
      exit 0
    fi
  fi

  if [[ -n "${RUNNER_PID:-}" ]] && ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    if [[ "${RUNNER_IS_CHILD:-false}" == "true" ]]; then
      wait "$RUNNER_PID"
      RC="$?"
    else
      RC="unknown_non_child"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] runner_exited rc=${RC}; no verified terminal flag yet" | tee -a "$LOG_PATH"
    RUNNER_PID=""
    RUNNER_IS_CHILD="false"
  fi

  if [[ -z "${RUNNER_PID:-}" && ! -f "$COMPLETE_FLAG" && ! -f "$NO_GO_FLAG" ]]; then
    start_or_attach_runner
  fi

  sleep "$INTERVAL"
done
