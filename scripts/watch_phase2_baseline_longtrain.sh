#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL:-20}"
LOG="${WATCH_LOG:-${ROOT}/results/env_v2_phase2/phase2_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] phase2_baseline_longtrain_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  set +e
  python - <<'PY' "${ROOT}" | tee -a "${LOG}"
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
complete_flag = root / "PHASE2_BASELINE_LONGTRAIN_COMPLETE.flag"
no_go_flag = root / "PHASE2_BASELINE_LONGTRAIN_NO_GO.flag"
ckpt_dir = root / "checkpoints/env_v2_phase2/attention_full_s0"
eval_dir = root / "results/env_v2_phase2/eval"
status_path = root / "results/env_v2_phase2/phase2_status.txt"
scenarios = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
base_steps = [250000, 500000, 750000, 1000000]
extra_steps = [1250000, 1500000]
required_artifacts = [
    "BASELINE_LONGTRAIN_ENV_V2_REPORT.md",
    "PHASE2_BASELINE_LONGTRAIN_FINAL_REPORT.md",
    "results/env_v2_phase2/baseline_longtrain_by_checkpoint_scenario.csv",
    "results/env_v2_phase2/baseline_longtrain_episode_metrics.csv",
    "results/env_v2_phase2/baseline_longtrain_reaction_breakdown.csv",
    "results/env_v2_phase2/baseline_longtrain_threat_metrics.csv",
    "results/env_v2_phase2/plots/no_response_curve.png",
    "results/env_v2_phase2/plots/near_miss_curve.png",
    "results/env_v2_phase2/plots/min_distance_curve.png",
    "results/env_v2_phase2/plots/collision_success_curve.png",
]
completed_ckpts = []
for step in base_steps + extra_steps:
    if (ckpt_dir / f"attention_full_s0_step{step}.zip").exists():
        completed_ckpts.append(step)
latest_checkpoint = max(completed_ckpts) if completed_ckpts else 0
eval_count = 0
for step in completed_ckpts:
    for scenario in scenarios:
        path = eval_dir / f"attention_full_s0_step{step}_{scenario}.csv"
        if path.exists() and path.stat().st_size > 0:
            eval_count += 1
expected_eval = 24
if latest_checkpoint >= 1250000:
    expected_eval = 36
pending_eval = max(expected_eval - eval_count, 0)
status_text = status_path.read_text(encoding="utf-8", errors="replace").strip() if status_path.exists() else "not_started"

def emit(watcher_status: str) -> None:
    print(
        f"[current_stage] Phase2 baseline long-training reproduction\n"
        f"[current_run] attention_full_s0_DynamicObstacleFlowEnv_train_flow_mixed\n"
        f"[latest_checkpoint] {latest_checkpoint}\n"
        f"[completed_eval_count] {eval_count}\n"
        f"[pending_eval_count] {pending_eval}\n"
        f"[watcher_status] {watcher_status}\n"
        f"[runner_status] {status_text}",
        flush=True,
    )

if no_go_flag.exists():
    final = root / "PHASE2_BASELINE_LONGTRAIN_FINAL_REPORT.md"
    text = no_go_flag.read_text(encoding="utf-8", errors="replace")
    if final.exists() and "NO-GO triggered" in final.read_text(encoding="utf-8", errors="replace"):
        emit("no_go_verified")
        print(text.strip(), flush=True)
        raise SystemExit(2)
    emit("no_go_flag_present_but_report_incomplete")
    raise SystemExit(0)

if complete_flag.exists():
    missing = []
    for item in required_artifacts:
        path = root / item
        if not path.exists() or path.stat().st_size == 0:
            missing.append(item)
    try:
        flag = complete_flag.read_text(encoding="utf-8", errors="replace")
        decision = ""
        for line in flag.splitlines():
            if line.startswith("terminal_decision="):
                decision = line.split("=", 1)[1].strip()
        valid_decision = decision in {
            "phase2_strong_reproduction_go_phase3",
            "phase2_weak_reproduction_go_phase3_with_limited_claim",
        }
        summary = pd.read_csv(root / "results/env_v2_phase2/baseline_longtrain_by_checkpoint_scenario.csv")
        episodes = pd.read_csv(root / "results/env_v2_phase2/baseline_longtrain_episode_metrics.csv")
        reaction = pd.read_csv(root / "results/env_v2_phase2/baseline_longtrain_reaction_breakdown.csv")
        threat = pd.read_csv(root / "results/env_v2_phase2/baseline_longtrain_threat_metrics.csv")
        required_cols = {
            "checkpoint_step",
            "scenario",
            "success_rate",
            "collision_rate",
            "near_miss_rate",
            "no_response_rate",
            "mean_min_distance",
            "min_distance_after_threat",
            "reaction_time_eval_style",
            "conditional_reaction_time",
            "nan_reaction_rate",
            "replacement_count",
            "distance_warning_cost_nonzero_rate",
        }
        cols_ok = required_cols.issubset(summary.columns)
        row_count_ok = len(summary) in {24, 36}
        episode_count_ok = len(episodes) in {1200, 1800}
        reaction_ok = len(reaction) == len(summary)
        threat_ok = len(threat) > 0
        if not missing and valid_decision and cols_ok and row_count_ok and episode_count_ok and reaction_ok and threat_ok:
            emit("complete_verified")
            print("COMPLETE Phase2 baseline long-training reproduction artifacts verified", flush=True)
            print(f"terminal_decision = {decision}", flush=True)
            print("next_recommended_phase = Phase 3 failure localization", flush=True)
            raise SystemExit(10)
        emit(
            "complete_flag_present_but_checks_failed "
            f"missing={missing} valid_decision={valid_decision} cols_ok={cols_ok} "
            f"row_count_ok={row_count_ok} episode_count_ok={episode_count_ok} reaction_ok={reaction_ok} threat_ok={threat_ok}"
        )
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        emit(f"verification_error={exc!r}")
        raise SystemExit(0)

emit("waiting")
PY
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 10 ]]; then
    exit 0
  fi
  if [[ "${rc}" -eq 2 ]]; then
    exit 2
  fi
  sleep "${INTERVAL}"
done
