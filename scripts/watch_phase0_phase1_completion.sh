#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL:-10}"
LOG="${WATCH_LOG:-${ROOT}/results/restart_phase0_phase1/logs/phase0_phase1_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] phase0_phase1_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  set +e
  python - <<'PY' "${ROOT}" | tee -a "${LOG}"
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
required = [
    "OLD_EXPERIMENTS_ASSETS_SUMMARY.md",
    "ENV_V2_DESIGN_REPORT.md",
    "ENV_V2_SANITY_REPORT.md",
    "PHASE0_PHASE1_FINAL_REPORT.md",
    "results/restart_phase0_phase1/old_assets_index.csv",
    "results/restart_phase0_phase1/env_v2/env_v2_sanity.csv",
    "results/restart_phase0_phase1/env_v2/env_v2_sanity_by_policy_scenario.csv",
    "results/restart_phase0_phase1/env_v2/cpa_distribution.csv",
    "results/restart_phase0_phase1/env_v2/ttc_distribution.csv",
    "results/restart_phase0_phase1/env_v2/replacement_count_distribution.csv",
    "results/restart_phase0_phase1/env_v2/active_obstacle_count_distribution.csv",
]
complete_flag = root / "PHASE0_PHASE1_COMPLETE.flag"
no_go_flag = root / "PHASE0_PHASE1_NO_GO.flag"
missing = []
for item in required:
    path = root / item
    if not path.exists() or path.stat().st_size == 0:
        missing.append(item)

status = {
    "current_stage": "Phase0+Phase1",
    "current_task": "watching artifacts and sanity thresholds",
    "completed_files": len(required) - len(missing),
    "pending_files": ",".join(missing[:8]) if missing else "none",
    "current_sanity_policy": "n/a",
    "current_sanity_scenario": "n/a",
    "watcher_status": "waiting",
}

def emit() -> None:
    print(
        f"[current_stage] {status['current_stage']}\n"
        f"[current_task] {status['current_task']}\n"
        f"[completed_files] {status['completed_files']}/{len(required)}\n"
        f"[pending_files] {status['pending_files']}\n"
        f"[current_sanity_policy] {status['current_sanity_policy']}\n"
        f"[current_sanity_scenario] {status['current_sanity_scenario']}\n"
        f"[watcher_status] {status['watcher_status']}",
        flush=True,
    )

if no_go_flag.exists():
    text = no_go_flag.read_text(encoding="utf-8", errors="replace")
    final = root / "PHASE0_PHASE1_FINAL_REPORT.md"
    if final.exists() and "NO-GO triggered" in final.read_text(encoding="utf-8", errors="replace"):
        status["watcher_status"] = "no_go_verified"
        emit()
        print(text.strip(), flush=True)
        raise SystemExit(2)
    status["watcher_status"] = "no_go_flag_present_but_report_incomplete"
    emit()
    raise SystemExit(0)

if complete_flag.exists() and not missing:
    try:
        sanity = pd.read_csv(root / "results/restart_phase0_phase1/env_v2/env_v2_sanity.csv")
        summary = pd.read_csv(root / "results/restart_phase0_phase1/env_v2/env_v2_sanity_by_policy_scenario.csv")
        cpa = pd.read_csv(root / "results/restart_phase0_phase1/env_v2/cpa_distribution.csv")
        required_cols = {
            "policy_name",
            "scenario",
            "episode_id",
            "success",
            "collision",
            "near_miss",
            "replacement_count",
            "init_collision",
            "nan_or_crash",
            "threat_valid_rate",
        }
        if not required_cols.issubset(sanity.columns):
            status["watcher_status"] = "sanity_csv_missing_required_columns"
            emit()
            raise SystemExit(0)
        threat_valid_rate = float(cpa["threat_valid"].mean()) if len(cpa) else float(sanity["threat_valid_rate"].mean())
        replacement_mean = float(sanity["replacement_count"].mean())
        init_collision_rate = float(sanity["init_collision"].mean())
        nan_or_crash = int(sanity["nan_or_crash"].sum())
        straight = sanity[sanity["policy_name"] == "straight_line"]
        reactive = sanity[sanity["policy_name"] == "reactive"]
        random_df = sanity[sanity["policy_name"] == "random"]
        straight_success = float(straight["success"].mean())
        straight_collision = float(straight["collision"].mean())
        straight_near = float(straight["near_miss"].mean())
        reactive_success = float(reactive["success"].mean())
        reactive_collision = float(reactive["collision"].mean())
        random_success = float(random_df["success"].mean())
        high_mean = float(cpa[cpa["threat_class"] == "high"]["planned_cpa"].mean())
        low_mean = float(cpa[cpa["threat_class"] == "low"]["planned_cpa"].mean())
        reactive_better = (
            (reactive_success - straight_success >= 0.10 or straight_collision - reactive_collision >= 0.10)
            and reactive_success >= straight_success - 0.05
        )
        not_too_easy = not (
            random_success > 0.8
            or (straight_success > 0.9 and straight_collision < 0.05 and straight_near < 0.10)
        )
        not_too_hard = not (
            reactive_success < 0.25 and reactive_collision > 0.45 and abs(reactive_success - straight_success) < 0.10
        )
        checks = [
            threat_valid_rate >= 0.8,
            replacement_mean > 0.0,
            init_collision_rate <= 0.02,
            nan_or_crash == 0,
            reactive_better,
            not_too_easy,
            not_too_hard,
            low_mean - high_mean > 1.0,
            len(summary) == 18,
        ]
        status["current_sanity_policy"] = "all"
        status["current_sanity_scenario"] = "all"
        if all(checks):
            status["watcher_status"] = (
                "complete_verified "
                f"threat_valid_rate={threat_valid_rate:.4f} replacement_mean={replacement_mean:.4f} "
                f"init_collision_rate={init_collision_rate:.4f} nan_or_crash={nan_or_crash} "
                f"reactive_success={reactive_success:.4f} straight_success={straight_success:.4f}"
            )
            emit()
            print("COMPLETE Phase0+Phase1 artifacts verified", flush=True)
            print("terminal_decision = phase0_phase1_complete", flush=True)
            print("next_recommended_phase = Phase 2 baseline long-training reproduction", flush=True)
            raise SystemExit(10)
        status["watcher_status"] = (
            "complete_flag_present_but_checks_failed "
            f"threat_valid_rate={threat_valid_rate:.4f} replacement_mean={replacement_mean:.4f} "
            f"init_collision_rate={init_collision_rate:.4f} nan_or_crash={nan_or_crash} "
            f"reactive_better={reactive_better} not_too_easy={not_too_easy} not_too_hard={not_too_hard}"
        )
        emit()
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        status["watcher_status"] = f"verification_error={exc!r}"
        emit()
        raise SystemExit(0)

emit()
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
