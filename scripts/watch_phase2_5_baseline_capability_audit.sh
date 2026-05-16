#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${WATCH_INTERVAL:-10}"
LOG="${WATCH_LOG:-${ROOT}/results/env_v2_phase2_5/phase2_5_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] phase2_5_baseline_capability_audit_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  set +e
  python - <<'PY' "${ROOT}" | tee -a "${LOG}"
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
out_dir = root / "results/env_v2_phase2_5"
complete_flag = root / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_COMPLETE.flag"
no_go_flag = root / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_NO_GO.flag"
status_path = out_dir / "phase2_5_status.txt"

required_tables = [
    "baseline_comparison_summary.csv",
    "attention_full_checkpoint_summary.csv",
    "scenario_difficulty_table.csv",
    "collision_breakdown_by_scenario.csv",
    "collision_breakdown_by_threat_class.csv",
    "collision_breakdown_by_motion_mode.csv",
    "reactive_vs_attention_gap.csv",
    "method_route_recommendation.md",
]
required_plots = [
    "plots/success_collision_by_scenario.png",
    "plots/attention_checkpoint_curves.png",
    "plots/reactive_vs_attention_collision.png",
    "plots/near_miss_min_distance_by_scenario.png",
    "plots/collision_by_threat_class.png",
    "plots/collision_by_motion_mode.png",
]
required_report = root / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_REPORT.md"
allowed_decisions = {
    "baseline_undertrained_or_unstable",
    "baseline_learns_progress_but_not_safety",
    "specific_scenarios_dominate_failure",
    "reactive_prior_promising",
    "env_or_reward_needs_revision",
    "data_insufficient_for_decision",
}
allowed_no_go = {
    "phase2_5_no_go_missing_phase1_sanity_data",
    "phase2_5_no_go_missing_phase2_eval_data",
    "phase2_5_no_go_metrics_incompatible",
    "phase2_5_no_go_episode_level_data_missing",
    "phase2_5_no_go_audit_inconclusive_requires_extra_eval",
}

completed_tables = []
pending_tables = []
for item in required_tables:
    path = out_dir / item
    if path.exists() and path.stat().st_size > 0:
        completed_tables.append(item)
    else:
        pending_tables.append(item)
completed_plots = []
pending_plots = []
for item in required_plots:
    path = out_dir / item
    if path.exists() and path.stat().st_size > 0:
        completed_plots.append(item)
    else:
        pending_plots.append(item)
status_text = status_path.read_text(encoding="utf-8", errors="replace").strip() if status_path.exists() else "not_started"
current_file = status_text


def emit(watcher_status: str) -> None:
    print(
        f"[current_stage] Phase2.5 baseline capability audit\n"
        f"[current_file_being_processed] {current_file}\n"
        f"[completed_tables] {len(completed_tables)}/{len(required_tables)} {','.join(completed_tables)}\n"
        f"[pending_tables] {','.join(pending_tables) if pending_tables else 'none'}\n"
        f"[completed_plots] {len(completed_plots)}/{len(required_plots)} {','.join(completed_plots)}\n"
        f"[pending_plots] {','.join(pending_plots) if pending_plots else 'none'}\n"
        f"[watcher_status] {watcher_status}",
        flush=True,
    )


def parse_flag(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


if no_go_flag.exists():
    values = parse_flag(no_go_flag)
    decision = values.get("terminal_decision", "")
    report_ok = required_report.exists() and "NO-GO triggered" in required_report.read_text(encoding="utf-8", errors="replace")
    if decision in allowed_no_go and report_ok and values.get("next_recommended_action"):
        emit("no_go_verified")
        print(f"NO-GO triggered: {decision}", flush=True)
        print(f"terminal_decision = {decision}", flush=True)
        print(f"next_recommended_action = {values.get('next_recommended_action')}", flush=True)
        raise SystemExit(2)
    emit(f"no_go_flag_present_but_incomplete decision={decision} report_ok={report_ok}")
    raise SystemExit(0)

if complete_flag.exists():
    values = parse_flag(complete_flag)
    decision = values.get("terminal_decision", "")
    next_action = values.get("next_recommended_action", "")
    missing = []
    for item in required_tables + required_plots:
        path = out_dir / item
        if not path.exists() or path.stat().st_size == 0:
            missing.append(item)
    if not required_report.exists() or required_report.stat().st_size == 0:
        missing.append("PHASE2_5_BASELINE_CAPABILITY_AUDIT_REPORT.md")
    checks = {}
    try:
        comparison = pd.read_csv(out_dir / "baseline_comparison_summary.csv")
        checkpoint = pd.read_csv(out_dir / "attention_full_checkpoint_summary.csv")
        scenario = pd.read_csv(out_dir / "scenario_difficulty_table.csv")
        scenario_breakdown = pd.read_csv(out_dir / "collision_breakdown_by_scenario.csv")
        threat = pd.read_csv(out_dir / "collision_breakdown_by_threat_class.csv")
        motion = pd.read_csv(out_dir / "collision_breakdown_by_motion_mode.csv")
        gap = pd.read_csv(out_dir / "reactive_vs_attention_gap.csv")
        checks["comparison_rows_ok"] = len(comparison) >= 5
        checks["checkpoint_rows_ok"] = len(checkpoint) == 6
        checks["scenario_rows_ok"] = len(scenario) == 6 and len(scenario_breakdown) == 6
        checks["threat_motion_rows_ok"] = len(threat) > 0 and len(motion) > 0
        checks["gap_rows_ok"] = len(gap) == 6
        checks["method_route_keys_ok"] = all(
            key in (out_dir / "method_route_recommendation.md").read_text(encoding="utf-8", errors="replace")
            for key in ["primary_diagnosis", "secondary_diagnosis", "recommended_next_step", "not_recommended_next_steps"]
        )
        checks["report_sections_ok"] = all(
            section in required_report.read_text(encoding="utf-8", errors="replace")
            for section in [
                "## 1. Executive Summary",
                "## 2. Inputs",
                "## 3. Attention_full Learning Audit",
                "## 4. Comparison with Random / Straight-Line / Reactive",
                "## 5. Scenario Difficulty",
                "## 6. Collision Breakdown",
                "## 7. Interpretation",
                "## 8. Method Route Recommendation",
                "## 9. Decision",
            ]
        )
    except Exception as exc:
        emit(f"verification_error={exc!r}")
        raise SystemExit(0)
    all_checks = all(checks.values())
    if not missing and decision in allowed_decisions and next_action and all_checks:
        emit("complete_verified")
        print("COMPLETE Phase2.5 baseline capability audit artifacts verified", flush=True)
        print(f"terminal_decision = {decision}", flush=True)
        print(f"next_recommended_action = {next_action}", flush=True)
        raise SystemExit(10)
    emit(f"complete_flag_present_but_checks_failed missing={missing} decision={decision} checks={checks}")
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
