#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG="${WATCH_LOG:-runs/logs/pre_ppo_priority_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] pre_ppo_priority_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  status="$(python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(".")
complete_flag = root / "PRE_PPO_PRIORITY_COMPLETE.flag"
no_go_flag = root / "PRE_PPO_PRIORITY_NO_GO.flag"
status_path = root / "results/pre_ppo_priority/status.json"

required_complete = [
    "P_MINUS_1_P2_PARETO_AUDIT_REPORT.md",
    "P0_ADAPTATION_VALIDATION_REPORT.md",
    "P0_5_BETA_COST_SCALE_SWEEP_REPORT.md",
    "PRE_PPO_PRIORITY_FINAL_REPORT.md",
    "results/pre_ppo_priority/pminus1_pareto_audit/p2_delta_by_seed_step_scenario.csv",
    "results/pre_ppo_priority/pminus1_pareto_audit/p2_delta_summary_by_scenario.csv",
    "results/pre_ppo_priority/pminus1_pareto_audit/p2_pareto_classification.csv",
    "results/pre_ppo_priority/p0_adaptation/p0_adaptation_by_variant_method_step_scenario.csv",
    "results/pre_ppo_priority/p0_adaptation/p0_adaptation_summary.csv",
    "results/pre_ppo_priority/p0_adaptation/p0_adaptation_delta_risk_minus_wide_d2.csv",
    "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_sweep_by_method_step_scenario.csv",
    "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_pareto_summary.csv",
    "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_delta_table.csv",
]

terminal_choices = {
    "risk_adaptation_supported",
    "risk_pareto_but_not_adaptive",
    "distance_margin_explains_risk",
    "risk_mainline_downgrade",
}
no_go_choices = {
    "pminus1_no_go_risk_not_pareto",
    "p0_no_go_environment_invalid",
    "p0_no_go_both_methods_fail_all_variants",
    "p05_no_go_training_unstable_or_nan",
}

def read_status():
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def rows(path):
    p = root / path
    if not p.exists() or p.stat().st_size == 0:
        return -1
    try:
        return len(pd.read_csv(p))
    except Exception:
        return -1

def terminal_from_flag(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.startswith("terminal_decision="):
            return line.split("=", 1)[1].strip()
    try:
        payload = json.loads(text)
        return str(payload.get("terminal_decision", ""))
    except Exception:
        return ""

missing = []
row_counts = {}
if complete_flag.exists():
    for item in required_complete:
        p = root / item
        if not p.exists() or p.stat().st_size == 0:
            missing.append(item)
    expected_rows = {
        "results/pre_ppo_priority/pminus1_pareto_audit/p2_delta_by_seed_step_scenario.csv": 48,
        "results/pre_ppo_priority/pminus1_pareto_audit/p2_delta_summary_by_scenario.csv": 8,
        "results/pre_ppo_priority/pminus1_pareto_audit/p2_pareto_classification.csv": 48,
        "results/pre_ppo_priority/p0_adaptation/p0_adaptation_by_variant_method_step_scenario.csv": 96,
        "results/pre_ppo_priority/p0_adaptation/p0_adaptation_delta_risk_minus_wide_d2.csv": 48,
        "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_sweep_by_method_step_scenario.csv": 144,
        "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_delta_table.csv": 96,
    }
    for item, minimum in expected_rows.items():
        count = rows(item)
        row_counts[item] = count
        if count < minimum:
            missing.append(f"{item}:rows<{minimum}:actual={count}")
    try:
        p0 = pd.read_csv(root / "results/pre_ppo_priority/p0_adaptation/p0_adaptation_summary.csv")
        row_counts["p0_summary"] = len(p0)
        if len(p0) < 12:
            missing.append("p0_adaptation_summary.csv:rows<12")
    except Exception:
        missing.append("p0_adaptation_summary.csv:readable")
    try:
        p05 = pd.read_csv(root / "results/pre_ppo_priority/p0_5_beta_sweep/p0_5_beta_pareto_summary.csv")
        row_counts["p05_pareto_summary"] = len(p05)
        if len(p05) < 3:
            missing.append("p0_5_beta_pareto_summary.csv:rows<3")
    except Exception:
        missing.append("p0_5_beta_pareto_summary.csv:readable")
    terminal = terminal_from_flag(complete_flag)
    complete = (not missing) and terminal in terminal_choices
    payload = {
        "mode": "complete_check",
        "complete": complete,
        "terminal_decision": terminal,
        "missing_count": len(missing),
        "missing": missing[:20],
        "row_counts": row_counts,
        **read_status(),
    }
elif no_go_flag.exists():
    terminal = terminal_from_flag(no_go_flag)
    final_report = root / "PRE_PPO_PRIORITY_FINAL_REPORT.md"
    final_text = final_report.read_text(encoding="utf-8", errors="ignore") if final_report.exists() else ""
    no_go_complete = (
        terminal in no_go_choices
        and final_report.exists()
        and final_report.stat().st_size > 0
        and "Stage" in final_text
        and "Trigger Metrics" in final_text
        and "Why Stop" in final_text
        and "Recommendation" in final_text
    )
    payload = {
        "mode": "no_go_check",
        "no_go": no_go_complete,
        "terminal_decision": terminal,
        "missing_count": 0 if no_go_complete else 1,
        "missing": [] if no_go_complete else ["no-go final report incomplete or invalid reason"],
        **read_status(),
    }
else:
    payload = {
        "mode": "running",
        "complete": False,
        "no_go": False,
        **read_status(),
    }

print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
PY
)"

  echo "[$(date '+%F %T')] status=${status}" | tee -a "${LOG}"
  terminal="$(python - <<PY
import json
data=json.loads('''${status}''')
print(data.get("terminal_decision",""))
PY
)"
  complete="$(python - <<PY
import json
data=json.loads('''${status}''')
print('1' if data.get('complete') else '0')
PY
)"
  no_go="$(python - <<PY
import json
data=json.loads('''${status}''')
print('1' if data.get('no_go') else '0')
PY
)"

  if [[ "${complete}" == "1" ]]; then
    echo "COMPLETE pre-PPO priority artifacts verified"
    echo "terminal_decision = ${terminal}"
    exit 0
  fi
  if [[ "${no_go}" == "1" ]]; then
    echo "NO-GO triggered: ${terminal}"
    echo "terminal_decision = ${terminal}"
    exit 0
  fi

  python - <<PY
import json
data=json.loads('''${status}''')
print(f"[current_stage] {data.get('current_stage','unknown')}")
print(f"[current_run] {data.get('current_run','unknown')}")
print(f"[latest_checkpoint] {data.get('latest_checkpoint','unknown')}")
print(f"[completed_eval_count] {data.get('completed_eval_count',0)}")
print(f"[pending_eval_count] {data.get('pending_eval_count',0)}")
print(f"[watcher_status] {data.get('watcher_status', data.get('mode','running'))}")
PY
  sleep "${INTERVAL}"
done
