#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG="${WATCH_LOG:-runs/logs/p1_5_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] p1_5_watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  status="$(python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(".")
missing = []

required = [
    "P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md",
    "results/p1_5_distance_wide/p1_5_by_seed_step_scenario.csv",
    "results/p1_5_distance_wide/p1_5_summary_by_method_step_scenario.csv",
    "results/p1_5_distance_wide/p1_5_main_750k_table.csv",
    "results/p1_5_distance_wide/p1_5_pareto_summary.csv",
    "results/p1_5_distance_wide/d_warning_sweep_seed0.csv",
    "results/p1_5_distance_wide/P1_5_COMPLETE.flag",
]
for item in required:
    path = root / item
    if not path.exists() or path.stat().st_size == 0:
        missing.append(item)

by_seed_rows = summary_rows = main_rows = pareto_rows = sweep_rows = 0
plot_count = len(list((root / "results/p1_5_distance_wide/plots").glob("*.png"))) if (root / "results/p1_5_distance_wide/plots").exists() else 0
sweep_plot_count = len(list((root / "results/p1_5_distance_wide/d_warning_sweep_seed0_plots").glob("*.png"))) if (root / "results/p1_5_distance_wide/d_warning_sweep_seed0_plots").exists() else 0

try:
    by_seed = pd.read_csv(root / "results/p1_5_distance_wide/p1_5_by_seed_step_scenario.csv")
    by_seed_rows = len(by_seed)
    required_cols = {
        "method", "seed", "step", "scenario", "d_warning", "success_rate", "collision_rate",
        "reaction_time_eval_style", "reaction_time_nan_style", "nan_reaction_rate",
        "mean_min_distance", "near_miss_rate", "distance_warning_cost_nonzero_rate",
        "distance_warning_cost_mean", "distance_warning_cost_p90", "distance_warning_cost_p95",
        "distance_warning_cost_max", "risk_sum_mean", "risk_max_mean", "mean_time",
        "checkpoint_path", "eval_csv_path",
    }
    if by_seed_rows != 208 or not required_cols.issubset(set(by_seed.columns)):
        missing.append("p1_5_by_seed_step_scenario.csv:208_rows_required_columns")
    else:
        expected = {
            ("attention_full", 0), ("attention_full", 1), ("attention_full", 2),
            ("attention_full_distance_penalty_d1", 0), ("attention_full_distance_penalty_d1", 1), ("attention_full_distance_penalty_d1", 2),
            ("attention_full_distance_penalty_wide_d2", 0), ("attention_full_distance_penalty_wide_d2", 1), ("attention_full_distance_penalty_wide_d2", 2),
            ("attention_full_risk_penalty", 0), ("attention_full_risk_penalty", 1), ("attention_full_risk_penalty", 2),
            ("attention_full_distance_penalty_mid_d15", 0),
        }
        seen = set(zip(by_seed["method"].astype(str), by_seed["seed"].astype(int)))
        if seen != expected:
            missing.append("p1_5_by_seed_step_scenario.csv:expected_method_seed_set")
        counts = by_seed.groupby(["method", "seed", "step"]).size()
        if len(counts) != 52 or counts.min() != 4 or counts.max() != 4:
            missing.append("p1_5_by_seed_step_scenario.csv:complete_step_scenario_grid")
        for p in by_seed["checkpoint_path"].astype(str):
            fp = root / p
            if not fp.exists() or fp.stat().st_size == 0:
                missing.append(f"missing_checkpoint:{p}")
                break
        for p in by_seed["eval_csv_path"].astype(str):
            fp = root / p
            if not fp.exists() or fp.stat().st_size == 0:
                missing.append(f"missing_eval_csv:{p}")
                break
except Exception:
    if (root / "results/p1_5_distance_wide/p1_5_by_seed_step_scenario.csv").exists():
        missing.append("p1_5_by_seed_step_scenario.csv:readable")

try:
    summary = pd.read_csv(root / "results/p1_5_distance_wide/p1_5_summary_by_method_step_scenario.csv")
    summary_rows = len(summary)
    required_cols = {"method", "step", "scenario", "seed_count", "mean_success_rate", "mean_reaction_time_eval_style", "mean_mean_time"}
    if summary_rows != 80 or not required_cols.issubset(set(summary.columns)):
        missing.append("p1_5_summary_by_method_step_scenario.csv:80_rows_required_columns")
except Exception:
    if (root / "results/p1_5_distance_wide/p1_5_summary_by_method_step_scenario.csv").exists():
        missing.append("p1_5_summary_by_method_step_scenario.csv:readable")

try:
    main = pd.read_csv(root / "results/p1_5_distance_wide/p1_5_main_750k_table.csv")
    main_rows = len(main)
    required_cols = {
        "method", "seed", "step", "sudden_reaction", "sudden_success", "sudden_collision",
        "sudden_min_distance", "sudden_near_miss", "random_success", "random_mean_time",
        "random_min_distance", "random_near_miss", "hard_success", "hard_collision",
        "hard_near_miss", "mixed_success", "mixed_collision", "mixed_near_miss",
    }
    if main_rows != 13 or not required_cols.issubset(set(main.columns)):
        missing.append("p1_5_main_750k_table.csv:13_rows_required_columns")
except Exception:
    if (root / "results/p1_5_distance_wide/p1_5_main_750k_table.csv").exists():
        missing.append("p1_5_main_750k_table.csv:readable")

try:
    pareto = pd.read_csv(root / "results/p1_5_distance_wide/p1_5_pareto_summary.csv")
    pareto_rows = len(pareto)
    required_cols = {"method", "seed", "step", "sudden_reaction", "random_mean_time", "hard_near_miss", "mixed_collision", "is_750k"}
    if pareto_rows != 52 or not required_cols.issubset(set(pareto.columns)):
        missing.append("p1_5_pareto_summary.csv:52_rows_required_columns")
except Exception:
    if (root / "results/p1_5_distance_wide/p1_5_pareto_summary.csv").exists():
        missing.append("p1_5_pareto_summary.csv:readable")

try:
    sweep = pd.read_csv(root / "results/p1_5_distance_wide/d_warning_sweep_seed0.csv")
    sweep_rows = len(sweep)
    required_cols = {"method", "seed", "step", "scenario", "d_warning", "sweep_label"}
    if sweep_rows != 64 or not required_cols.issubset(set(sweep.columns)):
        missing.append("d_warning_sweep_seed0.csv:64_rows_required_columns")
except Exception:
    if (root / "results/p1_5_distance_wide/d_warning_sweep_seed0.csv").exists():
        missing.append("d_warning_sweep_seed0.csv:readable")

if plot_count < 6:
    missing.append("results/p1_5_distance_wide/plots:>=6_png")
if sweep_plot_count < 3:
    missing.append("results/p1_5_distance_wide/d_warning_sweep_seed0_plots:>=3_png")

for method, seed in [
    ("attention_full_distance_penalty_wide_d2", 1),
    ("attention_full_distance_penalty_wide_d2", 2),
    ("attention_full_distance_penalty_mid_d15", 0),
]:
    log = root / f"runs/logs/train_p1_5_{method}_s{seed}.log"
    if not log.exists() or "TRAIN_END" not in log.read_text(encoding="utf-8", errors="ignore"):
        missing.append(f"train_log_complete:{log}")
        break

try:
    report = (root / "P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md").read_text(encoding="utf-8")
    for phrase in [
        "## 1. Purpose",
        "## 4. Main 750k Comparison",
        "## 7. Optional d_warning=1.5 Sweep",
        "## Required Answers",
        "P1.5 tests whether the risk_penalty advantage",
    ]:
        if phrase not in report:
            missing.append(f"report_missing:{phrase}")
            break
except Exception:
    if (root / "P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md").exists():
        missing.append("P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md:readable")

try:
    flag = json.loads((root / "results/p1_5_distance_wide/P1_5_COMPLETE.flag").read_text())
    if int(flag.get("by_seed_rows", 0)) != 208 or int(flag.get("summary_rows", 0)) != 80:
        missing.append("P1_5_COMPLETE.flag:manifest")
except Exception:
    if (root / "results/p1_5_distance_wide/P1_5_COMPLETE.flag").exists():
        missing.append("P1_5_COMPLETE.flag:valid_json")

payload = {
    "complete": len(missing) == 0,
    "missing": missing[:30],
    "missing_count": len(missing),
    "by_seed_rows": by_seed_rows,
    "summary_rows": summary_rows,
    "main_rows": main_rows,
    "pareto_rows": pareto_rows,
    "sweep_rows": sweep_rows,
    "plot_count": plot_count,
    "sweep_plot_count": sweep_plot_count,
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
    echo "[$(date '+%F %T')] COMPLETE P1.5 artifacts verified" | tee -a "${LOG}"
    exit 0
  fi
  sleep "${INTERVAL}"
done
