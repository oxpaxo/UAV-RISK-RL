#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL="${WATCH_INTERVAL_SECONDS:-60}"
LOG="${WATCH_LOG:-runs/logs/longtrain_gate2b_watcher.log}"
mkdir -p "$(dirname "${LOG}")"

required=(
  "results/preflight/CODE_STATUS_PREFLIGHT.md"
  "results/preflight/RESUME_PREFLIGHT_REPORT.md"
  "results/preflight/RISK_CONFIG_PREFLIGHT_REPORT.md"
  "results/preflight/cost_stats_attention_gate.csv"
  "results/preflight/COST_SCALE_PREFLIGHT_REPORT.md"
  "results/preflight/reaction_definition_check.md"
  "results/preflight/TRACE_FIELD_PREFLIGHT_REPORT.md"
  "results/preflight/CHECKPOINT_EVAL_INDEX.csv"
  "results/preflight/SAFETY_COST_PREFLIGHT_REPORT.md"
  "results/preflight/RISK_BIASED_ATTENTION_PREFLIGHT_REPORT.md"
  "results/preflight/GATE2B_CONFIG_LOGGING_PREFLIGHT_REPORT.md"
  "results/longtrain_baseline/attention_vs_risk_longtrain_summary.csv"
  "results/gate2b/gate2b_by_step.csv"
  "results/gate2b/gate2b_curve_diagnostics_summary.csv"
  "results/attention_seed1/attention_seed1_by_step.csv"
  "ATTENTION_RISK_2000K_BASELINE_REPORT.md"
  "GATE2B_PENALTY_1000K_REPORT.md"
  "ATTENTION_SEED1_1000K_REPORT.md"
  "FINAL_DIRECTION_DECISION_REPORT.md"
  "results/PIPELINE_COMPLETE.flag"
)

echo "[$(date '+%F %T')] watcher_start interval=${INTERVAL}" | tee -a "${LOG}"

while true; do
  missing=()
  for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
      missing+=("${path}")
    fi
  done

  png_longtrain_count="$(find results/longtrain_baseline/plots -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
  png_gate2b_count="$(find results/gate2b/plots -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
  preflight_trace_count="$(find results/preflight/traces -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
  gate_trace_count="$(find results/gate2b/traces -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"

  if [[ "${png_longtrain_count}" -lt 1 ]]; then
    missing+=("results/longtrain_baseline/plots/*.png")
  fi
  if [[ "${png_gate2b_count}" -lt 1 ]]; then
    missing+=("results/gate2b/plots/*.png")
  fi
  if [[ "${preflight_trace_count}" -lt 1 ]]; then
    missing+=("results/preflight/traces/*.csv")
  fi
  if [[ "${gate_trace_count}" -lt 1 ]]; then
    missing+=("results/gate2b/traces/*.csv")
  fi

  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] COMPLETE all completion artifacts present" | tee -a "${LOG}"
    exit 0
  fi

  echo "[$(date '+%F %T')] waiting missing_count=${#missing[@]} longtrain_png=${png_longtrain_count} gate2b_png=${png_gate2b_count} preflight_trace=${preflight_trace_count} gate_trace=${gate_trace_count}" | tee -a "${LOG}"
  printf '  missing: %s\n' "${missing[@]:0:12}" | tee -a "${LOG}"
  sleep "${INTERVAL}"
done
