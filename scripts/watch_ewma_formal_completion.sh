#!/usr/bin/env bash
set -euo pipefail

EXPECTED_EVAL=108
EXPECTED_DIAG=36
REPORT=EWMA_RISK_FORMAL_RECHECK_REPORT.md
SUMMARY_DIR=results/ewma_formal/summary

report_complete() {
  [[ -f "${REPORT}" ]] || return 1
  [[ -f "${SUMMARY_DIR}/ewma_formal_all_results.csv" ]] || return 1
  [[ -f "${SUMMARY_DIR}/ewma_formal_by_config_step.csv" ]] || return 1
  [[ -f "${SUMMARY_DIR}/ewma_formal_best_checkpoint.csv" ]] || return 1
  grep -q "^# 修正后 EWMA-Risk 正式复验报告" "${REPORT}" || return 1
  grep -q "^## 9. 最终判断" "${REPORT}" || return 1
  grep -q "best corrected EWMA-risk" "${REPORT}" || return 1
  return 0
}

while true
do
  EVAL_COUNT=$(find results/ewma_formal/eval -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
  DIAG_COUNT=$(find results/ewma_formal/diagnostics -path '*/summary/*episode_summary.csv' 2>/dev/null | wc -l | tr -d ' ')
  RUNNING=$(ps -eo cmd | rg -c 'run_ewma_formal_(train|eval|diagnostics)|python (train|eval)\.py .*ewma_formal|diagnose_sudden_turn.py --model_path checkpoints/ewma_formal' || true)
  TS=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[${TS}] WATCH eval=${EVAL_COUNT}/${EXPECTED_EVAL} diag=${DIAG_COUNT}/${EXPECTED_DIAG} running=${RUNNING}"

  if [[ "${EVAL_COUNT}" -ge "${EXPECTED_EVAL}" && "${DIAG_COUNT}" -ge "${EXPECTED_DIAG}" ]]; then
    python scripts/aggregate_ewma_formal_results.py
    if report_complete; then
      echo "[${TS}] WATCH_COMPLETE report=${REPORT}"
      exit 0
    fi
  fi

  sleep 30
done
