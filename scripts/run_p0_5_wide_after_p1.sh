#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG="${P05_WIDE_QUEUE_LOG:-runs/logs/p0_5_distance_wide_queue.log}"
INTERVAL="${P05_WIDE_QUEUE_INTERVAL_SECONDS:-60}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date '+%F %T')] p0_5_wide_queue_start interval=${INTERVAL}" | tee -a "${LOG}"

while [[ ! -s results/P0_P1_COMPLETE.flag ]]; do
  echo "[$(date '+%F %T')] waiting_for_p0_p1_complete_flag" | tee -a "${LOG}"
  sleep "${INTERVAL}"
done

echo "[$(date '+%F %T')] p0_p1_complete_flag_seen; starting P0.5-C wide ablation" | tee -a "${LOG}"
python scripts/run_p0_5_distance_followup.py --mode wide 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] p0_5_wide_queue_complete" | tee -a "${LOG}"
