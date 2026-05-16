#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n0_design_freeze"
LOG="$OUT_DIR/phase_n0_watcher.log"
STATUS="$OUT_DIR/phase_n0_status.txt"
COMPLETE_FLAG="$OUT_DIR/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag"

STOP_FLAGS=(
  "PHASE_N0_STOP_ENV_CORE_CHANGE_REQUIRED.flag"
  "PHASE_N0_STOP_OBSTACLE_ID_ALIGNMENT_FAILED.flag"
  "PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag"
  "PHASE_N0_STOP_REQUIRED_FIELDS_MISSING.flag"
  "PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag"
  "PHASE_N0_STOP_SPEC_CONFLICT.flag"
  "PHASE_N0_STOP_WATCHER_FAILED.flag"
)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/schema"
: > "$LOG"

echo "[watcher] Phase N0 watcher started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
echo "running" > "$STATUS"

rm -f "$COMPLETE_FLAG"
for flag in "${STOP_FLAGS[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

python scripts/check_envv2_gpsi_required_fields.py \
  --out-dir "$OUT_DIR" \
  --spec configs/gpsi_head_a_spec.yaml \
  --phase-a-dir results/env_v2_phase_a_eval_framework \
  --phase-b-dir results/env_v2_phase_b_geometry_filter_baselines \
  --scenarios eval_flow_id train_flow_mixed \
  --num-episodes 3 \
  --history-steps 20 \
  --future-times 1.0 2.0 4.0 \
  --write-dryrun-tables \
  2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$COMPLETE_FLAG" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    wait "$PID" || true
    exit 0
  fi

  for flag in "${STOP_FLAGS[@]}"; do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      wait "$PID" || true
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ -f "$COMPLETE_FLAG" ]]; then
      echo "complete" > "$STATUS"
      echo "[watcher] complete flag detected after runner exit" | tee -a "$LOG"
      exit 0
    fi
    for flag in "${STOP_FLAGS[@]}"; do
      if [[ -f "$OUT_DIR/$flag" ]]; then
        echo "stopped:$flag" > "$STATUS"
        echo "[watcher] stop flag detected after runner exit: $flag" | tee -a "$LOG"
        exit 2
      fi
    done
    touch "$OUT_DIR/PHASE_N0_STOP_WATCHER_FAILED.flag"
    echo "stopped:PHASE_N0_STOP_WATCHER_FAILED.flag" > "$STATUS"
    echo "[watcher] runner exited without complete or stop flag" | tee -a "$LOG"
    exit 2
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 20
done
