#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n1_gpsi_dataset"
DATA_DIR="data/gpsi_head_a_v1"
LOG="$OUT_DIR/phase_n1_watcher.log"
STATUS="$OUT_DIR/phase_n1_status.txt"
COMPLETE_FLAG="$OUT_DIR/PHASE_N1_GPSI_DATASET_COMPLETE.flag"

STOP_FLAGS=(
  "PHASE_N1_STOP_PHASE_N0_MISSING.flag"
  "PHASE_N1_STOP_ENV_CORE_CHANGE_REQUIRED.flag"
  "PHASE_N1_STOP_DATASET_BUILD_FAILED.flag"
  "PHASE_N1_STOP_LABEL_VALIDITY_FAILED.flag"
  "PHASE_N1_STOP_ID_ALIGNMENT_FAILED.flag"
  "PHASE_N1_STOP_DATA_LEAKAGE_FAILED.flag"
  "PHASE_N1_STOP_INSUFFICIENT_DATA.flag"
  "PHASE_N1_STOP_SCHEMA_MISMATCH.flag"
  "PHASE_N1_STOP_WATCHER_FAILED.flag"
)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$DATA_DIR"
: > "$LOG"

echo "[watcher] Phase N1 watcher started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
echo "running" > "$STATUS"

rm -f "$COMPLETE_FLAG"
for flag in "${STOP_FLAGS[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

if [[ ! -f "results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N1_STOP_PHASE_N0_MISSING.flag"
fi

(
  python scripts/build_gpsi_heada_dataset.py \
    --out-dir "$DATA_DIR" \
    --result-dir "$OUT_DIR" \
    --scenario train_flow_mixed \
    --num-episodes 100 \
    --eval-seed 2000 \
    --history-steps 20 \
    --future-times 1.0 2.0 4.0 \
    --split 0.70 0.15 0.15 \
    --format npz \
    --write-schema

  python scripts/inspect_gpsi_heada_dataset.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
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
    touch "$OUT_DIR/PHASE_N1_STOP_WATCHER_FAILED.flag"
    echo "stopped:PHASE_N1_STOP_WATCHER_FAILED.flag" > "$STATUS"
    echo "[watcher] runner exited without complete or stop flag" | tee -a "$LOG"
    exit 2
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
