#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n2_gpsi_heada_offline"
LOG="$OUT_DIR/phase_n2_watcher.log"
STATUS="$OUT_DIR/phase_n2_status.txt"
COMPLETE_FLAG="$OUT_DIR/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag"

STOP_FLAGS=(
  "PHASE_N2_STOP_PHASE_N1_MISSING.flag"
  "PHASE_N2_STOP_DATASET_READ_FAILED.flag"
  "PHASE_N2_STOP_SCHEMA_MISMATCH.flag"
  "PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag"
  "PHASE_N2_STOP_DELTA_NOT_LEARNABLE.flag"
  "PHASE_N2_STOP_NLL_TRAIN_FAILED.flag"
  "PHASE_N2_STOP_LOGVAR_COLLAPSE.flag"
  "PHASE_N2_STOP_CALIBRATION_FAILED.flag"
  "PHASE_N2_STOP_WATCHER_FAILED.flag"
)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" work_dirs/gpsi_heada_v1_delta_only work_dirs/gpsi_heada_v1_nll
: > "$LOG"

echo "[watcher] Phase N2 watcher started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
echo "running" > "$STATUS"

rm -f "$COMPLETE_FLAG"
for flag in "${STOP_FLAGS[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

if [[ ! -f "results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N2_STOP_PHASE_N1_MISSING.flag"
fi

(
  python scripts/train_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --out-dir work_dirs/gpsi_heada_v1_delta_only \
    --config configs/gpsi_heada_train_delta_only.yaml \
    --loss delta_smoothl1 \
    --epochs 20 \
    --batch-size 1024 \
    --seed 0

  python scripts/train_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --init work_dirs/gpsi_heada_v1_delta_only/best.pth \
    --out-dir work_dirs/gpsi_heada_v1_nll \
    --config configs/gpsi_heada_train_nll.yaml \
    --loss gaussian_nll \
    --logvar-clamp -5 3 \
    --epochs 20 \
    --batch-size 1024 \
    --seed 0

  python scripts/eval_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --delta-checkpoint work_dirs/gpsi_heada_v1_delta_only/best.pth \
    --nll-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
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
    touch "$OUT_DIR/PHASE_N2_STOP_WATCHER_FAILED.flag"
    echo "stopped:PHASE_N2_STOP_WATCHER_FAILED.flag" > "$STATUS"
    echo "[watcher] runner exited without complete or stop flag" | tee -a "$LOG"
    exit 2
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
