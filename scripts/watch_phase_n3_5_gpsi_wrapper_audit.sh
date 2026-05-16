#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3_5_gpsi_wrapper_audit"
LOG="$OUT_DIR/phase_n3_5_watcher.log"
STATUS="$OUT_DIR/phase_n3_5_status.txt"
CHECKPOINT="work_dirs/gpsi_heada_v1_nll/best.pth"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"
: > "$LOG"
echo "[watcher] Phase N3.5 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

stop_flags=(
  PHASE_N3_5_STOP_PHASE_N2_MISSING.flag
  PHASE_N3_5_STOP_GPSI_CHECKPOINT_MISSING.flag
  PHASE_N3_5_STOP_N3_ARTIFACTS_MISSING.flag
  PHASE_N3_5_STOP_OFFLINE_ONLINE_MISMATCH.flag
  PHASE_N3_5_STOP_INPUT_DISTRIBUTION_INVALID.flag
  PHASE_N3_5_STOP_OUTPUT_SCALE_INVALID.flag
  PHASE_N3_5_STOP_FEATURE_SCALE_INVALID.flag
  PHASE_N3_5_STOP_WRAPPER_REPAIR_FAILED.flag
  PHASE_N3_5_STOP_WATCHER_FAILED.flag
)

rm -f "$OUT_DIR/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag"
for flag in "${stop_flags[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

if [[ ! -f "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3_5_STOP_PHASE_N2_MISSING.flag"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  touch "$OUT_DIR/PHASE_N3_5_STOP_GPSI_CHECKPOINT_MISSING.flag"
fi

if [[ ! -f "results/env_v2_phase_n3_gpsi_ppo_no_shield/PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3_5_STOP_N3_ARTIFACTS_MISSING.flag"
fi

for flag in "${stop_flags[@]}"; do
  if [[ -f "$OUT_DIR/$flag" ]]; then
    echo "stopped:$flag" > "$STATUS"
    echo "[watcher] stop flag detected before launch: $flag" | tee -a "$LOG"
    exit 2
  fi
done

(
  set -euo pipefail
  echo "[watcher] running offline-online equivalence"
  python scripts/compare_gpsi_offline_online.py \
    --data-dir data/gpsi_head_a_v1 \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --split val \
    --num-samples 5000

  echo "[watcher] running online wrapper audit"
  python scripts/audit_gpsi_online_wrapper.py \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
    --num-episodes 10 \
    --policy random_or_straight_line \
    --write-input-output-stats

  echo "[watcher] running augmented feature-scale audit"
  python scripts/inspect_gpsi_augmented_features.py \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
    --num-episodes 10 \
    --policy random_or_straight_line \
    --write-plots

  echo "[watcher] finalizing report and complete flag"
  python scripts/finalize_phase_n3_5_gpsi_wrapper_audit.py \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    wait "$PID" || true
    exit 0
  fi

  for flag in "${stop_flags[@]}"; do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      wait "$PID" || true
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ -f "$OUT_DIR/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag" ]]; then
      echo "complete" > "$STATUS"
      echo "[watcher] complete flag detected after process exit" | tee -a "$LOG"
      exit 0
    fi
    for flag in "${stop_flags[@]}"; do
      if [[ -f "$OUT_DIR/$flag" ]]; then
        echo "stopped:$flag" > "$STATUS"
        echo "[watcher] stop flag detected after process exit: $flag" | tee -a "$LOG"
        exit 2
      fi
    done
    touch "$OUT_DIR/PHASE_N3_5_STOP_WATCHER_FAILED.flag"
    echo "stopped:process_exited_without_complete_flag" > "$STATUS"
    echo "[watcher] process exited without complete or stop flag" | tee -a "$LOG"
    exit 2
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
