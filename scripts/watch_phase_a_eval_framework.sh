#!/usr/bin/env bash
set -u -o pipefail

OUT_DIR="${PHASE_A_OUT_DIR:-results/env_v2_phase_a_eval_framework}"
LOG="$OUT_DIR/phase_a_watcher.log"
STATUS="$OUT_DIR/phase_a_status.txt"
NUM_EPISODES="${PHASE_A_NUM_EPISODES:-3}"
EVAL_SEED="${PHASE_A_EVAL_SEED:-1000}"
CHECKPOINT="${PHASE_A_CHECKPOINT:-checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip}"
INTERVAL_SECONDS="${PHASE_A_WATCH_INTERVAL_SECONDS:-5}"

STOP_FLAGS=(
  PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag
  PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag
  PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag
  PHASE_A_STOP_SCHEMA_UNIFICATION_FAILED.flag
  PHASE_A_STOP_WATCHER_FAILED.flag
)

mkdir -p "$OUT_DIR"
: > "$LOG" || {
  echo "[watcher] failed to write watcher log: $LOG"
  exit 2
}

log() {
  local message="$1"
  local stamp
  stamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$stamp] $message" | tee -a "$LOG"
}

write_status() {
  echo "$1" > "$STATUS" || {
    log "[watcher] failed to write status file: $STATUS"
    touch "$OUT_DIR/PHASE_A_STOP_WATCHER_FAILED.flag"
    exit 2
  }
}

rm -f "$OUT_DIR/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag"
for flag in "${STOP_FLAGS[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

log "[watcher] Phase A watcher started"
write_status "running:watcher_started"

python scripts/run_env_v2_phase_a_eval_framework.py \
  --out-dir "$OUT_DIR" \
  --num-episodes "$NUM_EPISODES" \
  --scenarios eval_flow_id eval_flow_high_speed \
  --policies random straight_line cpa_reactive attention_full filtered_attention_full \
  --checkpoint "$CHECKPOINT" \
  --eval-seed "$EVAL_SEED" \
  --write-traces \
  --heartbeat-seconds 5 \
  2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
    write_status "complete"
    log "[watcher] complete flag detected"
    exit 0
  fi

  for flag in "${STOP_FLAGS[@]}"; do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      write_status "stopped:$flag"
      log "[watcher] stop flag detected: $flag"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" >/dev/null 2>&1
    runner_code=$?
    if [[ -f "$OUT_DIR/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
      write_status "complete"
      log "[watcher] complete flag detected after runner exit"
      exit 0
    fi
    for flag in "${STOP_FLAGS[@]}"; do
      if [[ -f "$OUT_DIR/$flag" ]]; then
        write_status "stopped:$flag"
        log "[watcher] stop flag detected after runner exit: $flag"
        exit 2
      fi
    done
    touch "$OUT_DIR/PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag"
    write_status "stopped:process_exited_without_complete_flag"
    log "[watcher] process exited without complete flag runner_exit_code=$runner_code"
    exit 2
  fi

  write_status "running:pid=$PID"
  log "[watcher] still running pid=$PID"
  sleep "$INTERVAL_SECONDS"
done
