#!/usr/bin/env bash
set -u -o pipefail

OUT_DIR="${PHASE_B_OUT_DIR:-results/env_v2_phase_b_geometry_filter_baselines}"
LOG="$OUT_DIR/phase_b_watcher.log"
STATUS="$OUT_DIR/phase_b_status.txt"
CHECKPOINT="${PHASE_B_CHECKPOINT:-checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip}"
INTERVAL_SECONDS="${PHASE_B_WATCH_INTERVAL_SECONDS:-10}"
B0_EPISODES="${PHASE_B_B0_EPISODES:-3}"
B1_EPISODES="${PHASE_B_B1_EPISODES:-20}"
B2_EPISODES="${PHASE_B_B2_EPISODES:-50}"

STOP_FLAGS=(
  PHASE_B_STOP_PHASE_A_MISSING.flag
  PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag
  PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag
  PHASE_B_STOP_BASELINE_IMPL_FAILED.flag
  PHASE_B_STOP_EVAL_FAILED.flag
  PHASE_B_STOP_SCHEMA_MISMATCH.flag
  PHASE_B_STOP_RESOURCE_LIMIT.flag
  PHASE_B_STOP_WATCHER_FAILED.flag
)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces/sample_traces" "$OUT_DIR/traces/failure_traces" "$OUT_DIR/traces/formal_traces"
: > "$LOG" || {
  echo "[watcher] failed to write watcher log: $LOG"
  exit 2
}

log() {
  local stamp
  stamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$stamp] $1" | tee -a "$LOG"
}

write_status() {
  echo "$1" > "$STATUS" || {
    log "[watcher] failed to write status file: $STATUS"
    touch "$OUT_DIR/PHASE_B_STOP_WATCHER_FAILED.flag"
    exit 2
  }
}

rm -f "$OUT_DIR/PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag"
for flag in "${STOP_FLAGS[@]}"; do
  rm -f "$OUT_DIR/$flag"
done

log "[watcher] Phase B watcher started"
write_status "running:watcher_started"

if [[ ! -f "results/env_v2_phase_a_eval_framework/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_B_STOP_PHASE_A_MISSING.flag"
  write_status "stopped:PHASE_B_STOP_PHASE_A_MISSING.flag"
  log "[watcher] Phase A complete flag missing"
  exit 2
fi

(
  set -o pipefail
  python scripts/run_env_v2_phase_b_geometry_filter_baselines.py \
    --out-dir "$OUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --eval-seed 1000 \
    --stage full \
    --b0-episodes "$B0_EPISODES" \
    --b1-episodes "$B1_EPISODES" \
    --b2-episodes "$B2_EPISODES" \
    --write-traces \
    --heartbeat-seconds 20 \
  && python scripts/analyze_env_v2_phase_b_results.py \
    --result-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag" ]]; then
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
    if [[ -f "$OUT_DIR/PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag" ]]; then
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
    touch "$OUT_DIR/PHASE_B_STOP_EVAL_FAILED.flag"
    write_status "stopped:process_exited_without_complete_flag"
    log "[watcher] process exited without complete flag runner_exit_code=$runner_code"
    exit 2
  fi

  line_count="$(find "$OUT_DIR/tables" -maxdepth 1 -type f -name '*.csv' -print 2>/dev/null | wc -l | tr -d ' ')"
  trace_count="$(find "$OUT_DIR/traces" -type f -name '*.csv' -print 2>/dev/null | wc -l | tr -d ' ')"
  write_status "running:pid=$PID tables=$line_count traces=$trace_count"
  log "[watcher] still running pid=$PID tables=$line_count traces=$trace_count"
  sleep "$INTERVAL_SECONDS"
done
