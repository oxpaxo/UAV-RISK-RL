#!/usr/bin/env bash
set -eo pipefail

OUT_DIR="results/env_v2_phase_n4u_precheck_oracle_calibration_v2"
LOG_DIR="$OUT_DIR/logs"
WATCHER_LOG="$OUT_DIR/phase_n4u_precheck_oracle_calibration_v2_watcher.log"
STATUS_FILE="$OUT_DIR/phase_n4u_precheck_oracle_calibration_v2_status.txt"
GITHUB_SYNC_COMMIT="${GITHUB_SYNC_COMMIT:-}"
GITHUB_SYNC_STATUS="${GITHUB_SYNC_STATUS:-unknown}"

mkdir -p "$OUT_DIR/tables" "$OUT_DIR/plots" "$LOG_DIR"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$WATCHER_LOG"
}

stop_phase() {
  local flag="$1"
  local detail="$2"
  printf '%s\n' "$detail" > "$OUT_DIR/$flag"
  printf 'stopped:%s\n' "$flag" > "$STATUS_FILE"
  log "STOP $flag $detail"
  exit 0
}

log "PHASE_N4U_PRECHECK_V2_WATCHER_START heartbeat_interval=10min"
printf 'running\n' > "$STATUS_FILE"

log "PREFLIGHT_START"
[[ -f "codex_guide/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_GUIDE.md" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing guide"
[[ -f "envs/dynamic_obstacle_flow_env.py" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing EnvV2 core file"
[[ -f "scripts/run_env_v2_phase_b_geometry_filter_baselines.py" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing Phase B implementation"
[[ -f "scripts/eval_env_v2_phase_n4o_ordinary_shield.py" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing N4-O implementation"
[[ -f "results/env_v2_phase_b_geometry_filter_baselines/tables/phase_b_baseline_manifest.csv" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing Phase B manifest"
log "PREFLIGHT_OK"

log "STEP_A_CPA_TTC_AUDIT_START"
python -u scripts/audit_phase_n4u_precheck_oracle_calibration_v2.py \
  --result-dir "$OUT_DIR" \
  --github-sync-commit "$GITHUB_SYNC_COMMIT" \
  --github-sync-status "$GITHUB_SYNC_STATUS" \
  > "$LOG_DIR/phase_n4u_precheck_v2_step_a_audit.log" 2>&1 || {
    cat "$LOG_DIR/phase_n4u_precheck_v2_step_a_audit.log" >> "$WATCHER_LOG" || true
    stop_phase "STOP_CPA_TTC_AUDIT_FAILED.flag" "Step A audit command failed"
  }
cat "$LOG_DIR/phase_n4u_precheck_v2_step_a_audit.log" >> "$WATCHER_LOG" || true

if [[ -f "$OUT_DIR/STOP_PREMISE_INVALIDATED.flag" ]]; then
  log "STOP STOP_PREMISE_INVALIDATED.flag terminal_decision=phase_n4u_precheck_v2_stopped_premise_invalidated"
  exit 0
fi

if [[ -f "$OUT_DIR/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_COMPLETE.flag" ]]; then
  log "COMPLETE $(cat "$OUT_DIR/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_COMPLETE.flag")"
  exit 0
fi

stop_phase "STOP_ANALYSIS_FAILED.flag" "watcher reached unexpected state without complete or stop flag"
