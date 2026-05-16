#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUP_DIR="results/env_v2_phase_n3pf_ms_plus_n4o_supervisor"
mkdir -p "$SUP_DIR/logs"
SUP_LOG="$SUP_DIR/supervisor.log"
STATUS_FILE="$SUP_DIR/supervisor_status.txt"

N3_DIR="results/env_v2_phase_n3pf_ms_multiseed"
N4_DIR="results/env_v2_phase_n4o_ordinary_shield_fair_comparison"
N3_COMPLETE="$N3_DIR/PHASE_N3PF_MS_MULTI_SEED_COMPLETE.flag"
N4_COMPLETE="$N4_DIR/PHASE_N4O_ORDINARY_SHIELD_FAIR_COMPARISON_COMPLETE.flag"

N3_STOP_PATTERN="$N3_DIR/PHASE_N3PF_MS_STOP_*.flag"
N4_STOP_PATTERN="$N4_DIR/PHASE_N4O_STOP_*.flag"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$SUP_LOG"
}

latest_steps() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    echo "unavailable"
    return
  fi
  grep -E "N3FZ_TRAIN_HEARTBEAT|N3PF_MS_TRAIN_END|N3FZ_CHECKPOINT_SAVED" "$log_file" | tail -n 3 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g'
}

: > "$SUP_LOG"
echo "running" > "$STATUS_FILE"

log "starting N3PF-MS watcher and N4-O watcher"
bash scripts/watch_phase_n3pf_ms_multiseed.sh > "$SUP_DIR/logs/n3pf_ms_watcher_stdout.log" 2>&1 &
N3_PID=$!
bash scripts/watch_phase_n4o_ordinary_shield.sh > "$SUP_DIR/logs/n4o_watcher_stdout.log" 2>&1 &
N4_PID=$!
log "watcher PIDs n3pf_ms=$N3_PID n4o=$N4_PID"

LAST_HEARTBEAT=0
while true; do
  if compgen -G "$N3_STOP_PATTERN" > /dev/null; then
    echo "stopped:n3pf_ms" > "$STATUS_FILE"
    log "N3PF-MS stop flag detected: $(ls $N3_STOP_PATTERN)"
    exit 2
  fi
  if compgen -G "$N4_STOP_PATTERN" > /dev/null; then
    echo "stopped:n4o" > "$STATUS_FILE"
    log "N4-O stop flag detected: $(ls $N4_STOP_PATTERN)"
    exit 2
  fi
  if [[ -f "$N3_COMPLETE" && -f "$N4_COMPLETE" ]]; then
    echo "complete" > "$STATUS_FILE"
    log "both complete flags detected"
    wait "$N3_PID" || true
    wait "$N4_PID" || true
    exit 0
  fi
  if ! kill -0 "$N3_PID" 2>/dev/null && [[ ! -f "$N3_COMPLETE" ]]; then
    echo "stopped:n3pf_ms_watcher_exit" > "$STATUS_FILE"
    log "N3PF-MS watcher exited without complete flag"
    exit 2
  fi
  if ! kill -0 "$N4_PID" 2>/dev/null && [[ ! -f "$N4_COMPLETE" ]]; then
    echo "stopped:n4o_watcher_exit" > "$STATUS_FILE"
    log "N4-O watcher exited without complete flag"
    exit 2
  fi
  NOW=$(date +%s)
  if (( NOW - LAST_HEARTBEAT >= 300 )); then
    LAST_HEARTBEAT=$NOW
    N3_STATUS="$(cat "$N3_DIR/phase_n3pf_ms_status.txt" 2>/dev/null || echo pending)"
    N4_STATUS="$(cat "$N4_DIR/phase_n4o_status.txt" 2>/dev/null || echo pending)"
    S1="$(latest_steps "$N3_DIR/logs/phase_n3pf_ms_train_s1.log")"
    S2="$(latest_steps "$N3_DIR/logs/phase_n3pf_ms_train_s2.log")"
    LOAD="$(cut -d' ' -f1-3 /proc/loadavg)"
    MEM="$(free -h | awk '/Mem:/ {print "avail="$7" used="$3}')"
    DISK="$(df -h / | awk 'NR==2 {print "avail="$4" used="$3}')"
    log "heartbeat n3_status=$N3_STATUS n4_status=$N4_STATUS load=$LOAD mem=($MEM) disk=($DISK)"
    log "seed1_recent=$S1"
    log "seed2_recent=$S2"
  fi
  sleep 120
done
