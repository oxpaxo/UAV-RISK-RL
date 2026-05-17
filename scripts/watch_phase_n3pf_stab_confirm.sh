#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="results/env_v2_phase_n3pf_stab_confirm"
TABLE_PREFIX="phase_n3pf_stab_confirm"
LOG_DIR="$RESULT_DIR/logs"
STATUS_FILE="$RESULT_DIR/phase_n3pf_stab_confirm_status.txt"
WATCHER_LOG="$RESULT_DIR/phase_n3pf_stab_confirm_watcher.log"
COMPLETE_FLAG="$RESULT_DIR/PHASE_N3PF_STAB_CONFIRM_COMPLETE.flag"
REPORT_FILE="$RESULT_DIR/PHASE_N3PF_STAB_CONFIRM_REPORT.md"
CONFIG="configs/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_gated.yaml"
CHECKPOINT_STEPS=(500000 750000 1000000 1250000 1500000)
TRAIN_STEPS=1500000
HEARTBEAT_SECONDS=300

mkdir -p "$RESULT_DIR/tables" "$RESULT_DIR/plots" "$LOG_DIR"
: > "$WATCHER_LOG"
echo "running" > "$STATUS_FILE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$WATCHER_LOG"
}

stop_phase() {
    local flag="$1"
    local detail="$2"
    echo "$detail" > "$RESULT_DIR/$flag"
    echo "stopped:$flag" > "$STATUS_FILE"
    {
        echo "# Phase N3PF-STAB-CONFIRM Report"
        echo
        echo "\`terminal_decision = phase_n3pf_stab_confirm_stopped_${flag%.flag}\`"
        echo
        echo '```text'
        echo "$detail"
        echo '```'
    } > "$REPORT_FILE"
    log "STOP $flag $detail"
    exit 2
}

run_cmd() {
    local label="$1"
    shift
    local log_file="$LOG_DIR/${label}.log"
    log "RUN $label: $*"
    if ! "$@" > "$log_file" 2>&1; then
        tail -80 "$log_file" | tee -a "$WATCHER_LOG" || true
        return 1
    fi
}

preflight() {
    log "PREFLIGHT_START"
    for path in \
        "$CONFIG" \
        "models/gpsi_ppo_policy.py" \
        "envs/wrappers/gpsi_obs_wrapper.py" \
        "work_dirs/gpsi_heada_v1_nll/best.pth" \
        "results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_COMPLETE.flag"; do
        [[ -e "$path" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing required path: $path"
    done
    grep -q "GpsiGatedResidualExtractor" models/gpsi_ppo_policy.py || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing GpsiGatedResidualExtractor"
    {
        echo "nproc=$(nproc)"
        echo "nproc_all=$(nproc --all)"
        taskset -pc $$
        free -h
        df -h /
    } > "$LOG_DIR/phase_n3pf_stab_confirm_resource_preflight.log" 2>&1 || true
    log "PREFLIGHT_OK"
}

validate_training_entry() {
    local seed="$1"
    local out_dir="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}"
    run_cmd "phase_n3pf_stab_confirm_validate_seed${seed}" \
        python -u scripts/train_env_v2_gpsi_ppo_n3pf_stab.py \
        --config "$CONFIG" \
        --out-dir "$out_dir" \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --status-file "phase_n3pf_stab_confirm_status.txt" \
        --report-file "PHASE_N3PF_STAB_CONFIRM_REPORT.md" \
        --terminal-prefix "phase_n3pf_stab_confirm" \
        --stop-flag-mode confirm \
        --train-steps "$TRAIN_STEPS" \
        --checkpoint-steps "${CHECKPOINT_STEPS[@]}" \
        --seed "$seed" \
        --n-envs 4 \
        --device cpu \
        --validate-only || stop_phase "STOP_PREFLIGHT_FAILED.flag" "validate-only failed for seed ${seed}"
}

start_training_job() {
    local seed="$1"
    local out_dir="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}"
    local log_file="$LOG_DIR/phase_n3pf_stab_confirm_train_s${seed}.log"
    log "TRAIN_START seed=${seed} out_dir=${out_dir}" >&2
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
        python -u scripts/train_env_v2_gpsi_ppo_n3pf_stab.py \
        --config "$CONFIG" \
        --out-dir "$out_dir" \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --status-file "phase_n3pf_stab_confirm_status.txt" \
        --report-file "PHASE_N3PF_STAB_CONFIRM_REPORT.md" \
        --terminal-prefix "phase_n3pf_stab_confirm" \
        --stop-flag-mode confirm \
        --train-steps "$TRAIN_STEPS" \
        --checkpoint-steps "${CHECKPOINT_STEPS[@]}" \
        --seed "$seed" \
        --n-envs 4 \
        --device cpu \
        --heartbeat-seconds 300 \
        > "$log_file" 2>&1 &
    PIDS[$seed]="$!"
}

pid_alive() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

checkpoint_status() {
    local seed="$1"
    local out_dir="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}"
    local latest="none"
    for label in 1500 1250 1000 750 500; do
        if [[ -s "$out_dir/checkpoint_${label}k.zip" ]]; then
            latest="checkpoint_${label}k.zip"
            break
        fi
    done
    if [[ -s "$out_dir/final.zip" ]]; then
        latest="final.zip"
    fi
    echo "$latest"
}

target_ready() {
    local seed="$1"
    local out_dir="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}"
    [[ -s "$out_dir/final.zip" && -s "$out_dir/checkpoint_1500k.zip" ]]
}

lock_pid_for_seed() {
    local seed="$1"
    local lock="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}/TRAIN_RUNNING.lock"
    [[ -s "$lock" ]] || return 1
    python - "$lock" <<'PY'
import json, sys
try:
    print(int(json.load(open(sys.argv[1]))["pid"]))
except Exception:
    raise SystemExit(1)
PY
}

attach_or_start_seed() {
    local seed="$1"
    if target_ready "$seed"; then
        log "TRAIN_ALREADY_COMPLETE seed=${seed}"
        return 0
    fi
    local pid=""
    if pid="$(lock_pid_for_seed "$seed" 2>/dev/null)" && pid_alive "$pid"; then
        PIDS[$seed]="$pid"
        log "TRAIN_ATTACH seed=${seed} pid=${pid}"
        return 0
    fi
    start_training_job "$seed"
    log "PID seed=${seed} pid=${PIDS[$seed]}"
}

resource_healthy_for_optional() {
    local available_kb
    local disk_pct
    local load1
    available_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    disk_pct="$(df -P / | awk 'NR==2 {gsub("%","",$5); print $5}')"
    load1="$(awk '{print $1}' /proc/loadavg)"
    python - "$available_kb" "$disk_pct" "$load1" <<'PY'
import sys
avail_kb = float(sys.argv[1])
disk_pct = float(sys.argv[2])
load1 = float(sys.argv[3])
ok = avail_kb > 8 * 1024 * 1024 and disk_pct < 95 and load1 < 14
raise SystemExit(0 if ok else 1)
PY
}

monitor_training() {
    declare -A PIDS
    for seed in 0 1 2; do
        validate_training_entry "$seed"
    done
    for seed in 0 1 2; do
        attach_or_start_seed "$seed"
        sleep 2
    done

    local optional_seed3_started=0
    if target_ready 3; then
        optional_seed3_started=1
    fi
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        local running=0
        local failed=0
        for seed in 0 1 2 3; do
            if target_ready "$seed"; then
                unset "PIDS[$seed]" 2>/dev/null || true
                continue
            fi
            if [[ -z "${PIDS[$seed]:-}" ]]; then
                local attach_pid=""
                if attach_pid="$(lock_pid_for_seed "$seed" 2>/dev/null)" && pid_alive "$attach_pid"; then
                    PIDS[$seed]="$attach_pid"
                    log "TRAIN_REATTACH seed=${seed} pid=${attach_pid}"
                fi
            fi
        done
        for seed in "${!PIDS[@]}"; do
            if pid_alive "${PIDS[$seed]}"; then
                running=$((running + 1))
            else
                if ! target_ready "$seed"; then
                    failed=1
                fi
                unset "PIDS[$seed]"
            fi
        done
        if (( failed )); then
            stop_phase "STOP_TRAINING_FAILED.flag" "one or more training jobs failed"
        fi
        if (( optional_seed3_started == 0 )); then
            local now
            now="$(date +%s)"
            if (( now - start_ts >= 300 )); then
                if resource_healthy_for_optional; then
                    validate_training_entry 3
                    attach_or_start_seed 3
                    log "OPTIONAL_SEED3_STARTED pid=${PIDS[3]}"
                else
                    log "OPTIONAL_SEED3_SKIPPED resource health gate not satisfied"
                fi
                optional_seed3_started=1
            fi
        fi
        log "TRAIN_HEARTBEAT running=${running} load=$(cat /proc/loadavg) mem_available=$(awk '/MemAvailable/ {print $2 " kB"}' /proc/meminfo) disk=$(df -h / | awk 'NR==2 {print $4 " free " $5 " used"}')"
        for seed in 0 1 2 3; do
            if [[ -d "checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}" ]]; then
                log "TRAIN_STATUS seed=${seed} latest=$(checkpoint_status "$seed") pid=${PIDS[$seed]:-done_or_not_started}"
                tail -5 "$LOG_DIR/phase_n3pf_stab_confirm_train_s${seed}.log" >> "$WATCHER_LOG" 2>/dev/null || true
            fi
        done
        if target_ready 0 && target_ready 1 && target_ready 2 && { (( optional_seed3_started == 0 )) || target_ready 3; }; then
            break
        fi
        sleep "$HEARTBEAT_SECONDS"
    done
    log "TRAINING_ALL_DONE"
}

seed_runs_for_validation() {
    local runs=()
    for seed in 0 1 2 3; do
        local out_dir="checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s${seed}"
        if [[ -s "$out_dir/final.zip" && -s "$out_dir/checkpoint_1500k.zip" ]]; then
            runs+=(--run "s2d_confirm:${seed}:${CONFIG}:${out_dir}")
        fi
    done
    printf '%s\n' "${runs[@]}"
}

run_validation_selector_test_analysis() {
    mapfile -t RUN_ARGS < <(seed_runs_for_validation)
    if (( ${#RUN_ARGS[@]} < 6 )); then
        stop_phase "STOP_TRAINING_FAILED.flag" "fewer than mandatory seed0/1/2 completed runs were found"
    fi
    if [[ -s "$RESULT_DIR/tables/${TABLE_PREFIX}_validation_eval_summary_by_seed.csv" ]]; then
        log "VALIDATION_SKIP existing summary found"
    else
        run_cmd "phase_n3pf_stab_confirm_eval_validation" \
            python -u scripts/eval_env_v2_gpsi_ppo_n3pf_stab.py \
            --result-dir "$RESULT_DIR" \
            --table-prefix "$TABLE_PREFIX" \
            --status-file "phase_n3pf_stab_confirm_status.txt" \
            --report-file "PHASE_N3PF_STAB_CONFIRM_REPORT.md" \
            --terminal-prefix "phase_n3pf_stab_confirm" \
            --stop-flag-mode confirm \
            --eval-phase validation \
            "${RUN_ARGS[@]}" \
            --checkpoint-labels 500k 750k 1000k 1250k 1500k final \
            --eval-seeds 900 901 \
            --num-episodes 50 \
            --device cpu \
            --skip-raw-step-table \
            --heartbeat-seconds 300 || stop_phase "STOP_EVAL_FAILED.flag" "validation eval failed"
    fi

    if [[ -s "$RESULT_DIR/tables/${TABLE_PREFIX}_selector_decision.csv" ]]; then
        log "SELECTOR_SKIP existing selector decision found"
    else
        run_cmd "phase_n3pf_stab_confirm_selector" \
            python -u scripts/select_env_v2_phase_n3pf_stab_checkpoint.py \
            --result-dir "$RESULT_DIR" \
            --table-prefix "$TABLE_PREFIX" \
            --status-file "phase_n3pf_stab_confirm_status.txt" \
            --report-file "PHASE_N3PF_STAB_CONFIRM_REPORT.md" \
            --terminal-prefix "phase_n3pf_stab_confirm" \
            --stop-flag-mode confirm \
            --validation-seeds 900 901 || stop_phase "STOP_SELECTOR_CONTAMINATED.flag" "selector failed"
    fi

    local selected_file="$RESULT_DIR/tables/${TABLE_PREFIX}_selector_decision.csv"
    [[ -s "$selected_file" ]] || stop_phase "STOP_SELECTOR_CONTAMINATED.flag" "selector decision missing"

    mapfile -t TEST_RUN_ARGS < <(python - "$selected_file" "$CONFIG" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
config = sys.argv[2]
for row in csv.DictReader(path.open()):
    seed = int(float(row["training_seed"]))
    out_dir = str(Path(row["selected_checkpoint_path"]).parent)
    label = str(row["selected_checkpoint_label"])
    print("--run")
    print(f"s2d_confirm:{seed}:{config}:{out_dir}:{label}")
PY
)
    if (( ${#TEST_RUN_ARGS[@]} < 6 )); then
        stop_phase "STOP_SELECTOR_CONTAMINATED.flag" "selector did not produce mandatory seed run args"
    fi
    if [[ -s "$RESULT_DIR/tables/${TABLE_PREFIX}_test_eval_summary_by_seed.csv" ]]; then
        log "TEST_SKIP existing test summary found"
    else
        run_cmd "phase_n3pf_stab_confirm_eval_test_selected" \
            python -u scripts/eval_env_v2_gpsi_ppo_n3pf_stab.py \
            --result-dir "$RESULT_DIR" \
            --table-prefix "$TABLE_PREFIX" \
            --status-file "phase_n3pf_stab_confirm_status.txt" \
            --report-file "PHASE_N3PF_STAB_CONFIRM_REPORT.md" \
            --terminal-prefix "phase_n3pf_stab_confirm" \
            --stop-flag-mode confirm \
            --eval-phase test \
            "${TEST_RUN_ARGS[@]}" \
            --eval-seeds 1000 1001 1002 \
            --num-episodes 50 \
            --device cpu \
            --skip-raw-step-table \
            --heartbeat-seconds 300 || stop_phase "STOP_EVAL_FAILED.flag" "test eval failed"
    fi

    run_cmd "phase_n3pf_stab_confirm_analysis" \
        python -u scripts/analyze_env_v2_phase_n3pf_stab_confirm.py \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --github-sync-commit "${GITHUB_SYNC_COMMIT:-unknown}" \
        --github-sync-status "${GITHUB_SYNC_STATUS:-unknown}" || stop_phase "STOP_EVAL_FAILED.flag" "analysis failed"
}

main() {
    log "PHASE_N3PF_STAB_CONFIRM_WATCHER_START"
    preflight
    monitor_training
    run_validation_selector_test_analysis
    if [[ -s "$COMPLETE_FLAG" ]]; then
        log "COMPLETE $(cat "$COMPLETE_FLAG" | tr '\n' ' ')"
        exit 0
    fi
    stop_phase "STOP_EVAL_FAILED.flag" "analysis finished without complete flag"
}

main "$@"
