#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="results/env_v2_phase_n3pf_decouple"
TABLE_PREFIX="phase_n3pf_decouple"
LOG_DIR="$RESULT_DIR/logs"
STATUS_FILE="$RESULT_DIR/phase_n3pf_decouple_status.txt"
WATCHER_LOG="$RESULT_DIR/phase_n3pf_decouple_watcher.log"
COMPLETE_FLAG="$RESULT_DIR/PHASE_N3PF_DECOUPLE_COMPLETE.flag"
REPORT_FILE="$RESULT_DIR/PHASE_N3PF_DECOUPLE_REPORT.md"
HEARTBEAT_SECONDS=300
TRAIN_MAX_INITIAL=6
TRAIN_MAX_HEALTHY=8
EVAL_MAX_INITIAL=8

mkdir -p "$RESULT_DIR/tables" "$RESULT_DIR/plots" "$LOG_DIR"
: > "$WATCHER_LOG"
echo "running" > "$STATUS_FILE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

VARIANTS=(decouple_nk_obs decouple_nk_gpsi decouple_deepsets_obs decouple_deepsets_gpsi)
CONFIG_decouple_nk_obs="configs/env_v2_gpsi_heada_ppo_n3pf_decouple_nk_obs.yaml"
CONFIG_decouple_nk_gpsi="configs/env_v2_gpsi_heada_ppo_n3pf_decouple_nk_gpsi.yaml"
CONFIG_decouple_deepsets_obs="configs/env_v2_gpsi_heada_ppo_n3pf_decouple_deepsets_obs.yaml"
CONFIG_decouple_deepsets_gpsi="configs/env_v2_gpsi_heada_ppo_n3pf_decouple_deepsets_gpsi.yaml"
SCENARIOS=(eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat)

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
        echo "# Phase N3PF-DECOUPLE Report"
        echo
        echo "\`terminal_decision = phase_n3pf_decouple_stopped_${flag%.flag}\`"
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

config_for_variant() {
    local variant="$1"
    local var="CONFIG_${variant}"
    echo "${!var}"
}

out_dir_for() {
    local variant="$1"
    local seed="$2"
    echo "checkpoints/env_v2_gpsi_heada_ppo_n3pf_decouple_${variant}_s${seed}"
}

pid_alive() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

latest_checkpoint() {
    local out_dir="$1"
    local latest="none"
    for label in final 1500 1250 1000 750 500 250; do
        if [[ "$label" == "final" && -s "$out_dir/final.zip" ]]; then
            latest="final.zip"
            break
        fi
        if [[ "$label" != "final" && -s "$out_dir/checkpoint_${label}k.zip" ]]; then
            latest="checkpoint_${label}k.zip"
            break
        fi
    done
    echo "$latest"
}

target_ready() {
    local out_dir="$1"
    local target="$2"
    [[ -s "$out_dir/final.zip" && -s "$out_dir/checkpoint_$((target / 1000))k.zip" ]]
}

resource_healthy() {
    local available_kb disk_pct load1
    available_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    disk_pct="$(df -P / | awk 'NR==2 {gsub("%","",$5); print $5}')"
    load1="$(awk '{print $1}' /proc/loadavg)"
    python - "$available_kb" "$disk_pct" "$load1" <<'PY'
import sys
avail_kb = float(sys.argv[1])
disk_pct = float(sys.argv[2])
load1 = float(sys.argv[3])
ok = avail_kb > 8 * 1024 * 1024 and disk_pct < 95 and load1 < 15
raise SystemExit(0 if ok else 1)
PY
}

preflight() {
    log "PREFLIGHT_START"
    for path in \
        "codex_guide/PHASE_N3PF_DECOUPLE_GUIDE.md" \
        "models/gpsi_ppo_policy.py" \
        "envs/wrappers/gpsi_obs_wrapper.py" \
        "work_dirs/gpsi_heada_v1_nll/best.pth"; do
        [[ -e "$path" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing required path: $path"
    done
    grep -q "GpsiNearestKExtractor" models/gpsi_ppo_policy.py || stop_phase "STOP_SCHEMA_MISMATCH.flag" "missing GpsiNearestKExtractor"
    grep -q "GpsiDeepSetsExtractor" models/gpsi_ppo_policy.py || stop_phase "STOP_SCHEMA_MISMATCH.flag" "missing GpsiDeepSetsExtractor"
    for variant in "${VARIANTS[@]}"; do
        local cfg
        cfg="$(config_for_variant "$variant")"
        [[ -s "$cfg" ]] || stop_phase "STOP_PREFLIGHT_FAILED.flag" "missing config: $cfg"
    done
    {
        echo "nproc=$(nproc)"
        echo "nproc_all=$(nproc --all)"
        taskset -pc $$
        free -h
        df -h /
        python - <<'PY'
import os
print("os_cpu_count", os.cpu_count())
print("affinity_count", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "na")
PY
    } > "$LOG_DIR/phase_n3pf_decouple_resource_preflight.log" 2>&1 || true
    log "PREFLIGHT_OK"
}

validate_training_entries() {
    for variant in "${VARIANTS[@]}"; do
        local cfg
        cfg="$(config_for_variant "$variant")"
        for seed in 0 1 2; do
            local out_dir
            out_dir="$(out_dir_for "$variant" "$seed")"
            run_cmd "phase_n3pf_decouple_validate_${variant}_s${seed}" \
                python -u scripts/train_env_v2_gpsi_ppo_n3pf_stab.py \
                --config "$cfg" \
                --out-dir "$out_dir" \
                --result-dir "$RESULT_DIR" \
                --table-prefix "$TABLE_PREFIX" \
                --status-file "phase_n3pf_decouple_status.txt" \
                --report-file "PHASE_N3PF_DECOUPLE_REPORT.md" \
                --terminal-prefix "phase_n3pf_decouple" \
                --stop-flag-mode confirm \
                --train-steps 750000 \
                --checkpoint-steps 250000 500000 750000 \
                --seed "$seed" \
                --n-envs 4 \
                --device cpu \
                --validate-only || stop_phase "STOP_PREFLIGHT_FAILED.flag" "validate-only failed: ${variant} seed ${seed}"
        done
    done
}

start_train_job() {
    local variant="$1"
    local seed="$2"
    local target="$3"
    local extra_args=("${@:4}")
    local cfg out_dir log_file
    cfg="$(config_for_variant "$variant")"
    out_dir="$(out_dir_for "$variant" "$seed")"
    log_file="$LOG_DIR/phase_n3pf_decouple_train_${variant}_s${seed}_${target}.log"
    log "TRAIN_START variant=${variant} seed=${seed} target=${target} out_dir=${out_dir}" >&2
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
        python -u scripts/train_env_v2_gpsi_ppo_n3pf_stab.py \
        --config "$cfg" \
        --out-dir "$out_dir" \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --status-file "phase_n3pf_decouple_status.txt" \
        --report-file "PHASE_N3PF_DECOUPLE_REPORT.md" \
        --terminal-prefix "phase_n3pf_decouple" \
        --stop-flag-mode confirm \
        --train-steps "$target" \
        --checkpoint-steps "${extra_args[@]}" \
        --seed "$seed" \
        --n-envs 4 \
        --device cpu \
        --heartbeat-seconds 300 \
        > "$log_file" 2>&1 &
    echo "$!"
}

run_train_queue() {
    local stage="$1"
    local target="$2"
    local max_jobs="$3"
    shift 3
    local queue=("$@")
    declare -A PIDS
    declare -A JOBS
    local idx=0
    while true; do
        while (( idx < ${#queue[@]} && ${#PIDS[@]} < max_jobs )); do
            local item="${queue[$idx]}"
            idx=$((idx + 1))
            local variant="${item%%:*}"
            local seed="${item##*:}"
            local out_dir
            out_dir="$(out_dir_for "$variant" "$seed")"
            if target_ready "$out_dir" "$target"; then
                log "TRAIN_SKIP stage=${stage} variant=${variant} seed=${seed} target=${target}"
                continue
            fi
            local pid
            if [[ "$target" == "750000" ]]; then
                pid="$(start_train_job "$variant" "$seed" "$target" 250000 500000 750000)"
            elif [[ "$target" == "1000000" ]]; then
                pid="$(start_train_job "$variant" "$seed" "$target" 1000000 --resume-checkpoint "$out_dir/checkpoint_750k.zip" --parent-total-steps 750000)"
            else
                pid="$(start_train_job "$variant" "$seed" "$target" 1000000 1250000 1500000 --resume-checkpoint "$out_dir/checkpoint_750k.zip" --parent-total-steps 750000)"
            fi
            PIDS[$pid]="$item"
            JOBS[$item]="$pid"
            sleep 1
        done
        local failed=0
        for pid in "${!PIDS[@]}"; do
            local item="${PIDS[$pid]}"
            local variant="${item%%:*}"
            local seed="${item##*:}"
            local out_dir
            out_dir="$(out_dir_for "$variant" "$seed")"
            if pid_alive "$pid"; then
                :
            else
                if target_ready "$out_dir" "$target"; then
                    unset "PIDS[$pid]"
                else
                    failed=1
                fi
            fi
        done
        if (( failed )); then
            stop_phase "STOP_TRAINING_FAILED.flag" "one or more ${stage} training jobs failed"
        fi
        log "TRAIN_HEARTBEAT stage=${stage} running=${#PIDS[@]} queued=$(( ${#queue[@]} - idx )) load=$(cat /proc/loadavg) mem_available=$(awk '/MemAvailable/ {print $2 " kB"}' /proc/meminfo) disk=$(df -h / | awk 'NR==2 {print $4 " free " $5 " used"}')"
        for item in "${queue[@]}"; do
            local variant="${item%%:*}"
            local seed="${item##*:}"
            local out_dir
            out_dir="$(out_dir_for "$variant" "$seed")"
            [[ -d "$out_dir" ]] && log "TRAIN_STATUS stage=${stage} variant=${variant} seed=${seed} latest=$(latest_checkpoint "$out_dir")"
        done
        if (( idx >= ${#queue[@]} && ${#PIDS[@]} == 0 )); then
            break
        fi
        if [[ "$max_jobs" == "$TRAIN_MAX_INITIAL" ]] && resource_healthy; then
            max_jobs="$TRAIN_MAX_HEALTHY"
            log "TRAIN_CONCURRENCY_RAISE stage=${stage} max_jobs=${max_jobs}"
        fi
        sleep "$HEARTBEAT_SECONDS"
    done
    log "TRAIN_QUEUE_DONE stage=${stage}"
}

stage_a_queue() {
    local jobs=()
    for variant in "${VARIANTS[@]}"; do
        for seed in 0 1 2; do
            jobs+=("${variant}:${seed}")
        done
    done
    if resource_healthy; then
        for variant in "${VARIANTS[@]}"; do
            jobs+=("${variant}:3")
        done
        log "OPTIONAL_SEED3_INCLUDED stage=A" >&2
    else
        log "OPTIONAL_SEED3_SKIPPED stage=A resource health gate not satisfied" >&2
    fi
    printf '%s\n' "${jobs[@]}"
}

run_eval_jobs() {
    local phase="$1"
    local shard_root="$RESULT_DIR/eval_shards/${phase}"
    shift
    local jobs=("$@")
    rm -rf "$shard_root"
    mkdir -p "$shard_root"
    declare -A PIDS
    local idx=0
    while true; do
        while (( idx < ${#jobs[@]} && ${#PIDS[@]} < EVAL_MAX_INITIAL )); do
            local spec="${jobs[$idx]}"
            idx=$((idx + 1))
            IFS=':' read -r variant seed label eval_seed <<< "$spec"
            local cfg out_dir shard log_file
            cfg="$(config_for_variant "$variant")"
            out_dir="$(out_dir_for "$variant" "$seed")"
            shard="$shard_root/${variant}_s${seed}_${label}_e${eval_seed}"
            log_file="$LOG_DIR/phase_n3pf_decouple_eval_${phase}_${variant}_s${seed}_${label}_e${eval_seed}.log"
            mkdir -p "$shard"
            env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
                python -u scripts/eval_env_v2_gpsi_ppo_n3pf_stab.py \
                --result-dir "$shard" \
                --table-prefix "$TABLE_PREFIX" \
                --status-file "phase_n3pf_decouple_status.txt" \
                --report-file "PHASE_N3PF_DECOUPLE_REPORT.md" \
                --terminal-prefix "phase_n3pf_decouple" \
                --stop-flag-mode confirm \
                --eval-phase "$phase" \
                --run "${variant}:${seed}:${cfg}:${out_dir}:${label}" \
                --eval-seeds "$eval_seed" \
                --num-episodes 50 \
                --scenarios "${SCENARIOS[@]}" \
                --device cpu \
                --skip-raw-step-table \
                --heartbeat-seconds 300 \
                > "$log_file" 2>&1 &
            PIDS[$!]="$spec"
            sleep 1
        done
        local failed=0
        for pid in "${!PIDS[@]}"; do
            if pid_alive "$pid"; then
                :
            else
                wait "$pid" || failed=1
                unset "PIDS[$pid]"
            fi
        done
        if (( failed )); then
            stop_phase "STOP_EVAL_FAILED.flag" "one or more ${phase} eval shard jobs failed"
        fi
        log "EVAL_HEARTBEAT phase=${phase} running=${#PIDS[@]} queued=$(( ${#jobs[@]} - idx )) load=$(cat /proc/loadavg) mem_available=$(awk '/MemAvailable/ {print $2 " kB"}' /proc/meminfo)"
        if (( idx >= ${#jobs[@]} && ${#PIDS[@]} == 0 )); then
            break
        fi
        sleep "$HEARTBEAT_SECONDS"
    done
    run_cmd "phase_n3pf_decouple_merge_${phase}" \
        python -u scripts/merge_phase_n3pf_decouple_eval_shards.py \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --eval-phase "$phase" \
        --shard-root "$shard_root" || stop_phase "STOP_EVAL_FAILED.flag" "merge failed for ${phase}"
}

build_validation_jobs() {
    local jobs=()
    for variant in "${VARIANTS[@]}"; do
        for seed in 0 1 2 3; do
            local out_dir
            out_dir="$(out_dir_for "$variant" "$seed")"
            [[ -s "$out_dir/checkpoint_750k.zip" ]] || continue
            for label in 250k 500k 750k; do
                for eval_seed in 900 901; do
                    jobs+=("${variant}:${seed}:${label}:${eval_seed}")
                done
            done
        done
    done
    printf '%s\n' "${jobs[@]}"
}

stage_b_queue_from_plan() {
    python - "$RESULT_DIR/tables/${TABLE_PREFIX}_stage_b_plan.csv" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
for row in csv.DictReader(path.open()):
    if int(float(row["continue_stage_b"])):
        for seed in [0, 1, 2]:
            print(f'{row["variant"]}:{seed}:{int(float(row["stage_b_target_steps"]))}')
PY
}

build_test_jobs_from_selector() {
    python - "$RESULT_DIR/tables/${TABLE_PREFIX}_selector_decision.csv" "$RESULT_DIR/tables/${TABLE_PREFIX}_stage_b_plan.csv" <<'PY'
import csv, sys
from pathlib import Path
selector = list(csv.DictReader(Path(sys.argv[1]).open()))
plan = {row["variant"]: int(float(row["stage_b_target_steps"])) for row in csv.DictReader(Path(sys.argv[2]).open())}
for row in selector:
    seed = int(float(row["training_seed"]))
    if seed not in [0,1,2]:
        continue
    variant = row["variant"]
    label = row["selected_checkpoint_label"]
    target = plan.get(variant, 0)
    if target >= 1500000:
        label = "1500k"
    elif target >= 1000000:
        label = "1000k"
    for eval_seed in [1000, 1001, 1002]:
        print(f"{variant}:{seed}:{label}:{eval_seed}")
PY
}

run_selector() {
    run_cmd "phase_n3pf_decouple_selector_stage_a" \
        python -u scripts/select_env_v2_phase_n3pf_decouple.py \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --validation-seeds 900 901 \
        --stage-a || stop_phase "STOP_SELECTOR_CONTAMINATED.flag" "selector failed"
}

run_analysis() {
    run_cmd "phase_n3pf_decouple_analysis" \
        python -u scripts/analyze_env_v2_phase_n3pf_decouple.py \
        --result-dir "$RESULT_DIR" \
        --table-prefix "$TABLE_PREFIX" \
        --github-sync-commit "${GITHUB_SYNC_COMMIT:-73209069b87b1c88d891b8f9b23440e7734b9240}" \
        --github-sync-status "${GITHUB_SYNC_STATUS:-success}" || stop_phase "STOP_ANALYSIS_FAILED.flag" "analysis failed"
}

main() {
    log "PHASE_N3PF_DECOUPLE_WATCHER_START"
    preflight
    validate_training_entries
    mapfile -t STAGE_A_JOBS < <(stage_a_queue)
    run_train_queue "stage_a_750k" 750000 "$TRAIN_MAX_INITIAL" "${STAGE_A_JOBS[@]}"
    if [[ -s "$RESULT_DIR/tables/${TABLE_PREFIX}_validation_eval_summary_by_seed.csv" ]]; then
        log "VALIDATION_SKIP existing merged validation summary found"
    else
        mapfile -t VALIDATION_JOBS < <(build_validation_jobs)
        (( ${#VALIDATION_JOBS[@]} > 0 )) || stop_phase "STOP_EVAL_FAILED.flag" "no validation jobs built"
        run_eval_jobs validation "${VALIDATION_JOBS[@]}"
    fi
    if [[ -s "$RESULT_DIR/tables/${TABLE_PREFIX}_stage_b_plan.csv" && -s "$RESULT_DIR/tables/${TABLE_PREFIX}_selector_decision.csv" ]]; then
        log "SELECTOR_SKIP existing stage_b_plan and selector decision found"
    else
        run_selector
    fi
    mapfile -t STAGE_B_ITEMS < <(stage_b_queue_from_plan)
    if (( ${#STAGE_B_ITEMS[@]} == 0 )); then
        stop_phase "STOP_NO_VALID_VARIANT.flag" "selector produced no Stage B variants"
    fi
    local stage_b_jobs=()
    local target=0
    for item in "${STAGE_B_ITEMS[@]}"; do
        IFS=':' read -r variant seed item_target <<< "$item"
        stage_b_jobs+=("${variant}:${seed}")
        target="$item_target"
    done
    run_train_queue "stage_b_${target}" "$target" "$TRAIN_MAX_INITIAL" "${stage_b_jobs[@]}"
    mapfile -t TEST_JOBS < <(build_test_jobs_from_selector)
    (( ${#TEST_JOBS[@]} > 0 )) || stop_phase "STOP_EVAL_FAILED.flag" "no test jobs built"
    run_eval_jobs test "${TEST_JOBS[@]}"
    run_analysis
    if [[ -s "$COMPLETE_FLAG" ]]; then
        log "COMPLETE $(cat "$COMPLETE_FLAG" | tr '\n' ' ')"
        exit 0
    fi
    stop_phase "STOP_ANALYSIS_FAILED.flag" "analysis finished without complete flag"
}

main "$@"
