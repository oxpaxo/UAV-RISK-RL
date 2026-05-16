#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3p_noz_representation_ablation"
LOG="$OUT_DIR/phase_n3p_watcher.log"
STATUS="$OUT_DIR/phase_n3p_status.txt"
SCENARIOS=(eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat)

P1_DIR="checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0"
P2_DIR="checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0"
P3_DIR="checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces" "$P1_DIR" "$P2_DIR" "$P3_DIR"

rm -f "$OUT_DIR"/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag "$OUT_DIR"/PHASE_N3P_STOP_*.flag
rm -f "$OUT_DIR"/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md "$OUT_DIR"/phase_n3p_status.txt "$OUT_DIR"/phase_n3p_watcher.log
rm -f "$OUT_DIR"/tables/phase_n3p_*.csv "$OUT_DIR"/plots/*.png "$OUT_DIR"/logs/*.log
find "$OUT_DIR/traces" -type f -name '*.csv' -delete 2>/dev/null || true
for dir in "$P1_DIR" "$P2_DIR" "$P3_DIR"; do
  rm -f "$dir"/TRAIN_RUNNING.lock "$dir"/TRAIN_COMPLETE.flag "$dir"/TRAIN_STATUS.json
  rm -f "$dir"/checkpoint_250k.zip "$dir"/checkpoint_500k.zip "$dir"/final.zip "$dir"/best_by_eval.zip "$dir"/config_resolved.yaml
done

log() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG"
}

stop() {
  local flag="$1"
  local detail="$2"
  echo "$detail" > "$OUT_DIR/$flag"
  echo "stopped:$flag" > "$STATUS"
  log "STOP $flag: $detail"
  if [[ ! -f "$OUT_DIR/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md" ]]; then
    {
      echo "# Phase N3P No-Z Representation Ablation Report"
      echo
      echo "\`terminal_decision = phase_n3p_stopped_${flag#PHASE_N3P_STOP_}\`"
      echo
      echo '```text'
      echo "$detail"
      echo '```'
      echo
      echo "can_enter_N4: no"
    } > "$OUT_DIR/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md"
  fi
  exit 2
}

run_stage() {
  local name="$1"
  shift
  log "START $name: $*"
  echo "stage:$name" > "$STATUS"
  "$@"
  log "END $name"
}

log "Phase N3P watcher started"
echo "running" > "$STATUS"

[[ -f "work_dirs/gpsi_heada_v1_nll/best.pth" ]] || stop "PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag" "missing Gpsi checkpoint"
[[ -f "results/env_v2_phase_n3z2cf_corrected_z2_full/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag" ]] || stop "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag" "missing N3Z2CF complete flag"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip" ]] || stop "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag" "missing N3F no_z full final checkpoint"
[[ -f "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip" ]] || stop "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag" "missing attention_full reference checkpoint"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

run_stage "compile" python -m py_compile \
  envs/wrappers/gpsi_obs_wrapper.py \
  models/gpsi_ppo_policy.py \
  scripts/train_env_v2_gpsi_ppo_n3p.py \
  scripts/eval_env_v2_gpsi_ppo_n3p.py \
  scripts/analyze_env_v2_phase_n3p_results.py

run_stage "bash_n" bash -n scripts/watch_phase_n3p_noz_representation_ablation.sh

run_stage "train_p1_obs_delta_only" python scripts/train_env_v2_gpsi_ppo_n3p.py \
  --config configs/env_v2_gpsi_heada_ppo_n3p_obs_delta_only.yaml \
  --out-dir "$P1_DIR" \
  --train-steps 500000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3p_train_p1_obs_delta_only.log" 2>&1 || \
  stop "PHASE_N3P_STOP_TRAIN_FAILED.flag" "P1 train failed; see $OUT_DIR/logs/phase_n3p_train_p1_obs_delta_only.log"

run_stage "train_p2_logvar_scaled" python scripts/train_env_v2_gpsi_ppo_n3p.py \
  --config configs/env_v2_gpsi_heada_ppo_n3p_logvar_scaled.yaml \
  --out-dir "$P2_DIR" \
  --train-steps 500000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3p_train_p2_logvar_scaled.log" 2>&1 || \
  stop "PHASE_N3P_STOP_TRAIN_FAILED.flag" "P2 train failed; see $OUT_DIR/logs/phase_n3p_train_p2_logvar_scaled.log"

run_stage "train_p3_block_projected" python scripts/train_env_v2_gpsi_ppo_n3p.py \
  --config configs/env_v2_gpsi_heada_ppo_n3p_block_projected.yaml \
  --out-dir "$P3_DIR" \
  --train-steps 500000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3p_train_p3_block_projected.log" 2>&1 || \
  stop "PHASE_N3P_STOP_TRAIN_FAILED.flag" "P3 train failed; see $OUT_DIR/logs/phase_n3p_train_p3_block_projected.log"

run_stage "eval_p_variants" python scripts/eval_env_v2_gpsi_ppo_n3p.py \
  --result-dir "$OUT_DIR" \
  --configs obs_delta_only logvar_scaled block_projected \
  --eval-seed 1000 \
  --num-episodes 50 \
  --device cpu \
  --heartbeat-seconds 300 \
  --scenarios "${SCENARIOS[@]}" \
  --write-traces \
  > "$OUT_DIR/logs/phase_n3p_eval.log" 2>&1 || \
  stop "PHASE_N3P_STOP_EVAL_FAILED.flag" "N3P eval failed; see $OUT_DIR/logs/phase_n3p_eval.log"

run_stage "analysis_report" python scripts/analyze_env_v2_phase_n3p_results.py \
  --result-dir "$OUT_DIR" \
  --n3r-noz-success 0.4233 \
  --n3r-noz-collision 0.5767 \
  --n3f-noz-success 0.5633 \
  --n3f-noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  --z2-success 0.5067 \
  --z2-collision 0.4933 \
  > "$OUT_DIR/logs/phase_n3p_analysis.log" 2>&1 || \
  stop "PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag" "N3P analysis failed; see $OUT_DIR/logs/phase_n3p_analysis.log"

if [[ -f "$OUT_DIR/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag" ]]; then
  echo "complete" > "$STATUS"
  log "complete flag detected"
  exit 0
fi

stop "PHASE_N3P_STOP_WATCHER_FAILED.flag" "watcher reached end without complete flag"
