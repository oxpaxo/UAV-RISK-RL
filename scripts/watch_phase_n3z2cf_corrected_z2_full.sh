#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3z2cf_corrected_z2_full"
CKPT_DIR="checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0"
PARENT="checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip"
LOG="$OUT_DIR/phase_n3z2cf_watcher.log"
STATUS="$OUT_DIR/phase_n3z2cf_status.txt"
SCENARIOS=(eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces" "$CKPT_DIR"

rm -f "$OUT_DIR"/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag "$OUT_DIR"/PHASE_N3Z2CF_STOP_*.flag
rm -f "$OUT_DIR"/PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md "$OUT_DIR"/phase_n3z2cf_status.txt "$OUT_DIR"/phase_n3z2cf_watcher.log
rm -f "$OUT_DIR"/tables/phase_n3z2cf_*.csv "$OUT_DIR"/plots/*.png "$OUT_DIR"/logs/*.log
find "$OUT_DIR/traces" -type f -name '*.csv' -delete 2>/dev/null || true
rm -f "$CKPT_DIR"/TRAIN_RUNNING.lock "$CKPT_DIR"/TRAIN_COMPLETE.flag "$CKPT_DIR"/TRAIN_STATUS.json
rm -f "$CKPT_DIR"/parent_500k.zip "$CKPT_DIR"/checkpoint_750k.zip "$CKPT_DIR"/checkpoint_1000k.zip "$CKPT_DIR"/checkpoint_1250k.zip "$CKPT_DIR"/checkpoint_1500k.zip "$CKPT_DIR"/final.zip "$CKPT_DIR"/best_by_eval.zip "$CKPT_DIR"/config_resolved.yaml

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
  if [[ ! -f "$OUT_DIR/PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md" ]]; then
    {
      echo "# Phase N3Z2CF Corrected Z2 Full Report"
      echo
      echo "\`terminal_decision = phase_n3z2cf_stopped_${flag#PHASE_N3Z2CF_STOP_}\`"
      echo
      echo '```text'
      echo "$detail"
      echo '```'
      echo
      echo "Can enter N4: no."
    } > "$OUT_DIR/PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md"
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

log "Phase N3Z2CF watcher started"
echo "running" > "$STATUS"

[[ -f "$PARENT" ]] || stop "PHASE_N3Z2CF_STOP_PARENT_MISSING.flag" "missing fixed parent: $PARENT"
[[ -f "results/env_v2_phase_n3z2c_audit/PHASE_N3Z2C_AUDIT_COMPLETE.flag" ]] || stop "PHASE_N3Z2CF_STOP_PARENT_MISSING.flag" "missing N3Z2C-Audit complete flag"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

run_stage "compile" python -m py_compile \
  scripts/train_env_v2_gpsi_ppo_n3z2cf.py \
  scripts/eval_env_v2_gpsi_ppo_n3z2cf.py \
  scripts/analyze_env_v2_phase_n3z2cf_results.py

run_stage "train_corrected_z2_full" python scripts/train_env_v2_gpsi_ppo_n3z2cf.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml \
  --resume "$PARENT" \
  --out-dir "$CKPT_DIR" \
  --parent-total-steps 500000 \
  --additional-steps 1000000 \
  --target-total-steps 1500000 \
  --checkpoint-total-steps 750000 1000000 1250000 1500000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --reset-num-timesteps false \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3z2cf_train_corrected_full.log" 2>&1 || \
  stop "PHASE_N3Z2CF_STOP_TRAIN_FAILED.flag" "corrected Z2 full train failed; see $OUT_DIR/logs/phase_n3z2cf_train_corrected_full.log"

run_stage "eval_corrected_z2_full" python scripts/eval_env_v2_gpsi_ppo_n3z2cf.py \
  --result-dir "$OUT_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --old-z2-checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0 \
  --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --num-episodes 50 \
  --device cpu \
  --heartbeat-seconds 300 \
  --scenarios "${SCENARIOS[@]}" \
  --write-traces \
  > "$OUT_DIR/logs/phase_n3z2cf_eval.log" 2>&1 || \
  stop "PHASE_N3Z2CF_STOP_EVAL_FAILED.flag" "corrected Z2 full eval failed; see $OUT_DIR/logs/phase_n3z2cf_eval.log"

run_stage "analysis_report" python scripts/analyze_env_v2_phase_n3z2cf_results.py \
  --result-dir "$OUT_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  --old-z2c-success 0.4500 \
  --old-z2c-collision 0.5500 \
  > "$OUT_DIR/logs/phase_n3z2cf_analysis.log" 2>&1 || \
  stop "PHASE_N3Z2CF_STOP_DIAGNOSTICS_FAILED.flag" "corrected Z2 full analysis failed; see $OUT_DIR/logs/phase_n3z2cf_analysis.log"

if [[ -f "$OUT_DIR/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag" ]]; then
  echo "complete" > "$STATUS"
  log "complete flag detected"
  exit 0
fi

stop "PHASE_N3Z2CF_STOP_WATCHER_FAILED.flag" "watcher reached end without complete flag"
