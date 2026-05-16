#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3pfv_checkpoint_verification"
LOG="$OUT_DIR/phase_n3pfv_watcher.log"
STATUS="$OUT_DIR/phase_n3pfv_status.txt"
SCENARIOS=(eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces"

rm -f "$OUT_DIR"/PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag "$OUT_DIR"/PHASE_N3PFV_STOP_*.flag
rm -f "$OUT_DIR"/PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md "$OUT_DIR"/phase_n3pfv_status.txt "$OUT_DIR"/phase_n3pfv_watcher.log
rm -f "$OUT_DIR"/tables/phase_n3pfv_*.csv "$OUT_DIR"/plots/*.png "$OUT_DIR"/logs/*.log
find "$OUT_DIR/traces" -type f -name '*.csv' -delete 2>/dev/null || true

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
  if [[ ! -f "$OUT_DIR/PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md" ]]; then
    local reason="${flag#PHASE_N3PFV_STOP_}"
    reason="${reason%.flag}"
    {
      echo "# Phase N3PF-V Checkpoint Verification Report"
      echo
      echo "\`terminal_decision = phase_n3pfv_stopped_${reason}\`"
      echo
      echo '```text'
      echo "$detail"
      echo '```'
      echo
      echo "Can enter N4: no."
    } > "$OUT_DIR/PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md"
  fi
  exit 2
}

run_stage() {
  local name="$1"
  shift
  log "START $name: $*"
  echo "stage:$name" > "$STATUS"
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "FAILED $name rc=$rc"
    return "$rc"
  fi
  log "END $name"
}

log "Phase N3PF-V watcher started"
echo "running" > "$STATUS"

[[ -f "results/env_v2_phase_n3pf_block_projected_full/PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag" ]] || stop "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag" "missing N3PF complete flag"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip" ]] || stop "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag" "missing P3 checkpoint_1000k"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip" ]] || stop "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag" "missing P3 checkpoint_1500k"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip" ]] || stop "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag" "missing P3 final"
[[ -f "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip" ]] || stop "PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag" "missing attention reference"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip" ]] || stop "PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag" "missing no_z reference"
[[ -f "checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip" ]] || stop "PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag" "missing corrected Z2 reference"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

run_stage "compile" python -m py_compile \
  scripts/eval_env_v2_gpsi_ppo_n3pfv.py \
  scripts/analyze_env_v2_phase_n3pfv_results.py

run_stage "eval_verification" python scripts/eval_env_v2_gpsi_ppo_n3pfv.py \
  --result-dir "$OUT_DIR" \
  --eval-seeds 1000 1001 1002 \
  --num-episodes 50 \
  --scenarios "${SCENARIOS[@]}" \
  --policies p3_1000k p3_1500k p3_final attention_full no_z_full z2_corrected_full \
  --include-diagnostic-policies p3_parent_500k p3_1250k \
  --device cpu \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3pfv_eval.log" 2>&1 || \
  stop "PHASE_N3PFV_STOP_EVAL_FAILED.flag" "N3PF-V eval failed; see $OUT_DIR/logs/phase_n3pfv_eval.log"

run_stage "analysis_report" python scripts/analyze_env_v2_phase_n3pfv_results.py \
  --result-dir "$OUT_DIR" \
  --expected-seeds 1000 1001 1002 \
  --expected-episodes 50 \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --z2-success 0.5067 \
  --z2-collision 0.4933 \
  --p3-parent-success 0.5333 \
  --p3-parent-collision 0.4667 \
  > "$OUT_DIR/logs/phase_n3pfv_analysis.log" 2>&1 || \
  stop "PHASE_N3PFV_STOP_DIAGNOSTICS_FAILED.flag" "N3PF-V analysis failed; see $OUT_DIR/logs/phase_n3pfv_analysis.log"

if [[ -f "$OUT_DIR/PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag" ]]; then
  echo "complete" > "$STATUS"
  log "complete flag detected"
  exit 0
fi

stop "PHASE_N3PFV_STOP_WATCHER_FAILED.flag" "watcher reached end without complete flag"
