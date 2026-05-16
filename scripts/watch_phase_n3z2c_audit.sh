#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3z2c_audit"
CKPT_DIR="checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0"
LOG="$OUT_DIR/phase_n3z2c_audit_watcher.log"
STATUS="$OUT_DIR/phase_n3z2c_audit_status.txt"
SCENARIOS=(eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces" "$CKPT_DIR"

rm -f "$OUT_DIR"/PHASE_N3Z2C_AUDIT_COMPLETE.flag "$OUT_DIR"/PHASE_N3Z2C_AUDIT_STOP_*.flag
rm -f "$OUT_DIR"/PHASE_N3Z2C_AUDIT_REPORT.md "$OUT_DIR"/phase_n3z2c_audit_status.txt "$OUT_DIR"/phase_n3z2c_audit_watcher.log
rm -f "$OUT_DIR"/tables/phase_n3z2c_audit_*.csv "$OUT_DIR"/tables/phase_n3z2c_audit_*.json
rm -f "$OUT_DIR"/plots/*.png
rm -f "$OUT_DIR"/logs/*.log
find "$OUT_DIR/traces" -type f -name '*.csv' -delete 2>/dev/null || true
rm -f "$CKPT_DIR"/TRAIN_RUNNING.lock "$CKPT_DIR"/TRAIN_COMPLETE.flag "$CKPT_DIR"/TRAIN_STATUS.json
rm -f "$CKPT_DIR"/parent_500k.zip "$CKPT_DIR"/checkpoint_750k.zip "$CKPT_DIR"/final.zip "$CKPT_DIR"/best_by_eval.zip "$CKPT_DIR"/config_resolved.yaml

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
  if [[ ! -f "$OUT_DIR/PHASE_N3Z2C_AUDIT_REPORT.md" ]]; then
    {
      echo "# Phase N3Z2C-Audit Report"
      echo
      echo "\`terminal_decision = phase_n3z2c_audit_stopped_${flag#PHASE_N3Z2C_AUDIT_STOP_}\`"
      echo
      echo '```text'
      echo "$detail"
      echo '```'
      echo
      echo "Can enter N4: no."
    } > "$OUT_DIR/PHASE_N3Z2C_AUDIT_REPORT.md"
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

log "Phase N3Z2C-Audit watcher started"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n3z2c_z2_continuation/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag" ]]; then
  stop "PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag" "missing N3Z2C complete flag"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

run_stage "compile" python -m py_compile \
  scripts/audit_phase_n3z2c_parent_resume.py \
  scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py \
  scripts/eval_env_v2_gpsi_ppo_n3z2c_audit.py \
  scripts/analyze_env_v2_phase_n3z2c_audit.py

run_stage "audit_parent_resume_cpu" python scripts/audit_phase_n3z2c_parent_resume.py \
  --n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --n3fz-result-dir results/env_v2_phase_n3fz_noz_full_z_screen \
  --out-dir "$OUT_DIR" \
  > "$OUT_DIR/logs/phase_n3z2c_audit_parent_resume.log" 2>&1 || \
  stop "PHASE_N3Z2C_AUDIT_STOP_RESUME_SEMANTICS_UNRESOLVED.flag" "parent/resume/CPU audit failed; see $OUT_DIR/logs/phase_n3z2c_audit_parent_resume.log"

FIXED_PARENT="$(python - <<'PY'
import json
from pathlib import Path
p=Path('results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_parent_selection_fixed.json')
payload=json.loads(p.read_text())
print(payload['selected_path'])
PY
)"
if [[ ! -f "$FIXED_PARENT" ]]; then
  stop "PHASE_N3Z2C_AUDIT_STOP_CORRECTED_PARENT_MISSING.flag" "fixed parent missing: $FIXED_PARENT"
fi
cp "$FIXED_PARENT" "$CKPT_DIR/parent_500k.zip"
log "FIXED_PARENT selected: $FIXED_PARENT"

run_stage "train_corrected_short" python scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
  --resume "$FIXED_PARENT" \
  --out-dir "$CKPT_DIR" \
  --additional-steps 250000 \
  --parent-total-steps 500000 \
  --target-total-steps 750000 \
  --checkpoint-total-steps 750000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --heartbeat-seconds 300 \
  > "$OUT_DIR/logs/phase_n3z2c_audit_train_corrected_short.log" 2>&1 || \
  stop "PHASE_N3Z2C_AUDIT_STOP_TRAIN_FAILED.flag" "corrected short continuation failed; see $OUT_DIR/logs/phase_n3z2c_audit_train_corrected_short.log"

run_stage "eval_corrected_short" python scripts/eval_env_v2_gpsi_ppo_n3z2c_audit.py \
  --result-dir "$OUT_DIR" \
  --corrected-checkpoint-dir "$CKPT_DIR" \
  --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --num-episodes 50 \
  --device cpu \
  --scenarios "${SCENARIOS[@]}" \
  --write-traces \
  > "$OUT_DIR/logs/phase_n3z2c_audit_eval.log" 2>&1 || \
  stop "PHASE_N3Z2C_AUDIT_STOP_EVAL_FAILED.flag" "Audit eval failed; see $OUT_DIR/logs/phase_n3z2c_audit_eval.log"

run_stage "analysis_report" python scripts/analyze_env_v2_phase_n3z2c_audit.py \
  --result-dir "$OUT_DIR" \
  --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  > "$OUT_DIR/logs/phase_n3z2c_audit_analysis.log" 2>&1 || \
  stop "PHASE_N3Z2C_AUDIT_STOP_DIAGNOSTICS_FAILED.flag" "Audit analysis failed; see $OUT_DIR/logs/phase_n3z2c_audit_analysis.log"

if [[ -f "$OUT_DIR/PHASE_N3Z2C_AUDIT_COMPLETE.flag" ]]; then
  echo "complete" > "$STATUS"
  log "complete flag detected"
  exit 0
fi

stop "PHASE_N3Z2C_AUDIT_STOP_WATCHER_FAILED.flag" "watcher reached end without complete flag"
