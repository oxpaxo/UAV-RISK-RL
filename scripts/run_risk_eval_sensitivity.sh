#!/usr/bin/env bash
set -euo pipefail

DEVICE=cpu
EPISODES=30
OUT_DIR=results/risk_diagnostics_sensitivity
MODEL=checkpoints/risk_full_rbar_s0.zip

mkdir -p "${OUT_DIR}/traces" "${OUT_DIR}/plots" "${OUT_DIR}/summary"

run_diag() {
  local name="$1"
  local sigma_min="$2"
  local lambda_ewma="$3"
  local r_gate="$4"
  local use_rbar="$5"

  python scripts/diagnose_sudden_turn.py \
    --model_path "${MODEL}" \
    --agg risk \
    --seed 0 \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_dir "${OUT_DIR}/${name}" \
    --use_rbar "${use_rbar}" \
    --sigma_min "${sigma_min}" \
    --lambda_ewma "${lambda_ewma}" \
    --r_gate "${r_gate}"
}

run_diag default 0.05 0.10 5.0 true
run_diag sigma_min_0p10 0.10 0.10 5.0 true
run_diag lambda_0p15 0.05 0.15 5.0 true
run_diag lambda_0p20 0.05 0.20 5.0 true
run_diag Rgate_8 0.05 0.10 8.0 true
run_diag no_rbar 0.05 0.10 5.0 false
run_diag sigma0p10_Rgate8_no_rbar 0.10 0.10 8.0 false
