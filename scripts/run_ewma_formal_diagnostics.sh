#!/usr/bin/env bash
set -euo pipefail

DEVICE=cpu
EPISODES=30

mkdir -p results/ewma_formal/diagnostics

diag_one () {
  RUN_NAME=$1
  R_GATE=$2
  LAMBDA_EWMA=$3
  SIGMA_MIN=$4
  USE_RBAR=$5
  RBAR_FLOOR=$6
  SEED=$7
  STEP=$8

  MODEL_PATH="checkpoints/ewma_formal/${RUN_NAME}_s${SEED}_step${STEP}"
  OUT_DIR="results/ewma_formal/diagnostics/${RUN_NAME}_s${SEED}_step${STEP}"

  if [ ! -f "${MODEL_PATH}.zip" ]; then
    echo "WARNING: checkpoint not found: ${MODEL_PATH}.zip"
    return 0
  fi

  echo "=========================================="
  echo "Diagnosing ${RUN_NAME}, seed=${SEED}, step=${STEP}"
  echo "=========================================="

  python scripts/diagnose_sudden_turn.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --out_dir "${OUT_DIR}"
}

for SEED in 0 1 2
do
  for STEP in 100k 200k 300k 500k
  do
    diag_one Rgate8 8.0 0.10 0.05 true 0.0 "${SEED}" "${STEP}"
    diag_one Rgate8_lambda015_RbarFloor03 8.0 0.15 0.05 true 0.3 "${SEED}" "${STEP}"
    diag_one Rgate8_lambda015 8.0 0.15 0.05 true 0.0 "${SEED}" "${STEP}"
  done
done
