#!/usr/bin/env bash
set -euo pipefail

TOTAL_STEPS=100000
N_ENVS=8
DEVICE=cpu
SEED=0

mkdir -p checkpoints/ewma_short
mkdir -p runs/ewma_short

run_train () {
  RUN_NAME=$1
  R_GATE=$2
  LAMBDA_EWMA=$3
  SIGMA_MIN=$4
  USE_RBAR=$5
  RBAR_FLOOR=$6

  echo "=========================================="
  echo "Training ${RUN_NAME}"
  echo "r_gate=${R_GATE}, lambda_ewma=${LAMBDA_EWMA}, sigma_min=${SIGMA_MIN}, use_rbar=${USE_RBAR}, rbar_floor=${RBAR_FLOOR}"
  echo "=========================================="

  python train.py \
    --agg risk \
    --seed "${SEED}" \
    --total_steps "${TOTAL_STEPS}" \
    --n_envs "${N_ENVS}" \
    --device "${DEVICE}" \
    --scenario train_random_switch \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --run_name "${RUN_NAME}" \
    --log_dir "runs/ewma_short/${RUN_NAME}_s${SEED}" \
    --save_path "checkpoints/ewma_short/${RUN_NAME}_s${SEED}.zip" \
    --heartbeat_seconds 30
}

run_train baseline_short 5.0 0.10 0.05 true 0.0
run_train Rgate8 8.0 0.10 0.05 true 0.0
run_train lambda015 5.0 0.15 0.05 true 0.0
run_train Rgate8_lambda015 8.0 0.15 0.05 true 0.0
run_train Rgate8_lambda015_noRbar 8.0 0.15 0.05 false 0.0
run_train Rgate8_lambda015_RbarFloor03 8.0 0.15 0.05 true 0.3
