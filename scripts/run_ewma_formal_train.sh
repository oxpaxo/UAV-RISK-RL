#!/usr/bin/env bash
set -euo pipefail

TOTAL_STEPS=500000
N_ENVS=8
DEVICE=cpu
CKPT_STEPS=100000,200000,300000,500000

mkdir -p checkpoints/ewma_formal
mkdir -p runs/ewma_formal

run_train () {
  RUN_NAME=$1
  R_GATE=$2
  LAMBDA_EWMA=$3
  SIGMA_MIN=$4
  USE_RBAR=$5
  RBAR_FLOOR=$6
  SEED=$7

  echo "=========================================="
  echo "Formal training ${RUN_NAME}, seed=${SEED}"
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
    --save_checkpoints true \
    --checkpoint_steps "${CKPT_STEPS}" \
    --checkpoint_dir checkpoints/ewma_formal \
    --run_name "${RUN_NAME}" \
    --log_dir "runs/ewma_formal/${RUN_NAME}_s${SEED}" \
    --save_path "checkpoints/ewma_formal/${RUN_NAME}_s${SEED}_final.zip" \
    --heartbeat_seconds 30
}

for SEED in 0 1 2
do
  run_train Rgate8 8.0 0.10 0.05 true 0.0 "${SEED}"
  run_train Rgate8_lambda015_RbarFloor03 8.0 0.15 0.05 true 0.3 "${SEED}"
  run_train Rgate8_lambda015 8.0 0.15 0.05 true 0.0 "${SEED}"
done
