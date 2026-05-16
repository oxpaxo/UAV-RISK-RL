#!/usr/bin/env bash
set -euo pipefail

DEVICE=cpu
EPISODES=100

mkdir -p results/ewma_formal/eval

eval_one () {
  RUN_NAME=$1
  R_GATE=$2
  LAMBDA_EWMA=$3
  SIGMA_MIN=$4
  USE_RBAR=$5
  RBAR_FLOOR=$6
  SEED=$7
  STEP=$8

  MODEL_PATH="checkpoints/ewma_formal/${RUN_NAME}_s${SEED}_step${STEP}"

  if [ ! -f "${MODEL_PATH}.zip" ]; then
    echo "WARNING: checkpoint not found: ${MODEL_PATH}.zip"
    return 0
  fi

  echo "=========================================="
  echo "Evaluating ${RUN_NAME}, seed=${SEED}, step=${STEP}"
  echo "=========================================="

  python eval.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch \
    --device "${DEVICE}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --out_csv "results/ewma_formal/eval/${RUN_NAME}_s${SEED}_step${STEP}_random.csv"

  python eval.py \
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
    --out_csv "results/ewma_formal/eval/${RUN_NAME}_s${SEED}_step${STEP}_sudden.csv"

  python eval.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch_hard \
    --device "${DEVICE}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --out_csv "results/ewma_formal/eval/${RUN_NAME}_s${SEED}_step${STEP}_hard.csv"
}

for SEED in 0 1 2
do
  for STEP in 100k 200k 300k 500k
  do
    eval_one Rgate8 8.0 0.10 0.05 true 0.0 "${SEED}" "${STEP}"
    eval_one Rgate8_lambda015_RbarFloor03 8.0 0.15 0.05 true 0.3 "${SEED}" "${STEP}"
    eval_one Rgate8_lambda015 8.0 0.15 0.05 true 0.0 "${SEED}" "${STEP}"
  done
done
