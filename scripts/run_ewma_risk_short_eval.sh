#!/usr/bin/env bash
set -euo pipefail

DEVICE=cpu
EPISODES=100
SEED=0

mkdir -p results/ewma_short/eval
mkdir -p results/ewma_short/diagnostics

run_eval () {
  RUN_NAME=$1
  R_GATE=$2
  LAMBDA_EWMA=$3
  SIGMA_MIN=$4
  USE_RBAR=$5
  RBAR_FLOOR=$6

  MODEL_PATH="checkpoints/ewma_short/${RUN_NAME}_s${SEED}"

  python eval.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch \
    --device "${DEVICE}" \
    --out_csv "results/ewma_short/eval/${RUN_NAME}_s${SEED}_random.csv" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --heartbeat_seconds 10

  python eval.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_csv "results/ewma_short/eval/${RUN_NAME}_s${SEED}_sudden.csv" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --heartbeat_seconds 10

  python scripts/diagnose_sudden_turn.py \
    --model_path "${MODEL_PATH}" \
    --agg risk \
    --seed "${SEED}" \
    --episodes 30 \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --r_gate "${R_GATE}" \
    --lambda_ewma "${LAMBDA_EWMA}" \
    --sigma_min "${SIGMA_MIN}" \
    --use_rbar "${USE_RBAR}" \
    --rbar_floor "${RBAR_FLOOR}" \
    --out_dir "results/ewma_short/diagnostics/${RUN_NAME}_s${SEED}"
}

run_eval baseline_short 5.0 0.10 0.05 true 0.0
run_eval Rgate8 8.0 0.10 0.05 true 0.0
run_eval lambda015 5.0 0.15 0.05 true 0.0
run_eval Rgate8_lambda015 8.0 0.15 0.05 true 0.0
run_eval Rgate8_lambda015_noRbar 8.0 0.15 0.05 false 0.0
run_eval Rgate8_lambda015_RbarFloor03 8.0 0.15 0.05 true 0.3
