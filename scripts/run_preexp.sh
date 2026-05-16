#!/usr/bin/env bash
set -euo pipefail

TOTAL_STEPS=500000
N_ENVS=8
DEVICE=cpu
HEARTBEAT_SECONDS=30

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_and_log() {
  echo "[$(timestamp)] START $*"
  "$@"
  echo "[$(timestamp)] END   $*"
}

for SEED in 0 1 2
do
  run_and_log python train.py \
    --agg risk \
    --seed "${SEED}" \
    --total_steps "${TOTAL_STEPS}" \
    --n_envs "${N_ENVS}" \
    --device "${DEVICE}" \
    --log_dir "runs/risk_full_rbar_s${SEED}" \
    --save_path "checkpoints/risk_full_rbar_s${SEED}.zip" \
    --use_rbar true \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python train.py \
    --agg attention \
    --seed "${SEED}" \
    --total_steps "${TOTAL_STEPS}" \
    --n_envs "${N_ENVS}" \
    --device "${DEVICE}" \
    --log_dir "runs/attention_full_s${SEED}" \
    --save_path "checkpoints/attention_full_s${SEED}.zip" \
    --use_rbar false \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"
done
