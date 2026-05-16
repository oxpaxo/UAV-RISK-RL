#!/usr/bin/env bash
set -euo pipefail

EPISODES=100
DEVICE=cpu
HEARTBEAT_SECONDS=15

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
  run_and_log python eval.py \
    --model_path "checkpoints/risk_full_rbar_s${SEED}.zip" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch \
    --device "${DEVICE}" \
    --out_csv "results/risk_full_rbar_s${SEED}_random.csv" \
    --use_rbar true \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python eval.py \
    --model_path "checkpoints/risk_full_rbar_s${SEED}.zip" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch_hard \
    --device "${DEVICE}" \
    --out_csv "results/risk_full_rbar_s${SEED}_random_hard.csv" \
    --use_rbar true \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python eval.py \
    --model_path "checkpoints/risk_full_rbar_s${SEED}.zip" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_csv "results/risk_full_rbar_s${SEED}_sudden.csv" \
    --use_rbar true \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python eval.py \
    --model_path "checkpoints/attention_full_s${SEED}.zip" \
    --agg attention \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch \
    --device "${DEVICE}" \
    --out_csv "results/attention_full_s${SEED}_random.csv" \
    --use_rbar false \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python eval.py \
    --model_path "checkpoints/attention_full_s${SEED}.zip" \
    --agg attention \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_random_switch_hard \
    --device "${DEVICE}" \
    --out_csv "results/attention_full_s${SEED}_random_hard.csv" \
    --use_rbar false \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"

  run_and_log python eval.py \
    --model_path "checkpoints/attention_full_s${SEED}.zip" \
    --agg attention \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_csv "results/attention_full_s${SEED}_sudden.csv" \
    --use_rbar false \
    --heartbeat_seconds "${HEARTBEAT_SECONDS}"
done
