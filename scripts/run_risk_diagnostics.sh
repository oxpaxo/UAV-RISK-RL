#!/usr/bin/env bash
set -euo pipefail

DEVICE=cpu
EPISODES=30
OUT_DIR=results/risk_diagnostics

mkdir -p "${OUT_DIR}/traces" "${OUT_DIR}/plots" "${OUT_DIR}/summary"

for SEED in 0 1 2
do
  python scripts/diagnose_sudden_turn.py \
    --model_path "checkpoints/risk_full_rbar_s${SEED}.zip" \
    --agg risk \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_dir "${OUT_DIR}" \
    --use_rbar true

  python scripts/diagnose_sudden_turn.py \
    --model_path "checkpoints/attention_full_s${SEED}.zip" \
    --agg attention \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --scenario eval_sudden_turn \
    --device "${DEVICE}" \
    --out_dir "${OUT_DIR}" \
    --use_rbar false
done

python scripts/plot_risk_diagnostics.py \
  --trace_dir "${OUT_DIR}/traces" \
  --summary_dir "${OUT_DIR}/summary" \
  --out_dir "${OUT_DIR}/plots"

python scripts/summarize_risk_diagnostics.py \
  --summary_dir "${OUT_DIR}/summary" \
  --out_csv "${OUT_DIR}/summary/diagnostic_summary.csv" \
  --out_md "${OUT_DIR}/summary/diagnostic_summary.md"
