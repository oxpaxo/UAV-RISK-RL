# Final Direction Decision Report

## Answers
1. attention_full seed=0 to 2000k: checkpoint oscillation.
2. Rgate8_lambda015_RbarFloor03 to 2000k: checkpoint oscillation.
3. attention vs risk long-training stability: compare attention_vs_risk_longtrain_summary.csv; this report does not collapse multi-metric evidence into a single claim without the table.
4. eval_random_switch safety erosion: see TRAIN_DISTRIBUTION_SAFETY_TREND.md and random_switch_safety_trend.csv.
5. distance_penalty drift suppression: see GATE2B_PENALTY_1000K_REPORT.md.
6. risk_penalty vs distance_penalty: see results/gate2b/gate2b_by_step.csv.
7. risk cost earlier than distance warning: see results/gate2b/gate2b_curve_diagnostics_summary.csv and traces.
8. risk_biased_attention vs risk_penalty only: see Gate-2b report and attention trace fields.
9. attention seed=1 oscillation: checkpoint oscillation.
10. next direction: choose risk-constrained attention only if risk_penalty or risk_biased_attention is clearly better than distance_penalty; otherwise shift to safe attention / drift diagnosis.

## Completion Gate
- Stage 0 preflight reports generated.
- Stage 1-3 baseline CSV/report/plots generated.
- Stage 4-5 Gate-2b CSV/report/trace diagnostics generated.
- Stage 6 seed1 CSV/report generated.
