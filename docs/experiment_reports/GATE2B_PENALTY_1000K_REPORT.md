# Gate-2b Penalty 1000k Report

| run | step | reaction | success | collision | random_success |
|---|---:|---:|---:|---:|---:|
| attention_full_distance_penalty | 250000 | 0.2640 | 1.0000 | 0.0000 | 1.0000 |
| attention_full_distance_penalty | 500000 | 1.2320 | 1.0000 | 0.0000 | 1.0000 |
| attention_full_distance_penalty | 750000 | 11.0327 | 0.9800 | 0.0200 | 1.0000 |
| attention_full_distance_penalty | 1000000 | 7.5360 | 1.0000 | 0.0000 | 1.0000 |
| attention_full_risk_penalty | 250000 | 0.2000 | 0.4200 | 0.0000 | 0.4400 |
| attention_full_risk_penalty | 500000 | 0.2000 | 1.0000 | 0.0000 | 1.0000 |
| attention_full_risk_penalty | 750000 | 0.2080 | 1.0000 | 0.0000 | 1.0000 |
| attention_full_risk_penalty | 1000000 | 1.0760 | 1.0000 | 0.0000 | 1.0000 |
| risk_biased_attention_risk_penalty | 250000 | 0.2000 | 0.7600 | 0.0200 | 0.5000 |
| risk_biased_attention_risk_penalty | 500000 | 0.2000 | 1.0000 | 0.0000 | 1.0000 |
| risk_biased_attention_risk_penalty | 750000 | 0.2400 | 1.0000 | 0.0000 | 1.0000 |
| risk_biased_attention_risk_penalty | 1000000 | 1.0600 | 1.0000 | 0.0000 | 1.0000 |

## Required Answers
- risk cost earlier than distance warning: see gate2b_curve_diagnostics_summary.csv and trace CSVs.
- risk_penalty vs distance_penalty: compare sudden reaction/collision plus random_switch success.
- risk_biased_attention vs risk_penalty only: compare the two risk penalty rows and trace attention weights.
- penalty side effects: random_switch success/time are included in gate2b_by_step.csv.
