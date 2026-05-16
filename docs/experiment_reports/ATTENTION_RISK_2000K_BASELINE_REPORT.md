# Attention vs Risk 2000k Baseline Report

- attention_full: checkpoint oscillation based on sudden-turn reaction curve.
- risk_Rgate8_lambda015_RbarFloor03: checkpoint oscillation based on sudden-turn reaction curve.

## Sudden-Turn Reaction
| run | step | reaction | success | collision |
|---|---:|---:|---:|---:|
| attention_full | 250000 | 0.2080 | 1.0000 | 0.0000 |
| attention_full | 500000 | 1.8600 | 1.0000 | 0.0000 |
| attention_full | 750000 | 11.4240 | 1.0000 | 0.0000 |
| attention_full | 1000000 | 6.0760 | 1.0000 | 0.0000 |
| attention_full | 1250000 | 4.7720 | 1.0000 | 0.0000 |
| attention_full | 1500000 | 12.0000 | 0.9800 | 0.0200 |
| attention_full | 1750000 | 3.3429 | 0.9800 | 0.0200 |
| attention_full | 2000000 | 16.1360 | 1.0000 | 0.0000 |
| risk_Rgate8_lambda015_RbarFloor03 | 100000 | 0.4120 | 0.9800 | 0.0200 |
| risk_Rgate8_lambda015_RbarFloor03 | 200000 | 0.2480 | 1.0000 | 0.0000 |
| risk_Rgate8_lambda015_RbarFloor03 | 300000 | 7.7673 | 0.9800 | 0.0200 |
| risk_Rgate8_lambda015_RbarFloor03 | 500000 | 9.8160 | 1.0000 | 0.0000 |
| risk_Rgate8_lambda015_RbarFloor03 | 750000 | 21.3959 | 0.9800 | 0.0200 |
| risk_Rgate8_lambda015_RbarFloor03 | 1000000 | 8.8939 | 0.9800 | 0.0200 |
| risk_Rgate8_lambda015_RbarFloor03 | 1500000 | 4.5878 | 0.9800 | 0.0200 |
| risk_Rgate8_lambda015_RbarFloor03 | 2000000 | 12.9200 | 1.0000 | 0.0000 |

## Required Answers
- attention to 2000k: see classification above.
- risk to 2000k: see classification above.
- long-training stability: compare reaction, collision, and safety-cost curves in CSV/plots.
- pure risk hard weighting should continue only if it is at least as stable as attention.
