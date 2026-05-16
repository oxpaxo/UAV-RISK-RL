# P2 Environment Sanity Report Patched

- Passed patched Stage 0 gate: True
- Reasons: none
- Gate uses planned_threat_valid_rate, not realized_near_miss_rate.

## Random Policy 50 Episodes
| scenario | success | collision | near_miss | realized_near | planned_valid | scenario_valid | init_collision | cpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eval_mixed_v2 | 0.0000 | 0.0400 | 0.2400 | 0.5200 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| eval_threat_validated_sudden | 0.0000 | 0.2800 | 0.6000 | 0.8200 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Short PPO 50 Episodes
| scenario | success | collision | near_miss | realized_near | planned_valid | scenario_valid | init_collision | cpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eval_mixed_v2 | 0.7600 | 0.2400 | 0.7400 | 0.9000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| eval_threat_validated_sudden | 0.6200 | 0.3800 | 0.7800 | 0.9200 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
