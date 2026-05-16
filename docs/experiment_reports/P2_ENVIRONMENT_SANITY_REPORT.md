# P2 Environment Sanity Report

## Stage 0 Result
- Passed: False
- Reasons: threat_valid_rate_below_0p8

## Random Policy Rollout
| scenario | success | collision | near_miss | min_distance | scenario_valid | threat_valid | init_collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval_sinusoidal | 0.0000 | 0.2000 | 0.9000 | 0.8029 | 1.0000 | 1.0000 | 0.0000 |
| eval_accel_decel | 0.0000 | 0.5000 | 0.7000 | 0.8473 | 1.0000 | 1.0000 | 0.0000 |
| eval_ar1 | 0.0000 | 0.6000 | 0.8000 | 0.7490 | 1.0000 | 1.0000 | 0.0000 |
| eval_mixed_v2 | 0.0000 | 0.1000 | 0.3000 | 1.3070 | 1.0000 | 0.7000 | 0.0000 |
| eval_threat_validated_sudden | 0.0000 | 0.3000 | 0.6000 | 1.0022 | 1.0000 | 0.7000 | 0.0000 |

## Short PPO Sanity
| scenario | success | collision | near_miss | mean_time | reward | scenario_valid | threat_valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval_accel_decel | 0.3500 | 0.6500 | 1.0000 | 4.7800 | 1.9455 | 1.0000 | 1.0000 |
| eval_ar1 | 0.5500 | 0.4500 | 1.0000 | 6.8000 | 8.2997 | 1.0000 | 1.0000 |
| eval_mixed_v2 | 0.8000 | 0.2000 | 0.5500 | 9.1000 | 16.2673 | 1.0000 | 0.7500 |
| eval_random_switch | 0.3500 | 0.6500 | 0.9000 | 5.2500 | 2.3107 | 1.0000 | 1.0000 |
| eval_sinusoidal | 0.4500 | 0.5500 | 1.0000 | 6.5300 | 5.7998 | 1.0000 | 1.0000 |
