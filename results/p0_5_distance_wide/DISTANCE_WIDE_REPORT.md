# Distance Penalty Wide Ablation Report

## Scope
- New method: attention_full_distance_penalty_wide_d2.
- Train seed: 0.
- d_warning: 2.0.
- Total steps: 1000000; checkpoints: 250000, 500000, 750000, 1000000.
- Eval episodes: 50; eval_seed: 1000.

## Main Judgment
- wide distance_penalty approaches risk_penalty on sudden-turn reaction.
- At 750k sudden_turn: d1 reaction=11.0327 s, wide_d2 reaction=0.2000 s, risk reaction=0.2080 s.
- At 750k random_switch: wide_d2 success=1.000, risk success=1.000.

## 750k Comparison
| scenario | method | reaction | success | collision | mean_time | min_dist_mean | near_miss | cost_nonzero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eval_random_switch | attention_full | nan | 1.000 | 0.000 | 6.8240 | 1.1767 | 0.240 | 0.240 |
| eval_random_switch | attention_full_distance_penalty_d1 | nan | 1.000 | 0.000 | 6.8760 | 1.3091 | 0.120 | 0.120 |
| eval_random_switch | attention_full_distance_penalty_wide_d2 | nan | 1.000 | 0.000 | 8.7160 | 2.1611 | 0.000 | 0.400 |
| eval_random_switch | attention_full_risk_penalty | nan | 1.000 | 0.000 | 8.1000 | 2.0643 | 0.000 | 0.000 |
| eval_random_switch_hard | attention_full | nan | 0.940 | 0.060 | 6.3760 | 0.8442 | 0.840 | 0.840 |
| eval_random_switch_hard | attention_full_distance_penalty_d1 | nan | 0.900 | 0.100 | 6.1520 | 0.8410 | 0.840 | 0.840 |
| eval_random_switch_hard | attention_full_distance_penalty_wide_d2 | nan | 1.000 | 0.000 | 9.5040 | 1.6929 | 0.000 | 0.920 |
| eval_random_switch_hard | attention_full_risk_penalty | nan | 1.000 | 0.000 | 8.4680 | 1.5000 | 0.060 | 0.060 |
| eval_sudden_turn | attention_full | 11.4240 | 1.000 | 0.000 | 6.7800 | 1.0405 | 0.420 | 0.420 |
| eval_sudden_turn | attention_full_distance_penalty_d1 | 11.0327 | 0.980 | 0.020 | 6.7040 | 1.1184 | 0.320 | 0.320 |
| eval_sudden_turn | attention_full_distance_penalty_wide_d2 | 0.2000 | 1.000 | 0.000 | 8.6920 | 1.9948 | 0.000 | 0.520 |
| eval_sudden_turn | attention_full_risk_penalty | 0.2080 | 1.000 | 0.000 | 8.1120 | 1.9129 | 0.000 | 0.000 |
| mixed_uncertainty | attention_full | 1.6040 | 0.960 | 0.040 | 6.7680 | 0.9928 | 0.440 | 0.440 |
| mixed_uncertainty | attention_full_distance_penalty_d1 | 2.9640 | 0.980 | 0.020 | 6.9080 | 1.0150 | 0.520 | 0.520 |
| mixed_uncertainty | attention_full_distance_penalty_wide_d2 | 0.2000 | 0.860 | 0.140 | 9.3840 | 1.2157 | 0.360 | 0.920 |
| mixed_uncertainty | attention_full_risk_penalty | 0.2040 | 0.800 | 0.200 | 8.4160 | 1.0538 | 0.520 | 0.500 |

## Artifacts
- results/p0_5_distance_wide/distance_wide_by_step_scenario.csv
- results/p0_5_distance_wide/eval/*.csv
- checkpoints/p0_5_distance_wide/*.zip
