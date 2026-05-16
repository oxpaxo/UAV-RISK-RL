# P2 Stage 3 Seed0 Pareto Report

- Go to Stage 4: True
- Reasons: key_pareto_positive_scenarios=4, key_safety_close_scenarios=6, baseline_issue_scenarios=6, wide_d2_dominates_scenarios=2, risk_wide_d2_no_difference_scenarios=2, risk_safety_gap_scenarios=0

## Required Stage 3 Answers
1. train_mixed_modes_v2 baseline safety/reaction drift: yes; baseline_issue_scenarios=6.
2. risk_penalty safety close to wide_d2: yes; key_safety_close_scenarios=6.
3. risk_penalty more efficient than wide_d2: yes; key_pareto_positive_scenarios=4.
4. risk on Pareto front: yes under the Stage 3 gate.
5. sinusoidal / accel_decel / ar1 single-mode results:
   - eval_sinusoidal: attention_full: success=1.0000, collision=0.0000, near=0.2600, min_d=1.2078, time=6.8800; attention_full_distance_penalty_wide_d2: success=0.9900, collision=0.0100, near=0.0200, min_d=2.0492, time=8.0180; attention_full_risk_penalty: success=0.9800, collision=0.0200, near=0.0400, min_d=1.9177, time=7.7000
   - eval_accel_decel: attention_full: success=1.0000, collision=0.0000, near=0.1800, min_d=1.2232, time=6.8340; attention_full_distance_penalty_wide_d2: success=1.0000, collision=0.0000, near=0.0000, min_d=2.1124, time=7.9180; attention_full_risk_penalty: success=1.0000, collision=0.0000, near=0.0000, min_d=2.0088, time=7.6580
   - eval_ar1: attention_full: success=1.0000, collision=0.0000, near=0.4000, min_d=1.1402, time=6.8680; attention_full_distance_penalty_wide_d2: success=1.0000, collision=0.0000, near=0.0000, min_d=2.0332, time=7.9540; attention_full_risk_penalty: success=1.0000, collision=0.0000, near=0.0000, min_d=1.9063, time=7.6760
6. mixed_v2 stability: wide_d2 more stable or tied. attention_full_distance_penalty_wide_d2: success=1.0000, collision=0.0000, near=0.0200, min_d=2.3834, time=8.0680; attention_full_risk_penalty: success=1.0000, collision=0.0000, near=0.0000, min_d=2.3419, time=7.9420
7. motion-mode adaptation signal: no; risk_sum_mean range across scenarios=0.0098.
8. Stage 4 three-seed confirmation worth running: yes.

## Gate By Scenario
| scenario | close_safety | risk_faster | risk_safer | pareto_positive | wide_dominates | no_difference | baseline_issue | risk_time | wide_time | risk_near | wide_near | risk_min_d | wide_min_d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval_accel_decel | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 7.6580 | 7.9180 | 0.0000 | 0.0000 | 2.0088 | 2.1124 |
| eval_ar1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 7.6760 | 7.9540 | 0.0000 | 0.0000 | 1.9063 | 2.0332 |
| eval_mixed_v2 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 7.9420 | 8.0680 | 0.0000 | 0.0200 | 2.3419 | 2.3834 |
| eval_random_switch_hard | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 8.6060 | 9.2760 | 0.0900 | 0.0400 | 1.4148 | 1.6601 |
| eval_sinusoidal | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 7.7000 | 8.0180 | 0.0400 | 0.0200 | 1.9177 | 2.0492 |
| eval_sudden_turn | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 7.9500 | 8.1960 | 0.0000 | 0.0000 | 1.8944 | 2.0299 |
| eval_threat_validated_sudden | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 8.0920 | 8.1780 | 0.0000 | 0.0100 | 2.4631 | 2.4939 |

## Risk Adaptation Summary
| scenario | method | risk_sum | risk_max | dist_cost_nonzero | min_distance | mean_time | near_miss | collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eval_accel_decel | attention_full | 0.0285 | 0.0281 | 0.1800 | 1.2232 | 6.8340 | 0.1800 | 0.0000 |
| eval_accel_decel | attention_full_distance_penalty_wide_d2 | 0.0001 | 0.0001 | 0.3400 | 2.1124 | 7.9180 | 0.0000 | 0.0000 |
| eval_accel_decel | attention_full_risk_penalty | 0.0001 | 0.0001 | 0.0000 | 2.0088 | 7.6580 | 0.0000 | 0.0000 |
| eval_ar1 | attention_full | 0.0382 | 0.0376 | 0.4000 | 1.1402 | 6.8680 | 0.4000 | 0.0000 |
| eval_ar1 | attention_full_distance_penalty_wide_d2 | 0.0001 | 0.0001 | 0.4300 | 2.0332 | 7.9540 | 0.0000 | 0.0000 |
| eval_ar1 | attention_full_risk_penalty | 0.0002 | 0.0002 | 0.0000 | 1.9063 | 7.6760 | 0.0000 | 0.0000 |
| eval_mixed_v2 | attention_full | 0.0470 | 0.0469 | 0.5300 | 1.1900 | 6.8900 | 0.5300 | 0.0000 |
| eval_mixed_v2 | attention_full_distance_penalty_wide_d2 | 0.0022 | 0.0022 | 0.2500 | 2.3834 | 8.0680 | 0.0200 | 0.0000 |
| eval_mixed_v2 | attention_full_risk_penalty | 0.0008 | 0.0008 | 0.0000 | 2.3419 | 7.9420 | 0.0000 | 0.0000 |
| eval_random_switch | attention_full | 0.0508 | 0.0492 | 0.2900 | 1.2224 | 6.9360 | 0.2900 | 0.0000 |
| eval_random_switch | attention_full_distance_penalty_wide_d2 | 0.0004 | 0.0004 | 0.3100 | 2.1721 | 8.2000 | 0.0000 | 0.0000 |
| eval_random_switch | attention_full_risk_penalty | 0.0002 | 0.0002 | 0.0000 | 2.1298 | 8.1280 | 0.0000 | 0.0000 |
| eval_random_switch_hard | attention_full | 0.1472 | 0.1393 | 0.8700 | 0.8304 | 6.2980 | 0.8700 | 0.1000 |
| eval_random_switch_hard | attention_full_distance_penalty_wide_d2 | 0.0065 | 0.0061 | 0.9000 | 1.6601 | 9.2760 | 0.0400 | 0.0200 |
| eval_random_switch_hard | attention_full_risk_penalty | 0.0099 | 0.0089 | 0.0900 | 1.4148 | 8.6060 | 0.0900 | 0.0100 |
| eval_sinusoidal | attention_full | 0.0399 | 0.0396 | 0.2600 | 1.2078 | 6.8800 | 0.2600 | 0.0000 |
| eval_sinusoidal | attention_full_distance_penalty_wide_d2 | 0.0017 | 0.0017 | 0.3500 | 2.0492 | 8.0180 | 0.0200 | 0.0100 |
| eval_sinusoidal | attention_full_risk_penalty | 0.0048 | 0.0048 | 0.0400 | 1.9177 | 7.7000 | 0.0400 | 0.0200 |
| eval_sudden_turn | attention_full | 0.0504 | 0.0499 | 0.4900 | 1.1123 | 6.9180 | 0.4900 | 0.0000 |
| eval_sudden_turn | attention_full_distance_penalty_wide_d2 | 0.0008 | 0.0008 | 0.4600 | 2.0299 | 8.1960 | 0.0000 | 0.0000 |
| eval_sudden_turn | attention_full_risk_penalty | 0.0007 | 0.0007 | 0.0000 | 1.8944 | 7.9500 | 0.0000 | 0.0000 |
| eval_threat_validated_sudden | attention_full | 0.0536 | 0.0532 | 0.5400 | 1.1145 | 6.8700 | 0.5400 | 0.0200 |
| eval_threat_validated_sudden | attention_full_distance_penalty_wide_d2 | 0.0003 | 0.0003 | 0.1800 | 2.4939 | 8.1780 | 0.0100 | 0.0000 |
| eval_threat_validated_sudden | attention_full_risk_penalty | 0.0006 | 0.0006 | 0.0000 | 2.4631 | 8.0920 | 0.0000 | 0.0000 |
