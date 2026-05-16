# P2 Three-Seed Confirmation Report

- Stage 4 confirmation: True
- key_pareto_positive_scenarios=3
- key_safety_close_scenarios=5

## 750k/1000k Mean Across Seeds
| method | scenario | seeds | success | collision | reaction | near_miss | min_distance | mean_time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attention_full_distance_penalty_wide_d2 | eval_accel_decel | 3 | 1.0000 | 0.0000 | nan | 0.0033 | 2.1662 | 8.0620 |
| attention_full_risk_penalty | eval_accel_decel | 3 | 1.0000 | 0.0000 | nan | 0.0000 | 1.9721 | 7.6567 |
| attention_full_distance_penalty_wide_d2 | eval_ar1 | 3 | 1.0000 | 0.0000 | nan | 0.0100 | 2.0592 | 8.1293 |
| attention_full_risk_penalty | eval_ar1 | 3 | 0.9933 | 0.0067 | nan | 0.0067 | 1.8399 | 7.6773 |
| attention_full_distance_penalty_wide_d2 | eval_mixed_v2 | 3 | 0.9967 | 0.0033 | 0.2293 | 0.0267 | 2.2449 | 8.0340 |
| attention_full_risk_penalty | eval_mixed_v2 | 3 | 0.9967 | 0.0033 | 0.3160 | 0.0400 | 2.0913 | 7.8860 |
| attention_full_distance_penalty_wide_d2 | eval_random_switch | 3 | 1.0000 | 0.0000 | nan | 0.0033 | 2.2823 | 8.2887 |
| attention_full_risk_penalty | eval_random_switch | 3 | 1.0000 | 0.0000 | nan | 0.0033 | 2.1383 | 8.0127 |
| attention_full_distance_penalty_wide_d2 | eval_random_switch_hard | 3 | 0.9733 | 0.0133 | nan | 0.0233 | 1.6759 | 9.6347 |
| attention_full_risk_penalty | eval_random_switch_hard | 3 | 0.9900 | 0.0100 | nan | 0.0733 | 1.3635 | 8.2860 |
| attention_full_distance_penalty_wide_d2 | eval_sinusoidal | 3 | 0.9967 | 0.0033 | nan | 0.0133 | 2.1066 | 8.1087 |
| attention_full_risk_penalty | eval_sinusoidal | 3 | 0.9933 | 0.0067 | nan | 0.0200 | 1.8772 | 7.7027 |
| attention_full_distance_penalty_wide_d2 | eval_sudden_turn | 3 | 1.0000 | 0.0000 | 0.2153 | 0.0033 | 2.0640 | 8.4627 |
| attention_full_risk_penalty | eval_sudden_turn | 3 | 1.0000 | 0.0000 | 0.6420 | 0.0067 | 1.8292 | 8.0373 |
| attention_full_distance_penalty_wide_d2 | eval_threat_validated_sudden | 3 | 0.9967 | 0.0033 | 0.2187 | 0.0167 | 2.2978 | 8.2407 |
| attention_full_risk_penalty | eval_threat_validated_sudden | 3 | 0.9967 | 0.0033 | 0.2553 | 0.0167 | 2.1548 | 8.0167 |
