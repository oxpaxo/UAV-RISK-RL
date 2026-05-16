# P2 Stage 2 Rich Training Seed0 Report

## Scope
- Continued from patched Stage 1; Stage 0 and Stage 1 were not rerun.
- Trained M0/M1/M2 on train_mixed_modes_v2 with seed=0, total_steps=1000000, n_envs=16, device=cpu.
- Saved and evaluated checkpoints at 250k / 500k / 750k / 1000k with eval_seed=1000 and episodes=50.

## 750k/1000k Main Table
| method | step | scenario | success | collision | reaction | near_miss | min_distance | mean_time | threat_valid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| attention_full | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.1000 | 1.3276 | 6.9320 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 2.1043 | 8.0360 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 2.0363 | 7.8120 | 1.0000 |
| attention_full | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.2400 | 1.2217 | 6.9720 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 2.0424 | 8.0920 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 1.9257 | 7.7920 | 1.0000 |
| attention_full | 750000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.6160 | 0.4400 | 1.3278 | 7.0040 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.2160 | 0.0400 | 2.3507 | 8.1240 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 2.5761 | 8.1840 | 1.0000 |
| attention_full | 750000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.2000 | 1.3254 | 7.0360 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.0000 | 2.1978 | 8.2400 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.0000 | 2.1523 | 8.2800 | 1.0000 |
| attention_full | 750000 | eval_random_switch_hard | 0.9000 | 0.1000 | nan | 0.8400 | 0.8664 | 6.3600 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_random_switch_hard | 0.9800 | 0.0200 | nan | 0.0400 | 1.6747 | 8.7000 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_random_switch_hard | 1.0000 | 0.0000 | nan | 0.1000 | 1.3757 | 8.4720 | 1.0000 |
| attention_full | 750000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.1800 | 1.3147 | 6.9800 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.0400 | 2.0456 | 8.1200 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.0400 | 1.9491 | 7.8320 | 1.0000 |
| attention_full | 750000 | eval_sudden_turn | 1.0000 | 0.0000 | 1.5680 | 0.3800 | 1.2094 | 7.0040 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_sudden_turn | 1.0000 | 0.0000 | 0.2040 | 0.0000 | 2.0546 | 8.2320 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_sudden_turn | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 1.9342 | 8.1000 | 1.0000 |
| attention_full | 750000 | eval_threat_validated_sudden | 0.9800 | 0.0200 | 0.5280 | 0.4600 | 1.2281 | 6.9920 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2480 | 0.0000 | 2.4507 | 8.1720 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 2.6923 | 8.3760 | 1.0000 |
| attention_full | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.2600 | 1.1188 | 6.7360 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 2.1205 | 7.8000 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 1.9814 | 7.5040 | 1.0000 |
| attention_full | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.5600 | 1.0587 | 6.7640 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 2.0240 | 7.8160 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 1.8868 | 7.5600 | 1.0000 |
| attention_full | 1000000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.8080 | 0.6200 | 1.0521 | 6.7760 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.2120 | 0.0000 | 2.4162 | 8.0120 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.3360 | 0.0000 | 2.1078 | 7.7000 | 1.0000 |
| attention_full | 1000000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.3800 | 1.1194 | 6.8360 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.0000 | 2.1463 | 8.1600 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_random_switch | 1.0000 | 0.0000 | nan | 0.0000 | 2.1073 | 7.9760 | 1.0000 |
| attention_full | 1000000 | eval_random_switch_hard | 0.9000 | 0.1000 | nan | 0.9000 | 0.7943 | 6.2360 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_random_switch_hard | 0.9600 | 0.0200 | nan | 0.0400 | 1.6454 | 9.8520 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_random_switch_hard | 0.9800 | 0.0200 | nan | 0.0800 | 1.4539 | 8.7400 | 1.0000 |
| attention_full | 1000000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.3400 | 1.1009 | 6.7800 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.0000 | 2.0529 | 7.9160 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.0400 | 1.8862 | 7.5680 | 1.0000 |
| attention_full | 1000000 | eval_sudden_turn | 1.0000 | 0.0000 | 2.4960 | 0.6000 | 1.0152 | 6.8320 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_sudden_turn | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 2.0052 | 8.1600 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_sudden_turn | 1.0000 | 0.0000 | 1.8720 | 0.0000 | 1.8547 | 7.8000 | 1.0000 |
| attention_full | 1000000 | eval_threat_validated_sudden | 0.9800 | 0.0200 | 0.7200 | 0.6200 | 1.0010 | 6.7480 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2000 | 0.0200 | 2.5372 | 8.1840 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.3200 | 0.0000 | 2.2338 | 7.8080 | 1.0000 |
