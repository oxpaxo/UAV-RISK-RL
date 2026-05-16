# P2 Stage 1 OOD Evaluation Report

- Passed Stage 1 gate: True
- Reasons: none

## 750k / 1000k Existing Checkpoint OOD Summary
| method | step | scenario | success | collision | reaction | near_miss | min_distance | mean_time | threat_valid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| attention_full | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.1400 | 1.1364 | 6.7280 | 1.0000 |
| attention_full_distance_penalty_d1 | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.1400 | 1.2626 | 6.7920 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 2.0099 | 8.5480 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 1.9722 | 7.8400 | 1.0000 |
| attention_full | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.4000 | 1.0589 | 6.7640 | 1.0000 |
| attention_full_distance_penalty_d1 | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.1800 | 1.1753 | 6.8080 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 1.9478 | 8.5880 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0200 | 1.8779 | 7.8600 | 1.0000 |
| attention_full | 750000 | eval_mixed_v2 | 0.9800 | 0.0200 | 1.4600 | 0.5000 | 1.0505 | 6.7680 | 1.0000 |
| attention_full_distance_penalty_d1 | 750000 | eval_mixed_v2 | 1.0000 | 0.0000 | 2.1520 | 0.3200 | 1.1312 | 6.8000 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_mixed_v2 | 0.9800 | 0.0200 | 0.2000 | 0.0200 | 2.4789 | 8.3880 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_mixed_v2 | 0.9800 | 0.0200 | 0.2000 | 0.1200 | 2.1496 | 8.0120 | 1.0000 |
| attention_full | 750000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.4000 | 1.0700 | 6.7800 | 1.0000 |
| attention_full_distance_penalty_d1 | 750000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.3000 | 1.1867 | 6.7080 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.0200 | 1.9476 | 8.6000 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.0200 | 1.9189 | 7.9000 | 1.0000 |
| attention_full | 750000 | eval_threat_validated_sudden | 0.9800 | 0.0200 | 0.5755 | 0.5400 | 1.0473 | 6.6440 | 1.0000 |
| attention_full_distance_penalty_d1 | 750000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 2.8360 | 0.3600 | 1.1323 | 6.7800 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 750000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 2.5162 | 8.4240 | 1.0000 |
| attention_full_risk_penalty | 750000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2000 | 0.0800 | 2.2168 | 8.1520 | 1.0000 |
| attention_full | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.5600 | 1.0374 | 6.6920 | 1.0000 |
| attention_full_distance_penalty_d1 | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.1800 | 1.1685 | 6.7280 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 2.0706 | 8.0240 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_accel_decel | 1.0000 | 0.0000 | nan | 0.0000 | 1.9202 | 7.5040 | 1.0000 |
| attention_full | 1000000 | eval_ar1 | 0.9800 | 0.0200 | nan | 0.6600 | 0.9611 | 6.5920 | 1.0000 |
| attention_full_distance_penalty_d1 | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.4000 | 1.0911 | 6.7560 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 2.0004 | 8.0680 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_ar1 | 1.0000 | 0.0000 | nan | 0.0000 | 1.8410 | 7.5480 | 1.0000 |
| attention_full | 1000000 | eval_mixed_v2 | 0.9600 | 0.0400 | 0.4167 | 0.7000 | 0.9866 | 6.5920 | 1.0000 |
| attention_full_distance_penalty_d1 | 1000000 | eval_mixed_v2 | 0.9800 | 0.0200 | 1.7600 | 0.6600 | 1.0039 | 6.7120 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_mixed_v2 | 0.9800 | 0.0200 | 0.2120 | 0.0200 | 2.3936 | 8.0960 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_mixed_v2 | 1.0000 | 0.0000 | 0.2800 | 0.0400 | 2.4029 | 7.8800 | 1.0000 |
| attention_full | 1000000 | eval_sinusoidal | 0.9200 | 0.0800 | nan | 0.6400 | 0.9406 | 6.3400 | 1.0000 |
| attention_full_distance_penalty_d1 | 1000000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.4000 | 1.0924 | 6.7640 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_sinusoidal | 1.0000 | 0.0000 | nan | 0.0200 | 2.0106 | 8.1360 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_sinusoidal | 0.9800 | 0.0200 | nan | 0.0200 | 1.8477 | 7.5760 | 1.0000 |
| attention_full | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.5440 | 0.6600 | 0.9979 | 6.7680 | 1.0000 |
| attention_full_distance_penalty_d1 | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 1.7000 | 0.5600 | 1.0240 | 6.7280 | 1.0000 |
| attention_full_distance_penalty_wide_d2 | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2000 | 0.0000 | 2.4850 | 8.1120 | 1.0000 |
| attention_full_risk_penalty | 1000000 | eval_threat_validated_sudden | 1.0000 | 0.0000 | 0.2160 | 0.0000 | 2.3738 | 7.7280 | 1.0000 |
