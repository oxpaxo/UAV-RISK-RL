# Phase N3PF-MS Multi-Training-Seed Report

## Terminal Decision

`terminal_decision = phase_n3pf_ms_multi_seed_complete`

P3 block_projected training seeds 1 and 2 completed and were evaluated with the imported seed0/reference verification rows.

## Decision

| p3_multiseed_status | p3_1500k_training_seed_mean_success | p3_1500k_training_seed_mean_collision | attention_success_reference | attention_collision_reference | noz_success_reference | noz_collision_reference | stable_minimum_gate | strong_positive_gate | comparable_positive_gate | attention_full_multiseed_needed | n4_can_proceed | diagnostics_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unstable_or_collapsed | 0.5493 | 0.4507 | 0.6033 | 0.3967 | 0.5667 | 0.4333 | 0.0000 | 0.0000 | 0.0000 | no | no | 1.0000 |

## Training-Seed Summary

| checkpoint_label | training_seed_count | mean_success_rate | std_success_rate_across_training_seeds | min_success_rate | mean_collision_rate | std_collision_rate_across_training_seeds | max_collision_rate | collapse_warning_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000k | 3.0000 | 0.5404 | 0.1096 | 0.4211 | 0.4596 | 0.1096 | 0.5789 | 1.0000 |
| 1500k | 3.0000 | 0.5493 | 0.1101 | 0.4222 | 0.4507 | 0.1101 | 0.5778 | 1.0000 |
| final | 3.0000 | 0.5589 | 0.0587 | 0.4911 | 0.4411 | 0.0587 | 0.5089 | 1.0000 |

## Aggregate

| training_seed | method_key | checkpoint_label | num_eval_seeds | num_episodes_total | mean_success_rate | mean_collision_rate | mean_near_miss_rate | mean_raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.0000 | no_z_full | final | 3.0000 | 900.0000 | 0.5667 | 0.4333 | 0.5089 | 0.2383 |
| -1.0000 | attention_full | attention_full_1500k | 3.0000 | 900.0000 | 0.6033 | 0.3967 | 0.5700 | 0.2901 |
| 0.0000 | p3_s0_1000k | 1000k | 3.0000 | 900.0000 | 0.6367 | 0.3633 | 0.5422 | 0.2278 |
| 0.0000 | p3_s0_1500k | 1500k | 3.0000 | 900.0000 | 0.6167 | 0.3833 | 0.5933 | 0.3138 |
| 0.0000 | p3_s0_final | final | 3.0000 | 900.0000 | 0.5933 | 0.4067 | 0.5633 | 0.3092 |
| 1.0000 | p3_s1_1000k | 1000k | 3.0000 | 900.0000 | 0.5633 | 0.4367 | 0.5311 | 0.3004 |
| 1.0000 | p3_s1_1500k | 1500k | 3.0000 | 900.0000 | 0.6089 | 0.3911 | 0.5744 | 0.3303 |
| 1.0000 | p3_s1_final | final | 3.0000 | 900.0000 | 0.5922 | 0.4078 | 0.5622 | 0.3471 |
| 2.0000 | p3_s2_1000k | 1000k | 3.0000 | 900.0000 | 0.4211 | 0.5789 | 0.4078 | 0.2451 |
| 2.0000 | p3_s2_1500k | 1500k | 3.0000 | 900.0000 | 0.4222 | 0.5778 | 0.3989 | 0.2688 |
| 2.0000 | p3_s2_final | final | 3.0000 | 900.0000 | 0.4911 | 0.5089 | 0.4611 | 0.2910 |

## Pairwise Vs Attention

| training_seed | method_key | checkpoint_label | success_diff_vs_attention | collision_diff_vs_attention | better_both |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | p3_s0_1000k | 1000k | 0.0333 | -0.0333 | 1.0000 |
| 0.0000 | p3_s0_1500k | 1500k | 0.0133 | -0.0133 | 1.0000 |
| 0.0000 | p3_s0_final | final | -0.0100 | 0.0100 | 0.0000 |
| 1.0000 | p3_s1_1000k | 1000k | -0.0400 | 0.0400 | 0.0000 |
| 1.0000 | p3_s1_1500k | 1500k | 0.0056 | -0.0056 | 1.0000 |
| 1.0000 | p3_s1_final | final | -0.0111 | 0.0111 | 0.0000 |
| 2.0000 | p3_s2_1000k | 1000k | -0.1822 | 0.1822 | 0.0000 |
| 2.0000 | p3_s2_1500k | 1500k | -0.1811 | 0.1811 | 0.0000 |
| 2.0000 | p3_s2_final | final | -0.1122 | 0.1122 | 0.0000 |

## Scenario/Motion/Threat Summary

| training_seed | method_key | eval_seed | scenario | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_high_density | 0.6000 | 0.4000 |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_high_speed | 0.4000 | 0.6000 |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_high_threat | 0.6200 | 0.3800 |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_id | 0.6800 | 0.3200 |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_mixed_ood | 0.6200 | 0.3800 |
| 1.0000 | p3_s1_1500k | 1000.0000 | eval_flow_sudden_threat | 0.6800 | 0.3200 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_high_density | 0.6200 | 0.3800 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_high_speed | 0.4000 | 0.6000 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_high_threat | 0.6400 | 0.3600 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_id | 0.6800 | 0.3200 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_mixed_ood | 0.6400 | 0.3600 |
| 1.0000 | p3_s1_1500k | 1001.0000 | eval_flow_sudden_threat | 0.6800 | 0.3200 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_high_density | 0.6200 | 0.3800 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_high_speed | 0.4000 | 0.6000 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_high_threat | 0.6600 | 0.3400 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_id | 0.6800 | 0.3200 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_mixed_ood | 0.6600 | 0.3400 |
| 1.0000 | p3_s1_1500k | 1002.0000 | eval_flow_sudden_threat | 0.6800 | 0.3200 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_high_density | 0.4000 | 0.6000 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_high_threat | 0.4800 | 0.5200 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_id | 0.5200 | 0.4800 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_mixed_ood | 0.3400 | 0.6600 |
| 2.0000 | p3_s2_1500k | 1000.0000 | eval_flow_sudden_threat | 0.4600 | 0.5400 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_high_density | 0.3800 | 0.6200 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_high_threat | 0.4600 | 0.5400 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_id | 0.5200 | 0.4800 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_mixed_ood | 0.3200 | 0.6800 |
| 2.0000 | p3_s2_1500k | 1001.0000 | eval_flow_sudden_threat | 0.4400 | 0.5600 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_high_density | 0.3800 | 0.6200 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_high_threat | 0.4600 | 0.5400 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_id | 0.5400 | 0.4600 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_mixed_ood | 0.3200 | 0.6800 |
| 2.0000 | p3_s2_1500k | 1002.0000 | eval_flow_sudden_threat | 0.4400 | 0.5600 |
| training_seed | method_key | eval_seed | threat_motion_mode | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | p3_s1_1500k | 1000.0000 | accel_decel | 0.5263 | 0.4737 |
| 1.0000 | p3_s1_1500k | 1000.0000 | ar1_velocity | 0.7067 | 0.2933 |
| 1.0000 | p3_s1_1500k | 1000.0000 | crossing_or_sudden_threat | 0.6364 | 0.3636 |
| 1.0000 | p3_s1_1500k | 1000.0000 | linear | 0.5818 | 0.4182 |
| 1.0000 | p3_s1_1500k | 1000.0000 | sinusoidal_lateral | 0.5500 | 0.4500 |
| 1.0000 | p3_s1_1500k | 1001.0000 | accel_decel | 0.5345 | 0.4655 |
| 1.0000 | p3_s1_1500k | 1001.0000 | ar1_velocity | 0.7200 | 0.2800 |
| 1.0000 | p3_s1_1500k | 1001.0000 | crossing_or_sudden_threat | 0.6364 | 0.3636 |
| 1.0000 | p3_s1_1500k | 1001.0000 | linear | 0.5926 | 0.4074 |
| 1.0000 | p3_s1_1500k | 1001.0000 | sinusoidal_lateral | 0.5625 | 0.4375 |
| 1.0000 | p3_s1_1500k | 1002.0000 | accel_decel | 0.5424 | 0.4576 |
| 1.0000 | p3_s1_1500k | 1002.0000 | ar1_velocity | 0.7200 | 0.2800 |
| 1.0000 | p3_s1_1500k | 1002.0000 | crossing_or_sudden_threat | 0.6000 | 0.4000 |
| 1.0000 | p3_s1_1500k | 1002.0000 | linear | 0.6111 | 0.3889 |
| 1.0000 | p3_s1_1500k | 1002.0000 | sinusoidal_lateral | 0.5844 | 0.4156 |
| 2.0000 | p3_s2_1500k | 1000.0000 | accel_decel | 0.4364 | 0.5636 |
| 2.0000 | p3_s2_1500k | 1000.0000 | ar1_velocity | 0.4205 | 0.5795 |
| 2.0000 | p3_s2_1500k | 1000.0000 | crossing_or_sudden_threat | 0.4490 | 0.5510 |
| training_seed | method_key | eval_seed | threat_class | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | p3_s1_1500k | 1000.0000 | high | 0.5862 | 0.4138 |
| 1.0000 | p3_s1_1500k | 1000.0000 | medium | 1.0000 | 0.0000 |
| 1.0000 | p3_s1_1500k | 1001.0000 | high | 0.5966 | 0.4034 |
| 1.0000 | p3_s1_1500k | 1001.0000 | medium | 1.0000 | 0.0000 |
| 1.0000 | p3_s1_1500k | 1002.0000 | high | 0.6034 | 0.3966 |
| 1.0000 | p3_s1_1500k | 1002.0000 | medium | 1.0000 | 0.0000 |
| 2.0000 | p3_s2_1500k | 1000.0000 | high | 0.4175 | 0.5825 |
| 2.0000 | p3_s2_1500k | 1000.0000 | low | 1.0000 | 0.0000 |
| 2.0000 | p3_s2_1500k | 1000.0000 | medium | 0.6429 | 0.3571 |
| 2.0000 | p3_s2_1500k | 1001.0000 | high | 0.4035 | 0.5965 |
| 2.0000 | p3_s2_1500k | 1001.0000 | low | 1.0000 | 0.0000 |
| 2.0000 | p3_s2_1500k | 1001.0000 | medium | 0.6429 | 0.3571 |
| 2.0000 | p3_s2_1500k | 1002.0000 | high | 0.4049 | 0.5951 |
| 2.0000 | p3_s2_1500k | 1002.0000 | low | 1.0000 | 0.0000 |
| 2.0000 | p3_s2_1500k | 1002.0000 | medium | 0.6667 | 0.3333 |
| 0.0000 | p3_s0_1500k | 1000.0000 | high | 0.6181 | 0.3819 |
| 0.0000 | p3_s0_1500k | 1000.0000 | medium | 0.6667 | 0.3333 |
| 0.0000 | p3_s0_1500k | 1001.0000 | high | 0.6098 | 0.3902 |

## Diagnostics

| diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | adapter_output_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- |
| 1.0000 | 2.1573 | 2.9422 | 6.3131 | 0.0000 |

## Artifacts

### tables
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_checkpoint_integrity.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_command_manifest.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_config_manifest.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_decision.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_diagnostics_decision.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_episode_metrics.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_pairwise_vs_attention.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_resource_affinity.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_schema_check.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_threat_class_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_train_curve.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_train_heartbeat.csv`
- `results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_training_seed_summary.csv`
### plots
- `results/env_v2_phase_n3pf_ms_multiseed/plots/n3pf_ms_checkpoint_success_collision.png`
- `results/env_v2_phase_n3pf_ms_multiseed/plots/n3pf_ms_scenario_breakdown.png`
- `results/env_v2_phase_n3pf_ms_multiseed/plots/n3pf_ms_training_seed_stability.png`
### logs
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_analysis.log`
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_eval.log`
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_train_s1.log`
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_train_s2.log`
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_validate_s1.log`
- `results/env_v2_phase_n3pf_ms_multiseed/logs/phase_n3pf_ms_validate_s2.log`
- `results/env_v2_phase_n3pf_ms_multiseed/phase_n3pf_ms_watcher.log`
### flags
- none
