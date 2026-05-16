# Phase N4-O Ordinary Shield Fair Comparison Report

## Terminal Decision

`terminal_decision = phase_n4o_ordinary_shield_complete`

N4-O completed eval-only ordinary VO-like shield fair comparison. No PPO or Gpsi training was run.

## Decision

| p3_plus_shield_beats_attention_plus_shield | p3_plus_shield_beats_noz_plus_shield | ordinary_shield_gains_mostly_from_shield | intervention_diagnostics_clean | can_proceed_to_n4u | decision |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 1.0000 | mixed | 1.0000 | yes | P3 + ordinary shield beats attention + same ordinary shield with clean intervention diagnostics. |

## Aggregate

| method_key | base_policy | shield_enabled | num_eval_seeds | num_episodes_total | mean_success_rate | mean_collision_rate | mean_near_miss_rate | mean_progress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_no_shield | attention_full | 0.0000 | 3.0000 | 900.0000 | 0.6033 | 0.3967 | 0.5700 | 0.9571 |
| noz_no_shield | no_z_full | 0.0000 | 3.0000 | 900.0000 | 0.5667 | 0.4333 | 0.5089 | 0.9489 |
| p3_no_shield | p3_1500k | 0.0000 | 3.0000 | 900.0000 | 0.6167 | 0.3833 | 0.5933 | 0.9541 |
| attention_ordinary_vo | attention_full | 1.0000 | 3.0000 | 900.0000 | 0.8300 | 0.1700 | 0.7189 | 0.9880 |
| noz_ordinary_vo | no_z_full | 1.0000 | 3.0000 | 900.0000 | 0.8211 | 0.1756 | 0.6922 | 0.9870 |
| p3_ordinary_vo | p3_1500k | 1.0000 | 3.0000 | 900.0000 | 0.8811 | 0.1156 | 0.7722 | 0.9849 |

## No-Shield References

| method_key | base_policy | mean_success_rate | mean_collision_rate | mean_near_miss_rate | mean_progress |
| --- | --- | --- | --- | --- | --- |
| attention_no_shield | attention_full | 0.6033 | 0.3967 | 0.5700 | 0.9571 |
| noz_no_shield | no_z_full | 0.5667 | 0.4333 | 0.5089 | 0.9489 |
| p3_no_shield | p3_1500k | 0.6167 | 0.3833 | 0.5933 | 0.9541 |

## Same-Shield Comparison

| comparison | p3_success | ref_success | success_diff | p3_collision | ref_collision | collision_diff | p3_better_both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p3_ordinary_vo_vs_attention_ordinary_vo | 0.8811 | 0.8300 | 0.0511 | 0.1156 | 0.1700 | -0.0544 | 1.0000 |
| p3_ordinary_vo_vs_noz_ordinary_vo | 0.8811 | 0.8211 | 0.0600 | 0.1156 | 0.1756 | -0.0600 | 1.0000 |

## Phase B Context

| context_config | baseline_name | success_rate | collision_rate | near_miss_rate | progress | filter_trigger_rate | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vo_like_filter_h45_cpa1p2_h16 | vo_like_filter | 0.8333 | 0.1667 | 0.7200 | 0.9880 | 0.3509 | Phase B formal B2 |
| cpa_ttc_weighted_apf_alpha3 | cpa_ttc_weighted_apf | 0.7200 | 0.2800 | 0.7200 | 0.8964 | 0.0000 | Phase B formal B2 |

## Intervention Metrics

| method_key | shield_enabled | shield_trigger_rate | episode_filter_triggered_rate | filter_delta_norm_mean | filter_delta_norm_p95 | raw_action_unsafe_rate | filtered_action_unsafe_rate | raw_min_predicted_cpa | filtered_min_predicted_cpa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2901 | 0.2901 | 3.0044 | 3.0044 |
| attention_ordinary_vo | 1.0000 | 0.3508 | 1.0000 | 0.2409 | 0.9780 | 0.3356 | 0.0065 | 2.6352 | 3.2034 |
| noz_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2383 | 0.2383 | 3.5747 | 3.5747 |
| noz_ordinary_vo | 1.0000 | 0.3084 | 1.0000 | 0.2327 | 1.0342 | 0.2999 | 0.0071 | 3.2064 | 3.7538 |
| p3_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3138 | 0.3138 | 2.6530 | 2.6530 |
| p3_ordinary_vo | 1.0000 | 0.3827 | 1.0000 | 0.2552 | 0.9961 | 0.3664 | 0.0048 | 2.3578 | 3.0066 |

## Scenario/Motion/Threat Summary

| method_key | eval_seed | scenario | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | eval_flow_high_density | 0.8400 | 0.1600 |
| attention_ordinary_vo | 1000.0000 | eval_flow_high_speed | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1000.0000 | eval_flow_high_threat | 0.8000 | 0.2000 |
| attention_ordinary_vo | 1000.0000 | eval_flow_id | 0.9400 | 0.0600 |
| attention_ordinary_vo | 1000.0000 | eval_flow_mixed_ood | 0.7800 | 0.2200 |
| attention_ordinary_vo | 1000.0000 | eval_flow_sudden_threat | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1001.0000 | eval_flow_high_density | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1001.0000 | eval_flow_high_speed | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1001.0000 | eval_flow_high_threat | 0.8000 | 0.2000 |
| attention_ordinary_vo | 1001.0000 | eval_flow_id | 0.9400 | 0.0600 |
| attention_ordinary_vo | 1001.0000 | eval_flow_mixed_ood | 0.7600 | 0.2400 |
| attention_ordinary_vo | 1001.0000 | eval_flow_sudden_threat | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1002.0000 | eval_flow_high_density | 0.8200 | 0.1800 |
| attention_ordinary_vo | 1002.0000 | eval_flow_high_speed | 0.8400 | 0.1600 |
| attention_ordinary_vo | 1002.0000 | eval_flow_high_threat | 0.8000 | 0.2000 |
| attention_ordinary_vo | 1002.0000 | eval_flow_id | 0.9400 | 0.0600 |
| attention_ordinary_vo | 1002.0000 | eval_flow_mixed_ood | 0.7600 | 0.2400 |
| attention_ordinary_vo | 1002.0000 | eval_flow_sudden_threat | 0.8200 | 0.1800 |
| noz_ordinary_vo | 1000.0000 | eval_flow_high_density | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1000.0000 | eval_flow_high_speed | 0.9000 | 0.1000 |
| noz_ordinary_vo | 1000.0000 | eval_flow_high_threat | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1000.0000 | eval_flow_id | 0.8800 | 0.1200 |
| noz_ordinary_vo | 1000.0000 | eval_flow_mixed_ood | 0.8200 | 0.1800 |
| noz_ordinary_vo | 1000.0000 | eval_flow_sudden_threat | 0.7800 | 0.2000 |
| noz_ordinary_vo | 1001.0000 | eval_flow_high_density | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1001.0000 | eval_flow_high_speed | 0.9000 | 0.1000 |
| noz_ordinary_vo | 1001.0000 | eval_flow_high_threat | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1001.0000 | eval_flow_id | 0.8600 | 0.1400 |
| noz_ordinary_vo | 1001.0000 | eval_flow_mixed_ood | 0.8200 | 0.1800 |
| noz_ordinary_vo | 1001.0000 | eval_flow_sudden_threat | 0.7800 | 0.2000 |
| noz_ordinary_vo | 1002.0000 | eval_flow_high_density | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1002.0000 | eval_flow_high_speed | 0.9200 | 0.0800 |
| noz_ordinary_vo | 1002.0000 | eval_flow_high_threat | 0.7800 | 0.2200 |
| noz_ordinary_vo | 1002.0000 | eval_flow_id | 0.8600 | 0.1400 |
| noz_ordinary_vo | 1002.0000 | eval_flow_mixed_ood | 0.8000 | 0.2000 |
| noz_ordinary_vo | 1002.0000 | eval_flow_sudden_threat | 0.7800 | 0.2000 |
| method_key | eval_seed | threat_motion_mode | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | accel_decel | 0.8769 | 0.1231 |
| attention_ordinary_vo | 1000.0000 | ar1_velocity | 0.8451 | 0.1549 |
| attention_ordinary_vo | 1000.0000 | crossing_or_sudden_threat | 0.7692 | 0.2308 |
| attention_ordinary_vo | 1000.0000 | linear | 0.8571 | 0.1429 |
| attention_ordinary_vo | 1000.0000 | sinusoidal_lateral | 0.8182 | 0.1818 |
| attention_ordinary_vo | 1001.0000 | accel_decel | 0.8657 | 0.1343 |
| attention_ordinary_vo | 1001.0000 | ar1_velocity | 0.8406 | 0.1594 |
| attention_ordinary_vo | 1001.0000 | crossing_or_sudden_threat | 0.7692 | 0.2308 |
| attention_ordinary_vo | 1001.0000 | linear | 0.8611 | 0.1389 |
| attention_ordinary_vo | 1001.0000 | sinusoidal_lateral | 0.8026 | 0.1974 |
| attention_ordinary_vo | 1002.0000 | accel_decel | 0.8657 | 0.1343 |
| attention_ordinary_vo | 1002.0000 | ar1_velocity | 0.8406 | 0.1594 |
| attention_ordinary_vo | 1002.0000 | crossing_or_sudden_threat | 0.7736 | 0.2264 |
| attention_ordinary_vo | 1002.0000 | linear | 0.8889 | 0.1111 |
| attention_ordinary_vo | 1002.0000 | sinusoidal_lateral | 0.8000 | 0.2000 |
| noz_ordinary_vo | 1000.0000 | accel_decel | 0.8209 | 0.1791 |
| noz_ordinary_vo | 1000.0000 | ar1_velocity | 0.8507 | 0.1493 |
| noz_ordinary_vo | 1000.0000 | crossing_or_sudden_threat | 0.8281 | 0.1562 |
| method_key | eval_seed | threat_class | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | high | 0.8476 | 0.1524 |
| attention_ordinary_vo | 1000.0000 | medium | 0.7097 | 0.2903 |
| attention_ordinary_vo | 1001.0000 | high | 0.8401 | 0.1599 |
| attention_ordinary_vo | 1001.0000 | medium | 0.7097 | 0.2903 |
| attention_ordinary_vo | 1002.0000 | high | 0.8433 | 0.1567 |
| attention_ordinary_vo | 1002.0000 | medium | 0.7188 | 0.2812 |
| noz_ordinary_vo | 1000.0000 | high | 0.8216 | 0.1747 |
| noz_ordinary_vo | 1000.0000 | medium | 0.8387 | 0.1613 |
| noz_ordinary_vo | 1001.0000 | high | 0.8202 | 0.1760 |
| noz_ordinary_vo | 1001.0000 | medium | 0.8182 | 0.1818 |
| noz_ordinary_vo | 1002.0000 | high | 0.8233 | 0.1729 |
| noz_ordinary_vo | 1002.0000 | medium | 0.7941 | 0.2059 |
| p3_ordinary_vo | 1000.0000 | high | 0.8800 | 0.1164 |
| p3_ordinary_vo | 1000.0000 | medium | 0.8800 | 0.1200 |
| p3_ordinary_vo | 1001.0000 | high | 0.8800 | 0.1164 |
| p3_ordinary_vo | 1001.0000 | medium | 0.8800 | 0.1200 |
| p3_ordinary_vo | 1002.0000 | high | 0.8841 | 0.1123 |
| p3_ordinary_vo | 1002.0000 | medium | 0.8750 | 0.1250 |

## Artifacts

### tables
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_checkpoint_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_command_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_decision.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_episode_metrics.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_eval_summary_aggregate.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_eval_summary_by_seed.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_failure_cases.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_feature_block_stats.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_gpsi_output_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_intervention_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_motion_mode_breakdown.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_no_shield_reference_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_phase_b_context_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_raw_vs_filtered_safety.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_raw_vs_filtered_safety_steps.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_same_shield_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_scenario_breakdown.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_schema_check.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_shield_config_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_shield_gain_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_threat_class_breakdown.csv`
### plots
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/plots/n4o_scenario_breakdown.png`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/plots/n4o_shield_gain_collision_reduction.png`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/plots/n4o_shield_trigger_rate.png`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/plots/n4o_success_collision.png`
### logs
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/logs/phase_n4o_analysis.log`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/logs/phase_n4o_eval.log`
- `results/env_v2_phase_n4o_ordinary_shield_fair_comparison/phase_n4o_watcher.log`
### flags
- none
