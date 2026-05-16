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
| attention_no_shield | attention_full | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.9565 |
| noz_no_shield | no_z_full | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9808 |
| p3_no_shield | p3_1500k | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9908 |
| attention_ordinary_vo | attention_full | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| noz_ordinary_vo | no_z_full | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9837 |
| p3_ordinary_vo | p3_1500k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9890 |

## No-Shield References

| method_key | base_policy | mean_success_rate | mean_collision_rate | mean_near_miss_rate | mean_progress |
| --- | --- | --- | --- | --- | --- |
| attention_no_shield | attention_full | 0.0000 | 1.0000 | 0.0000 | 0.9565 |
| noz_no_shield | no_z_full | 1.0000 | 0.0000 | 1.0000 | 0.9808 |
| p3_no_shield | p3_1500k | 1.0000 | 0.0000 | 1.0000 | 0.9908 |

## Same-Shield Comparison

| comparison | p3_success | ref_success | success_diff | p3_collision | ref_collision | collision_diff | p3_better_both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p3_ordinary_vo_vs_attention_ordinary_vo | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| p3_ordinary_vo_vs_noz_ordinary_vo | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Phase B Context

| context_config | baseline_name | success_rate | collision_rate | near_miss_rate | progress | filter_trigger_rate | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vo_like_filter_h45_cpa1p2_h16 | vo_like_filter | 0.8333 | 0.1667 | 0.7200 | 0.9880 | 0.3509 | Phase B formal B2 |
| cpa_ttc_weighted_apf_alpha3 | cpa_ttc_weighted_apf | 0.7200 | 0.2800 | 0.7200 | 0.8964 | 0.0000 | Phase B formal B2 |

## Intervention Metrics

| method_key | shield_enabled | shield_trigger_rate | episode_filter_triggered_rate | filter_delta_norm_mean | filter_delta_norm_p95 | raw_action_unsafe_rate | filtered_action_unsafe_rate | raw_min_predicted_cpa | filtered_min_predicted_cpa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4314 | 0.4314 | 3.2198 | 3.2198 |
| attention_ordinary_vo | 1.0000 | 0.2778 | 1.0000 | 0.1772 | 0.7577 | 0.2685 | 0.0000 | 3.3938 | 4.2302 |
| noz_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2609 | 0.2609 | 3.1417 | 3.1417 |
| noz_ordinary_vo | 1.0000 | 0.2216 | 1.0000 | 0.1823 | 1.0229 | 0.2216 | 0.0000 | 3.2878 | 3.6008 |
| p3_no_shield | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2946 | 0.2946 | 3.0741 | 3.0741 |
| p3_ordinary_vo | 1.0000 | 0.1864 | 1.0000 | 0.1140 | 0.8546 | 0.1780 | 0.0000 | 3.3062 | 3.7301 |

## Scenario/Motion/Threat Summary

| method_key | eval_seed | scenario | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | eval_flow_id | 1.0000 | 0.0000 |
| noz_ordinary_vo | 1000.0000 | eval_flow_id | 1.0000 | 0.0000 |
| p3_ordinary_vo | 1000.0000 | eval_flow_id | 1.0000 | 0.0000 |
| method_key | eval_seed | threat_motion_mode | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | ar1_velocity | 1.0000 | 0.0000 |
| noz_ordinary_vo | 1000.0000 | ar1_velocity | 1.0000 | 0.0000 |
| p3_ordinary_vo | 1000.0000 | sinusoidal_lateral | 1.0000 | 0.0000 |
| method_key | eval_seed | threat_class | success_rate | collision_rate |
| --- | --- | --- | --- | --- |
| attention_ordinary_vo | 1000.0000 | high | 1.0000 | 0.0000 |
| noz_ordinary_vo | 1000.0000 | high | 1.0000 | 0.0000 |
| p3_ordinary_vo | 1000.0000 | high | 1.0000 | 0.0000 |

## Artifacts

### tables
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_checkpoint_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_command_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_decision.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_episode_metrics.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_eval_summary_aggregate.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_eval_summary_by_seed.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_failure_cases.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_feature_block_stats.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_gpsi_output_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_intervention_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_motion_mode_breakdown.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_no_shield_reference_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_phase_b_context_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_raw_vs_filtered_safety.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_raw_vs_filtered_safety_steps.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_same_shield_comparison.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_scenario_breakdown.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_schema_check.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_shield_config_manifest.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_shield_gain_summary.csv`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/tables/phase_n4o_threat_class_breakdown.csv`
### plots
- `results/env_v2_phase_n4o_ordinary_shield_smoke/plots/n4o_scenario_breakdown.png`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/plots/n4o_shield_gain_collision_reduction.png`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/plots/n4o_shield_trigger_rate.png`
- `results/env_v2_phase_n4o_ordinary_shield_smoke/plots/n4o_success_collision.png`
### logs
- `results/env_v2_phase_n4o_ordinary_shield_smoke/phase_n4o_watcher.log`
### flags
- `results/env_v2_phase_n4o_ordinary_shield_smoke/PHASE_N4O_STOP_EVAL_FAILED.flag`
