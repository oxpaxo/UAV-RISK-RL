# Phase N3PF-MS-AB Seed2 Collapse Audit + Minimal Rerun Sanity

## Terminal Decision

`terminal_decision = phase_n3pf_ms_seed2_ab_audit_rerun_complete`

## Step A Audit Summary

| decision_type | hard_error_found | do_step_b | stop_after_step_a | seed2_success_1500k | seed2_collision_1500k | config_mismatch_detail | eval_path_mismatch_detail | checkpoint_integrity_failed_detail | feature_gpsi_diagnostics_failed_detail | most_likely_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_bad_local_optimum_likely | 0.0000 | 1.0000 | 0.0000 | 0.4222 | 0.5778 | nan | nan | nan | nan | PPO seed-sensitive bad local optimum or training instability |

No hard config/path/checkpoint/Gpsi feature error was allowed past Step A. Config equivalence and checkpoint integrity details are below.

| field | seed1_value | seed2_value | same | allowed_difference |
| --- | --- | --- | --- | --- |
| policy_class | MultiInputPolicy | MultiInputPolicy | 1.0000 | 0.0000 |
| feature_adapter | block_projected_no_z | block_projected_no_z | 1.0000 | 0.0000 |
| include_z | False | False | 1.0000 | 0.0000 |
| delta_scale | 5 | 5 | 1.0000 | 0.0000 |
| logvar_clip | [-5.0,3.0] | [-5.0,3.0] | 1.0000 | 0.0000 |
| logvar_scale | 0.2 | 0.2 | 1.0000 | 0.0000 |
| n_envs | 4 | 4 | 1.0000 | 0.0000 |
| device | cpu | cpu | 1.0000 | 0.0000 |
| learning_rate | 0.0003 | 0.0003 | 1.0000 | 0.0000 |
| n_steps | 1024 | 1024 | 1.0000 | 0.0000 |
| batch_size | 256 | 256 | 1.0000 | 0.0000 |
| gamma | 0.99 | 0.99 | 1.0000 | 0.0000 |
| gae_lambda | 0.95 | 0.95 | 1.0000 | 0.0000 |
| clip_range | 0.2 | 0.2 | 1.0000 | 0.0000 |
| ent_coef | 0.01 | 0.01 | 1.0000 | 0.0000 |
| vf_coef | 0.5 | 0.5 | 1.0000 | 0.0000 |
| max_grad_norm | 0.5 | 0.5 | 1.0000 | 0.0000 |
| gpsi_checkpoint | work_dirs/gpsi_heada_v1_nll/best.pth | work_dirs/gpsi_heada_v1_nll/best.pth | 1.0000 | 0.0000 |
| gpsi_frozen | true | true | 1.0000 | 0.0000 |
| train_scenario | train_flow_mixed | train_flow_mixed | 1.0000 | 0.0000 |
| no_shield | True | True | 1.0000 | 0.0000 |
| action_filtering | False | False | 1.0000 | 0.0000 |
| use_safety_cost | False | False | 1.0000 | 0.0000 |
| obs_aug_dim | 30 | 30 | 1.0000 | 0.0000 |
| block_projector | {"activation":"tanh","delta_block_dim":9,"delta_project_dim":16,"logvar_block_dim":9,"logvar_project_dim":16,"obs_block_dim":12,"obs_project_dim":32} | {"activation":"tanh","delta_block_dim":9,"delta_project_dim":16,"logvar_block_dim":9,"logvar_project_dim":16,"obs_block_dim":12,"obs_project_dim":32} | 1.0000 | 0.0000 |
| training.seed | allowed | allowed | 0.0000 | 1.0000 |
| method_name | allowed | allowed | 0.0000 | 1.0000 |
| out_dir | allowed | allowed | 0.0000 | 1.0000 |

## Checkpoint Integrity

| training_seed | checkpoint_label | exists | model_num_timesteps | optimizer_state_present | policy_parameter_delta_vs_previous_checkpoint | eval_seed_count | scenario_count | episode_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 250k | 1.0000 | 250000.0000 | 1.0000 | nan | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | 500k | 1.0000 | 500000.0000 | 1.0000 | 13.0696 | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | 750k | 1.0000 | 750000.0000 | 1.0000 | 12.7888 | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | 1000k | 1.0000 | 1000000.0000 | 1.0000 | 12.5671 | 3.0000 | 6.0000 | 900.0000 |
| 1.0000 | 1250k | 1.0000 | 1250000.0000 | 1.0000 | 12.3818 | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | 1500k | 1.0000 | 1500000.0000 | 1.0000 | 12.4978 | 3.0000 | 6.0000 | 900.0000 |
| 1.0000 | final | 1.0000 | 1503232.0000 | 1.0000 | 2.0051 | 3.0000 | 6.0000 | 900.0000 |
| 2.0000 | 250k | 1.0000 | 250000.0000 | 1.0000 | nan | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 500k | 1.0000 | 500000.0000 | 1.0000 | 14.4577 | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 750k | 1.0000 | 750000.0000 | 1.0000 | 13.1838 | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 1000k | 1.0000 | 1000000.0000 | 1.0000 | 12.8459 | 3.0000 | 6.0000 | 900.0000 |
| 2.0000 | 1250k | 1.0000 | 1250000.0000 | 1.0000 | 12.8363 | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 1500k | 1.0000 | 1500000.0000 | 1.0000 | 12.6373 | 3.0000 | 6.0000 | 900.0000 |
| 2.0000 | final | 1.0000 | 1503232.0000 | 1.0000 | 2.1594 | 3.0000 | 6.0000 | 900.0000 |

## Intermediate Eval

| run_name | training_seed | method_key | checkpoint_label | num_eval_seeds | num_episodes_total | mean_success_rate | mean_collision_rate | mean_raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_seed1 | 1.0000 | p3_s1_1250k | 1250k | 2.0000 | 600.0000 | 0.5083 | 0.4917 | 0.2934 |
| original_seed1 | 1.0000 | p3_s1_250k | 250k | 2.0000 | 600.0000 | 0.3850 | 0.6150 | 0.2535 |
| original_seed1 | 1.0000 | p3_s1_500k | 500k | 2.0000 | 600.0000 | 0.3733 | 0.6267 | 0.2652 |
| original_seed1 | 1.0000 | p3_s1_750k | 750k | 2.0000 | 600.0000 | 0.5300 | 0.4700 | 0.2467 |
| original_seed2 | 2.0000 | p3_s2_1250k | 1250k | 2.0000 | 600.0000 | 0.5050 | 0.4950 | 0.2420 |
| original_seed2 | 2.0000 | p3_s2_250k | 250k | 2.0000 | 600.0000 | 0.3950 | 0.6050 | 0.3028 |
| original_seed2 | 2.0000 | p3_s2_500k | 500k | 2.0000 | 600.0000 | 0.4700 | 0.5300 | 0.2373 |
| original_seed2 | 2.0000 | p3_s2_750k | 750k | 2.0000 | 600.0000 | 0.3583 | 0.6417 | 0.2345 |

## Step B Rerun Result

| run_name | training_seed | method_key | checkpoint_label | num_eval_seeds | num_episodes_total | mean_success_rate | mean_collision_rate | mean_raw_unsafe_action_rate | mean_action_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed2_rerunA | 2.0000 | p3_s2_seed2_rerunA_1000k | 1000k | 3.0000 | 900.0000 | 0.4211 | 0.5789 | 0.2451 | 0.1041 |
| seed2_rerunA | 2.0000 | p3_s2_seed2_rerunA_1500k | 1500k | 3.0000 | 900.0000 | 0.4222 | 0.5778 | 0.2688 | 0.1104 |
| seed2_rerunA | 2.0000 | p3_s2_seed2_rerunA_final | final | 3.0000 | 900.0000 | 0.4911 | 0.5089 | 0.2910 | 0.1172 |

## Decision

| step_a_decision_type | step_a_hard_error_found | step_b_executed | seed2_rerunA_success_1500k | seed2_rerunA_collision_1500k | seed2_rerunA_recovered | seed3_sanity_executed | seed3_sanity_success_1500k | seed3_sanity_collision_1500k | seed3_sanity_healthy | collapse_most_likely_cause | allow_restore_p3_primary_n4_candidate | can_continue_N4U | needs_p3_stabilization_phase | attention_full_multiseed_needed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_bad_local_optimum_likely | 0.0000 | 1.0000 | 0.4222 | 0.5778 | 0.0000 | 0.0000 | nan | nan | 0.0000 | collapse_reproducible_or_inconclusive_architecture_instability | no | no | required | yes_before_decisive_attention_claim |

Interpretation:
- engineering_error_found: 0
- seed2 collapse most likely cause: collapse_reproducible_or_inconclusive_architecture_instability
- seed2_rerunA recovered: 0
- seed3_sanity status: not_run
- restore P3 primary N4 candidate: no
- can continue N4-U: no
- P3 stabilization phase: required
- attention_full multi-seed: yes_before_decisive_attention_claim

## Behavior Diagnostics

| policy | checkpoint_label | success_rate | collision_rate | raw_unsafe_action_rate | action_norm | action_delta | progress | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_seed0_original_1500k | nan | 0.6167 | 0.3833 | nan | nan | nan | nan | reported_baseline |
| p3_seed1_original_1500k | nan | 0.6089 | 0.3911 | nan | nan | nan | nan | reported_baseline |
| p3_seed2_original_1500k | nan | 0.4222 | 0.5778 | nan | nan | nan | nan | reported_baseline |
| attention_full_reference | nan | 0.6033 | 0.3967 | nan | nan | nan | nan | reported_baseline |
| no_z_full_reference | nan | 0.5667 | 0.4333 | nan | nan | nan | nan | reported_baseline |
| p3_s2_1000k | 1000k | 0.4211 | 0.5789 | 0.2451 | 1.5809 | 0.1041 | 0.9607 | N3PF-MS-original |
| p3_s2_1500k | 1500k | 0.4222 | 0.5778 | 0.2688 | 1.5538 | 0.1104 | 0.9526 | N3PF-MS-original |
| p3_s2_final | final | 0.4911 | 0.5089 | 0.2910 | 1.5561 | 0.1172 | 0.9460 | N3PF-MS-original |
| p3_s2_seed2_rerunA_1000k | 1000k | 0.4211 | 0.5789 | 0.2451 | 1.5809 | 0.1041 | 0.9607 | N3PF-MS-AB-rerun |
| p3_s2_seed2_rerunA_1500k | 1500k | 0.4222 | 0.5778 | 0.2688 | 1.5538 | 0.1104 | 0.9526 | N3PF-MS-AB-rerun |
| p3_s2_seed2_rerunA_final | final | 0.4911 | 0.5089 | 0.2910 | 1.5561 | 0.1172 | 0.9460 | N3PF-MS-AB-rerun |

## Scenario/Motion/Threat Summary

| run_name | method_key | eval_seed | scenario | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_high_density | 0.4000 | 0.6000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_high_threat | 0.4800 | 0.5200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_id | 0.5200 | 0.4800 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_mixed_ood | 0.3400 | 0.6600 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | eval_flow_sudden_threat | 0.4600 | 0.5400 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_high_density | 0.3800 | 0.6200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_high_threat | 0.4600 | 0.5400 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_id | 0.5200 | 0.4800 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_mixed_ood | 0.3200 | 0.6800 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | eval_flow_sudden_threat | 0.4400 | 0.5600 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_high_density | 0.3800 | 0.6200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_high_speed | 0.3800 | 0.6200 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_high_threat | 0.4600 | 0.5400 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_id | 0.5400 | 0.4600 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_mixed_ood | 0.3200 | 0.6800 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | eval_flow_sudden_threat | 0.4400 | 0.5600 |
| run_name | method_key | eval_seed | threat_motion_mode | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | accel_decel | 0.4364 | 0.5636 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | ar1_velocity | 0.4205 | 0.5795 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | crossing_or_sudden_threat | 0.4490 | 0.5510 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | linear | 0.3548 | 0.6452 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | sinusoidal_lateral | 0.4545 | 0.5455 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | accel_decel | 0.4386 | 0.5614 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | ar1_velocity | 0.4000 | 0.6000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | crossing_or_sudden_threat | 0.4510 | 0.5490 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | linear | 0.3000 | 0.7000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | sinusoidal_lateral | 0.4416 | 0.5584 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | accel_decel | 0.4483 | 0.5517 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | ar1_velocity | 0.4118 | 0.5882 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | crossing_or_sudden_threat | 0.4600 | 0.5400 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | linear | 0.3000 | 0.7000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | sinusoidal_lateral | 0.4286 | 0.5714 |
| run_name | method_key | eval_seed | threat_class | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | high | 0.4175 | 0.5825 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | low | 1.0000 | 0.0000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1000.0000 | medium | 0.6429 | 0.3571 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | high | 0.4035 | 0.5965 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | low | 1.0000 | 0.0000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1001.0000 | medium | 0.6429 | 0.3571 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | high | 0.4049 | 0.5951 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | low | 1.0000 | 0.0000 |
| seed2_rerunA | p3_s2_seed2_rerunA_1500k | 1002.0000 | medium | 0.6667 | 0.3333 |

## Artifacts

### tables
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_behavior_diagnostics.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_checkpoint_integrity_extended.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_checkpoint_path_audit.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_config_diff.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_decision.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_feature_gpsi_audit.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_intermediate_checkpoint_eval.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_intermediate_checkpoint_eval_by_seed.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2a_training_curve_diagnostics.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/tables/phase_n3pf_ms_seed2ab_eval_command_manifest.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2ab_eval_command_manifest.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_behavior_diagnostics.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_checkpoint_integrity.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_command_manifest.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_decision.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_rerun_config_manifest.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_resource_affinity.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_schema_check.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_threat_class_breakdown.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_train_curve.csv`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables/phase_n3pf_ms_seed2b_train_heartbeat.csv`
### plots
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/plots/seed2_original_vs_rerun_behavior.png`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/plots/seed2_original_vs_rerun_success_collision.png`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/plots/seed2_rerun_training_curve.png`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/plots/seed2_seed3_sanity_comparison.png`
### logs
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/logs/phase_n3pf_ms_seed2_ab_analysis.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/logs/phase_n3pf_ms_seed2_ab_bash_n.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/logs/phase_n3pf_ms_seed2_ab_py_compile.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/phase_n3pf_ms_seed2_ab_watcher.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/logs/phase_n3pf_ms_seed2a_audit.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit/logs/phase_n3pf_ms_seed2a_intermediate_eval.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/logs/phase_n3pf_ms_seed2b_eval_seed2_rerunA.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/logs/phase_n3pf_ms_seed2b_train_seed2_rerunA.log`
- `results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/logs/phase_n3pf_ms_seed2b_validate_seed2_rerunA.log`
### flags
- none
