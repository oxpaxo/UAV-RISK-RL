# Phase N3PF-STAB-CONFIRM Report

## Terminal Decision

`terminal_decision = phase_n3pf_stab_confirm_complete_s2d_failed`

GitHub sync status: `success_remote_main`; commit: `af2273c2850fc3d732d59c80f8a8c941a1adc73d`.

S2-D semantics: `attention_like_gated_gpsi`. This is not strict attention-preserving, because no attention_full warm-start, isomorphic parameter load, or distillation is used.

N4-U remains blocked. N4-O can only be rerun next phase if the candidate is confirmed.

## Preflight

| item | kind | ok | detail |
| --- | --- | --- | --- |
| guide | file | 1.0000 | codex_guide/PHASE_N3PF_STAB_CONFIRM_GUIDE.md |
| s2d_config | file | 1.0000 | configs/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_gated.yaml |
| gpsi_checkpoint | file | 1.0000 | work_dirs/gpsi_heada_v1_nll/best.pth |
| gated_extractor_class | GpsiGatedResidualExtractor | 1.0000 | GpsiGatedResidualExtractor in models/gpsi_ppo_policy.py |
| base_block_projected_class | GpsiBlockProjectedNoZExtractor | 1.0000 | GpsiBlockProjectedNoZExtractor in models/gpsi_ppo_policy.py |
| previous_stab_complete | file | 1.0000 | results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_COMPLETE.flag |

## Parallel Training Strategy

Mandatory S2-D seeds 0/1/2 were scheduled as separate CPU PPO jobs with n_envs=4, device=cpu, and OMP/MKL/OpenBLAS/NumExpr set to 1. Seed3 is optional and included only if the watcher launched and completed it.

## Selector Discipline

| validation_seeds | test_seeds | final_heldout_seeds | selector_used_only_validation | test_used_for_structure_or_checkpoint_selection | selector_rows |
| --- | --- | --- | --- | --- | --- |
| 900,901 | 1000,1001,1002 | 1100,1101,1102 | 1.0000 | 0.0000 | 4.0000 |

## Validation Selector

| training_seed | selected_checkpoint_label | success_rate | collision_rate | selection_score | selector_used_only_validation_seeds | test_seed_used_for_selection |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1250k | 0.4750 | 0.5250 | -0.5750 | 1.0000 | 0.0000 |
| 1.0000 | final | 0.4933 | 0.5067 | -0.5200 | 1.0000 | 0.0000 |
| 2.0000 | 1000k | 0.5933 | 0.4067 | -0.2200 | 1.0000 | 0.0000 |
| 3.0000 | final | 0.4750 | 0.5250 | -0.5750 | 1.0000 | 0.0000 |

## Validation Aggregate

| training_seed | checkpoint_label | success_rate | collision_rate | progress | raw_unsafe_action_rate | episodes | eval_seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1000k | 0.4167 | 0.5833 | 0.9363 | 0.2775 | 600.0000 | 2.0000 |
| 0.0000 | 1250k | 0.4750 | 0.5250 | 0.9554 | 0.2292 | 600.0000 | 2.0000 |
| 0.0000 | 1500k | 0.2883 | 0.7117 | 0.9095 | 0.3236 | 600.0000 | 2.0000 |
| 0.0000 | 500k | 0.4183 | 0.5817 | 0.9545 | 0.2617 | 600.0000 | 2.0000 |
| 0.0000 | 750k | 0.4467 | 0.5533 | 0.9444 | 0.2560 | 600.0000 | 2.0000 |
| 0.0000 | final | 0.3117 | 0.6883 | 0.9042 | 0.3115 | 600.0000 | 2.0000 |
| 1.0000 | 1000k | 0.3783 | 0.6217 | 0.9051 | 0.3191 | 600.0000 | 2.0000 |
| 1.0000 | 1250k | 0.3450 | 0.6550 | 0.9324 | 0.2940 | 600.0000 | 2.0000 |
| 1.0000 | 1500k | 0.4367 | 0.5633 | 0.9257 | 0.3261 | 600.0000 | 2.0000 |
| 1.0000 | 500k | 0.3783 | 0.6217 | 0.9221 | 0.2867 | 600.0000 | 2.0000 |
| 1.0000 | 750k | 0.4350 | 0.5650 | 0.9418 | 0.2730 | 600.0000 | 2.0000 |
| 1.0000 | final | 0.4933 | 0.5067 | 0.9342 | 0.3175 | 600.0000 | 2.0000 |
| 2.0000 | 1000k | 0.5933 | 0.4067 | 0.9726 | 0.2050 | 600.0000 | 2.0000 |
| 2.0000 | 1250k | 0.4567 | 0.5433 | 0.9426 | 0.2763 | 600.0000 | 2.0000 |
| 2.0000 | 1500k | 0.5200 | 0.4800 | 0.9393 | 0.2896 | 600.0000 | 2.0000 |
| 2.0000 | 500k | 0.4050 | 0.5950 | 0.9172 | 0.3039 | 600.0000 | 2.0000 |
| 2.0000 | 750k | 0.4483 | 0.5517 | 0.9402 | 0.2613 | 600.0000 | 2.0000 |
| 2.0000 | final | 0.4267 | 0.5733 | 0.9365 | 0.2912 | 600.0000 | 2.0000 |
| 3.0000 | 1000k | 0.3417 | 0.6583 | 0.8946 | 0.2408 | 600.0000 | 2.0000 |
| 3.0000 | 1250k | 0.3250 | 0.6750 | 0.9181 | 0.2882 | 600.0000 | 2.0000 |
| 3.0000 | 1500k | 0.4717 | 0.5283 | 0.9562 | 0.2415 | 600.0000 | 2.0000 |
| 3.0000 | 500k | 0.3850 | 0.6150 | 0.9423 | 0.2501 | 600.0000 | 2.0000 |
| 3.0000 | 750k | 0.4317 | 0.5683 | 0.9347 | 0.2523 | 600.0000 | 2.0000 |
| 3.0000 | final | 0.4750 | 0.5250 | 0.9607 | 0.2469 | 600.0000 | 2.0000 |

## Test Aggregate

| training_seed | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | episodes | eval_seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1250k | 0.5100 | 0.4900 | 0.4656 | 0.9558 | 0.2372 | 0.1139 | 900.0000 | 3.0000 |
| 1.0000 | final | 0.5200 | 0.4800 | 0.4867 | 0.9181 | 0.3211 | 0.1237 | 900.0000 | 3.0000 |
| 2.0000 | 1000k | 0.5067 | 0.4933 | 0.4600 | 0.9579 | 0.2100 | 0.0947 | 900.0000 | 3.0000 |
| 3.0000 | final | 0.4667 | 0.5333 | 0.4356 | 0.9444 | 0.2573 | 0.0938 | 900.0000 | 3.0000 |

## PPO Training Diagnostics

| variant | training_seed | approx_kl | clip_fraction | entropy_loss | policy_gradient_loss | value_loss | explained_variance | learning_rate | n_updates | std | log_std_mean | steps_last | fps_mean | fps_last | heartbeat_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stab_s2d_gated | 0.0000 | 0.0197 | 0.1701 | -4.2386 | -0.0081 | 5.9673 | 0.6296 | 0.0003 | 1840.4545 | 1.0149 | -0.0049 | 1448700.0000 | 219.3732 | 219.4433 | 22.0000 |
| stab_s2d_gated | 1.0000 | 0.0251 | 0.1907 | -4.5261 | -0.0069 | 6.6855 | 0.6008 | 0.0003 | 1857.7273 | 1.1418 | 0.0909 | 1460384.0000 | 221.3886 | 221.1606 | 22.0000 |
| stab_s2d_gated | 2.0000 | 0.0197 | 0.1696 | -4.5117 | -0.0067 | 6.1743 | 0.6020 | 0.0003 | 1853.6364 | 1.1314 | 0.0856 | 1457812.0000 | 220.8889 | 220.7795 | 22.0000 |
| stab_s2d_gated | 3.0000 | 0.0171 | 0.1510 | -4.4354 | -0.0067 | 7.8303 | 0.5449 | 0.0003 | 1863.1818 | 1.0858 | 0.0605 | 1469664.0000 | 221.2647 | 222.6410 | 22.0000 |

## Parameter Drift

| training_seed | selected_checkpoint_label | comparison | all_l2_delta | feature_extractor_l2_delta | actor_action_l2_delta | critic_value_l2_delta | log_std_l2_delta | gate_l2_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1250k | selected_vs_final | 17.8345 | 12.6668 | 5.9301 | 11.0635 | 0.2382 | 0.5583 |
| 0.0000 | 1250k | 1000k_vs_final | 24.3834 | 17.2181 | 8.1342 | 15.2248 | 0.3600 | 1.2175 |
| 1.0000 | final | selected_vs_final | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | final | 1000k_vs_final | 24.3510 | 16.8994 | 8.3213 | 15.4229 | 0.5199 | 1.2317 |
| 2.0000 | 1000k | selected_vs_final | 24.3350 | 17.2017 | 7.5891 | 15.4423 | 0.4880 | 1.0850 |
| 2.0000 | 1000k | 1000k_vs_final | 24.3350 | 17.2017 | 7.5891 | 15.4423 | 0.4880 | 1.0850 |
| 3.0000 | final | selected_vs_final | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3.0000 | final | 1000k_vs_final | 24.2456 | 17.1805 | 7.6528 | 15.2987 | 0.2537 | 1.0466 |

## Attention Full Audit

| training_seed | exists | source | formal_1500k_protocol_confirmed | path |
| --- | --- | --- | --- | --- |
| 0.0000 | 1.0000 | formal_phase2_1500k | 1.0000 | checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip |
| 1.0000 | 1.0000 | legacy_top_level_unknown_protocol | 0.0000 | checkpoints/attention_full_s1.zip |
| 2.0000 | 1.0000 | legacy_top_level_unknown_protocol | 0.0000 | checkpoints/attention_full_s2.zip |

## Final Decision Table

| terminal_decision | s2d_status | mandatory_training_seed_count | test_mean_success | test_mean_collision | test_min_success | test_max_collision | attention_full_reference_success | attention_full_reference_collision | noz_full_reference_success | noz_full_reference_collision | comparable_to_attention_seed0_reference | decisive_multiseed_attention_claim_allowed | n4o_can_rerun_next | n4u_blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_n3pf_stab_confirm_complete_s2d_failed | failed_stability_gate | 3.0000 | 0.5122 | 0.4878 | 0.5067 | 0.4933 | 0.6033 | 0.3967 | 0.5667 | 0.4333 | 0.0000 | 0.0000 | no | yes |

## Artifacts

### tables
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_attention_full_multiseed_audit.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_checkpoint_manifest.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_command_manifest.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_config_manifest.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_eval_command_manifest.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_final_decision.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_parameter_drift.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_ppo_training_diagnostics.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_preflight_check.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_resource_affinity.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_schema_check.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_selector_decision.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_selector_discipline.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_episode_metrics.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_test_threat_class_breakdown.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_train_curve.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_train_heartbeat.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_checkpoint_scores.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_episode_metrics.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_stab_confirm/tables/phase_n3pf_stab_confirm_validation_threat_class_breakdown.csv`
### plots
- `results/env_v2_phase_n3pf_stab_confirm/plots/phase_n3pf_stab_confirm_parameter_drift.png`
- `results/env_v2_phase_n3pf_stab_confirm/plots/phase_n3pf_stab_confirm_test_selected_success_collision.png`
- `results/env_v2_phase_n3pf_stab_confirm/plots/phase_n3pf_stab_confirm_training_fps.png`
- `results/env_v2_phase_n3pf_stab_confirm/plots/phase_n3pf_stab_confirm_validation_checkpoint_success_collision.png`
### logs
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_analysis.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_eval_test_selected.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_eval_validation.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_resource_preflight.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_selector.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_train_s0.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_train_s1.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_train_s2.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_train_s3.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_validate_seed0.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_validate_seed1.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_validate_seed2.log`
- `results/env_v2_phase_n3pf_stab_confirm/logs/phase_n3pf_stab_confirm_validate_seed3.log`
- `results/env_v2_phase_n3pf_stab_confirm/phase_n3pf_stab_confirm_watcher.log`
### flags
- none
