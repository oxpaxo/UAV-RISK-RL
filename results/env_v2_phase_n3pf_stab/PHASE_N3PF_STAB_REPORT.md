# Phase N3PF-STAB Report

## Terminal Decision

`terminal_decision = phase_n3pf_stab_complete`

GitHub sync status: `pushed`; commit: `f28272bd694d490aae649b9878c20f6a8c1cfb98`.

N4-U remains blocked in this phase. N4-O remains conditional positive evidence, but must be rerun only after a stable P3-STAB policy is confirmed.

## Repo Verification

| item | kind | ok | detail |
| --- | --- | --- | --- |
| models/gpsi_ppo_policy.py | file | 1.0000 | models/gpsi_ppo_policy.py |
| GpsiBlockProjectedNoZExtractor | text | 1.0000 | models/gpsi_ppo_policy.py |
| GpsiGatedResidualExtractor | text | 1.0000 | models/gpsi_ppo_policy.py |
| policies/obstacle_set_extractor.py | file | 1.0000 | policies/obstacle_set_extractor.py |
| ObstacleSetExtractor | text | 1.0000 | policies/obstacle_set_extractor.py |
| envs/wrappers/gpsi_obs_wrapper.py | file | 1.0000 | envs/wrappers/gpsi_obs_wrapper.py |
| configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml | file | 1.0000 | configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml |
| work_dirs/gpsi_heada_v1_nll/best.pth | file | 1.0000 | work_dirs/gpsi_heada_v1_nll/best.pth |
| results/env_v2_phase_n3pf_ms_multiseed | dir | 1.0000 | results/env_v2_phase_n3pf_ms_multiseed |
| results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun | dir | 1.0000 | results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun |

## Config Changes

| variant | changed_field | old_value | new_value | structure_changed |
| --- | --- | --- | --- | --- |
| stab_s1_lr2e4 | ppo.learning_rate | 0.0003 | 0.0002 | 0.0000 |
| stab_s1_lr1e4 | ppo.learning_rate | 0.0003 | 0.0001 | 0.0000 |
| stab_s2d_gated | ppo.feature_adapter | block_projected_no_z | gated_residual_no_z | 1.0000 |
| stab_s2d_gated | s2_variant |  | S2-D attention-like fallback | 1.0000 |

S2 implementation type: `S2-D attention_like_gated_gpsi`.
S2 attention preservation: no strict attention-preserving claim. It does not warm-start from trained attention_full and is only an attention-like fallback with gate initialized near zero.

Validation/test/final-heldout discipline: validation seeds are 900/901 and are the only selector inputs. Test seeds 1000/1001/1002 and final-heldout seeds 1100/1101/1102 are reserved for later frozen-candidate evaluation.

Selection metric is pre-registered as `success_rate - 2 * collision_rate`.

## Seed2 Collapse Diagnostics

| comparison | method_key | success_rate | collision_rate | raw_unsafe_action_rate | action_delta | raw_min_predicted_cpa | raw_min_predicted_ttc | progress | ab_per_step_trace_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed0_vs_seed2_original | p3_s0_1500k | 0.6167 | 0.3833 | 0.3138 | 0.1543 | 2.6807 | 1.8517 | 0.9541 | 0.0000 |
| seed0_vs_seed2_original | p3_s2_1500k | 0.4222 | 0.5778 | 0.2688 | 0.1104 | 2.7749 | 1.7470 | 0.9526 | 0.0000 |
| seed2_original_vs_seed2_rerunA | p3_s2_seed2_rerunA_1500k | 0.4222 | 0.5778 | 0.2688 | 0.1104 | 2.8221 | nan | 0.9526 | 0.0000 |

## AB Consistency

| engineering_error_found | seed2_collapse_reproduced | most_likely_cause | seed2_rerunA_recovered | behavior_rows_available |
| --- | --- | --- | --- | --- |
| 0.0000 | 1.0000 | collapse_reproducible_or_inconclusive_architecture_instability | 0.0000 | 1.0000 |

## Validation Screening

| variant | checkpoint_label | success_rate | collision_rate | progress | raw_unsafe_action_rate | action_delta | eval_seed_count | episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stab_s1_lr1e4 | 1000k | 0.4383 | 0.5617 | 0.9656 | 0.2314 | 0.0894 | 2.0000 | 600.0000 |
| stab_s1_lr1e4 | 500k | 0.2883 | 0.7117 | 0.9308 | 0.3024 | 0.1023 | 2.0000 | 600.0000 |
| stab_s1_lr1e4 | 750k | 0.2700 | 0.7300 | 0.9446 | 0.2463 | 0.0806 | 2.0000 | 600.0000 |
| stab_s1_lr1e4 | final | 0.4917 | 0.5083 | 0.9644 | 0.2349 | 0.0943 | 2.0000 | 600.0000 |
| stab_s1_lr2e4 | 1000k | 0.5583 | 0.4417 | 0.9758 | 0.1909 | 0.1183 | 2.0000 | 600.0000 |
| stab_s1_lr2e4 | 500k | 0.4117 | 0.5883 | 0.9177 | 0.2945 | 0.1307 | 2.0000 | 600.0000 |
| stab_s1_lr2e4 | 750k | 0.5533 | 0.4467 | 0.9650 | 0.2626 | 0.1141 | 2.0000 | 600.0000 |
| stab_s1_lr2e4 | final | 0.5117 | 0.4883 | 0.9714 | 0.2116 | 0.1291 | 2.0000 | 600.0000 |
| stab_s2d_gated | 1000k | 0.5933 | 0.4067 | 0.9726 | 0.2050 | 0.0987 | 2.0000 | 600.0000 |
| stab_s2d_gated | 500k | 0.4050 | 0.5950 | 0.9172 | 0.3039 | 0.1191 | 2.0000 | 600.0000 |
| stab_s2d_gated | 750k | 0.4483 | 0.5517 | 0.9402 | 0.2613 | 0.0898 | 2.0000 | 600.0000 |
| stab_s2d_gated | final | 0.4533 | 0.5467 | 0.9620 | 0.2196 | 0.0978 | 2.0000 | 600.0000 |

## Selector Decision

| variant | selected_checkpoint_label | success_rate | collision_rate | selection_score | seed2_screening_gate_pass | trend_improving_through_1000k | continue_to_1500k_recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stab_s1_lr1e4 | final | 0.4917 | 0.5083 | -0.5250 | 0.0000 | 0.0000 | 0.0000 |
| stab_s1_lr2e4 | 1000k | 0.5583 | 0.4417 | -0.3250 | 0.0000 | 1.0000 | 1.0000 |
| stab_s2d_gated | 1000k | 0.5933 | 0.4067 | -0.2200 | 1.0000 | 1.0000 | 1.0000 |

## Diagnostics Decision

| variant | diagnostics_ok | feature_nonfinite_count | delta_norm_1s_p95_max | adapter_output_l2_p95_max |
| --- | --- | --- | --- | --- |
| stab_s1_lr1e4 | 1.0000 | 0.0000 | 2.0357 | 5.7014 |
| stab_s1_lr2e4 | 1.0000 | 0.0000 | 2.0534 | 5.7916 |
| stab_s2d_gated | 1.0000 | 0.0000 | 2.1318 | 7.3426 |

## Final Decision

| phase_status | p3_stab_candidate_found | selected_screening_variant | seed2_screening_passed_any | continue_1500k_recommended_any | multi_seed_confirmation_status | attention_full_multiseed_needed | n4o_can_rerun_next | n4u_blocked | terminal_decision | available_attention_success | available_attention_collision | noz_success | noz_collision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed2_screening_passed_proceed_multiseed_confirmation | screening_only | stab_s2d_gated | 1.0000 | 1.0000 | not_run | yes_before_decisive_attention_claim | after_seed0_1_2_confirmation | yes | phase_n3pf_stab_complete | 0.6033 | 0.3967 | 0.5667 | 0.4333 |

## Artifacts

### tables
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_checkpoint_manifest.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_command_manifest.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_config_diff.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_config_manifest.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_diagnostics_decision.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_eval_command_manifest.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_final_decision.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_repo_verification.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_resource_affinity.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_schema_check.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_seed2_breakdown_diagnostics.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_seed2_collapse_diagnostics.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_seed2_rerun_consistency.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_selector_decision.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_train_curve.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_train_heartbeat.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_checkpoint_scores.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_episode_metrics.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_validation_threat_class_breakdown.csv`
### plots
- `results/env_v2_phase_n3pf_stab/plots/phase_n3pf_stab_seed2_collapse_summary.png`
- `results/env_v2_phase_n3pf_stab/plots/phase_n3pf_stab_validation_success_collision.png`
### logs
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_analysis.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_bash_n.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_py_compile.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_selector.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_train_lr1e4.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_train_lr2e4.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_train_s2d.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_validate_lr1e4.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_validate_lr2e4.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_validate_s2d.log`
- `results/env_v2_phase_n3pf_stab/logs/phase_n3pf_stab_validation_eval.log`
- `results/env_v2_phase_n3pf_stab/phase_n3pf_stab_watcher.log`
### flags
- none
