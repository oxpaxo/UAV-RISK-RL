# Phase N3PF Block-Projected Full Report

## Terminal Decision

`terminal_decision = phase_n3pf_block_projected_full_complete`

P3 block-projected no_z continuation completed from the fixed N3P checkpoint_500k parent with reset_num_timesteps=False.

## Final Candidate Decision

| p3_checkpoint_label | p3_success | p3_collision | p3_parent_success | p3_parent_collision | noz_success | noz_collision | attention_success | attention_collision | z2_success | z2_collision | p3_minus_noz_success | p3_minus_noz_collision | p3_minus_attention_success | p3_minus_attention_collision | p3_minus_z2_success | p3_minus_z2_collision | p3_beats_noz_gate | p3_beats_attention_gate | strong_m1_candidate | do_not_claim_beats_attention | diagnostics_ok | selected_n4_candidate | can_enter_n4 | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | 0.5967 | 0.4033 | 0.5333 | 0.4667 | 0.5633 | 0.4367 | 0.6100 | 0.3900 | 0.5067 | 0.4933 | 0.0333 | -0.0333 | -0.0133 | 0.0133 | 0.0900 | -0.0900 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | P3_block_projected | yes | P3 meets or beats no_z on both success and collision. |
- Can enter N4: yes.
- Selected N4 candidate: `P3_block_projected`.
- Attention claim guard: do_not_claim_beats_attention=1.

## Resume Semantics

| phase | selected_parent_path | selected_parent_sha256 | reset_num_timesteps | model_num_timesteps | model_parent_step_match | model_target_step_match | optimizer_state_restored | n_envs | n_steps | batch_size | feature_extractor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before_learn | checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip | 9ba7f68424e65413aead99a51bc1f7bc8de1f00dabbbf6bace45f14dc3bd7ab2 | 0.0000 | 500000.0000 | 1.0000 | nan | 1.0000 | 4.0000 | 1024.0000 | 256.0000 | GpsiBlockProjectedNoZExtractor |
| after_learn | checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip | 9ba7f68424e65413aead99a51bc1f7bc8de1f00dabbbf6bace45f14dc3bd7ab2 | 0.0000 | 1503520.0000 | nan | 1.0000 | 1.0000 | 4.0000 | 1024.0000 | 256.0000 | GpsiBlockProjectedNoZExtractor |

## Checkpoint Integrity

| checkpoint_label | checkpoint_path | sha256 | global_total_step | model_num_timesteps | parameter_l2_delta_vs_parent | parameter_max_abs_delta_vs_parent | optimizer_state_present | eval_path_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parent_500k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/parent_500k.zip | 9ba7f68424e65413aead99a51bc1f7bc8de1f00dabbbf6bace45f14dc3bd7ab2 | 500000.0000 | 500000.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 750k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_750k.zip | 184f5fb2122f3430aef7abfca68d600bd5cb7c07d9ce90d3b3ade4788348955a | 750000.0000 | 750000.0000 | 16.0113 | 0.2986 | 1.0000 | 1.0000 |
| 1000k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip | 5685cb07b5d861abac35f8c3fa4103cfaa5ef6c574d2dcb613f80b695e9c5a04 | 1000000.0000 | 1000000.0000 | 21.6394 | 0.6417 | 1.0000 | 1.0000 |
| 1250k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1250k.zip | b839daa9f610f9200b8f50cfd903b0cbfa9a96357ea2b53c43d7fabf49add1c9 | 1250000.0000 | 1250000.0000 | 26.6777 | 1.0725 | 1.0000 | 1.0000 |
| 1500k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip | 1aec7318cb0b745f3e98ea3d8e15996e40ac19cf416220cc7ead9926c567bd3b | 1500000.0000 | 1500000.0000 | 31.0108 | 1.3453 | 1.0000 | 1.0000 |
| final | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip | a7537e97403d135500408a3dc98b736ab2da997649ce4b87d8fc3dd7b524310a | 1500000.0000 | 1503520.0000 | 31.0840 | 1.3551 | 1.0000 | 1.0000 |
| best_by_eval | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/best_by_eval.zip | a7537e97403d135500408a3dc98b736ab2da997649ce4b87d8fc3dd7b524310a | 1500000.0000 | 1503520.0000 | 31.0840 | 1.3551 | 1.0000 | 1.0000 |

## Aggregate Eval

| method_key | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 0.6100 | 0.3900 | 0.5767 | 0.9572 | 0.2914 | 0.1388 | 4.8633 |
| block_projected_full | 1000k | 0.6300 | 0.3700 | 0.5367 | 0.9556 | 0.2275 | 0.1226 | 4.7367 |
| block_projected_full | 1250k | 0.4467 | 0.5533 | 0.4200 | 0.9224 | 0.3119 | 0.1164 | 4.2243 |
| block_projected_full | 1500k | 0.6200 | 0.3800 | 0.5967 | 0.9542 | 0.3131 | 0.1537 | 4.6516 |
| block_projected_full | 750k | 0.4933 | 0.5067 | 0.4300 | 0.9351 | 0.2392 | 0.1015 | 4.8418 |
| block_projected_full | best_by_eval | 0.5967 | 0.4033 | 0.5667 | 0.9451 | 0.3075 | 0.1509 | 4.7910 |
| block_projected_full | final | 0.5967 | 0.4033 | 0.5667 | 0.9451 | 0.3075 | 0.1509 | 4.7910 |
| block_projected_full | parent_500k | 0.5333 | 0.4667 | 0.4700 | 0.9282 | 0.2465 | 0.0911 | 5.1054 |
| n3f_no_z_full | final | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 0.0967 | 5.4235 |
| z2_corrected_full | z2_final | 0.5067 | 0.4933 | 0.4700 | 0.9260 | 0.2629 | 0.0905 | 5.2264 |

## Diagnostics

| method_key | diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | logvar_xy_1s_span_max | logvar_scaled_l2_p95_max | adapter_output_l2_p95_max | full_aug_obs_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_projected_full | 1.0000 | 2.0861 | 2.7372 | 0.0000 | 5.6387 | 3.0000 | 6.1791 | 5.1628 | 0.0000 |

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_750k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/parent_500k.zip`
### tables
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_attention_reference_comparison.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_checkpoint_integrity.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_command_manifest.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_config_manifest.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_diagnostics_decision.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_episode_metrics.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_eval_summary.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_feature_block_stats.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_final_candidate_decision.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_noz_reference_comparison.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_reference_comparison.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_resource_affinity.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_resume_semantics.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_scenario_breakdown.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_schema_check.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_threat_class_breakdown.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_train_curve.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_train_heartbeat.csv`
- `results/env_v2_phase_n3pf_block_projected_full/tables/phase_n3pf_z2_reference_comparison.csv`
### plots
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_action_dynamics.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_checkpoint_integrity.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_checkpoint_success_collision.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_feature_block_scale.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_gpsi_delta_norm.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_gpsi_logvar.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_raw_unsafe_by_checkpoint.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_scenario_breakdown.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_train_reward.png`
- `results/env_v2_phase_n3pf_block_projected_full/plots/n3pf_vs_noz_attention_success_collision.png`
### logs
- `results/env_v2_phase_n3pf_block_projected_full/logs/phase_n3pf_analysis.log`
- `results/env_v2_phase_n3pf_block_projected_full/logs/phase_n3pf_eval.log`
- `results/env_v2_phase_n3pf_block_projected_full/logs/phase_n3pf_train_block_projected_full.log`
- `results/env_v2_phase_n3pf_block_projected_full/phase_n3pf_watcher.log`
### flags
- `results/env_v2_phase_n3pf_block_projected_full/PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag`
