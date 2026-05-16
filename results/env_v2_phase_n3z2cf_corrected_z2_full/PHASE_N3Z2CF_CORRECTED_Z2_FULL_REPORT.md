# Phase N3Z2CF Corrected Z2 Full Report

## Terminal Decision

`terminal_decision = phase_n3z2cf_corrected_z2_full_complete`

Corrected Z2 full continuation completed from the fixed checkpoint_500k parent with reset_num_timesteps=False.

## Final Candidate Decision

| corrected_z2_checkpoint_label | corrected_z2_success | corrected_z2_collision | noz_success | noz_collision | attention_success | attention_collision | old_z2c_success | old_z2c_collision | corrected_minus_noz_success | corrected_minus_noz_collision | corrected_minus_attention_success | corrected_minus_attention_collision | corrected_minus_old_success | corrected_minus_old_collision | z2_beats_noz_gate | z2_beats_attention_gate | do_not_claim_beats_attention | diagnostics_ok | selected_n4_candidate | can_enter_n4 | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | 0.5067 | 0.4933 | 0.5633 | 0.4367 | 0.6100 | 0.3900 | 0.4500 | 0.5500 | -0.0567 | 0.0567 | -0.1033 | 0.1033 | 0.0567 | -0.0567 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | no_z | yes | corrected Z2 is not better than no_z under the decision rule; keep Z2 as ablation. |
- Can enter N4: yes.
- Selected N4 candidate: `no_z`.
- Attention claim guard: do_not_claim_beats_attention=1.

## Resume Semantics

| phase | loaded_checkpoint_path | parent_sha256 | reset_num_timesteps | model_num_timesteps | model_parent_step_match | model_target_step_match | optimizer_state_restored | n_envs | n_steps | batch_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before_learn | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip | 6c49b0a153d7579e6b099a22d0e9e604abce279647de2d811008c48e2103708d | 0.0000 | 500000.0000 | 1.0000 | nan | 1.0000 | 4.0000 | 1024.0000 | 256.0000 |
| after_learn | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip | 6c49b0a153d7579e6b099a22d0e9e604abce279647de2d811008c48e2103708d | 0.0000 | 1503520.0000 | nan | 1.0000 | 1.0000 | 4.0000 | 1024.0000 | 256.0000 |

## Checkpoint Integrity

| checkpoint_label | checkpoint_path | sha256 | global_total_step | model_num_timesteps | parameter_l2_delta_vs_parent | parameter_max_abs_delta_vs_parent | optimizer_state_present | eval_path_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parent_500k | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/parent_500k.zip | 6c49b0a153d7579e6b099a22d0e9e604abce279647de2d811008c48e2103708d | 500000.0000 | 500000.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 750k | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_750k.zip | 94be8bc66bcdd0dee427d446430cc7d50ce29134ce9fc828aa9f84bd9a7186f2 | 750000.0000 | 750000.0000 | 15.4620 | 0.2931 | 1.0000 | 1.0000 |
| 1000k | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1000k.zip | 42128c77520417c60b818ea6162b757fff8c0a38609b1eacebd12cf40c959e93 | 1000000.0000 | 1000000.0000 | 22.3799 | 0.6011 | 1.0000 | 1.0000 |
| 1250k | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1250k.zip | 01076c755d051f74bef4356486e28d84333ddce4689d0a2f1a7ff9b1344bd5be | 1250000.0000 | 1250000.0000 | 27.7985 | 0.9147 | 1.0000 | 1.0000 |
| 1500k | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1500k.zip | 725a10d1e18bbed3de18ebf4f27b5d5898558c813ff2b142e1ec9c63ba4232e7 | 1500000.0000 | 1500000.0000 | 32.8194 | 1.0534 | 1.0000 | 1.0000 |
| final | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip | bbdcc7255e27c6ccca7e41df0c007c9163e03d880771b1812d8350d6b021f208 | 1500000.0000 | 1503520.0000 | 32.9457 | 1.0586 | 1.0000 | 1.0000 |
| best_by_eval | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/best_by_eval.zip | bbdcc7255e27c6ccca7e41df0c007c9163e03d880771b1812d8350d6b021f208 | 1500000.0000 | 1503520.0000 | 32.9457 | 1.0586 | 1.0000 | 1.0000 |

## Aggregate Eval

| method_key | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 0.6100 | 0.3900 | 0.5767 | 0.9572 | 0.2914 | 0.1388 | 4.8633 |
| n3f_no_z_full | final | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 0.0967 | 5.4235 |
| z2_corrected_full | 1000k | 0.4600 | 0.5400 | 0.4200 | 0.9419 | 0.2551 | 0.0738 | 5.7286 |
| z2_corrected_full | 1250k | 0.5367 | 0.4633 | 0.4467 | 0.9273 | 0.2410 | 0.0905 | 5.5479 |
| z2_corrected_full | 1500k | 0.4767 | 0.5233 | 0.4267 | 0.9100 | 0.2656 | 0.0886 | 5.1355 |
| z2_corrected_full | 750k | 0.4800 | 0.5200 | 0.4167 | 0.9390 | 0.2266 | 0.0901 | 5.5538 |
| z2_corrected_full | best_by_eval | 0.5067 | 0.4933 | 0.4700 | 0.9260 | 0.2629 | 0.0905 | 5.2264 |
| z2_corrected_full | final | 0.5067 | 0.4933 | 0.4700 | 0.9260 | 0.2629 | 0.0905 | 5.2264 |
| z2_corrected_full | parent_500k | 0.4800 | 0.5200 | 0.4600 | 0.9400 | 0.2285 | 0.0832 | 5.5098 |
| z2_old_n3z2c | old_final | 0.4500 | 0.5500 | 0.3800 | 0.9253 | 0.2771 | 0.0881 | 5.5193 |

## Diagnostics

| method_key | diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | logvar_xy_1s_span_max | z_after_constraint_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| z2_corrected_full | 1.0000 | 1.6965 | 2.1465 | 0.0000 | 5.5170 | 4.0000 | 0.0000 |

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1000k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_1500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/checkpoint_750k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/parent_500k.zip`
### tables
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_attention_reference_comparison.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_checkpoint_integrity.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_command_manifest.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_config_manifest.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_diagnostics_decision.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_episode_metrics.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_eval_summary.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_feature_block_stats.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_final_candidate_decision.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_gpsi_output_summary.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_noz_reference_comparison.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_old_z2c_comparison.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_resource_affinity.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_resume_semantics.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_scenario_breakdown.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_schema_check.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_threat_class_breakdown.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_train_curve.csv`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/tables/phase_n3z2cf_train_heartbeat.csv`
### plots
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_action_dynamics.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_checkpoint_integrity.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_checkpoint_success_collision.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_feature_block_scale.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_gpsi_delta_norm.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_gpsi_logvar.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_raw_unsafe_by_checkpoint.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_scenario_breakdown.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_train_reward.png`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/plots/z2cf_vs_noz_attention_success_collision.png`
### logs
- `results/env_v2_phase_n3z2cf_corrected_z2_full/logs/phase_n3z2cf_analysis.log`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/logs/phase_n3z2cf_eval.log`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/logs/phase_n3z2cf_train_corrected_full.log`
- `results/env_v2_phase_n3z2cf_corrected_z2_full/phase_n3z2cf_watcher.log`
### flags
- `results/env_v2_phase_n3z2cf_corrected_z2_full/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag`
