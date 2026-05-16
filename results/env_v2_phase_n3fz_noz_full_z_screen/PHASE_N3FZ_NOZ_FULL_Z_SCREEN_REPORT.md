# Phase N3F/Z No-Z Full + Z Screen Report

## Terminal Decision

`terminal_decision = phase_n3fz_noz_full_z_screen_complete`

Phase N3F/Z complete. This is a repaired no-shield PPO no_z full rerun plus constrained z_i screening; it does not enter N4.

## Engineering Facts

- Phase N3R complete flag exists and N3R no_z 500k baseline is used for Z hard gates.
- Phase N3.5 complete flag exists and repaired `GpsiObsWrapper` was used.
- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`; Gpsi was frozen under `eval()` with no trainable parameters.
- EnvV2 core was not modified.
- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.
- Track 1 budget: N3F no_z `1500000` PPO steps, seed 0.
- Track 2 budget: Z1/Z2 `500000` PPO steps each, seed 0.
- Evaluation used `50` episodes per scenario over six scenarios.
- Logvar clip sanity: configs use `[-5, 3]`, already bounded tighter than `|logvar| <= 5`.

## Configs

- Track 1 `n3f_no_z_full`: `[obs_i, delta_hat_scaled, logvar_hat]`, dim 30.
- Z1 `z_l2_scale_4`: `[obs_i, z_i / (||z_i||_2 + eps) * 4, delta_hat_scaled, logvar_hat]`, dim 94.
- Z2 `z_layernorm_alpha_0p5`: `[obs_i, 0.5 * LayerNorm(z_i), delta_hat_scaled, logvar_hat]`, dim 94.
- Z3 `z_proj16_layernorm`: not run in this stage; optional and resource-gated.

## Main Results

| method_key | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- |
| n3f_no_z_full | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 5.4235 |
| z_l2_scale_4 | 0.2933 | 0.7067 | 0.2567 | 0.9136 | 0.2751 | 4.9649 |
| z_layernorm_alpha_0p5 | 0.4700 | 0.5300 | 0.4267 | 0.9376 | 0.2362 | 5.3009 |

## Z Hard Gate

| method_key | success_rate | collision_rate | passes_success_gate | passes_collision_gate | diagnostics_ok | z_hard_gate_pass | eligible_for_1_5m_continuation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| z_l2_scale_4 | 0.2933 | 0.7067 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | no |
| z_layernorm_alpha_0p5 | 0.4700 | 0.5300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
- N3R no_z 500k gate: success >= 0.4233 and collision <= 0.5767.
- Z winner recommendation: `z_layernorm_alpha_0p5` / `gpsi_z_layernorm_alpha_0p5_screen`.
- Z 1.5M continuation needed: yes.
- Z hard-gate decision: Continue Z winner gpsi_z_layernorm_alpha_0p5_screen to 1.5M before choosing a final N4 candidate.

## N4 Candidate

- no_z full N4 policy candidate: defer_until_z_continuation.
- Can enter N4 now: no.
- Recommendation: N3F no_z full is clean, but a Z variant passed the continuation gate; run that Z continuation before N4.
- Next step: run the Z winner 1.5M continuation before N4.

## Attention Reference Comparison

| method_key | scenario | success_rate_config | success_rate_attention | delta_success_rate | collision_rate_config | collision_rate_attention | delta_collision_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n3f_no_z_full | eval_flow_high_density | 0.5000 | 0.6400 | -0.1400 | 0.5000 | 0.3600 | 0.1400 |
| n3f_no_z_full | eval_flow_high_speed | 0.4000 | 0.5400 | -0.1400 | 0.6000 | 0.4600 | 0.1400 |
| n3f_no_z_full | eval_flow_high_threat | 0.5000 | 0.6200 | -0.1200 | 0.5000 | 0.3800 | 0.1200 |
| n3f_no_z_full | eval_flow_id | 0.7600 | 0.5800 | 0.1800 | 0.2400 | 0.4200 | -0.1800 |
| n3f_no_z_full | eval_flow_mixed_ood | 0.5600 | 0.5400 | 0.0200 | 0.4400 | 0.4600 | -0.0200 |
| n3f_no_z_full | eval_flow_sudden_threat | 0.6600 | 0.7400 | -0.0800 | 0.3400 | 0.2600 | 0.0800 |
| z_l2_scale_4 | eval_flow_high_density | 0.2400 | 0.6400 | -0.4000 | 0.7600 | 0.3600 | 0.4000 |
| z_l2_scale_4 | eval_flow_high_speed | 0.0800 | 0.5400 | -0.4600 | 0.9200 | 0.4600 | 0.4600 |
| z_l2_scale_4 | eval_flow_high_threat | 0.3400 | 0.6200 | -0.2800 | 0.6600 | 0.3800 | 0.2800 |
| z_l2_scale_4 | eval_flow_id | 0.3400 | 0.5800 | -0.2400 | 0.6600 | 0.4200 | 0.2400 |
| z_l2_scale_4 | eval_flow_mixed_ood | 0.3000 | 0.5400 | -0.2400 | 0.7000 | 0.4600 | 0.2400 |
| z_l2_scale_4 | eval_flow_sudden_threat | 0.4600 | 0.7400 | -0.2800 | 0.5400 | 0.2600 | 0.2800 |
| z_layernorm_alpha_0p5 | eval_flow_high_density | 0.3800 | 0.6400 | -0.2600 | 0.6200 | 0.3600 | 0.2600 |
| z_layernorm_alpha_0p5 | eval_flow_high_speed | 0.4000 | 0.5400 | -0.1400 | 0.6000 | 0.4600 | 0.1400 |
| z_layernorm_alpha_0p5 | eval_flow_high_threat | 0.4200 | 0.6200 | -0.2000 | 0.5800 | 0.3800 | 0.2000 |
| z_layernorm_alpha_0p5 | eval_flow_id | 0.5400 | 0.5800 | -0.0400 | 0.4600 | 0.4200 | 0.0400 |
| z_layernorm_alpha_0p5 | eval_flow_mixed_ood | 0.5000 | 0.5400 | -0.0400 | 0.5000 | 0.4600 | 0.0400 |
| z_layernorm_alpha_0p5 | eval_flow_sudden_threat | 0.5800 | 0.7400 | -0.1600 | 0.4200 | 0.2600 | 0.1600 |

## Gpsi Output Diagnostics

| method_key | delta_norm_1s_p95 | delta_norm_1s_max | logvar_xy_1s_mean | logvar_xy_1s_span | projected_std_radial_mean | projected_std_relvel_mean | z_norm_raw_p95 | z_norm_after_p95 | z_zero_norm_count_max | history_valid_ratio_mean | inactive_forwarded_count_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n3f_no_z_full | 1.6247 | 2.0285 | -4.6855 | 5.2606 | 0.1053 | 0.1059 | 49.1228 | 49.1228 | 0.0000 | 0.6386 | 0.0000 |
| z_l2_scale_4 | 1.9956 | 2.7461 | -4.5417 | 5.6425 | 0.1227 | 0.1227 | 39.7895 | 4.0000 | 0.0000 | 0.6195 | 0.0000 |
| z_layernorm_alpha_0p5 | 1.6697 | 2.3558 | -4.6523 | 5.3152 | 0.1046 | 0.1062 | 46.3334 | 4.0000 | 0.0000 | 0.6244 | 0.0000 |

## Feature Block Stats

| method_key | block | z_transform | not_applicable | l2_norm_p95 | max_abs_p95 | nan_count | inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.9816 | 1.6448 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 2.0895 | 1.7852 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 2.0227 | 1.6480 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.9018 | 1.4893 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.9179 | 1.5434 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.8309 | 1.4118 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.9598 | 1.5161 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.8344 | 1.4582 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.8410 | 1.3781 | 0.0000 | 0.0000 |
| n3f_no_z_full | delta_hat_9_after_scale | raw | 0.0000 | 1.9079 | 1.4412 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.9302 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.9485 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.8803 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.8540 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.7985 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.9656 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.8045 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.5081 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.3820 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | full_aug_obs | raw | 0.0000 | 14.5251 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | logvar_hat_9_clamped | raw | 0.0000 | 14.8319 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | logvar_hat_9_clamped | raw | 0.0000 | 14.8408 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | logvar_hat_9_clamped | raw | 0.0000 | 14.7585 | 5.0000 | 0.0000 | 0.0000 |
| n3f_no_z_full | logvar_hat_9_clamped | raw | 0.0000 | 14.7435 | 5.0000 | 0.0000 | 0.0000 |

## Raw Unsafe Diagnostics

| method_key | raw_unsafe_rate | raw_min_predicted_cpa | no_response_rate |
| --- | --- | --- | --- |
| n3f_no_z_full | 0.2277 | 3.5999 | 0.0000 |
| z_l2_scale_4 | 0.2696 | 2.9979 | 0.0000 |
| z_layernorm_alpha_0p5 | 0.2269 | 3.4251 | 0.0000 |

## Breakdown Outputs

Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3fz_noz_full_z_screen/tables/`.

## Remaining Risks

- This stage uses seed 0 only.
- N3F/Z still evaluates no-shield PPO; shield comparisons remain out of scope for this phase.

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/checkpoint_1000k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/checkpoint_1500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip`
### tables
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_attention_reference_comparison.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_aug_feature_block_stats.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_command_manifest.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_config_manifest.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_episode_metrics.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_eval_summary.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_gpsi_output_summary.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_scenario_breakdown.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_schema_check.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_threat_class_breakdown.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_train_curve.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_train_heartbeat.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_winner_recommendation.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_z_hard_gate_decision.csv`
- `results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_z_stats.csv`
### plots
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/aug_feature_block_scale_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/checkpoint_success_collision_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/gpsi_delta_norm_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/gpsi_logvar_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/raw_unsafe_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/scenario_breakdown_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/success_collision_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/train_reward_by_config.png`
- `results/env_v2_phase_n3fz_noz_full_z_screen/plots/z_hard_gate_by_variant.png`
### logs
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_analysis.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_eval.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_train_no_z_full.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_train_z_l2_scale4_sidecar.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_train_z_l2_scale_4.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_train_z_layernorm_alpha0p5_sidecar.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_train_z_layernorm_alpha_0p5.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/logs/phase_n3fz_z_stats.log`
- `results/env_v2_phase_n3fz_noz_full_z_screen/phase_n3fz_watcher.log`
### flags
- `results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag`
