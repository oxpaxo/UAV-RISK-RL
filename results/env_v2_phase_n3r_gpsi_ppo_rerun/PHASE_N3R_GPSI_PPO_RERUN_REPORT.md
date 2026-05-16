# Phase N3R Gpsi-PPO Rerun Report

## Terminal Decision

`terminal_decision = phase_n3r_gpsi_ppo_rerun_complete`

Phase N3R complete. This is a repaired no-shield PPO screening rerun with z-block ablation; it does not enter N4.

## Engineering Facts

- Phase N3.5 complete flag exists and repaired `GpsiObsWrapper` was used.
- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`; Gpsi was frozen under `eval()` with no trainable parameters.
- EnvV2 core was not modified for this phase.
- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.
- Training budget used: `500000` PPO steps per A/B/C config, seed 0.
- Evaluation used `50` episodes per scenario over six scenarios.

## N3.5 Repair Summary

The original N3 no-shield result is invalid because online normalization divided nonzero PPO UAV velocities by near-zero N2 hold-position velocity std. N3.5 fixed this by flooring degenerate checkpoint std dimensions and verified offline-online equivalence.

## Configs

- A `repaired-full-raw-z`: `[obs_i, z_i_raw, delta_hat_scaled, logvar_hat]`, dim 94.
- B `repaired-full-z-normalized`: `[obs_i, z_i_normalized, delta_hat_scaled, logvar_hat]`, dim 94. z stats are from N1 train split frozen-Gpsi forward only.
- C `repaired-no-z`: `[obs_i, delta_hat_scaled, logvar_hat]`, dim 30.

## Experiment-Supported Facts

| method_key | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- |
| no_z | 0.4233 | 0.5767 | 0.4033 | 0.9403 | 0.2650 | 5.3996 |
| raw_z | 0.3733 | 0.6267 | 0.3433 | 0.9263 | 0.2634 | 5.8144 |
| z_norm | 0.3667 | 0.6333 | 0.3367 | 0.9424 | 0.2361 | 5.3529 |

## Attention Reference Comparison

| method_key | scenario | success_rate_config | success_rate_attention | delta_success_rate | collision_rate_config | collision_rate_attention | delta_collision_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_z | eval_flow_high_density | 0.3400 | 0.6400 | -0.3000 | 0.6600 | 0.3600 | 0.3000 |
| no_z | eval_flow_high_speed | 0.3800 | 0.5400 | -0.1600 | 0.6200 | 0.4600 | 0.1600 |
| no_z | eval_flow_high_threat | 0.4600 | 0.6200 | -0.1600 | 0.5400 | 0.3800 | 0.1600 |
| no_z | eval_flow_id | 0.3400 | 0.5800 | -0.2400 | 0.6600 | 0.4200 | 0.2400 |
| no_z | eval_flow_mixed_ood | 0.5200 | 0.5400 | -0.0200 | 0.4800 | 0.4600 | 0.0200 |
| no_z | eval_flow_sudden_threat | 0.5000 | 0.7400 | -0.2400 | 0.5000 | 0.2600 | 0.2400 |
| raw_z | eval_flow_high_density | 0.3800 | 0.6400 | -0.2600 | 0.6200 | 0.3600 | 0.2600 |
| raw_z | eval_flow_high_speed | 0.2000 | 0.5400 | -0.3400 | 0.8000 | 0.4600 | 0.3400 |
| raw_z | eval_flow_high_threat | 0.2600 | 0.6200 | -0.3600 | 0.7400 | 0.3800 | 0.3600 |
| raw_z | eval_flow_id | 0.4000 | 0.5800 | -0.1800 | 0.6000 | 0.4200 | 0.1800 |
| raw_z | eval_flow_mixed_ood | 0.4800 | 0.5400 | -0.0600 | 0.5200 | 0.4600 | 0.0600 |
| raw_z | eval_flow_sudden_threat | 0.5200 | 0.7400 | -0.2200 | 0.4800 | 0.2600 | 0.2200 |
| z_norm | eval_flow_high_density | 0.3600 | 0.6400 | -0.2800 | 0.6400 | 0.3600 | 0.2800 |
| z_norm | eval_flow_high_speed | 0.2400 | 0.5400 | -0.3000 | 0.7600 | 0.4600 | 0.3000 |
| z_norm | eval_flow_high_threat | 0.4200 | 0.6200 | -0.2000 | 0.5800 | 0.3800 | 0.2000 |
| z_norm | eval_flow_id | 0.4400 | 0.5800 | -0.1400 | 0.5600 | 0.4200 | 0.1400 |
| z_norm | eval_flow_mixed_ood | 0.3400 | 0.5400 | -0.2000 | 0.6600 | 0.4600 | 0.2000 |
| z_norm | eval_flow_sudden_threat | 0.4000 | 0.7400 | -0.3400 | 0.6000 | 0.2600 | 0.3400 |

## Gpsi Output Diagnostics

| method_key | delta_norm_1s_p95 | delta_norm_1s_max | logvar_xy_1s_mean | logvar_xy_1s_span | projected_std_radial_mean | projected_std_relvel_mean | z_norm_raw_p95 | z_norm_after_p95 | history_valid_ratio_mean | inactive_forwarded_count_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_z | 1.8575 | 2.9764 | -4.4558 | 5.6174 | 0.1360 | 0.1381 | 44.1654 | 44.1654 | 0.5927 | 0.0000 |
| raw_z | 1.9108 | 2.7236 | -4.5296 | 5.6735 | 0.1245 | 0.1237 | 41.9016 | 41.9016 | 0.6290 | 0.0000 |
| z_norm | 1.9513 | 2.8220 | -4.5655 | 5.6594 | 0.1212 | 0.1199 | 50.0304 | 44.3532 | 0.6098 | 0.0000 |

## Augmented Feature Block Stats

| method_key | block | not_applicable | l2_norm_p95 | max_abs_p95 | nan_count | inf_count |
| --- | --- | --- | --- | --- | --- | --- |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.1451 | 1.5561 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.0957 | 1.5012 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.1057 | 1.5528 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.0603 | 1.5445 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.0701 | 1.5012 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.0753 | 1.5520 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.2459 | 1.5245 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.0090 | 1.4315 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 1.4452 | 1.1775 | 0.0000 | 0.0000 |
| no_z | delta_hat_9_after_scale | 0.0000 | 2.1563 | 1.6649 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.5262 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.4263 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.5413 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.5609 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.4945 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 15.0265 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.1093 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.3865 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 14.5625 | 5.0000 | 0.0000 | 0.0000 |
| no_z | full_aug_obs | 0.0000 | 13.7932 | 5.0000 | 0.0000 | 0.0000 |
| no_z | logvar_hat_9_clamped | 0.0000 | 14.4066 | 5.0000 | 0.0000 | 0.0000 |
| no_z | logvar_hat_9_clamped | 0.0000 | 14.3151 | 5.0000 | 0.0000 | 0.0000 |
| no_z | logvar_hat_9_clamped | 0.0000 | 14.4427 | 5.0000 | 0.0000 | 0.0000 |
| no_z | logvar_hat_9_clamped | 0.0000 | 14.4566 | 5.0000 | 0.0000 | 0.0000 |

## Raw Unsafe Diagnostics

| method_key | raw_unsafe_rate | raw_min_predicted_cpa | no_response_rate |
| --- | --- | --- | --- |
| no_z | 0.2415 | 3.4166 | 0.0000 |
| raw_z | 0.2530 | 3.6496 | 0.0000 |
| z_norm | 0.2256 | 3.3754 | 0.0000 |

## Breakdown Outputs

Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/`.

## Winner Recommendation

- Winner recommendation: `no_z` / `gpsi_no_z_repaired`.
- Recommendation: Winner for the next N3-full candidate is gpsi_no_z_repaired; run repaired no-shield PPO to 1.5M before N4.
- Need full N3 1.5M rerun: yes.
- Can enter N4: no.
- Next step before N4: run the recommended repaired no-shield N3-full 1.5M baseline if the winner is promising; otherwise repair/ablate and rerun N3R.

## Reasonable Inferences

- N3R is a screening comparison. A promising winner should be used for a full repaired N3 no-shield rerun before any shield comparison.
- If z-normalized or no-z wins over raw-z, raw latent scale is a likely PPO input conditioning problem rather than a Gpsi HeadA output-scale failure.

## Remaining Risks

- The run uses seed 0 only; the selected winner still needs longer-run and multi-seed confirmation before method-level claims.
- The attention reference is a 1.5M baseline; N3R A/B/C are 500k screening runs and are not a final compute-matched comparison.

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0/final.zip`
### tables
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_attention_reference_comparison.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_aug_feature_block_stats.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_command_manifest.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_config_manifest.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_episode_metrics.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_eval_summary.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_gpsi_output_summary.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_scenario_breakdown.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_schema_check.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_threat_class_breakdown.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_train_curve.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_train_heartbeat.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_winner_recommendation.csv`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_z_stats.csv`
### plots
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/aug_feature_block_scale_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/checkpoint_success_collision_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/gpsi_delta_norm_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/gpsi_logvar_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/raw_unsafe_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/scenario_breakdown_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/success_collision_by_config.png`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/plots/train_reward_by_config.png`
### logs
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_analysis.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_eval.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_train_no_z.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_train_raw_z.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_train_z_norm.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/logs/phase_n3r_z_stats.log`
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/phase_n3r_watcher.log`
### flags
- `results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag`
