# Phase N3P No-Z Representation Ablation Report

## Terminal Decision

`terminal_decision = phase_n3p_noz_representation_ablation_complete`

P1/P2/P3 500k no-shield representation screening completed with frozen Gpsi and EnvV2 core unchanged.

## Winner Recommendation

| method_key | success_rate_500k | collision_rate_500k | hard_gate_pass | clearly_better_than_n3r_noz_500k | not_worse_high_speed_high_threat | train_curve_still_improving | promote_to_1p5m | winner_if_any | overall_promote_to_1p5m | can_enter_N4_now |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_projected | 0.5333 | 0.4667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | block_projected | yes | no |
| obs_delta_only | 0.4667 | 0.5333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | block_projected | yes | no |
| logvar_scaled | 0.3467 | 0.6533 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | block_projected | yes | no |
- promote_to_1p5m: yes
- can_enter_N4_now: no
- recommendation: block_projected passes the hard gate, improves clearly over N3R no_z 500k, and merits a 1.5M continuation before N4.

## Reference Comparison

| method_key | success_rate | collision_rate | near_miss_rate | raw_unsafe_action_rate | hard_gate_pass | delta_success_vs_n3r_noz_500k | delta_collision_vs_n3r_noz_500k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| block_projected | 0.5333 | 0.4667 | 0.4700 | 0.2465 | 1.0000 | 0.1100 | -0.1100 |
| obs_delta_only | 0.4667 | 0.5333 | 0.4300 | 0.2283 | 1.0000 | 0.0434 | -0.0434 |
| logvar_scaled | 0.3467 | 0.6533 | 0.3133 | 0.3000 | 0.0000 | -0.0766 | 0.0766 |

## Diagnostics

| method_key | checkpoint_label | diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | logvar_xy_1s_span_max | feature_nonfinite_count | full_aug_obs_l2_p95_max | logvar_raw_l2_p95_max | logvar_scaled_l2_p95_max | adapter_output_l2_p95_max | feature_scale_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| obs_delta_only | 500k | 1.0000 | 1.5496 | 2.2469 | 0.0000 | 5.2751 | 0.0000 | 3.7072 | 15.0000 | nan | nan | 1.0000 |
| logvar_scaled | 500k | 1.0000 | 1.5267 | 2.2538 | 0.0000 | 5.2402 | 0.0000 | 4.5560 | 14.6251 | 2.9250 | nan | 1.0000 |
| block_projected | 500k | 1.0000 | 2.0861 | 2.7372 | 0.0000 | 5.5289 | 0.0000 | 5.1628 | 14.9807 | 2.9961 | 5.8895 | 1.0000 |

## Checkpoint Eval Aggregate

| method_key | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_projected | 250k | 0.2833 | 0.7167 | 0.2633 | 0.9013 | 0.2765 | 0.0684 | 5.1252 |
| block_projected | 500k | 0.5333 | 0.4667 | 0.4700 | 0.9282 | 0.2465 | 0.0911 | 5.1054 |
| block_projected | best_by_eval | 0.3967 | 0.6033 | 0.3767 | 0.9104 | 0.2791 | 0.0942 | 4.7256 |
| block_projected | final | 0.3967 | 0.6033 | 0.3767 | 0.9104 | 0.2791 | 0.0942 | 4.7256 |
| logvar_scaled | 250k | 0.3100 | 0.6900 | 0.2833 | 0.9020 | 0.2940 | 0.0753 | 5.8271 |
| logvar_scaled | 500k | 0.3467 | 0.6533 | 0.3133 | 0.8933 | 0.3000 | 0.0933 | 4.9373 |
| logvar_scaled | best_by_eval | 0.3567 | 0.6433 | 0.3100 | 0.9042 | 0.2998 | 0.0983 | 5.0386 |
| logvar_scaled | final | 0.3567 | 0.6433 | 0.3100 | 0.9042 | 0.2998 | 0.0983 | 5.0386 |
| obs_delta_only | 250k | 0.2833 | 0.7167 | 0.2533 | 0.8819 | 0.2854 | 0.0770 | 4.5520 |
| obs_delta_only | 500k | 0.4667 | 0.5333 | 0.4300 | 0.9508 | 0.2283 | 0.0974 | 5.5379 |
| obs_delta_only | best_by_eval | 0.4400 | 0.5600 | 0.4067 | 0.9404 | 0.2470 | 0.1009 | 5.4098 |
| obs_delta_only | final | 0.4400 | 0.5600 | 0.4067 | 0.9404 | 0.2470 | 0.1009 | 5.4098 |

## Interpretation

- whether logvar was harmful: yes.
- Gpsi output diagnostics remained bounded; no wrapper-scale regression was detected.
- N4 was not executed in this phase.

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/final.zip`
### tables
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_command_manifest.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_config_manifest.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_diagnostics_decision.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_episode_metrics.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_eval_summary.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_feature_block_stats.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_gpsi_output_summary.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_reference_comparison.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_resource_affinity.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_scenario_breakdown.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_scenario_gate_check.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_schema_check.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_threat_class_breakdown.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_train_curve.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_train_heartbeat.csv`
- `results/env_v2_phase_n3p_noz_representation_ablation/tables/phase_n3p_winner_recommendation.csv`
### plots
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_checkpoint_success_collision.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_feature_block_scale.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_gpsi_delta_norm.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_gpsi_logvar.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_raw_unsafe_by_variant.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_scenario_breakdown.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_success_collision_by_variant.png`
- `results/env_v2_phase_n3p_noz_representation_ablation/plots/n3p_train_reward.png`
### logs
- `results/env_v2_phase_n3p_noz_representation_ablation/logs/phase_n3p_analysis.log`
- `results/env_v2_phase_n3p_noz_representation_ablation/logs/phase_n3p_eval.log`
- `results/env_v2_phase_n3p_noz_representation_ablation/logs/phase_n3p_train_p1_obs_delta_only.log`
- `results/env_v2_phase_n3p_noz_representation_ablation/logs/phase_n3p_train_p2_logvar_scaled.log`
- `results/env_v2_phase_n3p_noz_representation_ablation/logs/phase_n3p_train_p3_block_projected.log`
- `results/env_v2_phase_n3p_noz_representation_ablation/phase_n3p_watcher.log`
### flags
- `results/env_v2_phase_n3p_noz_representation_ablation/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag`
