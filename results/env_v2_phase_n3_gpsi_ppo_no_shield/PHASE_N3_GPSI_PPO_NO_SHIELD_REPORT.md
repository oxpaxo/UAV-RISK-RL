# Phase N3 Gpsi-PPO No-Shield Report

## Terminal Decision

`terminal_decision = phase_n3_gpsi_ppo_no_shield_complete`

Phase N3 complete.
Gpsi-PPO no-shield raw policy evaluation is ready for Phase N4 shield fair comparison.

## Background And Goal

N3 trains a raw PPO velocity policy with frozen Gpsi-HeadA obstacle augmentation. It does not use a safety shield, action filtering, action projection, candidate velocity risk map, learned R(s,a), safety-cost PPO, or Gpsi fine-tuning.

## Experiment-Supported Facts

- Phase N2 complete flag exists.
- Gpsi NLL checkpoint exists and was loaded by wrapper.
- Gpsi freeze/schema check CSV was generated.
- Formal PPO checkpoints exist through 1500000 steps.
- Evaluation completed for 6 scenarios with required episodes.
- Attention reference comparison, scenario/motion/threat breakdowns, raw unsafe diagnostics, Gpsi diagnostics, plots, and sampled traces were generated.
- No safety shield, action filtering, action projection, dense safety cost, or Gpsi training was used.

## Gpsi Frozen And Wrapper

- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`.
- Wrapper: `envs/wrappers/gpsi_obs_wrapper.py`.
- Online inputs: `ego_current`, `obs_current`, `history_rel_pos`, `history_rel_vel`, `history_valid_mask`.
- Histories are keyed by `obstacle_id`; replacement creates a new left-padded history.
- Gpsi is set to `eval()`, all parameters use `requires_grad=False`, and forward runs under `torch.no_grad()`.

## Observation Schema

`obs_i_aug = [obs_i(12), z_i(64), delta_hat_i(9), logvar_hat_i(9)]`, so obstacle profile dim is `94`.

## Augmentation Normalization

- Gpsi input normalization uses train-split-only stats stored in the N2 checkpoint.
- `delta_hat` is divided by `delta_scale=5.0` for PPO input, while raw diagnostics keep unscaled values.
- `logvar_hat` is clamped to `[-5, 3]` before PPO input.
- `z_i` is not additionally normalized in N3 v1.

## PPO Backbone

Masked attention over active obstacles is used for both actor and critic through SB3 `MultiInputPolicy`; actor/critic MLP heads are symmetric with `pi=[128,128]`, `vf=[128,128]`.

## Checkpoint Eval Summary

| method | checkpoint_step | scenario | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | 1500000.0000 | eval_flow_high_density | 0.6400 | 0.3600 | 0.9600 | 0.9425 | 0.3186 |
| attention_full | 1500000.0000 | eval_flow_high_speed | 0.5400 | 0.4600 | 1.0000 | 0.9578 | 0.2671 |
| attention_full | 1500000.0000 | eval_flow_high_threat | 0.6200 | 0.3800 | 0.9600 | 0.9687 | 0.2991 |
| attention_full | 1500000.0000 | eval_flow_id | 0.5800 | 0.4200 | 0.9200 | 0.9634 | 0.2773 |
| attention_full | 1500000.0000 | eval_flow_mixed_ood | 0.5400 | 0.4600 | 0.9800 | 0.9403 | 0.3114 |
| attention_full | 1500000.0000 | eval_flow_sudden_threat | 0.7400 | 0.2600 | 0.9800 | 0.9707 | 0.2913 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | 0.2400 | 0.7600 | 1.0000 | 0.9195 | 0.3038 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_speed | 0.2000 | 0.8000 | 0.9800 | 0.8992 | 0.2731 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_threat | 0.2800 | 0.7200 | 1.0000 | 0.9241 | 0.2897 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_id | 0.4600 | 0.5400 | 0.9200 | 0.9560 | 0.2719 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_mixed_ood | 0.2600 | 0.7400 | 0.9400 | 0.9172 | 0.2745 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_sudden_threat | 0.3000 | 0.7000 | 0.9800 | 0.9370 | 0.2724 |

## Attention Reference Comparison

| scenario | success_rate_gpsi | success_rate_attention | delta_success_rate | collision_rate_gpsi | collision_rate_attention | delta_collision_rate | raw_unsafe_action_rate_gpsi | raw_unsafe_action_rate_attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval_flow_high_density | 0.1800 | 0.6400 | -0.4600 | 0.8200 | 0.3600 | 0.4600 | 0.3918 | 0.3186 |
| eval_flow_high_speed | 0.1600 | 0.5400 | -0.3800 | 0.8400 | 0.4600 | 0.3800 | 0.3464 | 0.2671 |
| eval_flow_high_threat | 0.2600 | 0.6200 | -0.3600 | 0.7400 | 0.3800 | 0.3600 | 0.3660 | 0.2991 |
| eval_flow_id | 0.4800 | 0.5800 | -0.1000 | 0.5200 | 0.4200 | 0.1000 | 0.3405 | 0.2773 |
| eval_flow_mixed_ood | 0.1800 | 0.5400 | -0.3600 | 0.8200 | 0.4600 | 0.3600 | 0.3602 | 0.3114 |
| eval_flow_sudden_threat | 0.3400 | 0.7400 | -0.4000 | 0.6600 | 0.2600 | 0.4000 | 0.3327 | 0.2913 |

## Motion-Mode Breakdown

| method | checkpoint_step | threat_motion_mode | success_rate | collision_rate | near_miss_rate | progress |
| --- | --- | --- | --- | --- | --- | --- |
| attention_full | 1500000.0000 | accel_decel | 0.5000 | 0.5000 | 0.9483 | 0.9525 |
| attention_full | 1500000.0000 | ar1_velocity | 0.6706 | 0.3294 | 0.9765 | 0.9631 |
| attention_full | 1500000.0000 | crossing_or_sudden_threat | 0.6939 | 0.3061 | 0.9796 | 0.9679 |
| attention_full | 1500000.0000 | linear | 0.5278 | 0.4722 | 0.9167 | 0.9380 |
| attention_full | 1500000.0000 | sinusoidal_lateral | 0.6111 | 0.3889 | 0.9861 | 0.9564 |
| gpsi_heada_ppo_no_shield | 500000.0000 | accel_decel | 0.3000 | 0.7000 | 0.9667 | 0.9179 |
| gpsi_heada_ppo_no_shield | 500000.0000 | ar1_velocity | 0.2586 | 0.7414 | 0.9655 | 0.9137 |
| gpsi_heada_ppo_no_shield | 500000.0000 | crossing_or_sudden_threat | 0.3571 | 0.6429 | 0.9762 | 0.9243 |
| gpsi_heada_ppo_no_shield | 500000.0000 | linear | 0.3171 | 0.6829 | 0.9756 | 0.9370 |
| gpsi_heada_ppo_no_shield | 500000.0000 | sinusoidal_lateral | 0.2626 | 0.7374 | 0.9697 | 0.9327 |
| gpsi_heada_ppo_no_shield | 1000000.0000 | accel_decel | 0.2353 | 0.7647 | 0.9559 | 0.8757 |
| gpsi_heada_ppo_no_shield | 1000000.0000 | ar1_velocity | 0.1781 | 0.8219 | 0.9863 | 0.9059 |

## Raw Action Unsafe Diagnostics

| method | checkpoint_step | scenario | motion_mode | threat_class | raw_unsafe_rate | raw_min_predicted_cpa |
| --- | --- | --- | --- | --- | --- | --- |
| attention_full | 1500000.0000 | eval_flow_high_density | accel_decel | high | 0.3652 | 2.2369 |
| attention_full | 1500000.0000 | eval_flow_high_density | accel_decel | low | 0.2748 | 2.4317 |
| attention_full | 1500000.0000 | eval_flow_high_density | accel_decel | medium | 0.2398 | 2.8094 |
| attention_full | 1500000.0000 | eval_flow_high_density | ar1_velocity | high | 0.4235 | 2.5045 |
| attention_full | 1500000.0000 | eval_flow_high_density | ar1_velocity | low | 0.2582 | 2.9911 |
| attention_full | 1500000.0000 | eval_flow_high_density | ar1_velocity | medium | 0.3452 | 2.6973 |
| attention_full | 1500000.0000 | eval_flow_high_density | crossing_or_sudden_threat | high | 0.4042 | 2.1303 |
| attention_full | 1500000.0000 | eval_flow_high_density | crossing_or_sudden_threat | low | 0.0909 | 4.1188 |
| attention_full | 1500000.0000 | eval_flow_high_density | crossing_or_sudden_threat | medium | 0.1596 | 3.5780 |
| attention_full | 1500000.0000 | eval_flow_high_density | linear | high | 0.5758 | 1.8453 |
| attention_full | 1500000.0000 | eval_flow_high_density | linear | low | 0.1111 | 4.0060 |
| attention_full | 1500000.0000 | eval_flow_high_density | linear | medium | 0.3480 | 2.8902 |

## Gpsi Output Diagnostics

| method | checkpoint_step | scenario | motion_mode | threat_class | mean_delta_norm_1s | mean_logvar_xy_1s | history_valid_ratio_nearest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | accel_decel | high | 20340.2734 | -5.0000 | 0.7862 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | accel_decel | medium | 27913.0918 | -5.0000 | 0.7875 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | ar1_velocity | high | 22925.2277 | -5.0000 | 0.7771 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | ar1_velocity | medium | 17402.5293 | -5.0000 | 0.2167 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | crossing_or_sudden_threat | high | 20917.5622 | -5.0000 | 0.7591 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | crossing_or_sudden_threat | medium | 35527.6992 | -5.0000 | 0.7000 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | linear | high | 20656.0462 | -5.0000 | 0.7076 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | linear | medium | 15962.6008 | -5.0000 | 0.5500 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | sinusoidal_lateral | high | 22240.6792 | -5.0000 | 0.7300 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_density | sinusoidal_lateral | medium | 16203.9879 | -5.0000 | 0.7261 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_speed | accel_decel | high | 20704.9209 | -5.0000 | 0.6180 |
| gpsi_heada_ppo_no_shield | 500000.0000 | eval_flow_high_speed | accel_decel | medium | 16376.2252 | -5.0000 | 0.4962 |

## Phase B Context

Phase B geometry/filter baselines remain background upper-bound context only. N3 direct comparison is `attention_full_1500k` versus `Gpsi-HeadA + PPO no shield`.

## Reasonable Inferences

- If no-shield Gpsi-PPO does not outperform attention_full, N4 can still proceed because N2 supports testing Gpsi uncertainty on the shield side.

## Risks / Warnings

- No blocking warning.

## Output Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1000k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0_smoke/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0_smoke/checkpoint_2k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_s0_smoke/final.zip`
### tables
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_attention_reference_comparison.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_command_manifest.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_episode_metrics.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_eval_summary.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_gpsi_forward_profile.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_gpsi_output_steps.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_gpsi_output_summary.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_scenario_breakdown.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_schema_check.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_threat_class_breakdown.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_train_curve.csv`
### plots
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/checkpoint_success_collision.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/gpsi_delta_norm_distribution.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/gpsi_logvar_distribution.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/motion_mode_breakdown.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/raw_unsafe_rate_by_checkpoint.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/scenario_breakdown.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/train_reward_curve.png`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/plots/train_success_collision_curve.png`
### traces
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep0.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep1.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep10.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep11.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep12.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep13.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep14.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep15.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep16.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep17.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep18.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep19.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep2.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep20.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep21.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep22.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep23.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep24.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep25.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep26.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep27.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep28.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep29.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep3.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep30.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep31.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep32.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep33.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep34.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep35.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep36.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep37.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep38.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep39.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep4.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep40.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep41.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep42.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep43.csv`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/traces/attention_full_attention_full_1500k_eval_flow_high_density_ep44.csv`
### logs
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/logs/phase_n3_analysis.log`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/logs/phase_n3_eval.log`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/logs/phase_n3_formal_train.log`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/logs/phase_n3_smoke.log`
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/phase_n3_watcher.log`
### flags
- `results/env_v2_phase_n3_gpsi_ppo_no_shield/PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag`

## N4 Readiness

Can enter Phase N4: yes.
