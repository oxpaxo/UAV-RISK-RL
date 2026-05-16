# Phase N1 Gpsi-HeadA Dataset Report

## Terminal Decision

`terminal_decision = phase_n1_gpsi_dataset_complete`

Phase N1 complete.
Gpsi-HeadA dataset is ready for Phase N2 offline Head A pilot.

## Background And Goal

N1 builds the supervised HeadA dataset for the new mainline: Gpsi-HeadA -> PPO velocity policy -> post-hoc VO/CPA-TTC Safety Shield.
No Gpsi/PPO training, shield implementation, safety-cost PPO, learned R(s,a), candidate velocity risk map, or 5-head Gpsi route is used.

## Phase N0 Dependency

- Phase N0 complete flag: `results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag`
- EnvV2-core action: `read_only`

## Spec Refinement

- `configs/gpsi_head_a_spec.yaml` now includes `uncertainty.type=diagonal_logvar`, dimensions `[x,y,z]`, `sigma2_direct_label=false`, and future shield usage for scalar margin, directional margin, trajectory tube, and candidate scoring.
- Reserved shield versions: V0 fixed margin, V1 scalar sigma2 margin, V2 directional sigma2 margin, V3 predicted-trajectory directional sigma2 tube, V4 V3 plus uncertainty-aware candidate scoring.

## Dataset Source And Split

- scenario: `train_flow_mixed`
- episodes: `100`
- split: episode-level `{'row_level_random_split': False, 'test': 0.15, 'train': 0.7, 'type': 'episode_level', 'val': 0.15}`
- row-level random split: forbidden and not used.

## Sample Definition

Each sample is `(episode_id, step, obstacle_id)`. Inputs include current ego state, current obstacle observation, same-id obstacle history, valid history mask, motion mode, threat class, planned CPA/TTC, distance, closing, and risk value.

## Label Definition

`delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]`, with `tau in {1s,2s,4s}`.
`sigma2` is not a direct label; later HeadA predicts diagonal `logvar_hat` and trains it through Gaussian NLL.

## Validity And Identity Rules

- History/future are keyed by `episode_id + obstacle_id`.
- `obstacle_slot` is current slot metadata only; slot reuse after replacement is never joined.
- `future_valid_mask` marks invalid horizons caused by episode end, inactive obstacle, or replacement before the horizon.
- Labels use world-frame residuals to avoid future UAV motion leakage.

## Dataset Files

- `data/gpsi_head_a_v1/train.npz`
- `data/gpsi_head_a_v1/val.npz`
- `data/gpsi_head_a_v1/test.npz`
- `data/gpsi_head_a_v1/schema.json`
- `data/gpsi_head_a_v1/dataset_manifest.json`

## Split Summary

| split | samples | episodes | valid 1s | valid 2s | valid 4s | full history rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 228957 | 70 | 209016 | 192966 | 161489 | 0.7189 |
| val | 49599 | 15 | 45257 | 41720 | 34748 | 0.7144 |
| test | 48096 | 15 | 43908 | 40528 | 33889 | 0.7182 |

## Label Validity By Horizon

| split | horizon | valid labels | valid rate |
| --- | ---: | ---: | ---: |
| train | 1.0 | 209016 | 0.9129 |
| train | 2.0 | 192966 | 0.8428 |
| train | 4.0 | 161489 | 0.7053 |
| val | 1.0 | 45257 | 0.9125 |
| val | 2.0 | 41720 | 0.8411 |
| val | 4.0 | 34748 | 0.7006 |
| test | 1.0 | 43908 | 0.9129 |
| test | 2.0 | 40528 | 0.8426 |
| test | 4.0 | 33889 | 0.7046 |

## Residual Sanity

- Linear residual max mean across split/horizon: `0.00000289`.
- Linear mode delta ~= 0 sanity passed.

## Leakage Check

- `train` vs `val`: episode_id_overlap=0, episode_seed_overlap=0, status=pass
- `train` vs `test`: episode_id_overlap=0, episode_seed_overlap=0, status=pass
- `val` vs `test`: episode_id_overlap=0, episode_seed_overlap=0, status=pass

## Plots

- `results/env_v2_phase_n1_gpsi_dataset/plots/residual_norm_by_motion_mode_1s.png`
- `results/env_v2_phase_n1_gpsi_dataset/plots/residual_norm_by_motion_mode_2s.png`
- `results/env_v2_phase_n1_gpsi_dataset/plots/residual_norm_by_motion_mode_4s.png`
- `results/env_v2_phase_n1_gpsi_dataset/plots/residual_norm_by_horizon.png`
- `results/env_v2_phase_n1_gpsi_dataset/plots/valid_label_rate_by_horizon.png`
- `results/env_v2_phase_n1_gpsi_dataset/plots/history_valid_length_distribution.png`

## Risks / Warnings

- No blocking warning.

## Output Tables And Logs

- `results/env_v2_phase_n1_gpsi_dataset/tables`
- `results/env_v2_phase_n1_gpsi_dataset/logs/phase_n1_dataset_build.log`
- `results/env_v2_phase_n1_gpsi_dataset/logs/phase_n1_dataset_inspect.log`
- `results/env_v2_phase_n1_gpsi_dataset/phase_n1_watcher.log`
- `results/env_v2_phase_n1_gpsi_dataset/phase_n1_status.txt`
- `results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag`

## Phase N2 Readiness

Can enter Phase N2: yes.
Phase N2 may start delta-only MSE/SmoothL1 warmup, Gaussian NLL with diagonal logvar, and direction-aware projected uncertainty calibration.
