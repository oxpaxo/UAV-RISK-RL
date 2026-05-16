# Phase N2 HeadA Offline Report

## Terminal Decision

`terminal_decision = phase_n2_heada_offline_complete`

Phase N2 complete.
Gpsi-HeadA offline model is ready for Phase N3 frozen-Gpsi PPO integration.

## Experiment-Supported Facts

- Phase N1 complete flag exists and train/val/test datasets were readable.
- Model inputs were restricted to `ego_current`, `obs_current`, `history_rel_pos`, `history_rel_vel`, and `history_valid_mask`.
- `future_pos_world`, `delta_label_world`, and `future_valid_mask` were used only for loss/evaluation, never as model inputs.
- Normalization statistics were computed from the train split only and stored in checkpoints.
- Delta-only training and Gaussian NLL training both completed without NaN/inf.
- Diagonal logvar was clamped to [-5, 3] during NLL loss/evaluation.

## Dataset Summary

| split | samples | valid labels |
| --- | ---: | ---: |
| train | 228957 | 563471 |
| val | 49599 | 121725 |
| test | 48096 | 118325 |

## Model Structure

GRU history encoder over relative position/velocity history, MLP current encoder over ego + current obstacle profile, fusion MLP to `z_i`, HeadA delta output `[T,D]`, and diagonal logvar output `[T,D]`.

## Core Metrics

- Zero residual test MSE: `0.466866`
- Delta-only test MSE: `0.099426`
- NLL model test MSE: `0.096426`
- NLL model test Gaussian NLL: `-1.793691`
- Mean nonlinear test MSE improvement over zero: `0.6692`

## Projected Uncertainty Calibration

- Mean corr(projected_std, |projected_error|) over test radial/relative-velocity directions: `0.6042`
- Directional projected uncertainty tables were generated for x-axis, y-axis, radial_xy, rel_velocity_xy, and error_direction_diag_only.

## Reasonable Inferences

- The offline HeadA residual predictor is learnable on nonlinear/stochastic obstacle residuals if nonlinear-mode MSE improves over the zero residual baseline.
- Diagonal logvar provides a usable starting point for N4 directional/tube shield experiments when projected calibration is finite and not collapsed.

## Risks / Warnings

- No blocking warning.

## Plots

- `results/env_v2_phase_n2_gpsi_heada_offline/plots/delta_loss_curve.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/nll_loss_curve.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/per_horizon_mse_bar.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/per_motion_mode_error_bar.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/logvar_by_motion_mode.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/projected_uncertainty_reliability.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/zscore_histogram.png`
- `results/env_v2_phase_n2_gpsi_heada_offline/plots/predicted_vs_error_scatter.png`

## Output Artifacts

- `work_dirs/gpsi_heada_v1_delta_only/best.pth`
- `work_dirs/gpsi_heada_v1_delta_only/last.pth`
- `work_dirs/gpsi_heada_v1_nll/best.pth`
- `work_dirs/gpsi_heada_v1_nll/last.pth`
- `results/env_v2_phase_n2_gpsi_heada_offline/tables`
- `results/env_v2_phase_n2_gpsi_heada_offline/logs/phase_n2_train_delta_only.log`
- `results/env_v2_phase_n2_gpsi_heada_offline/logs/phase_n2_train_nll.log`
- `results/env_v2_phase_n2_gpsi_heada_offline/logs/phase_n2_eval.log`
- `results/env_v2_phase_n2_gpsi_heada_offline/phase_n2_watcher.log`
- `results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag`

## Phase N3 Readiness

Can enter Phase N3: yes.
N3 should use frozen Gpsi, trainable PPO, augmented obstacle input `[obs_i, z_i, delta_hat_i, log_sigma2_i]`, no shield, masked-attention PPO backbone, and symmetric critic.
