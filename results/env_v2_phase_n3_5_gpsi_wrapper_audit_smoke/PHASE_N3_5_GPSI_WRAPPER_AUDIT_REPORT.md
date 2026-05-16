# Phase N3.5 Gpsi Wrapper Audit Report

## Terminal Decision

`terminal_decision = phase_n3_5_gpsi_wrapper_audit_complete`

Phase N3.5 complete. The audit found and repaired an engineering bug in the online Gpsi normalization path.

## Engineering Facts

- Phase N2 complete flag, Phase N3 artifacts, and `work_dirs/gpsi_heada_v1_nll/best.pth` were present.
- Offline-online equivalence passes after using the same normalization semantics in the N2 eval path and N3 wrapper path.
- Before repair, online 1s delta norm mean/max were `12167.9` / `12922.6`.
- After repair, online 1s delta norm mean/max are `0.85568` / `1.55512`.
- After repair, logvar_xy_1s mean/min/max are `-4.04876` / `-5` / `-2.40472`.
- Degenerate checkpoint std dimensions repaired: `9`.
- Active-mask audit inactive-forwarded count: `0`; duplicate active id count: `0`.
- History left-padding violations: `0`.

## Confirmed Bug

N2 data was collected with a hold-position policy, so the first three `ego_current` dimensions, normalized UAV velocity, have checkpoint std near `1e-6`. N3 online PPO supplies nonzero UAV velocity, so the legacy wrapper divided by near-zero std and pushed normalized Gpsi inputs to extreme scale. This reproduced the N3 symptom: huge `delta_hat` and logvar clamped to `-5`.

Repair action: `GpsiObsWrapper` now floors degenerate checkpoint std dimensions before normalization and exposes raw/normalized inputs plus unclamped logvar diagnostics.

## Offline-Online Equivalence

| component | max_abs_diff | mean_abs_diff | rmse_diff | corr | allclose_pass |
| --- | --- | --- | --- | --- | --- |
| normalized_ego_current | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| normalized_obs_current | 0.0 | 0.0 | 0.0 | 0.9999999999999999 | 1 |
| normalized_history_rel_pos | 0.0 | 0.0 | 0.0 | 0.9999999999999999 | 1 |
| normalized_history_rel_vel | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| normalized_history_valid_mask | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| z | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| delta_hat | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| logvar_hat | 0.0 | 0.0 | 0.0 | 0.9999999999999999 | 1 |

## Online And Feature Scale

| metric | value | status |
| --- | --- | --- |
| before_delta_norm_1s_p95 | 12916.59736328125 | observed |
| after_delta_norm_1s_p95 | 1.2632359862327576 | pass |
| after_delta_norm_1s_max | 1.555120587348938 | pass |
| after_logvar_xy_1s_span | 2.5952842235565186 | pass |
| repaired_degenerate_std_dims | 9 | pass |

| metric | value | status |
| --- | --- | --- |
| obs_i_12_l2_p95 | 1.7017205119132994 | pass |
| obs_i_12_max_abs | 1.0 | pass |
| z_i_64_l2_p95 | 22.375733280181883 | pass |
| z_i_64_max_abs | 9.24125862121582 | pass |
| delta_hat_9_after_scale_l2_p95 | 1.9750725984573365 | pass |
| delta_hat_9_after_scale_max_abs | 1.716282844543457 | pass |
| logvar_hat_9_clamped_l2_p95 | 13.072296476364135 | pass |
| logvar_hat_9_clamped_max_abs | 5.0 | pass |
| full_aug_obs_l2_p95 | 25.8930926322937 | pass |
| full_aug_obs_max_abs | 9.24125862121582 | pass |
| z_p95_to_obs_p95_l2_ratio | 13.148888506388234 | warn |

## Feature-Scale Finding

`z_i` after-fix p95 L2 norm is `22.3757` versus base `obs_i(12)` p95 L2 norm `1.70172`; ratio `13.1489`. This is not a stop condition here, but it remains a PPO input-scale risk because N3 v1 did not normalize `z_i`.

## Artifacts

- Tables: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/tables/`
- Plots: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/plots/`
- Logs: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/logs/` and `phase_n3_5_watcher.log`

## Decision

- N3 original result validity: invalid as a method conclusion. It used the legacy online normalization path that produced severe feature-scale corruption.
- Must rerun N3: yes. At minimum rerun N3-lite or full N3 with repaired wrapper and consider z-block normalization or z ablation.
- Can enter N4: no. N4 shield comparison is blocked until repaired N3/N3-lite establishes a valid no-shield baseline.
- Recommended rerun: repaired Gpsi wrapper, no shield, same PPO backbone, plus an ablation of `z_i` normalization or removing `z_i` while keeping delta/logvar.
