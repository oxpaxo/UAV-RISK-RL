# Phase N0 Design Freeze Report

## Background

Current mainline is Gpsi-HeadA supervised dynamic obstacle representation -> PPO velocity policy -> post-hoc VO/CPA-TTC Safety Shield with sigma2 uncertainty margin.
Old Phase C/D/E safety-cost planning, learned R(s,a), candidate velocity risk maps as PPO input, and full 5-head Gpsi are deprecated for this phase.

## Terminal Decision

`terminal_decision = phase_n0_design_freeze_complete`

Phase N0 complete. Gpsi-HeadA design and dataflow are ready for Phase N1 dataset construction.

## Experiment And Code-Audit Supported Facts

- EnvV2 `info` exposes active obstacle ids, world positions, world velocities, motion modes, threat classes, planned CPA/TTC values, UAV state, dt, and active obstacle count.
- EnvV2 observation exposes ego state, per-obstacle relative position/velocity features, active mask, distance, closing, threat class id, and risk value.
- Phase A and Phase B long obstacle tables preserve obstacle slot, obstacle id, active flag, world position, world velocity, distance, closing, planned CPA/TTC, threat class, motion mode, and risk value.
- Phase A/B per-step traces preserve UAV position/velocity, so relative position/velocity are reconstructable by joining long obstacle tables on scenario/episode/step.
- Small dry-run rows: `3159`; full-history rows: `1990`; replacement-nearby rows: `1196`.

## Design Decisions

- HeadA target is world-frame obstacle residual: `delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]`.
- `sigma2` is not a direct per-sample label. Gpsi outputs `delta_hat` and `log_sigma2_hat`; heteroscedastic uncertainty is learned with Gaussian NLL after delta-only MSE/SmoothL1 warmup.
- N3 first version freezes Gpsi and trains PPO only.
- N4 must run `lambda_uncertainty` sweep with `{0, small, medium, large}`; `lambda=0` is the fixed-margin shield reference.
- Obstacle identity is `episode_id + obstacle_id`; `obstacle_slot` is current ordering only and may be reused after replacement.

## Replacement And Label Rules

- History is built only from the same obstacle id within an episode. Missing early history is padded and marked by `valid_history_mask`.
- Future labels are valid only if the same obstacle id is active at `t+tau`; episode end, inactive obstacle, or replacement before horizon invalidates that horizon.
- Replacement at the same slot creates a new obstacle id. N1 must not join old and new ids even if the slot is unchanged.
- Future information is used only for supervised labels, never as inference input.

## Dry-Run Future Label Counts

| horizon | valid rows |
| --- | ---: |
| 1s | 2784 |
| 2s | 2486 |
| 4s | 1939 |

## Residual Sanity By Motion Mode

| tau | motion_mode | valid_count | mean_delta_norm | median_delta_norm |
| ---: | --- | ---: | ---: | ---: |
| 1.0 | accel_decel | 510 | 0.307872 | 0.169807 |
| 1.0 | ar1_velocity | 557 | 0.262871 | 0.234852 |
| 1.0 | crossing_or_sudden_threat | 536 | 0.063950 | 0.000001 |
| 1.0 | linear | 477 | 0.000001 | 0.000001 |
| 1.0 | sinusoidal_lateral | 704 | 0.352193 | 0.319917 |
| 2.0 | accel_decel | 445 | 1.003845 | 0.740296 |
| 2.0 | ar1_velocity | 492 | 0.623025 | 0.547146 |
| 2.0 | crossing_or_sudden_threat | 483 | 0.202117 | 0.000001 |
| 2.0 | linear | 422 | 0.000002 | 0.000001 |
| 2.0 | sinusoidal_lateral | 644 | 1.058102 | 0.962804 |
| 4.0 | accel_decel | 318 | 2.337265 | 2.131853 |
| 4.0 | ar1_velocity | 362 | 1.324446 | 1.192940 |
| 4.0 | crossing_or_sudden_threat | 397 | 0.623836 | 0.000002 |
| 4.0 | linear | 329 | 0.000003 | 0.000002 |
| 4.0 | sinusoidal_lateral | 533 | 1.938217 | 1.881346 |

## Risks And Unresolved Issues

- The three requested design markdown files were not present under the specified filenames; no spec conflict was detected because the Phase N0 guide and generated YAML define the active freeze.

## Output Artifacts

- `configs/gpsi_head_a_spec.yaml`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_required_fields_check.csv`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_obstacle_id_alignment_check.csv`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_history_future_label_check.csv`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_coordinate_frame_check.csv`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_phase_ab_artifact_check.csv`
- `results/env_v2_phase_n0_design_freeze/tables/phase_n0_spec_freeze_check.csv`
- `results/env_v2_phase_n0_design_freeze/schema/gpsi_head_a_dataset_schema_draft.json`
- `results/env_v2_phase_n0_design_freeze/schema/gpsi_head_a_model_io_schema_draft.json`
- `results/env_v2_phase_n0_design_freeze/logs/phase_n0_design_freeze.log`
- `results/env_v2_phase_n0_design_freeze/phase_n0_watcher.log`
- `results/env_v2_phase_n0_design_freeze/phase_n0_status.txt`
- `results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag`

## N1 Readiness

Can enter Phase N1: yes.
N1 should build only `data/gpsi_head_a_v1/{train,val,test,schema}` after this complete flag, using the schema draft and identity/valid-mask rules above.
