# Phase N3Z2C-Audit Report

## Terminal Decision

`terminal_decision = phase_n3z2c_audit_complete`

Phase N3Z2C-Audit complete. This stage audited parent selection, resume semantics, CPU affinity, and ran a corrected short continuation sanity from the fixed 500k parent.

## Final Decision

| fixed_parent_candidate | fixed_parent_path | old_parent_selection_wrong | cpu_affinity_finding | resume_reset_num_timesteps_false | optimizer_state_restored | diagnostics_ok | corrected_parent_success | corrected_parent_collision | corrected_750k_success | corrected_750k_collision | old_750k_success | old_750k_collision | noz_success | noz_collision | attention_success | attention_collision | corrected_minus_old_success | corrected_minus_old_collision | corrected_minus_noz_success | corrected_minus_noz_collision | corrected_beats_old_750_gate | corrected_still_degraded_vs_parent | need_corrected_z2_full_rerun | final_decision | can_enter_n4 | selected_n4_candidate_if_yes | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoint_500k | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip | 1.0000 | {"OMP_NUM_THREADS": "1", "affinity_count": 16, "cpu_max": "max 100000", "cpuset_cpus_effective": "0-15", "finding": "nproc_reports_omp_thread_limit_not_affinity_limit", "nproc": 1, "nproc_all": 16} | 1.0000 | 1.0000 | 1.0000 | 0.4800 | 0.5200 | 0.4800 | 0.5200 | 0.4667 | 0.5333 | 0.5633 | 0.4367 | 0.6100 | 0.3900 | 0.0133 | -0.0133 | -0.0833 | 0.0833 | 1.0000 | 0.0000 | yes | rerun_corrected_z2_full | no |  | Corrected parent materially changes short-continuation behavior; rerun corrected Z2 full before final N4 selection. |
- Can enter N4: no.
- Selected N4 candidate if yes: ``.
- Need corrected Z2 full rerun: yes.
- Decision: Corrected parent materially changes short-continuation behavior; rerun corrected Z2 full before final N4 selection.

## Parent Selection Audit

| candidate | exists | size_bytes | eval_label | success_rate | collision_rate | near_miss_rate | raw_unsafe_action_rate | diagnostics_ok | selected_by_old_rule | selected_by_fixed_rule | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoint_500k | 1.0000 | 1173883.0000 | 500k | 0.4800 | 0.5200 | 0.4600 | 0.2285 | 1.0000 | 0.0000 | 1.0000 | diagnostics normal; selected by success desc, collision asc, near_miss asc, raw_unsafe asc |
| final | 1.0000 | 1173883.0000 | final | 0.4700 | 0.5300 | 0.4267 | 0.2362 | 1.0000 | 0.0000 | 0.0000 | nan |
| best_by_eval | 1.0000 | 1173883.0000 | best_by_eval | 0.4700 | 0.5300 | 0.4267 | 0.2362 | 1.0000 | 1.0000 | 0.0000 | old N3Z2C selected this candidate |

## Resume Semantics Audit

| script | uses_reset_num_timesteps_false | uses_reset_num_timesteps_true | finding |
| --- | --- | --- | --- |
| old_n3z2c_train | 0.0000 | 1.0000 | old N3Z2C continuation used reset_num_timesteps=True; this is a resume-semantics bug |
| audit_corrected_train | 1.0000 | 0.0000 | audit corrected train must use reset_num_timesteps=False |
| phase | check | status | reset_num_timesteps | model_num_timesteps | optimizer_state_restored | n_envs | n_steps | batch_size | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nan | selected_parent_checkpoint | ok | nan | nan | nan | nan | nan | nan | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip |
| nan | old_n3z2c_reset_num_timesteps | bug_confirmed | nan | nan | nan | nan | nan | nan | scripts/train_env_v2_gpsi_ppo_n3z2c.py uses reset_num_timesteps=True during continuation |
| nan | audit_corrected_reset_num_timesteps | planned_corrected | nan | nan | nan | nan | nan | nan | scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py must log and use reset_num_timesteps=False |
| nan | global_step_plan | ok | nan | nan | nan | nan | nan | nan | parent_total_steps=500000, additional_steps=250000, target_total_steps=750000, checkpoint_total_steps=[750000] |
| nan | sb3_load_optimizer_state | ok | nan | nan | nan | nan | nan | nan | {"batch_size": 256, "ent_coef": 0.01, "gae_lambda": 0.95, "gamma": 0.99, "max_grad_norm": 0.5, "model_num_timesteps": 500000, "n_epochs": 10, "n_steps": 1024, "optimizer_state_entries": 25, "vf_coef": 0.5} |
| before_learn | nan | nan | 0.0000 | 500000.0000 | 1.0000 | 4.0000 | 1024.0000 | 256.0000 | reconstructed from selected parent checkpoint |
| after_learn | nan | nan | 0.0000 | 753952.0000 | 1.0000 | 4.0000 | 1024.0000 | 256.0000 | reconstructed from corrected final checkpoint: checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/final.zip |

## CPU Affinity Finding

| item | returncode | output |
| --- | --- | --- |
| nproc | 0.0000 | 1 |
| nproc_all | 0.0000 | 16 |
| taskset_current_shell | 0.0000 | pid 10928's current affinity list: 0-15 |
| python_cpu_affinity | 0.0000 | {"affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "affinity_count": 16, "os_cpu_count": 16} |
| cpuset_cpus_effective | 0.0000 | 0-15 |
| cpu_max | 0.0000 | max 100000 |
| cpu_affinity_finding | 0.0000 | {"OMP_NUM_THREADS": "1", "affinity_count": 16, "cpu_max": "max 100000", "cpuset_cpus_effective": "0-15", "finding": "nproc_reports_omp_thread_limit_not_affinity_limit", "nproc": 1, "nproc_all": 16} |

## Corrected Short Continuation Results

| method_key | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 0.6100 | 0.3900 | 0.5767 | 0.9572 | 0.2914 | 0.1388 | 4.8633 |
| n3f_no_z_full | final | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 0.0967 | 5.4235 |
| z2_corrected_parent | corrected_750k | 0.4800 | 0.5200 | 0.4167 | 0.9390 | 0.2266 | 0.0901 | 5.5538 |
| z2_corrected_parent | corrected_parent_500k | 0.4800 | 0.5200 | 0.4600 | 0.9400 | 0.2285 | 0.0832 | 5.5098 |
| z2_old_n3z2c | old_750k | 0.4667 | 0.5333 | 0.4433 | 0.9400 | 0.2599 | 0.0921 | 4.8778 |
| z2_old_n3z2c | old_parent_500k | 0.4700 | 0.5300 | 0.4267 | 0.9376 | 0.2362 | 0.0845 | 5.3009 |

## Diagnostics

| method_key | checkpoint_label | diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | logvar_xy_1s_span_max | z_after_constraint_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| z2_corrected_parent | corrected_750k | 1.0000 | 1.6438 | 1.8998 | 0.0000 | 5.3330 | 4.0000 | 0.0000 |

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/checkpoint_750k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/parent_500k.zip`
### tables
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_checkpoint_hash.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_checkpoint_success_collision_curve.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_command_manifest.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_config_manifest.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_decision.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_diagnostics_decision.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_episode_metrics.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_eval_summary.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_feature_block_stats.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_gpsi_output_summary.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_parent_selection_fixed.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_parent_selection_fixed.json`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_raw_unsafe_summary.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_resource_affinity.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_resume_semantics.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_scenario_breakdown.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_short_continuation_heartbeat.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_short_continuation_train_curve.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_threat_class_breakdown.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_audit_train_script_findings.csv`
- `results/env_v2_phase_n3z2c_audit/tables/phase_n3z2c_schema_check.csv`
### plots
- `results/env_v2_phase_n3z2c_audit/plots/audit_corrected_vs_old_750k_success_collision.png`
- `results/env_v2_phase_n3z2c_audit/plots/audit_feature_block_scale.png`
- `results/env_v2_phase_n3z2c_audit/plots/audit_parent_selection_comparison.png`
- `results/env_v2_phase_n3z2c_audit/plots/audit_raw_unsafe_comparison.png`
- `results/env_v2_phase_n3z2c_audit/plots/audit_scenario_breakdown.png`
- `results/env_v2_phase_n3z2c_audit/plots/audit_short_continuation_curve.png`
- `results/env_v2_phase_n3z2c_audit/plots/checkpoint_success_collision.png`
### logs
- `results/env_v2_phase_n3z2c_audit/logs/phase_n3z2c_audit_analysis.log`
- `results/env_v2_phase_n3z2c_audit/logs/phase_n3z2c_audit_eval.log`
- `results/env_v2_phase_n3z2c_audit/logs/phase_n3z2c_audit_parent_resume.log`
- `results/env_v2_phase_n3z2c_audit/logs/phase_n3z2c_audit_train_corrected_short.log`
- `results/env_v2_phase_n3z2c_audit/phase_n3z2c_audit_watcher.log`
### flags
- `results/env_v2_phase_n3z2c_audit/PHASE_N3Z2C_AUDIT_COMPLETE.flag`
