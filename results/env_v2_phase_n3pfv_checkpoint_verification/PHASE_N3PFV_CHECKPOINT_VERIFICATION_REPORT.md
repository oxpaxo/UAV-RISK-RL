# Phase N3PF-V Checkpoint Verification Report

## Terminal Decision

`terminal_decision = phase_n3pfv_checkpoint_verification_complete`

N3PF-V completed eval-only multi-seed verification. No PPO or Gpsi training was run.

## Candidate Decision

| selected_n4_candidate | can_enter_n4 | decision | diagnostics_ok | p3_1000k_success | p3_1000k_collision | p3_1500k_success | p3_1500k_collision | p3_final_success | p3_final_collision | attention_success | attention_collision | noz_success | noz_collision | p3_1500k_matches_or_exceeds_attention | p3_1000k_offline_eval_selected | do_not_claim_decisive_attention_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3_checkpoint_1500k | yes | P3 1500k is comparable to or better than attention_full; prefer exact 1.5M snapshot. | 1.0000 | 0.6367 | 0.3633 | 0.6167 | 0.3833 | 0.5933 | 0.4067 | 0.6033 | 0.3967 | 0.5667 | 0.4333 | 1.0000 | 0.0000 | 1.0000 |

## Aggregate Mean/Std

| method_key | checkpoint_label | num_eval_seeds | num_episodes_total | mean_success_rate | std_success_rate | mean_collision_rate | std_collision_rate | mean_near_miss_rate | mean_raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_parent_500k | parent_500k | 3.0000 | 900.0000 | 0.5400 | 0.0058 | 0.4600 | 0.0058 | 0.4767 | 0.2459 |
| p3_1000k | 1000k | 3.0000 | 900.0000 | 0.6367 | 0.0058 | 0.3633 | 0.0058 | 0.5422 | 0.2278 |
| p3_1250k | 1250k | 3.0000 | 900.0000 | 0.4456 | 0.0019 | 0.5544 | 0.0019 | 0.4189 | 0.3122 |
| p3_1500k | 1500k | 3.0000 | 900.0000 | 0.6167 | 0.0033 | 0.3833 | 0.0033 | 0.5933 | 0.3138 |
| p3_final | final | 3.0000 | 900.0000 | 0.5933 | 0.0033 | 0.4067 | 0.0033 | 0.5633 | 0.3092 |
| attention_full | attention_full_1500k | 3.0000 | 900.0000 | 0.6033 | 0.0067 | 0.3967 | 0.0067 | 0.5700 | 0.2901 |
| no_z_full | final | 3.0000 | 900.0000 | 0.5667 | 0.0033 | 0.4333 | 0.0033 | 0.5089 | 0.2383 |
| z2_corrected_full | z2_final | 3.0000 | 900.0000 | 0.5089 | 0.0019 | 0.4911 | 0.0019 | 0.4722 | 0.2604 |

## Pairwise

| policy_a | policy_b | success_diff_mean | collision_diff_mean | seed_success_diff_std | worst_scenario_success_diff | better_both |
| --- | --- | --- | --- | --- | --- | --- |
| p3_1000k | attention_full | 0.0333 | -0.0333 | 0.0120 | -0.1733 | 1.0000 |
| p3_1000k | no_z_full | 0.0700 | -0.0700 | 0.0033 | -0.3200 | 1.0000 |
| p3_1500k | attention_full | 0.0133 | -0.0133 | 0.0058 | -0.1200 | 1.0000 |
| p3_1500k | no_z_full | 0.0500 | -0.0500 | 0.0058 | -0.0867 | 1.0000 |
| p3_final | attention_full | -0.0100 | 0.0100 | 0.0058 | -0.1133 | 0.0000 |
| p3_final | no_z_full | 0.0267 | -0.0267 | 0.0058 | -0.0600 | 1.0000 |
| p3_1000k | p3_1500k | 0.0200 | -0.0200 | 0.0088 | -0.2333 | 1.0000 |
| p3_1500k | p3_final | 0.0233 | -0.0233 | 0.0000 | -0.0533 | 1.0000 |

## Per-Seed Summary

| method_key | checkpoint_label | eval_seed | episodes | success_rate | collision_rate | near_miss_rate | raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 1000.0000 | 300.0000 | 0.6100 | 0.3900 | 0.5767 | 0.2914 |
| attention_full | attention_full_1500k | 1001.0000 | 300.0000 | 0.6033 | 0.3967 | 0.5700 | 0.2905 |
| attention_full | attention_full_1500k | 1002.0000 | 300.0000 | 0.5967 | 0.4033 | 0.5633 | 0.2885 |
| no_z_full | final | 1000.0000 | 300.0000 | 0.5633 | 0.4367 | 0.5067 | 0.2398 |
| no_z_full | final | 1001.0000 | 300.0000 | 0.5667 | 0.4333 | 0.5100 | 0.2389 |
| no_z_full | final | 1002.0000 | 300.0000 | 0.5700 | 0.4300 | 0.5100 | 0.2361 |
| p3_1000k | 1000k | 1000.0000 | 300.0000 | 0.6300 | 0.3700 | 0.5367 | 0.2275 |
| p3_1000k | 1000k | 1001.0000 | 300.0000 | 0.6400 | 0.3600 | 0.5467 | 0.2283 |
| p3_1000k | 1000k | 1002.0000 | 300.0000 | 0.6400 | 0.3600 | 0.5433 | 0.2275 |
| p3_1250k | 1250k | 1000.0000 | 300.0000 | 0.4467 | 0.5533 | 0.4200 | 0.3119 |
| p3_1250k | 1250k | 1001.0000 | 300.0000 | 0.4467 | 0.5533 | 0.4200 | 0.3126 |
| p3_1250k | 1250k | 1002.0000 | 300.0000 | 0.4433 | 0.5567 | 0.4167 | 0.3123 |
| p3_1500k | 1500k | 1000.0000 | 300.0000 | 0.6200 | 0.3800 | 0.5967 | 0.3131 |
| p3_1500k | 1500k | 1001.0000 | 300.0000 | 0.6133 | 0.3867 | 0.5900 | 0.3146 |
| p3_1500k | 1500k | 1002.0000 | 300.0000 | 0.6167 | 0.3833 | 0.5933 | 0.3137 |
| p3_final | final | 1000.0000 | 300.0000 | 0.5967 | 0.4033 | 0.5667 | 0.3075 |
| p3_final | final | 1001.0000 | 300.0000 | 0.5900 | 0.4100 | 0.5600 | 0.3096 |
| p3_final | final | 1002.0000 | 300.0000 | 0.5933 | 0.4067 | 0.5633 | 0.3106 |
| p3_parent_500k | parent_500k | 1000.0000 | 300.0000 | 0.5333 | 0.4667 | 0.4700 | 0.2465 |
| p3_parent_500k | parent_500k | 1001.0000 | 300.0000 | 0.5433 | 0.4567 | 0.4800 | 0.2459 |
| p3_parent_500k | parent_500k | 1002.0000 | 300.0000 | 0.5433 | 0.4567 | 0.4800 | 0.2453 |
| z2_corrected_full | z2_final | 1000.0000 | 300.0000 | 0.5067 | 0.4933 | 0.4700 | 0.2629 |
| z2_corrected_full | z2_final | 1001.0000 | 300.0000 | 0.5100 | 0.4900 | 0.4733 | 0.2608 |
| z2_corrected_full | z2_final | 1002.0000 | 300.0000 | 0.5100 | 0.4900 | 0.4733 | 0.2574 |

## Scenario Focus

| method_key | checkpoint_label | eval_seed | scenario | success_rate | collision_rate | raw_unsafe_action_rate |
| --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_high_density | 0.6400 | 0.3600 | 0.3168 |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_high_speed | 0.5400 | 0.4600 | 0.2633 |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_high_threat | 0.6200 | 0.3800 | 0.2949 |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_id | 0.5800 | 0.4200 | 0.2739 |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_mixed_ood | 0.5400 | 0.4600 | 0.3100 |
| attention_full | attention_full_1500k | 1000.0000 | eval_flow_sudden_threat | 0.7400 | 0.2600 | 0.2898 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_high_density | 0.6200 | 0.3800 | 0.3161 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_high_speed | 0.5200 | 0.4800 | 0.2647 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_high_threat | 0.6200 | 0.3800 | 0.2946 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_id | 0.6000 | 0.4000 | 0.2680 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_mixed_ood | 0.5200 | 0.4800 | 0.3125 |
| attention_full | attention_full_1500k | 1001.0000 | eval_flow_sudden_threat | 0.7400 | 0.2600 | 0.2869 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_high_density | 0.6000 | 0.4000 | 0.3177 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_high_speed | 0.5000 | 0.5000 | 0.2634 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_high_threat | 0.6200 | 0.3800 | 0.2923 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_id | 0.6200 | 0.3800 | 0.2665 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_mixed_ood | 0.5200 | 0.4800 | 0.3072 |
| attention_full | attention_full_1500k | 1002.0000 | eval_flow_sudden_threat | 0.7200 | 0.2800 | 0.2839 |
| no_z_full | final | 1000.0000 | eval_flow_high_density | 0.5000 | 0.5000 | 0.2798 |
| no_z_full | final | 1000.0000 | eval_flow_high_speed | 0.4000 | 0.6000 | 0.2344 |
| no_z_full | final | 1000.0000 | eval_flow_high_threat | 0.5000 | 0.5000 | 0.2410 |
| no_z_full | final | 1000.0000 | eval_flow_id | 0.7600 | 0.2400 | 0.2201 |
| no_z_full | final | 1000.0000 | eval_flow_mixed_ood | 0.5600 | 0.4400 | 0.2533 |
| no_z_full | final | 1000.0000 | eval_flow_sudden_threat | 0.6600 | 0.3400 | 0.2099 |
| no_z_full | final | 1001.0000 | eval_flow_high_density | 0.5200 | 0.4800 | 0.2778 |
| no_z_full | final | 1001.0000 | eval_flow_high_speed | 0.4000 | 0.6000 | 0.2357 |
| no_z_full | final | 1001.0000 | eval_flow_high_threat | 0.5000 | 0.5000 | 0.2373 |
| no_z_full | final | 1001.0000 | eval_flow_id | 0.7400 | 0.2600 | 0.2183 |
| no_z_full | final | 1001.0000 | eval_flow_mixed_ood | 0.5800 | 0.4200 | 0.2533 |
| no_z_full | final | 1001.0000 | eval_flow_sudden_threat | 0.6600 | 0.3400 | 0.2112 |
| no_z_full | final | 1002.0000 | eval_flow_high_density | 0.5400 | 0.4600 | 0.2794 |
| no_z_full | final | 1002.0000 | eval_flow_high_speed | 0.4200 | 0.5800 | 0.2358 |
| no_z_full | final | 1002.0000 | eval_flow_high_threat | 0.4800 | 0.5200 | 0.2354 |
| no_z_full | final | 1002.0000 | eval_flow_id | 0.7400 | 0.2600 | 0.2188 |
| no_z_full | final | 1002.0000 | eval_flow_mixed_ood | 0.5800 | 0.4200 | 0.2377 |
| no_z_full | final | 1002.0000 | eval_flow_sudden_threat | 0.6600 | 0.3400 | 0.2097 |

## Motion/Threat Notes

| method_key | eval_seed | threat_motion_mode | episodes | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| p3_1000k | 1000.0000 | linear | 57.0000 | 0.5439 | 0.4561 |
| p3_1000k | 1001.0000 | linear | 59.0000 | 0.5424 | 0.4576 |
| p3_1000k | 1002.0000 | linear | 57.0000 | 0.5439 | 0.4561 |
| p3_1500k | 1000.0000 | linear | 34.0000 | 0.6765 | 0.3235 |
| p3_1500k | 1001.0000 | linear | 33.0000 | 0.6667 | 0.3333 |
| p3_1500k | 1002.0000 | linear | 32.0000 | 0.6875 | 0.3125 |
| p3_final | 1000.0000 | linear | 32.0000 | 0.4062 | 0.5938 |
| p3_final | 1001.0000 | linear | 33.0000 | 0.3939 | 0.6061 |
| p3_final | 1002.0000 | linear | 33.0000 | 0.4242 | 0.5758 |
| method_key | eval_seed | threat_class | episodes | success_rate | collision_rate |
| --- | --- | --- | --- | --- | --- |
| p3_1000k | 1000.0000 | high | 285.0000 | 0.6211 | 0.3789 |
| p3_1000k | 1000.0000 | medium | 15.0000 | 0.8000 | 0.2000 |
| p3_1000k | 1001.0000 | high | 286.0000 | 0.6294 | 0.3706 |
| p3_1000k | 1001.0000 | medium | 14.0000 | 0.8571 | 0.1429 |
| p3_1000k | 1002.0000 | high | 286.0000 | 0.6294 | 0.3706 |
| p3_1000k | 1002.0000 | medium | 14.0000 | 0.8571 | 0.1429 |
| p3_1500k | 1000.0000 | high | 288.0000 | 0.6181 | 0.3819 |
| p3_1500k | 1000.0000 | medium | 12.0000 | 0.6667 | 0.3333 |
| p3_1500k | 1001.0000 | high | 287.0000 | 0.6098 | 0.3902 |
| p3_1500k | 1001.0000 | medium | 13.0000 | 0.6923 | 0.3077 |
| p3_1500k | 1002.0000 | high | 287.0000 | 0.6132 | 0.3868 |
| p3_1500k | 1002.0000 | medium | 13.0000 | 0.6923 | 0.3077 |

## Diagnostics

| diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | adapter_output_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 2.0923 | 2.7372 | 0.0000 | 6.2935 | 0.0000 |

## Checkpoint Manifest

| policy_key | checkpoint_label | checkpoint_path | sha256 | source_phase | selected_for_required_eval | selected_for_diagnostic_eval |
| --- | --- | --- | --- | --- | --- | --- |
| p3_1000k | 1000k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip | 5685cb07b5d861abac35f8c3fa4103cfaa5ef6c574d2dcb613f80b695e9c5a04 | N3PF | 1.0000 | 0.0000 |
| p3_1500k | 1500k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip | 1aec7318cb0b745f3e98ea3d8e15996e40ac19cf416220cc7ead9926c567bd3b | N3PF | 1.0000 | 0.0000 |
| p3_final | final | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip | a7537e97403d135500408a3dc98b736ab2da997649ce4b87d8fc3dd7b524310a | N3PF | 1.0000 | 0.0000 |
| attention_full | attention_full_1500k | checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip | 711d68c688a4388efec34cafeb05faa8b979eaf480aaad1dd86f5bf60df82791 | Phase2 | 1.0000 | 0.0000 |
| no_z_full | final | checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip | 05ff221a8f996b8c2e5b156f400fabec9688f90cef6c9be89f3a869d370609f8 | N3FZ | 1.0000 | 0.0000 |
| z2_corrected_full | z2_final | checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip | bbdcc7255e27c6ccca7e41df0c007c9163e03d880771b1812d8350d6b021f208 | N3Z2CF | 1.0000 | 0.0000 |
| p3_parent_500k | parent_500k | checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip | 9ba7f68424e65413aead99a51bc1f7bc8de1f00dabbbf6bace45f14dc3bd7ab2 | N3P | 0.0000 | 1.0000 |
| p3_1250k | 1250k | checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1250k.zip | b839daa9f610f9200b8f50cfd903b0cbfa9a96357ea2b53c43d7fabf49add1c9 | N3PF | 0.0000 | 1.0000 |

## Artifacts

### tables
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_action_dynamics_summary.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_candidate_decision.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_checkpoint_manifest.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_command_manifest.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_diagnostics_decision.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_episode_metrics.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_eval_summary_aggregate.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_eval_summary_by_seed.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_feature_block_stats.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_gpsi_output_summary.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_pairwise_comparison.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_scenario_breakdown.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_schema_check.csv`
- `results/env_v2_phase_n3pfv_checkpoint_verification/tables/phase_n3pfv_threat_class_breakdown.csv`
### plots
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_action_dynamics.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_checkpoint_comparison.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_pairwise_vs_attention.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_raw_unsafe_comparison.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_scenario_breakdown.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_seed_stability.png`
- `results/env_v2_phase_n3pfv_checkpoint_verification/plots/n3pfv_success_collision_mean_ci.png`
### logs
- `results/env_v2_phase_n3pfv_checkpoint_verification/logs/phase_n3pfv_analysis.log`
- `results/env_v2_phase_n3pfv_checkpoint_verification/logs/phase_n3pfv_eval.log`
- `results/env_v2_phase_n3pfv_checkpoint_verification/phase_n3pfv_watcher.log`
### flags
- `results/env_v2_phase_n3pfv_checkpoint_verification/PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag`

Can enter N4: yes.
Selected N4 candidate: `P3_checkpoint_1500k`.
