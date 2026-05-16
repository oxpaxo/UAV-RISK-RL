# Phase N3Z2C Z2 Continuation Report

## Terminal Decision

`terminal_decision = phase_n3z2c_z2_continuation_complete`

Phase N3Z2C complete. This stage continues Z2 `z_layernorm_alpha_0p5` to total 1.5M and selects the final no-shield N4 candidate.

## Engineering Facts

- Phase N3F/Z complete flag exists.
- Repaired `GpsiObsWrapper` was used; Gpsi checkpoint `work_dirs/gpsi_heada_v1_nll/best.pth` stayed frozen.
- EnvV2 core was not modified.
- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.
- Logvar clip sanity: config uses `[-5, 3]`, already bounded tighter than `|logvar| <= 5`.

## Parent Selection

| candidate | path | exists | size | eval_label | success_rate | collision_rate | selected_parent_key | selected_parent_path | selection_reason | parent_total_steps | additional_steps | target_total_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoint_500k | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip | 1.0000 | 1173883.0000 | 500k | 0.4800 | 0.5200 | nan | nan | nan | nan | nan | nan |
| final | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip | 1.0000 | 1173883.0000 | final | 0.4700 | 0.5300 | nan | nan | nan | nan | nan | nan |
| best_by_eval | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip | 1.0000 | 1173883.0000 | best_by_eval | 0.4700 | 0.5300 | nan | nan | nan | nan | nan | nan |
| selected | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip | 1.0000 | 1173883.0000 | 500k_parent | 0.4700 | 0.5300 | best_by_eval | checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip | best_by_eval exists and is not worse than final in both success and collision | 500000.0000 | 1000000.0000 | 1500000.0000 |

## CPU Strategy

| timestamp | item | command | returncode | output |
| --- | --- | --- | --- | --- |
| 2026-05-13 12:54:38 | nproc | nproc | 0.0000 | 1 |
| 2026-05-13 12:54:38 | lscpu | lscpu | 0.0000 | Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      48 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             16
On-line CPU(s) list:                0-15
Vendor ID:                          AuthenticAMD
Model name:                         AMD EPYC 7402 24-Core Processor
CPU family:                         23
Model:                              49
Thread(s) per core:                 2
Core(s) per socket:                 8
Socket(s):                          1
Stepping:                           0
BogoMIPS:                           5599.82
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr wbnoinvd virt_ssbd arat umip rdpid arch_capabilities
L1d cache:                          512 KiB (8 instances)
L1i cache:                          512 KiB (8 instances)
L2 cache:                           4 MiB (8 instances)
L3 cache:                           32 MiB (2 instances)
NUMA node(s):                       1
NUMA node0 CPU(s):                  0-15
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Mitigation; untrained return thunk; SMT enabled with STIBP protection
Vulnerability Spec rstack overflow: Mitigation; safe RET
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Retpolines; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected |
| 2026-05-13 12:54:38 | free_h | free -h | 0.0000 | total        used        free      shared  buff/cache   available
Mem:            62Gi       3.9Gi        53Gi       0.0Ki       5.8Gi        58Gi
Swap:             0B          0B          0B |
| 2026-05-13 12:54:39 | df_h_root | df -h / | 0.0000 | Filesystem      Size  Used Avail Use% Mounted on
overlay          79G  4.6G   71G   7% / |
| 2026-05-13 12:54:40 | nvidia_smi | nvidia-smi | 0.0000 | Wed May 13 12:54:40 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:00:03.0 Off |                  N/A |
| 30%   34C    P0             56W /  350W |       0MiB /  24576MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+ |
| 2026-05-13 12:54:40 | git_status_short | git status --short | 128.0000 | fatal: not a git repository (or any of the parent directories): .git |
| 2026-05-13 12:54:40 | OMP_NUM_THREADS | env | 0.0000 | 1 |
| 2026-05-13 12:54:40 | MKL_NUM_THREADS | env | 0.0000 | 1 |
| 2026-05-13 12:54:40 | OPENBLAS_NUM_THREADS | env | 0.0000 | 1 |
| 2026-05-13 12:54:40 | NUMEXPR_NUM_THREADS | env | 0.0000 | 1 |

## n_envs Smoke Benchmark

| n_envs | status | log | fps_last | fps_max | last_total_timesteps | bad_signal | selected_n_envs | selection_reason | ppo_semantics_changed_if_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0000 | completed | results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_benchmark_nenv4.log | 270.9000 | 277.3000 | 4096.0000 | 0.0000 | 4.0000 | Kept n_envs=4 because increasing n_envs changes PPO rollout batch semantics versus N3F/Z; benchmark is audit-only. | 0.0000 |
| 8.0000 | skipped | nan | nan | nan | nan | 0.0000 | 4.0000 | Kept n_envs=4 because increasing n_envs changes PPO rollout batch semantics versus N3F/Z; benchmark is audit-only. | 1.0000 |
| 12.0000 | skipped | nan | nan | nan | nan | 0.0000 | 4.0000 | Kept n_envs=4 because increasing n_envs changes PPO rollout batch semantics versus N3F/Z; benchmark is audit-only. | 1.0000 |

## Main Results

| method_key | checkpoint_label | success_rate | collision_rate | near_miss_rate | progress | raw_unsafe_action_rate | action_delta | mean_min_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_full | attention_full_1500k | 0.6100 | 0.3900 | 0.5767 | 0.9572 | 0.2914 | 0.1388 | 4.8633 |
| n3f_no_z_full | best_by_eval | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 0.0967 | 5.4235 |
| n3f_no_z_full | final | 0.5633 | 0.4367 | 0.5067 | 0.9471 | 0.2398 | 0.0967 | 5.4235 |
| z_layernorm_alpha_0p5_cont_1p5m | 1500k | 0.4500 | 0.5500 | 0.4167 | 0.9331 | 0.2671 | 0.0864 | 5.5951 |
| z_layernorm_alpha_0p5_cont_1p5m | best_by_eval | 0.4500 | 0.5500 | 0.3800 | 0.9253 | 0.2771 | 0.0881 | 5.5193 |
| z_layernorm_alpha_0p5_cont_1p5m | final | 0.4500 | 0.5500 | 0.3800 | 0.9253 | 0.2771 | 0.0881 | 5.5193 |
| z_layernorm_alpha_0p5_cont_1p5m | parent_500k | 0.4700 | 0.5300 | 0.4267 | 0.9376 | 0.2362 | 0.0845 | 5.3009 |

## Candidate Decision

| z2_success_rate | z2_collision_rate | noz_success_rate | noz_collision_rate | attention_success_rate | attention_collision_rate | z2_minus_noz_success | z2_minus_noz_collision | z2_minus_attention_success | z2_minus_attention_collision | diagnostics_ok | selected_n4_candidate | include_both_as_ablation | z2_beats_attention_no_shield | can_enter_n4 | decision | attention_statement | checkpoint_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4500 | 0.5500 | 0.5633 | 0.4367 | 0.6100 | 0.3900 | -0.1133 | 0.1133 | -0.1600 | 0.1600 | 1.0000 | no_z_full | no | 0.0000 | yes | Z2 full is worse than no_z on success and collision; no_z remains primary and Z2 is ablation only. | Gpsi-PPO no-shield does not beat attention; comparison is close/weak or scenario-dependent. | checkpoint_1500k, final, and best_by_eval are evaluated as separate rows; best_by_eval is final unless a later selector overwrote it. |
- Selected N4 candidate: `no_z_full`.
- Can enter N4: yes.
- Decision: Z2 full is worse than no_z on success and collision; no_z remains primary and Z2 is ablation only.
- Attention comparison: Gpsi-PPO no-shield does not beat attention; comparison is close/weak or scenario-dependent.

## Diagnostics

| method_key | checkpoint_label | diagnostics_ok | delta_norm_1s_p95_max | delta_norm_1s_max | inactive_forwarded_count_max | logvar_xy_1s_span_max | z_after_constraint_l2_p95_max | feature_nonfinite_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| z_layernorm_alpha_0p5_cont_1p5m | final | 1.0000 | 1.7039 | 2.4449 | 0.0000 | 5.3584 | 4.0000 | 0.0000 |

## Gpsi Output Diagnostics

| method_key | delta_norm_1s_p95 | delta_norm_1s_max | logvar_xy_1s_mean | logvar_xy_1s_span | projected_std_radial_mean | projected_std_relvel_mean | inactive_forwarded_count_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| z_layernorm_alpha_0p5_cont_1p5m | 1.7039 | 2.4449 | -4.6431 | 5.3584 | 0.1067 | 0.1091 | 0.0000 |

## Feature Block Stats

| block | z_transform | l2_norm_p95 | max_abs_p95 | nan_count | inf_count |
| --- | --- | --- | --- | --- | --- |
| delta_hat_9_after_scale | layernorm | 2.0954 | 1.8873 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.6543 | 1.2607 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.8354 | 1.5088 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 2.0188 | 1.7180 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.9103 | 1.6221 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.8502 | 1.5613 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.9628 | 1.6716 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.9827 | 1.6737 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 1.8504 | 1.5043 | 0.0000 | 0.0000 |
| delta_hat_9_after_scale | layernorm | 2.1733 | 1.8945 | 0.0000 | 0.0000 |
| full_aug_obs | layernorm | 15.0291 | 5.0000 | 0.0000 | 0.0000 |
| full_aug_obs | layernorm | 15.0206 | 5.0000 | 0.0000 | 0.0000 |

## Checkpoint Semantics

- `parent_500k.zip` is a copy of the selected N3F/Z Z2 parent checkpoint and is evaluated as the 500k parent.
- `checkpoint_750k/1000k/1250k/1500k.zip` are saved by global total training step.
- `final.zip` is saved after continuation completion.
- `best_by_eval.zip` is copied from `final.zip` in this stage because no train-time eval selector is used; both are still evaluated as distinct labels.

## Breakdown Outputs

Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3z2c_z2_continuation/tables/`.

## Artifacts

### checkpoints
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/best_by_eval.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/checkpoint_1000k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/checkpoint_1250k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/checkpoint_1500k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/checkpoint_750k.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/final.zip`
- `checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/parent_500k.zip`
### tables
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_attention_reference_comparison.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_aug_feature_block_stats.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_checkpoint_eval_summary.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_command_manifest.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_config_manifest.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_cpu_benchmark.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_diagnostics_decision.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_episode_metrics.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_eval_summary.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_final_candidate_decision.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_gpsi_output_summary.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_motion_mode_breakdown.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_noz_reference_comparison.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_parent_checkpoint_selection.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_parent_screening_comparison.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_raw_unsafe_action_steps.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_raw_unsafe_action_summary.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_resource_preflight.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_scenario_breakdown.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_schema_check.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_threat_class_breakdown.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_train_curve.csv`
- `results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_train_heartbeat.csv`
### plots
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_action_dynamics.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_aug_feature_block_scale.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_checkpoint_success_collision.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_gpsi_delta_norm.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_gpsi_logvar.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_raw_unsafe_by_checkpoint.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_scenario_breakdown.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_train_reward.png`
- `results/env_v2_phase_n3z2c_z2_continuation/plots/z2_vs_noz_attention_success_collision.png`
### logs
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_analysis.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_benchmark_nenv4.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_eval.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_parent_selection.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_resource_preflight.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_resource_preflight_detail.log`
- `results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_train_z2_continuation.log`
- `results/env_v2_phase_n3z2c_z2_continuation/phase_n3z2c_watcher.log`
### flags
- `results/env_v2_phase_n3z2c_z2_continuation/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag`
