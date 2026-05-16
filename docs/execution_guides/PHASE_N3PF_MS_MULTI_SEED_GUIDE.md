# Phase N3PF-MS 指南：P3 Block-Projected Multi-Training-Seed Verification

## 0. 目标

本阶段用于验证 `P3 block_projected Gψ-HeadA PPO` 不是 seed0 偶然结果。只训练 P3 的新 training seeds，不接 shield，不改 EnvV2-core，不 fine-tune Gψ，不继续改网络结构。

当前参考结果：

```text
P3 checkpoint_1500k seed0 verification: success=0.6167, collision=0.3833
attention_full verification:             success=0.6033, collision=0.3967
no_z full verification:                   success=0.5667, collision=0.4333
```

本阶段必须回答：

```text
1. P3 seed1/seed2 是否稳定？
2. 多 training seed 下 P3 是否仍 comparable / slightly better than attention_full？
3. 是否需要后续补 attention_full seed1/seed2？
4. P3 是否仍适合作为 N4 primary candidate？
```

## 1. 禁止事项

```text
禁止进入 N4
禁止实现 shield / action filtering / safety cost
禁止训练或 fine-tune Gψ
禁止修改 EnvV2-core
禁止改变 P3 block_projected 架构
禁止继续做 P4/P5/P6 新结构
禁止覆盖 N3PF / N3PF-V / N3P 原始产物
禁止为了结果好看从 eval set 选择非预设 checkpoint 当主结果
```

## 2. 训练任务

训练：

```text
P3 block_projected seed1
P3 block_projected seed2
```

配置必须与 seed0 P3 一致：

```text
include_z = false
obs_i: 12 dims
delta_hat_scaled: 9 dims, delta_hat / 5.0
logvar_scaled: 9 dims, clamp(logvar, -5, 3) / 5.0
obs/delta/logvar 分块投影 + LayerNorm
adapter output约64维
Gψ frozen
PPO trainable
n_envs=4
device=cpu
train_steps=1,500,000
```

保存：

```text
250k / 500k / 750k / 1000k / 1250k / 1500k / final
```

输出目录：

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s1/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s2/
results/env_v2_phase_n3pf_ms_multiseed/
```

## 3. 并行与 CPU 使用

当前机器有 16 logical CPUs。不要通过提高单个 PPO job 的 `n_envs` 来榨干 CPU，因为这会改变 rollout semantics。推荐使用跨 seed 的进程级并行：

```text
seed1: n_envs=4
seed2: n_envs=4
```

推荐：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

可选 CPU affinity：

```bash
taskset -c 0-7  <seed1 command>
taskset -c 8-15 <seed2 command>
```

并行前必须 smoke test。若并行导致任一 job fps 下降过大、日志混乱、checkpoint 冲突、CPU/IO 不安全，则回退顺序执行。

可以与 N4-O 同时运行，但训练优先。若训练 fps 下降超过约15%，N4-O 应暂停或排队。

## 4. Eval protocol

对 seed1/seed2 的以下 checkpoints 评估：

```text
checkpoint_1500k
checkpoint_1000k
final
```

Eval seeds：

```text
1000 1001 1002
```

Scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

Episodes：

```text
50 per scenario per eval seed
```

每个 training seed / checkpoint 合计 900 episodes。

## 5. 指标

主指标：

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
mean_episode_reward
mean_episode_length
```

诊断：

```text
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
```

Breakdowns：

```text
scenario / motion mode / threat class
```

Gψ 与 feature：

```text
delta_norm_1s_p95/max
logvar span
adapter_output_l2_p95
feature_nonfinite_count
inactive_forwarded_count
```

## 6. 决策规则

P3 multi-seed 稳定的最低要求：

```text
seed1/seed2 都完成 1.5M
diagnostics 正常
checkpoint_1500k 不显著崩溃
不明显弱于 no_z full
```

建议阈值：

```text
collapse warning:
  checkpoint_1500k success < 0.55 或 collision > 0.45

strong positive:
  3-training-seed mean success >= attention_full + 0.02
  且 collision <= attention_full - 0.02

comparable positive:
  3-training-seed mean success >= attention_full - 0.01
  且 collision <= attention_full + 0.01
```

如 P3 seed1/2 稳定且 margin 小，建议后续补 attention_full seed1/2。若 P3 seed1/2 不稳，先分析 P3，不急着补 attention。

## 7. Required outputs

```text
PHASE_N3PF_MS_MULTI_SEED_REPORT.md
PHASE_N3PF_MS_MULTI_SEED_COMPLETE.flag
phase_n3pf_ms_status.txt
phase_n3pf_ms_watcher.log
```

Tables：

```text
phase_n3pf_ms_config_manifest.csv
phase_n3pf_ms_command_manifest.csv
phase_n3pf_ms_resource_affinity.csv
phase_n3pf_ms_train_heartbeat.csv
phase_n3pf_ms_train_curve.csv
phase_n3pf_ms_checkpoint_integrity.csv
phase_n3pf_ms_eval_summary_by_seed.csv
phase_n3pf_ms_eval_summary_aggregate.csv
phase_n3pf_ms_training_seed_summary.csv
phase_n3pf_ms_pairwise_vs_attention.csv
phase_n3pf_ms_scenario_breakdown.csv
phase_n3pf_ms_motion_mode_breakdown.csv
phase_n3pf_ms_threat_class_breakdown.csv
phase_n3pf_ms_raw_unsafe_action_summary.csv
phase_n3pf_ms_feature_block_stats.csv
phase_n3pf_ms_gpsi_output_summary.csv
phase_n3pf_ms_decision.csv
phase_n3pf_ms_schema_check.csv
```

## 8. Stop flags

```text
PHASE_N3PF_MS_STOP_CONFIG_MISMATCH.flag
PHASE_N3PF_MS_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3PF_MS_STOP_TRAIN_FAILED.flag
PHASE_N3PF_MS_STOP_EVAL_FAILED.flag
PHASE_N3PF_MS_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3PF_MS_STOP_PARALLEL_RESOURCE_UNSAFE.flag
PHASE_N3PF_MS_STOP_WATCHER_FAILED.flag
```

## 9. Watcher 要求

必须创建阻塞式 watcher。只有 complete flag 或 stop flag 触发时才能结束。不要中途暂停，不要只说“等待结果”。

```text
poll interval: about 120 seconds
chat/status heartbeat: about 300 seconds or longer
```

长训练不允许 60 秒刷屏。checkpoint、异常、stop、complete 时必须及时输出。
