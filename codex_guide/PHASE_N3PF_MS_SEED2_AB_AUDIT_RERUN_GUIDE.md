# Phase N3PF-MS-AB 指南：Seed2 Collapse Audit + Minimal Rerun Sanity

> 目标：先审计 P3 block_projected 的 seed2 collapse，再做最小重跑验证。  
> 性质：不进入 N4-U；不实现 shield；不改 EnvV2-core；不 fine-tune Gψ；不改 P3 架构或 PPO 超参。

## 0. 背景

Phase N3PF-MS 结果：

```text
P3 checkpoint_1500k:
  seed0: success=0.6167, collision=0.3833
  seed1: success=0.6089, collision=0.3911
  seed2: success=0.4222, collision=0.5778

3-seed mean:
  success=0.5493, collision=0.4507

attention_full reference:
  success=0.6033, collision=0.3967
```

当前结论：

```text
P3 seed0/seed1 可比或略优 attention_full；
seed2 明显 collapse；
P3 不能被直接视为稳定 primary N4 candidate；
N4-O 结果先归档，不直接推进 N4-U；
下一步必须审计 seed2 collapse。
```

本阶段两步：

```text
Step A: Seed2 collapse audit
  不训练，只审计 config / checkpoint / eval path / train curve / feature-Gψ / behavior / breakdown。

Step B: Minimal rerun sanity
  只有 Step A 未发现硬工程错误时，单独重跑 seed2_rerunA；
  可选新增 seed3_sanity；
  判断 seed2 collapse 是否可复现。
```

---

## 1. 禁止事项

```text
禁止进入 N4-U
禁止实现 ordinary shield 或 σ² shield
禁止 action filtering / safety cost
禁止训练或 fine-tune Gψ
禁止修改 EnvV2-core
禁止改变 P3 block_projected 架构
禁止改 PPO 超参来修 seed2
禁止引入 P4/P5/P6
禁止覆盖 N3PF-MS 原始 seed1/seed2 checkpoints
禁止覆盖 N3PF / N3PF-V / N4-O 原始产物
禁止发现工程错误后继续重跑并掩盖问题
```

允许：

```text
新增审计脚本
读取原始 logs / tables / checkpoints
补充 eval 已有 intermediate checkpoints
单独重跑 seed2
新增 seed3 sanity run
生成审计与重跑报告
```

---

## 2. 输出目录

```text
results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/
```

子目录：

```text
step_a_audit/
step_b_rerun/
```

重跑 checkpoint 不得覆盖原始目录：

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s2_rerunA/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s3_sanity/
```

---

## 3. Step A：Seed2 collapse audit

### 3.1 目标

Step A 必须回答：

```text
1. seed2 config 是否与 seed1 完全一致，除 seed/out-dir/log-dir 外；
2. seed2 checkpoint 是否真实训练到 1.5M；
3. seed2 checkpoint/eval path 是否错配；
4. seed2 optimizer / parameter / policy parameter 是否正常；
5. seed2 是否在 250k/500k/750k/1000k/1250k 中曾经正常；
6. seed2 train reward / entropy / KL / clip_fraction / value_loss 是否异常；
7. seed2 action_delta / action_norm / progress / raw_unsafe 是否显示过平滑或错误策略；
8. seed2 Gψ 输出和 feature block 是否正常；
9. collapse 更像工程问题、eval问题、资源问题，还是 PPO bad local optimum。
```

### 3.2 读取原始产物

必须读取：

```text
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_config_manifest.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_command_manifest.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_checkpoint_integrity.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_train_curve.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_train_heartbeat.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_eval_summary_aggregate.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_raw_unsafe_action_summary.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_feature_block_stats.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_gpsi_output_summary.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_scenario_breakdown.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_motion_mode_breakdown.csv
results/env_v2_phase_n3pf_ms_multiseed/tables/phase_n3pf_ms_threat_class_breakdown.csv
```

以及：

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s1/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s2/
```

### 3.3 Config equivalence audit

输出：

```text
phase_n3pf_ms_seed2a_config_diff.csv
```

检查：

```text
policy_class
feature_adapter
include_z
delta_scale
logvar_clip
logvar_scale
n_envs
device
learning_rate
n_steps
batch_size
gamma
gae_lambda
clip_range
ent_coef
vf_coef
max_grad_norm
Gψ checkpoint path
Gψ frozen
EnvV2 train/eval config
```

允许不同：

```text
seed
out_dir
log_dir
checkpoint_dir
```

若有非允许差异，创建：

```text
PHASE_N3PF_MS_AB_STOP_CONFIG_MISMATCH.flag
```

### 3.4 Checkpoint / eval path audit

输出：

```text
phase_n3pf_ms_seed2a_checkpoint_path_audit.csv
phase_n3pf_ms_seed2a_checkpoint_integrity_extended.csv
```

检查：

```text
checkpoint_label
training_seed
checkpoint_path
exists
size_bytes
sha256
model_num_timesteps
expected_total_steps
optimizer_state_present
policy_parameter_l2_norm
policy_parameter_delta_vs_previous_checkpoint
eval_rows_reference_this_path
eval_seed_count
scenario_count
episode_count
```

如果 eval path 错配，创建：

```text
PHASE_N3PF_MS_AB_STOP_EVAL_PATH_MISMATCH.flag
```

如果 checkpoint 异常，创建：

```text
PHASE_N3PF_MS_AB_STOP_CHECKPOINT_INTEGRITY_FAILED.flag
```

### 3.5 Intermediate checkpoint eval

如果原 N3PF-MS 只评估了 1000k/1500k/final，则补评：

```text
seed2 checkpoint_250k
seed2 checkpoint_500k
seed2 checkpoint_750k
seed2 checkpoint_1250k
```

建议同步补 seed1 对照。

轻量 protocol：

```text
eval_seeds = 1000,1001
episodes = 50 per scenario
scenarios = all 6
```

输出：

```text
phase_n3pf_ms_seed2a_intermediate_checkpoint_eval.csv
```

### 3.6 Training curve audit

输出：

```text
phase_n3pf_ms_seed2a_training_curve_diagnostics.csv
```

比较 seed1 vs seed2：

```text
episode_reward_mean
episode_length_mean
fps
approx_kl
clip_fraction
entropy_loss
explained_variance
value_loss
policy_gradient_loss
loss
learning_rate
action_std if logged
```

### 3.7 Behavior diagnostics

输出：

```text
phase_n3pf_ms_seed2a_behavior_diagnostics.csv
```

比较：

```text
success_rate
collision_rate
progress
mean_min_distance
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
episode_length
```

重点判断：

```text
seed2_low_action_delta
seed2_conservative_but_collision
seed2_raw_unsafe_mismatch
seed2_progress_collision_mismatch
```

### 3.8 Feature / Gψ audit

输出：

```text
phase_n3pf_ms_seed2a_feature_gpsi_audit.csv
```

检查：

```text
obs_i_l2_p95
delta_hat_l2_p95
logvar_scaled_l2_p95
adapter_output_l2_p95
delta_norm_1s_p95/max
logvar span
feature_nonfinite_count
inactive_forwarded_count
```

若异常，创建：

```text
PHASE_N3PF_MS_AB_STOP_FEATURE_GPSI_DIAGNOSTICS_FAILED.flag
```

### 3.9 Step A decision

输出：

```text
phase_n3pf_ms_seed2a_decision.csv
```

决策类型：

```text
engineering_error_found
eval_path_error_found
checkpoint_integrity_error_found
resource_parallel_issue_suspected
feature_gpsi_issue_found
ppo_bad_local_optimum_likely
inconclusive
```

若硬工程错误存在：

```text
do_step_b = false
stop_after_step_a = true
```

否则：

```text
do_step_b = true
```

---

## 4. Step B：Minimal rerun sanity

### 4.1 前置

只有 Step A 无硬工程错误时才进入 Step B。若 Step A 找到 config/path/checkpoint/Gψ 硬错误，停止并生成 partial report，不要用重跑掩盖。

### 4.2 目标

判断 collapse 是否可复现：

```text
1. seed2_rerunA 是否恢复；
2. seed3_sanity 是否正常；
3. 原 seed2 是偶发坏点、并行副作用、seed敏感，还是 P3 架构稳定性不足。
```

### 4.3 Rerun jobs

最小任务：

```text
Job B1: seed2_rerunA
  seed=2
  same P3 config
  train to 1.5M
  单独运行或最高优先级
  不并行 N4-O
```

可选任务：

```text
Job B2: seed3_sanity
  seed=3
  same P3 config
  train to 1.5M
```

默认建议：

```text
先单独跑 seed2_rerunA；
资源允许再跑 seed3_sanity；
不要与 N4-O 并行。
```

### 4.4 Training config

必须与 P3 seed0/1/2 一致：

```text
include_z=false
obs/delta/logvar block projection + LayerNorm
delta_hat_scaled = delta_hat / 5.0
logvar_scaled = clamp(logvar,-5,3) / 5.0
Gψ frozen
no shield
no action filtering
no dense safety cost
n_envs=4
device=cpu
train_steps=1.5M
```

不允许改变 PPO 超参。

### 4.5 Eval protocol

对 rerun checkpoints：

```text
checkpoint_1000k
checkpoint_1500k
final
```

Eval：

```text
eval_seeds = 1000,1001,1002
scenarios = all 6
episodes = 50 per scenario
```

### 4.6 Step B outputs

```text
phase_n3pf_ms_seed2b_rerun_config_manifest.csv
phase_n3pf_ms_seed2b_command_manifest.csv
phase_n3pf_ms_seed2b_resource_affinity.csv
phase_n3pf_ms_seed2b_checkpoint_integrity.csv
phase_n3pf_ms_seed2b_train_curve.csv
phase_n3pf_ms_seed2b_eval_summary_aggregate.csv
phase_n3pf_ms_seed2b_eval_summary_by_seed.csv
phase_n3pf_ms_seed2b_scenario_breakdown.csv
phase_n3pf_ms_seed2b_motion_mode_breakdown.csv
phase_n3pf_ms_seed2b_threat_class_breakdown.csv
phase_n3pf_ms_seed2b_raw_unsafe_action_summary.csv
phase_n3pf_ms_seed2b_behavior_diagnostics.csv
phase_n3pf_ms_seed2b_decision.csv
```

Plots：

```text
seed2_original_vs_rerun_success_collision.png
seed2_rerun_training_curve.png
seed2_original_vs_rerun_behavior.png
seed2_seed3_sanity_comparison.png
```

### 4.7 Step B decision

```text
If seed2_rerunA recovers to success >= 0.58 and collision <= 0.42:
  seed2 original likely bad training run or parallel/randomness issue.
  Recommend adding seed3 or using rerun as replacement only with transparent reporting.

If seed2_rerunA still collapses and seed3 is healthy:
  seed=2 may be unlucky but architecture is seed-sensitive.
  Need more seeds or stabilization.

If seed2_rerunA collapses and seed3 also collapses:
  P3 architecture/training stability is insufficient.
  Do not proceed N4-U as P3-primary.
  Start stabilization design phase.

If seed2_rerunA recovers but seed3 collapses:
  P3 has high variance.
  Need more seeds or training stabilization.

If inconclusive:
  Do not claim P3 stable.
```

---

## 5. Final report

输出：

```text
PHASE_N3PF_MS_SEED2_AB_AUDIT_RERUN_REPORT.md
PHASE_N3PF_MS_SEED2_AB_AUDIT_RERUN_COMPLETE.flag
phase_n3pf_ms_seed2_ab_status.txt
phase_n3pf_ms_seed2_ab_watcher.log
```

报告必须包含：

```text
1. Step A audit summary；
2. 是否发现 config/path/checkpoint/Gψ/feature 工程错误；
3. seed2 collapse 最可能原因；
4. Step B 是否执行；
5. seed2_rerunA 是否恢复；
6. seed3_sanity 是否正常；
7. 是否允许恢复 P3 primary N4 candidate；
8. 是否可以继续 N4-U；
9. 是否需要 P3 stabilization phase；
10. 是否需要 attention_full multi-seed；
11. 后续建议。
```

---

## 6. Completion criteria

只有满足以下条件才生成 complete flag：

```text
1. Step A 审计完成；
2. Step A decision 生成；
3. 若 Step A 无硬错误，Step B 至少完成 seed2_rerunA；
4. 若资源允许，seed3_sanity 也完成；
5. final report 生成；
6. watcher log/status 存在。
```

---

## 7. Stop flags

```text
PHASE_N3PF_MS_AB_STOP_CONFIG_MISMATCH.flag
PHASE_N3PF_MS_AB_STOP_EVAL_PATH_MISMATCH.flag
PHASE_N3PF_MS_AB_STOP_CHECKPOINT_INTEGRITY_FAILED.flag
PHASE_N3PF_MS_AB_STOP_FEATURE_GPSI_DIAGNOSTICS_FAILED.flag
PHASE_N3PF_MS_AB_STOP_TRAIN_FAILED.flag
PHASE_N3PF_MS_AB_STOP_EVAL_FAILED.flag
PHASE_N3PF_MS_AB_STOP_RESOURCE_UNSAFE.flag
PHASE_N3PF_MS_AB_STOP_WATCHER_FAILED.flag
```

---

## 8. Suggested commands

Codex should adapt paths.

```bash
python -m py_compile scripts/audit_env_v2_phase_n3pf_ms_seed2.py
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3pf_ms_seed2_rerun.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3pf_ms_seed2_ab.py
python -m py_compile scripts/analyze_env_v2_phase_n3pf_ms_seed2_ab.py
bash -n scripts/watch_phase_n3pf_ms_seed2_ab.sh
chmod +x scripts/watch_phase_n3pf_ms_seed2_ab.sh
```

Step A:

```bash
python scripts/audit_env_v2_phase_n3pf_ms_seed2.py   --source-result-dir results/env_v2_phase_n3pf_ms_multiseed   --out-dir results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit   --seed-good 1   --seed-bad 2   --include-seed0   --write-plots
```

Step B seed2 rerun:

```bash
python scripts/train_env_v2_gpsi_ppo_n3pf_ms_seed2_rerun.py   --seed 2   --run-name seed2_rerunA   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s2_rerunA   --train-steps 1500000   --n-envs 4   --device cpu   --heartbeat-seconds 300
```

Step B optional seed3 sanity:

```bash
python scripts/train_env_v2_gpsi_ppo_n3pf_ms_seed2_rerun.py   --seed 3   --run-name seed3_sanity   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s3_sanity   --train-steps 1500000   --n-envs 4   --device cpu   --heartbeat-seconds 300
```

Step B eval:

```bash
python scripts/eval_env_v2_gpsi_ppo_n3pf_ms_seed2_ab.py   --mode rerun_eval   --out-dir results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun   --runs seed2_rerunA seed3_sanity   --checkpoints checkpoint_1000k checkpoint_1500k final   --eval-seeds 1000 1001 1002   --num-episodes 50   --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat
```

Final analysis:

```bash
python scripts/analyze_env_v2_phase_n3pf_ms_seed2_ab.py   --result-dir results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun   --attention-success 0.6033   --attention-collision 0.3967   --noz-success 0.5667   --noz-collision 0.4333   --p3-s0-success 0.6167   --p3-s0-collision 0.3833   --p3-s1-success 0.6089   --p3-s1-collision 0.3911   --p3-s2-success 0.4222   --p3-s2-collision 0.5778
```

Watcher:

```bash
bash scripts/watch_phase_n3pf_ms_seed2_ab.sh
```

---

## 9. Watcher requirements

必须创建阻塞式 watcher：

```text
poll interval: about 120 seconds
chat/status heartbeat: about 300 seconds or longer
```

只有以下情况可以结束：

```text
1. complete flag 出现；
2. stop flag 出现。
```

不得中途暂停，不得只说等待结果。长训练不得 60 秒刷屏。
