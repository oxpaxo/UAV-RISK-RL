# Phase N3 指南：Gψ-HeadA 接入 PPO（Frozen Gψ / No Shield）

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 新主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3 - Gψ-HeadA + PPO No Shield`  
> 阶段性质：训练 PPO raw velocity policy；Gψ frozen；不接 shield；不训练 Gψ；不改 EnvV2-core。  
> 前置条件：Phase N2 已完成，存在 `PHASE_N2_HEADA_OFFLINE_COMPLETE.flag`，且 `work_dirs/gpsi_heada_v1_nll/best.pth` 可用。

---

## 0. 背景与当前状态

Phase N2 已完成离线 Head A pilot。核心事实：

```text
Zero residual test MSE: 0.466866
Delta-only test MSE:    0.099426
NLL model test MSE:     0.096426
NLL Gaussian NLL:       -1.793691
Mean nonlinear MSE improvement over zero: 0.6692
Projected uncertainty corr on radial / relative-velocity directions: 0.6042
```

N2 表明：

```text
1. Head A residual prediction 可学；
2. diagonal logvar 没有 collapse；
3. projected uncertainty 有稳定正相关，可作为后续 N4 directional / tube shield 的基础；
4. 但 N3 不测试 shield，只测试 Gψ 增强特征是否能改善 raw PPO policy。
```

Phase N3 的问题是：

```text
只把 frozen Gψ 输出作为 PPO observation augmentation，
不接后置 shield，
PPO raw policy 是否比 attention_full 更安全 / 更稳？
```

---

## 1. Phase N3 总目标

训练一个新 PPO policy：

```text
Gψ frozen
PPO trainable
obs_i_aug = [obs_i, z_i, delta_hat_i, logvar_hat_i]
masked-attention PPO backbone
symmetric actor-critic
no shield
```

N3 必须输出：

```text
1. Gψ online inference wrapper / feature adapter；
2. frozen-Gψ PPO training checkpoint；
3. checkpoint eval results；
4. attention_full_1500k reference comparison；
5. raw action unsafe diagnostics；
6. Gψ output trace diagnostics；
7. Phase N3 report + complete/stop flag + watcher log。
```

N3 不要求一定超过 Phase B 的 VO-like shield。N3 的主要对照是：

```text
attention_full_1500k
Gψ-HeadA + PPO no shield
```

Phase B geometry/filter baselines只作为背景上界 / 强 baseline context，不作为 N3 的直接目标。

---

## 2. 明确禁止事项

Phase N3 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止训练或 fine-tune Gψ；
3. 禁止接入任何 Safety Shield；
4. 禁止 action filtering / action projection；
5. 禁止训练 safety-cost PPO；
6. 禁止实现 learned R(s,a)、candidate velocity risk map、5-head Gψ；
7. 禁止使用 future label / future trajectory 作为 PPO input；
8. 禁止把 eval scenarios 混入 PPO training；
9. 禁止 PPO 反向传播更新 Gψ；
10. 禁止只看 aggregate success/collision 而不输出 motion-mode / scenario / raw unsafe diagnostics。
```

允许：

```text
1. 新增 Env wrapper / policy feature extractor；
2. 新增 Gψ frozen inference adapter；
3. 新增 PPO training script；
4. 新增 eval script 或复用 Phase A/B eval runner；
5. 新增 trace fields；
6. 新增 report / watcher / flags。
```

---

## 3. 推荐新增文件与目录

```text
models/
  gpsi_head_a.py                       # N2 已有，可复用
  gpsi_ppo_policy.py                   # 可选：attention-compatible policy / feature extractor

envs/wrappers/
  gpsi_obs_wrapper.py                  # 推荐：维护 obstacle history，计算 Gψ augmentation

scripts/
  train_env_v2_gpsi_ppo.py
  eval_env_v2_gpsi_ppo.py
  analyze_env_v2_phase_n3_results.py
  watch_phase_n3_gpsi_ppo_no_shield.sh

configs/
  env_v2_gpsi_heada_attention_ppo_v1.yaml

checkpoints/
  env_v2_gpsi_heada_ppo_s0/
    checkpoint_250k.zip
    checkpoint_500k.zip
    checkpoint_1000k.zip
    checkpoint_1500k.zip
    final.zip
    best_by_eval.zip
    config_resolved.yaml

results/
  env_v2_phase_n3_gpsi_ppo_no_shield/
    PHASE_N3_GPSI_PPO_NO_SHIELD_REPORT.md
    PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag
    phase_n3_status.txt
    phase_n3_watcher.log

    tables/
      phase_n3_train_curve.csv
      phase_n3_eval_summary.csv
      phase_n3_episode_metrics.csv
      phase_n3_checkpoint_eval_summary.csv
      phase_n3_attention_reference_comparison.csv
      phase_n3_scenario_breakdown.csv
      phase_n3_motion_mode_breakdown.csv
      phase_n3_raw_unsafe_action_summary.csv
      phase_n3_gpsi_output_summary.csv
      phase_n3_gpsi_forward_profile.csv
      phase_n3_schema_check.csv
      phase_n3_command_manifest.csv

    traces/
      sampled_success_traces/
      sampled_collision_traces/
      sampled_near_miss_traces/

    plots/
      train_reward_curve.png
      train_success_collision_curve.png
      checkpoint_success_collision.png
      scenario_breakdown.png
      motion_mode_breakdown.png
      raw_unsafe_rate_by_checkpoint.png
      gpsi_delta_norm_distribution.png
      gpsi_logvar_distribution.png

    logs/
      phase_n3_train.log
      phase_n3_eval.log
      phase_n3_analysis.log
```

---

## 4. Gψ online integration design

### 4.1 推荐实现：Env wrapper 负责 Gψ augmentation

由于 Gψ 需要 obstacle history 和 obstacle identity，推荐新增：

```text
envs/wrappers/gpsi_obs_wrapper.py
```

wrapper 职责：

```text
1. reset 时清空每个 env instance 的 obstacle history buffer；
2. step 后读取 info 中 active obstacle ids / positions / velocities；
3. 按 (episode_id, obstacle_id) 维护 history_i[t-H:t]；
4. replacement 后根据 obstacle_id 自动断开旧 history；
5. left-pad 缺失 history，并生成 history_valid_mask；
6. 用 torch.no_grad() 调用 frozen Gψ；
7. 构造 obs_i_aug；
8. 返回给 PPO 的 augmented observation。
```

不建议把 Gψ history 维护塞进 SB3 feature extractor，因为 feature extractor 通常只看到 observation，不稳定获得 info / obstacle_id。

### 4.2 Gψ 必须 frozen

必须满足：

```python
gpsi.eval()
for p in gpsi.parameters():
    p.requires_grad_(False)
```

PPO 训练期间：

```text
1. Gψ checkpoint 不更新；
2. 不保存 Gψ optimizer；
3. 不允许 PPO loss 反向传播到 Gψ；
4. Gψ forward 使用 torch.no_grad()。
```

### 4.3 输入字段

Gψ online input 必须与 N2 训练一致：

```text
ego_current
obs_current
history_rel_pos
history_rel_vel
history_valid_mask
```

禁止输入：

```text
future_pos_world
delta_label_world
future_valid_mask
motion_mode label as default input
```

motion_mode / threat_class 可用于 trace 和 analysis，不建议作为 Gψ forward input，除非 N2 训练时已明确使用。

---

## 5. PPO augmented observation schema

原 EnvV2 per-obstacle profile：

```text
obs_i ∈ R^12
```

Gψ 输出：

```text
z_i ∈ R^64
delta_hat_i ∈ R^[T,D]
logvar_hat_i ∈ R^[T,D]
T = 3 horizons: 1s, 2s, 4s
D = 3 axes: x,y,z
```

默认 augmented profile：

```text
obs_i_aug = concat(
  obs_i,                    # 12
  z_i,                      # 64
  delta_hat_i.flatten(),    # 9
  logvar_hat_i.flatten()    # 9
)

obs_i_aug_dim = 94
```

如果实现选择只给 PPO xy：

```text
obs_i_aug_dim = 12 + 64 + 6 + 6 = 88
```

但第一版推荐保留 xyz，后续由 PPO 自行学习是否使用 z。

### 5.1 augmentation normalization

必须避免 raw delta / logvar 数值尺度直接破坏 PPO。

推荐：

```text
delta_hat_norm = delta_hat / delta_scale
logvar_hat_clamped = clamp(logvar_hat, -5, 3)
z_i 可直接使用，或用 running mean/std 归一化，但必须记录。
```

`delta_scale` 必须来自 train split / N2 checkpoint 保存的 normalization stats，不能使用 val/test。

报告必须写清楚：

```text
1. delta_hat 是否 normalized；
2. logvar_hat 是否直接输入；
3. z_i 是否 normalized；
4. normalization stats 来源。
```

---

## 6. PPO backbone

第一版必须沿用 attention_full-compatible / masked attention 聚合。

```text
ego encoder: same or compatible with attention_full
obstacle encoder: MLP over obs_i_aug
masked attention over active obstacles
actor: velocity command a_raw
critic: symmetric input, same augmented observation
```

禁止第一版同时改成：

```text
nearest-K MLP
passing-rule block
asymmetric critic
V_nom / p_trigger derived features
```

这些留给 N5 / later ablation。

---

## 7. Training protocol

### 7.1 训练目标

保持 EnvV2 原 reward，不加入 safety cost。

```text
reward = original EnvV2 reward
no dense safety cost
no shield
no action filtering
```

目标是单独测试 Gψ augmentation 对 raw policy 的影响。

### 7.2 训练预算

为了公平对齐 `attention_full_1500k`，推荐正式训练：

```text
train_steps = 1,500,000
seed = 0
```

必须保存中间 checkpoints：

```text
250k
500k
1000k
1500k
```

如果资源有限，可以先完成 500k pilot，但不允许生成 complete flag 伪装成正式 N3。资源不足时应触发：

```text
PHASE_N3_STOP_RESOURCE_LIMIT.flag
```

并生成 partial report。

### 7.3 PPO 超参数

优先复用 attention_full_1500k 的 PPO 超参数：

```text
gamma
gae_lambda
n_steps
batch_size
learning_rate
clip_range
ent_coef
vf_coef
network hidden sizes
normalization / VecNormalize 设置
```

如果必须调整，报告必须说明原因。

---

## 8. Evaluation protocol

### 8.1 eval scenarios

必须使用 6 个 Eval scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

### 8.2 episodes

正式 eval：

```text
50 episodes per scenario per checkpoint/method
```

至少评估：

```text
attention_full_1500k reference
gpsi_heada_ppo checkpoint_500k
gpsi_heada_ppo checkpoint_1000k
gpsi_heada_ppo checkpoint_1500k / final
```

如果重新评估 attention_full 成本过高，可以引用 Phase B formal results，但必须在 report 中注明来源和协议一致性。

### 8.3 主指标

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
episode_min_distance
mean_time / episode_length
episode_return
```

### 8.4 必须输出 breakdown

```text
scenario-wise breakdown
motion-mode breakdown
threat-class breakdown
checkpoint-wise breakdown
```

---

## 9. Raw action unsafe diagnostics

N3 不接 shield，但必须诊断 raw action 是否 unsafe。

对每一步 raw action `a_raw` 计算 CPA/TTC 安全诊断：

```text
v_cmd = action_xy * v_uav_max
rel = obs_pos_xy - uav_pos_xy
rel_vel_pred = obs_vel_xy - v_cmd

tcpa = clip(-dot(rel, rel_vel_pred) / (||rel_vel_pred||^2 + eps), 0, horizon)
cpa = ||rel + rel_vel_pred * tcpa||
```

记录：

```text
raw_min_predicted_cpa
raw_min_predicted_ttc
raw_unsafe_action
raw_unsafe_obstacle_id
raw_unsafe_motion_mode
raw_unsafe_threat_class
```

建议 unsafe 判定：

```text
0 < tcpa < 4.5 and cpa < 1.2
```

同时可记录 cpa_safe=1.5 的辅助统计。

必须输出：

```text
phase_n3_raw_unsafe_action_summary.csv
```

按：

```text
method
checkpoint
scenario
motion_mode
threat_class
```

统计 raw unsafe rate。

---

## 10. Gψ output diagnostics

N3 trace 必须记录 Gψ 输出摘要，避免 PPO 是否使用 Gψ 无法解释。

每步至少记录：

```text
mean_delta_norm_1s
mean_delta_norm_2s
mean_delta_norm_4s
max_delta_norm_1s
max_delta_norm_2s
max_delta_norm_4s
mean_logvar_xy_1s
mean_logvar_xy_2s
mean_logvar_xy_4s
max_logvar_xy_1s
max_logvar_xy_2s
max_logvar_xy_4s
nearest_obstacle_delta_norm_1s/2s/4s
nearest_obstacle_logvar_xy_1s/2s/4s
projected_std_radial_nearest
projected_std_relvel_nearest
history_valid_ratio_nearest
```

必须输出：

```text
phase_n3_gpsi_output_summary.csv
```

按 scenario / motion mode / checkpoint 汇总。

---

## 11. Trace 要求

N3 trace 应复用 Phase A/B 统一 trace schema，并新增 Gψ 字段。

每个 checkpoint 至少保存：

```text
1. sampled success traces；
2. sampled collision traces；
3. sampled near-miss traces；
4. high raw-unsafe-rate traces。
```

`action_executed = action_raw`，且必须记录：

```text
filter_used = false
filter_triggered = false
filter_delta_norm = 0
```

以便后续 N4 与 shielded methods 对齐。

---

## 12. 命令清单

Codex 应根据实际 repo 调整命令，并在 report 中记录最终命令。

### 12.1 编译检查

```bash
python -m py_compile envs/wrappers/gpsi_obs_wrapper.py
python -m py_compile models/gpsi_ppo_policy.py
python -m py_compile scripts/train_env_v2_gpsi_ppo.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo.py
python -m py_compile scripts/analyze_env_v2_phase_n3_results.py
bash -n scripts/watch_phase_n3_gpsi_ppo_no_shield.sh
chmod +x scripts/watch_phase_n3_gpsi_ppo_no_shield.sh
```

如果某些文件不需要，例如没有 `models/gpsi_ppo_policy.py`，可省略，但 report 必须说明实际实现位置。

### 12.2 smoke test

```bash
python scripts/train_env_v2_gpsi_ppo.py \
  --config configs/env_v2_gpsi_heada_attention_ppo_v1.yaml \
  --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_s0_smoke \
  --train-steps 2048 \
  --seed 0 \
  --smoke-test
```

### 12.3 formal training

```bash
python scripts/train_env_v2_gpsi_ppo.py \
  --config configs/env_v2_gpsi_heada_attention_ppo_v1.yaml \
  --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_s0 \
  --train-steps 1500000 \
  --checkpoint-steps 250000 500000 1000000 1500000 \
  --seed 0
```

### 12.4 eval

```bash
python scripts/eval_env_v2_gpsi_ppo.py \
  --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --ppo-checkpoints \
    checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_500k.zip \
    checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1000k.zip \
    checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1500k.zip \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --out-dir results/env_v2_phase_n3_gpsi_ppo_no_shield \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --num-episodes 50 \
  --eval-seed 1000 \
  --write-traces \
  --raw-unsafe-diagnostics \
  --gpsi-output-diagnostics
```

### 12.5 analysis

```bash
python scripts/analyze_env_v2_phase_n3_results.py \
  --result-dir results/env_v2_phase_n3_gpsi_ppo_no_shield
```

### 12.6 watcher

```bash
bash scripts/watch_phase_n3_gpsi_ppo_no_shield.sh
```

---

## 13. Report 要求

必须输出：

```text
results/env_v2_phase_n3_gpsi_ppo_no_shield/PHASE_N3_GPSI_PPO_NO_SHIELD_REPORT.md
```

报告至少包含：

```text
1. 背景与 N3 目标；
2. N2 dependency check；
3. Gψ checkpoint 信息；
4. Gψ frozen 证明；
5. online history buffer / wrapper 设计；
6. obs_i_aug schema；
7. augmentation normalization；
8. PPO backbone 与超参数；
9. training curve；
10. checkpoint eval summary；
11. attention_full_1500k comparison；
12. scenario-wise breakdown；
13. motion-mode breakdown；
14. raw action unsafe diagnostics；
15. Gψ output diagnostics；
16. sampled traces 路径；
17. 与 Phase B geometry/filter baseline 的背景对照；
18. 是否建议进入 N4 shield comparison；
19. risks / warnings；
20. terminal_decision。
```

报告必须明确区分：

```text
experiment-supported facts
reasonable inferences
risks / unresolved issues
```

---

## 14. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag
```

必须满足：

```text
1. Phase N2 complete flag 存在；
2. Gψ NLL checkpoint 可加载；
3. Gψ 在 PPO 训练中 frozen；
4. online history wrapper 可运行；
5. no future label leakage；
6. PPO smoke test 通过；
7. formal PPO training 完成到指定 train_steps，或明确资源 stop；
8. checkpoints 存在；
9. eval 在 6 scenarios 完成；
10. attention_full reference comparison 存在；
11. scenario / motion-mode breakdown 存在；
12. raw unsafe diagnostics 存在；
13. Gψ output diagnostics 存在；
14. sampled traces 存在；
15. report 生成；
16. watcher log 与 status 文件存在；
17. 无未解释的 NaN/inf、training crash、schema mismatch。
```

### 14.1 方法效果不是 complete 的硬条件

如果 Gψ-PPO no shield 没有超过 attention_full，但训练和评估完整，仍可生成 complete flag。报告必须写明：

```text
N3 complete, but Gψ no-shield improvement is weak/negative.
N4 may still proceed because N2 suggests Gψ uncertainty may be more valuable on shield side.
```

但如果出现：

```text
success 接近 0；
collision 不降反升且训练明显异常；
Gψ forward 产生错误 / NaN；
PPO 学不到任何基本 goal progress；
```

则应触发 stop 或 warning，由 report 明确。

---

## 15. 停止条件

如出现以下问题，必须生成对应 stop flag、partial report 和 log：

```text
PHASE_N3_STOP_PHASE_N2_MISSING.flag
PHASE_N3_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3_STOP_GPSI_WRAPPER_FAILED.flag
PHASE_N3_STOP_GPSI_NOT_FROZEN.flag
PHASE_N3_STOP_SCHEMA_MISMATCH.flag
PHASE_N3_STOP_TRAIN_FAILED.flag
PHASE_N3_STOP_EVAL_FAILED.flag
PHASE_N3_STOP_RESOURCE_LIMIT.flag
PHASE_N3_STOP_TRACE_DIAGNOSTICS_FAILED.flag
PHASE_N3_STOP_WATCHER_FAILED.flag
```

触发示例：

```text
N2 complete flag 不存在；
Gψ checkpoint 不存在或无法加载；
online history / obstacle id 对齐失败；
PPO 训练时 Gψ 参数发生变化；
augmented observation shape 与 policy 不一致；
训练 NaN/inf；
eval 无法完成；
资源不足以完成正式 train_steps；
无法输出 raw unsafe / Gψ diagnostics；
watcher 失败。
```

---

## 16. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n3_gpsi_ppo_no_shield.sh
```

watcher 必须：

```text
1. 检查 Phase N2 complete flag；
2. 检查 Gψ NLL checkpoint；
3. 运行 smoke test；
4. 运行 formal PPO training；
5. 运行 eval；
6. 运行 analysis；
7. 持续轮询 complete / stop flag；
8. 持续输出状态；
9. 只有 complete flag 或 stop flag 出现才退出；
10. 不允许中途“等待用户确认”；
11. 不允许因为暂无新日志而退出；
12. 写入 phase_n3_watcher.log；
13. 写入 phase_n3_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3_gpsi_ppo_no_shield"
LOG="$OUT_DIR/phase_n3_watcher.log"
STATUS="$OUT_DIR/phase_n3_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces"

echo "[watcher] Phase N3 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3_STOP_PHASE_N2_MISSING.flag"
fi

(
  python scripts/train_env_v2_gpsi_ppo.py \
    --config configs/env_v2_gpsi_heada_attention_ppo_v1.yaml \
    --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_s0_smoke \
    --train-steps 2048 \
    --seed 0 \
    --smoke-test

  python scripts/train_env_v2_gpsi_ppo.py \
    --config configs/env_v2_gpsi_heada_attention_ppo_v1.yaml \
    --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_s0 \
    --train-steps 1500000 \
    --checkpoint-steps 250000 500000 1000000 1500000 \
    --seed 0

  python scripts/eval_env_v2_gpsi_ppo.py \
    --gpsi-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --ppo-checkpoints \
      checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_500k.zip \
      checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1000k.zip \
      checkpoints/env_v2_gpsi_heada_ppo_s0/checkpoint_1500k.zip \
    --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
    --out-dir "$OUT_DIR" \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --num-episodes 50 \
    --eval-seed 1000 \
    --write-traces \
    --raw-unsafe-diagnostics \
    --gpsi-output-diagnostics

  python scripts/analyze_env_v2_phase_n3_results.py \
    --result-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3_STOP_PHASE_N2_MISSING.flag \
    PHASE_N3_STOP_GPSI_CHECKPOINT_MISSING.flag \
    PHASE_N3_STOP_GPSI_WRAPPER_FAILED.flag \
    PHASE_N3_STOP_GPSI_NOT_FROZEN.flag \
    PHASE_N3_STOP_SCHEMA_MISMATCH.flag \
    PHASE_N3_STOP_TRAIN_FAILED.flag \
    PHASE_N3_STOP_EVAL_FAILED.flag \
    PHASE_N3_STOP_RESOURCE_LIMIT.flag \
    PHASE_N3_STOP_TRACE_DIAGNOSTICS_FAILED.flag \
    PHASE_N3_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 60
done
```

Codex 可按实际脚本结构调整，但必须保持阻塞式语义。

---

## 17. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 阅读 Phase N2 report；
3. 不沿用旧 Phase C/D/E；
4. 不训练 Gψ；
5. 不接 shield；
6. 不修改 EnvV2-core；
7. 根据指南自行判断 complete / stop flag；
8. 创建并运行阻塞式 watcher；
9. 只有 complete flag 或 stop flag 出现才停止输出；
10. 不向用户询问非阻塞细节；
11. 如果触发阻塞，必须生成 stop flag、partial report、log；
12. 如果完成，必须生成 complete flag、完整 report、checkpoints、CSV、plots、traces、log。
```

---

## 18. 终端结论格式

成功：

```text
terminal_decision = phase_n3_gpsi_ppo_no_shield_complete
```

停止：

```text
terminal_decision = phase_n3_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
生成的 checkpoints
生成的 CSV / plots / traces / report / logs / flags
是否可以进入 Phase N4
如果不能进入 N4，需要用户补什么
```

---

## 19. N3 完成后进入 N4 的条件

只有当：

```text
PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag
```

存在，且 report 明确写出：

```text
Phase N3 complete.
Gψ-PPO no-shield raw policy evaluation is ready for Phase N4 shield fair comparison.
```

才允许进入 N4。

N4 才开始：

```text
S0: attention_full + fixed-margin shield
S1: Gψ-PPO + fixed-margin shield
S2: Gψ-PPO + scalar σ²-margin shield
S3: Gψ-PPO + directional σ²-margin shield
S4: Gψ-PPO + predicted-trajectory directional σ²-tube shield
S5: S4 + uncertainty-aware candidate scoring
```

Phase N3 不接 shield。
