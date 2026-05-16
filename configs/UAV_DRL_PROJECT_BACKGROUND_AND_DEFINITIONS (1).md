# UAV / Dynamic Obstacle Avoidance / DRL 项目背景与定义

## 1. 项目目标

本项目研究 UAV / quadrotor 在持续动态障碍物流中的局部避障。控制接口是 continuous velocity command，学习算法主线是 PPO，环境是自定义 Gym-style `DynamicObstacleFlowEnv / EnvV2`。

目标不是单纯让 PPO 拿到更高 reward，而是让 UAV 在动态障碍物中稳定前进并降低碰撞：

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
raw_unsafe_action_rate
shield_trigger_rate
filter_delta_norm
action_delta / smoothness
scenario robustness
motion-mode robustness
threat-class robustness
```

当前项目希望尽力冲 ICRA / RA-L，因此后续必须长期注意两个审稿风险：

```text
1. 结果不能只是 seed0 或某个 checkpoint 略优。Gψ / two-step 架构比 attention_full 更复杂，必须尽量形成多 seed、shield 后、仿真/真机层面的稳定强优势。
2. baseline 不仅要强，还要常见、可被 UAV/robot dynamic obstacle avoidance/DRL 审稿人认可，例如 APF / DWA / VO / ORCA / CBF / MPC / DRL-VO / residual RL / safety-filtered RL。
```

---

## 2. EnvV2 定义

旧 3-ball Gym 环境中的 long-training no-response / reaction oscillation 没有在 EnvV2 复现，旧问题线已经结束。

当前环境为：

```text
DynamicObstacleFlowEnv / EnvV2
```

核心特征：

```text
5–8 active obstacles
obstacle replacement
multiple motion modes
threat class
planned CPA / TTC
train/eval split
多个 eval scenarios
```

当前判断：

```text
EnvV2 已冻结，不建议继续升级。
当前问题不是“PPO 长训后是否 no-response”，而是：
在 EnvV2 动态障碍物流中，learned policy 为什么还不够安全，以及 Gψ prediction / uncertainty / shield 是否能带来稳定增益。
```

---

## 3. Eval scenarios

常用 6 个 eval scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

注意：

```text
high_speed / high_threat / mixed_ood 通常更关键；
sudden_threat 不是当前主要 collision 来源；
high threat episodes 和 linear / accel_decel 等 motion modes 需要重点拆分分析。
```

---

## 4. Observation / Action / Metrics

PPO 输出：

```text
a_raw = raw velocity command
```

N4 之后 shield 输出：

```text
a_exec = shield(a_raw)
```

每个 obstacle 有当前 profile，通常包括：

```text
relative position
relative velocity
distance
closing / approach indicators
planned CPA
planned TTC
threat class
risk value
active mask
```

其中 CPA/TTC 是重要几何信息，但在 Gψ HeadA 输入中使用 CPA/TTC 存在 shortcut 风险：HeadA 标签是 obstacle 自身 world-frame residual，理论上主要由 obstacle motion dynamics 决定，而不是 UAV-relative collision geometry。后续需要 Gψ input ablation：

```text
Gψ-full
Gψ-no-cpa-ttc
Gψ-history-only-plus-kinematics
```

---

## 5. attention_full 定义

`attention_full` 是当前强 learning baseline：

```text
完整当前 obstacle profile + learned obstacle attention aggregation + PPO
```

它不用：

```text
obstacle history
Gψ
delta_hat
log σ²
shield
```

它做的是 variable obstacle set aggregation：

```text
ego feature -> query
obstacle feature -> key/value
masked softmax over active obstacles
weighted obstacle context
PPO action
```

直观理解：

```text
attention_full 学的是“当前这些障碍物里，哪些最该关注”。
```

`full` 表示使用完整 EnvV2 当前 obstacle profile，包括 CPA/TTC/threat/risk 等强几何特征。

---

## 6. Gψ / HeadA / two-step 定义

当前主线：

```text
Gψ supervised dynamic obstacle representation
+ PPO velocity policy
+ post-hoc safety shield
```

Gψ HeadA 学的是 obstacle 相对 constant-velocity extrapolation 的未来 residual：

```text
Δ_i(τ) = p_i(t+τ) - [p_i(t) + τ * v_i(t)]
```

输出：

```text
delta_hat_i(τ)
log_sigma2_i(τ)
```

常用 horizons：

```text
1s / 2s / 4s
```

当前 z_i 已降级为 ablation。主线使用 no_z / explicit HeadA features，尤其是 P3 block_projected。

---

## 7. Safety shield 定义

ordinary shield：

```text
不使用 σ²
基于 VO / CPA-TTC candidate velocity search
若 a_raw unsafe，则输出更安全的 a_exec
```

future N4-U uncertainty shield：

```text
使用 HeadA log σ² / covariance
基于危险方向投影增加 margin
在 candidate scoring 中利用 uncertainty
```

但 N4-U 当前被 P3 seed2 instability 阻塞。

---

## 8. Codex 工程约定

Codex 指南 MD 默认放在项目根目录：

```text
codex_guide/
```

只要 Codex 将要改代码，指南 MD 中必须写：

```bash
git status --short

git add -A

if git diff --cached --quiet; then
    echo "[sync] no local changes to commit"
else
    git commit -m "sync before codex changes"
fi

git push origin main
```

GitHub 仓库：

```text
https://github.com/oxpaxo/UAV-RISK-RL
```

给 Codex 的 prompt 必须提醒它去 `codex_guide/` 读取对应指南。

所有长任务要求 blocking watcher：

```text
Codex 根据指南自行判断任务完成标志；
watcher 监控任务；
只有 complete flag 或 stop flag/stop condition 触发时才结束；
否则不得中途暂停。
```

长训练 heartbeat 通常约 5 分钟一次，不要 60 秒刷屏。

当前机器：

```text
Ubuntu 22.04.5 LTS
AMD EPYC 7402
16 logical CPUs / 8 physical cores
RAM 62 GiB
RTX 3090 24GB
```

提速原则：

```text
不改变 PPO rollout semantics；
优先进程级并行；
不要随意提高单个 PPO job 的 n_envs。
```
