# UAV Risk-Guided Aggregation 预实验报告

## 1. 实验目的

本预实验用于验证在轻量级三维动态避障环境中，完整 risk-guided aggregation 是否相比 learned attention aggregation 具有更好的早期实验表现。

核心对比：

- `risk_full_rbar`
- `attention_full`

两者均使用相同的 12 维障碍物 profile，区别在于聚合机制不同。

## 2. 环境配置

### 2.1 服务器环境

- OS: Ubuntu 22.04
- Python: 3.10
- PyTorch: 2.7.0
- Stable-Baselines3: 已安装
- Gymnasium: 已安装
- Device used for PPO: cpu

### 2.2 仿真环境

- 空间范围: x,y∈[-10,10], z∈[0,5]
- 障碍物数量: 3
- 无人机半径: 0.2
- 障碍物半径: 0.3
- 最大速度: UAV 2.0, obstacle 1.5
- dt: 0.2
- max_steps: 200
- 障碍物运动模式: random_switch / sudden_turn
- sudden-turn 设置: t_turn=3.0s, turn_step=15

## 3. 方法设置

### 3.1 risk_full_rbar

risk 权重：

```text
w_i = risk_i^beta / sum_j risk_j^beta
```

R_bar 调制：

```text
c = tanh(R_sum / R_ref) * sum_i w_i h_i
```

### 3.2 attention_full

learned attention：

```text
q = W_q ego
k_i = W_k h_i
w_i = softmax(q^T k_i / sqrt(d))
c = sum_i w_i h_i
```

attention 模式不使用 R_bar 乘性调制，但策略输入仍包含 `R_max, R_sum`。

## 4. 训练设置

- total_steps: 由已完成实验结果决定，见第 8 节
- n_envs: 8
- seeds: 0, 1, 2
- PPO n_steps: 1024
- batch_size: 256
- learning_rate: 3e-4
- gamma: 0.99
- gae_lambda: 0.95
- ent_coef: 0.01

## 5. 环境 sanity check

### 5.1 check_env

结果：通过。

### 5.2 random rollout

| Metric | Value |
|---|---:|
| random_success_rate | 0.0000 |
| random_collision_rate | 0.3500 |
| mean_episode_min_distance | 1.0754 |

## 6. 正式评估结果

### 6.1 每 seed 结果

| Method | Seed | Success ↑ | Collision ↓ | MinDist ↑ | Time ↓ | Reaction ↓ |
|---|---:|---:|---:|---:|---:|---:|
| risk_full_rbar | 0 | 0.9800 | 0.0200 | 1.4166 | 6.6680 | 10.6343 |
| risk_full_rbar | 1 | 1.0000 | 0.0000 | 1.2929 | 6.7500 | 11.7480 |
| risk_full_rbar | 2 | 1.0000 | 0.0000 | 1.4739 | 6.7920 | 17.9140 |
| attention_full | 0 | 0.9900 | 0.0100 | 1.4957 | 6.9720 | 0.2000 |
| attention_full | 1 | 1.0000 | 0.0000 | 1.6622 | 7.8560 | 0.2220 |
| attention_full | 2 | 1.0000 | 0.0000 | 1.5031 | 6.9660 | 1.7000 |

### 6.2 汇总结果

| Method | Success ↑ | Collision ↓ | MinDist ↑ | Reaction ↓ |
|---|---:|---:|---:|---:|
| risk_full_rbar | 0.9933 ± 0.0094 | 0.0067 ± 0.0094 | 1.3945 ± 0.0755 | 13.4321 ± 3.2016 |
| attention_full | 0.9967 ± 0.0047 | 0.0033 ± 0.0047 | 1.5537 ± 0.0768 | 0.7073 ± 0.7020 |

### 6.3 Harder Eval: random_switch_hard

该评估只用于测试泛化，不参与训练。相比标准 `eval_random_switch`，其障碍物更贴近主航线、交互比例更高、速度上限更高。

| Method | Seed | Success ↑ | Collision ↓ | MinDist ↑ | Time ↓ |
|---|---:|---:|---:|---:|---:|
| risk_full_rbar | 0 | 0.8900 | 0.1100 | 0.8047 | 6.0600 |
| risk_full_rbar | 1 | 0.9600 | 0.0400 | 0.8789 | 6.5700 |
| risk_full_rbar | 2 | 0.9800 | 0.0200 | 0.8805 | 6.7180 |
| attention_full | 0 | 0.8800 | 0.1200 | 0.8181 | 6.2240 |
| attention_full | 1 | 0.9600 | 0.0400 | 0.9554 | 7.4380 |
| attention_full | 2 | 0.9700 | 0.0300 | 0.8160 | 6.6920 |

## 7. 结果分析

### 7.1 是否支持 risk-guided aggregation

结论：

```text
基本持平
```

依据：
- success rate: risk=0.9933, attention=0.9967
- collision rate: risk=0.0067, attention=0.0033
- minimum distance: risk=1.3945, attention=1.5537
- reaction time: risk=13.4321, attention=0.7073

### 7.2 若 risk_full_rbar 优于 attention_full

说明：

```text
risk-guided aggregation 在该预实验设定下具有继续投入价值。
```

### 7.3 若两者接近

说明：

```text
该结果不直接否定 risk 方案。下一步应进一步分析 risk_rise_time、risk-threat consistency 和权重可解释性。
```

### 7.4 若 risk_full_rbar 明显差于 attention_full

优先排查：

1. risk_i 是否几乎全为 0；
2. risk_i 是否几乎全为 1；
3. Sigma 是否过大或过小；
4. distance gate 是否过强；
5. beta 是否过大；
6. R_bar 是否把 context 压得太小；
7. reward 是否鼓励硬冲目标；
8. 训练环境是否过难或过易；
9. Sigma 在匀速阶段是否衰减到 sigma_min，导致突变时 risk 上升慢。

下一步建议转向：

```text
risk-initialized attention / risk residual attention
```

形式：

```text
w_final = (1 - alpha) * w_risk + alpha * w_attention
```

## 8. 已生成文件

- checkpoints: 见 `checkpoints/`
- results csv: 见 `results/`
- TensorBoard logs: 见 `runs/`
- scripts: `scripts/check_env.py`, `scripts/random_rollout.py`, `scripts/run_preexp.sh`, `scripts/run_eval.sh`, `scripts/aggregate_results.py`
- source files: `envs/dynamic_obstacle_env.py`, `policies/obstacle_set_extractor.py`, `train.py`, `eval.py`

## 9. Codex 执行总结

说明：

1. 已完成的文件：见第 8 节。
2. 成功运行的命令：需结合本次执行日志补充。
3. 失败或跳过的命令：需结合本次执行日志补充。
4. 当前最可信结论：见第 7 节。
5. 后续建议：若正式结果不足，优先检查风险信号动态范围与 sudden-turn 响应。
