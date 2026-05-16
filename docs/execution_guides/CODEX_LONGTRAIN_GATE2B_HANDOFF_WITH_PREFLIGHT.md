# 新 Codex 交接说明：必须先读

> 本文件是下一阶段 Codex 执行指南。  
> **请不要一上来直接启动 2000k longtrain 或 Gate-2b 主训练。**  
> 先完成本节的 Preflight Patch，确保 cost 量级、reaction 口径、trace 字段、checkpoint 索引和 Gate-2b 配置日志都可靠；否则后续结果很可能无法解释。

---

## A. 研究背景与当前方向

本课题是 UAV 动态障碍物避障 / 速度规划预实验。最初目标是验证：

```text
EWMA motion-uncertainty risk
+
risk-guided aggregation
+
PPO policy
```

能否胜过 learned attention aggregation。

经过多轮实验，方向已经发生变化：

```text
早期问题：
risk 是否能替代 attention？

当前问题：
PPO 长训下是否存在 safety/reaction oscillation？
risk cost 是否比普通 distance warning cost 有独立价值？
```

已有关键结果：

```text
1. pure EWMA-risk hard weighting 不能直接替代 learned attention。
2. 修正后的 EWMA-risk 在 100k/200k 早期 checkpoint 表现很好，但 300k/500k 后 sudden-turn reaction 明显退化。
3. attention_full seed=0 1000k gate 也出现 long-training safety/reaction oscillation：
   sudden_turn reaction_time:
     250k: 0.208
     500k: 1.860
     750k: 11.424
     1000k: 6.076
4. attention 在 750k 最差、1000k 有一定恢复，因此不能简单说“单调退化”，更准确是 checkpoint-level safety/reaction oscillation。
5. 当前最重要的实验问题是：
   - attention 和 risk 到 2000k 后是恢复、继续退化，还是震荡？
   - distance_penalty 是否能抑制 drift？
   - risk_penalty 是否优于 distance_penalty？
   - risk_biased_attention + risk_penalty 是否优于 risk_penalty only？
   - risk cost 是否真的早于 distance warning cost 上升？
```

当前研究口径：

```text
不要再把问题表述为“risk 比 attention 更聪明”。
更稳妥的表述是：
PPO-based dynamic obstacle avoidance 中可能存在 long-training safety/reaction oscillation；
motion-uncertainty risk 是否能作为比普通 distance warning 更早、更有效的 safety signal，需要 Gate-2b 验证。
```

---

## B. 当前代码状态与已确认事项

根据上一轮 Codex 自查，当前代码状态如下：

```text
1. train.py 已支持 resume:
   --resume_from
   --resume_global_step
   --save_checkpoints
   --checkpoint_steps
   --checkpoint_dir

2. resume 逻辑：
   model = PPO.load(checkpoint_path, env=env, device=device)
   model.learn(total_timesteps=remaining, reset_num_timesteps=False)

3. checkpoint 命名已支持 global step：
   global_step = resume_global_step + local_step

4. Rgate8_lambda015_RbarFloor03 已确认真实生效：
   R_gate = 8.0
   lambda_ewma = 0.15
   rbar_floor = 0.3
   R_bar = tanh(R_sum / r_ref)
   if rbar_floor > 0: R_bar = clamp(R_bar, min=rbar_floor)

5. Gate-2b 的 SafetyCostWrapper 已实现：
   shaped_reward = base_reward - beta_cost * cost_t
   distance_warning 使用 info["distance_warning_cost"]
   risk_sum 使用 info["risk_sum"]

6. risk_biased_attention 已在 logits 层面实现：
   scores = learned_score + lambda_bias * log(risk_i + eps)
   再进行 softmax

重要！！！当你读到这一行的时候，这是来自实验员的提示（你可以在输出窗口里输出提示信息，表示你已经了解这个情况了）：因为操作原因，codex把以上几条代码状态中描述的代码和文件等回退了，它们实际基本已经不存在；你看到的以目录下实际的代码状态为准，但是下面的关键缺口你仍需注意，因为它们是你切实可能会出现的问题，应该避免：
```

仍有关键缺口：

```text
1. attention gate 旧 eval CSV 还没有 cost 分位数字段；
2. beta_cost=5.0 是否对 distance_warning_cost 和 risk_sum 公平，尚未验证；
3. eval.py 与 diagnose_sudden_turn.py 的 reaction_time 未反应处理不一致；
4. trace 字段还不完整，暂时不足以严格判断 risk 是否早于 distance cost 上升；
5. 还没有统一的 checkpoint/eval/trace 索引总表；
6. Gate-2b 主实验尚未真正开始。
```

因此必须先完成下面的 Preflight Patch。

---

## C. Preflight Patch：主训练前必须先完成

### Patch 1：补跑 attention gate 的 cost 统计

不要直接启动 longtrain / Gate-2b。先用已有 `attention_full seed=0` 的：

```text
250k / 500k / 750k / 1000k checkpoint
```

补跑：

```text
eval_random_switch
eval_sudden_turn
```

每个场景：

```text
episodes = 50
eval_seed = 1000
```

必须输出：

```text
distance_warning_cost:
  p50 / p90 / p95 / max

risk_sum:
  p50 / p90 / p95 / max

risk_max:
  p50 / p90 / p95 / max

near_miss_rate
mean_min_distance
success_rate
collision_rate
reaction_time
```

并计算：

```text
beta_cost = 5.0 时：

beta_cost * distance_warning_cost_p90
beta_cost * risk_sum_p90
beta_cost * risk_max_p90
```

判断目标：

```text
如果 collision penalty 约为 -10，
则 beta_cost * cost_p90 最好落在 1–5 区间。
```

输出文件：

```text
results/preflight/cost_stats_attention_gate.csv
results/preflight/COST_SCALE_PREFLIGHT_REPORT.md
```

报告必须回答：

```text
1. distance_warning_cost 与 risk_sum 量级是否可比？
2. beta_cost=5.0 是否明显过大或过小？
3. Gate-2b 是否可以统一使用 beta_cost=5.0？
4. 如果不可以，建议 distance_penalty 和 risk_penalty 分别使用什么 beta_cost？
```

---

### Patch 2：统一 reaction_time 口径

当前已知：

```text
eval.py:
  deviation_lateral > 0.3
  连续 2 步满足
  未反应 episode 用 max_episode_time - turn_time 填充

diagnose_sudden_turn.py:
  阈值与连续步相同
  但未反应 episode 返回 NaN
```

这会导致不同表里的 mean_reaction_time 不可直接比较。

Codex 需要做：

```text
1. 保留 eval.py 的正式评估口径；
2. 在 diagnostic summary 中同时输出两种 reaction：
   - mean_reaction_eval_style
   - mean_reaction_nan_style
3. 同时输出：
   - nan_reaction_rate
   - no_response_count
   - total_episodes
4. 在所有报告中说明未反应 episode 的处理方式。
```

推荐统一函数：

```text
reaction_time_eval_style:
  未反应 -> max_episode_time - turn_time

reaction_time_nan_style:
  未反应 -> NaN
```

输出文件：

```text
results/preflight/reaction_definition_check.md
```

必须回答：

```text
1. eval.py 和 diagnostic script 是否共用同一个 reaction flag 逻辑？
2. 未反应 episode 在两个口径下如何处理？
3. 旧数据中 formal eval reaction 与 diagnostic reaction 的不一致是否主要由 NaN/上限填充差异造成？
```

---

### Patch 3：补全 --save_trace 字段

如果 `eval.py --save_trace` 当前不完整，必须补齐以下字段：

```text
episode
step
time
turn_time
turn_step
scenario
success
collision
uav_position
uav_velocity
goal_position
action
min_distance
distance_warning_cost
risk_sum
risk_max
risk_values
sigma_values
reaction_flag
reaction_time_current
attention_weights
attention_entropy
turning_obstacle_id
turning_obstacle_attention_weight
turning_obstacle_attention_rank
turning_obstacle_risk
R_sum
R_max
lateral_deviation
```

如果 `attention_weights` 暂时无法直接从 policy 输出，需要实现一种可追踪方案，例如：

```text
1. 在 ObstacleSetExtractor forward 中缓存 latest_attention_weights；
2. eval.py 在 step 后读取该缓存；
3. 或让 policy forward 返回附加 info；
4. 若暂时无法实现，必须在报告中明确写：
   attention_weights unavailable，并说明原因。
```

Trace 输出路径：

```text
results/preflight/traces/
```

文件命名：

```text
{method}_step{global_step}_{scenario}_ep{episode_id}.csv
```

最少补跑 trace：

```text
attention_full seed0:
  step 250k
  step 750k
  step 1000k

scenarios:
  eval_sudden_turn
  mixed_uncertainty

episodes:
  10
```

输出报告：

```text
results/preflight/TRACE_FIELD_PREFLIGHT_REPORT.md
```

必须回答：

```text
1. trace 是否能支持 risk_sum 与 distance_warning_cost 的时间先后比较？
2. trace 是否能支持 risk_biased_attention 改变 attention weights 的验证？
3. 缺失字段有哪些？
```

---

### Patch 4：生成统一 checkpoint / eval / trace 索引表

生成：

```text
CHECKPOINT_EVAL_INDEX.csv
```

建议路径：

```text
results/preflight/CHECKPOINT_EVAL_INDEX.csv
```

列必须包括：

```text
method
config
seed
global_step
checkpoint_path
can_resume
eval_random_switch_csv
eval_sudden_turn_csv
eval_random_switch_hard_csv
mixed_uncertainty_csv
trace_dir
notes
```

至少覆盖：

```text
1. attention_full seed0:
   250k / 500k / 750k / 1000k

2. Rgate8_lambda015_RbarFloor03 seed0:
   100k / 200k / 300k / 500k

3. ewma_formal 三配置三种子：
   Rgate8
   Rgate8_lambda015
   Rgate8_lambda015_RbarFloor03

4. 第一轮正式预实验：
   risk_full_rbar seed0/1/2
   attention_full seed0/1/2
```

如果某些路径不存在，不能静默跳过，必须写：

```text
missing
```

或：

```text
not_found
```

并在 notes 里说明。

---

### Patch 5：Gate-2b run 必须强制写入配置日志

在正式启动 Gate-2b 前，确保每个 run 的 log / report 都写入：

```text
method
profile_mode
profile_dim
agg
seed
train_seed
eval_seed
episode_seed_rule
total_steps
checkpoint_steps

use_safety_cost
cost_type
fallback_penalty
beta_cost

use_risk_bias
lambda_bias

R_gate
lambda_ewma
rbar_floor

d_safe
d_warning

resume_from
resume_global_step
checkpoint naming rule
```

建议每个 run 输出：

```text
run_config.json
```

路径：

```text
runs/{run_group}/{run_name}/run_config.json
```

Gate-2b 训练日志中必须出现：

```text
FALLBACK: cost-penalty, not PPO-Lagrangian
```

并且必须记录：

```text
base_reward
applied_cost
shaped_reward
fallback_penalty_active
```

输出报告：

```text
results/preflight/GATE2B_CONFIG_LOGGING_PREFLIGHT_REPORT.md
```

必须回答：

```text
1. Gate-2b 所有关键参数是否能被日志追踪？
2. penalty 是否能证明进入训练 reward？
3. risk_bias 是否能证明进入 attention logits？
```

---

## D. Preflight 完成标准

只有当以下文件都生成后，才能启动后续 2000k / Gate-2b 主实验：

```text
results/preflight/cost_stats_attention_gate.csv
results/preflight/COST_SCALE_PREFLIGHT_REPORT.md

results/preflight/reaction_definition_check.md

results/preflight/TRACE_FIELD_PREFLIGHT_REPORT.md
results/preflight/traces/*.csv

results/preflight/CHECKPOINT_EVAL_INDEX.csv

results/preflight/GATE2B_CONFIG_LOGGING_PREFLIGHT_REPORT.md
```

如果某一项无法完成，Codex 必须说明：

```text
1. 未完成项是什么；
2. 原因是什么；
3. 是否阻塞 longtrain；
4. 是否阻塞 Gate-2b；
5. 后续如何补救。
```

---

## E. Preflight 之后再执行原指南

Preflight 完成后，再从原指南的 Step 1 开始执行：

```text
Step 1:
attention_full seed0 -> 2000k
Rgate8_lambda015_RbarFloor03 seed0 -> 2000k

Step 2:
训练分布内 safety erosion 检查

Step 3:
attention vs risk 250k→2000k 曲线汇总

Step 4:
Gate-2b 三组 -> 1000k

Step 5:
Gate-2b trace 诊断

Step 6:
attention_full seed1 -> 1000k

Step 7:
方向判断
```

---



---

# F. 原执行指南：Long-Training Baseline 补全与 Gate-2b 验证（保留并继续执行）


> 用途：给 Codex 执行下一阶段实验。  
> 目标：先补齐 `attention_full` 与代表性 `EWMA-risk` 的长训轨迹，消除“baseline 后面可能恢复”的数据不公平；再执行 Gate-2b，判断 `distance_penalty`、`risk_penalty`、`risk_biased_attention + risk_penalty` 是否能抑制 long-training safety/reaction drift。  
> 当前优先级：**先补 baseline 长训轨迹，再跑 Gate-2b，最后补 seed 与环境扩展。**

---

## 0. 当前已知背景

### 0.1 attention_full seed=0 1000k gate 结果

已有 Gate-1 结果：

```text
method: attention_full
seed: 0
checkpoint: 250k / 500k / 750k / 1000k
```

`sudden_turn reaction_time`：

```text
250k: 0.208
500k: 1.860
750k: 11.424
1000k: 6.076
```

`hard` 场景：

```text
250k: success 0.96, collision 0.04
500k: success 0.98, collision 0.02
750k: success 0.94, collision 0.06
1000k: success 0.90, collision 0.10
```

已有报告自动判断：

```text
attention_reaction_degradation = true
attention_safety_degradation = true
```

但注意：

```text
attention 在 750k 最差，1000k 有一定恢复。
因此不能简单说 attention 单调退化；
更严谨的说法是 attention 出现 long-training safety/reaction oscillation。
```

### 0.2 修正 EWMA-risk 已知结果

修正 EWMA-risk formal 复验做到 500k：

```text
Rgate8_lambda015_RbarFloor03:
100k: reaction ≈ 0.3194
200k: reaction ≈ 0.2200
300k: reaction ≈ 3.0488
500k: reaction ≈ 6.4792
```

但 risk 线目前只有到 500k 的数据，缺少：

```text
750k / 1000k / 1500k / 2000k
```

因此不能断言：

```text
risk 继续训练后不会恢复。
```

当前必须补齐 risk 的长训轨迹，否则与 attention 的 1000k/2000k 对比不公平。

---

## 1. 最终执行顺序总览

请严格按以下顺序执行。

```text
Step 0:
先检查和补齐 train.py / eval.py 的必要参数、resume 支持、目录结构。

Step 1:
并行或串行补齐 baseline 长训轨迹：
1a) attention_full seed=0 继续到 2000k
1b) Rgate8_lambda015_RbarFloor03 seed=0 继续到 2000k

Step 2:
用已有或补跑的 G1 eval_random_switch 数据做训练分布内退化检查。

Step 3:
汇总 attention vs risk 的 250k → 2000k 完整曲线。

Step 4:
跑 Gate-2b 三组到 1000k：
- attention_full + distance_penalty
- attention_full + risk_penalty
- risk_biased_attention + risk_penalty

Step 5:
做 Gate-2b 曲线诊断：
- risk cost 是否早于 distance warning cost 上升；
- penalty 是否抑制 oscillation；
- random_switch 是否因 penalty 过强而掉 success。

Step 6:
补跑 attention_full seed=1 到 1000k，确认 oscillation 是否是单 seed 偶然。

Step 7:
基于所有结果做方向判断：
- 是否升级 obstacle modes；
- 是否实现真正 PPO-Lagrangian；
- 是否继续 risk-constrained attention；
- 是否转向 safe attention / drift diagnosis。
```

---

## 2. 时间与优先级原则

### 2.1 如果可以并行

Step 1a 和 Step 1b 并行启动：

```text
1a) attention_full seed=0 继续到 2000k
1b) Rgate8_lambda015_RbarFloor03 seed=0 继续到 2000k
```

预计总耗时约：

```text
3–4 小时
```

### 2.2 如果只能串行

先跑：

```text
1b) Rgate8_lambda015_RbarFloor03 seed=0 继续到 2000k
```

原因：

```text
risk 的数据缺口更大；
attention 已有 1000k；
risk 目前只有 500k。
```

然后再跑：

```text
1a) attention_full seed=0 继续到 2000k
```

### 2.3 当前不要做

当前不要做：

```text
1. 所有 risk 配置都跑 2000k；
2. 所有方法三种子；
3. 立即扩展复杂障碍物；
4. 立即实现真正 PPO-Lagrangian；
5. 直接写论文级结论。
```

---

## 3. Step 0：执行前代码能力检查与目录创建

### 3.1 检查现有文件

先执行：

```bash
pwd
ls
ls envs || true
ls policies || true
ls scripts || true
ls checkpoints || true
ls results || true
```

重点检查：

```text
train.py
eval.py
envs/dynamic_obstacle_env.py
policies/obstacle_set_extractor.py
```

### 3.2 创建目录

执行：

```bash
mkdir -p checkpoints/longtrain_baseline
mkdir -p runs/longtrain_baseline
mkdir -p results/longtrain_baseline/plots

mkdir -p checkpoints/gate2b
mkdir -p runs/gate2b
mkdir -p results/gate2b/eval
mkdir -p results/gate2b/traces
mkdir -p results/gate2b/plots
mkdir -p results/gate2b/summary

mkdir -p checkpoints/attention_seed1
mkdir -p runs/attention_seed1
mkdir -p results/attention_seed1
```

### 3.3 train.py 必须支持的基础参数

Codex 必须检查 `train.py` 是否支持以下参数。如果没有，需要先添加：

```text
--method
--profile_mode
--agg
--seed
--total_steps
--n_envs
--device
--scenario
--save_checkpoints
--checkpoint_steps
--checkpoint_dir
--log_dir
--save_path
--resume_from
```

### 3.4 eval.py 必须支持的基础参数

Codex 必须检查 `eval.py` 是否支持以下参数。如果没有，需要先添加：

```text
--model_path
--method
--profile_mode
--agg
--seed
--eval_seed
--episodes
--scenario
--device
--out_csv
--save_trace
--trace_dir
```

要求：

```text
eval_seed 默认值必须是 1000；
不要默认用 train seed 做 eval seed。
```

---

## 4. Resume 训练实现要求

### 4.1 优先尝试 resume

如果存在已有 checkpoint，优先从 checkpoint 继续训练，而不是从头训练。

SB3 PPO resume 的核心形式：

```python
from stable_baselines3 import PPO

model = PPO.load(checkpoint_path, env=env, device=device)
model.learn(
    total_timesteps=remaining_steps,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)
```

### 4.2 train.py 需要添加 --resume_from

如果 `train.py` 不支持 resume，Codex 需要添加：

```text
--resume_from <checkpoint_path>
```

逻辑：

```python
if args.resume_from:
    model = PPO.load(args.resume_from, env=env, device=args.device)
    reset_num_timesteps = False
else:
    model = PPO("MlpPolicy", env, ...)
    reset_num_timesteps = True

model.learn(
    total_timesteps=args.total_steps,
    reset_num_timesteps=reset_num_timesteps,
    callback=callback
)
```

注意：

```text
当 resume_from 不为空时，args.total_steps 表示“本次继续训练的步数”，不是最终全局步数。
```

或者也可以实现为：

```text
--target_total_steps
--current_total_steps
remaining_steps = target_total_steps - current_total_steps
```

只要报告写清楚即可。

### 4.3 CheckpointCallback 的编号映射

SB3 的 `CheckpointCallback` 通常按**本次 learn() 调用后的步数**保存，而不是自动按全局步数命名。

因此如果：

```text
从 1000k checkpoint resume，再继续训练 1000k
```

本次 callback 内的：

```text
250k / 500k / 750k / 1000k
```

对应全局：

```text
1250k / 1500k / 1750k / 2000k
```

Codex 必须在保存文件名中使用全局 step，避免混淆。

推荐实现：

```text
--resume_global_step 1000000
--checkpoint_steps 250000,500000,750000,1000000
```

保存时命名为：

```text
global_step = resume_global_step + local_step
```

例如：

```text
attention_full_s0_step1250000.zip
attention_full_s0_step1500000.zip
attention_full_s0_step1750000.zip
attention_full_s0_step2000000.zip
```

### 4.4 Resume 失败 fallback

如果 resume 出错，例如：

```text
checkpoint 不存在；
SB3 版本不兼容；
policy kwargs 不匹配；
env / observation space 不匹配；
参数不一致；
```

则从头训练到 2000k，并保存完整 checkpoint：

```text
250k / 500k / 750k / 1000k / 1250k / 1500k / 1750k / 2000k
```

报告中必须写：

```text
Resume failed, retrained from scratch to 2000k.
Reason: <error message>
```

---

## 5. Rgate8_lambda015_RbarFloor03 具体参数定义

### 5.1 与默认值不同的参数

`Rgate8_lambda015_RbarFloor03` 表示：

```text
R_gate = 8.0
lambda_ewma = 0.15
rbar_floor = 0.3
```

默认参考：

```text
R_gate 原始通常为 5.0
lambda_ewma 原始通常为 0.10
rbar_floor 原始可能不存在或为 0.0
```

其余 risk / environment / PPO 参数保持原始预实验默认值。

### 5.2 train.py 需要支持的参数

Codex 需要检查 `train.py` 是否已有这些参数。如果没有，需要添加：

```text
--R_gate
--lambda_ewma
--rbar_floor
```

如果项目里实际参数命名不同，例如：

```text
--risk_gate
--ewma_lambda
--Rbar_floor
```

则使用实际命名，但必须在报告中写清楚映射关系。

### 5.3 risk 配置命令行示例

示例：

```bash
python train.py \
  --method risk_full_rbar \
  --profile_mode full_12 \
  --agg risk \
  --seed 0 \
  --total_steps 1500000 \
  --resume_from checkpoints/path/to/risk_500k.zip \
  --resume_global_step 500000 \
  --R_gate 8.0 \
  --lambda_ewma 0.15 \
  --rbar_floor 0.3 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,1000000,1500000 \
  --checkpoint_dir checkpoints/longtrain_baseline \
  --log_dir runs/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0 \
  --save_path checkpoints/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_step2000000.zip
```

如果从 500k resume，则本次训练 1500k 后对应全局 2000k。  
本次 local checkpoints：

```text
250k / 500k / 1000k / 1500k
```

对应全局：

```text
750k / 1000k / 1500k / 2000k
```

---

## 6. Step 1a：attention_full seed=0 继续到 2000k

### 6.1 目的

已有 `attention_full seed=0` 到 1000k 的结果显示：

```text
250k 很好；
500k 变差；
750k 严重退化；
1000k 有一定恢复但仍明显差于 250k。
```

继续到 2000k 是为了判断：

```text
1. attention 是否继续恢复；
2. 是否再次恶化；
3. 是否呈现 checkpoint-level safety oscillation；
4. 1000k 的恢复是否只是暂时现象。
```

### 6.2 配置

```text
method: attention_full
profile_mode: full_12
seed: 0
start_checkpoint: 1000k attention_full checkpoint
target_global_steps: 2000k
remaining_steps: 1000k
new_global_checkpoints: 1250k / 1500k / 1750k / 2000k
n_envs: 16
device: cpu
scenario: train_random_switch
```

### 6.3 命令模板

根据实际 checkpoint 路径调整：

```bash
python train.py \
  --method attention_full \
  --profile_mode full_12 \
  --agg attention \
  --seed 0 \
  --total_steps 1000000 \
  --resume_from checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step1000000.zip \
  --resume_global_step 1000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000 \
  --checkpoint_dir checkpoints/longtrain_baseline \
  --log_dir runs/longtrain_baseline/attention_full_s0_resume1000k_to2000k \
  --save_path checkpoints/longtrain_baseline/attention_full_s0_step2000000.zip
```

如果不能 resume，则从头训练：

```bash
python train.py \
  --method attention_full \
  --profile_mode full_12 \
  --agg attention \
  --seed 0 \
  --total_steps 2000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000,1250000,1500000,1750000,2000000 \
  --checkpoint_dir checkpoints/longtrain_baseline \
  --log_dir runs/longtrain_baseline/attention_full_s0_from_scratch_2000k \
  --save_path checkpoints/longtrain_baseline/attention_full_s0_step2000000.zip
```

### 6.4 评估场景

每个 checkpoint 必评估：

```text
eval_random_switch
eval_sudden_turn
eval_random_switch_hard
mixed_uncertainty
```

每个场景：

```text
episodes = 50
eval_seed = 1000
```

### 6.5 输出

```text
results/longtrain_baseline/attention_full_s0_by_step.csv
```

必须包含：

```text
step
scenario
success_rate
collision_rate
mean_min_distance
reaction_time
near_miss_rate
safety_cost_mean
safety_cost_p90
safety_cost_p95
mean_time
```

---

## 7. Step 1b：Rgate8_lambda015_RbarFloor03 seed=0 继续到 2000k

### 7.1 目的

已有 risk 线只到 500k。  
需要验证：

```text
1. risk 是否在 750k / 1000k / 1500k / 2000k 恢复；
2. risk 是否和 attention 一样存在 checkpoint oscillation；
3. risk 是否持续差于 attention；
4. pure risk hard weighting 是否确实更不稳定。
```

### 7.2 配置

```text
method: risk_full_rbar / 或项目中对应的 risk hard weighting method
risk_config: Rgate8_lambda015_RbarFloor03
seed: 0
start_checkpoint: 500k risk checkpoint
target_global_steps: 2000k
remaining_steps: 1500k
new_global_checkpoints: 750k / 1000k / 1500k / 2000k
n_envs: 16
device: cpu
scenario: train_random_switch
```

### 7.3 命令模板

根据实际 checkpoint 路径调整：

```bash
python train.py \
  --method risk_full_rbar \
  --profile_mode full_12 \
  --agg risk \
  --seed 0 \
  --total_steps 1500000 \
  --resume_from checkpoints/path/to/Rgate8_lambda015_RbarFloor03_s0_step500000.zip \
  --resume_global_step 500000 \
  --R_gate 8.0 \
  --lambda_ewma 0.15 \
  --rbar_floor 0.3 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,1000000,1500000 \
  --checkpoint_dir checkpoints/longtrain_baseline \
  --log_dir runs/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_resume500k_to2000k \
  --save_path checkpoints/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_step2000000.zip
```

local step 到 global step 的映射：

```text
local 250k  -> global 750k
local 500k  -> global 1000k
local 1000k -> global 1500k
local 1500k -> global 2000k
```

如果不能 resume，则从头训练：

```bash
python train.py \
  --method risk_full_rbar \
  --profile_mode full_12 \
  --agg risk \
  --seed 0 \
  --total_steps 2000000 \
  --R_gate 8.0 \
  --lambda_ewma 0.15 \
  --rbar_floor 0.3 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000,1250000,1500000,1750000,2000000 \
  --checkpoint_dir checkpoints/longtrain_baseline \
  --log_dir runs/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_from_scratch_2000k \
  --save_path checkpoints/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_step2000000.zip
```

### 7.4 评估场景

每个 checkpoint 必评估：

```text
eval_random_switch
eval_sudden_turn
eval_random_switch_hard
mixed_uncertainty
```

每个场景：

```text
episodes = 50
eval_seed = 1000
```

### 7.5 输出

```text
results/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_by_step.csv
```

必须包含：

```text
step
scenario
success_rate
collision_rate
mean_min_distance
reaction_time
near_miss_rate
safety_cost_mean
safety_cost_p90
safety_cost_p95
mean_time
```

---

## 8. Step 2：用已有或补跑的 G1 数据检查训练分布内安全是否退化

### 8.1 目的

检查：

```text
训练分布内是否也出现 safety margin erosion？
```

### 8.2 数据条件

如果已有 `attention_full G1 eval_random_switch` 数据：

```text
直接分析。
```

如果没有：

```text
用 G1 的 250k / 500k / 750k / 1000k checkpoint 补跑 eval_random_switch。
```

补跑命令模板：

```bash
python eval.py \
  --model_path checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step250000.zip \
  --method attention_full \
  --profile_mode full_12 \
  --agg attention \
  --scenario eval_random_switch \
  --episodes 50 \
  --eval_seed 1000 \
  --device cpu \
  --out_csv results/longtrain_baseline/eval_random_switch_attention_s0_step250000.csv
```

对 500k / 750k / 1000k 重复。

### 8.3 检查内容

检查：

```text
success_rate
collision_rate
mean_time
mean_min_distance
near_miss_rate
distance_warning_cost
risk_sum
risk_max
```

### 8.4 判断

如果：

```text
success_rate 保持高；
mean_time 下降或保持；
mean_min_distance 下降；
near_miss_rate 上升；
safety_cost 上升；
```

则说明：

```text
策略在训练分布内可能也从“更安全”漂移到“更高效 / 更贴近通过”。
```

### 8.5 输出

```text
results/longtrain_baseline/random_switch_safety_trend.csv
TRAIN_DISTRIBUTION_SAFETY_TREND.md
```

---

## 9. Step 3：汇总 attention vs risk 的 250k→2000k 完整曲线

### 9.1 目的

回答：

```text
attention 和 risk 是单调退化，还是 checkpoint oscillation？
risk 是否在 1000k+ 恢复？
attention 是否在 1500k/2000k 继续恢复或再次恶化？
```

### 9.2 汇总曲线

生成以下曲线：

```text
1. sudden_turn reaction_time vs step
2. sudden_turn min_distance vs step
3. hard success/collision vs step
4. mixed success/collision vs step
5. random_switch min_distance / mean_time / near_miss vs step
6. safety_cost_mean / safety_cost_p95 vs step
```

每张图至少包含：

```text
attention_full seed=0
Rgate8_lambda015_RbarFloor03 seed=0
```

### 9.3 输出

```text
results/longtrain_baseline/attention_vs_risk_longtrain_summary.csv
results/longtrain_baseline/plots/*.png
ATTENTION_RISK_2000K_BASELINE_REPORT.md
```

---

## 10. Gate-2b 开始前的代码能力检查

Gate-2b 需要 `train.py` 和 `obstacle_set_extractor.py` 支持一些参数与逻辑。Codex 必须先检查并补齐。

### 10.1 train.py 必须支持的 Gate-2b 参数

```text
--use_safety_cost
--cost_type
--beta_cost
--use_risk_bias
--lambda_bias
--fallback_penalty
```

其中：

```text
cost_type 可选：
- none
- distance_warning
- risk_sum
```

### 10.2 cost-penalty fallback 实现

本阶段不强制实现真正 PPO-Lagrangian。

如果：

```text
--use_safety_cost true
--fallback_penalty true
```

则在 env step 后修改 reward：

```python
reward_new = reward_old - beta_cost * cost_t
```

其中：

```text
distance_warning:
  cost_t = max(0, d_warning - min_distance)^2

risk_sum:
  cost_t = R_sum
```

必须 log：

```text
FALLBACK: cost-penalty, not PPO-Lagrangian.
```

### 10.3 risk_biased_attention 实现

如果 `policies/obstacle_set_extractor.py` 不支持 risk-biased attention，Codex 需要添加。

形式：

```text
score_i = learned_score_i + lambda_bias * log(risk_i + eps)
w_i = softmax(score_i)
context = sum_i w_i h_i
```

要求：

```text
1. lambda_bias 可由 --lambda_bias 设置；
2. risk_i 从 obstacle profile 中读取；
3. eps 建议 1e-6；
4. 若 profile_mode 不含 risk_i，则 risk_bias 自动关闭或报清晰错误。
```

---

## 11. d_warning 与 d_safe 的区别

环境中可能已有：

```text
d_safe = 0.80
```

通常对应物理安全距离：

```text
r_uav + r_obs + margin
```

Gate-2b 的 distance penalty 不直接使用 `d_safe`，而使用：

```text
d_warning = 1.0
```

定义：

```text
distance_warning_cost = max(0, d_warning - min_distance)^2
```

说明：

```text
d_warning = 1.0 > d_safe = 0.80
warning zone 比 collision / physical safety zone 更大；
目的是在碰撞发生前就产生 dense cost signal。
```

因此请不要把：

```text
d_warning
```

和环境内部的：

```text
d_safe
```

混淆。

---

## 12. Gate-2b 三组到 1000k

### 12.1 目的

在 baseline 长训轨迹明确后，验证：

```text
1. distance_penalty 是否抑制 attention 的 safety/reaction drift；
2. risk_penalty 是否优于 distance_penalty；
3. risk_biased_attention + risk_penalty 是否优于 risk_penalty only。
```

### 12.2 方法

跑三组：

```text
G2b-1: attention_full + distance_penalty
G2b-2: attention_full + risk_penalty
G2b-3: risk_biased_attention + risk_penalty
```

### 12.3 训练配置

```text
seed: 0
total_steps: 1000k
checkpoint_steps: 250k / 500k / 750k / 1000k
n_envs: 16
device: cpu
scenario: train_random_switch
```

### 12.4 beta_cost 设置

初始：

```text
beta_cost = 5.0
```

但训练前需要基于已有 G1 eval trace / summary 估计 cost 量级：

```text
distance_warning_cost p50 / p90 / p95 / max
risk_sum p50 / p90 / p95 / max
```

目标：

```text
beta_cost * cost_p90 ≈ collision_penalty_abs 的 10%–50%
```

如果 collision penalty 是：

```text
-10
```

则目标：

```text
beta_cost * cost_p90 ≈ 1–5
```

如果偏离很大，需要调整 beta_cost，并在报告里说明。

### 12.5 命令模板

#### G2b-1: attention_full + distance_penalty

```bash
python train.py \
  --method attention_full_distance_penalty \
  --profile_mode full_12 \
  --agg attention \
  --seed 0 \
  --total_steps 1000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --use_safety_cost true \
  --cost_type distance_warning \
  --d_warning 1.0 \
  --fallback_penalty true \
  --beta_cost 5.0 \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000 \
  --checkpoint_dir checkpoints/gate2b \
  --log_dir runs/gate2b/attention_full_distance_penalty_s0 \
  --save_path checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip
```

#### G2b-2: attention_full + risk_penalty

```bash
python train.py \
  --method attention_full_risk_penalty \
  --profile_mode full_12 \
  --agg attention \
  --seed 0 \
  --total_steps 1000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --use_safety_cost true \
  --cost_type risk_sum \
  --fallback_penalty true \
  --beta_cost 5.0 \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000 \
  --checkpoint_dir checkpoints/gate2b \
  --log_dir runs/gate2b/attention_full_risk_penalty_s0 \
  --save_path checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip
```

#### G2b-3: risk_biased_attention + risk_penalty

```bash
python train.py \
  --method risk_biased_attention_risk_penalty \
  --profile_mode full_12 \
  --agg attention \
  --use_risk_bias true \
  --lambda_bias 0.2 \
  --seed 0 \
  --total_steps 1000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --use_safety_cost true \
  --cost_type risk_sum \
  --fallback_penalty true \
  --beta_cost 5.0 \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000 \
  --checkpoint_dir checkpoints/gate2b \
  --log_dir runs/gate2b/risk_biased_attention_risk_penalty_s0 \
  --save_path checkpoints/gate2b/risk_biased_attention_risk_penalty_s0_step1000000.zip
```

如果实际 `train.py` 参数名不同，Codex 应使用实际参数，但必须在报告中列出映射。

---

## 13. Gate-2b 评估

每个 checkpoint 必评估：

```text
eval_random_switch
eval_sudden_turn
eval_random_switch_hard
mixed_uncertainty
```

每个场景：

```text
episodes = 50
eval_seed = 1000
```

### 13.1 为什么 random_switch 必评估

`eval_random_switch` 用于判断：

```text
penalty 是否让策略过于保守；
是否牺牲训练分布内正常任务性能；
success / mean_time 是否变差。
```

如果某个 penalty 在 sudden_turn 更好，但 random_switch success 明显下降，则需要谨慎解释。

---

## 14. Step 5：Gate-2b 曲线诊断

### 14.1 如果 eval.py 不支持 --save_trace

Codex 需要在 `eval.py` 中实现 step-level trace 功能。

要求：

```text
--save_trace true
--trace_dir results/gate2b/traces
```

输出 CSV 命名：

```text
results/gate2b/traces/{method}_step{ckpt}_{scenario}_ep{ep_id}.csv
```

优先级：

```text
trace 功能在 Gate-2b 训练完成后实现即可；
不阻塞训练。
```

### 14.2 诊断场景

对以下 checkpoint 做 step-level trace：

```text
250k / 750k / 1000k
```

对以下场景：

```text
eval_sudden_turn
mixed_uncertainty
```

对以下方法：

```text
attention_full baseline
attention_full + distance_penalty
attention_full + risk_penalty
risk_biased_attention + risk_penalty
```

每组：

```text
diagnostic_episodes = 10
```

### 14.3 trace 每步字段

每步至少记录：

```text
episode
step
time
scenario
success
collision
uav_position
uav_velocity
goal_position
action
min_distance
risk_values
sigma_values
attention_weights
attention_entropy
turning_obstacle_id
turning_obstacle_attention_weight
turning_obstacle_attention_rank
turning_obstacle_risk
R_sum
R_max
distance_warning_cost
risk_sum
risk_max
lateral_deviation
reaction_flag
```

### 14.4 诊断曲线

每个 episode 画：

```text
1. risk_sum / risk_max vs time
2. distance_warning_cost vs time
3. min_distance vs time
4. action norm vs time
5. lateral deviation vs time
6. attention weight of turning obstacle vs time
7. turn_time vertical line
8. reaction_time vertical line
```

### 14.5 关键判断

#### risk cost 有预测式优势

如果：

```text
risk_sum / risk_max 在 turn 后明显早于 distance_warning_cost 上升；
且 risk_penalty 策略的 reaction_time 更短；
且 near_miss / collision 不增加；
```

则结论：

```text
risk cost 具有 prediction-style warning value。
```

#### risk cost 没有预测优势

如果：

```text
risk_sum 与 distance_warning_cost 同时上升；
或 risk_sum 更晚上升；
或 risk_penalty 没改善 reaction / near_miss；
```

则结论：

```text
risk cost 的独立价值不足；
不宜强推 risk-constrained attention。
```

#### risk bias 有价值

如果：

```text
risk_biased_attention + risk_penalty
优于
attention_full + risk_penalty
```

并且 attention 诊断显示：

```text
turning obstacle attention weight 更早上升；
turning obstacle top1/top2 rate 更高；
```

则结论：

```text
dual-role risk 有继续研究价值。
```

---

## 15. Step 6：attention_full seed=1 1000k

### 15.1 目的

确认 attention oscillation 不是 seed=0 偶然。

### 15.2 配置

```text
method: attention_full
profile_mode: full_12
seed: 1
total_steps: 1000k
checkpoint_steps: 250k / 500k / 750k / 1000k
n_envs: 16
device: cpu
scenario: train_random_switch
```

### 15.3 命令模板

```bash
python train.py \
  --method attention_full \
  --profile_mode full_12 \
  --agg attention \
  --seed 1 \
  --total_steps 1000000 \
  --n_envs 16 \
  --device cpu \
  --scenario train_random_switch \
  --save_checkpoints true \
  --checkpoint_steps 250000,500000,750000,1000000 \
  --checkpoint_dir checkpoints/attention_seed1 \
  --log_dir runs/attention_seed1/attention_full_s1_1000k \
  --save_path checkpoints/attention_seed1/attention_full_s1_step1000000.zip
```

### 15.4 评估

每个 checkpoint：

```text
eval_random_switch
eval_sudden_turn
eval_random_switch_hard
mixed_uncertainty
```

### 15.5 判断

#### 两个 seed 都退化

```text
结论：
attention long-training safety/reaction oscillation 不是单 seed 偶然。
```

#### seed=0 退化、seed=1 不退化

```text
结论：
attention stability 有 seed sensitivity。
需要三种子复验，不能强说普遍退化。
```

#### seed=1 也震荡但时间点不同

```text
结论：
checkpoint-level safety oscillation 存在，但峰值位置依赖 seed。
```

---

## 16. Step 7：最终方向判断

### 16.1 risk 有明确价值

如果：

```text
risk_penalty 明显优于 distance_penalty；
或 risk_biased_attention + risk_penalty 明显优于 risk_penalty only；
并且 risk cost 曲线早于 distance cost 上升；
```

结论：

```text
继续 Motion-Uncertainty Risk-Constrained Attention Policy。
```

下一步：

```text
1. 扩展 obstacle modes；
2. 三种子复验；
3. 实现真正 PPO-Lagrangian；
4. 准备论文级实验矩阵。
```

### 16.2 distance_penalty 足够，risk 无优势

如果：

```text
distance_penalty 和 risk_penalty 差不多；
risk_biased_attention 没有额外收益；
```

结论：

```text
risk 不适合作为主创新。
```

下一步：

```text
转向 safe attention / long-training drift diagnosis。
```

### 16.3 两种 penalty 都没用

如果：

```text
distance_penalty 和 risk_penalty 都不能抑制 oscillation；
```

结论：

```text
固定 penalty 版不足。
```

下一步：

```text
1. 实现真正 PPO-Lagrangian；
2. 调整 reward/cost；
3. 考虑 early stopping / checkpoint selection / stability regularization。
```

### 16.4 risk 和 attention 都长训震荡

如果：

```text
pure attention 和 pure risk 都在长训中表现出 oscillation；
```

结论：

```text
问题更可能是 PPO safety behavior oscillation，而非单一 aggregation 机制失效。
```

下一步：

```text
将问题定义为：
long-training safety behavior oscillation in PPO-based dynamic obstacle avoidance。
```

---

## 17. 推荐脚本

Codex 需要新增或修改以下脚本：

```text
scripts/run_longtrain_baseline_2000k.sh
scripts/eval_longtrain_baseline_2000k.sh
scripts/aggregate_longtrain_baseline.py

scripts/run_gate2b_penalty_1000k.sh
scripts/eval_gate2b_penalty_1000k.sh
scripts/diagnose_gate2b_curves.py
scripts/aggregate_gate2b_results.py

scripts/run_attention_seed1_1000k.sh
```

---

## 18. 最终交付物

Codex 最终需要输出：

```text
ATTENTION_RISK_2000K_BASELINE_REPORT.md
GATE2B_PENALTY_1000K_REPORT.md
ATTENTION_SEED1_1000K_REPORT.md
FINAL_DIRECTION_DECISION_REPORT.md
```

以及 CSV：

```text
results/longtrain_baseline/attention_vs_risk_longtrain_summary.csv
results/gate2b/gate2b_by_step.csv
results/gate2b/gate2b_curve_diagnostics_summary.csv
results/attention_seed1/attention_seed1_by_step.csv
```

以及图：

```text
results/longtrain_baseline/plots/
results/gate2b/plots/
```

---

## 19. 最终报告必须回答的问题

`FINAL_DIRECTION_DECISION_REPORT.md` 必须回答：

```text
1. attention_full seed=0 到 2000k 是恢复、继续退化，还是震荡？
2. Rgate8_lambda015_RbarFloor03 到 2000k 是否恢复？
3. attention 和 risk 谁的 long-training stability 更好？
4. eval_random_switch 上是否存在训练分布内 safety erosion？
5. distance_penalty 是否抑制 attention drift？
6. risk_penalty 是否优于 distance_penalty？
7. risk cost 是否早于 distance warning cost 上升？
8. risk_biased_attention 是否比 risk_penalty only 更好？
9. attention seed=1 是否复现 oscillation？
10. 下一步应该继续 risk-constrained attention，还是转向 safe attention / drift diagnosis？
```

---

## 20. Codex 执行提醒

执行时请注意：

```text
1. 先检查是否能从已有 checkpoint resume；
2. 若不能 resume，必须在报告中说明，并从头训练；
3. 所有 eval 使用 eval_seed=1000，不要复用 train seed；
4. 所有关键曲线必须保存为 CSV 和 PNG；
5. 不要因为某一 checkpoint 好或坏就直接下结论，要看完整曲线；
6. 所有结论必须区分：
   - 单 seed 现象；
   - checkpoint oscillation；
   - trend；
   - cost penalty 效果；
   - risk 的独立价值。
```