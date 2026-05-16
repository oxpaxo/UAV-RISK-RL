# Phase B 指南：EnvV2 Eval-only 几何 / Safety-filter Baseline Audit

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 阶段名称：`Phase B - Eval-only Geometry & Safety-filter Baseline Audit`  
> 阶段性质：不训练；不改 EnvV2-core；只评估几何控制器与 action-level safety filter。  
> 前置条件：Phase A 已完成，且存在 `PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag`。  
> 目标产物：几何 / filter baseline 统一评估结果、Pareto 图表、failure breakdown、阻塞式 watcher、Phase B 报告与完成/停止 flag。  

---

## 0. 背景简述

Phase A 已完成并建立了统一 evaluation runner、episode-level metrics schema、per-step trace schema、full active obstacle long-table logging，以及 safety-filter action metadata。

Phase A 报告已经确认：

```text
EnvV2-core frozen;
unified eval framework ready;
random / straight_line / cpa_reactive / attention_full / filtered_attention_full adapters ready;
episode-level metrics CSV and per-step traces available;
full active obstacle set logging available;
safety filter records action_raw / action_filtered / filter metadata.
```

当前 EnvV2 的核心问题不是旧 3-ball 中的 no-response，而是：

```text
attention_full PPO 会前进，但 collision 偏高；
CPA-reactive 几何策略 collision 低，但 success 低；
需要系统比较几何方法、action-level safety filter 与 attention_full PPO 的 safety-efficiency Pareto。
```

Phase B 的目标是：

```text
不训练任何新策略，先用 eval-only 方法判断：
1. 简单几何 baseline 到底有多强；
2. current CPA-reactive 的低 collision 是否来自 CPA/TTC 预测触发；
3. action-level safety filter 是否能直接修复 attention_full 的高 collision；
4. 后续是否有必要训练 safety cost PPO。
```

---

## 1. Phase B 总目标

Phase B 要在 frozen EnvV2-core 上完成一组 eval-only baseline audit。

必须比较的 baseline 分为四类：

```text
A. Reference baselines:
   random
   straight_line
   attention_full_1500k
   current_cpa_reactive

B. APF family:
   naive_apf
   velocity_aware_apf
   cpa_ttc_weighted_apf

C. CPA-reactive sweep:
   current_cpa_reactive
   variants around d_reactive / avoid_weight / cpa_trigger / horizon

D. Safety-filtered attention_full:
   attention_full + distance_filter
   attention_full + cpa_ttc_filter
   attention_full + vo_like_filter
```

Phase B 必须输出：

```text
1. success-collision Pareto;
2. progress-collision Pareto;
3. near_miss / min_distance comparison;
4. scenario-wise breakdown;
5. motion-mode breakdown;
6. threat-class breakdown;
7. filter intervention analysis;
8. top baseline/config ranking;
9. 是否可以进入 Phase C safety-cost training 的判断。
```

---

## 2. 明确禁止事项

Phase B 禁止：

```text
1. 禁止修改 EnvV2-core：
   - obstacle 数量范围；
   - motion modes；
   - train/eval scenario；
   - action dynamics；
   - reward；
   - termination；
   - collision/success/near_miss 定义。

2. 禁止训练任何新策略：
   - 不训练 safety cost PPO；
   - 不训练 nearest-K MLP PPO；
   - 不训练 GRU/LSTM；
   - 不训练 residual RL；
   - 不做 PPO-Lagrangian。

3. 禁止把 Phase A minimal filter 当作正式 baseline 直接汇报。
   Phase A minimal filter 只用于基础设施验证，Phase B 必须实现正式的 distance / CPA-TTC / VO-like filter。

4. 禁止只跑 attention_full 与一个 filter 后就停止。

5. 禁止只输出 aggregate metrics 而没有 scenario / motion / threat / intervention breakdown。

6. 禁止 watcher 在未完成、未触发 stop flag 时中途退出。
```

允许：

```text
1. 新增 eval-only controller；
2. 新增 APF / filter adapter；
3. 新增 parameter sweep；
4. 新增结果分析脚本；
5. 新增图表；
6. 新增 watcher 和 report；
7. 复用或轻度扩展 Phase A eval framework。
```

---

## 3. Phase B 推荐文件与目录结构

建议新增：

```text
scripts/
  run_env_v2_phase_b_geometry_filter_baselines.py
  analyze_env_v2_phase_b_results.py
  watch_phase_b_geometry_filter_baselines.sh

results/
  env_v2_phase_b_geometry_filter_baselines/
    phase_b_status.txt
    phase_b_watcher.log

    PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag
    PHASE_B_STOP_PHASE_A_MISSING.flag
    PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag
    PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag
    PHASE_B_STOP_BASELINE_IMPL_FAILED.flag
    PHASE_B_STOP_EVAL_FAILED.flag
    PHASE_B_STOP_SCHEMA_MISMATCH.flag
    PHASE_B_STOP_RESOURCE_LIMIT.flag
    PHASE_B_STOP_WATCHER_FAILED.flag

    PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md

    tables/
      phase_b_baseline_manifest.csv
      phase_b_episode_metrics.csv
      phase_b_eval_summary.csv
      phase_b_pareto_table.csv
      phase_b_top_configs.csv
      phase_b_scenario_breakdown.csv
      phase_b_motion_mode_breakdown.csv
      phase_b_threat_class_breakdown.csv
      phase_b_filter_intervention_summary.csv
      phase_b_failure_case_table.csv
      phase_b_command_manifest.csv
      phase_b_schema_check.csv

    traces/
      sample_traces/
        sample_<baseline>_<scenario>_trace.csv
      formal_traces/
        <optional gzip traces for selected top configs>

    plots/
      success_collision_pareto.png
      progress_collision_pareto.png
      near_miss_min_distance_comparison.png
      scenario_collision_heatmap.png
      motion_mode_collision_heatmap.png
      threat_class_collision_bar.png
      filter_rate_vs_collision.png
      filter_delta_distribution.png
      top_config_ranking.png

    logs/
      phase_b_geometry_filter_eval.log
      phase_b_analysis.log
```

如果项目已有更合适的目录规范，可以沿用，但必须保证：

```text
1. Phase B 产物集中；
2. flag 路径固定；
3. watcher 能检测 complete / stop flag；
4. report 能引用所有核心 CSV / plots / logs。
```

---

## 4. 前置检查

Phase B 启动前必须检查：

```text
1. Phase A complete flag 是否存在；
2. Phase A unified trace schema 是否可读取；
3. attention_full 1500k checkpoint 是否存在；
4. EnvV2-core 是否未被修改；
5. 当前 repo 能 import EnvV2 / policies / SB3 checkpoint；
6. 输出目录可写。
```

若 Phase A 产物缺失：

```text
PHASE_B_STOP_PHASE_A_MISSING.flag
```

若 checkpoint 缺失：

```text
PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag
```

若必须修改 EnvV2-core 才能继续：

```text
PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag
```

---

## 5. Baseline 定义

### 5.1 Reference baselines

Phase B 必须包含：

```text
random
straight_line
attention_full_1500k
current_cpa_reactive
```

`random` 与 `straight_line` 可以作为 sanity/reference，不一定进入 Pareto 主结论，但应保留在表中。

`attention_full_1500k` 必须使用 Phase 2 best / 1500k checkpoint：

```text
checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip
```

如果项目实际路径不同，Codex 应自动搜索合理路径；若无法定位，触发 checkpoint stop flag。

`current_cpa_reactive` 必须复用现有 sanity 中的 CPA-reactive 逻辑，不要重写成不同策略。

---

### 5.2 APF family

APF family 的目标不是替代 CPA-reactive，而是拆解几何避障能力来源。

#### 5.2.1 naive_apf

只使用当前距离，不使用 relative velocity / CPA / TTC。

建议形式：

```text
goal_vec = normalize(goal_xy - uav_xy)

for each active obstacle:
  rel = obs_pos_xy - uav_pos_xy
  d = ||rel||
  away = -rel / d
  if d < d0:
      rep_gain = w_rep * (1/d - 1/d0) / (d*d + eps)
      avoid += rep_gain * away

cmd_xy = normalize(w_goal * goal_vec + avoid)
```

推荐初始参数：

```text
d0 = 4.0
w_goal = 1.0
w_rep = 1.0
```

可小规模 sweep：

```text
d0 ∈ {3.0, 4.0, 5.0}
w_rep ∈ {0.6, 1.0, 1.6}
```

#### 5.2.2 velocity_aware_apf

在 naive APF 基础上加入 closing / relative velocity gain。

建议：

```text
closing = -dot(rel, rel_vel) / (||rel|| + eps)
closing_gain = clip(closing / 2.0, 0.0, 1.5)

rep_gain = base_rep_gain * (1.0 + alpha_closing * closing_gain)
```

推荐：

```text
alpha_closing ∈ {0.5, 1.0, 2.0}
```

#### 5.2.3 cpa_ttc_weighted_apf

在 velocity-aware APF 基础上加入短时 CPA / TTC 权重。

建议：

```text
tcpa = clip(-dot(rel, rel_vel) / (||rel_vel||^2 + eps), 0, horizon)
cpa_dist = ||rel + rel_vel * tcpa||

cpa_gain = clip((cpa_threshold - cpa_dist) / cpa_threshold, 0, 1)
tcpa_gain = clip((horizon - tcpa) / horizon, 0, 1)

risk_gain = 1 + alpha_cpa * cpa_gain * tcpa_gain
rep_gain = base_rep_gain * risk_gain
```

推荐：

```text
horizon = 4.5
cpa_threshold = 2.4
alpha_cpa ∈ {1.0, 2.0, 3.0}
```

---

### 5.3 CPA-reactive sweep

Current CPA-reactive 参数：

```text
d_reactive = 4.0
horizon = 4.5
cpa_trigger_distance = 2.4
final command = normalize(v_goal + 2.1 * avoid)
```

Phase B 必须做局部 sweep，而不是大规模盲目 grid。

推荐 one-factor-around-current sweep：

```text
current:
  d_reactive=4.0, horizon=4.5, cpa_trigger=2.4, avoid_weight=2.1

variants:
  d_reactive ∈ {3.0, 5.0}, others current
  cpa_trigger ∈ {1.8, 3.0}, others current
  horizon ∈ {3.0, 6.0}, others current
  avoid_weight ∈ {1.4, 2.8}, others current
```

共 9 个配置：

```text
cpa_reactive_current
cpa_reactive_d3
cpa_reactive_d5
cpa_reactive_cpa18
cpa_reactive_cpa30
cpa_reactive_h3
cpa_reactive_h6
cpa_reactive_w14
cpa_reactive_w28
```

如计算资源允许，可额外做 small grid，但不能影响主报告完成。

---

### 5.4 Safety-filtered attention_full

Safety filter 只修改 action，不训练 policy。

所有 filtered baselines 必须记录：

```text
action_raw
action_filtered
action_executed
filter_used
filter_triggered
filter_reason
filter_delta_norm
min_predicted_cpa_raw
min_predicted_cpa_filtered
min_ttc_raw
min_ttc_filtered
unsafe_obstacle_id
unsafe_obstacle_distance
unsafe_obstacle_tcpa
unsafe_obstacle_cpa
```

#### 5.4.1 distance_filter

目的：测试最简单的当前距离安全修正是否有效。

建议触发条件：

```text
nearest_distance < d_filter
and raw action would increase closing speed toward nearest obstacle
```

建议参数：

```text
d_filter ∈ {1.5, 2.0, 2.5}
```

修正方式可以是：

```text
v_filtered = normalize((1 - beta) * v_raw + beta * v_away_or_lateral)
```

推荐：

```text
beta ∈ {0.5, 0.8}
```

至少保留一个主配置：

```text
distance_filter_d2_beta08
```

#### 5.4.2 cpa_ttc_filter

目的：测试短时预测安全修正是否能直接修复 PPO。

对 raw action 计算未来短时 CPA/TTC：

```text
v_cmd = action_raw_xy * v_uav_max
rel = obs_pos_xy - uav_pos_xy
rel_vel_pred = obs_vel_xy - v_cmd
tcpa = clip(-dot(rel, rel_vel_pred) / (||rel_vel_pred||^2 + eps), 0, horizon)
cpa = ||rel + rel_vel_pred * tcpa||
```

触发条件：

```text
0 < tcpa < horizon and cpa < cpa_safe
```

推荐参数：

```text
horizon ∈ {3.0, 4.5}
cpa_safe ∈ {1.2, 1.5, 2.0}
```

修正方式：

```text
1. 生成 away / lateral candidate；
2. 与 raw action blend；
3. 或从候选速度中选择 closest-to-raw but safe。
```

至少保留一个主配置：

```text
cpa_ttc_filter_h45_cpa15
```

#### 5.4.3 vo_like_filter

目的：测试 velocity-space safety projection 是否是强 baseline。

建议实现：

```text
1. 输入 attention_full raw action；
2. 生成 candidate velocity set：
   - raw velocity；
   - goal velocity；
   - current velocity direction；
   - 16 or 24 headings × speed levels {0.4, 0.7, 1.0} * v_uav_max；
   - away / lateral candidates from nearest unsafe obstacle；

3. 对每个 candidate 计算 horizon 内 predicted CPA；
4. discard unsafe candidates:
   - predicted cpa < cpa_safe
   - or predicted collision distance < d_collision_margin

5. score safe candidates:
   score = 
     - lambda_raw * ||v_candidate - v_raw|| 
     + lambda_goal * progress_alignment
     + lambda_clearance * min_predicted_cpa
     - lambda_smooth * ||v_candidate - v_current||

6. 选择 score 最高的 candidate。
```

推荐参数：

```text
horizon = 4.5
cpa_safe ∈ {1.2, 1.5, 2.0}
num_headings = 16
speed_levels = {0.4, 0.7, 1.0}
```

至少保留一个主配置：

```text
vo_like_filter_h45_cpa15_h16
```

---

## 6. Eval 规模设计

Phase B 分为三个子阶段，由同一个 watcher 连续执行。

### B0: smoke test

目的：检查所有 baseline 能运行。

```text
scenarios:
  eval_flow_id
  eval_flow_high_speed

episodes:
  2 or 3 per scenario-baseline

baselines:
  all implemented Phase B baselines/configs
```

B0 不用于结论，只用于运行检查。

### B1: coarse audit

目的：筛选 sweep 配置。

```text
scenarios:
  all 6 eval scenarios:
    eval_flow_id
    eval_flow_high_density
    eval_flow_high_speed
    eval_flow_high_threat
    eval_flow_mixed_ood
    eval_flow_sudden_threat

episodes:
  20 per scenario-baseline/config

baselines:
  all required baselines/configs
```

B1 用于生成初步 Pareto 和 top configs。

### B2: formal confirmation

目的：对 top configs 做正式 50 episode 确认。

```text
scenarios:
  all 6 eval scenarios

episodes:
  50 per scenario-baseline/config

baselines:
  attention_full_1500k
  current_cpa_reactive
  top 3 CPA-reactive configs from B1
  top 2 APF-family configs from B1
  top 2 safety-filtered attention configs from B1
  random
  straight_line
```

如果资源不足以完成 B2，不能假装完成。应触发：

```text
PHASE_B_STOP_RESOURCE_LIMIT.flag
```

并生成 partial report，说明 B0/B1 已完成但 B2 未完成。

---

## 7. 统一 metrics 与分析要求

Phase B 必须沿用 Phase A episode-level schema，并可以新增字段，但不能删除 Phase A 核心列。

至少输出：

```text
success
collision
timeout
near_miss
progress
final_goal_distance
episode_return
episode_length_steps
mean_time
episode_min_distance
mean_min_distance
min_distance_after_threat
replacement_count
active_obstacle_count
mean_action_norm
mean_action_delta
max_action_delta
filter_used
filter_trigger_count
filter_trigger_rate
mean_filter_delta_norm
max_filter_delta_norm
```

### 7.1 Pareto 判定

必须生成 Pareto 表：

```text
baseline_name
config_name
success_rate
collision_rate
near_miss_rate
progress_mean
episode_min_distance_mean
mean_action_delta
filter_trigger_rate
is_pareto_success_collision
is_pareto_progress_collision
rank_score
```

建议 rank_score 只作为辅助，不作为唯一结论：

```text
rank_score = success_rate - 2.0 * collision_rate - 0.5 * near_miss_rate + 0.2 * progress_mean
```

Codex 可以调整，但必须在报告中写清楚。

### 7.2 breakdown

必须包含：

```text
scenario-wise breakdown
motion-mode breakdown
threat-class breakdown
filter intervention breakdown
failure-case table
```

### 7.3 filter intervention analysis

对 filtered attention_full，必须分析：

```text
filter_trigger_rate
mean_filter_delta_norm
max_filter_delta_norm
collision_when_filter_triggered
collision_when_filter_not_triggered
success_when_filter_triggered
success_when_filter_not_triggered
min_predicted_cpa_raw vs filtered
min_ttc_raw vs filtered
```

如果某些字段暂时 NaN，必须说明原因。

---

## 8. Trace 要求

Phase B 必须复用 Phase A trace schema。

### 8.1 B0 / B1

B0 和 B1 至少要保存 sampled traces：

```text
每个 baseline/config 至少保存 1 个 episode trace；
每个 scenario 至少保存若干 failure trace；
filtered baseline 必须保存 filter-triggered episode trace。
```

### 8.2 B2

B2 formal confirmation 建议保存：

```text
1. full traces for top configs；
2. 或 gzip-compressed traces；
3. 至少保存所有 collision episodes 与 representative success episodes。
```

不得完全不保存 trace。

---

## 9. 命令清单

Codex 应根据项目实际路径调整命令，并在报告中记录实际命令。

### 9.1 进入项目

```bash
cd /root/workspace/uav-risk-rl
```

### 9.2 编译检查

```bash
python -m py_compile scripts/run_env_v2_phase_b_geometry_filter_baselines.py
python -m py_compile scripts/analyze_env_v2_phase_b_results.py
bash -n scripts/watch_phase_b_geometry_filter_baselines.sh
chmod +x scripts/watch_phase_b_geometry_filter_baselines.sh
```

### 9.3 B0 smoke test

```bash
python scripts/run_env_v2_phase_b_geometry_filter_baselines.py \
  --out-dir results/env_v2_phase_b_geometry_filter_baselines_smoke \
  --checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --stage smoke \
  --num-episodes 3 \
  --scenarios eval_flow_id eval_flow_high_speed \
  --write-traces
```

### 9.4 正式 watcher

```bash
bash scripts/watch_phase_b_geometry_filter_baselines.sh
```

### 9.5 手动 formal 命令示例

```bash
python scripts/run_env_v2_phase_b_geometry_filter_baselines.py \
  --out-dir results/env_v2_phase_b_geometry_filter_baselines \
  --checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --stage full \
  --write-traces
```

### 9.6 分析脚本

```bash
python scripts/analyze_env_v2_phase_b_results.py \
  --result-dir results/env_v2_phase_b_geometry_filter_baselines
```

### 9.7 查看结果

```bash
find results/env_v2_phase_b_geometry_filter_baselines -maxdepth 3 -type f | sort
```

---

## 10. Phase B 报告要求

必须输出：

```text
results/env_v2_phase_b_geometry_filter_baselines/PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md
```

报告至少包含：

```text
1. 背景与 Phase B 目标；
2. Phase A 依赖与检查结果；
3. EnvV2-core freeze 复核；
4. 新增 / 修改文件清单；
5. baseline manifest；
6. 每个 baseline 的公式 / 参数 / config；
7. B0/B1/B2 eval 规模；
8. aggregate comparison；
9. Pareto frontier；
10. scenario-wise breakdown；
11. motion-mode breakdown；
12. threat-class breakdown；
13. filter intervention analysis；
14. failure cases；
15. top configs；
16. 是否有 geometry/filter baseline 打穿 attention_full；
17. 是否建议进入 Phase C safety-cost training；
18. Phase B completion criteria；
19. terminal_decision。
```

报告必须明确区分：

```text
experiment-supported facts
reasonable inferences
hypotheses for Phase C
```

---

## 11. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag
```

### 11.1 必须满足

```text
1. Phase A complete flag 存在；
2. EnvV2-core 未被主体修改；
3. attention_full checkpoint 可加载；
4. Phase B required baselines 均实现并写入 baseline_manifest；
5. B0 smoke test 完成；
6. B1 coarse audit 完成；
7. B2 formal confirmation 完成；
8. episode metrics CSV 非空；
9. eval summary CSV 非空；
10. Pareto table 非空；
11. scenario / motion / threat breakdown 非空；
12. filter intervention summary 非空；
13. 至少保存 sampled traces / failure traces；
14. plots 生成；
15. report 生成；
16. watcher log 和 status 文件存在；
17. 无未解释的 Python exception、empty CSV、schema mismatch。
```

### 11.2 如果 B2 未完成

如果 B0/B1 完成但 B2 因资源或运行时间无法完成：

```text
不允许生成 complete flag；
必须生成 PHASE_B_STOP_RESOURCE_LIMIT.flag；
必须生成 partial report；
terminal_decision = phase_b_stopped_resource_limit
```

---

## 12. 停止条件

### 12.1 Phase A 缺失

```text
PHASE_B_STOP_PHASE_A_MISSING.flag
```

### 12.2 需要修改 EnvV2-core

```text
PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag
```

### 12.3 checkpoint 缺失或无法加载

```text
PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag
```

### 12.4 baseline 实现失败

```text
PHASE_B_STOP_BASELINE_IMPL_FAILED.flag
```

### 12.5 eval 运行失败

```text
PHASE_B_STOP_EVAL_FAILED.flag
```

### 12.6 schema 不一致

```text
PHASE_B_STOP_SCHEMA_MISMATCH.flag
```

### 12.7 资源不足

```text
PHASE_B_STOP_RESOURCE_LIMIT.flag
```

### 12.8 watcher 失败

```text
PHASE_B_STOP_WATCHER_FAILED.flag
```

---

## 13. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_b_geometry_filter_baselines.sh
```

watcher 逻辑：

```text
1. 检查 Phase A complete flag；
2. 启动 B0 smoke；
3. 若 B0 通过，启动 B1 coarse audit；
4. 若 B1 通过，分析 top configs；
5. 启动 B2 formal confirmation；
6. 运行 analysis script；
7. 生成 report；
8. 只有检测到 complete flag 或 stop flag 才退出；
9. 不允许因为暂时没有新输出而退出；
10. 不允许只启动后台任务后结束；
11. 所有输出写入 phase_b_watcher.log；
12. 当前状态写入 phase_b_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_b_geometry_filter_baselines"
LOG="$OUT_DIR/phase_b_watcher.log"
STATUS="$OUT_DIR/phase_b_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$OUT_DIR/traces"

echo "[watcher] Phase B watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_a_eval_framework/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_B_STOP_PHASE_A_MISSING.flag"
fi

python scripts/run_env_v2_phase_b_geometry_filter_baselines.py \
  --out-dir "$OUT_DIR" \
  --checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --stage full \
  --write-traces \
  2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_B_STOP_PHASE_A_MISSING.flag \
    PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag \
    PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag \
    PHASE_B_STOP_BASELINE_IMPL_FAILED.flag \
    PHASE_B_STOP_EVAL_FAILED.flag \
    PHASE_B_STOP_SCHEMA_MISMATCH.flag \
    PHASE_B_STOP_RESOURCE_LIMIT.flag \
    PHASE_B_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_B_STOP_EVAL_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
```

Codex 可以根据实际脚本结构调整，但必须保持阻塞式语义。

---

## 14. Codex 执行原则

Codex 必须：

```text
1. 先阅读 Phase A 报告与本 Phase B 指南；
2. 复用 Phase A eval / trace schema；
3. 自己根据指南确定 complete / stop flag；
4. 不向用户询问非阻塞细节；
5. 优先做最小可运行版本，但不得省略 required baselines；
6. 没有 complete flag 或 stop flag，不得中途停止输出；
7. 如果触发阻塞，必须生成 stop flag、日志、partial report；
8. 如果完成，必须生成 complete flag、完整报告、CSV、plots、logs。
```

---

## 15. Phase B 结束时的 terminal_decision

成功：

```text
terminal_decision = phase_b_geometry_filter_baseline_complete
```

失败 / 停止：

```text
terminal_decision = phase_b_stopped_phase_a_missing
terminal_decision = phase_b_stopped_env_core_change_required
terminal_decision = phase_b_stopped_checkpoint_not_found
terminal_decision = phase_b_stopped_baseline_impl_failed
terminal_decision = phase_b_stopped_eval_failed
terminal_decision = phase_b_stopped_schema_mismatch
terminal_decision = phase_b_stopped_resource_limit
terminal_decision = phase_b_stopped_watcher_failed
```

---

## 16. Phase B 完成后进入 Phase C 的条件

只有当：

```text
PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag
```

存在，且报告明确写出：

```text
Phase B complete.
Geometry/filter baseline audit is ready for Phase C decision.
```

才允许进入 Phase C。

Phase C 才开始训练：

```text
attention_full + distance cost
attention_full + CPA cost
attention_full + TTC cost
attention_full + CPA/TTC combined cost
optional VO-style unsafe velocity cost
```

Phase B 不训练任何 safety-cost PPO。
