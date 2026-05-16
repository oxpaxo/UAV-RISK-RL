# Phase A 指南：EnvV2 评估框架与 Trace 统一化

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 阶段名称：`Phase A - Eval Framework & Trace Unification`  
> 阶段性质：工程地基阶段；不训练新策略；不做新 idea；不改 EnvV2-core。  
> 目标产物：统一 eval runner、统一 metrics、统一 per-step trace、阻塞式 watcher、Phase A 报告与完成/停止 flag。  

---

## 0. 背景简述

当前项目已经从旧 3-ball Gym 切换到 EnvV2 / `DynamicObstacleFlowEnv`。

旧 3-ball 阶段的 no-response / reaction oscillation 主线已经不再作为当前核心问题。EnvV2 Phase 2 显示：

```text
attention_full seed=0 训练到 1500k；
500k–1500k no_response_rate 均为 0；
旧 3-ball 的 long-training no-response / reaction oscillation 没有在 EnvV2 复现；
attention_full 1500k 平均 success ≈ 0.61，collision ≈ 0.39；
reactive collision ≈ 0.0867，但 success ≈ 0.0967。
```

因此当前问题不是：

```text
PPO 是否长训后 no-response？
```

而是：

```text
在 EnvV2 动态障碍物流中，attention_full PPO 为什么会前进但不够安全？
不同类别 baseline 的 safety-efficiency Pareto frontier 是什么？
```

接下来阶段不是直接提出新方法，而是执行：

```text
EnvV2 Baseline Strength Audit
```

Phase A 是该 audit 的第一步：先统一评估框架、trace schema 和报告格式。只有 Phase A 完成后，才能进入 Phase B 的 eval-only 几何 / filter baseline 比较。

---

## 1. Phase A 总目标

Phase A 的目标是建立一个统一、可复用、可追踪的评估框架，使后续所有 baseline 可以在同一套协议下比较。

Phase A 必须做到：

```text
1. 冻结 EnvV2-core；
2. 不训练新 PPO；
3. 不修改环境主体动力学、obstacle generation、reward、termination；
4. 建立统一 eval runner；
5. 支持多类 policy / controller / wrapper；
6. 统一 episode-level metrics CSV；
7. 统一 per-step trace schema；
8. safety-filter 类方法记录 raw action 与 filtered action；
9. 输出 Phase A markdown 报告；
10. 建立阻塞式 watcher；
11. 明确完成 flag 和停止 flag。
```

Phase A 不追求任何方法表现更好；只追求：

```text
所有 baseline 可以公平、可解释、可复现地被评估。
```

---

## 2. 明确禁止事项

Phase A 中禁止做以下事情：

```text
1. 禁止修改 EnvV2-core 的主体设定：
   - obstacle 数量范围；
   - motion modes；
   - train/eval scenarios；
   - collision / success / near_miss 定义；
   - action dynamics；
   - reward 函数；
   - termination 逻辑。

2. 禁止训练新策略：
   - 不训练 safety cost PPO；
   - 不训练 nearest-K MLP PPO；
   - 不训练 GRU / LSTM / temporal attention；
   - 不做 residual RL；
   - 不做 PPO-Lagrangian。

3. 禁止把 EWMA risk / risk-aware attention 作为新主线加入。

4. 禁止只为让某个 baseline 好看而改环境。

5. 禁止只输出 aggregate metrics 而没有 per-step trace。

6. 禁止 watcher 在未完成、未触发停止条件时提前退出。
```

允许做的修改只包括：

```text
1. 新增 eval script；
2. 新增 controller / policy adapter；
3. 新增 safety filter wrapper 的空接口或基础实现；
4. 新增 trace logger；
5. 新增 report generator；
6. 新增 watcher；
7. 新增结果目录与 flag 文件。
```

---

## 3. EnvV2-core 冻结声明

Phase A 必须在报告中显式写出 EnvV2-core freeze 声明。

建议写入 `PHASE_A_EVAL_FRAMEWORK_REPORT.md`：

```text
EnvV2-core was frozen in Phase A. This phase did not modify obstacle count ranges, motion modes, train/eval scenario definitions, action dynamics, reward function, collision/success/near-miss definitions, or termination logic. All changes were limited to evaluation infrastructure, policy/controller adapters, unified logging, trace schema, and watcher/report generation.
```

如果发现必须修改 EnvV2-core 才能继续，应立即停止，并生成 stop flag：

```text
PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag
```

---

## 4. 推荐目录结构

在项目根目录下建议新增：

```text
scripts/
  run_env_v2_phase_a_eval_framework.py
  watch_phase_a_eval_framework.sh

results/
  env_v2_phase_a_eval_framework/
    phase_a_status.txt
    phase_a_watcher.log
    PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag
    PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag
    PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag

    PHASE_A_EVAL_FRAMEWORK_REPORT.md

    tables/
      phase_a_eval_summary.csv
      phase_a_episode_metrics_sample.csv
      phase_a_trace_schema.csv
      phase_a_policy_adapter_check.csv
      phase_a_env_freeze_check.csv
      phase_a_command_manifest.csv

    traces/
      sample_random_trace.csv
      sample_straight_line_trace.csv
      sample_cpa_reactive_trace.csv
      sample_attention_full_trace.csv
      sample_filtered_attention_trace.csv

    logs/
      phase_a_eval_framework.log
```

如项目已有不同目录规范，可沿用现有风格，但必须保证：

```text
1. 所有 Phase A 产物集中在一个结果目录；
2. flag 文件路径固定；
3. watcher 能检测到完成或停止 flag；
4. 报告能引用所有关键产物路径。
```

---

## 5. 统一 eval runner 要求

建议新增或重构：

```text
scripts/run_env_v2_phase_a_eval_framework.py
```

该脚本应支持至少以下 policy/controller 类型：

```text
random
straight_line
cpa_reactive
attention_full
filtered_attention_full
```

Phase A 中 APF / VO / DWA 可以先只建立 adapter / placeholder 接口，不要求完成正式 sweep。Phase B 才跑完整 APF / VO / DWA baseline。

建议命令格式：

```bash
python scripts/run_env_v2_phase_a_eval_framework.py \
  --config configs/env_v2/phase_a_eval_framework.yaml \
  --out-dir results/env_v2_phase_a_eval_framework \
  --num-episodes 3 \
  --scenarios eval_flow_id eval_flow_high_speed \
  --policies random straight_line cpa_reactive attention_full filtered_attention_full \
  --checkpoint work_dirs/env_v2_attention_full_seed0/1500k.zip \
  --write-traces \
  --dry-run false
```

如果项目没有 yaml config，也可使用 argparse 纯命令行参数。

最低测试规模建议：

```text
scenarios:
  eval_flow_id
  eval_flow_high_speed

policies:
  random
  straight_line
  cpa_reactive
  attention_full
  filtered_attention_full

episodes per scenario-policy:
  2–3 episodes

目的：
  验证框架可运行，不追求统计显著性。
```

Phase A 完成不要求跑 6 scenarios × 50 episodes。完整规模留给 Phase B。

---

## 6. policy/controller adapter 规范

所有 policy/controller 都应统一暴露：

```python
def act(obs, info=None) -> dict:
    return {
        "action_raw": np.ndarray shape (3,),
        "action_filtered": np.ndarray shape (3,),
        "filter_triggered": bool,
        "filter_reason": str,
        "debug": dict,
    }
```

### 6.1 random

```text
action_raw:
  random sample from action space

action_filtered:
  same as action_raw

filter_triggered:
  false
```

### 6.2 straight_line

```text
action_raw:
  horizontal goal direction normalized to action scale

action_filtered:
  same as action_raw

filter_triggered:
  false
```

### 6.3 cpa_reactive

应复用当前 sanity 脚本中的 reactive 逻辑，不要重新发明一个不同版本。

当前 reactive 已知逻辑：

```text
goal direction:
  v_goal = normalize(goal - uav), z set to 0

parameters:
  d_reactive = 4.0
  horizon = 4.5
  CPA trigger distance = 2.4

for each obstacle:
  rel = obs_pos - uav
  rel_vel = obs_vel - uav_vel
  tcpa = clip(-dot(rel, rel_vel) / ||rel_vel||^2, 0, horizon)
  cpa_distance = ||rel + rel_vel * tcpa||

trigger if:
  distance < 4.0
  or
  0 < tcpa < 4.5 and cpa_distance < 2.4

avoidance:
  away = -rel / distance
  lateral = +/- perpendicular(v_goal) chosen toward away
  proximity_gain = (4.0 - min(distance, 4.0)) / 4.0
  cpa_gain = (2.4 - min(cpa_distance, 2.4)) / 2.4
  closing_gain = 1.0 if tcpa > 0 else 0.35
  avoid += (1.3 * proximity_gain + 2.2 * cpa_gain * closing_gain)
           * (0.65 * away + 0.55 * lateral)

final:
  command = normalize(v_goal + 2.1 * avoid)
```

Phase A 只需接入 current version，不做参数 sweep。参数 sweep 留到 Phase B。

### 6.4 attention_full

应加载已有 EnvV2 Phase 2 的 attention_full checkpoint，默认使用 best / 1500k checkpoint。

如 checkpoint 路径不可自动识别，应在报告中列出需要用户确认的路径，并触发 stop flag：

```text
PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag
```

### 6.5 filtered_attention_full

Phase A 中可以实现一个 minimal safety filter，用于验证 trace schema，不要求作为最终 baseline。

最低要求：

```text
1. 先调用 attention_full 得到 action_raw；
2. 调用 safety filter wrapper；
3. 输出 action_filtered；
4. 记录 filter_triggered / filter_reason / filter_delta_norm；
5. per-step trace 中能看到 raw 与 filtered 的差异。
```

Phase A 的 filter 可以非常保守或简单，但必须清楚标注：

```text
This minimal filter is for infrastructure validation only and is not a formal Phase B baseline.
```

---

## 7. episode-level metrics CSV 统一字段

建议 `phase_a_episode_metrics_sample.csv` 至少包含：

```text
method
policy_name
scenario
checkpoint_step
episode_id
episode_seed

success
collision
timeout
truncated
out_of_bounds
near_miss

progress
final_goal_distance
mean_time
episode_length_steps
episode_return

mean_min_distance
episode_min_distance
min_distance_after_threat

no_response
no_response_rate
reaction_time_eval_style
conditional_reaction_time

planned_cpa
planned_ttc
threat_class
motion_mode
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

如某些字段对某 baseline 不适用，用：

```text
NaN
```

不要删除列。

---

## 8. per-step trace schema

所有 baseline 的 per-step trace 应统一字段。建议至少包含：

```text
checkpoint_step
method
policy_name
scenario
episode_id
episode_seed

step
time
done
terminated
truncated

uav_pos_x
uav_pos_y
uav_pos_z
uav_vel_x
uav_vel_y
uav_vel_z

goal_pos_x
goal_pos_y
goal_pos_z
goal_dist

action_raw_x
action_raw_y
action_raw_z
action_filtered_x
action_filtered_y
action_filtered_z
action_executed_x
action_executed_y
action_executed_z

filter_used
filter_triggered
filter_reason
filter_delta_norm

min_distance
nearest_obstacle_id
nearest_obstacle_distance

threat_obstacle_id
threat_obstacle_index
threat_class
planned_cpa
planned_ttc
planned_ttc_remaining

lateral_deviation
away_from_threat_velocity
goal_directed_velocity
reaction_flag
no_response_flag

attention_entropy
threat_obstacle_attention_weight
threat_obstacle_attention_rank
```

### 8.1 full active obstacle set 记录

除了主 trace，每个 step 必须记录 full active obstacle set。可以采用两种方式之一。

#### 方案 A：wide columns

适合 max_obs=10：

```text
obs0_id
obs0_active
obs0_pos_x
obs0_pos_y
obs0_pos_z
obs0_vel_x
obs0_vel_y
obs0_vel_z
obs0_distance
obs0_closing
obs0_planned_cpa
obs0_planned_ttc
obs0_threat_class
obs0_motion_mode
...
obs9_motion_mode
```

#### 方案 B：long table

单独输出：

```text
phase_a_step_obstacles_sample.csv
```

字段：

```text
method
scenario
episode_id
episode_seed
step
time
obstacle_slot
obstacle_id
active
pos_x
pos_y
pos_z
vel_x
vel_y
vel_z
distance
closing
planned_cpa
planned_ttc
threat_class
motion_mode
risk_value
```

推荐方案 B，因为更清楚，也便于后续 groupby 分析。

Phase A 至少应实现一种。如果暂时只能实现 wide columns，应在报告中说明。

---

## 9. safety filter trace 字段要求

任何 filtered policy 都必须记录：

```text
action_raw_x
action_raw_y
action_raw_z

action_filtered_x
action_filtered_y
action_filtered_z

action_executed_x
action_executed_y
action_executed_z

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

如果 Phase A 的 minimal filter 暂时没有实现 `min_predicted_cpa_filtered` 等字段，可以先写 NaN，但列必须存在，并在报告中说明：

```text
Columns reserved for Phase B formal safety-filter baselines.
```

---

## 10. 统一 seed 规则

Phase A 应尽量使用和 Phase 2 attention eval 一致的 seed 规则：

```text
episode_seed = eval_seed + seed * 10000 + episode_id
```

例如：

```text
seed = 0
eval_seed = 1000
episode_id = 0..49
=> episode_seed = 1000..1049
```

但报告中必须说明：

```text
Because obstacle replacement depends on policy trajectory, identical reset seed does not guarantee identical obstacle schedules across policies. Phase A standardizes initial seeds and trace logging; stricter precomputed spawn schedule / decoupled obstacle RNG is deferred unless required by later comparisons.
```

如果 Codex 能低成本实现 decoupled obstacle RNG / precomputed spawn schedule，可在报告中列为 optional extension，但不要把它作为 Phase A 必需完成项，除非现有评估明显不可复现。

---

## 11. 命令清单

Codex 应根据项目实际路径调整命令，但报告中必须记录最终实际使用命令。

### 11.1 进入项目

示例：

```bash
cd /root/workspace/uav-risk-rl
```

### 11.2 快速检查环境

```bash
python - <<'PY'
import sys
print(sys.version)
try:
    import gymnasium
    print("gymnasium ok")
except Exception as e:
    print("gymnasium import failed:", e)

try:
    import stable_baselines3
    print("stable_baselines3 ok")
except Exception as e:
    print("stable_baselines3 import failed:", e)
PY
```

### 11.3 运行 Phase A eval framework

示例：

```bash
python scripts/run_env_v2_phase_a_eval_framework.py \
  --out-dir results/env_v2_phase_a_eval_framework \
  --num-episodes 3 \
  --scenarios eval_flow_id eval_flow_high_speed \
  --policies random straight_line cpa_reactive attention_full filtered_attention_full \
  --checkpoint work_dirs/env_v2_attention_full_seed0/1500k.zip \
  --eval-seed 1000 \
  --write-traces
```

### 11.4 查看结果文件

```bash
find results/env_v2_phase_a_eval_framework -maxdepth 3 -type f | sort
```

### 11.5 检查 CSV 头部

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results/env_v2_phase_a_eval_framework")
for p in sorted(root.rglob("*.csv")):
    try:
        df = pd.read_csv(p, nrows=3)
        print("\n==", p)
        print(df.head())
        print("columns:", list(df.columns))
    except Exception as e:
        print("failed:", p, e)
PY
```

### 11.6 启动 watcher

```bash
bash scripts/watch_phase_a_eval_framework.sh
```

---

## 12. Phase A 报告要求

必须输出：

```text
results/env_v2_phase_a_eval_framework/PHASE_A_EVAL_FRAMEWORK_REPORT.md
```

报告至少包含：

```text
1. 背景与阶段目标；
2. EnvV2-core freeze 声明；
3. 新增 / 修改文件清单；
4. eval runner 使用说明；
5. 支持的 policies / controllers；
6. episode-level metrics schema；
7. per-step trace schema；
8. full active obstacle set logging 方式；
9. safety filter trace 字段说明；
10. seed 规则与公平性说明；
11. smoke test 规模与结果；
12. 已生成 CSV / trace / logs / flags 路径；
13. Phase A 完成判据；
14. 若有未完成项，列出原因与是否阻塞 Phase B。
```

报告中必须明确结论：

```text
Phase A complete / not complete
```

---

## 13. 完成判据

只有同时满足以下条件，才允许生成完成 flag：

```text
results/env_v2_phase_a_eval_framework/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag
```

### 13.1 必须满足

```text
1. EnvV2-core 未被主体修改；
2. 统一 eval runner 存在并可运行；
3. 至少完成 smoke test：
   - scenarios 至少包括 eval_flow_id 和 eval_flow_high_speed；
   - policies 至少包括 random、straight_line、cpa_reactive；
   - 若 checkpoint 可用，还应包括 attention_full；
4. episode-level metrics CSV 存在，且包含统一字段；
5. per-step trace CSV 存在；
6. full active obstacle set 被记录；
7. filtered policy 的 action_raw / action_filtered 字段存在；
8. Phase A markdown report 存在；
9. watcher log 存在；
10. phase_a_status.txt 存在；
11. 无 Python exception / missing file / empty CSV / schema mismatch。
```

### 13.2 checkpoint 缺失时的特殊规则

如果 attention_full checkpoint 路径找不到，但其他框架均可运行，应触发：

```text
PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag
```

并在报告中说明：

```text
Phase A infrastructure is partially complete, but attention_full adapter could not be validated because the checkpoint path was missing.
```

这种情况不允许生成 complete flag。

---

## 14. 停止条件

如果出现以下情况，Codex 必须停止继续执行，并生成对应 stop flag 与报告。

### 14.1 需要修改 EnvV2-core

```text
PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag
```

触发条件：

```text
不修改 EnvV2-core 就无法实现统一 eval / trace；
或发现现有环境接口缺失关键状态且只能通过改环境主体解决。
```

### 14.2 评估框架运行失败

```text
PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag
```

触发条件：

```text
统一 eval runner 无法完成 smoke test；
CSV 为空；
trace 为空；
核心 import 失败；
环境 reset/step 失败。
```

### 14.3 checkpoint 缺失

```text
PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag
```

触发条件：

```text
attention_full checkpoint 不存在或无法加载。
```

### 14.4 schema 无法统一

```text
PHASE_A_STOP_SCHEMA_UNIFICATION_FAILED.flag
```

触发条件：

```text
不同 policy/controller 输出无法统一到共同 schema；
或必须大量删除关键字段才能运行。
```

### 14.5 watcher 自身失败

```text
PHASE_A_STOP_WATCHER_FAILED.flag
```

触发条件：

```text
watcher 无法运行；
watcher 无法检测 flag；
watcher 无法写入 log/status。
```

---

## 15. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_a_eval_framework.sh
```

watcher 逻辑：

```text
1. 启动 Phase A eval framework；
2. 持续轮询结果目录；
3. 持续输出当前状态；
4. 只有检测到 complete flag 或 stop flag 才退出；
5. 不允许因为中间暂无新日志就退出；
6. 不允许只启动后台任务后立即结束；
7. 所有输出写入 phase_a_watcher.log；
8. 当前状态写入 phase_a_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_a_eval_framework"
LOG="$OUT_DIR/phase_a_watcher.log"
STATUS="$OUT_DIR/phase_a_status.txt"

mkdir -p "$OUT_DIR"

echo "[watcher] Phase A watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

python scripts/run_env_v2_phase_a_eval_framework.py \
  --out-dir "$OUT_DIR" \
  --num-episodes 3 \
  --scenarios eval_flow_id eval_flow_high_speed \
  --policies random straight_line cpa_reactive attention_full filtered_attention_full \
  --eval-seed 1000 \
  --write-traces \
  2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag \
    PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag \
    PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag \
    PHASE_A_STOP_SCHEMA_UNIFICATION_FAILED.flag \
    PHASE_A_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 20
done
```

Codex 可以根据项目实际情况改写，但必须保持阻塞式语义。

---

## 16. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 自己根据本指南确定完成 flag 与 stop flag；
3. 不向用户反复询问非阻塞细节；
4. 优先做最小可运行版本；
5. 只要没有 complete flag 或 stop flag，就继续执行和输出；
6. 不得中途“暂停等待用户确认”；
7. 如果发现阻塞问题，必须生成 stop flag、报告和日志；
8. 如果任务完成，必须生成 complete flag、报告和日志。
```

---

## 17. Phase A 结束后应给用户的结论格式

最终报告应给出类似结论：

```text
terminal_decision = phase_a_eval_framework_complete
```

或：

```text
terminal_decision = phase_a_stopped_checkpoint_not_found
```

或：

```text
terminal_decision = phase_a_stopped_env_core_change_required
```

并说明：

```text
1. 是否可以进入 Phase B；
2. 如果不能，阻塞原因是什么；
3. 需要用户提供什么；
4. 哪些文件已经生成；
5. 哪些命令已经验证。
```

---

## 18. Phase A 完成后进入 Phase B 的条件

只有当：

```text
PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag
```

存在，且报告明确写出：

```text
Phase A complete.
Unified eval framework and trace schema are ready for Phase B.
```

才允许进入 Phase B。

Phase B 才开始正式比较：

```text
current CPA-reactive
CPA-reactive sweep
naive APF
velocity-aware APF
CPA/TTC-weighted APF
attention_full 1500k
attention_full + distance filter
attention_full + CPA/TTC filter
attention_full + VO-like filter
```

Phase A 不需要完成这些正式 baseline 结果。
