# Phase N0 指南：Gψ-HeadA 设计冻结与 EnvV2 数据链路审计

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 新主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N0 - Design Freeze & Dataflow Audit`  
> 阶段性质：设计冻结 + 代码审计 + 数据可得性检查；不训练；不构建正式数据集；不改 EnvV2-core。  
> 前置状态：EnvV2 已冻结；旧 Phase A/B 已完成；Phase B 证明 geometry/filter baseline 很强，后续方法必须与其正面对比。

---

## 0. 背景简述

当前项目已经从旧的 EnvV2 baseline audit / safety-cost Phase C 路线，切换到新的三段式主线：

```text
监督动态障碍物表征网络 Gψ
→ PPO velocity policy
→ 后置 Safety Shield
```

旧 3-ball 环境中的 long-training no-response / reaction oscillation 没有在 EnvV2 里复现，因此不再作为核心问题。

Phase A 已完成统一 eval / trace 框架。Phase B 已完成几何 / filter baseline audit。关键结果包括：

```text
attention_full_1500k:
  success ≈ 0.6100
  collision ≈ 0.3900

vo_like_filter_h45_cpa1p2_h16:
  success ≈ 0.8333
  collision ≈ 0.1667

cpa_ttc_weighted_apf_alpha3:
  success ≈ 0.7200
  collision ≈ 0.2800
```

因此，后续新方法不能只和 `attention_full` 比，必须正面对比强 geometry/filter baseline。

新方案第一版是：

```text
Gψ:
  action-independent per-obstacle dynamic representation network.
  Head A predicts residual Δ̂_i(τ) and uncertainty log σ̂_i²(τ).

PPO:
  uses augmented obstacle feature:
  obs_i_aug = [obs_i, z_i, Δ̂_i, σ̂_i²]

Safety Shield:
  post-hoc analytic VO/CPA-TTC candidate shield.
  safety margin uses σ̂_i².
```

Phase N0 的目标不是训练模型，而是确认这条数据链路在当前代码中可实现、字段可取、定义固定、风险点明确。N0 通过后才能进入 N1 数据集构建。

---

## 1. Phase N0 总目标

Phase N0 必须完成：

```text
1. 冻结第一版网络设计 spec；
2. 审计 EnvV2 / Phase A/B trace 是否支持 Gψ 数据构建；
3. 明确 obstacle history 的构造方法；
4. 明确 future label Δ_i(τ) 的构造方法；
5. 明确 obstacle id / slot / replacement 对齐规则；
6. 明确坐标系：world frame / relative frame 的取舍；
7. 明确 σ̂_i² 的训练方式；
8. 明确 N3 中 Gψ 默认 frozen；
9. 明确 N4 中 λ_uncertainty sweep；
10. 输出 report、schema、审计脚本、complete/stop flag 和阻塞式 watcher。
```

Phase N0 必须回答一个问题：

```text
当前代码与 trace 是否足以可靠构建 Gψ-HeadA 监督数据，并在后续接入 PPO + shield？
```

---

## 2. 本阶段必须写入 spec 的三个修订

Phase N0 必须把以下三个修订写入固定 spec 和报告。

### 2.1 修订一：σ² 不是直接标签

不要写成：

```text
future trajectory 生成 Δ_i(τ), σ_i²(τ) 标签
```

更严谨的写法是：

```text
future trajectory 直接生成 Δ_i(τ) 监督真值；
Gψ 输出 Δ̂_i(τ) 和 log σ̂_i²(τ)；
σ̂_i² 通过 Gaussian NLL 从 residual prediction error 中学习异方差不确定性。
```

第一版 loss 顺序：

```text
Stage A1:
  Δ-only MSE / SmoothL1

Stage A2:
  Gaussian NLL:
  L_NLL = 0.5 * [ ||Δ - Δ̂||² / σ̂² + log σ̂² ]

log σ̂² clamp:
  e.g. [-5, 3]，可按数据尺度调整。
```

除非后续真的实现了 empirical variance label，例如同一状态多 rollout 统计，否则不得把 σ² 写成 per-sample direct label。

### 2.2 修订二：N3 第一版默认 freeze Gψ

Phase N3 `Gψ-HeadA + PPO no shield` 中，第一版默认：

```text
Gψ frozen
PPO trainable
```

原因：

```text
1. 先验证离线监督学到的动态表征作为固定特征是否有用；
2. 避免 PPO objective 破坏 Gψ 的 residual / uncertainty 表征；
3. 如果 joint training 失败，不会混淆“Gψ 表征没用”和“联合训练不稳定”。
```

后续可以做：

```text
Gψ frozen vs Gψ fine-tuned
```

作为 ablation，但不进入第一版默认主线。

### 2.3 修订三：N4 必须做 λ_uncertainty sweep

Phase N4 `σ²-margin shield` 不允许只测单个 λ。

必须至少设计：

```text
λ_uncertainty ∈ {0, small, medium, large}
```

其中：

```text
λ = 0:
  等价于 fixed-margin shield。

λ > 0:
  才是 uncertainty-aware σ²-margin shield。
```

必须记录：

```text
collision
success
near_miss
progress
filter_trigger_rate
mean_filter_delta_norm
raw unsafe action rate
raw CPA
filtered CPA
```

如果 σ²-margin 结果不好，需要判断是：

```text
1. σ² 没有信息；
2. λ 设置不合适；
3. shield 过保守；
4. Gψ uncertainty calibration 不好。
```

---

## 3. 明确禁止事项

Phase N0 禁止：

```text
1. 禁止修改 EnvV2-core：
   - obstacle 数量范围；
   - motion modes；
   - train/eval scenario；
   - action dynamics；
   - reward；
   - termination；
   - collision/success/near_miss 定义。

2. 禁止训练 Gψ；
3. 禁止训练 PPO；
4. 禁止构建正式大规模数据集；
5. 禁止实现完整 Safety Shield；
6. 禁止引入 5-head Gψ；
7. 禁止引入 learned R(s,a)；
8. 禁止引入 candidate velocity risk map as PPO input；
9. 禁止把 old safety-cost Phase C 当作当前阶段目标；
10. 禁止跳过 obstacle id / replacement 审计直接进入 N1。
```

允许：

```text
1. 新增设计 spec yaml；
2. 新增字段审计脚本；
3. 新增数据链路 dry-run / small rollout check；
4. 新增 schema json；
5. 新增 report；
6. 新增 watcher；
7. 新增 stop / complete flag；
8. 只读式检查 EnvV2、Phase A/B traces 和 current scripts。
```

---

## 4. 推荐输出目录

建议新增：

```text
configs/
  gpsi_head_a_spec.yaml

scripts/
  check_envv2_gpsi_required_fields.py
  watch_phase_n0_design_freeze.sh

results/
  env_v2_phase_n0_design_freeze/
    phase_n0_status.txt
    phase_n0_watcher.log

    PHASE_N0_DESIGN_FREEZE_COMPLETE.flag
    PHASE_N0_STOP_ENV_CORE_CHANGE_REQUIRED.flag
    PHASE_N0_STOP_OBSTACLE_ID_ALIGNMENT_FAILED.flag
    PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag
    PHASE_N0_STOP_REQUIRED_FIELDS_MISSING.flag
    PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag
    PHASE_N0_STOP_SPEC_CONFLICT.flag
    PHASE_N0_STOP_WATCHER_FAILED.flag

    PHASE_N0_DESIGN_FREEZE_REPORT.md

    tables/
      phase_n0_required_fields_check.csv
      phase_n0_obstacle_id_alignment_check.csv
      phase_n0_history_future_label_check.csv
      phase_n0_coordinate_frame_check.csv
      phase_n0_phase_ab_artifact_check.csv
      phase_n0_spec_freeze_check.csv
      phase_n0_command_manifest.csv

    schema/
      gpsi_head_a_dataset_schema_draft.json
      gpsi_head_a_model_io_schema_draft.json

    logs/
      phase_n0_design_freeze.log
```

---

## 5. 固定设计 spec 要求

Codex 必须创建：

```text
configs/gpsi_head_a_spec.yaml
```

建议内容至少包括：

```yaml
version: gpsi_head_a_v1

environment:
  name: DynamicObstacleFlowEnv
  env_core_frozen: true

observation:
  ego_dim: 10
  obs_dim_per_obstacle: 12
  max_obs: 10
  mask_dim: 10
  uses_obstacle_history: true

gpsi:
  action_independent: true
  per_obstacle_encoder: true
  encoder_type_first_version: temporal_mlp_or_gru
  z_dim: 64
  head_a_required: true
  head_c_optional_regularizer: true
  disabled_heads:
    - head_b_delta_vo
    - head_d_v_nom
    - head_e_p_trigger

head_a:
  horizons_sec: [1.0, 2.0, 4.0]
  target_type: residual_to_constant_velocity
  residual_definition: "delta_i(tau)=p_i(t+tau)-[p_i(t)+tau*v_i(t)]"
  direct_label: delta_only
  uncertainty_learning: gaussian_nll
  logvar_clamp: [-5, 3]
  sigma2_is_direct_label: false

ppo:
  backbone_first_version: masked_attention_compatible
  actor_critic: symmetric
  gpsi_frozen_in_first_ppo_phase: true
  input_aug: "[obs_i, z_i, delta_hat_i, sigma2_hat_i]"

shield:
  type_first_version: vo_like_candidate_search
  uses_uncertainty_margin: true
  margin_formula: "r_safe_i=base_radius+lambda_uncertainty*sqrt(sigma2_hat_i)"
  lambda_uncertainty_sweep_required: true
  lambda_uncertainty_values: [0.0, small, medium, large]
  q_p_or_cbf_deferred: true

deferred:
  - learned_R_s_a
  - candidate_velocity_risk_map_as_ppo_input
  - full_5_head_gpsi
  - asymmetric_critic
  - passing_rule_mlp_first_version
  - qp_projection_first_version
```

如果项目使用 Python config 而非 YAML，也可以生成等价 `.py` / `.json`，但 report 必须写清楚 spec 路径。

---

## 6. 字段审计要求

建议新增：

```text
scripts/check_envv2_gpsi_required_fields.py
```

该脚本必须检查以下内容。

### 6.1 EnvV2 observation 字段

确认是否可取：

```text
ego state
obs_i current profile
mask
relative position
relative velocity
planned CPA/TTC
distance
closing
threat class
risk value
```

### 6.2 full active obstacle state

确认 Phase A/B trace 或 Env info 是否可取：

```text
obstacle id
obstacle slot
active flag
world position
world velocity
relative position
relative velocity
motion mode
threat class
planned CPA
planned TTC
risk value
```

### 6.3 history 构造

检查能否构造：

```text
history_i[t-H:t]
```

要求：

```text
1. 同一 obstacle id 连续；
2. active mask 正确；
3. episode 边界处截断；
4. replacement 后不把新障碍物拼进旧障碍物 history；
5. 缺失历史可以 padding，但必须有 valid_history_mask。
```

### 6.4 future label 构造

检查能否构造：

```text
p_i(t+τ), τ ∈ {1s,2s,4s}
```

要求：

```text
1. 同一 obstacle id 在未来仍 active；
2. 如果未来发生 replacement，则该 horizon label 无效；
3. 如果 episode 结束，则该 horizon label 无效；
4. 输出 valid_future_mask；
5. 不使用 future 信息作为推理输入。
```

### 6.5 坐标系检查

必须明确：

```text
推荐：
  Head A label 使用 obstacle world-frame residual：
  Δ_i(τ)=p_i_world(t+τ)-[p_i_world(t)+τ*v_i_world(t)]

原因：
  避免 future UAV motion 污染 obstacle dynamics label。
```

如果代码只能使用 relative frame，必须在 report 中说明：

```text
1. relative residual 是否混入 ego future motion；
2. 是否能用 ego current frame 固定；
3. 是否会导致 Gψ 学到 policy-dependent label；
4. 是否需要补 world-frame logging。
```

如果无法避免 label 被 ego future motion 污染，应触发：

```text
PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag
```

---

## 7. small dry-run 检查

Phase N0 不构建正式数据集，但建议做 small dry-run：

```text
scenarios:
  train_flow_mixed 或 eval_flow_id

episodes:
  2–5

horizons:
  1s, 2s, 4s

history:
  H steps，例如 10 或 20
```

输出检查表：

```text
phase_n0_history_future_label_check.csv
```

至少包含：

```text
episode_id
step
obstacle_id
obstacle_slot
history_valid
future_valid_1s
future_valid_2s
future_valid_4s
delta_norm_1s
delta_norm_2s
delta_norm_4s
motion_mode
replacement_boundary_nearby
reason_invalid_if_any
```

检查重点：

```text
1. linear mode Δ 是否大体接近 0；
2. nonlinear / stochastic mode Δ 是否明显更大；
3. replacement 附近是否被正确标 invalid；
4. inactive obstacle 是否被排除；
5. horizon 对应的 step offset 是否正确：
   tau_steps = round(tau / dt)
   dt 默认 0.2s，因此 1s=5 steps, 2s=10 steps, 4s=20 steps。
```

---

## 8. Phase A/B 产物依赖检查

Phase N0 必须检查：

```text
1. Phase A complete flag；
2. Phase B complete flag；
3. Phase A trace schema；
4. Phase B baseline report；
5. attention_full checkpoint；
6. Phase A/B 结果目录是否存在。
```

如果 Phase A/B 缺失，不一定完全阻塞 N0，但必须在报告中说明哪些内容缺失。

若缺失会导致无法验证统一 trace / obstacle long table，则触发：

```text
PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag
```

---

## 9. 命令清单

Codex 应根据实际项目路径调整命令，并在报告中记录最终实际命令。

### 9.1 进入项目

```bash
cd /root/workspace/uav-risk-rl
```

### 9.2 快速语法检查

```bash
python -m py_compile scripts/check_envv2_gpsi_required_fields.py
bash -n scripts/watch_phase_n0_design_freeze.sh
chmod +x scripts/watch_phase_n0_design_freeze.sh
```

### 9.3 运行字段审计

示例：

```bash
python scripts/check_envv2_gpsi_required_fields.py \
  --out-dir results/env_v2_phase_n0_design_freeze \
  --spec configs/gpsi_head_a_spec.yaml \
  --phase-a-dir results/env_v2_phase_a_eval_framework \
  --phase-b-dir results/env_v2_phase_b_geometry_filter_baselines \
  --scenarios eval_flow_id train_flow_mixed \
  --num-episodes 3 \
  --history-steps 20 \
  --future-times 1.0 2.0 4.0 \
  --write-dryrun-tables
```

### 9.4 启动 watcher

```bash
bash scripts/watch_phase_n0_design_freeze.sh
```

### 9.5 查看结果

```bash
find results/env_v2_phase_n0_design_freeze -maxdepth 3 -type f | sort
```

### 9.6 检查 CSV 头部

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results/env_v2_phase_n0_design_freeze")
for p in sorted(root.rglob("*.csv")):
    print("\n==", p)
    try:
        df = pd.read_csv(p, nrows=5)
        print(df.head())
        print("columns:", list(df.columns))
    except Exception as e:
        print("failed:", e)
PY
```

---

## 10. Phase N0 报告要求

必须输出：

```text
results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_REPORT.md
```

报告至少包含：

```text
1. 背景与当前新主线；
2. 明确旧 Phase C/D/E 作废；
3. EnvV2-core freeze 复核；
4. 三个小修订的落实：
   - σ² 不作为直接标签；
   - N3 默认 freeze Gψ；
   - N4 λ_uncertainty sweep；
5. Gψ-HeadA fixed spec；
6. observation / trace / obstacle long table 字段审计结果；
7. obstacle id / replacement 对齐规则；
8. history 构造规则；
9. future label 构造规则；
10. 坐标系选择；
11. dry-run 结果；
12. 是否需要修 trace / env info；
13. N1 dataset builder 的输入输出建议；
14. complete / stop 判据；
15. terminal_decision。
```

报告必须明确区分：

```text
experiment/code-audit supported facts
design decisions
risks / unresolved issues
```

---

## 11. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N0_DESIGN_FREEZE_COMPLETE.flag
```

### 11.1 必须满足

```text
1. EnvV2-core 未被主体修改；
2. configs/gpsi_head_a_spec.yaml 或等价 spec 存在；
3. spec 明确禁用 5-head / learned R(s,a) / candidate velocity risk map as PPO input；
4. spec 明确 σ² 不是 direct label；
5. spec 明确 N3 Gψ frozen first；
6. spec 明确 N4 λ_uncertainty sweep；
7. 字段审计脚本存在并可运行；
8. required fields check 表存在；
9. obstacle id alignment check 表存在；
10. history/future label dry-run 表存在；
11. 坐标系选择已写入 report；
12. replacement / invalid horizon 处理规则已写入 report；
13. report 生成；
14. watcher log 与 status 文件存在；
15. 无未解释的 Python exception、empty required table、schema conflict。
```

### 11.2 允许非阻塞项

以下可以作为 warning，不阻塞 complete flag，但必须写入 report：

```text
1. Phase A/B 某些非核心 plot 缺失；
2. Head C label 暂未完全确认；
3. formal dataset size 未统计；
4. λ_uncertainty 具体数值尚未最终定量，只给出 small/medium/large 占位；
5. Gψ encoder 具体 GRU / temporal MLP 尚未定型，但输入输出已定。
```

---

## 12. 停止条件

### 12.1 需要修改 EnvV2-core

```text
PHASE_N0_STOP_ENV_CORE_CHANGE_REQUIRED.flag
```

触发条件：

```text
不修改 EnvV2-core 就无法获取必要字段；
或必须改环境核心动力学/生成逻辑才能构建标签。
```

### 12.2 obstacle id 对齐失败

```text
PHASE_N0_STOP_OBSTACLE_ID_ALIGNMENT_FAILED.flag
```

触发条件：

```text
无法确认同一 obstacle 在 history 和 future 中的身份；
replacement 后 id/slot 混乱；
无法避免把不同障碍物拼成同一条轨迹。
```

### 12.3 history/future label 构造失败

```text
PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag
```

触发条件：

```text
无法构造 p_i(t+τ)；
horizon offset 不可靠；
relative label 混入不可接受的 ego future motion；
future mask 无法可靠生成。
```

### 12.4 必需字段缺失

```text
PHASE_N0_STOP_REQUIRED_FIELDS_MISSING.flag
```

触发条件：

```text
无法取得 obstacle pos/vel/id/mask 或 ego state 等 N1 必需字段。
```

### 12.5 trace schema 不足

```text
PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag
```

触发条件：

```text
Phase A/B trace 或 obstacle long table 不足以支持 dataset builder，
且无法通过非侵入式 eval info logging 获取。
```

### 12.6 spec 冲突

```text
PHASE_N0_STOP_SPEC_CONFLICT.flag
```

触发条件：

```text
三个上传设计文件之间存在无法自动解决的结构冲突；
例如是否直接监督 σ²、是否 PPO 反向更新 Gψ、是否使用 action-conditioned R(s,a)。
```

### 12.7 watcher 失败

```text
PHASE_N0_STOP_WATCHER_FAILED.flag
```

触发条件：

```text
watcher 无法运行；
无法检测 complete/stop flag；
无法写入 log/status。
```

---

## 13. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n0_design_freeze.sh
```

watcher 逻辑：

```text
1. 启动 Phase N0 字段审计脚本；
2. 持续轮询结果目录；
3. 持续输出当前状态；
4. 只有检测到 complete flag 或 stop flag 才退出；
5. 不允许因为中间暂无新日志就退出；
6. 不允许只启动后台任务后立即结束；
7. 所有输出写入 phase_n0_watcher.log；
8. 当前状态写入 phase_n0_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n0_design_freeze"
LOG="$OUT_DIR/phase_n0_watcher.log"
STATUS="$OUT_DIR/phase_n0_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/schema"

echo "[watcher] Phase N0 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

python scripts/check_envv2_gpsi_required_fields.py \
  --out-dir "$OUT_DIR" \
  --spec configs/gpsi_head_a_spec.yaml \
  --phase-a-dir results/env_v2_phase_a_eval_framework \
  --phase-b-dir results/env_v2_phase_b_geometry_filter_baselines \
  --scenarios eval_flow_id train_flow_mixed \
  --num-episodes 3 \
  --history-steps 20 \
  --future-times 1.0 2.0 4.0 \
  --write-dryrun-tables \
  2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N0_STOP_ENV_CORE_CHANGE_REQUIRED.flag \
    PHASE_N0_STOP_OBSTACLE_ID_ALIGNMENT_FAILED.flag \
    PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag \
    PHASE_N0_STOP_REQUIRED_FIELDS_MISSING.flag \
    PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag \
    PHASE_N0_STOP_SPEC_CONFLICT.flag \
    PHASE_N0_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 20
done
```

Codex 可以按实际脚本结构调整，但必须保持阻塞式语义。

---

## 14. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 同步阅读三个最新设计文件；
3. 不沿用旧 Phase C/D/E；
4. 根据指南自行确定 complete / stop flag；
5. 创建并运行阻塞式 watcher；
6. 只有 complete flag 或 stop flag 出现才停止本轮输出；
7. 不向用户询问非阻塞细节；
8. 优先做最小可运行审计；
9. 如果触发阻塞，必须生成 stop flag、partial report、log；
10. 如果完成，必须生成 complete flag、完整 report、CSV/schema/log。
```

---

## 15. Phase N0 结束时的 terminal_decision

成功：

```text
terminal_decision = phase_n0_design_freeze_complete
```

停止：

```text
terminal_decision = phase_n0_stopped_env_core_change_required
terminal_decision = phase_n0_stopped_obstacle_id_alignment_failed
terminal_decision = phase_n0_stopped_history_future_label_failed
terminal_decision = phase_n0_stopped_required_fields_missing
terminal_decision = phase_n0_stopped_trace_schema_insufficient
terminal_decision = phase_n0_stopped_spec_conflict
terminal_decision = phase_n0_stopped_watcher_failed
```

---

## 16. Phase N0 完成后进入 N1 的条件

只有当：

```text
PHASE_N0_DESIGN_FREEZE_COMPLETE.flag
```

存在，且 report 明确写出：

```text
Phase N0 complete.
Gψ-HeadA design and dataflow are ready for Phase N1 dataset construction.
```

才允许进入 N1。

N1 才开始正式构建：

```text
data/gpsi_head_a_v1/
  train.*
  val.*
  test.*
  schema.json
```

Phase N0 不构建正式训练数据集。
