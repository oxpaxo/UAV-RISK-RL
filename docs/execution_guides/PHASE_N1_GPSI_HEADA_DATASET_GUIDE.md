# Phase N1 指南：Gψ-HeadA 监督数据集构建

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：DynamicObstacleFlowEnv / EnvV2  
> 新主线：Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield  
> 阶段名称：Phase N1 - Gψ-HeadA Dataset Construction  
> 阶段性质：构建监督数据集；不训练 Gψ；不训练 PPO；不实现 shield；不修改 EnvV2-core。  
> 前置条件：Phase N0 已完成，存在 `results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag`。

---

## 0. 背景

Phase N0 已确认当前数据链路可进入 N1：

```text
1. EnvV2 info 暴露 active obstacle ids、world positions、world velocities、motion modes、threat classes、planned CPA/TTC、UAV state、dt、active obstacle count。
2. Phase A/B long obstacle tables 保留 obstacle slot、obstacle id、active flag、world position、world velocity、distance、closing、planned CPA/TTC、threat class、motion mode、risk value。
3. history / future label dry-run 可构造。
4. Head A target 固定为 world-frame obstacle residual：
   Δ_i(τ)=p_i_world(t+τ)-[p_i_world(t)+τv_i_world(t)]
5. σ² 不是 direct label；Gψ 输出 log σ̂²，通过 Gaussian NLL 学习 heteroscedastic uncertainty。
6. N3 第一版默认 freeze Gψ，只训练 PPO。
7. N4 必须做 λ_uncertainty sweep。
```

Phase N1 目标是正式生成 `Gψ-HeadA` 离线监督训练数据集，为 Phase N2 的 Head A offline pilot 做准备。

---

## 1. Phase N1 总目标

构建：

```text
data/gpsi_head_a_v1/
  train.npz
  val.npz
  test.npz
  schema.json
  dataset_manifest.json
  stats/*.csv
```

每条样本对应一个：

```text
(episode_id, step, obstacle_id)
```

输入包括：

```text
ego_t
current obs_i
obstacle history_i[t-H:t]
history_valid_mask
current obstacle id / slot / active
motion mode / threat class / planned CPA/TTC / distance / closing / risk_value
```

标签包括：

```text
delta_label_i(τ)=p_i_world(t+τ)-[p_i_world(t)+τv_i_world(t)]
τ ∈ {1s, 2s, 4s}
future_valid_mask_i(τ)
```

N1 不生成 σ² 监督标签。σ² 是后续 N2 模型输出的 diagonal log variance，通过 Gaussian NLL 学习。

---

## 2. 必须落实的 schema 更新：支持后续更充分使用 σ²

N1 虽然不实现 shield，但 schema 必须支持后续 shield 升级。

### 2.1 Head A 输出维度

后续 Head A 输出必须支持：

```text
delta_hat:  [num_horizons, state_dim]
logvar_hat: [num_horizons, state_dim]
```

其中：

```text
num_horizons = 3，对应 [1.0, 2.0, 4.0] 秒
state_dim 建议保留 xyz；后续 shield 至少使用 xy diagonal covariance
```

### 2.2 后续 shield 版本预留

schema / spec 中必须预留以下后续 shield 版本：

```text
V0 fixed-margin shield
V1 scalar σ²-margin shield
V2 directional σ²-margin shield
V3 predicted-trajectory directional σ²-tube shield
V4 V3 + uncertainty-aware candidate scoring
```

N1 必须 patch 或确认 `configs/gpsi_head_a_spec.yaml` 支持：

```yaml
uncertainty:
  type: diagonal_logvar
  dimensions: [x, y, z]
  sigma2_direct_label: false
  future_shield_usage:
    - scalar_margin
    - directional_margin
    - trajectory_tube
    - candidate_scoring
```

---

## 3. 禁止事项

Phase N1 禁止：

```text
1. 修改 EnvV2-core；
2. 训练 Gψ；
3. 训练 PPO；
4. 实现或评估 Safety Shield；
5. 把 σ² 写成 direct supervised label；
6. 按 row 随机划分 train/val/test；
7. 把不同 obstacle id 因 slot 相同而拼接成一条 history；
8. 把 future trajectory 泄漏到 inference input；
9. 回到 learned R(s,a)、candidate velocity risk map 或 5-head Gψ。
```

允许：

```text
1. 新增 dataset builder；
2. 新增 inspect 脚本；
3. patch / refine gpsi_head_a_spec.yaml；
4. 新增 schema、manifest、stats、plots、report、watcher、flags。
```

---

## 4. 推荐新增文件

```text
scripts/build_gpsi_heada_dataset.py
scripts/inspect_gpsi_heada_dataset.py
scripts/watch_phase_n1_gpsi_dataset.sh

data/gpsi_head_a_v1/
  train.npz
  val.npz
  test.npz
  schema.json
  dataset_manifest.json
  stats/

results/env_v2_phase_n1_gpsi_dataset/
  PHASE_N1_GPSI_DATASET_REPORT.md
  PHASE_N1_GPSI_DATASET_COMPLETE.flag
  phase_n1_status.txt
  phase_n1_watcher.log
  tables/*.csv
  plots/*.png
  logs/*.log
```

---

## 5. 数据来源和 split

默认数据来源：

```text
scenario = train_flow_mixed
```

推荐规模：

```text
300 episodes 起步；如果资源紧张，最低不要少于 100 episodes。
```

split 必须按 episode / episode_seed 划分：

```text
train: 70%
val:   15%
test:  15%
```

必须输出 `leakage_check.csv`，确认同一个 episode_id / episode_seed 不跨 split。

---

## 6. 样本字段

建议 `.npz` 包含：

```text
ego_current: [N, ego_dim]
obs_current: [N, 12]

history_pos_world: [N, H, 3]
history_vel_world: [N, H, 3]
history_rel_pos:   [N, H, 3]
history_rel_vel:   [N, H, 3]
history_valid_mask:[N, H]

delta_label_world: [N, T, 3]
future_valid_mask: [N, T]
future_times:      [T]

motion_mode_id: [N]
threat_class_id:[N]
obstacle_id:    [N]
obstacle_slot:  [N]
episode_id:     [N]
episode_seed:   [N]
step:           [N]
time:           [N]

distance:    [N]
closing:     [N]
planned_cpa: [N]
planned_ttc: [N]
risk_value:  [N]
```

如果保存 `future_pos_world` 或 `constant_velocity_pos_world`，必须在 schema 中标注：

```text
inspection / label-only，不允许作为 Gψ inference input。
```

---

## 7. label 构造规则

### 7.1 horizon offset

从 env info 读取 `dt`，或检查默认 `dt=0.2`：

```text
tau_steps = round(tau / dt)
1s = 5 steps
2s = 10 steps
4s = 20 steps
```

### 7.2 future valid

某 horizon label 仅当以下条件同时满足才有效：

```text
same episode
same obstacle_id
obstacle active at t
same obstacle_id active at t+τ
episode not ended before t+τ
replacement 没有打断该 obstacle identity
```

否则：

```text
future_valid_mask[τ] = 0
delta_label[τ] = 0 or NaN
loss 必须依赖 future_valid_mask
```

### 7.3 history valid

history 只使用同一 episode 内同一 obstacle_id 的过去状态。

```text
早期历史不足：left padding + history_valid_mask=0
slot 被 replacement 复用：不得接旧 obstacle history
```

### 7.4 坐标系

label 固定采用 world-frame obstacle residual：

```text
Δ_i(τ)=p_i_world(t+τ)-[p_i_world(t)+τv_i_world(t)]
```

理由：

```text
避免 future UAV motion 污染 obstacle dynamics label。
```

---

## 8. schema.json 要求

`data/gpsi_head_a_v1/schema.json` 必须至少包含：

```json
{
  "version": "gpsi_head_a_v1",
  "target": {
    "type": "world_frame_residual_to_constant_velocity",
    "formula": "delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]",
    "future_times_sec": [1.0, 2.0, 4.0],
    "sigma2_direct_label": false
  },
  "uncertainty_output_planned": {
    "type": "diagonal_logvar",
    "dimensions": ["x", "y", "z"],
    "used_by_future_shield": [
      "scalar_margin",
      "directional_margin",
      "trajectory_tube",
      "candidate_scoring"
    ]
  },
  "history": {
    "history_steps": 20,
    "padding": "left_pad_invalid",
    "valid_mask": true,
    "identity_key": ["episode_id", "obstacle_id"]
  },
  "split": {
    "type": "episode_level",
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
  }
}
```

---

## 9. inspect 脚本要求

`inspect_gpsi_heada_dataset.py` 必须输出：

```text
phase_n1_split_summary.csv
phase_n1_label_validity_by_horizon.csv
phase_n1_residual_stats_by_motion_mode.csv
phase_n1_residual_stats_by_horizon.csv
phase_n1_history_validity_stats.csv
phase_n1_replacement_boundary_stats.csv
phase_n1_leakage_check.csv
phase_n1_coordinate_frame_check.csv
phase_n1_sample_rows.csv
phase_n1_schema_check.csv
```

并生成 plots：

```text
residual_norm_by_motion_mode_1s.png
residual_norm_by_motion_mode_2s.png
residual_norm_by_motion_mode_4s.png
residual_norm_by_horizon.png
valid_label_rate_by_horizon.png
history_valid_length_distribution.png
```

必须检查：

```text
linear mode Δ≈0 sanity
nonlinear / stochastic mode residual 是否大于 linear
每个 split 非空
每个 horizon 有有效 label
episode-level leakage = 0
```

---

## 10. 命令清单

Codex 应按实际 repo 调整命令，并记录到 report。

### 10.1 编译检查

```bash
python -m py_compile scripts/build_gpsi_heada_dataset.py
python -m py_compile scripts/inspect_gpsi_heada_dataset.py
bash -n scripts/watch_phase_n1_gpsi_dataset.sh
chmod +x scripts/watch_phase_n1_gpsi_dataset.sh
```

### 10.2 构建数据集

```bash
python scripts/build_gpsi_heada_dataset.py   --out-dir data/gpsi_head_a_v1   --result-dir results/env_v2_phase_n1_gpsi_dataset   --scenario train_flow_mixed   --num-episodes 300   --eval-seed 2000   --history-steps 20   --future-times 1.0 2.0 4.0   --split 0.70 0.15 0.15   --format npz   --write-schema
```

### 10.3 inspect 数据集

```bash
python scripts/inspect_gpsi_heada_dataset.py   --data-dir data/gpsi_head_a_v1   --out-dir results/env_v2_phase_n1_gpsi_dataset
```

### 10.4 启动 watcher

```bash
bash scripts/watch_phase_n1_gpsi_dataset.sh
```

---

## 11. 报告要求

必须输出：

```text
results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_REPORT.md
```

报告至少包含：

```text
1. 背景与 N1 目标；
2. Phase N0 依赖检查；
3. spec refinement：diagonal logvar + future shield V0/V1/V2/V3/V4；
4. 数据来源与 episode 数量；
5. split 规则；
6. 样本定义；
7. 输入字段；
8. label 公式；
9. σ² 不是 direct label 的声明；
10. history / future validity 规则；
11. replacement / obstacle id 处理；
12. 坐标系说明；
13. dataset 文件路径；
14. schema.json 摘要；
15. split summary；
16. label validity by horizon；
17. residual stats by motion mode；
18. linear Δ≈0 sanity；
19. leakage check；
20. plots 路径；
21. risks / warnings；
22. 是否可以进入 Phase N2；
23. terminal_decision。
```

---

## 12. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N1_GPSI_DATASET_COMPLETE.flag
```

必须满足：

```text
1. Phase N0 complete flag 存在；
2. EnvV2-core 未被修改；
3. gpsi_head_a_spec.yaml 支持 diagonal logvar / future shield usage；
4. build / inspect 脚本存在并可运行；
5. train / val / test 文件存在且非空；
6. schema.json 和 dataset_manifest.json 存在；
7. train/val/test 按 episode split，无 leakage；
8. 每个 horizon 有有效 label；
9. history_valid_mask 和 future_valid_mask 存在；
10. obstacle_id alignment 规则被执行；
11. replacement 后 slot 复用没有导致 id 拼接；
12. linear mode Δ sanity 通过或有合理解释；
13. inspect tables 和 plots 生成；
14. report 生成；
15. watcher log 与 status 文件存在；
16. 无未解释 Python exception、empty dataset、schema mismatch。
```

建议但非强制：

```text
每个 split / horizon valid samples >= 1000；
nonlinear modes 的 Δ norm 大于 linear；
每种 motion mode 有足够样本。
```

---

## 13. 停止条件

如出现以下问题，必须生成对应 stop flag、partial report 和 log。

```text
PHASE_N1_STOP_PHASE_N0_MISSING.flag
PHASE_N1_STOP_ENV_CORE_CHANGE_REQUIRED.flag
PHASE_N1_STOP_DATASET_BUILD_FAILED.flag
PHASE_N1_STOP_LABEL_VALIDITY_FAILED.flag
PHASE_N1_STOP_ID_ALIGNMENT_FAILED.flag
PHASE_N1_STOP_DATA_LEAKAGE_FAILED.flag
PHASE_N1_STOP_INSUFFICIENT_DATA.flag
PHASE_N1_STOP_SCHEMA_MISMATCH.flag
PHASE_N1_STOP_WATCHER_FAILED.flag
```

触发示例：

```text
找不到 N0 complete flag；
必须改 EnvV2-core 才能构建数据集；
rollout / 写文件失败；
future_valid_mask 基本为空；
history/future 不能保证同一 obstacle_id；
同一 episode 出现在多个 split；
train/val/test 任一为空；
schema 与实际数据字段不一致；
watcher 失败。
```

---

## 14. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n1_gpsi_dataset.sh
```

watcher 必须：

```text
1. 检查 Phase N0 complete flag；
2. 启动 dataset builder；
3. 启动 inspect 脚本；
4. 持续轮询 complete / stop flag；
5. 持续输出当前状态；
6. 只有 complete flag 或 stop flag 出现才退出；
7. 不允许中途“等待用户确认”；
8. 不允许因为暂无新日志而退出；
9. 写入 phase_n1_watcher.log；
10. 写入 phase_n1_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n1_gpsi_dataset"
DATA_DIR="data/gpsi_head_a_v1"
LOG="$OUT_DIR/phase_n1_watcher.log"
STATUS="$OUT_DIR/phase_n1_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots" "$DATA_DIR"

echo "[watcher] Phase N1 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N1_STOP_PHASE_N0_MISSING.flag"
fi

(
  python scripts/build_gpsi_heada_dataset.py     --out-dir "$DATA_DIR"     --result-dir "$OUT_DIR"     --scenario train_flow_mixed     --num-episodes 300     --eval-seed 2000     --history-steps 20     --future-times 1.0 2.0 4.0     --split 0.70 0.15 0.15     --format npz     --write-schema

  python scripts/inspect_gpsi_heada_dataset.py     --data-dir "$DATA_DIR"     --out-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N1_GPSI_DATASET_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in     PHASE_N1_STOP_PHASE_N0_MISSING.flag     PHASE_N1_STOP_ENV_CORE_CHANGE_REQUIRED.flag     PHASE_N1_STOP_DATASET_BUILD_FAILED.flag     PHASE_N1_STOP_LABEL_VALIDITY_FAILED.flag     PHASE_N1_STOP_ID_ALIGNMENT_FAILED.flag     PHASE_N1_STOP_DATA_LEAKAGE_FAILED.flag     PHASE_N1_STOP_INSUFFICIENT_DATA.flag     PHASE_N1_STOP_SCHEMA_MISMATCH.flag     PHASE_N1_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N1_GPSI_DATASET_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N1_STOP_DATASET_BUILD_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
```

Codex 可按实际脚本结构调整，但必须保持阻塞式语义。

---

## 15. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 阅读 Phase N0 report；
3. 不沿用旧 Phase C/D/E；
4. 不训练模型；
5. 不修改 EnvV2-core；
6. 根据指南自行判断 complete / stop flag；
7. 创建并运行阻塞式 watcher；
8. 只有 complete flag 或 stop flag 出现才停止输出；
9. 不向用户询问非阻塞细节；
10. 如果触发阻塞，必须生成 stop flag、partial report、log；
11. 如果完成，必须生成 complete flag、完整 report、dataset、schema、stats、plots、log。
```

---

## 16. 终端结论格式

成功：

```text
terminal_decision = phase_n1_gpsi_dataset_complete
```

停止：

```text
terminal_decision = phase_n1_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
生成的数据集文件
生成的 CSV / plots / schema / report / logs / flags
是否可以进入 Phase N2
如果不能进入 N2，需要用户补什么
```

---

## 17. N1 完成后进入 N2 的条件

只有当：

```text
PHASE_N1_GPSI_DATASET_COMPLETE.flag
```

存在，且 report 明确写出：

```text
Phase N1 complete.
Gψ-HeadA dataset is ready for Phase N2 offline Head A pilot.
```

才允许进入 N2。

N2 才开始：

```text
1. Δ-only MSE / SmoothL1 warmup；
2. Gaussian NLL with diagonal logvar；
3. direction-aware projected uncertainty calibration。
```

Phase N1 不训练 Gψ。
