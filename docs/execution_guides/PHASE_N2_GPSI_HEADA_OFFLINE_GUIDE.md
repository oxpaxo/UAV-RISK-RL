# Phase N2 指南：Gψ-HeadA 离线可学习性验证

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 新主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N2 - Gψ-HeadA Offline Pilot`  
> 阶段性质：离线监督训练与校准评估；不接 PPO；不实现 shield；不训练 RL；不修改 EnvV2-core。  
> 前置条件：Phase N1 已完成，存在 `results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag`，且 `data/gpsi_head_a_v1/` 数据集可用。

---

## 0. 背景与当前状态

Phase N1 已完成，正式构建了 `Gψ-HeadA` 监督数据集：

```text
train.npz: 228,957 samples
val.npz:    49,599 samples
test.npz:   48,096 samples
```

数据集采用 episode-level split，无 row-level leakage。有效标签数量充足：

```text
train:
  1s: 209,016
  2s: 192,966
  4s: 161,489

val:
  1s: 45,257
  2s: 41,720
  4s: 34,748

test:
  1s: 43,908
  2s: 40,528
  4s: 33,889
```

N1 已确认：

```text
1. label = world-frame residual:
   Δ_i(τ)=p_i_world(t+τ)-[p_i_world(t)+τv_i_world(t)]

2. σ² 不是 direct label；
   后续 Gψ 输出 diagonal logvar，并通过 Gaussian NLL 学习。

3. schema 已预留：
   scalar σ²-margin
   directional σ²-margin
   predicted-trajectory uncertainty tube
   uncertainty-aware candidate scoring
```

Phase N2 的目标是验证 Head A 是否离线可学。只有 N2 通过，才允许进入 N3：`Gψ frozen + PPO no shield`。

---

## 1. Phase N2 总目标

Phase N2 必须回答：

```text
Gψ-HeadA 能否从 obstacle history / current profile 中稳定预测未来 residual Δ̂_i(τ)，并学习到有校准意义的 diagonal uncertainty log σ̂_i²(τ)？
```

N2 分三步：

```text
N2.0: Dataset loader / normalization / baseline sanity
N2.1: Δ-only MSE / SmoothL1 warmup
N2.2: diagonal Gaussian NLL 训练 log σ̂²
N2.3: motion-mode-wise + direction-aware projected uncertainty calibration
```

---

## 2. 明确禁止事项

Phase N2 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止重新构建正式数据集，除非 N1 数据集读取失败；
3. 禁止训练 PPO；
4. 禁止接入 RL / SB3 training；
5. 禁止实现或评估 Safety Shield；
6. 禁止把 σ² 当作 direct supervised label；
7. 禁止把 future_pos_world 喂给 Gψ input；
8. 禁止使用 val/test 统计量做 normalization；
9. 禁止回到 learned R(s,a)、candidate velocity risk map、5-head Gψ；
10. 禁止因为 Δ-only 没学好就强行进入 NLL；
11. 禁止 NLL 爆炸或 logvar collapse 后仍生成 complete flag。
```

允许：

```text
1. 新增 PyTorch dataset / dataloader；
2. 新增 Gψ-HeadA 模型；
3. 新增训练脚本；
4. 新增评估 / 校准脚本；
5. 新增 plots / tables / report / watcher；
6. 用 train split 统计 normalization；
7. 做小规模 architecture / loss sanity，但必须记录清楚。
```

---

## 3. 推荐新增文件与目录

```text
models/
  gpsi_head_a.py

scripts/
  train_gpsi_heada.py
  eval_gpsi_heada.py
  watch_phase_n2_gpsi_heada_offline.sh

configs/
  gpsi_heada_train_delta_only.yaml
  gpsi_heada_train_nll.yaml

work_dirs/
  gpsi_heada_v1_delta_only/
    best.pth
    last.pth
    train_log.csv
    config_resolved.yaml

  gpsi_heada_v1_nll/
    best.pth
    last.pth
    train_log.csv
    config_resolved.yaml

results/
  env_v2_phase_n2_gpsi_heada_offline/
    PHASE_N2_HEADA_OFFLINE_REPORT.md
    PHASE_N2_HEADA_OFFLINE_COMPLETE.flag
    phase_n2_status.txt
    phase_n2_watcher.log

    tables/
      phase_n2_dataset_loader_check.csv
      phase_n2_constant_velocity_baseline.csv
      phase_n2_delta_only_metrics.csv
      phase_n2_nll_metrics.csv
      phase_n2_per_horizon_metrics.csv
      phase_n2_per_motion_mode_metrics.csv
      phase_n2_per_axis_logvar_stats.csv
      phase_n2_projected_uncertainty_calibration.csv
      phase_n2_calibration_bins.csv
      phase_n2_command_manifest.csv
      phase_n2_schema_check.csv

    plots/
      delta_loss_curve.png
      nll_loss_curve.png
      per_horizon_mse_bar.png
      per_motion_mode_error_bar.png
      logvar_by_motion_mode.png
      projected_uncertainty_reliability.png
      zscore_histogram.png
      predicted_vs_error_scatter.png

    logs/
      phase_n2_train_delta_only.log
      phase_n2_train_nll.log
      phase_n2_eval.log
```

---

## 4. 模型设计要求

### 4.1 输入

模型输入必须来自 N1 数据集的 inference-available 字段，不允许使用 future label 字段。

推荐输入：

```text
ego_current
obs_current
history_rel_pos
history_rel_vel
history_valid_mask
```

可选输入：

```text
history_pos_world
history_vel_world
```

默认不要把 `motion_mode_id` 作为训练输入。motion mode 应主要用于评估分组，避免 Gψ 过度依赖显式模式标签。

禁止作为模型输入：

```text
future_pos_world
constant_velocity_pos_world
delta_label_world
future_valid_mask
```

### 4.2 网络结构

第一版建议轻量 per-obstacle temporal encoder：

```text
history encoder:
  GRU 或 temporal MLP

current encoder:
  MLP(obs_current + ego_current)

fusion:
  concat(history_embedding, current_embedding)
  → MLP
  → z_i

Head A:
  z_i → delta_hat [T,D]
  z_i → logvar_hat [T,D]
```

推荐默认：

```text
history_steps = 20
future_times = [1.0, 2.0, 4.0]
state_dim = 3
z_dim = 64
hidden_dim = 128
encoder = GRU first, temporal MLP optional
activation = Tanh or ReLU, but must be recorded
```

### 4.3 输出

```text
delta_hat:
  shape [batch, T, D]

logvar_hat:
  shape [batch, T, D]
```

其中：

```text
T = 3
D = 3
```

后续 shield 至少会使用 xy diagonal covariance：

```text
Σ_i(τ) = diag(σ_x²(τ), σ_y²(τ))
```

因此 N2 必须训练 per-axis diagonal logvar，而不是单个 scalar logvar。

---

## 5. Loss 设计

### 5.1 N2.1 Δ-only warmup

先训练 delta prediction，不使用 logvar loss。

推荐：

```text
L_delta = masked SmoothL1(delta_hat, delta_label)
```

或：

```text
L_delta = masked MSE(delta_hat, delta_label)
```

mask：

```text
future_valid_mask [batch, T]
valid_mask = future_valid_mask[..., None]
```

只对 valid horizon 计算 loss。

### 5.2 N2.2 Gaussian NLL

从 delta-only checkpoint 初始化，加入 diagonal logvar：

```text
L_NLL = 0.5 * [ exp(-logvar) * (delta - delta_hat)^2 + logvar ]
```

同样使用 future_valid_mask。

数值稳定要求：

```text
logvar_clamp = [-5, 3]
gradient clipping 建议开启
NLL 阶段 lr 建议 <= delta-only lr
```

必须监控：

```text
NLL
delta MSE / SmoothL1
mean logvar
min / max logvar
per-axis logvar
per-horizon logvar
```

禁止：

```text
1. 用手工 σ² label 做 supervised MSE；
2. 无 mask 地把 invalid horizon 纳入 loss；
3. NLL 阶段只优化 logvar 不保持 delta 质量；
4. logvar 不 clamp；
5. logvar collapse 后仍判定完成。
```

---

## 6. Baseline 与 sanity

### 6.1 Constant-velocity baseline

因为 label 是 residual to constant velocity，所以：

```text
constant velocity baseline predicts Δ̂=0
```

必须计算：

```text
MSE_zero_delta
SmoothL1_zero_delta
per-horizon zero baseline
per-motion-mode zero baseline
```

Gψ 至少应在 nonlinear / stochastic modes 上优于 zero baseline。linear mode 下 zero baseline 极强是正常现象。

### 6.2 Motion-mode residual sanity

必须按 motion mode 统计：

```text
linear
accel_decel
ar1_velocity
sinusoidal_lateral
crossing_or_sudden_threat
```

检查：

```text
linear Δ≈0
nonlinear residual larger
Gψ 是否主要改善 nonlinear modes
```

### 6.3 Horizon sanity

必须按 horizon 统计：

```text
1s / 2s / 4s
```

一般预期：

```text
4s error > 2s error > 1s error
```

但 Gψ 相对 zero baseline 的 improvement 也应分 horizon 报告。

---

## 7. 方向性 uncertainty calibration

这是 N2 的新增重点。后续 shield 不只用 scalar σ²，而要用：

```text
margin_i(v,τ) = base_margin + k * sqrt(n^T Σ_i(τ) n)
```

因此 N2 必须评估 projected uncertainty 是否有意义。

### 7.1 per-axis logvar 统计

输出：

```text
σ_x²(τ), σ_y²(τ), σ_z²(τ)
```

按以下维度统计：

```text
split
horizon
motion_mode
axis
```

统计：

```text
mean
median
p10
p90
min
max
```

### 7.2 projected uncertainty 评估

定义 residual error：

```text
e_i(τ) = Δ_i(τ) - Δ̂_i(τ)
```

对于方向 n：

```text
projected_error = n^T e_i(τ)
projected_var = n^T Σ_i(τ) n
projected_std = sqrt(projected_var)
z = projected_error / (projected_std + eps)
```

方向 n 至少包括：

```text
1. x-axis: [1,0]
2. y-axis: [0,1]
3. current radial direction from UAV to obstacle, normalized in xy
4. relative velocity direction in xy if norm > eps
5. residual error direction for diagnostic only
```

注意：residual error direction 不能作为 shield inference 的真实方向，只能作为 upper-bound diagnostic。

输出 calibration metrics：

```text
mean |z|
std z
percentage |z| < 1
percentage |z| < 2
NLL projected
correlation(projected_std, |projected_error|)
calibration bins by predicted std
```

理想但非硬性：

```text
|z|<1 接近 68%
|z|<2 接近 95%
```

第一版不要求完美，但如果完全不相关，需要报告风险。

### 7.3 scalar vs diagonal diagnostic

从 diagonal logvar 派生 scalar：

```text
scalar_std_trace = sqrt(σ_x² + σ_y²)
scalar_std_max = max(sqrt(σ_x²), sqrt(σ_y²))
```

比较：

```text
directional projected std
vs scalar std
```

报告中必须说明：

```text
direction-aware shield 是否有基础；
如果 diagonal uncertainty 不稳定，N4 directional shield 可能风险较高。
```

---

## 8. 训练规模建议

N2 是 pilot，不需要三种子，但必须足够稳定。

推荐：

```text
seed = 0
batch_size = 512 或 1024
epochs_delta_only = 20-50
epochs_nll = 20-50
optimizer = Adam / AdamW
lr_delta_only = 1e-3 或 3e-4
lr_nll = 3e-4 或 1e-4
grad_clip = 1.0
```

Early stopping：

```text
delta-only: monitor val masked delta MSE
NLL: monitor val NLL + no severe delta degradation
```

---

## 9. 评估指标

必须输出：

```text
masked MSE
masked SmoothL1
Gaussian NLL
per-horizon MSE / NLL
per-motion-mode MSE / NLL
per-axis MSE / logvar stats
zero-baseline comparison
relative improvement over zero baseline
projected uncertainty calibration metrics
valid sample counts used in evaluation
```

核心表：

```text
phase_n2_delta_only_metrics.csv
phase_n2_nll_metrics.csv
phase_n2_per_horizon_metrics.csv
phase_n2_per_motion_mode_metrics.csv
phase_n2_per_axis_logvar_stats.csv
phase_n2_projected_uncertainty_calibration.csv
phase_n2_calibration_bins.csv
```

---

## 10. 命令清单

Codex 应根据实际 repo 调整命令，并在 report 中记录实际命令。

### 10.1 编译检查

```bash
python -m py_compile models/gpsi_head_a.py
python -m py_compile scripts/train_gpsi_heada.py
python -m py_compile scripts/eval_gpsi_heada.py
bash -n scripts/watch_phase_n2_gpsi_heada_offline.sh
chmod +x scripts/watch_phase_n2_gpsi_heada_offline.sh
```

### 10.2 Delta-only 训练

```bash
python scripts/train_gpsi_heada.py \
  --data-dir data/gpsi_head_a_v1 \
  --out-dir work_dirs/gpsi_heada_v1_delta_only \
  --config configs/gpsi_heada_train_delta_only.yaml \
  --loss delta_smoothl1 \
  --epochs 30 \
  --batch-size 512 \
  --seed 0
```

### 10.3 NLL 训练

```bash
python scripts/train_gpsi_heada.py \
  --data-dir data/gpsi_head_a_v1 \
  --init work_dirs/gpsi_heada_v1_delta_only/best.pth \
  --out-dir work_dirs/gpsi_heada_v1_nll \
  --config configs/gpsi_heada_train_nll.yaml \
  --loss gaussian_nll \
  --logvar-clamp -5 3 \
  --epochs 30 \
  --batch-size 512 \
  --seed 0
```

### 10.4 评估

```bash
python scripts/eval_gpsi_heada.py \
  --data-dir data/gpsi_head_a_v1 \
  --delta-checkpoint work_dirs/gpsi_heada_v1_delta_only/best.pth \
  --nll-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out-dir results/env_v2_phase_n2_gpsi_heada_offline
```

### 10.5 watcher

```bash
bash scripts/watch_phase_n2_gpsi_heada_offline.sh
```

---

## 11. 报告要求

必须输出：

```text
results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_REPORT.md
```

报告至少包含：

```text
1. 背景与 N2 目标；
2. N1 dependency check；
3. 数据集摘要；
4. 模型结构；
5. 输入字段说明；
6. 确认 future label 未进入 input；
7. normalization 规则；
8. Δ-only 训练设置与结果；
9. Gaussian NLL 训练设置与结果；
10. zero baseline 对比；
11. per-horizon 指标；
12. per-motion-mode 指标；
13. per-axis logvar stats；
14. projected uncertainty calibration；
15. scalar vs directional uncertainty diagnostic；
16. plots 路径；
17. risks / warnings；
18. 是否可以进入 N3；
19. terminal_decision。
```

报告必须明确区分：

```text
experiment-supported facts
reasonable inferences
risks / unresolved issues
```

---

## 12. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N2_HEADA_OFFLINE_COMPLETE.flag
```

必须满足：

```text
1. Phase N1 complete flag 存在；
2. train/val/test 数据集可读取；
3. schema 与数据字段一致；
4. 训练脚本与评估脚本存在并可运行；
5. Δ-only 模型训练完成；
6. Δ-only val loss 相对 zero baseline 在非-linear modes 有改善；
7. NLL 模型训练完成；
8. NLL 没有 NaN / inf；
9. logvar 没有全塌成常数；
10. logvar clamp 生效；
11. per-horizon / per-motion-mode 指标生成；
12. projected uncertainty calibration 表生成；
13. plots 生成；
14. best checkpoint 存在；
15. report 生成；
16. watcher log 与 status 文件存在。
```

建议通过条件；若不满足必须写 warning：

```text
1. Gψ 在 all nonlinear modes 上优于 zero baseline；
2. 4s horizon 虽然更难，但不是完全失控；
3. projected_std 与 |projected_error| 有正相关；
4. |z|<1 与 |z|<2 覆盖率不要严重异常；
5. linear mode delta prediction 不被模型弄坏。
```

---

## 13. 停止条件

如出现以下问题，必须生成对应 stop flag、partial report 和 log：

```text
PHASE_N2_STOP_PHASE_N1_MISSING.flag
PHASE_N2_STOP_DATASET_READ_FAILED.flag
PHASE_N2_STOP_SCHEMA_MISMATCH.flag
PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag
PHASE_N2_STOP_DELTA_NOT_LEARNABLE.flag
PHASE_N2_STOP_NLL_TRAIN_FAILED.flag
PHASE_N2_STOP_LOGVAR_COLLAPSE.flag
PHASE_N2_STOP_CALIBRATION_FAILED.flag
PHASE_N2_STOP_WATCHER_FAILED.flag
```

---

## 14. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n2_gpsi_heada_offline.sh
```

watcher 必须：

```text
1. 检查 Phase N1 complete flag；
2. 启动 delta-only 训练；
3. 启动 NLL 训练；
4. 启动 eval 脚本；
5. 持续轮询 complete / stop flag；
6. 持续输出当前状态；
7. 只有 complete flag 或 stop flag 出现才退出；
8. 不允许中途“等待用户确认”；
9. 不允许因为暂无新日志而退出；
10. 写入 phase_n2_watcher.log；
11. 写入 phase_n2_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n2_gpsi_heada_offline"
LOG="$OUT_DIR/phase_n2_watcher.log"
STATUS="$OUT_DIR/phase_n2_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N2 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N2_STOP_PHASE_N1_MISSING.flag"
fi

(
  python scripts/train_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --out-dir work_dirs/gpsi_heada_v1_delta_only \
    --config configs/gpsi_heada_train_delta_only.yaml \
    --loss delta_smoothl1 \
    --epochs 30 \
    --batch-size 512 \
    --seed 0

  python scripts/train_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --init work_dirs/gpsi_heada_v1_delta_only/best.pth \
    --out-dir work_dirs/gpsi_heada_v1_nll \
    --config configs/gpsi_heada_train_nll.yaml \
    --loss gaussian_nll \
    --logvar-clamp -5 3 \
    --epochs 30 \
    --batch-size 512 \
    --seed 0

  python scripts/eval_gpsi_heada.py \
    --data-dir data/gpsi_head_a_v1 \
    --delta-checkpoint work_dirs/gpsi_heada_v1_delta_only/best.pth \
    --nll-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N2_STOP_PHASE_N1_MISSING.flag \
    PHASE_N2_STOP_DATASET_READ_FAILED.flag \
    PHASE_N2_STOP_SCHEMA_MISMATCH.flag \
    PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag \
    PHASE_N2_STOP_DELTA_NOT_LEARNABLE.flag \
    PHASE_N2_STOP_NLL_TRAIN_FAILED.flag \
    PHASE_N2_STOP_LOGVAR_COLLAPSE.flag \
    PHASE_N2_STOP_CALIBRATION_FAILED.flag \
    PHASE_N2_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag"
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
2. 阅读 Phase N1 report；
3. 不沿用旧 Phase C/D/E；
4. 不训练 PPO；
5. 不实现 shield；
6. 不修改 EnvV2-core；
7. 根据指南自行判断 complete / stop flag；
8. 创建并运行阻塞式 watcher；
9. 只有 complete flag 或 stop flag 出现才停止输出；
10. 不向用户询问非阻塞细节；
11. 如果触发阻塞，必须生成 stop flag、partial report、log；
12. 如果完成，必须生成 complete flag、完整 report、checkpoints、CSV、plots、log。
```

---

## 16. 终端结论格式

成功：

```text
terminal_decision = phase_n2_heada_offline_complete
```

停止：

```text
terminal_decision = phase_n2_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
生成的 checkpoints
生成的 CSV / plots / report / logs / flags
是否可以进入 Phase N3
如果不能进入 N3，需要用户补什么
```

---

## 17. N2 完成后进入 N3 的条件

只有当：

```text
PHASE_N2_HEADA_OFFLINE_COMPLETE.flag
```

存在，且 report 明确写出：

```text
Phase N2 complete.
Gψ-HeadA offline model is ready for Phase N3 frozen-Gψ PPO integration.
```

才允许进入 N3。

N3 才开始：

```text
Gψ frozen
PPO trainable
obs_i_aug = [obs_i, z_i, Δ̂_i, log σ̂_i²]
no shield
masked-attention PPO backbone
symmetric critic
```

Phase N2 不接 PPO。
