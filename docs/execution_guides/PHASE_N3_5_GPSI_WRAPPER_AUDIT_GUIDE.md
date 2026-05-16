# Phase N3.5 指南：Gψ Online Wrapper / Feature-Scale Audit

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3.5 - Gψ Online Wrapper & Feature-Scale Audit`  
> 阶段性质：工程审计 / bug 定位 / 特征尺度校验；不训练 PPO；不实现 shield；不改 EnvV2-core。  
> 前置条件：Phase N3 已完成，但 N3 结果出现严重异常，需要先审计 online Gψ 接入。  

---

## 0. 背景与问题定位

Phase N2 离线结果表明：

```text
Zero residual test MSE: 0.466866
Delta-only test MSE:   0.099426
NLL model test MSE:    0.096426
NLL test Gaussian NLL: -1.793691
Projected uncertainty corr: about 0.6042
```

这说明 Gψ-HeadA 在离线监督任务上是可学习的，残差尺度应为米级 / 小量级。

但 Phase N3 的 online PPO 接入结果异常：

```text
Gψ-PPO no-shield 1500k:
  success ≈ 0.267
  collision ≈ 0.733

attention_full_1500k:
  success ≈ 0.610
  collision ≈ 0.390
```

更严重的是 N3 diagnostics 里出现：

```text
mean_delta_norm_1s ≈ 1e4–3e4
mean_delta_norm_2s / 4s 更大
mean_logvar_xy_1s 全部贴到 -5
projected_std 近似恒定 sqrt(exp(-5)) ≈ 0.082
```

这和 N2 离线结果冲突。因此 N3 不能直接解释为“Gψ 方法失败”。更合理的当前判断是：

```text
N3 online Gψ wrapper / normalization / feature slicing / feature-scale pipeline 存在高风险。
```

Phase N3.5 的目标是在进入 N4 shield 之前，把 online Gψ 工程链路彻底查干净。

---

## 1. Phase N3.5 总目标

N3.5 必须回答：

```text
N3 中的 online Gψ 输出异常，到底来自：
1. wrapper / normalization / field-order / slicing bug；
2. online EnvV2 input distribution 与 N1/N2 train distribution 不一致；
3. z_i / delta_hat / logvar_hat 特征尺度未规范化，污染 PPO 输入；
4. diagnostics 统计错误；
5. 还是工程正确但方法本身不适合 PPO no-shield。
```

N3.5 不训练 PPO，不实现 shield，不修改 EnvV2-core。它只做：

```text
1. offline-online equivalence test；
2. online input distribution audit；
3. Gψ output scale audit；
4. PPO augmented feature block-scale audit；
5. wrapper / normalization / slicing bug 修复；
6. 修复后 smoke forward / short rollout 复检；
7. 给出是否需要重跑 N3 的明确结论。
```

---

## 2. 明确禁止事项

Phase N3.5 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止训练 PPO；
3. 禁止训练 / fine-tune Gψ；
4. 禁止实现 N4 safety shield；
5. 禁止引入 safety-cost PPO；
6. 禁止回到 learned R(s,a) / candidate velocity risk map / 5-head Gψ；
7. 禁止用错误的 N3 结果直接推进 N4；
8. 禁止在未完成 offline-online equivalence 前重跑正式 1.5M PPO；
9. 禁止只看 aggregate success/collision 而不检查 Gψ feature scale。
```

允许：

```text
1. 修复 Gψ online wrapper；
2. 修复 input normalization / field ordering / slicing；
3. 增加 feature-scale normalization；
4. 新增 diagnostic scripts；
5. 新增 short rollout / forward-only audit；
6. 新增 N3 rerun recommendation report；
7. 新增 watcher / flags / tables / plots。
```

---

## 3. 推荐新增 / 修改文件

建议新增：

```text
scripts/
  audit_gpsi_online_wrapper.py
  compare_gpsi_offline_online.py
  inspect_gpsi_augmented_features.py
  watch_phase_n3_5_gpsi_wrapper_audit.sh

results/
  env_v2_phase_n3_5_gpsi_wrapper_audit/
    PHASE_N3_5_GPSI_WRAPPER_AUDIT_REPORT.md
    PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag
    phase_n3_5_status.txt
    phase_n3_5_watcher.log

    tables/
      phase_n3_5_offline_online_equivalence.csv
      phase_n3_5_input_distribution_compare.csv
      phase_n3_5_output_scale_summary.csv
      phase_n3_5_aug_feature_block_stats.csv
      phase_n3_5_field_order_check.csv
      phase_n3_5_history_buffer_check.csv
      phase_n3_5_active_mask_check.csv
      phase_n3_5_normalization_check.csv
      phase_n3_5_slicing_check.csv
      phase_n3_5_short_rollout_output_summary.csv
      phase_n3_5_repair_actions.csv
      phase_n3_5_rerun_recommendation.csv
      phase_n3_5_command_manifest.csv

    plots/
      offline_vs_online_delta_scatter.png
      offline_vs_online_logvar_scatter.png
      online_delta_norm_distribution_before_after.png
      online_logvar_distribution_before_after.png
      aug_feature_block_scale.png
      input_distribution_shift.png
      z_norm_distribution.png

    logs/
      phase_n3_5_offline_online_compare.log
      phase_n3_5_online_audit.log
      phase_n3_5_feature_scale.log
```

如项目已有目录规范，可调整，但必须保证：

```text
1. 所有 N3.5 产物集中；
2. complete / stop flags 路径固定；
3. watcher 可检测状态；
4. report 能引用核心 CSV / plots / logs。
```

---

## 4. 核心检查 1：Offline-online equivalence test

这是 N3.5 最关键的检查。

### 4.1 测试目的

验证同一批 N1 val/test 样本，分别走：

```text
A. N2 eval path
B. N3 online wrapper / adapter path
```

是否得到一致输出。

### 4.2 输入样本

从 N1 数据集中抽样：

```text
split: val + test
samples: 1000–5000
覆盖：
  multiple motion modes
  multiple horizons
  full-history and partial-history
  different threat classes
```

### 4.3 必须比较

对同一输入样本，比较：

```text
normalized_ego_current
normalized_obs_current
normalized_history_rel_pos
normalized_history_rel_vel
history_valid_mask

z_i
delta_hat
logvar_hat
```

输出：

```text
max_abs_diff
mean_abs_diff
rmse_diff
corr
allclose_pass
```

### 4.4 通过标准

理想：

```text
delta_hat / logvar_hat / z_i 基本一致；
数值误差只来自 float tolerance。
```

建议判据：

```text
max_abs_diff(delta_hat) < 1e-4 或合理 tolerance
max_abs_diff(logvar_hat) < 1e-4 或合理 tolerance
history_valid_mask 完全一致
```

如果同一样本输出明显不同，应触发：

```text
PHASE_N3_5_STOP_OFFLINE_ONLINE_MISMATCH.flag
```

除非 Codex 已修复并重新通过 equivalence test。

---

## 5. 核心检查 2：Online EnvV2 input distribution audit

### 5.1 目的

确认 EnvV2 online wrapper 产生的 Gψ 输入与 N1/N2 train split 统计一致。

### 5.2 检查字段

```text
ego_current
obs_current
history_rel_pos
history_rel_vel
history_valid_mask
active_mask
obstacle_id
obstacle_slot
```

### 5.3 分布比较

对每个 input block 输出：

```text
mean
std
min
max
p01
p05
p50
p95
p99
nan_count
inf_count
```

并与 N1 train split 对比。

重点检查：

```text
1. obs_current 12维字段顺序是否一致；
2. history_rel_pos / history_rel_vel 单位是否一致；
3. history_valid_mask 是否被反转；
4. replacement 后是否 reset history；
5. inactive / padded obstacle 是否被送入 Gψ；
6. online relative position / velocity 是否与 N1 dataset 定义一致；
7. ego_current 维度和字段顺序是否一致。
```

### 5.4 输出表

```text
phase_n3_5_input_distribution_compare.csv
phase_n3_5_field_order_check.csv
phase_n3_5_history_buffer_check.csv
phase_n3_5_active_mask_check.csv
```

---

## 6. 核心检查 3：Gψ output scale audit

### 6.1 当前异常

N3 原始 diagnostics 显示：

```text
mean_delta_norm_1s: 1e4–3e4
logvar_xy_1s: all -5
projected_std: constant ≈ 0.082
```

N3.5 必须解释并修复这个异常。

### 6.2 合理范围

参考 N2 offline：

```text
delta_hat norm:
  应为 O(0–几米/十几米)，不是 O(1e4)

logvar_hat:
  不应全部 clamp 到 -5；
  应有 horizon / motion-mode / axis 结构。

projected_std:
  不应恒定；
  应随 motion mode / horizon / direction 变化。
```

### 6.3 输出表

```text
phase_n3_5_output_scale_summary.csv
```

字段至少包括：

```text
scenario
checkpoint_or_policy
motion_mode
threat_class
history_valid_ratio_bin
active_only
delta_norm_1s_mean/median/p95/max
delta_norm_2s_mean/median/p95/max
delta_norm_4s_mean/median/p95/max
logvar_xy_1s_mean/median/min/max
logvar_xy_2s_mean/median/min/max
logvar_xy_4s_mean/median/min/max
projected_std_radial_mean/std
projected_std_relvel_mean/std
z_norm_mean/median/p95/max
nan_count
inf_count
```

### 6.4 必须区分 before / after

如果 Codex 做了修复，必须输出 before/after 对比：

```text
before_fix
after_fix
```

并生成：

```text
online_delta_norm_distribution_before_after.png
online_logvar_distribution_before_after.png
```

---

## 7. 核心检查 4：PPO augmented feature block-scale audit

N3 使用：

```text
obs_i_aug = [obs_i(12), z_i(64), delta_hat_i(9), logvar_hat_i(9)]
```

N3.5 必须检查四个 block 的尺度。

### 7.1 检查 block

```text
obs_block
z_block
delta_block_before_scale
delta_block_after_scale
logvar_block
full_aug_obs
```

### 7.2 输出统计

每个 block 输出：

```text
mean
std
l2_norm_mean
l2_norm_median
l2_norm_p95
min
max
p01
p99
nan_count
inf_count
```

### 7.3 特别关注 z_i

N3 report 已说明：

```text
z_i is not additionally normalized in N3 v1.
```

N3.5 必须判断是否需要：

```text
1. z_i LayerNorm；
2. z_i train-set normalization；
3. tanh clamp；
4. 降维；
5. 暂时去掉 z_i，只用 delta_hat/logvar_hat；
6. 分块 normalization。
```

这不是必须立即训练，但必须给出工程建议。

---

## 8. 必须检查的具体 bug 点

N3.5 必须逐项检查并在 report 中写出 pass/fail。

### 8.1 Checkpoint load

```text
1. checkpoint path 是否为 work_dirs/gpsi_heada_v1_nll/best.pth；
2. model config 是否和 N2 一致；
3. state_dict key 是否完全匹配；
4. normalization stats 是否从 checkpoint 加载；
5. device / dtype 是否一致。
```

### 8.2 Input normalization

```text
1. 使用 train-split-only stats；
2. 没有重复 normalization；
3. 没有漏 normalization；
4. obs_current / ego_current / history blocks 各自用正确 stats；
5. online values 在 normalized 后没有极端 OOD。
```

### 8.3 Field order

```text
1. N1 obs_current 12维字段顺序；
2. EnvV2 online obs_i 12维字段顺序；
3. N3 wrapper 切片顺序；
4. ego_current 字段顺序；
5. history_rel_pos / history_rel_vel 维度顺序。
```

### 8.4 History buffer

```text
1. keyed by obstacle_id，而不是 slot；
2. replacement 后 reset / left pad；
3. early history padding 正确；
4. valid_history_mask 语义正确；
5. inactive obstacle 不污染 active history。
```

### 8.5 Output slicing

```text
1. z_i slice = correct 64 dims；
2. delta_hat slice = correct [T,D]；
3. logvar_hat slice = correct [T,D]；
4. flattened order 与 N2 一致；
5. PPO aug obs 拼接顺序与 schema 一致。
```

### 8.6 Diagnostics

```text
1. delta_norm 是否用 raw delta_hat 统计；
2. delta_scale 是否只用于 PPO input，不重复影响 diagnostics；
3. logvar 是否 clamp 前 / clamp 后分别可记录；
4. projected_std 是否根据 logvar 正确计算；
5. active-only 统计是否正确。
```

---

## 9. 修复策略

Codex 可以在 N3.5 内修复 wrapper/diagnostics，但必须记录所有修改。

### 9.1 如果是 normalization / slicing bug

必须：

```text
1. 修复代码；
2. rerun offline-online equivalence；
3. rerun short online rollout audit；
4. 输出 before/after；
5. report 写明 bug cause。
```

### 9.2 如果是 z_i scale 问题

不直接 full retrain PPO，但要给出建议：

```text
N3-rerun option:
  A. z_i train-stat normalization
  B. z_i LayerNorm
  C. obs + delta/logvar only，不喂 z_i
  D. small projection MLP with normalization
```

### 9.3 如果工程全通过但 PPO 仍差

必须给出：

```text
N3 method-level diagnosis
```

例如：

```text
1. Gψ augmented input维度过高，PPO难训练；
2. no-shield raw PPO无法直接使用 prediction features；
3. Gψ贡献可能主要在 N4 shield 侧；
4. 需要 N3-lite ablation，而不是直接判死。
```

---

## 10. 命令清单

Codex 应按实际路径调整命令，并记录最终命令。

### 10.1 编译检查

```bash
python -m py_compile scripts/audit_gpsi_online_wrapper.py
python -m py_compile scripts/compare_gpsi_offline_online.py
python -m py_compile scripts/inspect_gpsi_augmented_features.py
bash -n scripts/watch_phase_n3_5_gpsi_wrapper_audit.sh
chmod +x scripts/watch_phase_n3_5_gpsi_wrapper_audit.sh
```

### 10.2 Offline-online equivalence

```bash
python scripts/compare_gpsi_offline_online.py \
  --data-dir data/gpsi_head_a_v1 \
  --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --wrapper envs/wrappers/gpsi_obs_wrapper.py \
  --out-dir results/env_v2_phase_n3_5_gpsi_wrapper_audit \
  --split val \
  --num-samples 5000
```

### 10.3 Online wrapper audit

```bash
python scripts/audit_gpsi_online_wrapper.py \
  --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out-dir results/env_v2_phase_n3_5_gpsi_wrapper_audit \
  --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
  --num-episodes 10 \
  --policy random_or_straight_line \
  --write-input-output-stats
```

### 10.4 Augmented feature scale inspection

```bash
python scripts/inspect_gpsi_augmented_features.py \
  --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out-dir results/env_v2_phase_n3_5_gpsi_wrapper_audit \
  --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
  --num-episodes 10 \
  --write-plots
```

### 10.5 Watcher

```bash
bash scripts/watch_phase_n3_5_gpsi_wrapper_audit.sh
```

---

## 11. 报告要求

必须输出：

```text
results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_REPORT.md
```

报告至少包含：

```text
1. 背景：N3 success/collision 异常；
2. N2 vs N3 输出尺度冲突；
3. N3.5 目标；
4. 代码修改清单；
5. checkpoint load check；
6. offline-online equivalence test；
7. input distribution audit；
8. field order check；
9. history buffer / replacement check；
10. active mask check；
11. output slicing check；
12. delta/logvar/z output scale audit；
13. PPO augmented feature block-scale audit；
14. diagnostics correctness check；
15. 修复前后对比；
16. 是否确认 N3 原结果有效 / 无效 / 不可解释；
17. 是否需要重跑 N3；
18. 推荐重跑配置；
19. 是否可以进入 N4；
20. terminal_decision。
```

报告必须明确区分：

```text
engineering facts
confirmed bugs
remaining risks
method-level hypotheses
```

---

## 12. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag
```

必须满足：

```text
1. Phase N2 complete flag 存在；
2. N3 report / N3 outputs 可读取；
3. Gψ checkpoint 可加载；
4. offline-online equivalence 已执行；
5. input distribution audit 已执行；
6. output scale audit 已执行；
7. augmented feature block-scale audit 已执行；
8. field order / history buffer / active mask / slicing checks 均有结果；
9. 若发现 bug，已修复并 rerun audit；
10. online delta_hat norm 不再是 1e4 量级，除非证明原 diagnostics 错误且已修；
11. logvar 不再无解释地全部贴 -5；
12. projected_std 不再无解释地恒定；
13. report 生成；
14. watcher log 与 status 文件存在；
15. 明确给出：是否需要重跑 N3、是否允许进入 N4。
```

注意：

```text
N3.5 complete 不等于可以直接进入 N4。
如果结论是“必须重跑 N3”，则 complete flag 仍可生成，但 report 必须写：
Can enter N4: no, rerun N3 required.
```

---

## 13. 停止条件

如出现以下问题，必须生成 stop flag、partial report 和 log。

```text
PHASE_N3_5_STOP_PHASE_N2_MISSING.flag
PHASE_N3_5_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3_5_STOP_N3_ARTIFACTS_MISSING.flag
PHASE_N3_5_STOP_OFFLINE_ONLINE_MISMATCH.flag
PHASE_N3_5_STOP_INPUT_DISTRIBUTION_INVALID.flag
PHASE_N3_5_STOP_OUTPUT_SCALE_INVALID.flag
PHASE_N3_5_STOP_FEATURE_SCALE_INVALID.flag
PHASE_N3_5_STOP_WRAPPER_REPAIR_FAILED.flag
PHASE_N3_5_STOP_WATCHER_FAILED.flag
```

触发示例：

```text
N2 complete flag 缺失；
Gψ checkpoint 缺失；
N3 artifacts 缺失；
offline-online 同样本输出无法对齐；
online input 分布无法解释；
delta/logvar 输出尺度异常且无法修复；
aug feature block 尺度严重失控且无法提出修复；
watcher 无法运行。
```

---

## 14. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n3_5_gpsi_wrapper_audit.sh
```

watcher 必须：

```text
1. 检查 Phase N2 complete flag；
2. 检查 N3 artifacts；
3. 启动 offline-online equivalence；
4. 启动 online wrapper audit；
5. 启动 feature-scale audit；
6. 如有修复，重新运行 audit；
7. 持续轮询 complete / stop flag；
8. 持续输出当前状态；
9. 只有 complete flag 或 stop flag 出现才退出；
10. 不允许中途等待用户确认；
11. 不允许因为暂无新日志退出；
12. 写入 phase_n3_5_watcher.log；
13. 写入 phase_n3_5_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3_5_gpsi_wrapper_audit"
LOG="$OUT_DIR/phase_n3_5_watcher.log"
STATUS="$OUT_DIR/phase_n3_5_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3.5 watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3_5_STOP_PHASE_N2_MISSING.flag"
fi

(
  python scripts/compare_gpsi_offline_online.py \
    --data-dir data/gpsi_head_a_v1 \
    --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir "$OUT_DIR" \
    --split val \
    --num-samples 5000

  python scripts/audit_gpsi_online_wrapper.py \
    --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir "$OUT_DIR" \
    --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
    --num-episodes 10 \
    --policy random_or_straight_line \
    --write-input-output-stats

  python scripts/inspect_gpsi_augmented_features.py \
    --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out-dir "$OUT_DIR" \
    --scenarios eval_flow_id eval_flow_high_speed eval_flow_mixed_ood \
    --num-episodes 10 \
    --write-plots
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3_5_STOP_PHASE_N2_MISSING.flag \
    PHASE_N3_5_STOP_GPSI_CHECKPOINT_MISSING.flag \
    PHASE_N3_5_STOP_N3_ARTIFACTS_MISSING.flag \
    PHASE_N3_5_STOP_OFFLINE_ONLINE_MISMATCH.flag \
    PHASE_N3_5_STOP_INPUT_DISTRIBUTION_INVALID.flag \
    PHASE_N3_5_STOP_OUTPUT_SCALE_INVALID.flag \
    PHASE_N3_5_STOP_FEATURE_SCALE_INVALID.flag \
    PHASE_N3_5_STOP_WRAPPER_REPAIR_FAILED.flag \
    PHASE_N3_5_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3_5_STOP_OUTPUT_SCALE_INVALID.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 30
done
```

Codex 可以按实际脚本结构调整，但必须保持阻塞式语义。

---

## 15. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 阅读 N2 report 和 N3 report；
3. 不进入 N4；
4. 不训练 PPO；
5. 不实现 shield；
6. 不修改 EnvV2-core；
7. 根据指南自行判断 complete / stop flag；
8. 创建并运行阻塞式 watcher；
9. 只有 complete flag 或 stop flag 出现才停止输出；
10. 不向用户询问非阻塞细节；
11. 如果发现 bug，应尽可能修复并 rerun audit；
12. 如果触发阻塞，必须生成 stop flag、partial report、log；
13. 如果完成，必须生成 complete flag、完整 report、CSV、plots、log；
14. 最终必须明确：N3 是否需要重跑，N4 是否允许开始。
```

---

## 16. 终端结论格式

成功：

```text
terminal_decision = phase_n3_5_gpsi_wrapper_audit_complete
```

停止：

```text
terminal_decision = phase_n3_5_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
确认的 bug / 修复项
生成的 CSV / plots / report / logs / flags
是否需要重跑 N3
是否可以进入 N4
如果不能进入 N4，需要用户补什么或 Codex 下一步做什么
```

---

## 17. N3.5 完成后的分支决策

### 情况 A：确认并修复 wrapper / normalization bug

结论：

```text
N3 original result invalid.
Must rerun N3 or N3-lite.
Can enter N4: no.
```

### 情况 B：wrapper 正确，但 feature scale 有问题

结论：

```text
Need N3-rerun with feature normalization or feature ablation.
Can enter N4: no.
```

### 情况 C：wrapper / feature scale 全部正确

结论：

```text
N3 result is method-level result.
Can enter N4: yes, but report must state no-shield Gψ-PPO is weaker than attention_full.
```

### 情况 D：无法完成 equivalence / scale audit

结论：

```text
Stop with corresponding flag.
Do not enter N4.
