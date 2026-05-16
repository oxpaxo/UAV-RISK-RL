# Phase N3R 指南：Repaired Gψ-PPO No-Shield Rerun + z-block Ablation

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3R - Repaired Gψ-PPO No-Shield Rerun`  
> 阶段性质：修复后 PPO 短训重跑 + z-block normalization / no-z ablation；不接 shield；不训练 / fine-tune Gψ；不改 EnvV2-core。  
> 前置条件：Phase N3.5 已完成，确认并修复 Gψ online wrapper normalization bug。

---

## 0. 背景与当前判断

Phase N3 原始 Gψ-PPO no-shield 结果无效。Phase N3.5 已确认并修复 online Gψ normalization bug：

```text
N2 数据由 hold-position policy 采集；
ego_current 中 UAV velocity 相关维度 std 接近 1e-6；
N3 online PPO 产生非零 UAV velocity 后，legacy wrapper 除以近零 std；
normalized Gψ input 爆炸；
导致 delta_hat 万级、logvar clamp 到 -5。
```

修复后：

```text
offline-online equivalence after-fix 全部通过；
z/delta/logvar max_abs_diff = 0.0；
delta_norm_1s_p95 从约 22779.90 恢复到约 1.35；
delta_norm_1s_max 约 1.79；
logvar 不再无解释恒定，logvar_xy_1s_span 约 3.96。
```

但 N3.5 也发现新的 feature-scale 风险：

```text
z_i_64 p95 L2 norm ≈ 27.54
obs_i_12 p95 L2 norm ≈ 1.85
z_p95_to_obs_p95_l2_ratio ≈ 14.87
```

因此，修复后的 N3 不能只重跑原始设计。Phase N3R 必须同时做：

```text
A. repaired-full-raw-z
B. repaired-full-z-normalized
C. repaired-no-z
```

每个先训练 500k steps；如果资源不足，可降到 250k，但不得直接训练 1.5M。

---

## 1. Phase N3R 总目标

N3R 必须回答：

```text
1. 修复 wrapper 后，Gψ-PPO 是否恢复到合理性能区间？
2. raw z_i 是否继续干扰 PPO？
3. z_i normalization 是否必要？
4. 去掉 z_i、只保留 explicit Head A 输出 Δ̂/logvar 是否更稳？
5. 哪个版本值得进入后续 N3-full 或 N4 shield 对照？
```

本阶段训练三个 no-shield PPO screening configs：

```text
A. repaired-full-raw-z
   obs_i_aug = [obs_i, z_i_raw, delta_hat_scaled, logvar_hat]

B. repaired-full-z-normalized
   obs_i_aug = [obs_i, z_i_normalized, delta_hat_scaled, logvar_hat]

C. repaired-no-z
   obs_i_aug = [obs_i, delta_hat_scaled, logvar_hat]
```

共同要求：

```text
1. Gψ frozen；
2. PPO trainable；
3. no shield；
4. no action filtering；
5. no dense safety cost；
6. EnvV2 original reward；
7. masked-attention-compatible PPO backbone；
8. symmetric actor / critic；
9. same training budget；
10. same eval protocol；
11. same trace / diagnostics schema。
```

---

## 2. 明确禁止事项

Phase N3R 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止训练或 fine-tune Gψ；
3. 禁止实现 / 接入 safety shield；
4. 禁止 action filtering / projection；
5. 禁止加入 dense safety cost；
6. 禁止使用 learned R(s,a)；
7. 禁止使用 candidate velocity risk map as PPO input；
8. 禁止回到 5-head Gψ；
9. 禁止直接训练 1.5M 而不做 A/B/C 对比；
10. 禁止在 Gψ online output scale 异常时继续训练；
11. 禁止只看 aggregate success/collision，不看 raw unsafe 和 Gψ diagnostics。
```

允许：

```text
1. 使用修复后的 GpsiObsWrapper；
2. 为 z_i 计算 train-split-only normalization stats；
3. 新增 Gψ-PPO feature adapter；
4. 新增 no-z / z-normalized config；
5. 训练 A/B/C 三个 500k no-shield PPO；
6. 评估每个 config 的 checkpoints；
7. 输出 diagnostics、report、winner recommendation；
8. 若 500k 资源不足，可降为 250k，但必须在 report 中说明。
```

---

## 3. 三个训练配置定义

### 3.1 Config A：repaired-full-raw-z

目的：验证仅修复 wrapper 后，原 N3 设计是否恢复正常。

```text
name: gpsi_full_raw_z_repaired

input:
  obs_i: 12 dims
  z_i_raw: 64 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

total obstacle dim:
  94

normalization:
  obs_i: existing EnvV2/SB3 scale
  z_i: no extra normalization
  delta_hat: divided by delta_scale=5.0
  logvar_hat: clamped to [-5, 3]
```

风险：

```text
z_i scale may dominate PPO input.
```

---

### 3.2 Config B：repaired-full-z-normalized

目的：验证 z_i block normalization 是否解决尺度问题。

```text
name: gpsi_full_z_norm_repaired

input:
  obs_i: 12 dims
  z_i_normalized: 64 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

total obstacle dim:
  94
```

z_i normalization 要求：

```text
1. 使用 N1 train split 上 frozen Gψ forward 得到 z_i train statistics；
2. 只使用 train split，不使用 val/test；
3. 保存 z_mean[64], z_std[64]；
4. 对 degenerate z_std 加 floor，例如 1e-3；
5. online wrapper 中使用：z_i_normalized = (z_i - z_mean) / z_std；
6. 记录 z_norm before / after。
```

可选替代：

```text
LayerNorm(z_i)
```

但第一版优先使用 train-stat normalization，因为更可审计。

---

### 3.3 Config C：repaired-no-z

目的：判断 64维 latent z_i 是否干扰 PPO；只保留显式 Head A 输出。

```text
name: gpsi_no_z_repaired

input:
  obs_i: 12 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

total obstacle dim:
  30
```

解释：

```text
C 不代表最终方法一定不用 z_i；
C 是工程消融，检验 PPO 是否更容易利用显式 Δ̂/logvar，而不是高维 latent。
```

如果 C 明显优于 A/B，后续 N3-full 可先用 C 作为 no-shield baseline，并把 z_i 使用放到后续 feature ablation。

---

## 4. 训练规模

默认训练规模：

```text
A/B/C each:
  500,000 PPO steps
  seed = 0
  checkpoints: 250k, 500k
```

如果资源紧张，可降为：

```text
A/B/C each:
  250,000 PPO steps
```

但必须在 report 中明确：

```text
N3R used 250k screening budget due to resource constraints.
```

不得直接训练 1.5M。

---

## 5. 评估协议

每个 config 需要评估：

```text
checkpoint_250k
checkpoint_500k
final
best_by_eval if available
```

eval scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

默认 episodes：

```text
50 per scenario per config/checkpoint
```

如果资源不足可降到：

```text
20 per scenario
```

但必须报告。

必须包含 reference：

```text
attention_full_1500k
```

原始 N3 只能作为 invalid reference，不得作为有效方法 baseline。

Phase B 几何/filter baseline 仍作为背景上界：

```text
vo_like_filter_h45_cpa1p2_h16
cpa_ttc_weighted_apf_alpha3
```

但 N3R 的直接比较对象主要是：

```text
A vs B vs C vs attention_full_1500k
```

---

## 6. 必须记录的 diagnostics

### 6.1 Gψ output diagnostics

每个 config / checkpoint / scenario 记录：

```text
delta_norm_1s / 2s / 4s mean/median/p95/max
logvar_xy_1s / 2s / 4s mean/min/max/span
projected_std_radial mean/std
projected_std_relvel mean/std
z_norm_raw mean/median/p95/max
z_norm_after mean/median/p95/max
history_valid_ratio
nan_count
inf_count
```

要求：

```text
delta_norm 不得回到 1e4 量级；
logvar 不得无解释全贴 -5；
projected_std 不得无解释恒定；
inactive-forwarded count 必须为 0。
```

### 6.2 Augmented feature block stats

每个 config 记录：

```text
obs_block_l2_p95
z_block_l2_p95_raw
z_block_l2_p95_after_norm
delta_block_l2_p95
logvar_block_l2_p95
full_aug_obs_l2_p95
max_abs per block
nan/inf per block
```

Config C 中 z block 应标为 not applicable。

### 6.3 PPO policy diagnostics

必须输出：

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
episode_min_distance
episode_length
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
```

### 6.4 Breakdown

必须按以下维度输出：

```text
scenario
motion mode
threat class
checkpoint step
config
```

---

## 7. 推荐新增 / 修改文件

```text
configs/
  env_v2_gpsi_heada_ppo_n3r_raw_z.yaml
  env_v2_gpsi_heada_ppo_n3r_z_norm.yaml
  env_v2_gpsi_heada_ppo_n3r_no_z.yaml

scripts/
  compute_gpsi_z_stats.py
  train_env_v2_gpsi_ppo_n3r.py
  eval_env_v2_gpsi_ppo_n3r.py
  analyze_env_v2_phase_n3r_results.py
  watch_phase_n3r_gpsi_ppo_rerun.sh
```

可以复用 N3 的脚本，但必须避免覆盖原 N3 artifacts。

输出目录建议：

```text
checkpoints/
  env_v2_gpsi_heada_ppo_n3r_raw_z_s0/
  env_v2_gpsi_heada_ppo_n3r_z_norm_s0/
  env_v2_gpsi_heada_ppo_n3r_no_z_s0/

results/
  env_v2_phase_n3r_gpsi_ppo_rerun/
    PHASE_N3R_GPSI_PPO_RERUN_REPORT.md
    PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag
    phase_n3r_status.txt
    phase_n3r_watcher.log

    tables/
      phase_n3r_config_manifest.csv
      phase_n3r_z_stats.csv
      phase_n3r_train_curve.csv
      phase_n3r_checkpoint_eval_summary.csv
      phase_n3r_eval_summary.csv
      phase_n3r_attention_reference_comparison.csv
      phase_n3r_scenario_breakdown.csv
      phase_n3r_motion_mode_breakdown.csv
      phase_n3r_threat_class_breakdown.csv
      phase_n3r_raw_unsafe_action_summary.csv
      phase_n3r_gpsi_output_summary.csv
      phase_n3r_aug_feature_block_stats.csv
      phase_n3r_winner_recommendation.csv
      phase_n3r_command_manifest.csv
      phase_n3r_schema_check.csv

    plots/
      success_collision_by_config.png
      checkpoint_success_collision_by_config.png
      raw_unsafe_by_config.png
      aug_feature_block_scale_by_config.png
      gpsi_delta_norm_by_config.png
      gpsi_logvar_by_config.png
      train_reward_by_config.png
      scenario_breakdown_by_config.png

    logs/
      phase_n3r_train_raw_z.log
      phase_n3r_train_z_norm.log
      phase_n3r_train_no_z.log
      phase_n3r_eval.log
      phase_n3r_analysis.log
```

---

## 8. 命令清单

Codex 应按实际 repo 路径调整命令，并在 report 中记录最终实际命令。

### 8.1 编译检查

```bash
python -m py_compile scripts/compute_gpsi_z_stats.py
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3r.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3r.py
python -m py_compile scripts/analyze_env_v2_phase_n3r_results.py
bash -n scripts/watch_phase_n3r_gpsi_ppo_rerun.sh
chmod +x scripts/watch_phase_n3r_gpsi_ppo_rerun.sh
```

### 8.2 计算 z stats

```bash
python scripts/compute_gpsi_z_stats.py \
  --data-dir data/gpsi_head_a_v1 \
  --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
  --out results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_z_stats.csv \
  --out-npz work_dirs/gpsi_heada_v1_nll/z_stats_train_split.npz \
  --split train \
  --max-samples 200000
```

### 8.3 训练 A/B/C

```bash
python scripts/train_env_v2_gpsi_ppo_n3r.py \
  --config configs/env_v2_gpsi_heada_ppo_n3r_raw_z.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0 \
  --train-steps 500000 \
  --seed 0

python scripts/train_env_v2_gpsi_ppo_n3r.py \
  --config configs/env_v2_gpsi_heada_ppo_n3r_z_norm.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0 \
  --train-steps 500000 \
  --seed 0

python scripts/train_env_v2_gpsi_ppo_n3r.py \
  --config configs/env_v2_gpsi_heada_ppo_n3r_no_z.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0 \
  --train-steps 500000 \
  --seed 0
```

### 8.4 评估

```bash
python scripts/eval_env_v2_gpsi_ppo_n3r.py \
  --result-dir results/env_v2_phase_n3r_gpsi_ppo_rerun \
  --configs raw_z z_norm no_z \
  --eval-seed 1000 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --write-traces
```

### 8.5 分析

```bash
python scripts/analyze_env_v2_phase_n3r_results.py \
  --result-dir results/env_v2_phase_n3r_gpsi_ppo_rerun
```

### 8.6 watcher

```bash
bash scripts/watch_phase_n3r_gpsi_ppo_rerun.sh
```

---

## 9. 报告要求

必须输出：

```text
results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_REPORT.md
```

报告至少包含：

```text
1. 背景：N3 原结果因 wrapper bug 无效；
2. N3.5 修复摘要；
3. N3R 目标；
4. A/B/C config 定义；
5. z stats 计算方式；
6. 训练预算；
7. eval 协议；
8. training curves；
9. checkpoint eval summary；
10. A/B/C aggregate comparison；
11. attention_full reference comparison；
12. scenario / motion / threat breakdown；
13. raw unsafe action analysis；
14. Gψ output diagnostics；
15. augmented feature block-scale analysis；
16. winner recommendation；
17. 是否需要 full 1.5M rerun；
18. 是否可以进入 N4；
19. terminal_decision。
```

报告必须明确区分：

```text
engineering facts
experiment-supported facts
reasonable inferences
remaining risks
```

---

## 10. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag
```

必须满足：

```text
1. Phase N3.5 complete flag 存在；
2. repaired GpsiObsWrapper 被使用；
3. Gψ checkpoint 可加载；
4. Gψ frozen；
5. A/B/C 三个 configs 均训练完成到指定 budget；
6. 每个 config 至少保存 250k 和 final checkpoint；
7. eval 完成；
8. Gψ output scale 正常；
9. augmented feature block stats 生成；
10. raw unsafe diagnostics 生成；
11. scenario/motion/threat breakdown 生成；
12. report 生成；
13. watcher log 与 status 文件存在；
14. 明确 winner / no-winner；
15. 明确是否需要 full N3 1.5M rerun；
16. 明确是否可进入 N4。
```

注意：

```text
N3R complete 不等于必须进入 N4。
如果所有 A/B/C 都很差或 diagnostics 异常，应 complete 但 report 写：
Can enter N4: no.
Recommended next: repair / ablation / N3R rerun.
```

---

## 11. 停止条件

如出现以下问题，必须生成 stop flag、partial report 和 log。

```text
PHASE_N3R_STOP_PHASE_N3_5_MISSING.flag
PHASE_N3R_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3R_STOP_Z_STATS_FAILED.flag
PHASE_N3R_STOP_WRAPPER_SCALE_INVALID.flag
PHASE_N3R_STOP_TRAIN_FAILED.flag
PHASE_N3R_STOP_EVAL_FAILED.flag
PHASE_N3R_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3R_STOP_RESOURCE_LIMIT.flag
PHASE_N3R_STOP_WATCHER_FAILED.flag
```

触发示例：

```text
N3.5 complete flag 缺失；
Gψ checkpoint 缺失；
z stats 无法计算；
修复后的 wrapper 输出又出现 delta 万级 / logvar 恒定；
A/B/C 任一关键 config 无法训练；
eval 失败；
diagnostics 无法生成；
资源不足以完成 250k/500k screening；
watcher 失败。
```

---

## 12. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n3r_gpsi_ppo_rerun.sh
```

watcher 必须：

```text
1. 检查 Phase N3.5 complete flag；
2. 计算 z stats；
3. 依次或并行训练 A/B/C；
4. 评估 checkpoints；
5. 运行 analysis；
6. 生成 report；
7. 持续轮询 complete / stop flag；
8. 持续输出当前状态；
9. 只有 complete flag 或 stop flag 出现才退出；
10. 不允许中途等待用户确认；
11. 不允许因为暂无新日志退出；
12. 写入 phase_n3r_watcher.log；
13. 写入 phase_n3r_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3r_gpsi_ppo_rerun"
LOG="$OUT_DIR/phase_n3r_watcher.log"
STATUS="$OUT_DIR/phase_n3r_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3R watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3R_STOP_PHASE_N3_5_MISSING.flag"
fi

(
  python scripts/compute_gpsi_z_stats.py \
    --data-dir data/gpsi_head_a_v1 \
    --checkpoint work_dirs/gpsi_heada_v1_nll/best.pth \
    --out "$OUT_DIR/tables/phase_n3r_z_stats.csv" \
    --out-npz work_dirs/gpsi_heada_v1_nll/z_stats_train_split.npz \
    --split train \
    --max-samples 200000

  python scripts/train_env_v2_gpsi_ppo_n3r.py \
    --config configs/env_v2_gpsi_heada_ppo_n3r_raw_z.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0 \
    --train-steps 500000 \
    --seed 0

  python scripts/train_env_v2_gpsi_ppo_n3r.py \
    --config configs/env_v2_gpsi_heada_ppo_n3r_z_norm.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0 \
    --train-steps 500000 \
    --seed 0

  python scripts/train_env_v2_gpsi_ppo_n3r.py \
    --config configs/env_v2_gpsi_heada_ppo_n3r_no_z.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0 \
    --train-steps 500000 \
    --seed 0

  python scripts/eval_env_v2_gpsi_ppo_n3r.py \
    --result-dir "$OUT_DIR" \
    --configs raw_z z_norm no_z \
    --eval-seed 1000 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --write-traces

  python scripts/analyze_env_v2_phase_n3r_results.py \
    --result-dir "$OUT_DIR"
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3R_STOP_PHASE_N3_5_MISSING.flag \
    PHASE_N3R_STOP_GPSI_CHECKPOINT_MISSING.flag \
    PHASE_N3R_STOP_Z_STATS_FAILED.flag \
    PHASE_N3R_STOP_WRAPPER_SCALE_INVALID.flag \
    PHASE_N3R_STOP_TRAIN_FAILED.flag \
    PHASE_N3R_STOP_EVAL_FAILED.flag \
    PHASE_N3R_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3R_STOP_RESOURCE_LIMIT.flag \
    PHASE_N3R_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3R_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 60
done
```

Codex 可以按实际脚本结构调整，但必须保持阻塞式语义。

---

## 13. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 阅读 N3.5 report；
3. 不进入 N4；
4. 不训练 / fine-tune Gψ；
5. 不实现 shield；
6. 不修改 EnvV2-core；
7. 根据指南自行判断 complete / stop flag；
8. 创建并运行阻塞式 watcher；
9. 只有 complete flag 或 stop flag 出现才停止输出；
10. 不向用户询问非阻塞细节；
11. 如果触发阻塞，必须生成 stop flag、partial report、log；
12. 如果完成，必须生成 complete flag、完整 report、checkpoints、CSV、plots、log；
13. 最终必须明确：哪个 config 值得 full 1.5M rerun，是否允许进入 N4。
```

---

## 14. 终端结论格式

成功：

```text
terminal_decision = phase_n3r_gpsi_ppo_rerun_complete
```

停止：

```text
terminal_decision = phase_n3r_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
A/B/C 训练预算
A/B/C 主要结果
winner recommendation
生成的 checkpoints
生成的 CSV / plots / report / logs / flags
是否需要 full N3 1.5M rerun
是否可以进入 N4
如果不能进入 N4，下一步做什么
```

---

## 15. N3R 完成后的分支决策

### 情况 A：某个 config 明显恢复正常且接近 / 超过 attention_full

```text
Recommended:
  用该 config 做 full N3 1.5M rerun；
  然后进入 N4 shield。
```

### 情况 B：某个 config 明显优于其他 Gψ config，但仍弱于 attention_full

```text
Recommended:
  可做 full N3 rerun 或直接作为 N4 Gψ policy candidate；
  但 report 必须说明 no-shield Gψ raw policy 不强。
```

### 情况 C：A/B/C 全部很差，但 diagnostics 正常

```text
Recommended:
  方法层面问题；
  Gψ may be useful mainly for shield side；
  N4 可考虑使用 attention_full + Gψ shield only / Gψ as shield predictor；
  但不能声称 Gψ-PPO no-shield 有效。
```

### 情况 D：diagnostics 再次异常

```text
Recommended:
  不进入 N4；
  回到 wrapper / feature audit。
```
