# Phase N3P 指南：no_z Representation Conditioning Ablation

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：DynamicObstacleFlowEnv / EnvV2  
> 阶段名称：Phase N3P - no_z Representation Conditioning Ablation  
> 阶段性质：Gψ-HeadA 显式输出的 PPO 输入表征修复筛查；不接 shield；不训练 / fine-tune Gψ；不改 EnvV2-core。  
> 前置条件：N3Z2CF 已完成，当前最强 no-shield Gψ-PPO candidate 为 no_z full，但仍弱于 attention_full。

---

## 0. 背景与当前问题

当前 no-shield policy 结果：

```text
attention_full_1500k:
  success = 0.6100
  collision = 0.3900

N3F no_z full 1.5M:
  success = 0.5633
  collision = 0.4367

corrected Z2 z_layernorm_alpha_0p5 full 1.5M:
  success = 0.5067
  collision = 0.4933
```

当前主线 candidate 是：

```text
selected current N4 candidate: no_z full
z_i status: ablation only
```

但如果论文叙事希望把 `Gψ-HeadA + PPO` 本身作为 two-step policy-level 创新，M1 no-shield 至少应接近、最好超过 `attention_full`。否则审稿人会问：

```text
为什么不用更简单、更强的 attention_full？
为什么要引入前置监督 Gψ / two-step 架构？
```

当前最可能的问题不是 Gψ 不可学习，而是 Gψ 输出给 PPO 的表征形式不适合 on-policy learning。重点怀疑：

```text
1. logvar block 数值范数过大，可能压制 obs_i / Δ̂；
2. Δ̂ / logvar 直接拼接不是 decision-ready safety feature；
3. 30维输入相对 12维 attention_full 增加了 PPO sample-efficiency 负担；
4. Δ̂ 与 obs_i 里的 planned CPA/TTC 信息存在部分边际重叠；
5. PPO 需要额外学会“何时使用 / 忽略”这些预测特征。
```

Phase N3P 的目的不是继续研究 z_i，而是围绕当前主候选 no_z 做小规模表征修复筛查，判断 M1 no-shield 是否还有机会追上或超过 attention_full。

---

## 1. Phase N3P 总目标

本阶段只做 500k screening，不直接跑 1.5M。

训练三个 no-shield PPO variants：

```text
P1: obs_delta_only
  obs_i_aug = [obs_i, delta_hat_scaled]
  目的：检验 logvar 是否拖累 PPO。

P2: obs_delta_logvar_scaled
  obs_i_aug = [obs_i, delta_hat_scaled, logvar_scaled]
  目的：保留 uncertainty 信息，但把 logvar block 缩到合理范数。

P3: block_projected_no_z
  obs_i / delta_hat / uncertainty 各自投影和归一化后再融合。
  目的：检验 block-wise conditioning 是否解决 sample-efficiency / scale 问题。
```

共同要求：

```text
Gψ frozen
PPO trainable
no shield
no action filtering
no dense safety cost
EnvV2-core unchanged
same eval protocol
same seed=0
train budget = 500k each
```

本阶段必须回答：

```text
1. 去掉 logvar 后是否优于 N3R no_z 500k？
2. 缩放 logvar 后是否优于 raw-logvar no_z？
3. block-wise projection 是否改善 PPO sample efficiency？
4. 是否存在一个 representation candidate 值得后续 full 1.5M？
5. 如果 P1/P2/P3 都不行，是否应停止继续修 no-shield PPO，转入 N4 shield？
```

---

## 2. 明确禁止事项

Phase N3P 禁止：

```text
1. 禁止进入 N4；
2. 禁止实现 shield；
3. 禁止 action filtering / projection；
4. 禁止加入 dense safety cost；
5. 禁止训练或 fine-tune Gψ；
6. 禁止修改 EnvV2-core；
7. 禁止继续做 z_i 相关 variants；
8. 禁止重训 attention_full；
9. 禁止覆盖 N3F / N3Z2CF / N3R 原始产物；
10. 禁止直接跑 1.5M；
11. 禁止只按 train reward 选 winner，必须看 eval success/collision 和 breakdown。
```

允许：

```text
1. 改变 PPO 输入表征；
2. 改变 feature adapter / projection；
3. 对 logvar 做缩放、tanh、clip、block-level normalization；
4. 训练 P1/P2/P3 各 500k；
5. 评估并推荐是否有 winner 进入后续 1.5M。
```

---

## 3. Watcher 与 token 输出要求

本阶段包含三个 500k 短训练，预计仍可能耗时较长。为节省 token：

```text
watcher poll interval: 120 seconds
chat/status heartbeat interval: about 300 seconds
```

要求：

```text
1. watcher 必须阻塞式运行；
2. 日志可以持续写文件；
3. 聊天输出不要每 60 秒刷屏；
4. 除 checkpoint、阶段切换、异常、stop flag、complete flag 外，约 5 分钟输出一次简短状态；
5. 输出内容应压缩为 current job / step / fps / latest checkpoint / ETA；
6. complete 或 stop 时必须及时输出。
```

---

## 4. CPU / 性能要求

固定实验机器：

```text
OS: Ubuntu 22.04.5 LTS
CPU: AMD EPYC 7402
physical cores: 8
logical CPUs: 16
RAM: 62 GiB
GPU: RTX 3090 24 GiB
```

N3Z2C-Audit 已确认：

```text
nproc --all = 16
Python affinity_count = 16
cpuset = 0-15
cpu.max = max 100000
nproc=1 是 OMP_NUM_THREADS=1 影响，不是 cgroup/affinity 限死。
```

本阶段仍以科学可比性优先：

```text
default n_envs = 4
device = cpu
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

如果 Codex 希望并行跑 P1/P2/P3，需要先做资源风险判断：

```text
1. 不允许抢占导致任一训练崩溃；
2. 不允许 watcher 混乱；
3. 不允许覆盖 checkpoint；
4. 不允许日志交叉；
5. 不允许显著改变 PPO rollout semantics；
6. 若并行，必须每个 job 独立 out-dir / log / PID / stop handling。
```

优先策略：

```text
sequential by default
parallel only if Codex can guarantee safe process isolation
```

---

## 5. Baselines / reference metrics

本阶段必须引用但不重训：

```text
N3R no_z 500k:
  success = 0.4233
  collision = 0.5767

N3F no_z full 1.5M:
  success = 0.5633
  collision = 0.4367

attention_full_1500k:
  success = 0.6100
  collision = 0.3900

corrected Z2 full:
  success = 0.5067
  collision = 0.4933
```

500k screening 的硬门槛：

```text
success_rate >= 0.4233
collision_rate <= 0.5767
diagnostics_ok = true
```

如果没有 candidate 过硬门槛：

```text
stop no-shield representation tuning;
proceed to N4 with no_z full as candidate.
```

---

## 6. Variant P1：obs_delta_only

### 6.1 目的

直接检验 raw logvar block 是否拖累 PPO。

### 6.2 Input

```text
obs_i: 12 dims
delta_hat_scaled: 9 dims
obs_aug_dim = 21
```

### 6.3 Transform

```text
delta_hat_scaled = delta_hat / 5.0
```

No logvar.

### 6.4 Interpretation

```text
If P1 > no_z 500k:
  logvar likely hurts PPO in raw concatenation form.

If P1 < no_z 500k:
  uncertainty information may be useful, or delta-only is insufficient.
```

---

## 7. Variant P2：obs_delta_logvar_scaled

### 7.1 目的

保留 logvar 信息，但避免 raw logvar L2 norm 过大。

### 7.2 Input

```text
obs_i: 12 dims
delta_hat_scaled: 9 dims
logvar_scaled: 9 dims
obs_aug_dim = 30
```

### 7.3 Default transform

Use bounded scaled logvar:

```text
logvar_clamped = clamp(logvar, -5, 3)
logvar_scaled = logvar_clamped / 5.0
```

Expected block scale:

```text
logvar_scaled_l2_p95 should be around 2-3, not 14-15.
```

Optional diagnostics only:

```text
std = exp(0.5 * logvar_clamped)
std_l2_p95
```

Do not use std as policy input in P2 unless explicitly configured. Keep P2 as the clean scale-control test.

### 7.4 Interpretation

```text
If P2 > P1 and > no_z 500k:
  uncertainty is useful, but raw logvar scale was harmful.

If P2 ≈ no_z or worse:
  logvar uncertainty may not be policy-useful in this form.
```

---

## 8. Variant P3：block_projected_no_z

### 8.1 目的

测试 block-wise conditioning 是否改善 sample efficiency。

Raw concatenation forces PPO to learn from heterogeneous blocks:

```text
obs_i scale ≈ 1-2
delta scale ≈ 1-2
raw logvar scale ≈ 14-15
```

P3 explicitly projects blocks before fusion.

### 8.2 Input blocks

```text
obs_i: 12 dims
delta_hat_scaled: 9 dims
logvar_scaled: 9 dims
```

### 8.3 Adapter

Recommended minimal adapter:

```text
obs_i -> Linear(12, 32) -> LayerNorm -> Tanh/ReLU
delta_hat_scaled -> Linear(9, 16) -> LayerNorm -> Tanh/ReLU
logvar_scaled -> Linear(9, 16) -> LayerNorm -> Tanh/ReLU
concat -> 64 dims per obstacle embedding input
```

Then feed into masked-attention-compatible PPO backbone.

Notes:

```text
1. Gψ stays frozen；
2. adapter belongs to PPO policy, trainable；
3. record adapter parameter count；
4. do not alter EnvV2-core；
5. do not add shield。
```

### 8.4 Interpretation

```text
If P3 wins:
  The issue was not Gψ information but raw concatenation / conditioning.

If P3 does not win:
  HeadA features may have limited marginal value for raw PPO.
```

---

## 9. Training protocol

Each variant:

```text
train_steps = 500,000
seed = 0
checkpoints:
  250k
  500k
  final
  best_by_eval if available
```

Training must log:

```text
model_num_timesteps
n_envs
n_steps
batch_size
learning_rate / schedule if available
clip_range / schedule if available
fps
episode reward rolling mean
policy entropy
approx_kl
clip_fraction
value_loss
explained_variance
```

---

## 10. Eval protocol

Evaluate:

```text
P1 250k / 500k / final / best_by_eval
P2 250k / 500k / final / best_by_eval
P3 250k / 500k / final / best_by_eval
```

Scenarios:

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

Episodes:

```text
50 per scenario
```

Eval seed:

```text
1000
```

References:

```text
N3R no_z 500k
N3F no_z full
attention_full
corrected Z2 full
```

---

## 11. Required diagnostics

### 11.1 Feature block stats

For each variant / checkpoint / scenario:

```text
obs_i_12 l2_p95
delta_hat_9_after_scale l2_p95
logvar_raw_l2_p95 if available
logvar_scaled_l2_p95 if used
full_aug_obs_l2_p95
adapter_output_l2_p95 if P3
max_abs_p95 per block
nan_count
inf_count
```

Hard expectation:

```text
P2 logvar_scaled_l2_p95 should be far lower than raw logvar_l2_p95.
P3 adapter output should be finite and not collapse.
```

### 11.2 Gψ output diagnostics

```text
delta_norm_1s / 2s / 4s mean / median / p95 / max
logvar_xy_1s / 2s / 4s mean / min / max / span
projected_std_radial mean / std
projected_std_relvel mean / std
history_valid_ratio
inactive_forwarded_count
nan_count
inf_count
```

### 11.3 PPO / safety diagnostics

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
episode_length
episode_reward
```

Special attention:

```text
1. high_density / high_speed / high_threat / sudden_threat；
2. raw unsafe low but collision high；
3. action_delta too low；
4. whether P variants improve no_z weak scenarios。
```

---

## 12. Decision rules

### 12.1 500k hard gate

A P variant passes if:

```text
success_rate >= 0.4233
collision_rate <= 0.5767
diagnostics_ok = true
```

### 12.2 Winner ranking

Among variants passing hard gate, rank by:

```text
1. lower collision_rate
2. higher success_rate
3. lower near_miss_rate
4. lower raw_unsafe_action_rate
5. better scenario robustness
6. cleaner feature block scale
```

### 12.3 Promote to 1.5M?

Recommend a variant for 1.5M only if:

```text
1. It passes hard gate；
2. It is clearly better than N3R no_z 500k；
3. It is not worse than no_z 500k on high_speed/high_threat；
4. feature diagnostics are clean；
5. training curve still improving or at least not collapsing。
```

If P1/P2/P3 all fail or barely match no_z:

```text
Do not keep tuning no-shield representation.
Proceed to N4 using no_z full as main candidate.
```

---

## 13. Output directory and files

Output directory:

```text
results/env_v2_phase_n3p_noz_representation_ablation/
```

Checkpoint directories:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0/
checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0/
checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/
```

Required files:

```text
PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md
PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag
phase_n3p_status.txt
phase_n3p_watcher.log
```

Required tables:

```text
phase_n3p_config_manifest.csv
phase_n3p_command_manifest.csv
phase_n3p_resource_affinity.csv
phase_n3p_train_curve.csv
phase_n3p_train_heartbeat.csv
phase_n3p_checkpoint_eval_summary.csv
phase_n3p_eval_summary.csv
phase_n3p_reference_comparison.csv
phase_n3p_scenario_breakdown.csv
phase_n3p_motion_mode_breakdown.csv
phase_n3p_threat_class_breakdown.csv
phase_n3p_raw_unsafe_action_summary.csv
phase_n3p_gpsi_output_summary.csv
phase_n3p_feature_block_stats.csv
phase_n3p_winner_recommendation.csv
phase_n3p_schema_check.csv
```

Required plots:

```text
n3p_success_collision_by_variant.png
n3p_checkpoint_success_collision.png
n3p_train_reward.png
n3p_feature_block_scale.png
n3p_raw_unsafe_by_variant.png
n3p_scenario_breakdown.png
n3p_gpsi_delta_norm.png
n3p_gpsi_logvar.png
```

---

## 14. Stop flags

Create partial report and stop if needed:

```text
PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag
PHASE_N3P_STOP_CONFIG_INVALID.flag
PHASE_N3P_STOP_TRAIN_FAILED.flag
PHASE_N3P_STOP_EVAL_FAILED.flag
PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3P_STOP_FEATURE_SCALE_INVALID.flag
PHASE_N3P_STOP_WATCHER_FAILED.flag
```

Stop examples:

```text
Gψ checkpoint missing；
N3F no_z / attention reference missing；
P1/P2/P3 config cannot be validated；
training fails；
eval fails；
feature stats nonfinite；
P2 logvar_scaled still has raw-logvar scale；
watcher exits without complete/stop。
```

---

## 15. Completion criteria

Only create:

```text
PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag
```

when all are true:

```text
1. P1/P2/P3 configs validated；
2. Gψ frozen confirmed；
3. no shield/action filtering/safety cost confirmed；
4. all required variants trained to 500k；
5. eval completed；
6. feature block stats generated；
7. Gψ diagnostics normal；
8. scenario/motion/threat breakdown generated；
9. winner recommendation generated；
10. report generated；
11. watcher log and status exist。
```

Complete does not mean a variant is promoted. Report must explicitly state:

```text
promote_to_1p5m: yes/no
winner_if_any: P1/P2/P3/none
can_enter_N4_now: yes/no
```

---

## 16. Suggested commands

Codex should adapt paths.

### 16.1 Compile

```bash
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3p.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3p.py
python -m py_compile scripts/analyze_env_v2_phase_n3p_results.py
bash -n scripts/watch_phase_n3p_noz_representation_ablation.sh
chmod +x scripts/watch_phase_n3p_noz_representation_ablation.sh
```

### 16.2 Train P1

```bash
python scripts/train_env_v2_gpsi_ppo_n3p.py   --config configs/env_v2_gpsi_heada_ppo_n3p_obs_delta_only.yaml   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0   --train-steps 500000   --seed 0   --n-envs 4   --device cpu   --heartbeat-seconds 300
```

### 16.3 Train P2

```bash
python scripts/train_env_v2_gpsi_ppo_n3p.py   --config configs/env_v2_gpsi_heada_ppo_n3p_logvar_scaled.yaml   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0   --train-steps 500000   --seed 0   --n-envs 4   --device cpu   --heartbeat-seconds 300
```

### 16.4 Train P3

```bash
python scripts/train_env_v2_gpsi_ppo_n3p.py   --config configs/env_v2_gpsi_heada_ppo_n3p_block_projected.yaml   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0   --train-steps 500000   --seed 0   --n-envs 4   --device cpu   --heartbeat-seconds 300
```

### 16.5 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3p.py   --result-dir results/env_v2_phase_n3p_noz_representation_ablation   --configs obs_delta_only logvar_scaled block_projected   --eval-seed 1000   --num-episodes 50   --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat   --write-traces
```

### 16.6 Analysis

```bash
python scripts/analyze_env_v2_phase_n3p_results.py   --result-dir results/env_v2_phase_n3p_noz_representation_ablation   --n3r-noz-success 0.4233   --n3r-noz-collision 0.5767   --n3f-noz-success 0.5633   --n3f-noz-collision 0.4367   --attention-success 0.6100   --attention-collision 0.3900   --z2-success 0.5067   --z2-collision 0.4933
```

### 16.7 Watcher

```bash
bash scripts/watch_phase_n3p_noz_representation_ablation.sh
```

---

## 17. Watcher pseudo-code

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3p_noz_representation_ablation"
LOG="$OUT_DIR/phase_n3p_watcher.log"
STATUS="$OUT_DIR/phase_n3p_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3P watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/train_env_v2_gpsi_ppo_n3p.py     --config configs/env_v2_gpsi_heada_ppo_n3p_obs_delta_only.yaml     --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0     --train-steps 500000     --seed 0     --n-envs 4     --device cpu     --heartbeat-seconds 300

  python scripts/train_env_v2_gpsi_ppo_n3p.py     --config configs/env_v2_gpsi_heada_ppo_n3p_logvar_scaled.yaml     --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0     --train-steps 500000     --seed 0     --n-envs 4     --device cpu     --heartbeat-seconds 300

  python scripts/train_env_v2_gpsi_ppo_n3p.py     --config configs/env_v2_gpsi_heada_ppo_n3p_block_projected.yaml     --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0     --train-steps 500000     --seed 0     --n-envs 4     --device cpu     --heartbeat-seconds 300

  python scripts/eval_env_v2_gpsi_ppo_n3p.py     --result-dir "$OUT_DIR"     --configs obs_delta_only logvar_scaled block_projected     --eval-seed 1000     --num-episodes 50     --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat     --write-traces

  python scripts/analyze_env_v2_phase_n3p_results.py     --result-dir "$OUT_DIR"     --n3r-noz-success 0.4233     --n3r-noz-collision 0.5767     --n3f-noz-success 0.5633     --n3f-noz-collision 0.4367     --attention-success 0.6100     --attention-collision 0.3900     --z2-success 0.5067     --z2-collision 0.4933
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in     PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag     PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag     PHASE_N3P_STOP_CONFIG_INVALID.flag     PHASE_N3P_STOP_TRAIN_FAILED.flag     PHASE_N3P_STOP_EVAL_FAILED.flag     PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag     PHASE_N3P_STOP_FEATURE_SCALE_INVALID.flag     PHASE_N3P_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3P_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  sleep 120
done
```

---

## 18. Terminal decision format

Success:

```text
terminal_decision = phase_n3p_noz_representation_ablation_complete
```

Stop:

```text
terminal_decision = phase_n3p_stopped_<reason>
```

Must report:

```text
new / modified files
actual commands
P1 / P2 / P3 results
feature scale findings
whether logvar was harmful
winner recommendation
promote_to_1p5m yes/no
can_enter_N4 yes/no
if not, next required action
```
