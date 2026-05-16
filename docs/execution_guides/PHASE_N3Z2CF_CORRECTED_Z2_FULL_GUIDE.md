# Phase N3Z2CF 指南：Corrected Z2 Full Continuation + Final No-Shield Candidate Decision

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3Z2CF - Corrected Z2 Full Continuation`  
> 阶段性质：从修正后的 Z2 500k parent 重新做 full continuation 到 1.5M，并与 no_z full / attention_full 做最终 no-shield candidate 决策。  
> 前置条件：Phase N3Z2C-Audit 已完成，确认旧 N3Z2C 存在 parent selection bug 与 `reset_num_timesteps=True` resume bug。  

---

## 0. 背景与审计结论

Phase N3Z2C-Audit 已确认：

```text
1. fixed parent selection:
   正确 parent 应为 checkpoint_500k.zip，而不是 best_by_eval/final.zip。

2. old parent selection:
   旧 N3Z2C 错选 best_by_eval/final；
   但 checkpoint_500k 的 success/collision 更优。

3. resume semantics:
   旧 N3Z2C continuation 使用 reset_num_timesteps=True；
   这是 continuation bug。
   corrected audit 使用 reset_num_timesteps=False，SB3 optimizer state 已恢复。

4. CPU affinity:
   nproc=1 是 OMP_NUM_THREADS=1 影响；
   nproc --all=16，Python affinity_count=16，cpuset=0-15，cpu.max=max 100000；
   当前不是 cgroup / affinity 限死为 1 核。

5. corrected short continuation sanity:
   fixed checkpoint_500k → corrected 750k:
     success = 0.4800
     collision = 0.5200

   old N3Z2C 750k:
     success = 0.4667
     collision = 0.5333
```

因此旧 N3Z2C 的 Z2 1.5M 结果不能作为最终结论。本阶段必须从正确的 `checkpoint_500k.zip` 出发，用 `reset_num_timesteps=False` 重跑 Z2 full continuation 到 total 1.5M。

---

## 1. 本阶段目标

本阶段只做一条主线：

```text
Z2 corrected full continuation:
  parent = checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip
  parent_total_steps = 500,000
  additional_steps = 1,000,000
  target_total_steps = 1,500,000
  reset_num_timesteps = False
```

然后比较：

```text
corrected Z2 full 1.5M
vs
N3F no_z full 1.5M
vs
attention_full 1.5M
```

必须回答：

```text
1. corrected Z2 full 是否超过 no_z full？
2. corrected Z2 full 是否接近或超过 attention_full？
3. corrected Z2 是否改善 no_z 的短板场景：
   high_density / high_speed / high_threat / sudden_threat？
4. corrected Z2 是否仍存在 long-training policy drift？
5. 最终 N4 应使用：
   no_z full、corrected Z2 full、还是两者都作为 ablation？
```

---

## 2. 明确禁止事项

Phase N3Z2CF 禁止：

```text
1. 禁止进入 N4；
2. 禁止实现 shield；
3. 禁止 action filtering / projection；
4. 禁止加入 dense safety cost；
5. 禁止训练或 fine-tune Gψ；
6. 禁止修改 EnvV2-core；
7. 禁止重训 no_z；
8. 禁止重训 Z1；
9. 禁止复用旧 N3Z2C 的错误 parent；
10. 禁止使用 reset_num_timesteps=True；
11. 禁止覆盖 N3Z2C / N3Z2C-Audit / N3FZ 原始产物；
12. 禁止只看 train reward 决策，必须使用 eval success/collision 与 breakdown。
```

允许：

```text
1. 使用 repaired GpsiObsWrapper；
2. 使用 fixed Z2 checkpoint_500k parent；
3. 使用 reset_num_timesteps=False；
4. 继续训练 Z2 到 total 1.5M；
5. 保存 total-step checkpoints；
6. 做 eval / diagnostics / final candidate decision；
7. 报告是否可以进入 N4。
```

---

## 3. Watcher 与 token 输出要求

本阶段是长训练。为了节省 token：

```text
watcher poll interval: 120 seconds
chat/status heartbeat interval: 300 seconds or longer
```

要求：

```text
1. watcher 必须阻塞式运行；
2. 日志可以持续写文件；
3. 聊天输出不要每 60 秒刷屏；
4. 除 checkpoint、阶段切换、异常、stop flag、complete flag 外，约 5 分钟输出一次简短状态即可；
5. 如果训练较长但正常运行，输出内容应压缩：
   current total step / target step / fps / ETA / latest checkpoint；
6. complete 或 stop 时必须及时输出。
```

---

## 4. CPU / 性能要求

当前机器固定：

```text
OS: Ubuntu 22.04.5 LTS
CPU: AMD EPYC 7402
physical cores: 8
logical CPUs: 16
RAM: 62 GiB
GPU: RTX 3090 24 GiB
```

N3Z2C-Audit 已确认不是 cgroup/affinity 限死 1 核：

```text
nproc --all = 16
Python affinity_count = 16
cpuset = 0-15
cpu.max = max 100000
```

本阶段必须继续记录资源信息，但训练配置以科学可比性优先。

### 4.1 环境变量

建议：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

这些变量用于避免每个 env worker 内部多线程 oversubscription。不要再把 `nproc=1` 误判为 CPU affinity 只有 1 核。

### 4.2 n_envs

默认保持：

```text
n_envs = 4
```

理由：

```text
1. 与 Z2 原训练 / audit short continuation 保持一致；
2. 避免改变 PPO rollout semantics；
3. 本阶段目的是 corrected full continuation，不是新性能优化实验。
```

如果 Codex 要尝试更高 `n_envs`，必须满足：

```text
1. non-destructive benchmark；
2. 不覆盖正式 checkpoint；
3. 能保持 effective batch / PPO rollout semantics 可比；
4. report 明确说明没有改变科学比较口径；
5. 否则回退 n_envs=4。
```

本阶段优先建议不改 `n_envs`，只记录 CPU affinity 与训练 fps。

---

## 5. Z2 corrected full 配置

### 5.1 Method

```text
method_key: z2_corrected_full
method: gpsi_z_layernorm_alpha_0p5_corrected_full_1p5m
```

### 5.2 Input

```text
obs_i: 12 dims
z_layernorm_scaled: 64 dims
delta_hat_scaled: 9 dims
logvar_hat: 9 dims

obs_aug_dim = 94
```

### 5.3 z transform

```text
z_ln = LayerNorm(z_i)
z_used = 0.5 * z_ln
```

Expected feature scale:

```text
z_after_constraint_l2_p95 ≈ 4
```

### 5.4 logvar

Use current bounded setting:

```text
logvar_clamp = [-5, 3]
```

This is tighter than `|logvar| <= 5`.

### 5.5 Gψ

```text
checkpoint = work_dirs/gpsi_heada_v1_nll/best.pth
Gψ frozen = true
Gψ eval() = true
requires_grad_any = false
```

### 5.6 PPO

```text
PPO trainable
same PPO backbone as Z2
same EnvV2 reward
no shield
no safety cost
no action filtering
n_envs = 4 unless explicitly justified
reset_num_timesteps = False
```

---

## 6. Parent checkpoint and resume semantics

### 6.1 Required parent

Use:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip
```

Do **not** use:

```text
best_by_eval.zip
final.zip
```

unless `checkpoint_500k.zip` is missing and user approves.

### 6.2 Required resume semantics

Must log:

```text
selected_parent_path
selected_parent_sha256
selected_parent_success
selected_parent_collision

model.num_timesteps before learn
reset_num_timesteps = False
optimizer_state_restored = True
n_envs
n_steps
batch_size
learning_rate / lr schedule if available
clip_range / clip schedule if available
```

Before training, assert:

```text
model.num_timesteps == 500000
```

If it is not exactly 500000, either:

```text
1. explain SB3 step rounding / saved timestep semantics；
2. or stop with PHASE_N3Z2CF_STOP_RESUME_SEMANTICS_INVALID.flag。
```

### 6.3 Step accounting

```text
parent_total_steps = 500,000
additional_steps = 1,000,000
target_total_steps = 1,500,000
```

Use global total-step checkpoint names:

```text
checkpoint_750k.zip
checkpoint_1000k.zip
checkpoint_1250k.zip
checkpoint_1500k.zip
final.zip
best_by_eval.zip
```

---

## 7. Checkpoint integrity requirements

Because earlier plots showed suspicious identical metrics between parent and short continuation, this phase must explicitly record checkpoint integrity.

For each checkpoint:

```text
parent_500k
checkpoint_750k
checkpoint_1000k
checkpoint_1250k
checkpoint_1500k
final
best_by_eval
```

must output:

```text
checkpoint_label
checkpoint_path
exists
size_bytes
sha256
global_total_step
model_num_timesteps
parameter_l2_delta_vs_parent
parameter_max_abs_delta_vs_parent
policy_parameter_l2_delta_vs_parent
optimizer_state_present
eval_row_checkpoint_path
```

Output table:

```text
phase_n3z2cf_checkpoint_integrity.csv
```

If `parameter_l2_delta_vs_parent == 0` for a non-parent checkpoint, stop unless there is a clear reason.

---

## 8. Training checkpoints and eval protocol

### 8.1 Training

Train from 500k to 1.5M:

```text
additional_steps = 1,000,000
```

Save:

```text
750k
1000k
1250k
1500k
final
best_by_eval
```

### 8.2 Eval

Evaluate:

```text
parent_500k
corrected_750k
corrected_1000k
corrected_1250k
corrected_1500k
corrected_final
corrected_best_by_eval
```

References:

```text
N3F no_z full final/best_by_eval:
  success = 0.5633
  collision = 0.4367

attention_full_1500k:
  success = 0.6100
  collision = 0.3900

old N3Z2C final:
  success = 0.4500
  collision = 0.5500

audit corrected_750k:
  success = 0.4800
  collision = 0.5200
```

Eval scenarios:

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

---

## 9. Diagnostics

### 9.1 Gψ output diagnostics

Every checkpoint / scenario:

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

Hard requirements:

```text
delta_norm not 1e4 scale
inactive_forwarded_count = 0
feature_nonfinite_count = 0
z_after_constraint_l2_p95 around 4
```

### 9.2 Feature block stats

Every checkpoint / scenario:

```text
obs_i_12 l2_p95
z_i_64_raw l2_p95
z_i_64_after_constraint l2_p95
delta_hat_9_after_scale l2_p95
logvar_hat_9_clamped l2_p95
full_aug_obs l2_p95
max_abs_p95 per block
nan_count
inf_count
```

### 9.3 PPO / action diagnostics

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
episode_min_distance
raw_unsafe_action_rate
raw_safe_margin_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
episode_length
episode_reward
```

Special focus:

```text
1. Does raw unsafe increase after 500k?
2. Does action_delta stay too low?
3. Does Z2 improve high_density / high_speed / high_threat / sudden_threat?
4. Does Z2 still drift after 1000k?
```

---

## 10. Decision rules

### 10.1 Compare corrected Z2 full vs no_z full

Baseline:

```text
no_z full:
  success = 0.5633
  collision = 0.4367
```

Decision:

```text
If corrected Z2 success >= 0.5633 AND collision <= 0.4367:
  corrected Z2 becomes primary N4 candidate.

If corrected Z2 success lower but collision lower:
  keep both no_z and Z2 as N4 candidates; mark tradeoff.

If corrected Z2 success higher but collision higher:
  keep both only if collision increase is small and scenario breakdown supports it;
  otherwise no_z remains primary.

If corrected Z2 is worse on both:
  no_z remains primary N4 candidate;
  Z2 becomes ablation only.
```

### 10.2 Compare with attention_full

Reference:

```text
attention_full:
  success = 0.6100
  collision = 0.3900
```

If neither no_z nor Z2 beats attention:

```text
Do not claim Gψ-PPO no-shield beats attention_full.
```

### 10.3 Can enter N4?

Can enter N4 only if:

```text
1. corrected Z2 full completed；
2. checkpoint integrity is clean；
3. resume semantics are clean；
4. diagnostics are normal；
5. final N4 candidate decision is explicit。
```

---

## 11. Output directory and files

Output directory:

```text
results/env_v2_phase_n3z2cf_corrected_z2_full/
```

Checkpoint directory:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/
```

Required files:

```text
PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md
PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag
phase_n3z2cf_status.txt
phase_n3z2cf_watcher.log
```

Required tables:

```text
phase_n3z2cf_command_manifest.csv
phase_n3z2cf_config_manifest.csv
phase_n3z2cf_resource_affinity.csv
phase_n3z2cf_resume_semantics.csv
phase_n3z2cf_checkpoint_integrity.csv
phase_n3z2cf_train_curve.csv
phase_n3z2cf_train_heartbeat.csv
phase_n3z2cf_checkpoint_eval_summary.csv
phase_n3z2cf_eval_summary.csv
phase_n3z2cf_noz_reference_comparison.csv
phase_n3z2cf_attention_reference_comparison.csv
phase_n3z2cf_old_z2c_comparison.csv
phase_n3z2cf_scenario_breakdown.csv
phase_n3z2cf_motion_mode_breakdown.csv
phase_n3z2cf_threat_class_breakdown.csv
phase_n3z2cf_raw_unsafe_action_summary.csv
phase_n3z2cf_gpsi_output_summary.csv
phase_n3z2cf_feature_block_stats.csv
phase_n3z2cf_final_candidate_decision.csv
phase_n3z2cf_schema_check.csv
```

Required plots:

```text
z2cf_checkpoint_success_collision.png
z2cf_vs_noz_attention_success_collision.png
z2cf_scenario_breakdown.png
z2cf_raw_unsafe_by_checkpoint.png
z2cf_action_dynamics.png
z2cf_feature_block_scale.png
z2cf_gpsi_delta_norm.png
z2cf_gpsi_logvar.png
z2cf_train_reward.png
z2cf_checkpoint_integrity.png
```

---

## 12. Stop flags

Create partial report and stop if needed:

```text
PHASE_N3Z2CF_STOP_PARENT_MISSING.flag
PHASE_N3Z2CF_STOP_RESUME_SEMANTICS_INVALID.flag
PHASE_N3Z2CF_STOP_TRAIN_FAILED.flag
PHASE_N3Z2CF_STOP_EVAL_FAILED.flag
PHASE_N3Z2CF_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3Z2CF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag
PHASE_N3Z2CF_STOP_WATCHER_FAILED.flag
```

Stop examples:

```text
checkpoint_500k parent missing；
model.num_timesteps cannot be reconciled with 500k；
reset_num_timesteps=False not used；
optimizer state not restored；
checkpoint_750k/final parameter delta vs parent is zero；
eval row path does not match checkpoint label；
Gψ diagnostics abnormal；
watcher exits without complete/stop。
```

---

## 13. Completion criteria

Only create:

```text
PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag
```

when all are true:

```text
1. fixed parent checkpoint_500k used；
2. reset_num_timesteps=False confirmed；
3. optimizer state restored；
4. corrected Z2 reaches total 1.5M；
5. required checkpoints saved；
6. checkpoint integrity table generated；
7. eval completed；
8. diagnostics normal；
9. scenario/motion/threat/raw unsafe breakdown generated；
10. final candidate decision generated；
11. report generated；
12. watcher log and status exist。
```

Complete does not necessarily mean Z2 wins. Report must say:

```text
Can enter N4: yes/no
Selected N4 candidate: no_z / corrected_Z2 / both / undecided
```

---

## 14. Suggested commands

Codex should adapt paths.

### 14.1 Compile

```bash
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3z2cf.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3z2cf.py
python -m py_compile scripts/analyze_env_v2_phase_n3z2cf_results.py
bash -n scripts/watch_phase_n3z2cf_corrected_z2_full.sh
chmod +x scripts/watch_phase_n3z2cf_corrected_z2_full.sh
```

### 14.2 Train

```bash
python scripts/train_env_v2_gpsi_ppo_n3z2cf.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml \
  --resume checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0 \
  --parent-total-steps 500000 \
  --additional-steps 1000000 \
  --target-total-steps 1500000 \
  --checkpoint-total-steps 750000 1000000 1250000 1500000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --reset-num-timesteps false \
  --heartbeat-seconds 300
```

### 14.3 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3z2cf.py \
  --result-dir results/env_v2_phase_n3z2cf_corrected_z2_full \
  --checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0 \
  --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --write-traces
```

### 14.4 Analysis

```bash
python scripts/analyze_env_v2_phase_n3z2cf_results.py \
  --result-dir results/env_v2_phase_n3z2cf_corrected_z2_full \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  --old-z2c-success 0.4500 \
  --old-z2c-collision 0.5500
```

### 14.5 Watcher

```bash
bash scripts/watch_phase_n3z2cf_corrected_z2_full.sh
```

---

## 15. Watcher pseudo-code

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3z2cf_corrected_z2_full"
LOG="$OUT_DIR/phase_n3z2cf_watcher.log"
STATUS="$OUT_DIR/phase_n3z2cf_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3Z2CF watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/train_env_v2_gpsi_ppo_n3z2cf.py \
    --config configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml \
    --resume checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0 \
    --parent-total-steps 500000 \
    --additional-steps 1000000 \
    --target-total-steps 1500000 \
    --checkpoint-total-steps 750000 1000000 1250000 1500000 \
    --seed 0 \
    --n-envs 4 \
    --device cpu \
    --reset-num-timesteps false \
    --heartbeat-seconds 300

  python scripts/eval_env_v2_gpsi_ppo_n3z2cf.py \
    --result-dir "$OUT_DIR" \
    --checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0 \
    --eval-seed 1000 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --write-traces

  python scripts/analyze_env_v2_phase_n3z2cf_results.py \
    --result-dir "$OUT_DIR" \
    --noz-success 0.5633 \
    --noz-collision 0.4367 \
    --attention-success 0.6100 \
    --attention-collision 0.3900 \
    --old-z2c-success 0.4500 \
    --old-z2c-collision 0.5500
) 2>&1 | tee -a "$LOG" &
PID=$!

LAST_CHAT_HEARTBEAT=0

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3Z2CF_STOP_PARENT_MISSING.flag \
    PHASE_N3Z2CF_STOP_RESUME_SEMANTICS_INVALID.flag \
    PHASE_N3Z2CF_STOP_TRAIN_FAILED.flag \
    PHASE_N3Z2CF_STOP_EVAL_FAILED.flag \
    PHASE_N3Z2CF_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3Z2CF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag \
    PHASE_N3Z2CF_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3Z2CF_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  # Poll every 120s. Codex should keep chat output concise:
  # about once every 300s unless checkpoint/stop/complete/exception happens.
  sleep 120
done
```

---

## 16. Terminal decision format

Success:

```text
terminal_decision = phase_n3z2cf_corrected_z2_full_complete
```

Stop:

```text
terminal_decision = phase_n3z2cf_stopped_<reason>
```

Must report:

```text
new / modified files
actual commands
parent checkpoint path and sha256
reset_num_timesteps=False confirmation
optimizer state restored confirmation
checkpoint integrity results
corrected Z2 1.5M result
comparison vs no_z full
comparison vs attention_full
scenario/motion/threat breakdown summary
selected N4 candidate
whether N4 can start
if not, next required action
```
