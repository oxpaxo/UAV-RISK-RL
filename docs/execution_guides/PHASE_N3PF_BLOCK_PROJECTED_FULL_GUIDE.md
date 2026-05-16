# Phase N3PF 指南：Block-Projected no_z Full Continuation + Final No-Shield Candidate Decision

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3PF - Block-Projected no_z Full Continuation`  
> 阶段性质：从 N3P `block_projected` 500k checkpoint 做 corrected full continuation 到 1.5M，并与 no_z full / attention_full / corrected Z2 做最终 no-shield candidate 决策。  
> 前置条件：Phase N3P 已完成，`block_projected` 是 P1/P2/P3 500k screening winner。  

---

## 0. 背景与当前结论

Phase N3P 已完成。结果如下：

```text
P1 obs_delta_only @ 500k:
  success = 0.4667
  collision = 0.5333
  hard gate passed

P2 logvar_scaled @ 500k:
  success = 0.3467
  collision = 0.6533
  hard gate failed

P3 block_projected @ 500k:
  success = 0.5333
  collision = 0.4667
  hard gate passed
  current winner
```

关键解释：

```text
1. raw logvar L2 p95 约 14.6–15.0；
2. P2/P3 的 policy logvar_scaled L2 p95 约 2.9–3.0，说明 logvar 尺度修复有效；
3. 但 P2 仍然失败，说明“单纯缩放 logvar”不足；
4. P3 block-wise projection + LayerNorm 显著改善，说明 raw concatenation / block conditioning 是重要瓶颈；
5. Gψ diagnostics bounded，无 wrapper-scale regression；
6. 当前不能进入 N4，需先做 P3 full continuation。
```

必须注意：N3P 中 `block_projected` 的 `checkpoint_500k` 明显优于 `final/best_by_eval`：

```text
block_projected checkpoint_500k:
  success = 0.5333
  collision = 0.4667

block_projected final/best_by_eval:
  success = 0.3967
  collision = 0.6033
```

因此本阶段必须从 `checkpoint_500k.zip` 作为 parent 继续，不得从 `final.zip` 或 `best_by_eval.zip` 继续。

---

## 1. 本阶段目标

本阶段只做一条主线：

```text
P3 block_projected corrected full continuation:
  parent = checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip
  additional_steps ≈ 1,000,000
  target_total_steps ≈ 1,500,000
  reset_num_timesteps = False
```

然后比较：

```text
P3 block_projected full 1.5M
vs
N3F no_z full 1.5M
vs
attention_full 1.5M
vs
corrected Z2 full 1.5M
```

必须回答：

```text
1. P3 full 是否超过 no_z full？
2. P3 full 是否接近或超过 attention_full？
3. P3 full 是否改善 no_z 的短板场景：
   high_density / high_speed / high_threat / sudden_threat？
4. P3 是否解决 raw concatenation / logvar conditioning 对 PPO 的拖累？
5. 最终 N4 应使用：
   P3 block_projected、no_z full、还是两者都作为 ablation？
```

---

## 2. 明确禁止事项

Phase N3PF 禁止：

```text
1. 禁止进入 N4；
2. 禁止实现 shield；
3. 禁止 action filtering / projection；
4. 禁止加入 dense safety cost；
5. 禁止训练或 fine-tune Gψ；
6. 禁止修改 EnvV2-core；
7. 禁止重训 attention_full；
8. 禁止重训 P1/P2；
9. 禁止继续做 z_i variants；
10. 禁止从 block_projected final/best_by_eval 继续；
11. 禁止使用 reset_num_timesteps=True；
12. 禁止覆盖 N3P / N3F / N3Z2CF 原始产物；
13. 禁止只按 train reward 选 final candidate。
```

允许：

```text
1. 使用 repaired GpsiObsWrapper；
2. 使用 P3 block_projected checkpoint_500k parent；
3. 使用 reset_num_timesteps=False；
4. 继续训练 P3 到 total 1.5M；
5. 保存 total-step checkpoints；
6. 做 eval / diagnostics / final candidate decision；
7. 报告是否可以进入 N4。
```

---

## 3. Watcher 与 token 输出要求

本阶段是长训练。为了节省 token：

```text
watcher poll interval: 120 seconds
chat/status heartbeat interval: about 300 seconds or longer
```

要求：

```text
1. watcher 必须阻塞式运行；
2. 日志可以持续写文件；
3. 聊天输出不要每 60 秒刷屏；
4. 除 checkpoint、阶段切换、异常、stop flag、complete flag 外，约 5 分钟输出一次简短状态；
5. 输出内容应压缩为 current total step / target step / fps / ETA / latest checkpoint；
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

本阶段以科学可比性优先：

```text
default n_envs = 4
device = cpu
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

不要为了提速改变 PPO rollout semantics。若 Codex 想 benchmark 更高 `n_envs`，必须非破坏性执行并证明不会改变比较口径；否则保持 `n_envs=4`。

---

## 5. P3 block_projected 配置

### 5.1 Method

```text
method_key: block_projected_full
method: n3pf_block_projected_no_z_full_1p5m
```

### 5.2 Input blocks

```text
obs_i: 12 dims
delta_hat_scaled: 9 dims
logvar_scaled: 9 dims
include_z: false
```

### 5.3 Transforms

```text
delta_hat_scaled = delta_hat / 5.0

logvar_clamped = clamp(logvar, -5, 3)
logvar_scaled = logvar_clamped / 5.0
```

### 5.4 Adapter

Use exactly the N3P block_projected design unless implementation audit finds a mismatch:

```text
obs_i -> Linear(12, 32) -> LayerNorm -> activation
delta_hat_scaled -> Linear(9, 16) -> LayerNorm -> activation
logvar_scaled -> Linear(9, 16) -> LayerNorm -> activation
concat -> 64-dim per-obstacle adapter output
```

Then feed into the same masked-attention-compatible PPO backbone.

Must record:

```text
adapter parameter count
adapter output L2 p95
feature_adapter = block_projected_no_z
obs_aug_dim / adapter_output_dim
```

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
same EnvV2 reward
no shield
no safety cost
no action filtering
n_envs = 4
reset_num_timesteps = False
```

---

## 6. Parent checkpoint and resume semantics

### 6.1 Required parent

Use:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip
```

Do **not** use:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/final.zip
checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/best_by_eval.zip
```

unless `checkpoint_500k.zip` is missing and user explicitly approves.

### 6.2 Required resume semantics

Must log:

```text
selected_parent_path
selected_parent_sha256
selected_parent_success = 0.5333
selected_parent_collision = 0.4667

model.num_timesteps before learn
reset_num_timesteps = False
optimizer_state_restored = True
optimizer_state_entries
n_envs
n_steps
batch_size
learning_rate / lr schedule if available
clip_range / clip schedule if available
```

Because SB3 rollout overshoot may occur, handle parent step as:

```text
expected_parent_total_steps ≈ 500,000
allowed_rollout_overshoot = one rollout quantum
```

If parent `model.num_timesteps` is not exactly 500000, report the exact value and reconcile it with SB3 rollout/checkpoint semantics. If it cannot be reconciled, stop.

### 6.3 Step accounting

```text
parent_total_steps ≈ 500,000
additional_steps ≈ 1,000,000
target_total_steps ≈ 1,500,000
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

If local-step training saves local names, map them explicitly to global total steps.

---

## 7. Checkpoint integrity requirements

Because N3P showed `checkpoint_500k` much better than `final/best_by_eval`, this phase must explicitly record checkpoint integrity and eval path.

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
phase_n3pf_checkpoint_integrity.csv
```

If `parameter_l2_delta_vs_parent == 0` for a non-parent checkpoint, stop unless there is a clear reason.

---

## 8. Training checkpoints and eval protocol

### 8.1 Training

Train from 500k to 1.5M:

```text
additional_steps ≈ 1,000,000
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
N3P block_projected parent 500k:
  success = 0.5333
  collision = 0.4667

N3F no_z full:
  success = 0.5633
  collision = 0.4367

attention_full:
  success = 0.6100
  collision = 0.3900

corrected Z2 full:
  success = 0.5067
  collision = 0.4933

P1 obs_delta_only 500k:
  success = 0.4667
  collision = 0.5333
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

### 9.1 Feature block stats

Every checkpoint / scenario:

```text
obs_i_12 l2_p95
delta_hat_9_after_scale l2_p95
logvar_raw_9_clamped l2_p95
logvar_scaled_9_policy l2_p95
adapter_output_64 l2_p95
full_aug_obs / policy_input l2_p95
max_abs_p95 per block
nan_count
inf_count
```

Expected:

```text
logvar_scaled_l2_p95 ≈ 2.9–3.0
adapter_output_l2_p95 finite and not collapsed
```

### 9.2 Gψ output diagnostics

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
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
episode_length
episode_reward
```

Special focus:

```text
1. Does P3 full improve high_density / high_speed / high_threat / sudden_threat?
2. Does raw unsafe decrease or increase after 500k?
3. Does action_delta remain too low?
4. Does performance drift after 1000k?
5. Which checkpoint is best: 750k/1000k/1250k/1500k/final?
```

---

## 10. Decision rules

### 10.1 Compare P3 full vs no_z full

Reference:

```text
no_z full:
  success = 0.5633
  collision = 0.4367
```

Decision:

```text
If P3 success >= 0.5633 AND collision <= 0.4367:
  P3 becomes primary N4 candidate over no_z.

If P3 success lower but collision lower:
  keep both P3 and no_z as N4 candidates; mark tradeoff.

If P3 success higher but collision higher:
  keep both only if collision increase is small and scenario breakdown supports it;
  otherwise no_z remains primary.

If P3 is worse on both:
  no_z remains primary N4 candidate;
  P3 becomes ablation only.
```

### 10.2 Compare P3 full vs attention_full

Reference:

```text
attention_full:
  success = 0.6100
  collision = 0.3900
```

If P3 does not beat attention:

```text
Do not claim Gψ-PPO no-shield beats attention_full.
```

If P3 beats or matches attention within a small margin:

```text
Mark as strong M1 candidate and proceed to N4 with P3 + same shield comparison.
```

### 10.3 Can enter N4?

Can enter N4 only if:

```text
1. P3 full completed；
2. checkpoint integrity is clean；
3. resume semantics are clean；
4. diagnostics are normal；
5. final N4 candidate decision is explicit。
```

---

## 11. Output directory and files

Output directory:

```text
results/env_v2_phase_n3pf_block_projected_full/
```

Checkpoint directory:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/
```

Required files:

```text
PHASE_N3PF_BLOCK_PROJECTED_FULL_REPORT.md
PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag
phase_n3pf_status.txt
phase_n3pf_watcher.log
```

Required tables:

```text
phase_n3pf_command_manifest.csv
phase_n3pf_config_manifest.csv
phase_n3pf_resource_affinity.csv
phase_n3pf_resume_semantics.csv
phase_n3pf_checkpoint_integrity.csv
phase_n3pf_train_curve.csv
phase_n3pf_train_heartbeat.csv
phase_n3pf_checkpoint_eval_summary.csv
phase_n3pf_eval_summary.csv
phase_n3pf_reference_comparison.csv
phase_n3pf_noz_reference_comparison.csv
phase_n3pf_attention_reference_comparison.csv
phase_n3pf_z2_reference_comparison.csv
phase_n3pf_scenario_breakdown.csv
phase_n3pf_motion_mode_breakdown.csv
phase_n3pf_threat_class_breakdown.csv
phase_n3pf_raw_unsafe_action_summary.csv
phase_n3pf_gpsi_output_summary.csv
phase_n3pf_feature_block_stats.csv
phase_n3pf_final_candidate_decision.csv
phase_n3pf_schema_check.csv
```

Required plots:

```text
n3pf_checkpoint_success_collision.png
n3pf_vs_noz_attention_success_collision.png
n3pf_scenario_breakdown.png
n3pf_raw_unsafe_by_checkpoint.png
n3pf_action_dynamics.png
n3pf_feature_block_scale.png
n3pf_gpsi_delta_norm.png
n3pf_gpsi_logvar.png
n3pf_train_reward.png
n3pf_checkpoint_integrity.png
```

---

## 12. Stop flags

Create partial report and stop if needed:

```text
PHASE_N3PF_STOP_PARENT_MISSING.flag
PHASE_N3PF_STOP_RESUME_SEMANTICS_INVALID.flag
PHASE_N3PF_STOP_CONFIG_MISMATCH.flag
PHASE_N3PF_STOP_TRAIN_FAILED.flag
PHASE_N3PF_STOP_EVAL_FAILED.flag
PHASE_N3PF_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3PF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag
PHASE_N3PF_STOP_WATCHER_FAILED.flag
```

Stop examples:

```text
checkpoint_500k parent missing；
model.num_timesteps cannot be reconciled with 500k；
reset_num_timesteps=False not used；
optimizer state not restored；
block_projected adapter config differs from N3P；
checkpoint_750k/final parameter delta vs parent is zero；
eval row path does not match checkpoint label；
Gψ diagnostics abnormal；
watcher exits without complete/stop。
```

---

## 13. Completion criteria

Only create:

```text
PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag
```

when all are true:

```text
1. fixed parent checkpoint_500k used；
2. block_projected config matches N3P；
3. reset_num_timesteps=False confirmed；
4. optimizer state restored；
5. P3 reaches total ~1.5M；
6. required checkpoints saved；
7. checkpoint integrity table generated；
8. eval completed；
9. diagnostics normal；
10. scenario/motion/threat/raw unsafe breakdown generated；
11. final candidate decision generated；
12. report generated；
13. watcher log and status exist。
```

Complete does not necessarily mean P3 wins. Report must say:

```text
Can enter N4: yes/no
Selected N4 candidate: P3 / no_z / both / undecided
```

---

## 14. Suggested commands

Codex should adapt paths.

### 14.1 Compile

```bash
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3pf.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3pf.py
python -m py_compile scripts/analyze_env_v2_phase_n3pf_results.py
bash -n scripts/watch_phase_n3pf_block_projected_full.sh
chmod +x scripts/watch_phase_n3pf_block_projected_full.sh
```

### 14.2 Train

```bash
python scripts/train_env_v2_gpsi_ppo_n3pf.py   --config configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml   --resume checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0   --parent-total-steps 500000   --additional-steps 1000000   --target-total-steps 1500000   --checkpoint-total-steps 750000 1000000 1250000 1500000   --seed 0   --n-envs 4   --device cpu   --reset-num-timesteps false   --heartbeat-seconds 300
```

### 14.3 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3pf.py   --result-dir results/env_v2_phase_n3pf_block_projected_full   --checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0   --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0   --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip   --eval-seed 1000   --num-episodes 50   --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat   --write-traces
```

### 14.4 Analysis

```bash
python scripts/analyze_env_v2_phase_n3pf_results.py   --result-dir results/env_v2_phase_n3pf_block_projected_full   --noz-success 0.5633   --noz-collision 0.4367   --attention-success 0.6100   --attention-collision 0.3900   --z2-success 0.5067   --z2-collision 0.4933   --parent-success 0.5333   --parent-collision 0.4667
```

### 14.5 Watcher

```bash
bash scripts/watch_phase_n3pf_block_projected_full.sh
```

---

## 15. Watcher pseudo-code

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3pf_block_projected_full"
LOG="$OUT_DIR/phase_n3pf_watcher.log"
STATUS="$OUT_DIR/phase_n3pf_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3PF watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/train_env_v2_gpsi_ppo_n3pf.py     --config configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml     --resume checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip     --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0     --parent-total-steps 500000     --additional-steps 1000000     --target-total-steps 1500000     --checkpoint-total-steps 750000 1000000 1250000 1500000     --seed 0     --n-envs 4     --device cpu     --reset-num-timesteps false     --heartbeat-seconds 300

  python scripts/eval_env_v2_gpsi_ppo_n3pf.py     --result-dir "$OUT_DIR"     --checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0     --eval-seed 1000     --num-episodes 50     --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat     --write-traces

  python scripts/analyze_env_v2_phase_n3pf_results.py     --result-dir "$OUT_DIR"     --noz-success 0.5633     --noz-collision 0.4367     --attention-success 0.6100     --attention-collision 0.3900     --z2-success 0.5067     --z2-collision 0.4933     --parent-success 0.5333     --parent-collision 0.4667
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in     PHASE_N3PF_STOP_PARENT_MISSING.flag     PHASE_N3PF_STOP_RESUME_SEMANTICS_INVALID.flag     PHASE_N3PF_STOP_CONFIG_MISMATCH.flag     PHASE_N3PF_STOP_TRAIN_FAILED.flag     PHASE_N3PF_STOP_EVAL_FAILED.flag     PHASE_N3PF_STOP_DIAGNOSTICS_FAILED.flag     PHASE_N3PF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag     PHASE_N3PF_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3PF_BLOCK_PROJECTED_FULL_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3PF_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  sleep 120
done
```

---

## 16. Terminal decision format

Success:

```text
terminal_decision = phase_n3pf_block_projected_full_complete
```

Stop:

```text
terminal_decision = phase_n3pf_stopped_<reason>
```

Must report:

```text
new / modified files
actual commands
parent checkpoint path and sha256
reset_num_timesteps=False confirmation
optimizer state restored confirmation
checkpoint integrity results
P3 full 1.5M result
comparison vs no_z full
comparison vs attention_full
comparison vs corrected Z2
scenario/motion/threat breakdown summary
selected N4 candidate
whether N4 can start
if not, next required action
```
