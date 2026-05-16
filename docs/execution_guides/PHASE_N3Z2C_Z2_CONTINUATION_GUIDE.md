# Phase N3Z2C 指南：z_layernorm_alpha_0p5 1.5M Continuation + Final Candidate Decision

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3Z2C - Z2 1.5M Continuation and Final No-Shield Candidate Decision`  
> 阶段性质：Z2 续训到 1.5M + 与 no_z full / attention_full 做最终 no-shield 候选比较；不接 shield；不训练 / fine-tune Gψ；不改 EnvV2-core。  
> 前置条件：Phase N3F/Z 已完成，`z_layernorm_alpha_0p5` 通过 hard gate，no_z full 已跑到 1.5M。  

---

## 0. 背景与当前结论

Phase N3F/Z 已完成，核心结果：

```text
n3f_no_z_full:
  success = 0.5633
  collision = 0.4367

z_l2_scale_4:
  success = 0.2933
  collision = 0.7067
  hard gate failed

z_layernorm_alpha_0p5:
  success = 0.4700
  collision = 0.5300
  hard gate passed
```

Phase N3F/Z 的 hard gate 来自 N3R no_z 500k：

```text
success >= 0.4233
collision <= 0.5767
```

Z2 `z_layernorm_alpha_0p5` 通过了这两个条件，因此不能再直接砍掉 z_i。当前判断是：

```text
1. no_z full 是一个有效、干净、接近 attention_full 的 no-shield Gψ-PPO baseline；
2. Z1 z_l2_scale_4 明确失败，可以砍掉；
3. Z2 z_layernorm_alpha_0p5 通过筛查，必须续训到 1.5M；
4. 当前不能进入 N4；需要先比较 Z2 1.5M 与 no_z full 1.5M。
```

本阶段的目标是把 Z2 续训到 1.5M，并给出最终 no-shield candidate 决策。

---

## 1. Phase N3Z2C 总目标

本阶段只做一件主事：

```text
将 z_layernorm_alpha_0p5 从 500k continuation 到 total 1.5M。
```

然后比较：

```text
Z2 1.5M
vs
N3F no_z full 1.5M
vs
attention_full 1.5M reference
```

必须回答：

```text
1. Z2 1.5M 是否超过 no_z full？
2. Z2 1.5M 是否接近或超过 attention_full？
3. Z2 是否改善 no_z 的短板场景：
   high_density / high_speed / high_threat / sudden_threat？
4. Z2 的 raw unsafe / action dynamics / feature scale 是否正常？
5. 最终 N4 应使用哪个 no-shield policy candidate：
   no_z full, Z2 full, or both as ablation？
```

---

## 2. 明确禁止事项

Phase N3Z2C 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止训练或 fine-tune Gψ；
3. 禁止实现 / 接入 safety shield；
4. 禁止 action filtering / projection；
5. 禁止加入 dense safety cost；
6. 禁止使用 learned R(s,a)；
7. 禁止使用 candidate velocity risk map as PPO input；
8. 禁止回到 5-head Gψ；
9. 禁止重训 Z1；
10. 禁止改变 no_z 已完成结果；
11. 禁止覆盖 N3F/Z artifacts；
12. 禁止只按 train reward 决策，必须以 eval success/collision 和 breakdown 为准。
```

允许：

```text
1. 使用 repaired GpsiObsWrapper；
2. 续训 Z2 z_layernorm_alpha_0p5 到 total 1.5M；
3. 保留 Z2 feature transform：
   z_used = 0.5 * LayerNorm(z_i)
4. 使用 logvar clamp [-5, 3] 或等价更紧 bounded setting；
5. 评估 Z2 checkpoints；
6. 与 no_z full / attention_full / Z2 500k 对比；
7. 输出 report / tables / plots / complete flag。
```

---

## 3. CPU / 资源利用要求

本项目当前在以下固定机器上运行：

```text
OS: Ubuntu 22.04.5 LTS
Kernel: Linux 5.15.0-113-generic x86_64
CPU: AMD EPYC 7402 24-Core Processor
CPU cores: 8 physical / 16 logical
RAM: 62 GiB
GPU: NVIDIA RTX 3090, 24 GiB VRAM
```

当前主要问题之一是 CPU 使用率偏低，训练吞吐不足。因此 Codex 在本阶段必须加入资源利用审计和安全提速策略。

### 3.1 安全优先级

性能优化不得破坏：

```text
1. watcher 阻塞式运行；
2. checkpoint 保存；
3. eval / analysis 完整性；
4. 训练可恢复性；
5. 日志持续输出；
6. PPO 训练稳定性；
7. 科学可比性。
```

禁止：

```text
1. 无监控地后台启动训练；
2. 盲目开过多并行进程；
3. 同时跑多个会抢 CPU/GPU/IO 的正式训练；
4. 为了提速改 EnvV2-core；
5. 为了提速关闭 checkpoint / heartbeat；
6. 因 CPU 优化导致 watcher 提前退出。
```

### 3.2 必须做的资源 preflight

训练前必须记录：

```text
nproc
lscpu summary
free -h
df -h /
nvidia-smi
current git diff summary if available
```

并写入：

```text
phase_n3z2c_resource_preflight.csv
phase_n3z2c_command_manifest.csv
```

### 3.3 n_envs / CPU 利用策略

Phase N3F/Z 的 command manifest 显示此前训练多使用：

```text
--n-envs 4
```

本阶段要尽可能提高 CPU 利用率，但不能牺牲可比性。Codex 必须执行以下逻辑：

```text
1. 首先确认 Z2 continuation 是否能安全 resume；
2. 优先保持 PPO 超参和 Z2 config 不变；
3. 对 n_envs 做 non-destructive throughput smoke benchmark；
4. 如果提高 n_envs 会改变 PPO rollout semantics 且无法保持等效 batch，则不要强行改变；
5. 如果代码支持保持 effective rollout batch / batch_size 语义稳定，且 smoke 通过，可选择更高 n_envs；
6. 最终选择必须在 report 里说明：
   - selected n_envs
   - reason
   - benchmark fps
   - CPU utilization estimate
   - whether PPO semantics changed
```

推荐候选：

```text
n_envs candidates:
  4, 8, 12

Do not use 16 unless smoke confirms stable and memory/IO are safe.
```

建议环境变量：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

理由：避免每个 env worker 内部再开多线程造成 oversubscription。若实测变慢，必须记录并可回退。

### 3.4 benchmark 方式

允许进行短 smoke benchmark：

```text
5k–20k steps
no overwrite official checkpoints
separate temp output dir
heartbeat enabled
```

必须比较：

```text
fps
CPU utilization rough estimate
GPU utilization rough estimate
no NaN / inf
checkpoint write success
watcher log success
```

如果 benchmark 不稳定，回退到：

```text
n_envs = 4
```

科学结果优先，速度第二。

---

## 4. Z2 continuation 配置

### 4.1 Config

```text
method_key: z_layernorm_alpha_0p5_cont_1p5m

input:
  obs_i: 12 dims
  z_layernorm_scaled: 64 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

z transform:
  z_ln = LayerNorm(z_i)
  z_used = 0.5 * z_ln

total obstacle dim:
  94
```

### 4.2 Gψ

```text
checkpoint = work_dirs/gpsi_heada_v1_nll/best.pth
Gψ frozen = true
Gψ eval() = true
requires_grad_any = false
```

### 4.3 PPO

```text
PPO trainable
same PPO backbone as N3F/Z Z2
same EnvV2 reward
no shield
no safety cost
no action filtering
```

### 4.4 logvar clip

Use existing bounded setting:

```text
logvar_clamp = [-5, 3]
```

This is already tighter than:

```text
|logvar| <= 5
```

Report must state:

```text
logvar clip sanity: already bounded tighter than abs5.
```

---

## 5. Resume / parent checkpoint selection

Z2 continuation must start from an existing Z2 500k checkpoint unless there is a clear technical reason to restart.

Candidate parents:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip
```

Codex must inspect available checkpoints and choose parent by this rule:

```text
1. Prefer best_by_eval if it exists and its eval result is not worse than final in both success and collision;
2. Otherwise use final.zip;
3. If final.zip missing, use checkpoint_500k.zip;
4. Record chosen parent and reason.
```

Report must clarify the step accounting:

```text
parent_total_steps = 500k
additional_steps = 1,000,000
target_total_steps = 1,500,000
```

If the training script reports local continuation steps instead of global total steps, filenames and metadata must explicitly store both:

```text
local_step
total_step
```

---

## 6. Checkpoints and eval protocol

### 6.1 Required checkpoints

Z2 continuation must save at total step:

```text
750k
1000k
1250k
1500k
final
best_by_eval
```

If implementation only supports local checkpointing, map:

```text
local 250k -> total 750k
local 500k -> total 1000k
local 750k -> total 1250k
local 1000k -> total 1500k
```

### 6.2 Required eval

Evaluate:

```text
Z2 parent 500k
Z2 750k
Z2 1000k
Z2 1250k
Z2 1500k
Z2 final
Z2 best_by_eval
```

Reference comparisons:

```text
N3F no_z full final/best_by_eval
attention_full_1500k
N3F/Z Z2 500k screening result
N3R no_z 500k gate
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

### 6.3 Checkpoint/final口径

N3F/Z 里出现过 checkpoint_1500k 与 final/best_by_eval 数值不同的问题。N3Z2C 必须明确：

```text
1. checkpoint_1500k 保存时机；
2. final.zip 保存时机；
3. best_by_eval.zip 的选择依据；
4. 每个 eval row 对应哪个 checkpoint file；
5. final / best_by_eval 是否复用同一参数；
6. 不得混用 checkpoint_1500k 和 final 的指标。
```

---

## 7. Diagnostics 必须保留

### 7.1 Gψ output diagnostics

每个 checkpoint / scenario：

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

Hard sanity：

```text
delta_norm 不得回到 1e4；
logvar 不得无解释全部恒定；
projected_std 不得无解释恒定；
inactive_forwarded_count 必须为 0。
```

### 7.2 Feature block stats

每个 checkpoint / scenario：

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

For Z2, expected:

```text
z_i_64_after_constraint l2_p95 ≈ 4
```

### 7.3 PPO / action diagnostics

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
```

Special focus:

```text
1. high_density / high_speed / high_threat / sudden_threat；
2. raw unsafe low but collision high；
3. action_delta too low causing delayed reaction；
4. whether Z2 improves no_z weaknesses.
```

---

## 8. Decision rules after Z2 continuation

### 8.1 Primary comparison

Compare final / best_by_eval Z2 1.5M against no_z full:

```text
no_z full:
  success = 0.5633
  collision = 0.4367
```

Decision:

```text
If Z2 success >= no_z success AND Z2 collision <= no_z collision:
  Z2 becomes primary N4 Gψ-PPO policy candidate.

If Z2 improves one metric but worsens the other:
  mark as tradeoff;
  default to lower collision for shield safety path,
  but report must preserve both candidates.

If Z2 is worse on both success and collision:
  no_z remains primary N4 Gψ-PPO candidate;
  Z2 becomes ablation only.
```

### 8.2 Attention comparison

Compare against attention_full:

```text
attention_full:
  success = 0.6100
  collision = 0.3900
```

If Z2 or no_z still below attention_full:

```text
Do not claim Gψ-PPO no-shield beats attention.
```

Instead write:

```text
Gψ-PPO no-shield is close / weaker / scenario-dependent.
```

### 8.3 N4 readiness

Can enter N4 only if:

```text
1. Z2 continuation complete;
2. final candidate selected;
3. diagnostics normal;
4. no unresolved checkpoint/eval口径 issue;
5. report clearly states whether N4 should use no_z, Z2, or both.
```

---

## 9. Recommended output files

Suggested new / modified files:

```text
configs/
  env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml

scripts/
  train_env_v2_gpsi_ppo_n3z2c.py
  eval_env_v2_gpsi_ppo_n3z2c.py
  analyze_env_v2_phase_n3z2c_results.py
  watch_phase_n3z2c_z2_continuation.sh
```

You may reuse N3F/Z scripts if safer, but must not overwrite N3F/Z outputs.

Output directory:

```text
results/env_v2_phase_n3z2c_z2_continuation/
  PHASE_N3Z2C_Z2_CONTINUATION_REPORT.md
  PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag
  phase_n3z2c_status.txt
  phase_n3z2c_watcher.log

  tables/
    phase_n3z2c_resource_preflight.csv
    phase_n3z2c_cpu_benchmark.csv
    phase_n3z2c_config_manifest.csv
    phase_n3z2c_parent_checkpoint_selection.csv
    phase_n3z2c_train_curve.csv
    phase_n3z2c_train_heartbeat.csv
    phase_n3z2c_checkpoint_eval_summary.csv
    phase_n3z2c_eval_summary.csv
    phase_n3z2c_attention_reference_comparison.csv
    phase_n3z2c_noz_reference_comparison.csv
    phase_n3z2c_scenario_breakdown.csv
    phase_n3z2c_motion_mode_breakdown.csv
    phase_n3z2c_threat_class_breakdown.csv
    phase_n3z2c_raw_unsafe_action_summary.csv
    phase_n3z2c_gpsi_output_summary.csv
    phase_n3z2c_aug_feature_block_stats.csv
    phase_n3z2c_final_candidate_decision.csv
    phase_n3z2c_command_manifest.csv
    phase_n3z2c_schema_check.csv

  plots/
    z2_checkpoint_success_collision.png
    z2_vs_noz_attention_success_collision.png
    z2_scenario_breakdown.png
    z2_raw_unsafe_by_checkpoint.png
    z2_action_dynamics.png
    z2_aug_feature_block_scale.png
    z2_gpsi_delta_norm.png
    z2_gpsi_logvar.png
    z2_train_reward.png

  logs/
    phase_n3z2c_train_z2_continuation.log
    phase_n3z2c_eval.log
    phase_n3z2c_analysis.log
```

Checkpoint directory:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0/
  checkpoint_750k.zip
  checkpoint_1000k.zip
  checkpoint_1250k.zip
  checkpoint_1500k.zip
  final.zip
  best_by_eval.zip
```

---

## 10. Command examples

Codex should adjust paths to actual repo.

### 10.1 Compile

```bash
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3z2c.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3z2c.py
python -m py_compile scripts/analyze_env_v2_phase_n3z2c_results.py
bash -n scripts/watch_phase_n3z2c_z2_continuation.sh
chmod +x scripts/watch_phase_n3z2c_z2_continuation.sh
```

### 10.2 Resource preflight

```bash
nproc
lscpu
free -h
df -h /
nvidia-smi
```

### 10.3 Optional safe throughput benchmark

```bash
python scripts/train_env_v2_gpsi_ppo_n3z2c.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
  --resume checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_benchmark_nenv4 \
  --train-steps 10000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --benchmark-only \
  --heartbeat-seconds 10
```

Repeat for n_envs 8 / 12 only if safe and non-overwriting.

### 10.4 Z2 continuation

```bash
python scripts/train_env_v2_gpsi_ppo_n3z2c.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
  --resume <selected_z2_500k_parent_checkpoint> \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0 \
  --additional-steps 1000000 \
  --target-total-steps 1500000 \
  --checkpoint-total-steps 750000 1000000 1250000 1500000 \
  --seed 0 \
  --n-envs <selected_safe_n_envs> \
  --device <selected_safe_device> \
  --heartbeat-seconds 30
```

### 10.5 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3z2c.py \
  --result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --z2-checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0 \
  --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --write-traces
```

### 10.6 Analysis

```bash
python scripts/analyze_env_v2_phase_n3z2c_results.py \
  --result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900
```

### 10.7 Watcher

```bash
bash scripts/watch_phase_n3z2c_z2_continuation.sh
```

---

## 11. Completion criteria

Only create:

```text
PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag
```

when all are true:

```text
1. Phase N3F/Z complete flag exists;
2. Z2 parent checkpoint exists and parent selection is recorded;
3. resource preflight is recorded;
4. throughput benchmark is either completed or explicitly skipped for safety/comparability;
5. Z2 continuation reaches total 1.5M;
6. required checkpoints are saved;
7. eval completes;
8. Gψ diagnostics normal;
9. feature block stats generated;
10. raw unsafe/action diagnostics generated;
11. scenario/motion/threat breakdown generated;
12. checkpoint/final/best_by_eval口径 documented;
13. final candidate decision generated;
14. report generated;
15. watcher log and status exist.
```

Complete does not necessarily mean N4 can start. Report must explicitly state:

```text
Can enter N4: yes/no
Selected N4 candidate: no_z / Z2 / both / undecided
```

---

## 12. Stop flags

Create partial report + log if stopped.

```text
PHASE_N3Z2C_STOP_PHASE_N3FZ_MISSING.flag
PHASE_N3Z2C_STOP_Z2_PARENT_MISSING.flag
PHASE_N3Z2C_STOP_RESOURCE_BENCHMARK_FAILED.flag
PHASE_N3Z2C_STOP_WRAPPER_SCALE_INVALID.flag
PHASE_N3Z2C_STOP_TRAIN_FAILED.flag
PHASE_N3Z2C_STOP_EVAL_FAILED.flag
PHASE_N3Z2C_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3Z2C_STOP_CHECKPOINT_AMBIGUITY.flag
PHASE_N3Z2C_STOP_WATCHER_FAILED.flag
```

Stop examples:

```text
Phase N3F/Z complete flag missing;
Z2 parent checkpoint missing;
wrapper outputs delta万级 again;
training cannot resume safely;
eval fails;
diagnostics missing;
checkpoint_1500k/final/best_by_eval cannot be mapped;
watcher fails.
```

---

## 13. Watcher requirements

Create:

```text
scripts/watch_phase_n3z2c_z2_continuation.sh
```

Watcher must:

```text
1. check N3F/Z complete flag;
2. run resource preflight;
3. select parent checkpoint;
4. optionally run safe CPU/n_envs benchmark;
5. train Z2 continuation;
6. eval checkpoints;
7. run analysis;
8. write report;
9. poll complete / stop flags;
10. continue output until complete or stop;
11. never exit just because logs are temporarily quiet;
12. write phase_n3z2c_watcher.log;
13. write phase_n3z2c_status.txt.
```

Pseudo-code:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3z2c_z2_continuation"
LOG="$OUT_DIR/phase_n3z2c_watcher.log"
STATUS="$OUT_DIR/phase_n3z2c_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3Z2C watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3Z2C_STOP_PHASE_N3FZ_MISSING.flag"
fi

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  # resource preflight
  nproc | tee -a "$OUT_DIR/logs/phase_n3z2c_resource_preflight.log"
  lscpu | tee -a "$OUT_DIR/logs/phase_n3z2c_resource_preflight.log"
  free -h | tee -a "$OUT_DIR/logs/phase_n3z2c_resource_preflight.log"
  df -h / | tee -a "$OUT_DIR/logs/phase_n3z2c_resource_preflight.log"
  nvidia-smi | tee -a "$OUT_DIR/logs/phase_n3z2c_resource_preflight.log"

  # Codex should implement parent selection and optional benchmark here.

  python scripts/train_env_v2_gpsi_ppo_n3z2c.py \
    --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
    --resume "<selected_z2_parent>" \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0 \
    --additional-steps 1000000 \
    --target-total-steps 1500000 \
    --checkpoint-total-steps 750000 1000000 1250000 1500000 \
    --seed 0 \
    --n-envs "<selected_safe_n_envs>" \
    --device "<selected_safe_device>" \
    --heartbeat-seconds 30

  python scripts/eval_env_v2_gpsi_ppo_n3z2c.py \
    --result-dir "$OUT_DIR" \
    --z2-checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0 \
    --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
    --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
    --eval-seed 1000 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --write-traces

  python scripts/analyze_env_v2_phase_n3z2c_results.py \
    --result-dir "$OUT_DIR" \
    --noz-success 0.5633 \
    --noz-collision 0.4367 \
    --attention-success 0.6100 \
    --attention-collision 0.3900
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3Z2C_STOP_PHASE_N3FZ_MISSING.flag \
    PHASE_N3Z2C_STOP_Z2_PARENT_MISSING.flag \
    PHASE_N3Z2C_STOP_RESOURCE_BENCHMARK_FAILED.flag \
    PHASE_N3Z2C_STOP_WRAPPER_SCALE_INVALID.flag \
    PHASE_N3Z2C_STOP_TRAIN_FAILED.flag \
    PHASE_N3Z2C_STOP_EVAL_FAILED.flag \
    PHASE_N3Z2C_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3Z2C_STOP_CHECKPOINT_AMBIGUITY.flag \
    PHASE_N3Z2C_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3Z2C_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  echo "[watcher] still running..." | tee -a "$LOG"
  sleep 60
done
```

Codex may adjust implementation, but blocking semantics are mandatory.

---

## 14. Terminal decision

Success:

```text
terminal_decision = phase_n3z2c_z2_continuation_complete
```

Stop:

```text
terminal_decision = phase_n3z2c_stopped_<reason>
```

Must report:

```text
new / modified files
actual commands
selected parent checkpoint
selected n_envs / CPU strategy
Z2 1.5M result
comparison with no_z full
comparison with attention_full
checkpoint/final口径
candidate decision
whether N4 can start
if N4 cannot start, what next
```
