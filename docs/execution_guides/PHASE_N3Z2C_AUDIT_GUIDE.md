# Phase N3Z2C-Audit 指南：Parent / Resume / CPU-Affinity Audit + Short Corrected Continuation Sanity

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3Z2C-Audit - Z2 Continuation Audit and Short Sanity`  
> 阶段性质：工程审计 + 短续训 sanity；不进入 N4；不实现 shield；不训练 / fine-tune Gψ；不改 EnvV2-core。  
> 前置条件：Phase N3Z2C 已完成，但 Z2 continuation 结果和过程口径存在疑点。  

---

## 0. 背景与问题

Phase N3F/Z 的结论：

```text
n3f_no_z_full:
  success = 0.5633
  collision = 0.4367

z_layernorm_alpha_0p5 500k:
  success ≈ 0.4700 / 0.4800
  collision ≈ 0.5300 / 0.5200
  passed hard gate
```

Phase N3Z2C 将 Z2 `z_layernorm_alpha_0p5` 续训到 total 1.5M，但结果不理想：

```text
Z2 parent_500k:
  success ≈ 0.4700
  collision ≈ 0.5300

Z2 final 1.5M:
  success = 0.4500
  collision = 0.5500

no_z full 1.5M:
  success = 0.5633
  collision = 0.4367

attention_full_1500k:
  success = 0.6100
  collision = 0.3900
```

从 N3Z2C report 和表格中看，Gψ forward diagnostics 没有复现 N3 原始 wrapper 爆炸问题：

```text
delta_norm_1s_p95 正常；
inactive_forwarded_count = 0；
z_after_constraint_l2_p95 ≈ 4；
feature_nonfinite_count = 0。
```

但 N3Z2C 仍有两个明显疑点：

```text
1. parent selection 不严谨：
   checkpoint_500k 的 success/collision 优于 final/best_by_eval，
   但 Codex 选择了 best_by_eval/final 作为 parent。

2. resource preflight 中 nproc=1，但 lscpu 显示 16 CPUs：
   需要确认是 cgroup/affinity 限制，还是记录方式异常。
```

此外，N3Z2C 作为 continuation 还必须检查：

```text
1. SB3 learn() 是否 reset_num_timesteps=False；
2. optimizer state / lr schedule / clip schedule 是否完整恢复；
3. checkpoint_500k / final / best_by_eval 文件是否相同；
4. checkpoint/eval label 是否映射正确；
5. corrected parent 短续训后是否仍快速退化。
```

本阶段的目标是：**不直接重跑完整 1.5M，不进入 N4；先阻塞式审计这些过程问题，并从 corrected parent 做短 continuation sanity。**

---

## 1. Phase N3Z2C-Audit 总目标

必须回答：

```text
Q1: N3Z2C 的 parent selection 是否错误？
Q2: Z2 continuation 的 resume / scheduler / optimizer state 是否正确？
Q3: CPU 可用核数为什么 nproc=1，但 lscpu=16？
Q4: 从真正最优 500k parent 做短 continuation，是否仍出现性能退化？
Q5: 当前能否接受 N3Z2C 的结论：no_z_full 是 N4 candidate，Z2 只保留为 ablation？
Q6: 或者是否必须修复后重跑 Z2 continuation？
```

本阶段只做：

```text
1. static audit；
2. checkpoint hash / parent selection audit；
3. resume semantics audit；
4. CPU affinity / cgroup audit；
5. corrected parent short continuation sanity；
6. eval / comparison / report。
```

---

## 2. 明确禁止事项

Phase N3Z2C-Audit 禁止：

```text
1. 禁止进入 N4；
2. 禁止实现 shield；
3. 禁止训练或 fine-tune Gψ；
4. 禁止修改 EnvV2-core；
5. 禁止重跑完整 Z2 1.5M；
6. 禁止重训 no_z；
7. 禁止覆盖 N3Z2C / N3FZ / N3R 原始产物；
8. 禁止只因短 sanity reward 好看就改最终结论；
9. 禁止忽略 checkpoint parent 选择错误；
10. 禁止频繁 watcher 输出刷屏。
```

允许：

```text
1. 新增审计脚本；
2. 修复 parent selection 规则；
3. 增加 resume / scheduler / reset_num_timesteps 日志；
4. 从 corrected checkpoint_500k parent 做短 continuation sanity；
5. 对 corrected short continuation 做 eval；
6. 输出是否需要重跑 Z2 continuation 的决策。
```

---

## 3. Watcher 输出频率要求

本阶段有短续训，但不是极长训练。默认：

```text
watcher poll interval: 120 seconds
chat/status heartbeat interval: 300 seconds
```

含义：

```text
1. watcher 可以每 120 秒检查 complete/stop flag；
2. 不要每次 poll 都向聊天输出长状态；
3. 除 checkpoint、阶段切换、异常、stop flag、complete flag 外，约 5 分钟输出一次简短状态即可；
4. 日志文件可以持续写，不要用聊天刷屏消耗 token。
```

如短 smoke test 阶段出错，可临时更频繁输出；正式 sanity 训练阶段应节制输出。

---

## 4. CPU / 资源审计要求

当前硬件预期：

```text
OS: Ubuntu 22.04.5 LTS
CPU: AMD EPYC 7402
physical cores: 8
logical CPUs: 16
RAM: 62 GiB
GPU: RTX 3090 24 GiB
```

N3Z2C report 出现：

```text
nproc = 1
lscpu CPU(s) = 16
```

本阶段必须解释该冲突。

运行并记录：

```bash
nproc
nproc --all
lscpu
taskset -pc $$
python - <<'PY'
import os
print("os.cpu_count", os.cpu_count())
print("affinity_count", len(os.sched_getaffinity(0)))
print("affinity", sorted(os.sched_getaffinity(0)))
PY
cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || true
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
free -h
df -h /
nvidia-smi
```

必须输出：

```text
phase_n3z2c_audit_resource_affinity.csv
```

如果 `sched_getaffinity` 只有 1 个 CPU，则 report 必须写明：

```text
当前进程实际只获得 1 CPU affinity/cgroup 配额；
Codex 无法在该执行上下文中榨干 16 logical CPUs；
需要用户或运行环境解除 CPU affinity/cgroup 限制。
```

如果 affinity 有 16，但 `nproc=1`，需要查明是命令执行方式 / env 问题。

---

## 5. Parent selection audit

### 5.1 必须检查的候选

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip
checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip
```

### 5.2 必须输出

```text
phase_n3z2c_audit_checkpoint_hash.csv
phase_n3z2c_audit_parent_selection_fixed.csv
```

字段：

```text
candidate
path
exists
size_bytes
sha256
eval_label
success_rate
collision_rate
near_miss_rate
raw_unsafe_action_rate
selected_by_old_rule
selected_by_fixed_rule
selection_reason
```

### 5.3 修正后的选择规则

在所有存在且有 eval record 的 parent candidates 中选择：

```text
1. 先排除 diagnostics 不正常的 checkpoint；
2. 优先 success_rate 更高；
3. success_rate 相同时 collision_rate 更低；
4. collision_rate 相同时 near_miss_rate 更低；
5. 再看 raw_unsafe_action_rate；
6. 如果存在 Pareto tradeoff，需要写明 tradeoff 并默认选择 lower collision 版本。
```

当前预期：

```text
checkpoint_500k 应该被 fixed rule 选中。
```

如果不是，必须解释原因。

---

## 6. Resume semantics audit

必须审计训练脚本和训练日志，确认：

```text
1. SB3 load 是否完整加载 optimizer state；
2. learn() 是否使用 reset_num_timesteps=False；
3. additional_steps / target_total_steps / parent_total_steps 语义是否正确；
4. checkpoint filenames 是否使用 global total step；
5. learning_rate schedule 是否从正确 progress_remaining 继续；
6. clip_range schedule 是否从正确 progress_remaining 继续；
7. n_envs / n_steps / batch_size 是否与 N3FZ Z2 原训练一致；
8. VecNormalize / normalization stats 是否存在且正确恢复，如果项目使用了它；
9. resumed model 的 policy architecture / obs dim / z transform 是否与 Z2 完全一致。
```

必须输出：

```text
phase_n3z2c_audit_resume_semantics.csv
phase_n3z2c_audit_train_script_findings.csv
```

如果无法从代码或日志确认 reset/scheduler/optimizer state，需要 report 写成：

```text
resume semantics unresolved
```

并触发 stop 或 warning，视严重程度决定。

建议 Codex 在训练脚本中增加只读/日志型输出：

```text
loaded_checkpoint_path
parent_total_steps
reset_num_timesteps
model.num_timesteps before learn
model.num_timesteps after learn
learning_rate current
clip_range current
n_envs
n_steps
batch_size
obs_dim
```

不得改变训练逻辑，除非发现确定 bug。

---

## 7. Corrected short continuation sanity

### 7.1 目的

从 corrected parent，也就是 fixed rule 选择的 parent，做短续训，判断 Z2 是否仍快速退化。

### 7.2 默认训练长度

```text
additional_steps = 250,000
parent_total_steps = 500,000
target_total_steps = 750,000
```

如果资源或时间非常紧张，可先做：

```text
additional_steps = 100,000
```

但正式 sanity 推荐 250k，因为 N3Z2C 里已有 old selected-parent 750k，可直接比较。

### 7.3 输出目录

不得覆盖 N3Z2C：

```text
checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/
results/env_v2_phase_n3z2c_audit/
```

### 7.4 训练配置

必须与 Z2 保持一致：

```text
z_transform = layernorm
alpha = 0.5
Gψ frozen
no shield
no safety cost
no action filtering
EnvV2-core unchanged
n_envs = same as original Z2 unless audit proves safe and equivalent change
device = cpu unless original used otherwise
```

### 7.5 Eval

评估：

```text
corrected_parent_500k
corrected_cont_600k if 100k eval exists
corrected_cont_750k
old_N3Z2C_parent_500k
old_N3Z2C_750k
N3F no_z full final
attention_full
```

默认：

```text
6 scenarios × 50 episodes
eval_seed = 1000
```

如果先做 100k smoke，可以用：

```text
6 scenarios × 20 episodes
```

但 final sanity report 仍建议 50 episodes。

### 7.6 判断逻辑

如果 corrected 750k：

```text
success >= old_N3Z2C_750k success
AND collision <= old_N3Z2C_750k collision
```

说明 corrected parent 至少不差。

如果 corrected 750k 接近或超过 Z2 parent 500k：

```text
Z2 may still be viable; consider corrected full continuation.
```

如果 corrected 750k 仍明显退化：

```text
Z2 long-training instability likely real;
accept no_z_full as N4 candidate.
```

---

## 8. Required tables / plots

输出目录：

```text
results/env_v2_phase_n3z2c_audit/
```

必须包含：

```text
PHASE_N3Z2C_AUDIT_REPORT.md
PHASE_N3Z2C_AUDIT_COMPLETE.flag
phase_n3z2c_audit_status.txt
phase_n3z2c_audit_watcher.log
```

Tables：

```text
phase_n3z2c_audit_resource_affinity.csv
phase_n3z2c_audit_checkpoint_hash.csv
phase_n3z2c_audit_parent_selection_fixed.csv
phase_n3z2c_audit_resume_semantics.csv
phase_n3z2c_audit_train_script_findings.csv
phase_n3z2c_audit_command_manifest.csv
phase_n3z2c_audit_short_continuation_train_curve.csv
phase_n3z2c_audit_short_continuation_heartbeat.csv
phase_n3z2c_audit_checkpoint_eval_summary.csv
phase_n3z2c_audit_eval_summary.csv
phase_n3z2c_audit_scenario_breakdown.csv
phase_n3z2c_audit_raw_unsafe_summary.csv
phase_n3z2c_audit_gpsi_output_summary.csv
phase_n3z2c_audit_feature_block_stats.csv
phase_n3z2c_audit_decision.csv
```

Plots：

```text
audit_parent_selection_comparison.png
audit_corrected_vs_old_750k_success_collision.png
audit_short_continuation_curve.png
audit_raw_unsafe_comparison.png
audit_scenario_breakdown.png
audit_feature_block_scale.png
```

---

## 9. Stop flags

出现以下问题必须 stop 或 partial report：

```text
PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag
PHASE_N3Z2C_AUDIT_STOP_PARENT_CANDIDATES_MISSING.flag
PHASE_N3Z2C_AUDIT_STOP_RESUME_SEMANTICS_UNRESOLVED.flag
PHASE_N3Z2C_AUDIT_STOP_CORRECTED_PARENT_MISSING.flag
PHASE_N3Z2C_AUDIT_STOP_TRAIN_FAILED.flag
PHASE_N3Z2C_AUDIT_STOP_EVAL_FAILED.flag
PHASE_N3Z2C_AUDIT_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3Z2C_AUDIT_STOP_CPU_AFFINITY_UNRESOLVED.flag
PHASE_N3Z2C_AUDIT_STOP_WATCHER_FAILED.flag
```

Stop examples：

```text
N3Z2C complete flag missing；
parent candidates missing；
reset_num_timesteps / optimizer state cannot be audited；
corrected parent cannot be loaded；
short continuation fails；
eval fails；
Gψ diagnostics abnormal；
CPU affinity/cgroup conflict cannot be explained；
watcher exits without complete/stop。
```

---

## 10. Completion criteria

Only create:

```text
PHASE_N3Z2C_AUDIT_COMPLETE.flag
```

when all are true:

```text
1. N3Z2C complete flag exists；
2. parent candidates are hashed and compared；
3. fixed parent selection rule is applied；
4. resume semantics are audited；
5. CPU affinity/cgroup conflict is audited；
6. corrected parent is selected；
7. corrected short continuation sanity is completed；
8. eval is completed；
9. diagnostics are normal；
10. report clearly decides:
    - accept original N3Z2C conclusion, or
    - rerun corrected Z2 full continuation, or
    - stop due unresolved engineering issue；
11. watcher log and status exist。
```

Complete does **not** automatically mean N4 can start.

Report must explicitly state:

```text
Can enter N4: yes/no
Selected N4 candidate if yes: no_z_full / Z2 / both
Need corrected Z2 full rerun: yes/no
```

---

## 11. Suggested commands

Codex should adapt paths to repo.

### 11.1 Compile

```bash
python -m py_compile scripts/audit_phase_n3z2c_parent_resume.py
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3z2c_audit.py
python -m py_compile scripts/analyze_env_v2_phase_n3z2c_audit.py
bash -n scripts/watch_phase_n3z2c_audit.sh
chmod +x scripts/watch_phase_n3z2c_audit.sh
```

### 11.2 Audit parent / resume / CPU

```bash
python scripts/audit_phase_n3z2c_parent_resume.py \
  --n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --n3fz-result-dir results/env_v2_phase_n3fz_noz_full_z_screen \
  --out-dir results/env_v2_phase_n3z2c_audit
```

### 11.3 Corrected short continuation

```bash
python scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
  --resume <fixed_selected_parent_checkpoint> \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0 \
  --additional-steps 250000 \
  --parent-total-steps 500000 \
  --target-total-steps 750000 \
  --checkpoint-total-steps 750000 \
  --seed 0 \
  --n-envs 4 \
  --device cpu \
  --heartbeat-seconds 300
```

### 11.4 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3z2c_audit.py \
  --result-dir results/env_v2_phase_n3z2c_audit \
  --corrected-checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0 \
  --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --noz-reference-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --attention-checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --write-traces
```

### 11.5 Analysis

```bash
python scripts/analyze_env_v2_phase_n3z2c_audit.py \
  --result-dir results/env_v2_phase_n3z2c_audit \
  --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --attention-success 0.6100 \
  --attention-collision 0.3900
```

### 11.6 Watcher

```bash
bash scripts/watch_phase_n3z2c_audit.sh
```

---

## 12. Watcher pseudo-code

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3z2c_audit"
LOG="$OUT_DIR/phase_n3z2c_audit_watcher.log"
STATUS="$OUT_DIR/phase_n3z2c_audit_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3Z2C-Audit watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n3z2c_z2_continuation/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag"
fi

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/audit_phase_n3z2c_parent_resume.py \
    --n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
    --n3fz-result-dir results/env_v2_phase_n3fz_noz_full_z_screen \
    --out-dir "$OUT_DIR"

  python scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py \
    --config configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml \
    --resume "<fixed_selected_parent_checkpoint>" \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0 \
    --additional-steps 250000 \
    --parent-total-steps 500000 \
    --target-total-steps 750000 \
    --checkpoint-total-steps 750000 \
    --seed 0 \
    --n-envs 4 \
    --device cpu \
    --heartbeat-seconds 300

  python scripts/eval_env_v2_gpsi_ppo_n3z2c_audit.py \
    --result-dir "$OUT_DIR" \
    --corrected-checkpoint-dir checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0 \
    --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
    --eval-seed 1000 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --write-traces

  python scripts/analyze_env_v2_phase_n3z2c_audit.py \
    --result-dir "$OUT_DIR" \
    --old-n3z2c-result-dir results/env_v2_phase_n3z2c_z2_continuation \
    --noz-success 0.5633 \
    --noz-collision 0.4367 \
    --attention-success 0.6100 \
    --attention-collision 0.3900
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3Z2C_AUDIT_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag \
    PHASE_N3Z2C_AUDIT_STOP_PARENT_CANDIDATES_MISSING.flag \
    PHASE_N3Z2C_AUDIT_STOP_RESUME_SEMANTICS_UNRESOLVED.flag \
    PHASE_N3Z2C_AUDIT_STOP_CORRECTED_PARENT_MISSING.flag \
    PHASE_N3Z2C_AUDIT_STOP_TRAIN_FAILED.flag \
    PHASE_N3Z2C_AUDIT_STOP_EVAL_FAILED.flag \
    PHASE_N3Z2C_AUDIT_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3Z2C_AUDIT_STOP_CPU_AFFINITY_UNRESOLVED.flag \
    PHASE_N3Z2C_AUDIT_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3Z2C_AUDIT_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3Z2C_AUDIT_STOP_TRAIN_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  # Poll every 120s; concise heartbeat about every 300s should be implemented by Codex.
  sleep 120
done
```

---

## 13. Terminal decision format

Success:

```text
terminal_decision = phase_n3z2c_audit_complete
```

Stop:

```text
terminal_decision = phase_n3z2c_audit_stopped_<reason>
```

Must list:

```text
new / modified files
actual commands
CPU affinity finding
fixed parent selection result
resume semantics result
corrected short continuation result
comparison with old N3Z2C 750k
decision: accept no_z N4 candidate / rerun corrected Z2 full / unresolved
whether N4 can start
if not, next required action
```
