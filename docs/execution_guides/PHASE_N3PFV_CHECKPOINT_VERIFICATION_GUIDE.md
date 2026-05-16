# Phase N3PF-V 指南：P3 Block-Projected Checkpoint Verification Eval

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：DynamicObstacleFlowEnv / EnvV2  
> 阶段名称：Phase N3PF-V - P3 Block-Projected Checkpoint Verification Eval  
> 阶段性质：只做复评估 / 稳健性验证；不训练；不接 shield；不改 EnvV2-core；不 fine-tune Gψ。  
> 前置条件：Phase N3PF 已完成，P3 `block_projected` 已跑到 1.5M，并出现 1000k / 1500k 超过 attention_full、final 略低于 attention_full 的 checkpoint 差异。

---

## 0. 背景与当前结论

Phase N3PF 当前结果：

```text
attention_full_1500k:
  success = 0.6100
  collision = 0.3900

N3F no_z full:
  success = 0.5633
  collision = 0.4367

corrected Z2 full:
  success = 0.5067
  collision = 0.4933

P3 block_projected parent_500k:
  success = 0.5333
  collision = 0.4667

P3 block_projected checkpoint_1000k:
  success = 0.6300
  collision = 0.3700

P3 block_projected checkpoint_1500k:
  success = 0.6200
  collision = 0.3800

P3 block_projected final:
  success = 0.5967
  collision = 0.4033
```

这说明：

```text
1. P3 block_projected 明确优于 no_z raw concatenation；
2. P3 在 checkpoint_1000k 和 checkpoint_1500k 上超过 attention_full；
3. final.zip 因 rollout overshoot / end-of-training 保存口径略低于 attention_full；
4. 当前主要问题不是继续改网络，而是确认 P3 checkpoint 的稳健性与 N4 candidate 选择口径。
```

本阶段目标是通过多 seed / 多 episode 复评估，回答：

```text
1. P3 checkpoint_1000k 是否稳定超过 attention_full？
2. P3 checkpoint_1500k 是否稳定超过 attention_full？
3. P3 final 略低于 attention 是否只是 300 episode eval 噪声？
4. P3 checkpoint_1000k 的 eval_flow_id 弱点是否稳定存在？
5. P3 checkpoint_1500k 是否更均衡、更适合作为 N4 主候选？
6. N4 应使用 P3 1000k、P3 1500k、P3 final、还是 no_z？
```

---

## 1. 本阶段总目标

本阶段只做 evaluation，不做任何 training。

必须评估的 policy / checkpoint：

```text
P3_block_projected_1000k
P3_block_projected_1500k
P3_block_projected_final

attention_full_1500k
no_z_full
corrected_Z2_full
```

推荐同时评估的 diagnostic checkpoints：

```text
P3_block_projected_parent_500k
P3_block_projected_1250k
```

说明：

```text
P3 1250k 在单 seed eval 中明显退化，可用于分析 checkpoint instability；
P3 parent_500k 可用于确认 continuation gain。
```

如果计算时间紧张，可只评估前 6 个 required checkpoints；但 report 需说明 skipped diagnostic checkpoints。

---

## 2. 明确禁止事项

Phase N3PF-V 禁止：

```text
1. 禁止训练任何 PPO；
2. 禁止 fine-tune Gψ；
3. 禁止进入 N4；
4. 禁止实现 shield；
5. 禁止 action filtering / projection；
6. 禁止加入 dense safety cost；
7. 禁止修改 EnvV2-core；
8. 禁止改 policy architecture；
9. 禁止覆盖 N3PF / N3P / N3F / N3Z2CF 原始产物；
10. 禁止使用本阶段 eval seed 反过来训练或调参；
11. 禁止只看总 success/collision，不看 scenario/motion/threat breakdown。
```

允许：

```text
1. 新增 eval / analysis / watcher 脚本；
2. 对已有 checkpoints 做多 seed eval；
3. 输出稳定性、置信区间、checkpoint 推荐；
4. 给出是否可以进入 N4 的决策。
```

---

## 3. Evaluation protocol

默认使用三个 eval seeds：

```text
1000
1001
1002
```

每个 seed 下：

```text
6 scenarios × 50 episodes = 300 episodes
```

每个 policy / checkpoint 总计：

```text
3 seeds × 300 episodes = 900 episodes
```

如果资源不足，允许降级为：

```text
2 seeds: 1000, 1001
```

但必须在 report 中说明：

```text
verification_eval_degraded = true
reason = <reason>
```

必须覆盖 scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

不要为了加速减少 episode，除非生成 degraded flag 并在 report 中说明。

---

## 4. Checkpoint paths

Codex 必须根据实际 repo 检查路径。推荐路径如下：

### 4.1 P3 block_projected

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip
```

Diagnostic optional：

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1250k.zip
checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip
```

### 4.2 References

```text
attention_full:
  checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip

no_z_full:
  checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip
  or N3F reported best/final checkpoint path

corrected_Z2_full:
  checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip
```

If exact reference path differs, Codex must discover it from prior manifests and record the resolved path.

---

## 5. Checkpoint integrity and manifest

Before eval, output:

```text
phase_n3pfv_checkpoint_manifest.csv
```

Required fields:

```text
policy_key
checkpoint_label
checkpoint_path
exists
size_bytes
sha256
source_phase
expected_success_single_seed_if_known
expected_collision_single_seed_if_known
selected_for_required_eval
selected_for_diagnostic_eval
```

If a required checkpoint is missing, create:

```text
PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag
```

---

## 6. Metrics and diagnostics

### 6.1 Main metrics

For every policy / checkpoint / eval_seed / scenario:

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
mean_episode_reward
mean_episode_length
```

### 6.2 Raw unsafe / action diagnostics

For every policy / checkpoint / eval_seed / scenario:

```text
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
```

### 6.3 Breakdowns

Required breakdowns:

```text
scenario breakdown
motion mode breakdown
threat class breakdown
```

Special focus:

```text
1. eval_flow_id:
   P3 1000k had weak ID in the single-seed result; verify if stable.

2. high_speed / high_threat:
   P3 had strong gains here; verify if stable.

3. linear motion mode:
   P3 final was weak on linear in single-seed result; verify if checkpoint-specific or stable.

4. raw unsafe:
   P3 1000k had lower raw unsafe than 1500k/final; verify if this explains performance.
```

---

## 7. Statistical summary

For each policy/checkpoint aggregate over eval seeds:

```text
mean_success
std_success
mean_collision
std_collision
mean_near_miss
std_near_miss
mean_raw_unsafe
std_raw_unsafe
num_eval_seeds
num_episodes_total
```

Also compute pairwise differences:

```text
P3_1000k - attention_full
P3_1500k - attention_full
P3_final - attention_full
P3_1000k - no_z
P3_1500k - no_z
P3_final - no_z
```

For each pair:

```text
success_diff
collision_diff
scenario_level_diff
seed_level_diff
```

Use simple confidence summaries if easy:

```text
binomial standard error for success/collision
Wilson interval or normal approx interval
```

If this is too costly, at minimum provide mean/std over seeds and raw episode count.

---

## 8. Candidate decision rules

### 8.1 Primary N4 candidate

Decision preference:

```text
1. Prefer checkpoint_1500k if it is statistically comparable to or better than attention_full and no_z.
   Reason: exact 1.5M snapshot, cleaner selection口径.

2. Use checkpoint_1000k as primary only if:
   - it clearly beats checkpoint_1500k across seeds;
   - its scenario breakdown is not pathologically weak on eval_flow_id or other core scenarios;
   - report explicitly labels it as offline-eval-selected checkpoint.

3. Use final.zip only if:
   - it matches or exceeds checkpoint_1500k across seeds;
   - or exact 1500k is unavailable / invalid.

4. Fall back to no_z only if:
   - P3 1000k/1500k/final are worse than no_z on both success and collision;
   - or P3 diagnostics / checkpoint integrity fails.
```

### 8.2 Attention comparison

If P3 checkpoint_1500k or 1000k:

```text
mean_success >= attention_success
AND
mean_collision <= attention_collision
```

then report:

```text
P3 matches/exceeds attention_full under verification eval.
```

If difference is small and intervals overlap:

```text
P3 is comparable to attention_full, not decisively better.
```

If P3 below attention but above no_z:

```text
P3 improves Gψ-PPO no-shield over raw no_z but does not conclusively beat attention_full.
```

### 8.3 Can enter N4?

Can enter N4 if:

```text
1. required checkpoints evaluated;
2. verification eval complete;
3. candidate decision explicit;
4. no unresolved checkpoint / eval path issue;
5. no diagnostics regression.
```

N4 candidate should be one of:

```text
P3_checkpoint_1500k
P3_checkpoint_1000k
P3_final
no_z_full
both P3_1500k and no_z as ablation
```

---

## 9. Output directory and files

Output directory:

```text
results/env_v2_phase_n3pfv_checkpoint_verification/
```

Required files:

```text
PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md
PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag
phase_n3pfv_status.txt
phase_n3pfv_watcher.log
```

Required tables:

```text
phase_n3pfv_checkpoint_manifest.csv
phase_n3pfv_command_manifest.csv
phase_n3pfv_eval_summary_by_seed.csv
phase_n3pfv_eval_summary_aggregate.csv
phase_n3pfv_pairwise_comparison.csv
phase_n3pfv_scenario_breakdown.csv
phase_n3pfv_motion_mode_breakdown.csv
phase_n3pfv_threat_class_breakdown.csv
phase_n3pfv_raw_unsafe_action_summary.csv
phase_n3pfv_action_dynamics_summary.csv
phase_n3pfv_candidate_decision.csv
phase_n3pfv_schema_check.csv
```

Recommended plots:

```text
n3pfv_success_collision_mean_ci.png
n3pfv_checkpoint_comparison.png
n3pfv_scenario_breakdown.png
n3pfv_seed_stability.png
n3pfv_raw_unsafe_comparison.png
n3pfv_action_dynamics.png
n3pfv_pairwise_vs_attention.png
```

---

## 10. Stop flags

Create partial report and stop if needed:

```text
PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag
PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag
PHASE_N3PFV_STOP_EVAL_FAILED.flag
PHASE_N3PFV_STOP_SCHEMA_INVALID.flag
PHASE_N3PFV_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3PFV_STOP_WATCHER_FAILED.flag
```

Stop examples:

```text
P3 checkpoint_1000k or 1500k missing；
attention/no_z/Z2 reference path cannot be resolved；
eval crashes；
eval output schema missing required fields；
raw unsafe / scenario breakdown cannot be generated；
watcher exits without complete/stop。
```

---

## 11. Completion criteria

Only create:

```text
PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag
```

when all are true:

```text
1. checkpoint manifest generated；
2. all required policies/checkpoints evaluated；
3. at least eval seeds 1000/1001/1002 completed, or degraded mode explicitly documented；
4. aggregate metrics generated；
5. scenario/motion/threat breakdown generated；
6. raw unsafe/action diagnostics generated；
7. pairwise comparison against attention_full/no_z generated；
8. final N4 candidate decision generated；
9. report generated；
10. watcher log and status exist。
```

Complete does not automatically mean P3 beats attention. Report must say:

```text
Can enter N4: yes/no
Selected N4 candidate: P3_1000k / P3_1500k / P3_final / no_z / both / undecided
```

---

## 12. Suggested commands

Codex should adapt paths.

### 12.1 Compile

```bash
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3pfv.py
python -m py_compile scripts/analyze_env_v2_phase_n3pfv_results.py
bash -n scripts/watch_phase_n3pfv_checkpoint_verification.sh
chmod +x scripts/watch_phase_n3pfv_checkpoint_verification.sh
```

### 12.2 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3pfv.py \
  --result-dir results/env_v2_phase_n3pfv_checkpoint_verification \
  --eval-seeds 1000 1001 1002 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --policies p3_1000k p3_1500k p3_final attention_full no_z_full z2_corrected_full \
  --include-diagnostic-policies p3_parent_500k p3_1250k \
  --write-traces
```

### 12.3 Analysis

```bash
python scripts/analyze_env_v2_phase_n3pfv_results.py \
  --result-dir results/env_v2_phase_n3pfv_checkpoint_verification \
  --attention-success 0.6100 \
  --attention-collision 0.3900 \
  --noz-success 0.5633 \
  --noz-collision 0.4367 \
  --z2-success 0.5067 \
  --z2-collision 0.4933 \
  --p3-parent-success 0.5333 \
  --p3-parent-collision 0.4667
```

### 12.4 Watcher

```bash
bash scripts/watch_phase_n3pfv_checkpoint_verification.sh
```

---

## 13. Watcher pseudo-code

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3pfv_checkpoint_verification"
LOG="$OUT_DIR/phase_n3pfv_watcher.log"
STATUS="$OUT_DIR/phase_n3pfv_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3PF-V watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

(
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/eval_env_v2_gpsi_ppo_n3pfv.py \
    --result-dir "$OUT_DIR" \
    --eval-seeds 1000 1001 1002 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --policies p3_1000k p3_1500k p3_final attention_full no_z_full z2_corrected_full \
    --include-diagnostic-policies p3_parent_500k p3_1250k \
    --write-traces

  python scripts/analyze_env_v2_phase_n3pfv_results.py \
    --result-dir "$OUT_DIR" \
    --attention-success 0.6100 \
    --attention-collision 0.3900 \
    --noz-success 0.5633 \
    --noz-collision 0.4367 \
    --z2-success 0.5067 \
    --z2-collision 0.4933 \
    --p3-parent-success 0.5333 \
    --p3-parent-collision 0.4667
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag \
    PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag \
    PHASE_N3PFV_STOP_EVAL_FAILED.flag \
    PHASE_N3PFV_STOP_SCHEMA_INVALID.flag \
    PHASE_N3PFV_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3PFV_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3PFV_STOP_EVAL_FAILED.flag"
      echo "stopped:process_exited_without_complete_flag" > "$STATUS"
      echo "[watcher] process exited without complete flag" | tee -a "$LOG"
      exit 2
    fi
  fi

  # Eval-only stage; poll every 120s, concise chat/status heartbeat about every 300s.
  sleep 120
done
```

---

## 14. Terminal decision format

Success:

```text
terminal_decision = phase_n3pfv_checkpoint_verification_complete
```

Stop:

```text
terminal_decision = phase_n3pfv_stopped_<reason>
```

Must report:

```text
new / modified files
actual commands
evaluated checkpoints and seeds
P3 1000k aggregate result
P3 1500k aggregate result
P3 final aggregate result
comparison vs attention_full
comparison vs no_z full
scenario/motion/threat summary
selected N4 candidate
whether N4 can start
if not, next required action
```
