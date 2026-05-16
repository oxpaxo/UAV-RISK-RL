# Phase N3F/Z 指南：no_z Full Rerun + z_i Scale-Fixed Screening

> 项目：UAV / Dynamic Obstacle Avoidance / DRL  
> 环境：`DynamicObstacleFlowEnv` / EnvV2  
> 当前主线：`Gψ-HeadA + PPO velocity policy + uncertainty-aware Safety Shield`  
> 阶段名称：`Phase N3F/Z - Repaired no_z Full Rerun + z_i Constrained Screening`  
> 阶段性质：修复后 no-shield PPO 主线重训 + z_i epistemic screening；不接 shield；不训练 / fine-tune Gψ；不改 EnvV2-core。  
> 前置条件：Phase N3R 已完成，确认 `no_z` 是 500k screening winner，但 z_i 价值尚未被充分验证。  

---

## 0. 背景与当前判断

Phase N3 原始结果无效，因为 N3.5 已确认 online Gψ normalization bug：

```text
N2 使用 hold-position 数据采集；
ego_current 中 UAV velocity 相关维度 std 接近 1e-6；
N3 online PPO 产生非零 UAV velocity 后，被 legacy wrapper 除以近零 std；
Gψ 输入爆炸；
delta_hat 变成万级；
logvar clamp 到 -5。
```

N3.5 已修复 wrapper，并确认：

```text
before-fix delta_norm_1s_p95 ≈ 22779.90
after-fix delta_norm_1s_p95 ≈ 1.35
offline-online equivalence after-fix:
  z / delta / logvar max_abs_diff = 0.0
```

Phase N3R 在修复后的 wrapper 上完成 A/B/C 500k screening：

```text
A raw_z:
  success = 0.3733
  collision = 0.6267
  raw unsafe = 0.2634

B z_norm:
  success = 0.3667
  collision = 0.6333
  raw unsafe = 0.2361

C no_z:
  success = 0.4233
  collision = 0.5767
  raw unsafe = 0.2650
```

N3R 说明 `no_z` 是当前最稳的 repaired Gψ-PPO no-shield candidate。但这不能直接判死 `z_i`，因为 N3R 仍无法拒绝两个假设：

```text
H1: z_i 被尺度问题淹没。scale-fixed 后，z_i 可能有用。

H2: z_i 本质上与 Head A 显式输出 Δ̂/logvar 冗余。
    NLL loss 只奖励 z_i 编码 decode Δ/logvar 所需信息；
    scale-fix 后仍可能救不回来。
```

所以本阶段的目标不是“再试一次 z_i 看能不能赢”，而是用有硬门槛的 screening 切开 H1 和 H2：

```text
如果 scale-fixed z_i variants 赢过 no_z 500k baseline：
  H1 获得支持，z_i 值得续训到 1.5M。

如果 Z1/Z2/Z3 全输给 no_z 500k baseline：
  H1 被当前证据削弱，第一版不再纠结 z_i，资源回主路径。
```

---

## 1. Phase N3F/Z 总目标

本阶段采用双轨设计。

### Track 1：主路径，必须跑

```text
N3F-no_z @ 1.5M, seed 0
```

目的：

```text
建立修复后、工程干净、compute-matched 的 Gψ-HeadA explicit-output PPO no-shield baseline。
```

配置：

```text
obs_i_aug = [obs_i, delta_hat_scaled, logvar_hat]
Gψ frozen
PPO trainable
no shield
no action filter
no dense safety cost
EnvV2 original reward
masked-attention-compatible PPO
symmetric actor / critic
```

如果有 spare GPU，可补：

```text
N3F-no_z @ 1.5M, seed 1
N3F-no_z @ 1.5M, seed 2
```

### Track 2：screening 路径，有 spare GPU 时跑

```text
Z1: z_l2_scale_4          @ 500k, seed 0
Z2: z_layernorm_alpha_0p5 @ 500k, seed 0
Z3: z_proj16_layernorm    @ 500k, seed 0  # optional
```

目的：

```text
验证 z_i 的失败是否只是尺度/conditioning 问题。
```

硬验收门槛：

```text
Z variant 500k success_rate >= 0.4233
AND
Z variant 500k collision_rate <= 0.5767
```

两个条件必须同时满足，才允许把该 Z variant 升级到 1.5M。

如果 Z1/Z2/Z3 全部不达标：

```text
砍掉 z_i 分支；
资源回到 no_z 主路径；
第一版主线采用 no_z。
```

---

## 2. 核心对照与假设

### 2.1 no_z 主路径回答什么问题

```text
Q1: 修复 wrapper 后，Gψ 显式 Head A 输出 Δ̂/logvar 给 PPO 是否有 no-shield 价值？
```

它不回答：

```text
z_i 是否有价值。
```

### 2.2 Z1/Z2/Z3 回答什么问题

```text
Q2: z_i 在合理尺度约束后是否能超越 no_z 500k baseline？
```

结果解释：

```text
Z wins:
  H1 成立或至少得到支持；
  z_i 有信息但此前被 scale/conditioning 淹没；
  将 Z winner 续训到 1.5M。

Z loses:
  在当前 PPO backbone 与 Gψ 训练目标下，z_i 没有稳定边际收益；
  H2 更可信；
  不再为第一版主线纠结 z_i。
```

### 2.3 fallback 不是当前主路径

只有在 Z1/Z2/Z3 全输、但仍强烈需要继续研究 z_i 时，才考虑 fallback：

```text
重训 Gψ:
  add LayerNorm at z output
  z_dim = 32

前置 gate:
  必须先离线验证 N2 NLL / Δ MSE / calibration 没退化；
  再接 PPO。
```

---

## 3. 明确禁止事项

Phase N3F/Z 禁止：

```text
1. 禁止修改 EnvV2-core；
2. 禁止训练或 fine-tune 当前 Gψ；
3. 禁止实现 / 接入 safety shield；
4. 禁止 action filtering / projection；
5. 禁止加入 dense safety cost；
6. 禁止使用 learned R(s,a)；
7. 禁止使用 candidate velocity risk map as PPO input；
8. 禁止回到 5-head Gψ；
9. 禁止把 Z variant 500k 结果未达标时继续升到 1.5M；
10. 禁止只按 reward 选择 winner，必须看 success/collision；
11. 禁止只看 aggregate，不看 scenario/motion/threat/raw unsafe/feature diagnostics；
12. 禁止覆盖 N3R artifacts。
```

允许：

```text
1. 使用 repaired GpsiObsWrapper；
2. 对 z_i 做 L2 scale / LayerNorm / projection；
3. 对 logvar 做 bounded clip sanity；
4. 训练 no_z 1.5M；
5. 训练 Z1/Z2/Z3 500k screening；
6. 如果 Z winner 达标，后续再续训到 1.5M；
7. 输出完整 report / tables / plots / flags。
```

---

## 4. Track 1：N3F-no_z 1.5M

### 4.1 配置

```text
name: n3f_no_z_full
input:
  obs_i: 12 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

total obstacle dim:
  30

Gψ:
  checkpoint = work_dirs/gpsi_heada_v1_nll/best.pth
  frozen = true
  repaired wrapper = true
  no z_i included
```

### 4.2 训练

```text
seed = 0
train_steps = 1,500,000
checkpoints:
  250k
  500k
  1000k
  1500k
  final
  best_by_eval if available
```

### 4.3 logvar clip sanity

当前 no_z 使用 `logvar_hat`。为避免 logvar block 过度主导 PPO 输入，本阶段必须至少做一个 sanity check：

```text
N3F-no_z-primary:
  logvar clip = existing project default, e.g. [-5, 3] if already used.

N3F-no_z-logvar-abs5 sanity:
  logvar clip = [-5, 5]
```

说明：

```text
如果当前 default 已经比 [-5,5] 更紧，例如 [-5,3]，则 sanity 可记录为 already bounded.
如果实现方便，可跑一个 250k no_z-logvar-abs5 sanity。
如果 default 与 abs5 完全等价或更保守，report 中说明无需额外训练。
```

不要因为 logvar sanity 扩大成完整新主线。

---

## 5. Track 2：z_i constrained screening

### 5.1 Z1：z_l2_scale_4

目的：直接控制 z_i block 的整体范数，避免高维 latent 压倒 obs/delta/logvar。

```text
z_scaled = z_i / (||z_i||_2 + eps) * target_norm
target_norm = 4.0
```

配置：

```text
name: z_l2_scale_4
input:
  obs_i: 12 dims
  z_l2_scaled: 64 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

train_steps = 500,000
seed = 0
```

必须记录：

```text
z_l2_raw_p95
z_l2_scaled_p95
z_target_norm
z_zero_norm_count
```

---

### 5.2 Z2：z_layernorm_alpha_0p5

目的：使用 sample-wise LayerNorm 避免 train-stat z_norm 对稀疏/偏态 latent 的脆弱性。

```text
z_ln = LayerNorm(z_i)
z_scaled = alpha * z_ln
alpha = 0.5
```

如果 LayerNorm 后每维方差约 1，则 64 维 L2 大约：

```text
sqrt(64) * 0.5 ≈ 4
```

配置：

```text
name: z_layernorm_alpha_0p5
input:
  obs_i: 12 dims
  z_layernorm_scaled: 64 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

train_steps = 500,000
seed = 0
```

必须记录：

```text
z_l2_raw_p95
z_l2_layernorm_p95
z_l2_scaled_p95
layernorm_eps
alpha
```

---

### 5.3 Z3：z_proj16_layernorm，可选

目的：让 PPO 学一个小 adapter，从 64 维 z 中提取低维有效成分，避免直接拼接高维 latent。

```text
z_i
→ Linear(64, 16)
→ LayerNorm
→ Tanh or no Tanh, fixed by config
→ optional scale
```

配置：

```text
name: z_proj16_layernorm
input:
  obs_i: 12 dims
  z_proj16: 16 dims
  delta_hat_scaled: 9 dims
  logvar_hat: 9 dims

total obstacle dim:
  46

train_steps = 500,000
seed = 0
```

注意：

```text
Gψ 仍 frozen；
projection adapter 属于 PPO policy / feature extractor，可训练；
report 必须把它标为 trainable z adapter，不可与 frozen z 直接拼接混淆。
```

Z3 资源不足时可以不跑，但必须在报告中写明。

---

## 6. Z 变体验收硬条件

Z variant 500k 必须同时满足：

```text
success_rate >= 0.4233
collision_rate <= 0.5767
```

其中：

```text
0.4233 / 0.5767 来自 N3R no_z 500k final。
```

如果 Z variant 只满足一个条件，例如：

```text
success 高但 collision 更高；
collision 低但 success 明显低；
```

则不升级 1.5M。

winner ranking 只能在通过硬门槛的 Z variants 中进行。排序优先级：

```text
1. lower collision_rate
2. higher success_rate
3. lower near_miss_rate
4. lower raw_unsafe_action_rate
5. higher progress
6. better scenario robustness
```

---

## 7. 评估协议

所有 configs 需要使用同一 eval protocol：

```text
eval scenarios:
  eval_flow_id
  eval_flow_high_density
  eval_flow_high_speed
  eval_flow_high_threat
  eval_flow_mixed_ood
  eval_flow_sudden_threat

episodes:
  50 per scenario

eval_seed:
  same as N3R / Phase A/B protocol, e.g. 1000
```

N3F-no_z 需要评估：

```text
checkpoint_250k
checkpoint_500k
checkpoint_1000k
checkpoint_1500k
final
best_by_eval if available
```

Z variants 需要评估：

```text
checkpoint_250k
checkpoint_500k
final
best_by_eval if available
```

必须包含 reference：

```text
attention_full_1500k
N3R no_z 500k
N3 original invalid result as engineering-invalid reference only
```

---

## 8. 必须记录的 diagnostics

### 8.1 Gψ output diagnostics

每个 config / checkpoint / scenario：

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

要求：

```text
delta_norm 不得回到 1e4 量级；
logvar 不得无解释全贴 -5；
projected_std 不得无解释恒定；
inactive_forwarded_count 必须为 0。
```

### 8.2 Feature block stats

每个 config：

```text
obs_block_l2_p95
z_block_l2_raw_p95
z_block_l2_after_constraint_p95
delta_block_l2_p95
logvar_block_l2_p95
full_aug_obs_l2_p95
max_abs per block
nan/inf per block
```

Z1/Z2/Z3 必须证明：

```text
z after constraint 的 L2 p95 被压到合理量级；
推荐目标：约 2–8，而不是 30–50。
```

### 8.3 PPO / safety diagnostics

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
episode_length
raw_unsafe_action_rate
raw_min_predicted_cpa
action_norm
action_delta
no_response_rate if available
```

特别关注：

```text
raw unsafe 低但 collision 高；
action_delta 过低导致反应不足；
scenario-specific failure。
```

### 8.4 Breakdown

必须按以下维度输出：

```text
config
checkpoint
scenario
motion mode
threat class
```

---

## 9. 推荐新增 / 修改文件

```text
configs/
  env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml
  env_v2_gpsi_heada_ppo_n3z_l2_scale4.yaml
  env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5.yaml
  env_v2_gpsi_heada_ppo_n3z_proj16_layernorm.yaml  # optional

scripts/
  train_env_v2_gpsi_ppo_n3fz.py
  eval_env_v2_gpsi_ppo_n3fz.py
  analyze_env_v2_phase_n3fz_results.py
  watch_phase_n3fz_noz_full_z_screen.sh
```

可以复用 N3R scripts，但必须：

```text
1. 不覆盖 N3R artifacts；
2. 明确 config names；
3. 明确 Track 1 / Track 2；
4. 输出 hard-gate decision for Z variants。
```

输出目录建议：

```text
checkpoints/
  env_v2_gpsi_heada_ppo_n3f_no_z_s0/
  env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0/
  env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/
  env_v2_gpsi_heada_ppo_n3z_proj16_layernorm_s0/

results/
  env_v2_phase_n3fz_noz_full_z_screen/
    PHASE_N3FZ_NOZ_FULL_Z_SCREEN_REPORT.md
    PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag
    phase_n3fz_status.txt
    phase_n3fz_watcher.log

    tables/
      phase_n3fz_config_manifest.csv
      phase_n3fz_train_curve.csv
      phase_n3fz_checkpoint_eval_summary.csv
      phase_n3fz_eval_summary.csv
      phase_n3fz_attention_reference_comparison.csv
      phase_n3fz_n3r_baseline_comparison.csv
      phase_n3fz_scenario_breakdown.csv
      phase_n3fz_motion_mode_breakdown.csv
      phase_n3fz_threat_class_breakdown.csv
      phase_n3fz_raw_unsafe_action_summary.csv
      phase_n3fz_gpsi_output_summary.csv
      phase_n3fz_aug_feature_block_stats.csv
      phase_n3fz_z_gate_decision.csv
      phase_n3fz_full_rerun_recommendation.csv
      phase_n3fz_command_manifest.csv
      phase_n3fz_schema_check.csv

    plots/
      noz_full_checkpoint_success_collision.png
      z_screen_success_collision.png
      z_screen_gate_plot.png
      raw_unsafe_by_config.png
      scenario_breakdown_by_config.png
      aug_feature_block_scale_by_config.png
      gpsi_delta_norm_by_config.png
      gpsi_logvar_by_config.png
      train_reward_by_config.png

    logs/
      phase_n3fz_train_no_z_full.log
      phase_n3fz_train_z_l2_scale4.log
      phase_n3fz_train_z_layernorm.log
      phase_n3fz_train_z_proj16.log
      phase_n3fz_eval.log
      phase_n3fz_analysis.log
```

---

## 10. 命令清单

Codex 应按实际 repo 路径调整命令，并在 report 中记录最终实际命令。

### 10.1 编译检查

```bash
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3fz.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3fz.py
python -m py_compile scripts/analyze_env_v2_phase_n3fz_results.py
bash -n scripts/watch_phase_n3fz_noz_full_z_screen.sh
chmod +x scripts/watch_phase_n3fz_noz_full_z_screen.sh
```

### 10.2 训练 no_z full

```bash
python scripts/train_env_v2_gpsi_ppo_n3fz.py \
  --config configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
  --train-steps 1500000 \
  --seed 0
```

### 10.3 训练 Z1/Z2/Z3 screening

```bash
python scripts/train_env_v2_gpsi_ppo_n3fz.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z_l2_scale4.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0 \
  --train-steps 500000 \
  --seed 0

python scripts/train_env_v2_gpsi_ppo_n3fz.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0 \
  --train-steps 500000 \
  --seed 0

# optional, only if resources allow
python scripts/train_env_v2_gpsi_ppo_n3fz.py \
  --config configs/env_v2_gpsi_heada_ppo_n3z_proj16_layernorm.yaml \
  --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_proj16_layernorm_s0 \
  --train-steps 500000 \
  --seed 0
```

### 10.4 Eval

```bash
python scripts/eval_env_v2_gpsi_ppo_n3fz.py \
  --result-dir results/env_v2_phase_n3fz_noz_full_z_screen \
  --configs no_z_full z_l2_scale4 z_layernorm_alpha0p5 z_proj16_layernorm \
  --eval-seed 1000 \
  --num-episodes 50 \
  --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
  --write-traces
```

If Z3 is not run, omit it from `--configs`.

### 10.5 Analysis

```bash
python scripts/analyze_env_v2_phase_n3fz_results.py \
  --result-dir results/env_v2_phase_n3fz_noz_full_z_screen \
  --n3r-noz-success 0.4233 \
  --n3r-noz-collision 0.5767
```

### 10.6 Watcher

```bash
bash scripts/watch_phase_n3fz_noz_full_z_screen.sh
```

---

## 11. 报告要求

必须输出：

```text
results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_REPORT.md
```

报告至少包含：

```text
1. 背景：N3 invalid、N3.5 repaired、N3R screening result；
2. H1 vs H2 framing；
3. Track 1 / Track 2 设计；
4. no_z full config；
5. Z1/Z2/Z3 config；
6. logvar clip sanity；
7. training budgets；
8. eval protocol；
9. no_z 1.5M checkpoint curve；
10. Z variants 500k comparison；
11. Z hard-gate decision；
12. attention_full reference comparison；
13. N3R no_z 500k reference comparison；
14. scenario / motion / threat breakdown；
15. raw unsafe and action dynamics analysis；
16. Gψ output diagnostics；
17. feature block scale analysis；
18. whether any Z variant should be promoted to 1.5M；
19. whether no_z full is strong enough for N4；
20. whether can enter N4；
21. terminal_decision。
```

报告必须明确区分：

```text
engineering facts
experiment-supported facts
hypothesis test result for H1/H2
reasonable inferences
remaining risks
```

---

## 12. 完成判据

只有同时满足以下条件，才允许生成：

```text
PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag
```

必须满足：

```text
1. Phase N3R complete flag 存在；
2. repaired GpsiObsWrapper 被使用；
3. Gψ checkpoint 可加载；
4. Gψ frozen；
5. no_z full seed0 训练到 1.5M；
6. Z1/Z2 至少完成 500k screening，除非资源 stop / report 明确说明；
7. Z3 若资源允许则完成，否则标 optional not run；
8. eval 完成；
9. Gψ output scale 正常；
10. feature block stats 生成；
11. raw unsafe diagnostics 生成；
12. scenario/motion/threat breakdown 生成；
13. Z hard-gate decision 生成；
14. no_z full N4 decision 生成；
15. report 生成；
16. watcher log 与 status 文件存在；
17. 明确是否可进入 N4；
18. 明确是否有 Z winner 需要 1.5M continuation。
```

注意：

```text
Complete flag 不等于一定进入 N4。
如果 no_z full 很弱或 diagnostics 异常，report 必须写：
Can enter N4: no.
```

---

## 13. 停止条件

如出现以下问题，必须生成 stop flag、partial report 和 log。

```text
PHASE_N3FZ_STOP_PHASE_N3R_MISSING.flag
PHASE_N3FZ_STOP_GPSI_CHECKPOINT_MISSING.flag
PHASE_N3FZ_STOP_WRAPPER_SCALE_INVALID.flag
PHASE_N3FZ_STOP_NOZ_FULL_TRAIN_FAILED.flag
PHASE_N3FZ_STOP_Z_SCREEN_TRAIN_FAILED.flag
PHASE_N3FZ_STOP_EVAL_FAILED.flag
PHASE_N3FZ_STOP_DIAGNOSTICS_FAILED.flag
PHASE_N3FZ_STOP_RESOURCE_LIMIT.flag
PHASE_N3FZ_STOP_WATCHER_FAILED.flag
```

触发示例：

```text
N3R complete flag 缺失；
Gψ checkpoint 缺失；
修复后的 wrapper 输出又出现 delta 万级 / logvar 恒定；
no_z full 无法训练；
Z1/Z2 无法完成且没有合理资源解释；
eval 失败；
diagnostics 无法生成；
资源不足以完成主路径；
watcher 失败。
```

---

## 14. 阻塞式 watcher 要求

必须新增：

```text
scripts/watch_phase_n3fz_noz_full_z_screen.sh
```

watcher 必须：

```text
1. 检查 Phase N3R complete flag；
2. 训练 no_z full 1.5M；
3. 训练 Z1/Z2 500k；
4. 资源允许则训练 Z3 500k；
5. eval all completed configs；
6. run analysis；
7. generate report；
8. 持续轮询 complete / stop flag；
9. 持续输出当前状态；
10. 只有 complete flag 或 stop flag 出现才退出；
11. 不允许中途等待用户确认；
12. 不允许因为暂无新日志退出；
13. 写入 phase_n3fz_watcher.log；
14. 写入 phase_n3fz_status.txt。
```

伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/env_v2_phase_n3fz_noz_full_z_screen"
LOG="$OUT_DIR/phase_n3fz_watcher.log"
STATUS="$OUT_DIR/phase_n3fz_status.txt"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/tables" "$OUT_DIR/plots"

echo "[watcher] Phase N3F/Z watcher started" | tee -a "$LOG"
echo "running" > "$STATUS"

if [[ ! -f "results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag" ]]; then
  touch "$OUT_DIR/PHASE_N3FZ_STOP_PHASE_N3R_MISSING.flag"
fi

(
  python scripts/train_env_v2_gpsi_ppo_n3fz.py \
    --config configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0 \
    --train-steps 1500000 \
    --seed 0

  python scripts/train_env_v2_gpsi_ppo_n3fz.py \
    --config configs/env_v2_gpsi_heada_ppo_n3z_l2_scale4.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0 \
    --train-steps 500000 \
    --seed 0

  python scripts/train_env_v2_gpsi_ppo_n3fz.py \
    --config configs/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5.yaml \
    --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0 \
    --train-steps 500000 \
    --seed 0

  # Optional Z3 can be enabled if resources allow.
  # python scripts/train_env_v2_gpsi_ppo_n3fz.py \
  #   --config configs/env_v2_gpsi_heada_ppo_n3z_proj16_layernorm.yaml \
  #   --out-dir checkpoints/env_v2_gpsi_heada_ppo_n3z_proj16_layernorm_s0 \
  #   --train-steps 500000 \
  #   --seed 0

  python scripts/eval_env_v2_gpsi_ppo_n3fz.py \
    --result-dir "$OUT_DIR" \
    --configs no_z_full z_l2_scale4 z_layernorm_alpha0p5 \
    --eval-seed 1000 \
    --num-episodes 50 \
    --scenarios eval_flow_id eval_flow_high_density eval_flow_high_speed eval_flow_high_threat eval_flow_mixed_ood eval_flow_sudden_threat \
    --write-traces

  python scripts/analyze_env_v2_phase_n3fz_results.py \
    --result-dir "$OUT_DIR" \
    --n3r-noz-success 0.4233 \
    --n3r-noz-collision 0.5767
) 2>&1 | tee -a "$LOG" &
PID=$!

while true; do
  if [[ -f "$OUT_DIR/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag" ]]; then
    echo "complete" > "$STATUS"
    echo "[watcher] complete flag detected" | tee -a "$LOG"
    exit 0
  fi

  for flag in \
    PHASE_N3FZ_STOP_PHASE_N3R_MISSING.flag \
    PHASE_N3FZ_STOP_GPSI_CHECKPOINT_MISSING.flag \
    PHASE_N3FZ_STOP_WRAPPER_SCALE_INVALID.flag \
    PHASE_N3FZ_STOP_NOZ_FULL_TRAIN_FAILED.flag \
    PHASE_N3FZ_STOP_Z_SCREEN_TRAIN_FAILED.flag \
    PHASE_N3FZ_STOP_EVAL_FAILED.flag \
    PHASE_N3FZ_STOP_DIAGNOSTICS_FAILED.flag \
    PHASE_N3FZ_STOP_RESOURCE_LIMIT.flag \
    PHASE_N3FZ_STOP_WATCHER_FAILED.flag
  do
    if [[ -f "$OUT_DIR/$flag" ]]; then
      echo "stopped:$flag" > "$STATUS"
      echo "[watcher] stop flag detected: $flag" | tee -a "$LOG"
      exit 2
    fi
  done

  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || true
    if [[ ! -f "$OUT_DIR/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag" ]]; then
      touch "$OUT_DIR/PHASE_N3FZ_STOP_NOZ_FULL_TRAIN_FAILED.flag"
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

## 15. Codex 执行原则

Codex 必须：

```text
1. 先阅读本指南；
2. 阅读 N3R report；
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
13. 最终必须明确：no_z full 是否可作为 N4 policy candidate，是否有 Z winner 需要续训。
```

---

## 16. 终端结论格式

成功：

```text
terminal_decision = phase_n3fz_noz_full_z_screen_complete
```

停止：

```text
terminal_decision = phase_n3fz_stopped_<reason>
```

必须列出：

```text
新增 / 修改文件
实际运行命令
Track 1 no_z full 主要结果
Track 2 Z variants 主要结果
Z hard-gate decision
是否有 Z winner 需要 1.5M continuation
生成的 checkpoints
生成的 CSV / plots / report / logs / flags
是否可以进入 N4
如果不能进入 N4，下一步做什么
```

---

## 17. 完成后的分支决策

### 情况 A：no_z full 接近 / 超过 attention_full

```text
N3F no_z becomes valid no-shield Gψ-PPO baseline.
Proceed to N4 shield comparison.
```

### 情况 B：no_z full 仍弱于 attention_full，但 Z variant 500k 过硬门槛

```text
Promote Z winner to 1.5M continuation.
Do not enter N4 until Z winner full result is available, unless user explicitly chooses no_z for N4.
```

### 情况 C：no_z full 仍弱，Z variants 全输

```text
First-version Gψ-PPO no-shield is weak.
Adopt no_z as the cleanest Gψ policy candidate if needed;
shift main value test to N4 shield-side use of Δ̂/logvar.
Do not continue z_i in first-version PPO path.
```

### 情况 D：Z variants 全输，但 user still wants z_i

```text
Fallback:
  retrain Gψ with LayerNorm at z output and z_dim=32.
  Must pass N2-style offline NLL / Δ MSE / calibration gate before PPO.
