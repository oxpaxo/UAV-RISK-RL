# Phase N4-O 指南：Ordinary VO-like Shield Fair Comparison

## 0. 目标

本阶段只做 ordinary VO-like shield 的 fair comparison。不训练 PPO，不使用 σ² uncertainty margin，不改 EnvV2-core。后续 N4-U 才做 uncertainty-aware shield。

当前 no-shield 参考：

```text
P3 checkpoint_1500k: success=0.6167, collision=0.3833
attention_full:      success=0.6033, collision=0.3967
no_z full:           success=0.5667, collision=0.4333
```

Phase B strongest context：

```text
vo_like_filter_h45_cpa1p2_h16:
  success=0.8333, collision=0.1667, near_miss=0.7200, progress=0.9880, trigger=0.3509

cpa_ttc_weighted_apf_alpha3:
  success=0.7200, collision=0.2800, near_miss=0.7200, progress=0.8964
```

本阶段回答：

```text
1. P3 + same ordinary shield 是否优于 attention_full + same shield？
2. P3 + same ordinary shield 是否优于 no_z + same shield？
3. ordinary shield 本身贡献多大？
4. 是否值得进入 N4-U 做 σ² uncertainty-aware shield？
```

## 1. 禁止事项

```text
禁止训练任何 PPO
禁止 fine-tune Gψ
禁止实现 σ² uncertainty-aware shield
禁止使用 directional uncertainty margin
禁止修改 EnvV2-core
禁止只给 P3 用 shield 而不给 attention/no_z 用同一个 shield
禁止根据 P3 单独调 shield 参数
禁止覆盖 Phase B / N3PF / N3PF-V 原始产物
```

## 2. Required methods

No-shield references：

```text
attention_full
no_z_full
P3_checkpoint_1500k
```

Same-shield policies：

```text
attention_full + ordinary VO-like shield
no_z_full + ordinary VO-like shield
P3_checkpoint_1500k + ordinary VO-like shield
```

Geometry/filter context：

```text
vo_like_filter_h45_cpa1p2_h16
cpa_ttc_weighted_apf_alpha3
```

Optional diagnostic：

```text
P3_checkpoint_1000k + ordinary shield
P3_final + ordinary shield
```

## 3. Shield definition

使用 Phase B strongest ordinary VO-like shield：

```text
config: vo_like_filter_h45_cpa1p2_h16
baseline: vo_like_filter
```

必须从 Phase B manifest/config 解析真实参数，不要凭名字猜。

普通 shield 必须满足：

```text
takes raw policy action a_raw
checks VO/CPA-TTC style safety over candidate velocities
outputs a_exec
uses_sigma2 = false
uses_future_truth = false
same_params_for_attention_noz_p3 = true
```

## 4. Eval protocol

Eval seeds：

```text
1000 1001 1002
```

Scenarios：

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

Episodes：

```text
50 per scenario per seed
```

Each method：900 episodes。

若资源不足可降级为 seeds 1000/1001，但必须记录 degraded mode 和原因。

## 5. Checkpoints

```text
P3:
checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip

attention_full:
checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip

no_z_full:
checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip
或从 N3F manifest 解析
```

必须输出：

```text
phase_n4o_checkpoint_manifest.csv
phase_n4o_shield_config_manifest.csv
```

## 6. Metrics

Main：

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
mean_episode_length
mean_episode_reward
```

Shield-specific：

```text
shield_trigger_rate
shield_intervention_steps
filter_delta_norm_mean
filter_delta_norm_p95
max_filter_delta_norm
raw_action_unsafe_rate
raw_min_predicted_cpa
filtered_min_predicted_cpa
raw_ttc_min
filtered_ttc_min
a_raw_norm
a_exec_norm
action_delta_raw
action_delta_exec
progress_loss_due_to_filter
```

Breakdowns：

```text
scenario / motion mode / threat class / failure cases
```

Key comparisons：

```text
P3 + shield vs P3 no-shield
attention + shield vs attention no-shield
no_z + shield vs no_z no-shield
P3 + shield vs attention + shield
P3 + shield vs no_z + shield
P3 + shield vs Phase B VO-like context
```

## 7. Decision rules

N4-O 有效条件：

```text
same shield params for attention/no_z/P3
ordinary shield uses_sigma2=false
eval protocol identical
```

P3 + ordinary shield positive if：

```text
collision lower than P3 no-shield
success not severely degraded
progress acceptable
intervention rate not excessive
comparable or better than attention_full + same shield
```

Proceed to N4-U if：

```text
P3 + ordinary shield competitive
intervention diagnostics clean
remaining collisions / near misses suggest uncertainty margin may help
ordinary shield does not solve everything equally for all policies
```

If attention + ordinary shield dominates P3 + ordinary shield, stop and analyze before N4-U.

## 8. 并行策略

N4-O 可以与 N3PF-MS 同时运行，因为 N4-O 是 eval-only；但训练优先。

```text
若与 N3PF-MS 同时启动：
1. N4-O lower priority；
2. 若训练 fps 下降 >15% 或 CPU/IO/memory 不安全，N4-O 暂停或排队；
3. 独立 result dir / logs / locks / complete-stop flags；
4. 不干扰 N3PF-MS checkpoint 写入。
```

## 9. Outputs

```text
results/env_v2_phase_n4o_ordinary_shield_fair_comparison/
PHASE_N4O_ORDINARY_SHIELD_FAIR_COMPARISON_REPORT.md
PHASE_N4O_ORDINARY_SHIELD_FAIR_COMPARISON_COMPLETE.flag
phase_n4o_status.txt
phase_n4o_watcher.log
```

Tables：

```text
phase_n4o_checkpoint_manifest.csv
phase_n4o_shield_config_manifest.csv
phase_n4o_command_manifest.csv
phase_n4o_eval_summary_by_seed.csv
phase_n4o_eval_summary_aggregate.csv
phase_n4o_no_shield_reference_comparison.csv
phase_n4o_same_shield_comparison.csv
phase_n4o_phase_b_context_comparison.csv
phase_n4o_intervention_summary.csv
phase_n4o_raw_vs_filtered_safety.csv
phase_n4o_scenario_breakdown.csv
phase_n4o_motion_mode_breakdown.csv
phase_n4o_threat_class_breakdown.csv
phase_n4o_failure_cases.csv
phase_n4o_decision.csv
phase_n4o_schema_check.csv
```

## 10. Stop flags

```text
PHASE_N4O_STOP_CHECKPOINT_MISSING.flag
PHASE_N4O_STOP_PHASE_B_CONFIG_MISSING.flag
PHASE_N4O_STOP_SHIELD_CONFIG_MISMATCH.flag
PHASE_N4O_STOP_SIGMA_USED_BY_ORDINARY_SHIELD.flag
PHASE_N4O_STOP_EVAL_FAILED.flag
PHASE_N4O_STOP_SCHEMA_INVALID.flag
PHASE_N4O_STOP_WATCHER_FAILED.flag
```

## 11. Watcher 要求

必须创建阻塞式 watcher。只有 complete flag 或 stop flag 触发时才能结束。

```text
poll interval: about 120 seconds
chat/status heartbeat: about 300 seconds or longer
```

不要 60 秒刷屏。checkpoint/path missing、shield mismatch、sigma accidentally used、eval failed、schema invalid、complete/stop 必须及时输出。
