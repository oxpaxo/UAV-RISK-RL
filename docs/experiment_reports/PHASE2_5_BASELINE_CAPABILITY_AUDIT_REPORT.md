# Phase 2.5 Baseline Capability Audit Report

## 1. Executive Summary
terminal_decision = reactive_prior_promising
secondary_diagnosis = baseline_learns_progress_but_not_safety
next_recommended_action = 优先做 reactive prior + learned residual / velocity-obstacle-like safety filter / RL constrained by simple geometric avoider；不继续追 no-response。

attention_full best checkpoint is 1500000. It clearly learns task progress relative to random/straight_line (success=0.6100, progress=0.9572), but collision remains high (collision=0.3900, near_miss=0.9667). Reactive is much safer (collision=0.0867) but has lower success.

## 2. Inputs
- Phase 1 sanity episode data: `results/restart_phase0_phase1/env_v2/env_v2_sanity.csv`
- Phase 1 sanity summary: `results/restart_phase0_phase1/env_v2/env_v2_sanity_by_policy_scenario.csv`
- Phase 2 final report: `PHASE2_BASELINE_LONGTRAIN_FINAL_REPORT.md`
- Phase 2 checkpoint summary: `results/env_v2_phase2/baseline_longtrain_by_checkpoint_scenario.csv`
- Phase 2 episode metrics: `results/env_v2_phase2/baseline_longtrain_episode_metrics.csv`
- Phase 2 reaction metrics: `results/env_v2_phase2/baseline_longtrain_reaction_breakdown.csv`
- Phase 2 threat metrics: `results/env_v2_phase2/baseline_longtrain_threat_metrics.csv`

## 3. Attention_full Learning Audit
attention_full learns the basic task: success and progress improve strongly over early training, but the checkpoint curve is not monotonic. 1250k regresses before 1500k recovers, so there is checkpoint oscillation.
| checkpoint_step | success_rate_mean | collision_rate_mean | near_miss_rate_mean | progress_mean | mean_min_distance_mean | min_distance_after_threat_mean | scenario_best_count | is_global_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250000 | 0.1133 | 0.8867 | 0.9967 | 0.5177 | 3.6696 | 0.6712 | 0 | False |
| 500000 | 0.3067 | 0.6933 | 0.9767 | 0.9095 | 5.1235 | 0.8117 | 0 | False |
| 750000 | 0.5233 | 0.4767 | 0.9567 | 0.9472 | 5.1616 | 0.9635 | 2 | False |
| 1000000 | 0.5133 | 0.4867 | 0.9367 | 0.9538 | 5.1200 | 0.9442 | 1 | False |
| 1250000 | 0.4267 | 0.5733 | 0.9667 | 0.9114 | 4.2706 | 0.8319 | 0 | False |
| 1500000 | 0.6100 | 0.3900 | 0.9667 | 0.9572 | 4.8633 | 0.9355 | 3 | True |

## 4. Comparison with Random / Straight-Line / Reactive
attention_full is clearly better than random and straight_line on success/progress. Compared with reactive, it has much higher success (0.6100 vs 0.0967) but much worse collision (0.3900 vs 0.0867). This indicates a strong simple-geometry safety prior that the learned policy does not reproduce.
| method | checkpoint_step | success_rate | collision_rate | near_miss_rate | progress | mean_min_distance | min_distance_after_threat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | sanity | 0.0000 | 0.0000 | 0.0400 | 0.0912 | 7.1353 | 3.5191 |
| straight_line | sanity | 0.0433 | 0.9567 | 0.0433 | 0.3548 | 3.9183 | 0.4969 |
| reactive | sanity | 0.0967 | 0.0867 | 0.2967 | 0.9351 | 3.7406 | 1.5646 |
| attention_full_1500k | 1500000 | 0.6100 | 0.3900 | 0.9667 | 0.9572 | 4.8633 | 0.9355 |
| attention_full_best_checkpoint | 1500000 | 0.6100 | 0.3900 | 0.9667 | 0.9572 | 4.8633 | 0.9355 |

Reactive vs attention collision gaps by scenario:
| scenario | reactive_collision_rate | attention_collision_rate | collision_gap_attention_minus_reactive | reactive_success_rate | attention_success_rate |
| --- | --- | --- | --- | --- | --- |
| eval_flow_high_density | 0.0200 | 0.3600 | 0.3400 | 0.0200 | 0.6400 |
| eval_flow_high_speed | 0.0400 | 0.4600 | 0.4200 | 0.1000 | 0.5400 |
| eval_flow_high_threat | 0.1400 | 0.3800 | 0.2400 | 0.0800 | 0.6200 |
| eval_flow_id | 0.1800 | 0.4200 | 0.2400 | 0.2200 | 0.5800 |
| eval_flow_mixed_ood | 0.1200 | 0.4600 | 0.3400 | 0.0600 | 0.5400 |
| eval_flow_sudden_threat | 0.0200 | 0.2600 | 0.2400 | 0.1000 | 0.7400 |

## 5. Scenario Difficulty
Hardest scenarios at the best checkpoint by collision count/rate: eval_flow_high_speed, eval_flow_mixed_ood, eval_flow_id.
Failures are not dominated by sudden_threat; sudden_threat is the easiest by collision at the best checkpoint.
| scenario | success_rate | collision_rate | near_miss_rate | mean_min_distance | min_distance_after_threat | progress | no_response_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval_flow_high_speed | 0.5400 | 0.4600 | 1.0000 | 4.4615 | 0.7574 | 0.9578 | 0.0000 |
| eval_flow_mixed_ood | 0.5400 | 0.4600 | 0.9800 | 4.8153 | 0.8787 | 0.9403 | 0.0000 |
| eval_flow_id | 0.5800 | 0.4200 | 0.9200 | 5.3711 | 1.0004 | 0.9634 | 0.0000 |
| eval_flow_high_threat | 0.6200 | 0.3800 | 0.9600 | 4.9312 | 0.9171 | 0.9687 | 0.0000 |
| eval_flow_high_density | 0.6400 | 0.3600 | 0.9600 | 4.6375 | 1.1385 | 0.9425 | 0.0000 |
| eval_flow_sudden_threat | 0.7400 | 0.2600 | 0.9800 | 4.9631 | 0.9208 | 0.9707 | 0.0000 |

Collision share by scenario:
| scenario | collision_count | collision_rate | collision_share | success_rate | progress |
| --- | --- | --- | --- | --- | --- |
| eval_flow_high_speed | 23 | 0.4600 | 0.1966 | 0.5400 | 0.9578 |
| eval_flow_mixed_ood | 23 | 0.4600 | 0.1966 | 0.5400 | 0.9403 |
| eval_flow_id | 21 | 0.4200 | 0.1795 | 0.5800 | 0.9634 |
| eval_flow_high_threat | 19 | 0.3800 | 0.1624 | 0.6200 | 0.9687 |
| eval_flow_high_density | 18 | 0.3600 | 0.1538 | 0.6400 | 0.9425 |
| eval_flow_sudden_threat | 13 | 0.2600 | 0.1111 | 0.7400 | 0.9707 |

## 6. Collision Breakdown
Threat/motion labels are available in episode-level Phase 2 data, so collision breakdowns use attention_full at the best checkpoint.

By threat class:
| threat_class | episodes | collision_count | collision_rate | collision_share | planned_cpa | planned_ttc |
| --- | --- | --- | --- | --- | --- | --- |
| high | 286 | 112 | 0.3916 | 0.9573 | 0.5475 | 4.3520 |
| medium | 14 | 5 | 0.3571 | 0.0427 | 1.7264 | 4.4672 |

By motion mode:
| motion_mode | episodes | collision_count | collision_rate | collision_share | planned_cpa | planned_ttc |
| --- | --- | --- | --- | --- | --- | --- |
| accel_decel | 58 | 29 | 0.5000 | 0.2479 | 0.5919 | 4.7710 |
| linear | 36 | 17 | 0.4722 | 0.1453 | 0.5687 | 4.9714 |
| sinusoidal_lateral | 72 | 28 | 0.3889 | 0.2393 | 0.5938 | 4.5252 |
| ar1_velocity | 85 | 28 | 0.3294 | 0.2393 | 0.6122 | 4.0281 |
| crossing_or_sudden_threat | 49 | 15 | 0.3061 | 0.1282 | 0.6359 | 3.7416 |

CPA/TTC binned failure correlation:
| variable | bin | episodes | collision_rate | value_mean | value_min | value_max |
| --- | --- | --- | --- | --- | --- | --- |
| planned_cpa | (0.349, 0.418] | 75.0000 | 0.4133 | 0.3795 | 0.3501 | 0.4166 |
| planned_cpa | (0.418, 0.507] | 75.0000 | 0.4800 | 0.4651 | 0.4186 | 0.5053 |
| planned_cpa | (0.507, 0.663] | 75.0000 | 0.3333 | 0.5785 | 0.5079 | 0.6625 |
| planned_cpa | (0.663, 2.17] | 75.0000 | 0.3333 | 0.9870 | 0.6633 | 2.1702 |
| planned_cpa_pearson | collision | 300.0000 | -0.0690 | 0.6025 | 0.3501 | 2.1702 |
| planned_ttc | (0.799, 3.052] | 75.0000 | 0.2133 | 1.1902 | 0.8000 | 3.0195 |
| planned_ttc | (3.052, 4.633] | 75.0000 | 0.5333 | 3.8744 | 3.0629 | 4.6276 |
| planned_ttc | (4.633, 6.208] | 75.0000 | 0.3867 | 5.3122 | 4.6385 | 6.2021 |
| planned_ttc | (6.208, 8.878] | 75.0000 | 0.4267 | 7.0528 | 6.2253 | 8.8778 |
| planned_ttc_pearson | collision | 300.0000 | 0.1752 | 4.3574 | 0.8000 | 8.8778 |

## 7. Interpretation
The EnvV2 issue is not the old no-response degradation. Phase 2 showed no_response_rate becomes zero after 500k. The current problem is capability mismatch: attention_full learns forward progress and reaches goals, but it does not achieve the safety margin of the simple reactive policy.
random is low-collision because it mostly fails to make progress, so it is not a capable baseline. straight_line is unsafe. reactive is safe but under-achieves success. attention_full is capable but unsafe.

## 8. Method Route Recommendation
recommended_next_step: 优先做 reactive prior + learned residual / velocity-obstacle-like safety filter / RL constrained by simple geometric avoider；不继续追 no-response。

not_recommended_next_steps:
- 继续追旧 3-ball no-response degradation 或 Phase 3 failure localization
- 单纯把 attention_full 训练到 2000k
- 在 seed=0 路线不清楚时启动 seed=1/2
- 直接上 temporal/risk-aware attention 或 PPO-Lagrangian
- 把当前结果包装成 benchmark 论文主线

No new PPO training is required to complete this audit. Additional data, if needed later, should be eval-only logging.

## 9. Decision
terminal_decision = reactive_prior_promising
next_recommended_action = 优先做 reactive prior + learned residual / velocity-obstacle-like safety filter / RL constrained by simple geometric avoider；不继续追 no-response。
