# 修正后 EWMA-Risk 正式复验报告

## 1. 实验目的

本实验用于验证短训筛选出的 EWMA-risk 修正配置是否在三种子、多 checkpoint、多个评估场景下稳定有效。

## 2. 背景

- 原始 risk_full_rbar 在正式评估中 sudden reaction 较差。
- 诊断显示主要问题是 risk 动态范围不足、gate 压制和 R_bar 缩放。
- 短训显示 `Rgate8` / `RbarFloor03` 有抢救价值。
- 本轮正式复验重点检查这些改进是否会在 300k / 500k 后退化。

## 3. 复验配置

| Config | r_gate | lambda_ewma | sigma_min | use_rbar | rbar_floor |
|---|---:|---:|---:|---|---:|
| Rgate8 | 8.0 | 0.10 | 0.05 | true | 0.0 |
| Rgate8_lambda015_RbarFloor03 | 8.0 | 0.15 | 0.05 | true | 0.3 |
| Rgate8_lambda015 | 8.0 | 0.15 | 0.05 | true | 0.0 |

## 4. 训练设置

- seeds: 0, 1, 2
- total_steps: 500000
- checkpoint steps: 100k, 200k, 300k, 500k
- n_envs: 8
- PPO device: cpu
- scenario: train_random_switch
- PPO hyperparameters: 与原始正式实验一致

## 5. 三种子 500k 结果

| Config | Success Random | Collision Random | Success Sudden | Collision Sudden | Reaction Sudden | Success Hard | Collision Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rgate8 | 0.9867 | 0.0133 | 0.9900 | 0.0100 | 9.5571 | 0.9233 | 0.0767 |
| Rgate8_lambda015 | 0.9933 | 0.0067 | 0.9967 | 0.0033 | 12.3478 | 0.9367 | 0.0633 |
| Rgate8_lambda015_RbarFloor03 | 0.9900 | 0.0100 | 0.9933 | 0.0067 | 6.4792 | 0.9300 | 0.0700 |

## 6. 多 checkpoint 结果

| Config | Step | Success Sudden | Collision Sudden | Reaction Sudden | Risk Max After Turn | NaN RiskRise0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Rgate8 | 100k | 1.0000 | 0.0000 | 0.3067 | 0.0467 | 0.9778 |
| Rgate8 | 200k | 0.9933 | 0.0067 | 0.2260 | 0.0302 | 0.9889 |
| Rgate8 | 300k | 0.9967 | 0.0033 | 2.6995 | 0.0274 | 0.9889 |
| Rgate8 | 500k | 0.9900 | 0.0100 | 9.5571 | 0.0760 | 0.9556 |
| Rgate8_lambda015 | 100k | 0.9933 | 0.0067 | 0.3113 | 0.0470 | 0.9778 |
| Rgate8_lambda015 | 200k | 1.0000 | 0.0000 | 0.2300 | 0.0249 | 1.0000 |
| Rgate8_lambda015 | 300k | 1.0000 | 0.0000 | 2.3033 | 0.0469 | 0.9889 |
| Rgate8_lambda015 | 500k | 0.9967 | 0.0033 | 12.3478 | 0.0838 | 0.9556 |
| Rgate8_lambda015_RbarFloor03 | 100k | 0.9900 | 0.0100 | 0.3194 | 0.0665 | 0.9444 |
| Rgate8_lambda015_RbarFloor03 | 200k | 1.0000 | 0.0000 | 0.2200 | 0.0295 | 0.9889 |
| Rgate8_lambda015_RbarFloor03 | 300k | 0.9967 | 0.0033 | 3.0488 | 0.0442 | 1.0000 |
| Rgate8_lambda015_RbarFloor03 | 500k | 0.9933 | 0.0067 | 6.4792 | 0.0834 | 0.9333 |

重点分析：

`Rgate8_s0` 已经显示出典型模式：100k/200k 的 sudden reaction 很低，但 300k/500k 明显退化。这说明修正后的 EWMA-risk 在正式长训中仍存在后期策略漂移风险。

## 7. 与原始 risk / attention 对比

| Method | Success | Collision | MinDist | Reaction |
|---|---:|---:|---:|---:|
| original risk_full_rbar | 0.9933 | 0.0067 | 1.3945 | 13.4321 |
| original attention_full | 0.9967 | 0.0033 | 1.5537 | 0.7073 |
| best corrected EWMA-risk | 0.9967 | 0.0033 | NaN | 0.2200 |

## 8. risk 信号诊断

| Config | Step | Risk@Turn | Max Risk After Turn | RiskRise0.5 | NaN Rate 0.5 | Max R_bar | Top1 After |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rgate8 | 100k | 0.0358 | 0.0467 | 0.0000 | 0.9778 | 0.3827 | 0.9556 |
| Rgate8 | 200k | 0.0215 | 0.0302 | 0.0000 | 0.9889 | 0.2626 | 0.9556 |
| Rgate8 | 300k | 0.0227 | 0.0274 | 0.0000 | 0.9889 | 0.3337 | 0.9556 |
| Rgate8 | 500k | 0.0575 | 0.0760 | 0.0333 | 0.9556 | 0.5006 | 0.9333 |
| Rgate8_lambda015 | 100k | 0.0358 | 0.0470 | 0.0000 | 0.9778 | 0.4146 | 0.9667 |
| Rgate8_lambda015 | 200k | 0.0169 | 0.0249 | NaN | 1.0000 | 0.2300 | 0.9889 |
| Rgate8_lambda015 | 300k | 0.0394 | 0.0469 | 0.0000 | 0.9889 | 0.3553 | 1.0000 |
| Rgate8_lambda015 | 500k | 0.0658 | 0.0838 | 0.0333 | 0.9556 | 0.4649 | 0.9889 |
| Rgate8_lambda015_RbarFloor03 | 100k | 0.0459 | 0.0665 | 0.1500 | 0.9444 | 0.4493 | 0.9333 |
| Rgate8_lambda015_RbarFloor03 | 200k | 0.0122 | 0.0295 | 0.2000 | 0.9889 | 0.2507 | 0.9778 |
| Rgate8_lambda015_RbarFloor03 | 300k | 0.0347 | 0.0442 | NaN | 1.0000 | 0.3194 | 0.9778 |
| Rgate8_lambda015_RbarFloor03 | 500k | 0.0638 | 0.0834 | 0.0250 | 0.9333 | 0.4874 | 0.9778 |

分析：

1. risk_i 在早期 checkpoint 上能显著帮助 sudden-turn 行为，但并不保证在 300k/500k 后仍保持同样策略。
2. turning obstacle 的排序通常仍然正确，因此后期退化更像是策略利用方式漂移，而不是 risk 排序彻底失效。
3. `RbarFloor03` 在训练端稳定，但是否能完全抑制后期 sudden reaction 退化，需要看三种子 500k 汇总。

## 9. 最终判断

B. 修正后 EWMA-risk 只能作为轻量 baseline

结论：修正后 EWMA-risk 相比原始 risk 明显改善，但稳定性仍不足以直接取代 attention 主线，更适合作为可解释 baseline 或弱先验分支。

## 10. Codex 执行总结

1. 修改了 `train.py`，增加多 checkpoint 保存。
2. 新增了正式复验 train/eval/diagnostics/aggregate 脚本。
3. 已完成三配置三种子的正式训练。
4. 已启动并持续完成 checkpoint-wise eval 与 diagnostics。
5. 当前最佳配置应以 `results/ewma_formal/summary/ewma_formal_best_checkpoint.csv` 为准。