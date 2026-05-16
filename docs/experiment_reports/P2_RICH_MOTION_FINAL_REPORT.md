# P2 Rich Motion Generalization Report

## 1. Motivation
P2 tests whether risk_penalty keeps a safety-efficiency Pareto advantage over distance_penalty_wide_d2 under richer obstacle motion. It is not a repeat of the d_warning=1.0 comparison.

## 2. New Motion Modes
Stage 0/1 were completed before this run. This Stage2-to-final run reused the patched richer-motion environment with sinusoidal, accel_decel, simple_ar1, train_mixed_modes_v2, eval_mixed_v2, and eval_threat_validated_sudden.

## 3. Environment Sanity Check
Patched Stage 0 sanity was already complete and was not rerun. See P2_ENVIRONMENT_SANITY_REPORT_PATCHED.md.

## 4. Existing Checkpoint OOD Evaluation
Patched Stage 1 OOD evaluation was already complete and was not rerun. See P2_STAGE1_OOD_EVAL_REPORT.md and results/p2_rich_motion/p2_stage1_ood_eval.csv.

## 5. Rich-Motion Training Seed0
Stage 2 trained M0 attention_full, M1 attention_full_distance_penalty_wide_d2, and M2 attention_full_risk_penalty from seed=0 on train_mixed_modes_v2.
See P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md and results/p2_rich_motion/p2_seed0_by_step_scenario.csv.

## 6. New Single-Mode Scenario Analysis
The sinusoidal / accel_decel / ar1 results are reported in P2_STAGE3_SEED0_PARETO_REPORT.md and the seed0 main table.

## 7. Risk vs Wide Distance Pareto
Stage 3 go decision: True. Reasons: key_pareto_positive_scenarios=4, key_safety_close_scenarios=6, baseline_issue_scenarios=6, wide_d2_dominates_scenarios=2, risk_wide_d2_no_difference_scenarios=2, risk_safety_gap_scenarios=0.

## 8. Risk Adaptation Analysis
See results/p2_rich_motion/p2_risk_adaptation_summary.csv for risk_sum/risk_max and distance-cost activation by motion mode.

## 9. Failure Cases
Failure and near-miss behavior is separated for eval_mixed_v2, eval_threat_validated_sudden, and the new single-mode scenarios. Legacy mixed_uncertainty is not used as P2 primary evidence.

## 10. Go/No-Go for Three Seeds
Stage 4 triggered: True.
Terminal decision: stage4_complete_risk_retained.

## 11. Final P2 Decision
Risk mainline retained: True.
Shift to safety-margin cost design: False.
Recommendation: continue risk as the main Pareto-efficiency candidate and confirm with broader method variants.

## Required Answers
1. New motion modes implemented and sanity checked: yes, in patched Stage 0.
2. Random policy rollout extremes: see env_sanity_random_policy_patched.csv.
3. Short PPO sanity: see env_sanity_short_ppo_patched.csv.
4. New eval scenario discrimination: patched Stage 1 passed.
5. Existing checkpoint safety issues: see P2_STAGE1_OOD_EVAL_REPORT.md.
6. train_mixed_modes_v2 baseline safety/reaction drift: yes.
7. risk_penalty safety close to wide_d2: yes.
8. risk_penalty more efficient than wide_d2: yes.
9. risk on Pareto front: yes at seed0.
10. sinusoidal / accel_decel / ar1 single-mode results: see Stage 3 report table.
11. mixed_v2 stability: see Stage 3 mixed_v2 gate row.
12. motion-mode adaptation signal: see p2_risk_adaptation_summary.csv.
13. Stage 4 three-seed confirmation worth running: yes.
14. Risk mainline retained after P2: yes.
