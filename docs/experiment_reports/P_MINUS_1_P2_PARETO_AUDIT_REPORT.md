# Stage P-1 P2 Pareto Audit Report

## Scope
- Source: results/p2_rich_motion/p2_three_seed_summary.csv when available.
- Compared attention_full_risk_penalty against attention_full_distance_penalty_wide_d2.
- Primary split: scenario / seed / checkpoint at 750k and 1000k.

## Classification Summary
- Pareto-positive scenarios: eval_accel_decel, eval_ar1, eval_mixed_v2, eval_random_switch, eval_random_switch_hard, eval_sinusoidal, eval_sudden_turn, eval_threat_validated_sudden
- Majority wide_d2-dominated scenarios: none
- P-1 no-go triggered: False

| scenario | rows | pareto_positive | efficiency_safety_gap | wide_d2_dominates | no_difference | d_time | d_near | d_collision | d_min_distance | d_reaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval_accel_decel | 6 | 6 | 0 | 0 | 0 | -0.4053 | -0.0033 | 0.0000 | -0.1941 | nan |
| eval_ar1 | 6 | 4 | 2 | 0 | 0 | -0.4520 | -0.0033 | 0.0067 | -0.2193 | nan |
| eval_mixed_v2 | 6 | 2 | 1 | 1 | 2 | -0.1480 | 0.0133 | 0.0000 | -0.1536 | 0.0867 |
| eval_random_switch | 6 | 5 | 0 | 0 | 1 | -0.2760 | 0.0000 | 0.0000 | -0.1439 | nan |
| eval_random_switch_hard | 6 | 3 | 3 | 0 | 0 | -1.3487 | 0.0500 | -0.0033 | -0.3124 | nan |
| eval_sinusoidal | 6 | 5 | 1 | 0 | 0 | -0.4060 | 0.0067 | 0.0033 | -0.2295 | nan |
| eval_sudden_turn | 6 | 6 | 0 | 0 | 0 | -0.4253 | 0.0033 | 0.0000 | -0.2348 | 0.4267 |
| eval_threat_validated_sudden | 6 | 3 | 1 | 1 | 1 | -0.2240 | -0.0000 | 0.0000 | -0.1430 | 0.0367 |

## Required Answers
1. risk Pareto-positive evidence appears in: eval_accel_decel, eval_ar1, eval_mixed_v2, eval_random_switch, eval_random_switch_hard, eval_sinusoidal, eval_sudden_turn, eval_threat_validated_sudden.
2. wide_d2 majority dominance appears in: none.
3. Whether risk is merely faster with safety loss is shown by risk_efficiency_but_safety_gap counts above.
4. 750k and 1000k consistency is available in p2_pareto_classification.csv.
5. Seed-level stability is available in p2_delta_by_seed_step_scenario.csv.
6. Continue to P0/P0.5: True.
