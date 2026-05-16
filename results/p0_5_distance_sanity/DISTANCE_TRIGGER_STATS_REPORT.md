# Distance Trigger Stats Report

## Scope
- Existing seed=0 eval CSVs only; no new training was used for P0.5-B.
- Methods: attention_full, attention_full_distance_penalty_d1, attention_full_risk_penalty.
- Steps: 500000, 750000, 1000000.
- Scenarios: eval_random_switch, eval_sudden_turn.
- distance_warning_cost_* percentiles are computed over the 50 per-episode distance_warning_cost_max values.

## Core Answer
- attention_full: max nonzero episode rate across requested rows = 0.700.
- attention_full_distance_penalty_d1: max nonzero episode rate across requested rows = 0.440.
- attention_full_risk_penalty: max nonzero episode rate across requested rows = 0.000.
- At least one method/scenario has a meaningful distance_warning_cost trigger rate; inspect the detailed table.

## Detailed Rows
| method | step | scenario | nonzero_rate | cost_p95 | cost_max | min_dist_mean | min_dist_min | near_miss | success | collision | reaction_eval | nan_reaction_rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_full | 500000 | eval_random_switch | 0.100 | 0.0153 | 0.0515 | 1.3811 | 0.7731 | 0.100 | 1.000 | 0.000 | nan | nan |
| attention_full | 500000 | eval_sudden_turn | 0.220 | 0.0202 | 0.0579 | 1.1669 | 0.7594 | 0.220 | 1.000 | 0.000 | 1.8600 | 0.040 |
| attention_full | 750000 | eval_random_switch | 0.240 | 0.0196 | 0.0653 | 1.1767 | 0.7445 | 0.240 | 1.000 | 0.000 | nan | nan |
| attention_full | 750000 | eval_sudden_turn | 0.420 | 0.0319 | 0.0612 | 1.0405 | 0.7525 | 0.420 | 1.000 | 0.000 | 11.4240 | 0.300 |
| attention_full | 1000000 | eval_random_switch | 0.400 | 0.0352 | 0.1604 | 1.1278 | 0.5994 | 0.400 | 1.000 | 0.000 | nan | nan |
| attention_full | 1000000 | eval_sudden_turn | 0.700 | 0.0817 | 0.1003 | 0.9420 | 0.6833 | 0.700 | 1.000 | 0.000 | 6.0760 | 0.140 |
| attention_full_distance_penalty_d1 | 500000 | eval_random_switch | 0.060 | 0.0000 | 0.0233 | 1.5402 | 0.8475 | 0.060 | 1.000 | 0.000 | nan | nan |
| attention_full_distance_penalty_d1 | 500000 | eval_sudden_turn | 0.140 | 0.0048 | 0.1939 | 1.3306 | 0.5596 | 0.140 | 1.000 | 0.000 | 1.2320 | 0.020 |
| attention_full_distance_penalty_d1 | 750000 | eval_random_switch | 0.120 | 0.0169 | 0.0367 | 1.3091 | 0.8084 | 0.120 | 1.000 | 0.000 | nan | nan |
| attention_full_distance_penalty_d1 | 750000 | eval_sudden_turn | 0.320 | 0.0433 | 0.3297 | 1.1184 | 0.4258 | 0.320 | 0.980 | 0.020 | 11.0327 | 0.300 |
| attention_full_distance_penalty_d1 | 1000000 | eval_random_switch | 0.240 | 0.0220 | 0.0596 | 1.1789 | 0.7558 | 0.240 | 1.000 | 0.000 | nan | nan |
| attention_full_distance_penalty_d1 | 1000000 | eval_sudden_turn | 0.440 | 0.0403 | 0.0454 | 1.0466 | 0.7870 | 0.440 | 1.000 | 0.000 | 7.5360 | 0.180 |
| attention_full_risk_penalty | 500000 | eval_random_switch | 0.000 | 0.0000 | 0.0000 | 2.2206 | 1.4546 | 0.000 | 1.000 | 0.000 | nan | nan |
| attention_full_risk_penalty | 500000 | eval_sudden_turn | 0.000 | 0.0000 | 0.0000 | 2.0302 | 1.2589 | 0.000 | 1.000 | 0.000 | 0.2000 | 0.000 |
| attention_full_risk_penalty | 750000 | eval_random_switch | 0.000 | 0.0000 | 0.0000 | 2.0643 | 1.2241 | 0.000 | 1.000 | 0.000 | nan | nan |
| attention_full_risk_penalty | 750000 | eval_sudden_turn | 0.000 | 0.0000 | 0.0000 | 1.9129 | 1.0955 | 0.000 | 1.000 | 0.000 | 0.2080 | 0.000 |
| attention_full_risk_penalty | 1000000 | eval_random_switch | 0.000 | 0.0000 | 0.0000 | 2.0522 | 1.2894 | 0.000 | 1.000 | 0.000 | nan | nan |
| attention_full_risk_penalty | 1000000 | eval_sudden_turn | 0.000 | 0.0000 | 0.0000 | 1.8883 | 1.2303 | 0.000 | 1.000 | 0.000 | 1.0760 | 0.020 |

## Artifacts
- results/p0_5_distance_sanity/distance_trigger_stats.csv
