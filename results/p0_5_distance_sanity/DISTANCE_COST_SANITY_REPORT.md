# Distance Cost Sanity Report

## Scope
- d_warning = 1.0.
- Recomputed formula: max(0, d_warning - min_distance) ** 2.
- Covered existing step=750000 eval_sudden_turn traces for attention_full, attention_full_distance_penalty, and attention_full_risk_penalty.

## Summary By Method
| method | episodes | min(min_distance) | triggered episodes | mismatch rows | bug rows | zero-reasonable rows |
|---|---:|---:|---:|---:|---:|---:|
| attention_full | 10 | 0.8210 | 6 | 0 | 0 | 4 |
| attention_full_distance_penalty | 10 | 0.7641 | 5 | 0 | 0 | 5 |
| attention_full_risk_penalty | 10 | 1.3665 | 0 | 0 | 0 | 10 |

## Judgment
- No cost computation or trace recording mismatch was found in the covered traces.
- 19/30 episodes have min(min_distance) > 1.0 and both trace/recomputed distance_warning_cost equal to 0.

## Episode Rows
| method | episode | min_distance | trace_max | count_dist_lt_1 | recomputed_max | max_abs_diff | judgment |
|---|---:|---:|---:|---:|---:|---:|---|
| attention_full | 0 | 1.1349 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full | 1 | 0.8210 | 0.0321 | 2 | 0.0321 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full | 2 | 0.9751 | 0.0006 | 2 | 0.0006 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full | 3 | 0.8860 | 0.0130 | 2 | 0.0130 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full | 4 | 1.0345 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full | 5 | 1.0508 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full | 6 | 0.8218 | 0.0317 | 2 | 0.0317 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full | 7 | 1.1731 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full | 8 | 0.9193 | 0.0065 | 1 | 0.0065 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full | 9 | 0.9561 | 0.0019 | 2 | 0.0019 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_distance_penalty | 0 | 1.2484 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_distance_penalty | 1 | 0.7641 | 0.0557 | 3 | 0.0557 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_distance_penalty | 2 | 1.2829 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_distance_penalty | 3 | 0.9737 | 0.0007 | 1 | 0.0007 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_distance_penalty | 4 | 1.1467 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_distance_penalty | 5 | 1.1598 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_distance_penalty | 6 | 0.9366 | 0.0040 | 1 | 0.0040 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_distance_penalty | 7 | 1.1211 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_distance_penalty | 8 | 0.9569 | 0.0019 | 1 | 0.0019 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_distance_penalty | 9 | 0.9335 | 0.0044 | 2 | 0.0044 | 0.00000000 | OK_TRIGGERED_CONSISTENT |
| attention_full_risk_penalty | 0 | 2.1310 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 1 | 1.3665 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 2 | 2.0681 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 3 | 2.0776 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 4 | 2.7553 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 5 | 1.7400 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 6 | 1.6492 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 7 | 2.3569 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 8 | 1.5191 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |
| attention_full_risk_penalty | 9 | 1.9280 | 0.0000 | 0 | 0.0000 | 0.00000000 | OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING |

## Artifacts
- results/p0_5_distance_sanity/distance_cost_sanity.csv
