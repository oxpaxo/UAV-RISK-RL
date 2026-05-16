# Stage P0 Adaptation Validation Report

## Scope
- Variants: high_speed_obstacles and small_space.
- Methods: attention_full_distance_penalty_wide_d2 and attention_full_risk_penalty.
- Seed 0, 750k training, checkpoints 250k/500k/750k, 50 eval episodes per scenario.

## Variant Judgments at 750k
| variant | risk_pareto_scenarios | wide_failure | wide_overconservative | fixed_margin_still_strong | risk_adaptation_supported | wide_time | risk_time | wide_near | risk_near |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_speed | 3 | 0 | 0 | 1 | 0 | 9.1510 | 8.9040 | 0.0550 | 0.0725 |
| small_space | 5 | 0 | 0 | 1 | 0 | 7.1740 | 6.2080 | 0.1125 | 0.1575 |

## Required Answers
- high_speed: wide_d2 failure=False, overconservative=False, risk Pareto scenarios=3, risk_adaptation_supported=False.
- small_space: wide_d2 failure=False, overconservative=False, risk Pareto scenarios=5, risk_adaptation_supported=False.
