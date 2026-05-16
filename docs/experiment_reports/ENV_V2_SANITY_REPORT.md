# ENV V2 Sanity Report

## Run Configuration

- env: DynamicObstacleFlowEnv
- scenarios: eval_flow_id,eval_flow_high_density,eval_flow_high_speed,eval_flow_high_threat,eval_flow_mixed_ood,eval_flow_sudden_threat
- policies: random,straight_line,reactive
- episodes per policy/scenario: 50
- eval_seed: 1000
- dt: 0.2
- max_steps: 500

## Overall Metrics

- threat_valid_rate: 1.0000
- replacement_count_mean: 84.5867
- init_collision_rate: 0.0000
- nan_or_crash: 0
- out_of_bounds_rate: 0.0000
- random_success_rate: 0.0000
- straight_success_rate: 0.0433
- straight_collision_rate: 0.9567
- reactive_success_rate: 0.0967
- reactive_collision_rate: 0.0867
- reactive_success_advantage: 0.0533
- reactive_collision_reduction: 0.8700
- high_cpa_mean: 0.7492
- low_cpa_mean: 3.5018
- high_low_cpa_gap: 2.7526

## Policy/Scenario Summary

| policy | scenario | episodes | success | collision | near_miss | min_distance_mean | replacement_mean | threat_valid | active_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | eval_flow_high_density | 50 | 0.0000 | 0.0000 | 0.0400 | 3.2906 | 115.8200 | 1.0000 | 9.0800 |
| random | eval_flow_high_speed | 50 | 0.0000 | 0.0000 | 0.1200 | 2.4824 | 189.6800 | 1.0000 | 6.3800 |
| random | eval_flow_high_threat | 50 | 0.0000 | 0.0000 | 0.0000 | 3.6563 | 71.2600 | 1.0000 | 6.6600 |
| random | eval_flow_id | 50 | 0.0000 | 0.0000 | 0.0000 | 4.5746 | 88.8400 | 1.0000 | 6.2600 |
| random | eval_flow_mixed_ood | 50 | 0.0000 | 0.0000 | 0.0600 | 3.2706 | 90.2400 | 1.0000 | 6.6800 |
| random | eval_flow_sudden_threat | 50 | 0.0000 | 0.0000 | 0.0200 | 3.8402 | 73.3600 | 1.0000 | 6.3600 |
| reactive | eval_flow_high_density | 50 | 0.0200 | 0.0200 | 0.1400 | 1.8795 | 152.3800 | 1.0000 | 8.9600 |
| reactive | eval_flow_high_speed | 50 | 0.1000 | 0.0400 | 0.3400 | 1.5620 | 220.8600 | 1.0000 | 6.6600 |
| reactive | eval_flow_high_threat | 50 | 0.0800 | 0.1400 | 0.3000 | 1.4603 | 111.3800 | 1.0000 | 6.2400 |
| reactive | eval_flow_id | 50 | 0.2200 | 0.1800 | 0.2800 | 1.3534 | 106.8200 | 1.0000 | 6.3400 |
| reactive | eval_flow_mixed_ood | 50 | 0.0600 | 0.1200 | 0.4200 | 1.4512 | 121.0800 | 1.0000 | 6.3400 |
| reactive | eval_flow_sudden_threat | 50 | 0.1000 | 0.0200 | 0.3000 | 1.6810 | 110.4000 | 1.0000 | 6.7400 |
| straight_line | eval_flow_high_density | 50 | 0.0400 | 0.9600 | 0.0400 | 0.5033 | 13.0800 | 1.0000 | 9.0600 |
| straight_line | eval_flow_high_speed | 50 | 0.0000 | 1.0000 | 0.0000 | 0.4705 | 14.0200 | 1.0000 | 6.5800 |
| straight_line | eval_flow_high_threat | 50 | 0.0000 | 1.0000 | 0.0000 | 0.4894 | 6.5400 | 1.0000 | 6.3200 |
| straight_line | eval_flow_id | 50 | 0.0400 | 0.9600 | 0.0400 | 0.4960 | 13.0800 | 1.0000 | 6.3800 |
| straight_line | eval_flow_mixed_ood | 50 | 0.0600 | 0.9400 | 0.0600 | 0.4943 | 10.1200 | 1.0000 | 6.3200 |
| straight_line | eval_flow_sudden_threat | 50 | 0.1200 | 0.8800 | 0.1200 | 0.5280 | 13.6000 | 1.0000 | 6.6800 |

## Gate Checks

- random policy not 100% success: True
- straight-line not 100% success: True
- reactive clearly better than straight-line: True
- threat_valid_rate >= 0.8: True
- high-threat CPA lower than low-threat CPA: True
- replacement_count mean > 0: True
- init_collision_rate close to 0: True
- nan_or_crash = 0: True

terminal_decision = phase0_phase1_complete
