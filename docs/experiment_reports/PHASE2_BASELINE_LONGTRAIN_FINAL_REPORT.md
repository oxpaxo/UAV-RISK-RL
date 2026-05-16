# Phase 2 Baseline Long-Training Reproduction Report

## 1. Executive Summary

terminal_decision = phase2_no_go_no_reproduction
next_recommended_phase = blocked

## 2. Training Setup

attention_full seed=0 trained on DynamicObstacleFlowEnv/train_flow_mixed to 1500000 steps.
Checkpoints are saved under `checkpoints/env_v2_phase2/attention_full_s0`.

## 3. Evaluation Setup

Scenarios: eval_flow_id, eval_flow_high_density, eval_flow_high_speed, eval_flow_high_threat, eval_flow_mixed_ood, eval_flow_sudden_threat. Episodes per checkpoint/scenario: 50. eval_seed=1000.

## 4. Learning Sanity

Final mean success_rate=0.6100; final mean progress=0.9572.

## 5. Long-Training Curves

| step | scenario | success | collision | near_miss | no_response | mean_min_distance | min_distance_after_threat |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 250000 | eval_flow_high_density | 0.0400 | 0.9600 | 1.0000 | 0.0200 | 3.6793 | 0.6920 |
| 250000 | eval_flow_high_speed | 0.0200 | 0.9800 | 1.0000 | 0.0000 | 3.3965 | 0.5375 |
| 250000 | eval_flow_high_threat | 0.1200 | 0.8800 | 1.0000 | 0.0000 | 3.6609 | 0.7373 |
| 250000 | eval_flow_id | 0.2400 | 0.7600 | 0.9800 | 0.0600 | 3.9789 | 0.8002 |
| 250000 | eval_flow_mixed_ood | 0.0800 | 0.9200 | 1.0000 | 0.0000 | 3.3708 | 0.5742 |
| 250000 | eval_flow_sudden_threat | 0.1800 | 0.8200 | 1.0000 | 0.0200 | 3.9313 | 0.6859 |
| 500000 | eval_flow_high_density | 0.2600 | 0.7400 | 1.0000 | 0.0000 | 4.8703 | 0.9429 |
| 500000 | eval_flow_high_speed | 0.2400 | 0.7600 | 1.0000 | 0.0000 | 4.7704 | 0.5799 |
| 500000 | eval_flow_high_threat | 0.3000 | 0.7000 | 0.9800 | 0.0000 | 5.3660 | 0.9295 |
| 500000 | eval_flow_id | 0.3600 | 0.6400 | 0.9400 | 0.0000 | 5.5001 | 0.8328 |
| 500000 | eval_flow_mixed_ood | 0.3800 | 0.6200 | 0.9600 | 0.0000 | 4.9754 | 0.7992 |
| 500000 | eval_flow_sudden_threat | 0.3000 | 0.7000 | 0.9800 | 0.0000 | 5.2590 | 0.7862 |
| 750000 | eval_flow_high_density | 0.4400 | 0.5600 | 0.9800 | 0.0000 | 4.7487 | 1.0327 |
| 750000 | eval_flow_high_speed | 0.4400 | 0.5600 | 0.9200 | 0.0000 | 5.2633 | 0.8045 |
| 750000 | eval_flow_high_threat | 0.6600 | 0.3400 | 0.9200 | 0.0000 | 5.2994 | 0.9647 |
| 750000 | eval_flow_id | 0.6400 | 0.3600 | 0.9400 | 0.0000 | 5.3128 | 1.0685 |
| 750000 | eval_flow_mixed_ood | 0.4800 | 0.5200 | 1.0000 | 0.0000 | 5.0922 | 0.9000 |
| 750000 | eval_flow_sudden_threat | 0.4800 | 0.5200 | 0.9800 | 0.0000 | 5.2534 | 1.0105 |
| 1000000 | eval_flow_high_density | 0.3000 | 0.7000 | 1.0000 | 0.0000 | 4.7423 | 0.8338 |
| 1000000 | eval_flow_high_speed | 0.4400 | 0.5600 | 0.9600 | 0.0000 | 4.9665 | 0.7785 |
| 1000000 | eval_flow_high_threat | 0.6400 | 0.3600 | 0.9200 | 0.0000 | 5.3331 | 0.9869 |
| 1000000 | eval_flow_id | 0.5600 | 0.4400 | 0.9600 | 0.0000 | 5.4337 | 1.1034 |
| 1000000 | eval_flow_mixed_ood | 0.5400 | 0.4600 | 0.8400 | 0.0000 | 5.0070 | 1.0520 |
| 1000000 | eval_flow_sudden_threat | 0.6000 | 0.4000 | 0.9400 | 0.0000 | 5.2371 | 0.9105 |
| 1250000 | eval_flow_high_density | 0.4000 | 0.6000 | 1.0000 | 0.0000 | 3.9076 | 0.9468 |
| 1250000 | eval_flow_high_speed | 0.3000 | 0.7000 | 0.9800 | 0.0000 | 4.1402 | 0.6419 |
| 1250000 | eval_flow_high_threat | 0.4200 | 0.5800 | 0.9800 | 0.0000 | 4.3038 | 0.6965 |
| 1250000 | eval_flow_id | 0.5000 | 0.5000 | 0.9400 | 0.0000 | 4.6364 | 0.9823 |
| 1250000 | eval_flow_mixed_ood | 0.4600 | 0.5400 | 0.9400 | 0.0000 | 4.1786 | 0.8602 |
| 1250000 | eval_flow_sudden_threat | 0.4800 | 0.5200 | 0.9600 | 0.0000 | 4.4568 | 0.8636 |
| 1500000 | eval_flow_high_density | 0.6400 | 0.3600 | 0.9600 | 0.0000 | 4.6375 | 1.1385 |
| 1500000 | eval_flow_high_speed | 0.5400 | 0.4600 | 1.0000 | 0.0000 | 4.4615 | 0.7574 |
| 1500000 | eval_flow_high_threat | 0.6200 | 0.3800 | 0.9600 | 0.0000 | 4.9312 | 0.9171 |
| 1500000 | eval_flow_id | 0.5800 | 0.4200 | 0.9200 | 0.0000 | 5.3711 | 1.0004 |
| 1500000 | eval_flow_mixed_ood | 0.5400 | 0.4600 | 0.9800 | 0.0000 | 4.8153 | 0.8787 |
| 1500000 | eval_flow_sudden_threat | 0.7400 | 0.2600 | 0.9800 | 0.0000 | 4.9631 | 0.9208 |

## 6. Reaction Metric Breakdown

The no-response metric is action-based: after a planned threat window begins, the policy must emit continuous lateral/away-from-threat response. It is not physical completion latency.

| step | scenario | eval_style | conditional | nan_reaction_rate | no_response_rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 250000 | eval_flow_high_density | 2.3469 | 0.3167 | 0.0400 | 0.0200 |
| 250000 | eval_flow_high_speed | 0.3800 | 0.3800 | 0.0000 | 0.0000 |
| 250000 | eval_flow_high_threat | 0.3480 | 0.3480 | 0.0000 | 0.0000 |
| 250000 | eval_flow_id | 6.1120 | 0.3234 | 0.0600 | 0.0600 |
| 250000 | eval_flow_mixed_ood | 0.3200 | 0.3200 | 0.0000 | 0.0000 |
| 250000 | eval_flow_sudden_threat | 2.3440 | 0.3551 | 0.0200 | 0.0200 |
| 500000 | eval_flow_high_density | 0.2600 | 0.2600 | 0.0000 | 0.0000 |
| 500000 | eval_flow_high_speed | 0.2040 | 0.2040 | 0.0000 | 0.0000 |
| 500000 | eval_flow_high_threat | 0.2520 | 0.2520 | 0.0000 | 0.0000 |
| 500000 | eval_flow_id | 0.3160 | 0.3160 | 0.0000 | 0.0000 |
| 500000 | eval_flow_mixed_ood | 0.2400 | 0.2400 | 0.0000 | 0.0000 |
| 500000 | eval_flow_sudden_threat | 0.2680 | 0.2680 | 0.0000 | 0.0000 |
| 750000 | eval_flow_high_density | 0.2600 | 0.2600 | 0.0000 | 0.0000 |
| 750000 | eval_flow_high_speed | 0.2040 | 0.2040 | 0.0000 | 0.0000 |
| 750000 | eval_flow_high_threat | 0.2520 | 0.2520 | 0.0000 | 0.0000 |
| 750000 | eval_flow_id | 0.3680 | 0.3680 | 0.0000 | 0.0000 |
| 750000 | eval_flow_mixed_ood | 0.2400 | 0.2400 | 0.0000 | 0.0000 |
| 750000 | eval_flow_sudden_threat | 0.2680 | 0.2680 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_high_density | 0.2600 | 0.2600 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_high_speed | 0.2040 | 0.2040 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_high_threat | 0.2520 | 0.2520 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_id | 0.3680 | 0.3680 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_mixed_ood | 0.2400 | 0.2400 | 0.0000 | 0.0000 |
| 1000000 | eval_flow_sudden_threat | 0.2680 | 0.2680 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_high_density | 0.2600 | 0.2600 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_high_speed | 0.2040 | 0.2040 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_high_threat | 0.2520 | 0.2520 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_id | 0.3680 | 0.3680 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_mixed_ood | 0.2400 | 0.2400 | 0.0000 | 0.0000 |
| 1250000 | eval_flow_sudden_threat | 0.2680 | 0.2680 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_high_density | 0.2600 | 0.2600 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_high_speed | 0.2040 | 0.2040 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_high_threat | 0.2520 | 0.2520 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_id | 0.3680 | 0.3680 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_mixed_ood | 0.2400 | 0.2400 | 0.0000 | 0.0000 |
| 1500000 | eval_flow_sudden_threat | 0.2680 | 0.2680 | 0.0000 | 0.0000 |

## 7. Scenario-Wise Findings

- No scenario met the configured reproduction thresholds.

## 8. Reproduction Decision

NO-GO triggered: phase2_no_go_no_reproduction
Triggering metrics: {'max_step': 1500000, 'first_step': 250000, 'hit_scenarios': [], 'hit_scenario_count': 0, 'mean_final_success_rate': 0.61, 'mean_final_progress': 0.9572205441305308, 'peak_no_response_rate': 0.06, 'peak_near_miss_rate': 1.0, 'min_mean_min_distance': 3.3707589786983103}
Cannot enter Phase 3 because Phase 2 did not establish a reproducible baseline degradation signal.

## 9. Interpretation

If reproduction is strong or weak, the result supports investigating no-response / safety-margin erosion in Env V2. Claims remain bounded by seed=0 and the scenario coverage above.

## 10. Next Step

Enter Phase 3 failure localization if complete; otherwise fix training/eval/env or re-evaluate the direction according to the no-go reason.
