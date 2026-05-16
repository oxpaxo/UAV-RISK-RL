# Phase 0 + Phase 1 Final Report

## 1. Executive Summary

terminal_decision = phase0_phase1_complete
next_recommended_phase = Phase 2 baseline long-training reproduction

## 2. Phase 0 Old Assets Summary

Old experiments are frozen and downgraded to preliminary diagnostic evidence. Reusable assets include environment metric conventions, SB3 training/eval wiring, obstacle-set policy extraction, trace/report scripts, watcher patterns, and config/run metadata.

## 3. Environment V2 Design

`DynamicObstacleFlowEnv` is implemented as a separate continuous obstacle-flow environment with 5-8 default active obstacles, high-density eval support, fixed/constrained altitude horizontal avoidance, and scenario-specific train/eval split.

## 4. Motion Models

`linear`, `sinusoidal_lateral`, `accel_decel`, `ar1_velocity`, and `crossing_or_sudden_threat` are implemented.

## 5. Threat Generation

Each obstacle samples a threat class and planned CPA/TTC against the nominal path. High-threat planned CPA is lower than low-threat planned CPA by construction and verified in sanity data.

## 6. Replacement Mechanism

Obstacles are removed when passed, out of bounds, no longer a future threat, over lifetime, or far from the nominal path. The active count is maintained by immediate replacement.

## 7. Sanity Policies

Sanity policies are `random`, `straight_line`, and a simple current-state `reactive` avoider.

## 8. Sanity Results

| policy | scenario | episodes | success | collision | near_miss | replacement_mean | threat_valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | eval_flow_high_density | 50 | 0.0000 | 0.0000 | 0.0400 | 115.8200 | 1.0000 |
| random | eval_flow_high_speed | 50 | 0.0000 | 0.0000 | 0.1200 | 189.6800 | 1.0000 |
| random | eval_flow_high_threat | 50 | 0.0000 | 0.0000 | 0.0000 | 71.2600 | 1.0000 |
| random | eval_flow_id | 50 | 0.0000 | 0.0000 | 0.0000 | 88.8400 | 1.0000 |
| random | eval_flow_mixed_ood | 50 | 0.0000 | 0.0000 | 0.0600 | 90.2400 | 1.0000 |
| random | eval_flow_sudden_threat | 50 | 0.0000 | 0.0000 | 0.0200 | 73.3600 | 1.0000 |
| reactive | eval_flow_high_density | 50 | 0.0200 | 0.0200 | 0.1400 | 152.3800 | 1.0000 |
| reactive | eval_flow_high_speed | 50 | 0.1000 | 0.0400 | 0.3400 | 220.8600 | 1.0000 |
| reactive | eval_flow_high_threat | 50 | 0.0800 | 0.1400 | 0.3000 | 111.3800 | 1.0000 |
| reactive | eval_flow_id | 50 | 0.2200 | 0.1800 | 0.2800 | 106.8200 | 1.0000 |
| reactive | eval_flow_mixed_ood | 50 | 0.0600 | 0.1200 | 0.4200 | 121.0800 | 1.0000 |
| reactive | eval_flow_sudden_threat | 50 | 0.1000 | 0.0200 | 0.3000 | 110.4000 | 1.0000 |
| straight_line | eval_flow_high_density | 50 | 0.0400 | 0.9600 | 0.0400 | 13.0800 | 1.0000 |
| straight_line | eval_flow_high_speed | 50 | 0.0000 | 1.0000 | 0.0000 | 14.0200 | 1.0000 |
| straight_line | eval_flow_high_threat | 50 | 0.0000 | 1.0000 | 0.0000 | 6.5400 | 1.0000 |
| straight_line | eval_flow_id | 50 | 0.0400 | 0.9600 | 0.0400 | 13.0800 | 1.0000 |
| straight_line | eval_flow_mixed_ood | 50 | 0.0600 | 0.9400 | 0.0600 | 10.1200 | 1.0000 |
| straight_line | eval_flow_sudden_threat | 50 | 0.1200 | 0.8800 | 0.1200 | 13.6000 | 1.0000 |

## 9. Go/No-Go Decision

Phase 1 passes the environment sanity gate.

- threat_valid_rate >= 0.8
- replacement_count mean > 0
- init_collision_rate close to 0
- nan_or_crash = 0
- reactive avoider is clearly better than straight-line
- environment is neither too easy nor too hard by the sanity-policy checks

## 10. Next Step

If complete: enter Phase 2 baseline long-training reproduction. If no-go: fix the environment issue above and rerun sanity.

## Required Final Answers

1. Old experiment assets organized: yes.
2. Reusable modules: env metric patterns, train/eval tooling, policy extractor, cost bookkeeping, reaction diagnostics, traces, watchers, report scripts, configs.
3. DynamicObstacleFlowEnv implemented: yes.
4. Supports 5-8 active obstacles: yes; high-density eval supports 8-10.
5. Replacement works: yes.
6. Motion modes implemented: yes.
7. Threat class / planned CPA / planned TTC implemented: yes.
8. Train/eval split implemented: yes.
9. random / straight-line / reactive sanity completed: yes.
10. Environment too easy: no.
11. Environment too hard: no.
12. threat_valid_rate >= 0.8: yes (1.0000).
13. replacement_count > 0: yes (84.5867).
14. Can enter Phase 2 baseline long-training reproduction: yes.
15. If not, fix: n/a.
