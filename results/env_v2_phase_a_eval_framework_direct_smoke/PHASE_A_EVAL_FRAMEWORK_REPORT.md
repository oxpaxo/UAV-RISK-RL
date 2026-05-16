# Phase A Eval Framework Report

## 1. Background And Goal

Phase A establishes a unified evaluation runner and trace schema for EnvV2 baseline auditing. It does not train a new policy and does not introduce a new method claim.

## 2. EnvV2-Core Freeze Statement

EnvV2-core was frozen in Phase A. This phase did not modify obstacle count ranges, motion modes, train/eval scenario definitions, action dynamics, reward function, collision/success/near-miss definitions, or termination logic. All changes were limited to evaluation infrastructure, policy/controller adapters, unified logging, trace schema, and watcher/report generation.

## 3. Added Or Modified Files

- `scripts/run_env_v2_phase_a_eval_framework.py`
- `scripts/watch_phase_a_eval_framework.sh`
- `results/env_v2_phase_a_eval_framework_direct_smoke/` result artifacts

## 4. Eval Runner Usage

```bash
python scripts/run_env_v2_phase_a_eval_framework.py \
  --out-dir results/env_v2_phase_a_eval_framework_direct_smoke \
  --num-episodes 1 \
  --scenarios eval_flow_id eval_flow_high_speed \
  --policies random straight_line cpa_reactive attention_full filtered_attention_full \
  --checkpoint checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip \
  --eval-seed 1000 \
  --write-traces
```

## 5. Supported Policies / Controllers

- `random`: action-space sample with fixed per-episode RNG.
- `straight_line`: normalized horizontal goal direction.
- `cpa_reactive`: current CPA-reactive logic reused from `scripts/run_env_v2_sanity.py`.
- `attention_full`: SB3 PPO attention checkpoint adapter.
- `filtered_attention_full`: Phase A minimal safety-filter wrapper around `attention_full`.

The minimal filter is for infrastructure validation only and is not a formal Phase B baseline.

## 6. Episode-Level Metrics Schema

`tables/phase_a_episode_metrics_sample.csv` uses 38 fixed fields: `method, policy_name, scenario, checkpoint_step, episode_id, episode_seed, success, collision, timeout, truncated, out_of_bounds, near_miss, progress, final_goal_distance, mean_time, episode_length_steps, episode_return, mean_min_distance, episode_min_distance, min_distance_after_threat, no_response, no_response_rate, reaction_time_eval_style, conditional_reaction_time, planned_cpa, planned_ttc, threat_class, motion_mode, replacement_count, active_obstacle_count, mean_action_norm, mean_action_delta, max_action_delta, filter_used, filter_trigger_count, filter_trigger_rate, mean_filter_delta_norm, max_filter_delta_norm`.

## 7. Per-Step Trace Schema

`traces/sample_<policy>_trace.csv` uses 59 fixed trace fields. `tables/phase_a_trace_schema.csv` records the trace and obstacle-long schema.

## 8. Full Active Obstacle Set Logging

`tables/phase_a_step_obstacles_sample.csv` uses long-table logging. Every per-step active obstacle is recorded with slot, id, position, velocity, distance, closing speed, planned CPA/TTC, threat class, motion mode, and risk value.

## 9. Safety Filter Trace Fields

`filtered_attention_full` records `action_raw`, `action_filtered`, `action_executed`, `filter_triggered`, `filter_reason`, `filter_delta_norm`, `min_predicted_cpa_raw`, `min_predicted_cpa_filtered`, `min_ttc_raw`, `min_ttc_filtered`, and unsafe-obstacle metadata. These columns are reserved for Phase B formal safety-filter baselines and populated by the Phase A minimal filter where available.

## 10. Seed Rule And Fairness Note

`episode_seed = eval_seed + seed * 10000 + episode_id`, with `eval_seed=1000` and `seed=0`.

Because obstacle replacement depends on policy trajectory, identical reset seed does not guarantee identical obstacle schedules across policies. Phase A standardizes initial seeds and trace logging; stricter precomputed spawn schedule / decoupled obstacle RNG is deferred unless required by later comparisons.

## 11. Smoke Test Scale And Results

- scenarios: `eval_flow_id, eval_flow_high_speed`
- policies: `random, straight_line, cpa_reactive, attention_full, filtered_attention_full`
- episodes per scenario-policy: `1`
- checkpoint: `checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip`

| policy | scenario | episodes | success | collision | near_miss | mean_min_distance | filter_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_full | eval_flow_high_speed | 1 | 1.0000 | 0.0000 | 1.0000 | 4.4814 | 0.0000 |
| attention_full | eval_flow_id | 1 | 0.0000 | 1.0000 | 0.0000 | 5.8916 | 0.0000 |
| cpa_reactive | eval_flow_high_speed | 1 | 1.0000 | 0.0000 | 1.0000 | 3.7134 | 0.0000 |
| cpa_reactive | eval_flow_id | 1 | 0.0000 | 0.0000 | 1.0000 | 3.5655 | 0.0000 |
| filtered_attention_full | eval_flow_high_speed | 1 | 0.0000 | 1.0000 | 0.0000 | 3.5983 | 0.8374 |
| filtered_attention_full | eval_flow_id | 1 | 1.0000 | 0.0000 | 1.0000 | 4.1919 | 0.8294 |
| random | eval_flow_high_speed | 1 | 0.0000 | 0.0000 | 0.0000 | 5.8482 | 0.0000 |
| random | eval_flow_id | 1 | 0.0000 | 0.0000 | 0.0000 | 8.4945 | 0.0000 |
| straight_line | eval_flow_high_speed | 1 | 0.0000 | 1.0000 | 0.0000 | 4.0908 | 0.0000 |
| straight_line | eval_flow_id | 1 | 0.0000 | 1.0000 | 0.0000 | 6.3699 | 0.0000 |

## 12. Generated Artifacts

- `tables/phase_a_eval_summary.csv`
- `tables/phase_a_episode_metrics_sample.csv`
- `tables/phase_a_trace_schema.csv`
- `tables/phase_a_policy_adapter_check.csv`
- `tables/phase_a_env_freeze_check.csv`
- `tables/phase_a_command_manifest.csv`
- `tables/phase_a_step_obstacles_sample.csv`
- `traces/sample_random_trace.csv`
- `traces/sample_straight_line_trace.csv`
- `traces/sample_cpa_reactive_trace.csv`
- `traces/sample_attention_full_trace.csv`
- `traces/sample_filtered_attention_trace.csv`
- `logs/phase_a_eval_framework.log`
- `phase_a_status.txt`
- `phase_a_watcher.log` after watcher execution

## 13. Phase A Completion Criteria

- EnvV2-core freeze documented and checked.
- Unified runner exists and completed the smoke test.
- Required policies are supported.
- Episode metrics CSV has the fixed schema.
- Per-step traces exist for each policy.
- Full active obstacle set is logged with a long table.
- Filtered policy records raw and filtered actions plus filter metadata.
- Report, status, log, and flag files are generated.

## 14. Conclusion

Phase A complete.
Unified eval framework and trace schema are ready for Phase B.
