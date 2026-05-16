# P0 Trace Predictive Analysis Report

## Scope
- Core trace: attention_full_risk_penalty, seed=0, step=750000, eval_sudden_turn, 10 episodes.
- Source traces: results/gate2b/traces.
- Required trace fields are available, including attention_weights.

## Main Finding
- P0 does not cleanly support a strict early-warning claim under the configured thresholds.
- Episodes with distance_warning_cost rising after turn: 0/10.
- Episodes with positive lead_time_sum: 0/10 among all episodes.
- Median lead_time_sum: nan s; median lead_steps_sum: nan steps.
- 750k sudden reaction: risk_penalty=0.208 s, distance_penalty=11.033 s.

## Required Answers
1. risk_sum / risk_max earlier than distance_warning_cost: not established by the default rise rule.
2. Lead amount: median nan s / nan steps for valid lead episodes.
3. Majority: 0/10 episodes have lead_time_sum > 0.
4. Reaction consistency: risk_penalty is much faster than distance_penalty at 750k, but early-warning evidence is limited when distance_warning_cost never rises.
5. If inconsistent: likely causes include conservative delta_risk=0.02 relative to this trace scale, policy avoiding the warning zone, or reward shaping rather than earlier warning.
6. Paper narrative: use cautious wording: risk_penalty is empirically effective, but the strict early-warning mechanism is not fully established by P0.

## Episode Metrics
| episode | risk_rise_sum | dist_rise | lead_s | lead_steps | reaction | min_dist_after_turn | near_miss | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | nan | nan | nan | nan | 0.200 | 2.681 | 0 | 0 |
| 1 | nan | nan | nan | nan | 0.200 | 2.178 | 0 | 0 |
| 2 | nan | nan | nan | nan | 0.200 | 3.131 | 0 | 0 |
| 3 | nan | nan | nan | nan | 0.200 | 2.097 | 0 | 0 |
| 4 | nan | nan | nan | nan | 0.200 | 2.755 | 0 | 0 |
| 5 | nan | nan | nan | nan | 0.200 | 2.492 | 0 | 0 |
| 6 | nan | nan | nan | nan | 0.200 | 2.656 | 0 | 0 |
| 7 | nan | nan | nan | nan | 0.200 | 3.123 | 0 | 0 |
| 8 | nan | nan | nan | nan | 0.200 | 3.315 | 0 | 0 |
| 9 | nan | nan | nan | nan | 0.200 | 2.809 | 0 | 0 |

## Artifacts
- results/p0_trace/p0_trace_summary.csv
- results/p0_trace/p0_trace_mean_curve.csv
- results/p0_trace/plots/risk_vs_distance_ep*.png
- results/p0_trace/plots/risk_vs_distance_mean.png
