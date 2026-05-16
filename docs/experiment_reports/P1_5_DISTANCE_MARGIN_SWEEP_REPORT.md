# P1.5 Distance Margin Sweep Report

## 1. Purpose
P1.5 tests whether the risk_penalty advantage is explained by motion-uncertainty risk itself or by a wider / denser distance safety margin. This must be settled before P2 environment upgrades, otherwise P2 could over-attribute a generic distance-margin effect to risk.

## 2. Experiment Setup
- Reused methods: attention_full, attention_full_distance_penalty_d1, attention_full_risk_penalty.
- New required method: attention_full_distance_penalty_wide_d2, d_warning=2.0, seeds 1 and 2; seed 0 reused from P0.5-C.
- Optional executed method: attention_full_distance_penalty_mid_d15, d_warning=1.5, seed 0.
- beta_cost=5.0, fallback_penalty=true, profile_mode=full_12, train scenario=train_random_switch.
- Checkpoints: 250000, 500000, 750000, 1000000.
- Eval scenarios: eval_random_switch, eval_sudden_turn, eval_random_switch_hard, mixed_uncertainty.
- Eval episodes: 50; eval_seed=1000.

## 3. Reused Runs and New Runs
- Reused P1 three-seed evals for baseline, d1 distance penalty, and risk penalty.
- Reused P0.5-C wide_d2 seed=0 checkpoints/evals.
- Newly trained/evaluated wide_d2 seeds 1 and 2.
- Newly trained/evaluated mid_d15 seed 0.

## 4. Main 750k Comparison
| method | seed | sudden_reaction | sudden_success | sudden_collision | random_mean_time | hard_near_miss | mixed_collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_full | 0 | 11.4240 | 1.000 | 0.000 | 6.8240 | 0.840 | 0.040 |
| attention_full | 1 | 1.3760 | 1.000 | 0.000 | 6.6840 | 0.940 | 0.200 |
| attention_full | 2 | 8.5080 | 1.000 | 0.000 | 6.9240 | 0.800 | 0.200 |
| attention_full_distance_penalty_d1 | 0 | 11.0327 | 0.980 | 0.020 | 6.8760 | 0.840 | 0.020 |
| attention_full_distance_penalty_d1 | 1 | 1.8080 | 1.000 | 0.000 | 6.9960 | 0.660 | 0.100 |
| attention_full_distance_penalty_d1 | 2 | 5.0280 | 1.000 | 0.000 | 6.9480 | 0.700 | 0.080 |
| attention_full_distance_penalty_mid_d15 | 0 | 2.7040 | 1.000 | 0.000 | 7.3280 | 0.120 | 0.040 |
| attention_full_distance_penalty_wide_d2 | 0 | 0.2000 | 1.000 | 0.000 | 8.7160 | 0.000 | 0.140 |
| attention_full_distance_penalty_wide_d2 | 1 | 0.2000 | 0.980 | 0.020 | 8.6800 | 0.000 | 0.200 |
| attention_full_distance_penalty_wide_d2 | 2 | 0.2000 | 1.000 | 0.000 | 8.6600 | 0.000 | 0.220 |
| attention_full_risk_penalty | 0 | 0.2080 | 1.000 | 0.000 | 8.1000 | 0.060 | 0.200 |
| attention_full_risk_penalty | 1 | 0.2000 | 1.000 | 0.000 | 8.0040 | 0.080 | 0.260 |
| attention_full_risk_penalty | 2 | 0.2000 | 1.000 | 0.000 | 8.2760 | 0.020 | 0.140 |

## 5. Three-Seed Wide Distance Result
- wide_d2 mean sudden reaction at 750k: 0.2000 s.
- risk_penalty mean sudden reaction at 750k: 0.2027 s.
- wide_d2 mean hard near_miss at 750k: 0.0000.
- risk_penalty mean hard near_miss at 750k: 0.0533.
- wide_d2 seeds with sudden reaction <= 1s: 3/3.
- risk_penalty seeds with sudden reaction <= 1s: 3/3.

## 6. Safety-Efficiency Trade-off
- wide_d2 mean random mean_time at 750k: 8.6853 s.
- risk_penalty mean random mean_time at 750k: 8.1267 s.
- d1 mean sudden reaction at 750k: 5.9562 s.
- baseline mean sudden reaction at 750k: 7.1027 s.
- Decision class: A. wide_d2 is close to risk on safety, but is slower / more conservative on random-switch efficiency.

## 7. Optional d_warning=1.5 Sweep
- d15 seed=0 was executed.
- The seed0 sweep is provided in d_warning_sweep_seed0.csv and d_warning_sweep_seed0_plots/*.png.
- Risk on the seed0 sudden_reaction/random_time Pareto front: yes.

## 8. Decision
- Selected case: A.
- Interpretation: wide_d2 is close to risk on safety, but is slower / more conservative on random-switch efficiency.

## 9. Next Recommendation
- Enter P2, but compare risk_penalty against wide_d2 directly. The defensible risk claim is better safety-efficiency trade-off, not strict early warning.
- P2 should compare: attention_full baseline, attention_full_distance_penalty_wide_d2, attention_full_risk_penalty; optionally keep d1 as a reference.

## Required Answers
1. wide_d2 seed=1/2 close to risk_penalty: compare table; mean wide_d2 sudden reaction 0.2000 s vs risk 0.2027 s.
2. Sudden reaction difference: -0.0027 s at 750k mean.
3. wide_d2 slower/more conservative: random mean_time wide=8.6853 s vs risk=8.1267 s.
4. risk safety-efficiency advantage over wide_d2: yes.
5. d_warning=1.5 continuous trade-off: see d_warning_sweep_seed0.csv and plots.
6. risk on distance-margin sweep Pareto front: yes.
7. Enter P2: yes.
8. P2 methods: attention_full baseline, distance_penalty_wide_d2, risk_penalty, optional distance_penalty_d1.

## Artifacts
- results/p1_5_distance_wide/p1_5_by_seed_step_scenario.csv
- results/p1_5_distance_wide/p1_5_summary_by_method_step_scenario.csv
- results/p1_5_distance_wide/p1_5_main_750k_table.csv
- results/p1_5_distance_wide/p1_5_pareto_summary.csv
- results/p1_5_distance_wide/d_warning_sweep_seed0.csv
- results/p1_5_distance_wide/plots/*.png
- results/p1_5_distance_wide/d_warning_sweep_seed0_plots/*.png
