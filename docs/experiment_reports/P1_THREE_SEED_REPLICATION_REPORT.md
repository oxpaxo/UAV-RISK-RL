# P1 Three-Seed Replication Report

## 1. Experiment Setup
- Methods: attention_full, attention_full_distance_penalty, attention_full_risk_penalty.
- Seeds: 0, 1, 2.
- Total steps: 1000000.
- Checkpoints: 250000, 500000, 750000, 1000000.
- Scenarios: eval_random_switch, eval_sudden_turn, eval_random_switch_hard, mixed_uncertainty.
- Eval episodes: 50; eval_seed: 1000.
- beta_distance: 5.0; beta_risk: 5.0.
- reaction_time definition: eval.py reaction_time_eval_style.

## 2. Existing Runs Reused
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step250000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step250000_random.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step250000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step250000_hard.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step250000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step250000_sudden.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step250000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step250000_mixed.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step500000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step500000_random.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step500000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step500000_hard.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step500000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step500000_sudden.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step500000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step500000_mixed.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step750000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step750000_random.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step750000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step750000_hard.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step750000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step750000_sudden.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step750000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step750000_mixed.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step1000000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step1000000_random.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step1000000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step1000000_hard.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step1000000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step1000000_sudden.csv.
- attention_full seed=0: checkpoint checkpoints/longtrain_baseline/attention_full_s0_step1000000.zip; eval source results/longtrain_baseline/eval/attention_full_s0_step1000000_mixed.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step250000.zip; eval source results/attention_seed1/eval/attention_full_s1_step250000_random.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step250000.zip; eval source results/attention_seed1/eval/attention_full_s1_step250000_hard.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step250000.zip; eval source results/attention_seed1/eval/attention_full_s1_step250000_sudden.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step250000.zip; eval source results/attention_seed1/eval/attention_full_s1_step250000_mixed.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step500000.zip; eval source results/attention_seed1/eval/attention_full_s1_step500000_random.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step500000.zip; eval source results/attention_seed1/eval/attention_full_s1_step500000_hard.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step500000.zip; eval source results/attention_seed1/eval/attention_full_s1_step500000_sudden.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step500000.zip; eval source results/attention_seed1/eval/attention_full_s1_step500000_mixed.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step750000.zip; eval source results/attention_seed1/eval/attention_full_s1_step750000_random.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step750000.zip; eval source results/attention_seed1/eval/attention_full_s1_step750000_hard.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step750000.zip; eval source results/attention_seed1/eval/attention_full_s1_step750000_sudden.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step750000.zip; eval source results/attention_seed1/eval/attention_full_s1_step750000_mixed.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step1000000.zip; eval source results/attention_seed1/eval/attention_full_s1_step1000000_random.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step1000000.zip; eval source results/attention_seed1/eval/attention_full_s1_step1000000_hard.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step1000000.zip; eval source results/attention_seed1/eval/attention_full_s1_step1000000_sudden.csv.
- attention_full seed=1: checkpoint checkpoints/attention_seed1/attention_full_s1_step1000000.zip; eval source results/attention_seed1/eval/attention_full_s1_step1000000_mixed.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step250000_random.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step250000_hard.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step250000_sudden.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step250000_mixed.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step500000_random.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step500000_hard.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step500000_sudden.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step500000_mixed.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step750000_random.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step750000_hard.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step750000_sudden.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step750000_mixed.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step1000000_random.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step1000000_hard.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step1000000_sudden.csv.
- attention_full_distance_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_distance_penalty_s0_step1000000_mixed.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step250000_random.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step250000_hard.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step250000_sudden.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step250000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step250000_mixed.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step500000_random.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step500000_hard.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step500000_sudden.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step500000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step500000_mixed.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step750000_random.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step750000_hard.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step750000_sudden.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step750000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step750000_mixed.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step1000000_random.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step1000000_hard.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step1000000_sudden.csv.
- attention_full_risk_penalty seed=0: checkpoint checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip; eval source results/gate2b/eval/attention_full_risk_penalty_s0_step1000000_mixed.csv.

New or completed P1 training runs:
- attention_full seed=2.
- attention_full_distance_penalty seed=1.
- attention_full_distance_penalty seed=2.
- attention_full_risk_penalty seed=1.
- attention_full_risk_penalty seed=2.

## 3. Main Results: sudden_turn
| method | seed | step | reaction | success | collision | min_distance | near_miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_full | 0 | 250000 | 0.208 | 1.000 | 0.000 | 1.334 | 0.100 |
| attention_full | 0 | 500000 | 1.860 | 1.000 | 0.000 | 1.167 | 0.220 |
| attention_full | 0 | 750000 | 11.424 | 1.000 | 0.000 | 1.040 | 0.420 |
| attention_full | 0 | 1000000 | 6.076 | 1.000 | 0.000 | 0.942 | 0.700 |
| attention_full | 1 | 250000 | 0.216 | 1.000 | 0.000 | 1.305 | 0.120 |
| attention_full | 1 | 500000 | 2.692 | 1.000 | 0.000 | 1.203 | 0.300 |
| attention_full | 1 | 750000 | 1.376 | 1.000 | 0.000 | 1.006 | 0.600 |
| attention_full | 1 | 1000000 | 5.044 | 1.000 | 0.000 | 1.074 | 0.520 |
| attention_full | 2 | 250000 | 0.228 | 1.000 | 0.000 | 1.402 | 0.120 |
| attention_full | 2 | 500000 | 3.248 | 1.000 | 0.000 | 1.280 | 0.160 |
| attention_full | 2 | 750000 | 8.508 | 1.000 | 0.000 | 1.182 | 0.280 |
| attention_full | 2 | 1000000 | 10.356 | 1.000 | 0.000 | 0.959 | 0.700 |
| attention_full_distance_penalty | 0 | 250000 | 0.264 | 1.000 | 0.000 | 1.502 | 0.040 |
| attention_full_distance_penalty | 0 | 500000 | 1.232 | 1.000 | 0.000 | 1.331 | 0.140 |
| attention_full_distance_penalty | 0 | 750000 | 11.033 | 0.980 | 0.020 | 1.118 | 0.320 |
| attention_full_distance_penalty | 0 | 1000000 | 7.536 | 1.000 | 0.000 | 1.047 | 0.440 |
| attention_full_distance_penalty | 1 | 250000 | 0.252 | 1.000 | 0.000 | 1.435 | 0.120 |
| attention_full_distance_penalty | 1 | 500000 | 0.200 | 1.000 | 0.000 | 1.408 | 0.080 |
| attention_full_distance_penalty | 1 | 750000 | 1.808 | 1.000 | 0.000 | 1.296 | 0.100 |
| attention_full_distance_penalty | 1 | 1000000 | 7.408 | 1.000 | 0.000 | 1.221 | 0.140 |
| attention_full_distance_penalty | 2 | 250000 | 0.292 | 1.000 | 0.000 | 1.530 | 0.020 |
| attention_full_distance_penalty | 2 | 500000 | 0.200 | 1.000 | 0.000 | 1.414 | 0.080 |
| attention_full_distance_penalty | 2 | 750000 | 5.028 | 1.000 | 0.000 | 1.250 | 0.120 |
| attention_full_distance_penalty | 2 | 1000000 | 7.180 | 1.000 | 0.000 | 1.228 | 0.200 |
| attention_full_risk_penalty | 0 | 250000 | 0.200 | 0.420 | 0.000 | 2.647 | 0.000 |
| attention_full_risk_penalty | 0 | 500000 | 0.200 | 1.000 | 0.000 | 2.030 | 0.000 |
| attention_full_risk_penalty | 0 | 750000 | 0.208 | 1.000 | 0.000 | 1.913 | 0.000 |
| attention_full_risk_penalty | 0 | 1000000 | 1.076 | 1.000 | 0.000 | 1.888 | 0.000 |
| attention_full_risk_penalty | 1 | 250000 | 0.200 | 0.860 | 0.000 | 2.599 | 0.000 |
| attention_full_risk_penalty | 1 | 500000 | 0.200 | 1.000 | 0.000 | 2.044 | 0.020 |
| attention_full_risk_penalty | 1 | 750000 | 0.200 | 1.000 | 0.000 | 1.910 | 0.000 |
| attention_full_risk_penalty | 1 | 1000000 | 0.208 | 0.980 | 0.020 | 1.905 | 0.020 |
| attention_full_risk_penalty | 2 | 250000 | 0.200 | 0.580 | 0.000 | 2.670 | 0.000 |
| attention_full_risk_penalty | 2 | 500000 | 0.200 | 1.000 | 0.000 | 2.095 | 0.000 |
| attention_full_risk_penalty | 2 | 750000 | 0.200 | 1.000 | 0.000 | 1.916 | 0.000 |
| attention_full_risk_penalty | 2 | 1000000 | 0.212 | 1.000 | 0.000 | 1.868 | 0.000 |

## 4. Random Switch Side Effects
| method | seed | success | mean_time | min_distance | near_miss |
|---|---:|---:|---:|---:|---:|
| attention_full | 0 | 1.000 | 6.824 | 1.177 | 0.240 |
| attention_full | 1 | 0.980 | 6.684 | 1.244 | 0.460 |
| attention_full | 2 | 1.000 | 6.924 | 1.668 | 0.180 |
| attention_full_distance_penalty | 0 | 1.000 | 6.876 | 1.309 | 0.120 |
| attention_full_distance_penalty | 1 | 1.000 | 6.996 | 1.456 | 0.240 |
| attention_full_distance_penalty | 2 | 1.000 | 6.948 | 1.728 | 0.100 |
| attention_full_risk_penalty | 0 | 1.000 | 8.100 | 2.064 | 0.000 |
| attention_full_risk_penalty | 1 | 1.000 | 8.004 | 2.079 | 0.000 |
| attention_full_risk_penalty | 2 | 1.000 | 8.276 | 2.418 | 0.000 |

## 5. Hard and Mixed Results
| scenario | method | seed | reaction | success | collision | near_miss |
|---|---|---:|---:|---:|---:|---:|
| eval_random_switch_hard | attention_full | 0 | nan | 0.940 | 0.060 | 0.840 |
| eval_random_switch_hard | attention_full | 1 | nan | 0.980 | 0.020 | 0.940 |
| eval_random_switch_hard | attention_full | 2 | nan | 0.980 | 0.020 | 0.800 |
| eval_random_switch_hard | attention_full_distance_penalty | 0 | nan | 0.900 | 0.100 | 0.840 |
| eval_random_switch_hard | attention_full_distance_penalty | 1 | nan | 1.000 | 0.000 | 0.660 |
| eval_random_switch_hard | attention_full_distance_penalty | 2 | nan | 1.000 | 0.000 | 0.700 |
| eval_random_switch_hard | attention_full_risk_penalty | 0 | nan | 1.000 | 0.000 | 0.060 |
| eval_random_switch_hard | attention_full_risk_penalty | 1 | nan | 1.000 | 0.000 | 0.080 |
| eval_random_switch_hard | attention_full_risk_penalty | 2 | nan | 1.000 | 0.000 | 0.020 |
| mixed_uncertainty | attention_full | 0 | 1.604 | 0.960 | 0.040 | 0.440 |
| mixed_uncertainty | attention_full | 1 | 1.300 | 0.800 | 0.200 | 0.800 |
| mixed_uncertainty | attention_full | 2 | 0.440 | 0.800 | 0.200 | 0.760 |
| mixed_uncertainty | attention_full_distance_penalty | 0 | 2.964 | 0.980 | 0.020 | 0.520 |
| mixed_uncertainty | attention_full_distance_penalty | 1 | 0.500 | 0.900 | 0.100 | 0.660 |
| mixed_uncertainty | attention_full_distance_penalty | 2 | 1.144 | 0.920 | 0.080 | 0.800 |
| mixed_uncertainty | attention_full_risk_penalty | 0 | 0.204 | 0.800 | 0.200 | 0.520 |
| mixed_uncertainty | attention_full_risk_penalty | 1 | 0.216 | 0.740 | 0.260 | 0.640 |
| mixed_uncertainty | attention_full_risk_penalty | 2 | 0.200 | 0.860 | 0.140 | 0.540 |

## 6. Replication Decision
- Decision: risk_penalty core effect replicated.
- Replicated seeds by 750k core rule: [0, 1, 2].
- Seeds where risk_penalty has lower sudden-turn reaction than distance_penalty at 750k: [0, 1, 2].
- Seeds where distance_penalty reaction exceeds 5s at 750k: [0, 2].

## 7. Interpretation
- risk_penalty is stable enough across this three-seed check to support the core Gate-2b effect.
- Compare p1_summary_by_method_step_scenario.csv for mean/std across seeds.

## 8. Next Step Recommendation
- A. Enter P2 environment upgrade, while keeping beta/cost-scale diagnostics in the appendix.

## Interaction with P0.5 Distance Sanity / Wide Distance Ablation

1. distance_warning_cost=0 implementation check: 0 bug-like rows found; 19/30 rows are zero because min_distance stayed above d_warning=1.0.
2. d_warning=1.0 trigger rate: attention_full max=0.700, attention_full_distance_penalty_d1 max=0.440, attention_full_risk_penalty max=0.000. The trigger is not uniformly sparse.
3. d_warning=2.0 ablation at 750k sudden_turn: d1=11.0327 s, wide_d2=0.2000 s, risk=0.2080 s.
4. Current mechanism interpretation: wider/dense safety-margin support is sufficient to approach risk_penalty in this ablation.

## Artifacts
- results/p1_three_seed/p1_by_seed_step_scenario.csv
- results/p1_three_seed/p1_summary_by_method_step_scenario.csv
- results/p1_three_seed/plots/*.png
