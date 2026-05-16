# Phase B Geometry Filter Baseline Report

## 1. Background And Phase B Goal

Phase B is an eval-only audit of geometry controllers and action-level safety filters on frozen EnvV2. No new PPO was trained and EnvV2-core was not modified.

## 2. Phase A Dependency Check

- Phase A complete flag: checked.
- Phase A episode and trace schema: reused and extended without deleting core columns.
- attention_full 1500k checkpoint: loaded from `checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip`.

## 3. EnvV2-Core Freeze Recheck

The Phase A EnvV2-core hash was compared with the current `envs/dynamic_obstacle_flow_env.py` hash before evaluation. Phase B did not modify obstacle counts, motion modes, train/eval scenarios, action dynamics, reward, termination, or collision/success/near-miss definitions.

## 4. Added / Modified Files

- `scripts/run_env_v2_phase_b_geometry_filter_baselines.py`
- `scripts/analyze_env_v2_phase_b_results.py`
- `scripts/watch_phase_b_geometry_filter_baselines.sh`
- `results/env_v2_phase_b_geometry_filter_baselines_full_smoke/` result artifacts

## 5. Baseline Manifest

Baseline definitions are recorded in `tables/phase_b_baseline_manifest.csv`. Required families include random, straight_line, attention_full_1500k, current_cpa_reactive, APF family, CPA-reactive sweep, and distance / CPA-TTC / VO-like attention filters.

## 6. Baseline Formulas / Parameters

- `naive_apf`: goal attraction plus inverse-distance repulsion with `d0` and `w_rep` sweep.
- `velocity_aware_apf`: naive APF multiplied by closing-speed gain.
- `cpa_ttc_weighted_apf`: velocity-aware APF multiplied by CPA/TTC short-horizon gain.
- `cpa_reactive_sweep`: one-factor sweep around current `d_reactive=4.0`, `horizon=4.5`, `cpa_trigger=2.4`, `avoid_weight=2.1`.
- `distance_filter`: filters attention action when nearest obstacle is close and raw action increases closing.
- `cpa_ttc_filter`: filters attention action when predicted CPA/TTC under raw action is unsafe.
- `vo_like_filter`: selects a velocity candidate from a small velocity set by CPA safety and progress/raw-distance score.

## 7. B0 / B1 / B2 Eval Scale

- b0_smoke: 84 episode rows
- b1_coarse: 252 episode rows
- b2_formal: 66 episode rows

## 8. Aggregate Comparison

| baseline | config | success | collision | near_miss | progress | min_distance | rank_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vo_like_filter | vo_like_filter_h45_cpa2_h16 | 1.0000 | 0.0000 | 0.3333 | 0.9865 | 1.5110 | 1.0306 |
| vo_like_filter | vo_like_filter_h45_cpa1p5_h16 | 1.0000 | 0.0000 | 0.5000 | 0.9870 | 1.4238 | 0.9474 |
| cpa_ttc_weighted_apf | cpa_ttc_weighted_apf_alpha2 | 1.0000 | 0.0000 | 1.0000 | 0.9871 | 1.0445 | 0.6974 |
| cpa_ttc_weighted_apf | cpa_ttc_weighted_apf_alpha3 | 1.0000 | 0.0000 | 1.0000 | 0.9842 | 1.1163 | 0.6968 |
| cpa_reactive_sweep | cpa_reactive_d5 | 0.3333 | 0.0000 | 0.1667 | 0.9026 | 1.7835 | 0.4305 |
| attention_full_1500k | attention_full_1500k | 0.8333 | 0.1667 | 0.8333 | 0.9789 | 0.8306 | 0.2791 |
| cpa_reactive_sweep | cpa_reactive_d3 | 0.3333 | 0.0000 | 0.6667 | 0.9708 | 1.4315 | 0.1942 |
| cpa_reactive_sweep | cpa_reactive_cpa18 | 0.5000 | 0.1667 | 0.3333 | 0.9679 | 1.4174 | 0.1936 |
| random | random | 0.0000 | 0.0000 | 0.0000 | 0.0859 | 3.3265 | 0.0172 |
| current_cpa_reactive | cpa_reactive_current | 0.1667 | 0.0000 | 0.8333 | 0.9591 | 1.3357 | -0.0582 |
| straight_line | straight_line | 0.1667 | 0.8333 | 0.1667 | 0.4218 | 0.5901 | -1.4990 |

## 9. Pareto Frontier

- success/collision Pareto configs: `vo_like_filter_h45_cpa2_h16, vo_like_filter_h45_cpa1p5_h16, cpa_ttc_weighted_apf_alpha2, cpa_ttc_weighted_apf_alpha3`
- progress/collision Pareto configs: `cpa_ttc_weighted_apf_alpha2`

## 10. Scenario-Wise Breakdown

`tables/phase_b_scenario_breakdown.csv` contains formal scenario-level success, collision, near-miss, progress, and distance metrics.

## 11. Motion-Mode Breakdown

`tables/phase_b_motion_mode_breakdown.csv` contains formal collision/success breakdown by threat motion mode.

## 12. Threat-Class Breakdown

`tables/phase_b_threat_class_breakdown.csv` contains formal collision/success breakdown by threat class.

## 13. Filter Intervention Analysis

| baseline | config | trigger_rate | collision_triggered | collision_not_triggered | cpa_raw | cpa_filtered |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vo_like_filter | vo_like_filter_h45_cpa1p5_h16 | 0.3724 | 0.0000 | nan | 2.7353 | 3.3649 |
| vo_like_filter | vo_like_filter_h45_cpa2_h16 | 0.5946 | 0.0000 | nan | 2.2832 | 3.3938 |

## 14. Failure Cases

`tables/phase_b_failure_case_table.csv` contains 55 sampled failure rows sorted by collision / near-miss / minimum distance.

## 15. Top Configs

`tables/phase_b_top_configs.csv` records the B1-selected configs used for B2 confirmation. `tables/phase_b_pareto_table.csv` records the formal ranking.

## 16. Did Geometry / Filters Beat Attention Full?

Experiment-supported fact: at least one formal B2 config achieved lower collision while preserving attention_full_1500k success within 0.05.
Configs: `vo_like_filter_h45_cpa2_h16, vo_like_filter_h45_cpa1p5_h16, cpa_ttc_weighted_apf_alpha2, cpa_ttc_weighted_apf_alpha3`.

Reasonable inference: Phase C should focus on training-time safety costs only after using this audit to choose whether action-level filters are strong enough as deployment wrappers or diagnostic baselines.

Hypotheses for Phase C: CPA/TTC costs and VO-style unsafe-velocity costs are plausible if formal filters reduce collision without destroying progress; otherwise geometry-only intervention is insufficient.

## 17. Phase C Recommendation

If Phase B complete, Phase C may start safety-cost training decisions using the formal Pareto and filter intervention tables.

## 18. Completion Criteria

- Phase A complete flag exists.
- EnvV2-core freeze rechecked.
- attention_full checkpoint loaded.
- B0, B1, and B2 completed.
- required CSVs, plots, sampled/failure traces, report, logs, and flag are generated.

## 19. Decision

Phase B complete.
Geometry/filter baseline audit is ready for Phase C decision.

terminal_decision = phase_b_geometry_filter_baseline_complete
