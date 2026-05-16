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
- `results/env_v2_phase_b_geometry_filter_baselines/` result artifacts

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

- b0_smoke: 252 episode rows
- b1_coarse: 5040 episode rows
- b2_formal: 3300 episode rows

## 8. Aggregate Comparison

| baseline | config | success | collision | near_miss | progress | min_distance | rank_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vo_like_filter | vo_like_filter_h45_cpa1p2_h16 | 0.8333 | 0.1667 | 0.7200 | 0.9880 | 1.1546 | 0.3376 |
| vo_like_filter | vo_like_filter_h45_cpa1p5_h16 | 0.7067 | 0.2767 | 0.3533 | 0.9879 | 1.1125 | 0.1742 |
| cpa_reactive_sweep | cpa_reactive_d5 | 0.0667 | 0.0433 | 0.2733 | 0.9060 | 1.7812 | 0.0245 |
| random | random | 0.0000 | 0.0000 | 0.0300 | 0.0889 | 3.5845 | 0.0028 |
| cpa_ttc_weighted_apf | cpa_ttc_weighted_apf_alpha3 | 0.7200 | 0.2800 | 0.7200 | 0.8964 | 0.8401 | -0.0207 |
| current_cpa_reactive | cpa_reactive_current | 0.1133 | 0.0967 | 0.2667 | 0.9332 | 1.5612 | -0.0267 |
| cpa_reactive_sweep | cpa_reactive_cpa30 | 0.0333 | 0.0767 | 0.2300 | 0.9300 | 1.7154 | -0.0490 |
| cpa_reactive_sweep | cpa_reactive_w28 | 0.0633 | 0.0900 | 0.2600 | 0.9276 | 1.6737 | -0.0612 |
| cpa_ttc_weighted_apf | cpa_ttc_weighted_apf_alpha2 | 0.6867 | 0.3133 | 0.6867 | 0.8557 | 0.8205 | -0.1122 |
| attention_full_1500k | attention_full_1500k | 0.6100 | 0.3900 | 0.5767 | 0.9572 | 0.8130 | -0.2669 |
| straight_line | straight_line | 0.0267 | 0.9733 | 0.0267 | 0.3369 | 0.4971 | -1.8660 |

## 9. Pareto Frontier

- success/collision Pareto configs: `vo_like_filter_h45_cpa1p2_h16, cpa_reactive_d5, random, cpa_reactive_current`
- progress/collision Pareto configs: `vo_like_filter_h45_cpa1p2_h16, cpa_reactive_d5, random, cpa_reactive_current, cpa_reactive_cpa30`

## 10. Scenario-Wise Breakdown

`tables/phase_b_scenario_breakdown.csv` contains formal scenario-level success, collision, near-miss, progress, and distance metrics.

## 11. Motion-Mode Breakdown

`tables/phase_b_motion_mode_breakdown.csv` contains formal collision/success breakdown by threat motion mode.

## 12. Threat-Class Breakdown

`tables/phase_b_threat_class_breakdown.csv` contains formal collision/success breakdown by threat class.

## 13. Filter Intervention Analysis

| baseline | config | trigger_rate | collision_triggered | collision_not_triggered | cpa_raw | cpa_filtered |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vo_like_filter | vo_like_filter_h45_cpa1p2_h16 | 0.3509 | 0.1667 | nan | 2.6371 | 3.2060 |
| vo_like_filter | vo_like_filter_h45_cpa1p5_h16 | 0.4409 | 0.2767 | nan | 2.5015 | 3.2342 |

## 14. Failure Cases

`tables/phase_b_failure_case_table.csv` contains 200 sampled failure rows sorted by collision / near-miss / minimum distance.

## 15. Top Configs

`tables/phase_b_top_configs.csv` records the B1-selected configs used for B2 confirmation. `tables/phase_b_pareto_table.csv` records the formal ranking.

## 16. Did Geometry / Filters Beat Attention Full?

Experiment-supported fact: at least one formal B2 config achieved lower collision while preserving attention_full_1500k success within 0.05.
Configs: `vo_like_filter_h45_cpa1p2_h16, vo_like_filter_h45_cpa1p5_h16, cpa_ttc_weighted_apf_alpha3, cpa_ttc_weighted_apf_alpha2`.

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
