# Phase N4U-PRECHECK-ORACLE-CALIBRATION v2 Report

## Terminal Decision

`terminal_decision = phase_n4u_precheck_v2_stopped_premise_invalidated`

GitHub sync status: `success`; commit: `4765acac`.

This run stopped at Step A by design. No formal N4-U eval, no PPO training, no Gpsi fine-tuning, and no EnvV2-core modification were performed.

## CPA/TTC Audit

EnvV2 `obs_i` planned CPA/TTC are stored obstacle planned quantities sampled/constructed at spawn time. They are not runtime candidate-velocity CPA/TTC values. Phase B and N4-O shields independently recompute analytic constant-velocity CPA/TTC from current obstacle state and each candidate action.

| component | file | line | formula_or_source | uses_constant_velocity_analytic_runtime_cpa_ttc | uses_ground_truth_future_trajectory | depends_on_current_uav_velocity | depends_on_candidate_velocity | units_and_normalization | invalid_handling | audit_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EnvV2 observation obs_i planned_cpa | envs/dynamic_obstacle_flow_env.py | 622 | obs_i[6] = obstacle.planned_cpa / 5.0 | 0 | 0 | 0 | 0 | meters, normalized by /5.0 in obs_i | stored per active obstacle; inactive obs rows zero-padded by mask | spawn-time planned design quantity, not runtime analytic CPA |
| EnvV2 observation obs_i planned_ttc | envs/dynamic_obstacle_flow_env.py | 623 | obs_i[7] = obstacle.planned_ttc / 20.0 | 0 | 0 | 0 | 0 | seconds, normalized by /20.0 in obs_i | info also reports remaining planned_ttc clipped at >=0 | spawn-time planned design quantity, not runtime analytic TTC |
| EnvV2 obstacle spawn planned_cpa | envs/dynamic_obstacle_flow_env.py | 348 | planned_cpa sampled from CPA_RANGES[threat_class] | 0 | 0 | 0 | 0 | meters before obs normalization | _planned_threat_valid checks class range and finite value | not computed from current relative position/velocity |
| EnvV2 obstacle spawn planned_ttc | envs/dynamic_obstacle_flow_env.py | 352 | planned_ttc sampled from scenario ttc_range, then may be adjusted by target_s/path progress | 0 | 0 | 0 | 0 | seconds before obs normalization | _planned_threat_valid requires finite and 0.5<=planned_ttc<=20 | not computed as tcpa=-dot(rel,rel_vel)/||rel_vel||^2 |
| Phase B runtime CPA/TTC | scripts/run_env_v2_phase_b_geometry_filter_baselines.py | 253 | tcpa=clip(-dot(rel,rel_vel)/||rel_vel||^2,0,horizon); cpa=||rel+rel_vel*tcpa|| | 1 | 0 | 1 | 1 | meters and seconds, unnormalized | if no obstacles -> NaN; if rel_speed_sq<=1e-8 -> tcpa=0, cpa=current distance | independent runtime analytic CPA/TTC recomputation for candidate actions |
| N4-O ordinary VO shield | scripts/eval_env_v2_phase_n4o_ordinary_shield.py | 171 | loads Phase B vo_like_filter_h45_cpa1p2_h16 manifest and calls Phase B act/filter path | 1 | 0 | 1 | 1 | same Phase B unnormalized meters/seconds | inherits Phase B logic | parameter-equivalent ordinary VO wrapper around Phase B implementation |


## Premise Gate

| gate | envv2_obs_planned_cpa_ttc_is_constant_velocity_runtime_analytic | envv2_obs_planned_cpa_ttc_is_ground_truth_future_oracle | shield_cpa_ttc_is_constant_velocity_runtime_analytic | phase_b_n4o_formula_compatible | premise_valid | stop_flag | terminal_decision | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_env_constant_velocity_correction_route | 0 | 0 | 1 | 1 | 0 | STOP_PREMISE_INVALIDATED.flag | phase_n4u_precheck_v2_stopped_premise_invalidated | EnvV2 obs planned_cpa/planned_ttc are spawn-time planned design quantities, while Phase B/N4-O shields independently recompute runtime constant-velocity CPA/TTC for candidate actions. Therefore the framing 'Gpsi corrects EnvV2 planned CPA/TTC constant-velocity analytic quantity' is invalid for clean EnvV2. |


## Strong Geometric Anchor Manifest

| strong_geom_anchor | manifest_path | summary_path | manifest_exists | formal_summary_exists | baseline_name | kind | filter_used | horizon | cpa_safe | num_headings | candidate_velocity_set | candidate_count_nominal | scoring_logic | unsafe_action_handling | formal_success | formal_collision | formal_near_miss | formal_progress | formal_filter_trigger_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase B vo_like_filter_h45_cpa1p2_h16 | results/env_v2_phase_b_geometry_filter_baselines/tables/phase_b_baseline_manifest.csv | results/env_v2_phase_b_geometry_filter_baselines/tables/phase_b_eval_summary.csv | 1 | 1 | vo_like_filter | vo_like_filter | 1 | 4.5 | 1.2 | 16 | raw_action; goal_action; current_action; away_lateral_action; headings at speeds 0.4/0.7/1.0 over num_headings=16 | 52 | -1.2*||cand-raw|| + 0.8*progress_alignment + 0.35*min_cpa - 0.2*||cand-current_action|| | trigger if runtime analytic candidate CPA has 0<=tcpa<=horizon and min_cpa<cpa_safe; select best safe candidate if any, else best_any | 0.8333333333333334 | 0.1666666666666666 | 0.72 | 0.987975154553813 | 0.3508952195574182 |


## Phase B / N4-O Equivalence

| phase_b_config | n4o_manifest_path | n4o_manifest_exists | n4o_same_params_for_attention_noz_p3 | n4o_ordinary_shield_uses_sigma2 | n4o_ordinary_shield_uses_future_truth | parameter_equivalent_to_phase_b_anchor | formula_equivalent_to_phase_b_anchor | equivalence_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vo_like_filter_h45_cpa1p2_h16 | results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_shield_config_manifest.csv | 1 | 1 | 0 | 0 | 1 | 1 | N4-O ordinary VO loads Phase B manifest params and uses Phase B vo_like_filter logic; EnvV2 obs planned CPA/TTC are not the same quantity. |


## Required Consequence

- Stop the clean EnvV2 constant-velocity-correction framing.
- Do not proceed to oracle headroom, Gpsi point-correction, uncertainty calibration, adaptive margin, or stress smoke under this invalid premise in this run.
- A future phase can be reframed around candidate-action runtime CPA/TTC correction directly inside the Phase B/N4-O shield geometry, but it must not describe EnvV2 `planned_cpa/planned_ttc` as the constant-velocity analytic target.

## Artifacts

- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_command_manifest.csv`
- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_cpa_ttc_source_audit.csv`
- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_github_sync.csv`
- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_phase_b_n4o_equivalence.csv`
- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_premise_gate.csv`
- `results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/phase_n4u_precheck_v2_strong_geom_anchor_manifest.csv`
