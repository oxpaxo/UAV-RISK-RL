# Old Experiments Assets Summary

## Freeze Decision

The legacy 3-ball Gym line is frozen. It remains useful only as hypothesis source, metric-system source, failure-mode clue, baseline design experience, and engineering toolchain foundation.

## Required Conclusions

- Old 3-ball Gym results are downgraded to preliminary diagnostic evidence.
- risk hard weighting / risk prior / risk_penalty are not pushed as the main innovation.
- d_warning=1.0 is a narrow/sparse weak baseline.
- d_warning=2.0 is a strong safety baseline.
- beta / cost scale is an important confound.
- Long-training degradation should be described as no_response_rate / response reliability oscillation, not true reaction latency becoming slower.
- The new environment must re-validate whether no-response / safety-margin erosion exists.
- The current restart does not add a new method; it rebuilds and sanity-checks the environment first.

## Asset Scan Summary

- Total indexed assets: 86
- found: 86
- not_found: 0
- needs_update: 0

| module_or_file | status | purpose | can_reuse_for_env_v2 | notes |
| --- | --- | --- | --- | --- |
| envs/dynamic_obstacle_env.py | found | legacy 3-obstacle Gymnasium environment, rich-motion patches, risk/distance info fields | partial | Reuse observation conventions, info metrics, and motion snippets; do not continue method work on old 3-ball environment. |
| train.py | found | SB3 PPO training entry, Monitor/VecEnv wiring, safety-cost wrapper, run config writing | partial | Useful later for Phase 2 baseline reproduction only; not executed in Phase 0/1. |
| eval.py | found | checkpoint evaluation, episode metrics, trace export, attention snapshots | partial | Metric conventions are reusable; eval-style reaction_time is not true reaction latency. |
| policies/obstacle_set_extractor.py | found | risk, mean, and learned-attention obstacle set aggregation feature extractor | partial | May be reused once Phase 2 trains PPO on Env V2; not modified for new methods here. |
| policies/obstacle_set_extractor.py | found | attention_full implementation and optional risk bias hooks | partial | Attention tooling is frozen as engineering baseline; no temporal/risk-aware attention added. |
| train.py | found | SafetyCostWrapper for distance_warning and risk_sum reward shaping | partial | Cost-scale/beta confounds must be controlled; no new risk formula in this phase. |
| envs/dynamic_obstacle_env.py | found | legacy distance_warning_cost, risk_sum, and risk_max info fields | partial | Metric pattern reusable, but formulas are not promoted as primary innovation. |
| eval.py | found | no-response/reaction-style lateral deviation metrics for sudden-turn eval | partial | Use as diagnostic lineage only; describe long-train degradation as response reliability/no_response oscillation. |
| scripts/diagnose_sudden_turn.py | found | risk/reaction trace diagnostics around sudden-turn events | partial | Useful for later failure analysis after Env V2 baseline reproduction. |
| eval.py | found | per-step trace CSV writing with positions, risk, attention, and deviation fields | partial | Trace infrastructure can be adapted after Phase 1 sanity passes. |
| results/gate2b/traces | found | legacy trace artifact directory referenced by P0 analysis | no | Old evidence only; not a source of Env V2 results. |
| scripts/watch_p0_p1_completion.sh | found | legacy blocking watcher pattern for staged research runs | yes | Pattern reused for scripts/watch_phase0_phase1_completion.sh. |
| scripts/watch_p2_completion.sh | found | legacy Phase 2 completion verifier | partial | Verification style reusable; Phase 2 itself is not executed here. |
| scripts/aggregate_results.py | found | legacy result aggregation helper | partial | Aggregation style reusable for future PPO results. |
| scripts/summarize_risk_diagnostics.py | found | legacy diagnostic report writer | partial | Report pattern reusable; conclusions are preliminary diagnostics. |
| configs | found | configuration directory for current and future environment settings | yes | Env V2 config is stored under configs/env_v2/. |
| scripts/watch_ewma_formal_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_longtrain_gate2b_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_p0_p1_p0_5_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_p1_5_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_p2_stage2_to_final.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_p2_threat_patch_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_phase0_phase1_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/watch_pre_ppo_priority_completion.sh | found | blocking watcher shell scripts | yes | Discovered during Phase 0 scan. |
| scripts/plot_risk_diagnostics.py | found | plotting helpers | partial | Discovered during Phase 0 scan. |
| scripts/run_p0_5_distance_followup.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_p0_p1_replication.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_p1_5_distance_wide.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_p2_rich_motion.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_p2_stage2_to_final.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_p2_threat_patch_stage0_stage1.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| scripts/run_pre_ppo_priority.py | found | staged experiment orchestration scripts | partial | Discovered during Phase 0 scan. |
| ATTENTION_RISK_2000K_BASELINE_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| ATTENTION_SEED1_1000K_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| EWMA_RISK_FORMAL_RECHECK_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| EWMA_RISK_SHORT_TRAIN_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| FINAL_DIRECTION_DECISION_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| GATE2B_PENALTY_1000K_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P0_5_BETA_COST_SCALE_SWEEP_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P0_ADAPTATION_VALIDATION_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P0_TRACE_PREDICTIVE_ANALYSIS_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P1_THREE_SEED_REPLICATION_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_ENVIRONMENT_SANITY_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_ENVIRONMENT_SANITY_REPORT_PATCHED.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_RICH_MOTION_FINAL_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_STAGE1_OOD_EVAL_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_STAGE3_SEED0_PARETO_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P2_THREE_SEED_CONFIRMATION_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| PREEXP_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| PRE_PPO_PRIORITY_FINAL_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| P_MINUS_1_P2_PARETO_AUDIT_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| RISK_DIAGNOSTIC_REPORT.md | found | legacy markdown reports | partial | Discovered during Phase 0 scan. |
| runs/attention_seed1/attention_full_s1_1000k/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/gate2b/attention_full_distance_penalty_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/gate2b/attention_full_risk_penalty_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/gate2b/risk_biased_attention_risk_penalty_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/longtrain_baseline/attention_full_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p0_5_distance_wide/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_5_distance_wide/attention_full_distance_penalty_mid_d15_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_5_distance_wide/attention_full_distance_penalty_wide_d2_s1/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_5_distance_wide/attention_full_distance_penalty_wide_d2_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_three_seed/attention_full_distance_penalty_s1/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_three_seed/attention_full_distance_penalty_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_three_seed/attention_full_risk_penalty_s1/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_three_seed/attention_full_risk_penalty_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p1_three_seed/attention_full_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_distance_penalty_wide_d2_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_distance_penalty_wide_d2_s1/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_distance_penalty_wide_d2_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_risk_penalty_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_risk_penalty_s1/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_risk_penalty_s2/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/attention_full_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/p2_rich_motion/p2_short_attention_full_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_5_beta_sweep/risk_beta10_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_5_beta_sweep/risk_beta2_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_5_beta_sweep/wide_d2_beta10_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_5_beta_sweep/wide_d2_beta2_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_adaptation/high_speed_risk_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_adaptation/high_speed_wide_d2_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_adaptation/small_space_risk_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/pre_ppo_priority/p0_adaptation/small_space_wide_d2_s0/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |
| runs/preflight/resume_smoke/run_config.json | found | stored run configs | partial | Discovered during Phase 0 scan. |

## Reusable Modules

- Environment patterns: observation dict, info fields, rich-motion snippets, and risk/distance metric bookkeeping are reusable with updates.
- Training/evaluation tooling: SB3 entrypoints, run configs, Monitor/VecEnv use, checkpoint evaluation, and CSV/report conventions are reusable later in Phase 2.
- Policy tooling: obstacle-set feature extractor and attention/risk aggregation code can serve future baselines, but no new attention/risk method is added in Phase 0/1.
- Watcher/report tooling: blocking watcher and staged report patterns are reusable for this restart.

## Downgraded Or Rejected Routes

- risk_penalty, risk hard weighting, and risk prior are not treated as a primary contribution without Env V2 re-validation.
- d_warning=1.0 comparisons are weak because the warning band is narrow/sparse.
- d_warning=2.0 is retained as a strong safety baseline candidate for later reproduction.
- beta/cost-scale sweeps remain confounds that must be controlled before method claims.
- Legacy sudden-turn reaction_time is an eval-style diagnostic and must not be described as physical reaction latency.

## Phase 0 Output

- `OLD_EXPERIMENTS_ASSETS_SUMMARY.md`
- `results/restart_phase0_phase1/old_assets_index.csv`
