# Method Route Recommendation

primary_diagnosis: reactive_prior_promising
secondary_diagnosis: baseline_learns_progress_but_not_safety
best_checkpoint: 1500000

recommended_next_step:
- 优先做 reactive prior + learned residual / velocity-obstacle-like safety filter / RL constrained by simple geometric avoider；不继续追 no-response。
- Treat the reactive avoider as a strong geometric prior but not a complete policy, because it is much safer while attention_full reaches the goal more often.

not_recommended_next_steps:
- 继续追旧 3-ball no-response degradation 或 Phase 3 failure localization
- 单纯把 attention_full 训练到 2000k
- 在 seed=0 路线不清楚时启动 seed=1/2
- 直接上 temporal/risk-aware attention 或 PPO-Lagrangian
- 把当前结果包装成 benchmark 论文主线
