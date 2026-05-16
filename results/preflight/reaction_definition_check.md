# Reaction Definition Check

- turn_time: env.turn_step * env.dt; default turn_step=15, dt=0.2, turn_time=3.0 seconds.
- reaction flag: lateral desired velocity relative to goal direction > 0.3.
- consecutive_steps: 2.
- eval-style no response: max_episode_time - turn_time.
- nan-style no response: NaN.
- eval.py writes reaction_time_eval_style, reaction_time_nan_style, nan_reaction_rate, no_response_count, total_episodes.
- diagnostic mismatch in old data is primarily expected from NaN vs upper-bound fill for no-response episodes.
