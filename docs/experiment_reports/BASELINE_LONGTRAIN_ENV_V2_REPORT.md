# Baseline Longtrain Env V2 Report

## Setup

- method: attention_full
- seed: 0
- env: DynamicObstacleFlowEnv
- train scenario: train_flow_mixed
- trained steps: 1500000
- n_envs: 16
- safety cost: disabled
- PPO-Lagrangian/adaptive lambda/new risk formula: not used

## Metrics Note

`reaction_time_eval_style` uses a timeout penalty for no-response episodes. It can rise because `no_response_rate` rises and must not be interpreted as every episode having slower physical reaction latency.
`reaction_time_nan_style` / `conditional_reaction_time` only averages episodes where the action-based response criterion fired.
