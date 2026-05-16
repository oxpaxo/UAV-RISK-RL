# Code Status Preflight

Current directory code was inspected directly. Old handoff claims are not assumed.

## train.py key parameters
- --method: yes
- --profile_mode: yes
- --agg: yes
- --seed: yes
- --total_steps: yes
- --n_envs: yes
- --device: yes
- --scenario: yes
- --save_checkpoints: yes
- --checkpoint_steps: yes
- --checkpoint_dir: yes
- --log_dir: yes
- --save_path: yes
- --resume_from: yes
- --resume_global_step: yes
- --use_safety_cost: yes
- --cost_type: yes
- --beta_cost: yes
- --fallback_penalty: yes
- --d_warning: yes
- --use_risk_bias: yes
- --lambda_bias: yes

## eval.py key parameters
- --model_path: yes
- --method: yes
- --profile_mode: yes
- --agg: yes
- --seed: yes
- --eval_seed: yes
- --episodes: yes
- --scenario: yes
- --device: yes
- --out_csv: yes
- --save_trace: yes
- --trace_dir: yes

## Aggregation modes
- risk: True
- attention: True
- mean: True
- risk_biased_attention logits: True
- latest_attention_weights cache: True

## DynamicObstacleEnv info fields
- risk_sum: True
- risk_max: True
- distance_warning_cost: True
- risk_values: True
- sigma_values: True
