# Gate-2b Config Logging Preflight Report

- train.py writes runs/{group}/{run_name}/run_config.json.
- run_config.json includes method, profile, seed, total_steps, checkpoint_steps, cost settings, risk-bias settings, risk config, d_safe, d_warning, resume settings, and checkpoint naming rule.
- Penalty reward path is logged through SafetyCostWrapper info fields.
- risk_bias path is logged by --use_risk_bias and --lambda_bias and implemented in ObstacleSetExtractor logits.
