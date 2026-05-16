# Risk Biased Attention Preflight Report

- Implementation: score_i = learned_score_i + lambda_bias * log(risk_i + eps).
- Default lambda_bias: 0.2.
- risk_i is read from full_12 obstacle profile final dimension.
- latest_attention_weights are cached for trace export.
- A smoke comparison is covered by trace availability and by using --use_risk_bias in Gate-2b run_config.
