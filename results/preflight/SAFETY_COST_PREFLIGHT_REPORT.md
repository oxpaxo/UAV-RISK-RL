# Safety Cost Preflight Report

- distance_penalty enters reward through SafetyCostWrapper when --use_safety_cost true --fallback_penalty true --cost_type distance_warning.
- risk_penalty enters reward through SafetyCostWrapper when --use_safety_cost true --fallback_penalty true --cost_type risk_sum.
- Training info records base_reward, applied_cost, shaped_reward, fallback_penalty_active.
- Default beta_cost: 5.0.
- The training log prints: FALLBACK: cost-penalty, not PPO-Lagrangian.
