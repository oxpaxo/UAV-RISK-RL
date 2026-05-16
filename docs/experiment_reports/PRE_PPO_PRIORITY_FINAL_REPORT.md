# Pre-PPO Priority Experiments Final Report

## 1. Motivation
This run resolves the three pre-PPO-Lagrangian questions: P2 Pareto scope, fixed-margin adaptation under two variants, and beta/cost-scale confounding.

## 2. P-1 Pareto Audit
- Pareto-positive scenarios: eval_accel_decel, eval_ar1, eval_mixed_v2, eval_random_switch, eval_random_switch_hard, eval_sinusoidal, eval_sudden_turn, eval_threat_validated_sudden.
- Majority wide_d2-dominated scenarios: none.

## 3. P0 Adaptation Validation
- high_speed wide failure: False; overconservative: False; risk adaptation supported: False.
- small_space wide failure: False; overconservative: False; risk adaptation supported: False.

## 4. P0.5 Beta / Cost-Scale Sweep
- 750k wide_d2 beta-sweep cover rate over risk beta=5: 0.5000.
- wide_d2 beta sweep covers risk beta=5 under the configured rule: True.

## 5. Integrated Decision
terminal_decision = distance_margin_explains_risk

## 6. Next Recommendation
- Downgrade risk as a main innovation and pivot toward safety-margin cost design principles and beta/margin tuning.

## Required Answers
1. P2 Pareto audit: risk advantage holds in eval_accel_decel, eval_ar1, eval_mixed_v2, eval_random_switch, eval_random_switch_hard, eval_sinusoidal, eval_sudden_turn, eval_threat_validated_sudden scenarios under the classification rule.
2. risk is not globally dominated by wide_d2; majority-dominated scenarios: none.
3. high_speed wide_d2 failure/overconservative: failure=False, overconservative=False.
4. high_speed risk Pareto scenarios at 750k: 3.
5. small_space wide_d2 failure/overconservative: failure=False, overconservative=False.
6. small_space risk Pareto scenarios at 750k: 5.
7. beta sweep: wide_d2 covers risk beta=5 = True with 750k cover_rate=0.5000.
8. risk beta=5 tuning-only explanation: supported.
9. adaptive-lambda / PPO-Lagrangian now: no.
10. risk mainline: distance_margin_explains_risk.
