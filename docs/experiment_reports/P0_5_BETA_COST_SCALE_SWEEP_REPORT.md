# Stage P0.5 Beta / Cost-Scale Sweep Report

## Scope
- Trained wide_d2 beta=2/10 and risk beta=2/10.
- Included existing P2 seed0 beta=5 reference rows for wide_d2 and risk.
- Evaluated 250k/500k/750k across the eight P2 scenarios.

## Pareto Coverage
- 750k wide_d2 beta sweep cover rate over risk beta=5: 0.5000
- distance_beta_covers_risk: True

| step | scenarios | wide_cover_count | wide_cover_rate | risk_not_covered_count |
|---:|---:|---:|---:|---:|
| 250000 | 8 | 5 | 0.6250 | 3 |
| 500000 | 8 | 0 | 0.0000 | 8 |
| 750000 | 8 | 4 | 0.5000 | 4 |

## Required Answers
1. wide_d2 beta sweep covers risk beta=5 at 750k: True.
2. risk beta=5 being pure tuning luck is supported by this coverage rule.
3. The adaptive-lambda / PPO-Lagrangian decision is deferred to the integrated final report.
