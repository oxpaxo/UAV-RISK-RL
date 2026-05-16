## Interaction with P0.5 Distance Sanity / Wide Distance Ablation

1. distance_warning_cost=0 implementation check: 0 bug-like rows found; 19/30 rows are zero because min_distance stayed above d_warning=1.0.
2. d_warning=1.0 trigger rate: attention_full max=0.700, attention_full_distance_penalty_d1 max=0.440, attention_full_risk_penalty max=0.000. The trigger is not uniformly sparse.
3. d_warning=2.0 ablation: P0.5-C not available yet.
4. Current mechanism interpretation: P0 trace does not support a strict post-turn early-warning mechanism; the working hypothesis is dense safety-margin regularization / wider cost support until P0.5-C finishes.
