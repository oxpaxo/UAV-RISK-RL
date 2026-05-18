# Phase N4U-RUNTIME-CV-ORACLE-CALIBRATION Guide

## 0. Phase purpose

This guide replaces the invalidated `planned_cpa/planned_ttc correction` framing with the corrected runtime-shield framing:

```text
Gψ should correct the runtime constant-relative-velocity CPA/TTC used by the strong Phase B / N4-O VO shield.
```

Previous Step A stopped correctly with:

```text
terminal_decision = phase_n4u_precheck_v2_stopped_premise_invalidated
```

What died:

```text
Gψ Δ̂ corrects EnvV2 obs_i planned_cpa/planned_ttc constant-velocity error.
```

What remains valid:

```text
Gψ Δ̂ may correct the runtime candidate-action CPA/TTC used by the deployable VO shield.
```

This phase must not revive Gψ-as-PPO-feature. It is an eval / analysis / shield-side precheck.

---

## 1. Hard facts imported from previous audit

Do not rediscover these unless a file/path mismatch is found.

Previous audit established:

```text
1. EnvV2 obs_i planned_cpa is a spawn/design quantity sampled from threat-class CPA ranges.
   It is not runtime constant-velocity analytic CPA/TTC.

2. EnvV2 obs_i planned_ttc is a design-time / path-progress quantity.
   It does not use ground-truth future obstacle trajectory.

3. Phase B vo_like_filter computes runtime candidate-action CPA/TTC independently:
   tcpa = clip(-dot(rel, rel_vel) / ||rel_vel||², 0, horizon)
   cpa  = ||rel + rel_vel * tcpa||

4. Phase B runtime CPA/TTC is constant-relative-velocity analytic CPA/TTC.
   It depends on candidate velocity.

5. N4-O ordinary VO is parameter/formula equivalent to the Phase B best deployable filter.

6. Strong deployable anchor:
   vo_like_filter_h45_cpa1p2_h16
   horizon = 4.5
   cpa_safe = 1.2
   num_headings = 16
   formal success = 0.8333
   formal collision = 0.1667
```

Therefore, this phase targets the shield runtime CPA/TTC, not EnvV2 `planned_cpa/planned_ttc`.

---

## 2. Required GitHub sync before code changes

Before changing any code or config, sync the current project to GitHub:

```bash
git status --short

git add -A

if git diff --cached --quiet; then
    echo "[sync] no local changes to commit"
else
    git commit -m "sync before codex changes"
fi

git push origin main
```

Repository:

```text
https://github.com/oxpaxo/UAV-RISK-RL
```

If remote push fails because credentials are unavailable, record the failure in the report and continue only after a local commit is created. Do not claim remote sync success unless it actually succeeds.

---

## 3. Output locations

Guide file in repo:

```text
codex_guide/PHASE_N4U_RUNTIME_CV_ORACLE_CALIBRATION_GUIDE.md
```

Result root:

```text
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/
```

Required artifacts:

```text
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/PHASE_N4U_RUNTIME_CV_ORACLE_CALIBRATION_REPORT.md
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/PHASE_N4U_RUNTIME_CV_ORACLE_CALIBRATION_COMPLETE.flag
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/phase_n4u_runtime_cv_oracle_calibration_watcher.log
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/phase_n4u_runtime_cv_oracle_calibration_status.txt
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/tables/
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/plots/
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/logs/
```

---

## 4. Phase structure and conditional execution

```text
Step 0. Reframed premise preflight
Step 1. Oracle headroom vs strong runtime-CV geometric anchor
Step 2. Online σ² calibration, including heteroscedastic and ensemble uncertainty
Step 3. Fixed vs σ²-adaptive margin
Step 4. Conditional nonlinear/noisy stress oracle smoke
```

Execution logic:

```text
Run Step 0.
Run Step 1.

If Step 1 shows clean EnvV2 oracle headroom:
    Run Step 2 and Step 3.

If Step 1 shows no clean EnvV2 oracle headroom:
    Do not run Step 2/3 on clean EnvV2 as if N4-U is viable.
    Run Step 4 stress oracle smoke.

If Step 1 is statistically inconclusive:
    Expand episodes if feasible.
    If still inconclusive, stop with an inconclusive terminal decision.
```

Do not skip Step 1. It is the only gate deciding whether calibration/adaptive-margin experiments are meaningful on clean EnvV2.

---

# 5. Step 0 — Reframed premise preflight

## 5.1 Purpose

Do not redo the full previous audit. Verify that the previous audit files exist and that the new target is the runtime VO CPA/TTC.

Required checks:

```text
previous v2 report exists
previous CPA/TTC source audit exists
previous premise gate exists
Phase B vo_like_filter implementation exists
N4-O ordinary VO imports or equivalent implementation exists
strong anchor manifest exists or can be reconstructed
attention_full formal seed0 checkpoint exists
Gψ NLL checkpoint exists
```

## 5.2 Mandatory reframe addendum

Write:

```text
tables/phase_n4u_runtime_cv_reframe_addendum.csv
```

It must state:

```text
old invalid target:
    EnvV2 obs_i planned_cpa/planned_ttc

new valid target:
    Phase B / N4-O runtime candidate-action constant-relative-velocity CPA/TTC

reason:
    planned_cpa/ttc are design quantities;
    runtime VO CPA/TTC is analytic constant-relative-velocity and depends on candidate action.
```

## 5.3 Stop conditions

If the previous audit files are missing or inconsistent, re-run only the minimal audit necessary to confirm:

```text
runtime VO CPA/TTC is constant-relative-velocity;
planned_cpa/ttc are not oracle-like;
N4-O ordinary VO is equivalent to Phase B best anchor.
```

If any of these fails, stop with:

```text
STOP_PREMISE_RECHECK_FAILED.flag
```

---

# 6. Statistical protocol

This phase produces go/no-go decisions. Point estimates are not enough.

## 6.1 Evaluation seeds

Use:

```text
validation seeds: 900, 901
test seeds: 1000, 1001, 1002
final-heldout seeds: 1100, 1101, 1102
```

Validation may be used for small grid choices.

Test is for frozen comparison.

Final-heldout is only used if a candidate passes test gate and the report freezes it first.

## 6.2 Episode scale

For test-critical comparisons:

```text
6 scenarios
50 episodes per scenario per eval seed
3 eval seeds
900 episodes per method
```

Validation may use:

```text
minimum 30 episodes per scenario per eval seed
```

but all Step 1 oracle headroom test comparisons must use full test scale unless a resource stop condition is triggered.

## 6.3 Bootstrap confidence intervals

All critical deltas must report bootstrap 95% CI.

Required deltas:

```text
oracle_future_shield - strong_geom_anchor
oracle_future_shield - runtime_cv_same_structure
Gψ_point_fixed - strong_geom_anchor
Gψ_point_fixed - oracle_future_shield
heteroscedastic_adaptive - matched_Gψ_fixed
ensemble_adaptive - matched_Gψ_fixed
calibrated_directional_hetero - matched_Gψ_fixed
calibrated_directional_ensemble - matched_Gψ_fixed
stress_oracle - stress_strong_geom_anchor, if Step 4 runs
```

Bootstrap requirements:

```text
stratify by scenario and eval_seed at minimum
resamples >= 2000
if runtime is excessive, use >=1000 and report the reason
parallelize bootstrap computation
```

## 6.4 Noise floor

Define:

```text
no meaningful difference:
    |success_diff| < 0.02 and CI crosses 0
    or |collision_diff| < 0.02 and CI crosses 0
```

Do not make go/no-go decisions based on point estimates inside the noise floor.

## 6.5 Oracle headroom gates

Strong clean-headroom pass:

```text
success_diff oracle - strong_anchor has CI lower >= +0.05
and collision_diff oracle - strong_anchor has CI upper <= -0.05
```

Targeted clean-headroom pass:

```text
aggregate may be smaller,
but at least two pre-registered failure groups have:
    success_diff CI lower >= +0.05
    collision_diff CI upper <= 0
and aggregate collision is not meaningfully worse.
```

Pre-registered failure groups:

```text
eval_flow_high_speed
eval_flow_high_density
eval_flow_mixed_ood
eval_flow_high_threat
accel_decel
sinusoidal_lateral
crossing_or_sudden_threat
sudden_threat
```

No clean headroom:

```text
aggregate success_diff CI upper <= +0.02
and no pre-registered failure group satisfies targeted pass.
```

Ambiguous:

```text
CI crosses zero or falls between the above cases.
```

If ambiguous and resources allow, expand episodes before terminal decision. If still ambiguous, report inconclusive and do not proceed to full N4-U.

---

# 7. Step 1 — Oracle headroom vs strong runtime-CV anchor

## 7.1 Purpose

Primary gate:

```text
If the deployable VO shield had perfect future obstacle trajectories instead of runtime constant-velocity CPA/TTC, would it materially improve?
```

If not, clean EnvV2 does not provide enough prediction headroom for this contribution.

## 7.2 Required methods

Use frozen `attention_full` formal seed0 as base policy where applicable.

Evaluate at minimum:

```text
H0. attention_full_no_shield
H1. attention_full + strong_geom_anchor runtime-CV VO
H2. standalone/context strong_geom_anchor vo_like_filter_h45_cpa1p2_h16
H3. attention_full + oracle_future_shield_same_structure
H4. attention_full + Gψ_point_corrected_fixed_margin
```

Important:

```text
H1 is not generic ordinary VO.
H1 must use the strongest deployable Phase B/N4-O equivalent anchor.
Headroom gate is H3 - strongest deployable anchor, not H3 - weak/default VO.
```

If H1 and H2 differ because one uses attention raw action and the other is a standalone filter, report both. The headroom gate must use the stronger deployable geometric row as the anchor.

## 7.3 Strong anchor manifest

Write:

```text
tables/phase_n4u_runtime_cv_strong_anchor_manifest.csv
```

Include:

```text
implementation file/function
horizon
cpa_safe
num_headings
candidate velocity set
scoring formula
progress term
raw-distance term
whether candidate velocity is used in rel_vel
formal Phase B success/collision
N4-O equivalence status
```

## 7.4 Oracle future shield design

Oracle future shield is an upper bound, not deployable.

Use the same candidate velocity set and scoring as the strong anchor.

Only replace the obstacle future source:

```text
runtime-CV anchor:
    obstacle future = current obstacle position + current obstacle velocity * τ

oracle future:
    obstacle future = simulator/scripted ground-truth obstacle future at τ
```

Do not change:

```text
candidate set
progress scoring
action constraints
episode termination
collision/success definitions
base policy
```

Required validity checks:

```text
future trajectory alignment
coordinate frame consistency
horizon consistency
candidate set equivalence
no success/collision label leakage
oracle does not peek at future ego actions beyond the candidate-velocity assumption
```

Write:

```text
tables/phase_n4u_runtime_cv_oracle_validity_checks.csv
```

## 7.5 Gψ point-corrected fixed-margin shield

Use Gψ point predictions to replace runtime-CV obstacle future:

```text
p_gpsi(t+τ) = p_current + v_current * τ + Δ̂(τ)
```

Then compute candidate CPA/TTC from the corrected future.

This method does not use σ² yet.

It is the in-house two-step-style point-correction baseline.

Required outputs:

```text
cpa_cv(v)
ttc_cv(v)
cpa_oracle(v)
ttc_oracle(v)
cpa_gpsi(v)
ttc_gpsi(v)
delta_cpa_gpsi(v)
delta_ttc_gpsi(v)
gpsi_vs_oracle_candidate_agreement_rate
gpsi_vs_cv_candidate_change_rate
```

## 7.6 Step 1 decision

If Step 1 clean-headroom pass:

```text
continue to Step 2 and Step 3
```

If Step 1 no clean headroom:

```text
do not run Step 2/3 on clean EnvV2 as if contribution is viable
run Step 4 nonlinear/noisy stress oracle smoke
```

If Step 1 ambiguous:

```text
increase episodes if feasible;
otherwise report inconclusive and stop without full N4-U.
```

---

# 8. Step 2 — Online uncertainty calibration

Run Step 2 only if Step 1 shows clean EnvV2 headroom, or if Step 4 stress headroom passes and calibration is being run for stress.

## 8.1 Mandatory uncertainty sources

Compare both:

```text
A. HeadA heteroscedastic σ² from Gaussian NLL model
B. Ensemble σ² from Gψ ensemble disagreement
```

Ensemble is mandatory.

If compatible ensemble checkpoints exist, use them.

If not, train a minimal ensemble only after Step 1 passes:

```text
ensemble size K = 5
same dataset/spec as HeadA where possible
different seeds
same output delta target
```

If K=5 is impossible due to resource failure, K=3 may be used only with explicit report justification. If no credible ensemble is available, write:

```text
STOP_ENSEMBLE_UNAVAILABLE.flag
```

and do not claim uncertainty ablation completeness.

## 8.2 Required calibration data

For each active obstacle and horizon:

```text
delta_hat
log_sigma2_hetero
ensemble_mean_delta
ensemble_sigma2
actual future residual
squared error
projected error along approach direction
projected predicted variance
scenario
motion_mode
threat_class
horizon
valid label mask
```

Horizon split:

```text
1s
2s
4s
```

## 8.3 Required metrics

For heteroscedastic and ensemble uncertainty, report side-by-side:

```text
MSE by horizon
Gaussian NLL where applicable
variance/error correlation
coverage at 1σ / 2σ / 3σ
expected calibration error style bins
reliability diagram data
projected uncertainty calibration
by scenario
by motion mode
by threat class
```

## 8.4 Post-hoc calibration

Fit on validation only:

```text
σ²_cal = α * σ²
```

Required variants:

```text
global α_hetero
horizon-specific α_hetero
global α_ensemble
horizon-specific α_ensemble
```

Report validation and test before/after:

```text
NLL
coverage
ECE
projected calibration
```

Do not fit α on test.

---

# 9. Step 3 — Fixed vs adaptive margin

Run Step 3 only after Step 1 passes.

## 9.1 Required variants

Compare:

```text
M0. strong_geom_anchor runtime-CV fixed margin
M1. Gψ point-corrected fixed margin
M2. Gψ heteroscedastic scalar σ margin
M3. Gψ heteroscedastic directional σ margin
M4. Gψ calibrated heteroscedastic directional σ margin
M5. Gψ ensemble scalar σ margin
M6. Gψ ensemble directional σ margin
M7. Gψ calibrated ensemble directional σ margin
M8. optional calibrated combined hetero+ensemble directional margin
M9. oracle future shield upper bound
```

## 9.2 Margin formulas

Fixed:

```text
margin = base_margin
```

Scalar uncertainty:

```text
margin_i(τ) = base_margin + k * scalar_std_i(τ)
```

Directional uncertainty:

```text
margin_i(v, τ) = base_margin + k * sqrt(n_i(v,τ)^T Σ_i(τ) n_i(v,τ))
```

Calibrated directional:

```text
Σ_cal = α * Σ
margin_i(v, τ) = base_margin + k * sqrt(n_i(v,τ)^T Σ_cal_i(τ) n_i(v,τ))
```

Where:

```text
v = candidate ego velocity
τ = horizon
n_i(v,τ) = unit approach direction under candidate velocity and corrected obstacle future
Σ_i(τ) = diagonal covariance from exp(log_sigma2_i(τ)) or ensemble covariance
```

## 9.3 Validation grid

Use validation only.

Required small grid:

```text
k ∈ {0.0, 0.5, 1.0, 1.5, 2.0}
base_margin ∈ {strong anchor default, default ± small offset}
horizon aggregation ∈ {nearest horizon, worst-case, near-horizon weighted}
```

The grid must include `k=0`.

This guarantees a matched fixed-margin control inside the same implementation path.

## 9.4 Positive gate

Adaptive margin is positive only if, vs matched fixed Gψ-corrected margin:

```text
success_diff 95% CI lower > 0
collision_diff 95% CI upper < 0
progress does not meaningfully collapse
near_miss / filtered unsafe improves or stays acceptable
gain appears in failure-mode breakdown, not only aggregate noise
uncertainty calibration is acceptable or validation-calibrated
```

If adaptive wins only by globally increasing conservatism and causing large progress loss, report:

```text
conservative margin effect, not intelligent uncertainty use
```

---

# 10. Step 4 — Conditional nonlinear/noisy stress oracle smoke

Step 4 is mandatory if Step 1 finds no clean EnvV2 headroom.

It may also run as secondary evidence if Step 1 passes and resources are healthy.

## 10.1 Purpose

If clean EnvV2 does not reward prediction, test whether realistic stress creates prediction headroom.

Do not weaken baselines by hiding CPA/TTC. Increase task difficulty fairly.

## 10.2 Implementation

Do not modify frozen EnvV2 core.

Use wrapper/config stress modes:

```text
EnvV2-Stress-Nonlinear
EnvV2-NoisyState
EnvV2-NoisyNonlinear
```

All methods face the same stressed observations/states.

## 10.3 Stress types

Use controlled stress such as:

```text
stronger accel_decel
stronger sinusoidal_lateral
piecewise velocity switch
sudden lateral maneuver
turning/curved obstacle
short acceleration burst
position noise
velocity noise
short dropout
low-pass filtering delay
small bias
```

## 10.4 Required stress comparisons

At smoke scale compare:

```text
stress_strong_geom_anchor
stress_oracle_future_shield
stress_Gψ_point_fixed, if feasible
```

Use bootstrap CI as in Step 1.

If stress oracle headroom exists, report:

```text
clean EnvV2 has low prediction headroom;
stress/noisy suite is required for a prediction-based contribution.
```

---

# 11. Full-pipeline parallelism and resource policy

The user has confirmed prior maximum parallel settings still underused the machine:

```text
training CPU ~50%
eval CPU ~17%
```

This phase must use aggressive process-level parallelism while preserving correctness.

## 11.1 Resource targets

```text
eval CPU target: >=70%, ideally 80–90%
analysis CPU target: >=70% during bootstrap/aggregation
ensemble training CPU target, if any: >=80%, ideally near 90%
```

## 11.2 Per-worker environment

For each eval/training/analysis worker:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

Use process-level parallelism rather than intra-worker thread oversubscription.

## 11.3 Eval worker scaling

Start aggressively:

```text
initial eval concurrency = 32 workers
```

Then scale:

```text
if CPU <70%, RAM/disk/log healthy:
    increase to 48 workers

if CPU still <80%, no IO/RAM pressure:
    increase to 64 workers

if CPU reaches 80–90% or per-worker throughput collapses:
    hold current concurrency

if RAM/disk/IO/log stalls occur:
    reduce concurrency by 25–50%, optional/stress workers first
```

If 64 is too high for the machine, the watcher must back off and report why.

## 11.4 Ensemble training scaling

If Step 1 passes and ensemble training is needed:

```text
start K=5 ensemble jobs concurrently if resources allow
target CPU >=80%
keep each job device=cpu and thread env vars = 1
if unhealthy, train 3 first, then remaining 2
```

## 11.5 Analysis scaling

Bootstrap CI and calibration aggregation must be parallelized.

Suggested:

```text
analysis workers start = 16
increase to 24 if CPU remains low and memory healthy
cache intermediate aggregates
avoid repeated full reads of large CSVs
```

## 11.6 Health thresholds

Stop or scale down if:

```text
disk free < 10 GB
RAM free < 8 GB
worker stale > 45 minutes
mandatory worker repeated failure
mean throughput collapses > 50% for > 20 minutes
```

---

# 12. Watcher behavior

Create:

```text
scripts/watch_phase_n4u_runtime_cv_oracle_calibration.sh
```

Watcher requirements:

```text
blocking watcher
Codex determines completion flag from guide
may exit only on complete flag or stop flag/condition
otherwise continue running
```

Adaptive heartbeat:

```text
initial interval = 10 minutes
if progress per 10 minutes is small but healthy/stable:
    increase to 20 minutes
if jobs near completion:
    reduce to 10 minutes
if debugging/resource pressure/stale logs/finalization:
    reduce to 5 minutes
```

Every heartbeat must report:

```text
current step
current concurrency
CPU/RAM/disk
completed shards
running workers
stale workers
estimated remaining work
next heartbeat interval
```

---

# 13. Suggested files

Prefer phase-specific utilities without breaking old scripts.

Suggested additions:

```text
utils/gpsi_risk_correction.py
utils/gpsi_uncertainty_calibration.py
utils/bootstrap_ci.py

scripts/eval_env_v2_n4u_runtime_cv_oracle_headroom.py
scripts/eval_env_v2_n4u_runtime_cv_gpsi_point.py
scripts/train_gpsi_heada_ensemble.py
scripts/analyze_env_v2_n4u_online_calibration.py
scripts/eval_env_v2_n4u_runtime_cv_adaptive_margin.py
scripts/eval_env_v2_n4u_stress_oracle_smoke.py
scripts/analyze_env_v2_phase_n4u_runtime_cv_oracle_calibration.py
scripts/watch_phase_n4u_runtime_cv_oracle_calibration.sh
```

Reuse existing Phase B / N4-O shield code whenever possible.

Do not create a new generic VO implementation if the existing strong anchor can be imported.

---

# 14. Preflight checks

Run:

```bash
python -m py_compile utils/gpsi_risk_correction.py
python -m py_compile utils/gpsi_uncertainty_calibration.py
python -m py_compile utils/bootstrap_ci.py
python -m py_compile scripts/eval_env_v2_n4u_runtime_cv_oracle_headroom.py
python -m py_compile scripts/eval_env_v2_n4u_runtime_cv_gpsi_point.py
python -m py_compile scripts/train_gpsi_heada_ensemble.py
python -m py_compile scripts/analyze_env_v2_n4u_online_calibration.py
python -m py_compile scripts/eval_env_v2_n4u_runtime_cv_adaptive_margin.py
python -m py_compile scripts/eval_env_v2_n4u_stress_oracle_smoke.py
python -m py_compile scripts/analyze_env_v2_phase_n4u_runtime_cv_oracle_calibration.py
bash -n scripts/watch_phase_n4u_runtime_cv_oracle_calibration.sh
```

Also verify:

```text
attention_full checkpoint exists
Gψ checkpoint exists
Phase B strong anchor import works
candidate set equality between anchor/oracle/Gψ variants
future oracle trajectory access works
Gψ normalization/denormalization is correct
bootstrap CI script works on a small sample
```

---

# 15. Stop flags

Use:

```text
STOP_RESOURCE_UNSAFE.flag
STOP_PREFLIGHT_FAILED.flag
STOP_PREMISE_RECHECK_FAILED.flag
STOP_ORACLE_IMPL_INVALID.flag
STOP_GPSI_RISK_IMPL_INVALID.flag
STOP_ENSEMBLE_UNAVAILABLE.flag
STOP_CALIBRATION_INVALID.flag
STOP_EVAL_FAILED.flag
STOP_SELECTOR_CONTAMINATED.flag
STOP_ANALYSIS_FAILED.flag
```

Stop immediately for:

```text
oracle implementation invalid
strong anchor mismatch
candidate set mismatch
coordinate frame mismatch
selector/test contamination
resource unsafe
```

---

# 16. Complete conditions

Write complete flag only when one valid branch completes.

Branch A — Clean headroom pass:

```text
Step 1 complete with clean oracle headroom
Step 2 calibration complete
Step 3 adaptive-margin smoke complete
report written
```

Branch B — No clean headroom:

```text
Step 1 complete with no clean headroom
Step 4 stress oracle smoke complete
report written
```

Branch C — Inconclusive:

```text
Step 1 completed
episode expansion attempted or infeasible
report written with inconclusive terminal decision
```

Complete flag:

```text
results/env_v2_phase_n4u_runtime_cv_oracle_calibration/PHASE_N4U_RUNTIME_CV_ORACLE_CALIBRATION_COMPLETE.flag
```

---

# 17. Terminal decisions

Use exactly one:

```text
phase_n4u_runtime_cv_complete_ready_for_n4u
phase_n4u_runtime_cv_complete_oracle_headroom_but_calibration_blocked
phase_n4u_runtime_cv_complete_oracle_headroom_but_gpsi_prediction_blocked
phase_n4u_runtime_cv_complete_no_clean_headroom_stress_promising
phase_n4u_runtime_cv_complete_no_clean_headroom_stress_not_promising
phase_n4u_runtime_cv_complete_adaptive_margin_failed
phase_n4u_runtime_cv_complete_oracle_headroom_inconclusive
phase_n4u_runtime_cv_stopped_resource_unsafe
phase_n4u_runtime_cv_stopped_preflight_failed
phase_n4u_runtime_cv_stopped_premise_recheck_failed
phase_n4u_runtime_cv_stopped_oracle_impl_invalid
phase_n4u_runtime_cv_stopped_gpsi_risk_impl_invalid
phase_n4u_runtime_cv_stopped_ensemble_unavailable
phase_n4u_runtime_cv_stopped_eval_failed
phase_n4u_runtime_cv_stopped_selector_contaminated
phase_n4u_runtime_cv_stopped_analysis_failed
```

Meaning:

```text
ready_for_n4u:
    clean oracle headroom exists, σ² calibration usable, adaptive margin beats matched fixed margin.

oracle_headroom_but_calibration_blocked:
    clean oracle headroom exists but uncertainty is not usable.

oracle_headroom_but_gpsi_prediction_blocked:
    clean oracle headroom exists but Gψ point correction is too far from oracle or harmful.

no_clean_headroom_stress_promising:
    clean EnvV2 has low headroom but stress/noisy suite shows headroom.

no_clean_headroom_stress_not_promising:
    neither clean nor stress settings show enough prediction headroom.

adaptive_margin_failed:
    point correction may work, but σ² adaptive margin does not beat matched fixed margin.

oracle_headroom_inconclusive:
    CI remains inconclusive after feasible expansion.
```

---

# 18. Final report requirements

The final report must include:

```text
terminal_decision
GitHub sync status and commit
changed files
previous v2 audit facts imported
reframe addendum
strong anchor manifest
candidate set equivalence checks
oracle implementation validity checks
statistical protocol
episode counts
bootstrap CI settings
oracle headroom vs strong anchor
oracle headroom by scenario/motion/threat
Gψ point correction vs strong anchor and oracle
online heteroscedastic σ² calibration
online ensemble σ² calibration
hetero vs ensemble calibration comparison
temperature scaling values
fixed vs adaptive margin results
adaptive margin bootstrap CIs
failure-mode breakdown
stress oracle smoke results if clean headroom fails
resource utilization and concurrency scaling
watcher heartbeat intervals
whether clean EnvV2 supports prediction-based contribution
whether stress/noisy suite should become primary
whether formal N4-U can start
N4-O status
N4-U status
recommended next phase
```

Explicitly state:

```text
This phase targets runtime candidate-action CPA/TTC in the strong VO shield,
not EnvV2 obs_i planned_cpa/planned_ttc.
```
