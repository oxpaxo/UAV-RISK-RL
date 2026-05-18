# Phase N4U-PRECHECK-ORACLE-CALIBRATION v2 Guide

## 0. Why this guide exists

This phase is a **go/no-go precheck** before any formal N4-U. It must decide whether the project has a defensible path after the failure of Gψ-as-PPO-feature.

Previous decisions:

```text
Gψ-as-PPO-feature is no longer the primary route.
P3 / S2-D / non-attention decoupling did not produce stable multi-seed PPO gains.
Do not continue tuning z_i, raw Δ̂/logvar PPO inputs, S2-D gates, or non-attention PPO aggregators.
Do not proceed to formal N4-U before this precheck passes.
```

New route under test:

```text
frozen stable policy
+ strongest deployable geometric shield baseline
+ Gψ risk-level correction of CPA/TTC
+ online-calibrated uncertainty
+ σ²-adaptive margin
```

This phase must answer five questions:

```text
Q1. Premise check:
    Are EnvV2 planned CPA/TTC actually constant-velocity analytic quantities?
    If not, the “Gψ corrects constant-velocity CPA/TTC” framing is invalid.

Q2. Oracle headroom:
    Does a perfect future-trajectory shield beat the strongest deployable geometric baseline?

Q3. Gψ point correction:
    Does Gψ point prediction close any part of the oracle-vs-geometry gap?

Q4. Uncertainty calibration:
    Is HeadA heteroscedastic σ² calibrated online, and how does it compare to ensemble uncertainty?

Q5. Adaptive margin:
    Does σ²-adaptive margin beat a matched fixed-margin shield with statistically meaningful confidence?
```

This phase is primarily eval-only and analysis-only. Do not train PPO.

---

## 1. Required GitHub sync before changes

Before changing code/config/scripts, sync the current project to GitHub:

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

If remote push fails due to authentication, create the local commit, record the remote-push failure in the report, and do not claim remote sync success.

---

## 2. Output locations

Place this guide in the repo as:

```text
codex_guide/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_GUIDE.md
```

Use result root:

```text
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/
```

Required artifacts:

```text
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_REPORT.md
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_COMPLETE.flag
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/phase_n4u_precheck_oracle_calibration_v2_watcher.log
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/phase_n4u_precheck_oracle_calibration_v2_status.txt
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/tables/
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/plots/
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/logs/
```

---

## 3. Non-negotiable baseline anchor

The oracle headroom gate must be anchored to the **strongest deployable geometric baseline**, not an arbitrary ordinary/default VO implementation.

The required deployable geometric anchors are:

```text
Phase B strongest VO-like filter:
  vo_like_filter_h45_cpa1p2_h16
  success=0.8333
  collision=0.1667
  near_miss=0.7200
  progress=0.9880
  filter_trigger_rate=0.3509

N4-O same-shield attention ordinary VO:
  attention_full + ordinary VO
  success=0.8300
  collision=0.1700

N4-O same-shield P3 seed0 + ordinary VO:
  P3_1500k + ordinary VO
  success=0.8811
  collision=0.1156
  This is conditional on unstable P3 seed0 and must not be used as a stable deployed baseline,
  but it is useful as a system-potential upper context.
```

For this phase, define:

```text
strong_geom_anchor = strongest deployable geometric method that is not dependent on an unstable learned seed.
Default required anchor:
  Phase B vo_like_filter_h45_cpa1p2_h16
```

If the implementation uses the N4-O ordinary VO wrapper, verify and report that its parameters match the Phase B best VO-like filter:

```text
horizon = 4.5
cpa_safe = 1.2
num_headings = 16
same candidate velocity set
same scoring logic
same unsafe-action handling
```

The headroom quantity must be:

```text
oracle_headroom_success = oracle_future_shield_success - strong_geom_anchor_success
oracle_headroom_collision = strong_geom_anchor_collision - oracle_future_shield_collision
```

Do not define headroom relative to:

```text
attention_full no shield
a generic/default VO
a weaker CPA/TTC APF
an unverified shield variant
```

Context rows may include attention/no_z/no-shield, but the go/no-go decision must use `strong_geom_anchor`.

---

## 4. Statistical protocol

This phase cannot make go/no-go decisions from noisy point estimates.

### 4.1 Episode scale

Unless resource safety requires a smaller smoke scale, use:

```text
validation seeds: 900, 901
test seeds: 1000, 1001, 1002
final-heldout seeds: 1100, 1101, 1102

scenarios: all 6 standard eval scenarios
episodes per scenario per eval seed: 50
```

This gives:

```text
validation full scale: 2 × 6 × 50 = 600 episodes per method
test full scale: 3 × 6 × 50 = 900 episodes per frozen method
final-heldout full scale: 3 × 6 × 50 = 900 episodes per frozen method
```

If runtime forces a reduced validation scale, minimum allowed:

```text
30 episodes per scenario per validation seed
```

But test-scale final comparisons must use 50 episodes per scenario per eval seed unless a stop flag explains why not.

### 4.2 Bootstrap confidence intervals

For all key comparisons, compute paired or stratified bootstrap 95% confidence intervals.

Required comparisons:

```text
oracle_future_shield - strong_geom_anchor
gpsi_point_corrected_fixed_margin - strong_geom_anchor
gpsi_point_corrected_fixed_margin - ordinary_cv_fixed_margin
heteroscedastic_sigma_adaptive - matched_fixed_margin
ensemble_sigma_adaptive - matched_fixed_margin
calibrated_directional_sigma_adaptive - matched_fixed_margin
calibrated_directional_sigma_adaptive - ensemble_sigma_adaptive
```

Bootstrap stratification:

```text
stratify by scenario and eval_seed at minimum;
also report breakdown by motion_mode and threat_class where sample counts permit.
```

Use at least:

```text
bootstrap resamples = 2000
```

If runtime is high, allow 1000 but record it.

### 4.3 Pass/fail gates

A comparison is considered positive only if:

```text
success_diff 95% CI lower bound > 0
AND collision_diff 95% CI upper bound < 0
```

where:

```text
success_diff = method_A_success - method_B_success
collision_diff = method_A_collision - method_B_collision
```

For failure-mode targeted claims, the method must satisfy:

```text
aggregate result is not worse beyond noise floor
AND at least one pre-registered failure group has a positive CI-supported collision reduction
AND progress does not collapse.
```

Noise floor:

```text
If |success_diff| < 0.02 and |collision_diff| < 0.02 with CI crossing 0,
treat as no meaningful difference.
```

Do not let a point estimate alone trigger `ready_for_n4u`.

### 4.4 Test/validation discipline

Validation may be used for:

```text
choosing base_margin/k/horizon aggregation
temperature scaling σ²
choosing scalar vs directional margin variant for frozen test
```

Test may not be used for:

```text
parameter tuning
variant selection before frozen declaration
changing the stress/noise settings
changing CPA/TTC implementation
```

If test results are used to modify the method, write:

```text
STOP_SELECTOR_CONTAMINATED.flag
```

---

## 5. Main phase structure

Run:

```text
Step A. CPA/TTC source audit and premise gate
Step B. Strong-baseline anchored oracle headroom
Step C. Gψ point-estimate risk-level correction
Step D. Online uncertainty calibration: heteroscedastic σ² and ensemble σ²
Step E. Fixed vs adaptive margin smoke with statistical gates
Step F. Conditional mandatory nonlinear/noisy stress smoke
```

---

# 6. Step A — CPA/TTC source audit and premise gate

## 6.1 Purpose

The project framing depends on this fact:

```text
EnvV2 planned CPA/TTC and ordinary VO CPA/TTC are constant-velocity analytic estimates.
```

If they are already ground-truth future-aware or oracle-like, then the “Gψ corrects constant-velocity CPA/TTC” premise is invalid.

## 6.2 Required audit questions

From code, identify:

```text
1. How are obs_i planned_cpa / planned_ttc computed?
2. Are they based on constant-velocity extrapolation from current position/velocity?
3. Do they use ground-truth future obstacle trajectory?
4. Do they use current UAV planned velocity or candidate velocity?
5. What are units, clipping, normalization, invalid-value handling?
6. Does Phase B vo_like_filter compute CPA/TTC independently of obs_i planned_cpa/ttc?
7. Does N4-O ordinary shield directly reuse Phase B vo_like_filter?
8. Are EnvV2, Phase B, and N4-O formulas exactly same, parameter-compatible, or only conceptually similar?
```

## 6.3 Required outputs

```text
tables/phase_n4u_precheck_v2_cpa_ttc_audit.csv
tables/phase_n4u_precheck_v2_cpa_ttc_formula_manifest.csv
tables/phase_n4u_precheck_v2_strong_geom_anchor_manifest.csv
```

## 6.4 Premise gate

If planned_cpa/ttc or shield CPA/TTC are ground-truth future-aware/oracle-like, this is not a soft wording issue.

Write terminal decision:

```text
phase_n4u_precheck_v2_stopped_premise_invalidated
```

Then report:

```text
The clean EnvV2 contribution framing based on correcting constant-velocity CPA/TTC is invalid.
Recommended next route:
  stress/noisy-state setting
  or redesign around another failure mode
```

If planned_cpa/ttc are constant-velocity based but strong_geom_anchor independently recomputes CPA/TTC, continue.

---

# 7. Step B — Strong-baseline anchored oracle headroom

## 7.1 Required methods

Use frozen `attention_full` as the base learned policy for policy+shield methods.

Mandatory rows:

```text
B0. attention_full_no_shield
B1. attention_full + strong_geom_anchor ordinary VO
B2. strongest deployable geometric baseline standalone/context row
B3. attention_full + oracle_future_shield
B4. attention_full + Gψ point-corrected fixed-margin shield
```

`B1` must match Phase B best VO-like parameters unless audit proves an exact reusable N4-O equivalent.

Optional context rows:

```text
no_z_no_shield
no_z + strong_geom_anchor ordinary VO
cpa_ttc_weighted_apf_alpha3
P3_seed0 + ordinary VO as conditional historical context only
```

## 7.2 Oracle future shield

Oracle shield is an upper bound, not deployable.

For each candidate velocity:

```text
Use simulator ground-truth future obstacle positions over the shield horizon.
Keep candidate velocity set identical to strong_geom_anchor ordinary VO.
Keep scoring/progress logic identical except obstacle future source.
Do not use future ego actions beyond the same constant-candidate-velocity assumption used by ordinary VO.
```

Required validity checks:

```text
oracle future labels exist and align with current timestep
coordinate frame matches ordinary VO/Gψ computation
oracle future does not leak success/collision labels directly
candidate velocity set identical to ordinary VO
all methods evaluated on same episodes/seeds/scenarios
```

## 7.3 Headroom gate

Compute with 95% bootstrap CI:

```text
oracle_future_shield - strong_geom_anchor
```

Terminal implications:

```text
If oracle does not beat strong_geom_anchor with CI-supported collision reduction:
  clean EnvV2 has insufficient prediction headroom.
  Step F nonlinear/noisy stress becomes mandatory.

If oracle beats strong_geom_anchor:
  prediction headroom exists; continue to Gψ point correction and uncertainty tests.
```

Do not use oracle-vs-weak-ordinary as the gate.

---

# 8. Step C — Gψ point-estimate risk correction

## 8.1 Purpose

Implement risk-level interface, not PPO feature interface.

For candidate velocity `v`, compute:

```text
cpa_cv(v)
ttc_cv(v)
cpa_gpsi(v)
ttc_gpsi(v)
delta_cpa(v) = cpa_gpsi(v) - cpa_cv(v)
delta_ttc(v) = ttc_gpsi(v) - ttc_cv(v)
```

Corrected obstacle future:

```text
p_gpsi(t+τ) = p_current + τ * v_current + Δ̂(τ)
```

Use corrected future trajectory to compute candidate safety.

## 8.2 Required utility

Add or update:

```text
utils/gpsi_risk_correction.py
```

Functions should include:

```text
compute_cv_future_positions(...)
compute_gpsi_corrected_future_positions(...)
compute_oracle_future_positions(...)
compute_candidate_cpa_ttc_from_future(...)
compute_delta_cpa_ttc(...)
```

Do not modify EnvV2 core.

## 8.3 Required diagnostics

```text
delta_cpa distribution
delta_ttc distribution
gpsi_vs_oracle_cpa_error
gpsi_vs_oracle_ttc_error
gpsi_vs_ordinary_candidate_change_rate
gpsi_vs_oracle_candidate_agreement_rate
failure mode breakdown of correction error
```

## 8.4 Interpretation

Report explicitly:

```text
Gψ point-corrected CPA/TTC is a two-step-style risk-level baseline.
It is not by itself the main novelty.
The main novelty candidate is calibrated uncertainty-aware shielding.
```

---

# 9. Step D — Online uncertainty calibration: heteroscedastic and ensemble

## 9.1 Purpose

A σ²-adaptive shield is only meaningful if uncertainty is credible online.

Evaluate both:

```text
HeadA heteroscedastic σ² from Gaussian NLL
ensemble σ² from prediction disagreement
```

The ensemble baseline is mandatory. Reviewers will ask for it; collecting it now avoids rerunning this entire precheck.

## 9.2 Ensemble implementation

If multiple existing Gψ checkpoints/seeds already exist, use them if protocol-compatible.

Otherwise train or load a small ensemble as a phase subtask only if needed:

```text
ensemble size: 3 minimum
same architecture/data/splits as HeadA NLL where feasible
different random seeds
do not alter EnvV2
record training provenance
```

If training the ensemble is too expensive for this phase, create:

```text
STOP_CALIBRATION_INVALID.flag
```

only if no credible ensemble can be produced. Prefer completing a minimal 3-member ensemble because it is central to the uncertainty comparison.

Ensemble uncertainty:

```text
ensemble_mean_delta = mean_j Δ̂_j
ensemble_epistemic_var = var_j Δ̂_j
```

For fair margin comparison, report:

```text
heteroscedastic aleatoric σ²
ensemble epistemic σ²
optional combined σ² = heteroscedastic + ensemble
```

## 9.3 Required online calibration data

Collect under online eval distributions:

```text
frozen attention_full no shield
attention_full + strong_geom_anchor ordinary VO
attention_full + Gψ point-corrected fixed shield
stress/noisy variants if Step F triggered
```

For each active obstacle and horizon:

```text
delta_hat
log_sigma2
ensemble_delta_hats
ensemble_sigma2
actual future residual
squared error
projected error along approach direction
projected heteroscedastic variance
projected ensemble variance
scenario
motion_mode
threat_class
distance/risk/ttc bins if available
valid label mask
```

Horizons:

```text
1s / 2s / 4s
```

## 9.4 Calibration metrics

For heteroscedastic σ² and ensemble σ² side by side:

```text
MSE by horizon
Gaussian NLL where applicable
variance/error correlation
1σ / 2σ / 3σ coverage
ECE-style calibration bins
reliability diagram data
projected uncertainty calibration
calibration by scenario/motion/threat
```

## 9.5 Post-hoc calibration

Fit on validation only:

```text
σ²_cal = α * σ²
```

For both uncertainty sources:

```text
α_hetero
α_ensemble
optional α_by_horizon
```

Report validation and test:

```text
NLL before/after
coverage before/after
ECE before/after
```

Do not fit on test.

---

# 10. Step E — Fixed vs adaptive margin smoke

## 10.1 Required methods

Compare:

```text
M0. strong_geom_anchor fixed margin
M1. Gψ point-corrected fixed margin
M2. Gψ point-corrected heteroscedastic scalar σ margin
M3. Gψ point-corrected heteroscedastic directional σ margin
M4. Gψ point-corrected calibrated heteroscedastic directional σ margin
M5. Gψ point-corrected ensemble scalar σ margin
M6. Gψ point-corrected ensemble directional σ margin
M7. Gψ point-corrected calibrated ensemble directional σ margin
M8. optional combined hetero+ensemble calibrated directional margin
M9. oracle_future_shield upper bound
```

The comparison of M4 vs M1 and M7 vs M1 is mandatory.

## 10.2 Margin formulas

Fixed:

```text
margin = base_margin
```

Scalar σ:

```text
margin_i(τ) = base_margin + k * scalar_std_i(τ)
```

Directional:

```text
margin_i(v,τ) = base_margin + k * sqrt(n_i(v,τ)^T Σ_i(τ) n_i(v,τ))
```

Calibrated directional:

```text
Σ_cal = α * Σ
margin_i(v,τ) = base_margin + k * sqrt(n_i(v,τ)^T Σ_cal_i(τ) n_i(v,τ))
```

Where:

```text
v = candidate velocity
τ = horizon
n_i(v,τ) = candidate-relative approach direction
Σ_i(τ) = diagonal covariance
```

If shield is xy-plane only, document how z variance is handled.

## 10.3 Validation grid

Use validation only.

Small grid:

```text
k ∈ {0.0, 0.5, 1.0, 1.5, 2.0}
base_margin ∈ {strong_geom_anchor default, default ± small offset if safe}
horizon aggregation ∈ {near-horizon weighted, worst-case, min-cpa horizon}
```

Do not run a large hyperparameter search.

## 10.4 Required outputs

For each method:

```text
success
collision
near_miss
progress
shield_trigger_rate
filter_delta_norm_mean/p95
raw_action_unsafe_rate
filtered_action_unsafe_rate
min_predicted_cpa before/after
candidate change rate vs strong_geom_anchor
candidate agreement with oracle
scenario breakdown
motion mode breakdown
threat class breakdown
bootstrap CI for key pairwise differences
```

## 10.5 Adaptive margin pass gate

Adaptive margin is positive only if:

```text
adaptive method beats matched fixed Gψ margin with CI-supported collision reduction;
success does not decrease beyond noise floor;
progress does not collapse;
gain appears in at least one pre-registered failure mode;
calibration is acceptable or corrected using validation-only scaling.
```

If adaptive wins only by being more conservative everywhere:

```text
report conservative margin effect, not intelligent uncertainty use.
```

---

# 11. Step F — Conditional mandatory nonlinear/noisy stress smoke

Step F is not always optional.

## 11.1 Trigger

If Step B finds:

```text
oracle_future_shield does not beat strong_geom_anchor on clean EnvV2 with CI-supported gains
```

then Step F becomes mandatory before writing a `no_clean_env_headroom_use_stress` terminal decision.

Do not conclude “use stress” without running at least stress smoke.

## 11.2 Purpose

Do not weaken baselines by hiding CPA/TTC.

Instead, fairly increase task difficulty:

```text
more nonlinear obstacle dynamics
realistic state-estimation noise
```

All methods face the same stressed observations/states.

## 11.3 Implementation rules

Do not modify frozen EnvV2 core permanently.

Implement as wrapper/config/stress mode where possible:

```text
EnvV2-Stress-Nonlinear
EnvV2-NoisyState
EnvV2-NoisyNonlinear
```

## 11.4 Stress types

Nonlinear:

```text
stronger accel_decel
stronger sinusoidal_lateral
piecewise velocity switch
sudden lateral maneuver
turning/curved obstacle
short acceleration burst
```

Noisy state:

```text
position noise
velocity noise
short dropout
low-pass delay
small bias
```

All methods except oracle upper bound consume the same noisy estimated state.

## 11.5 Stress smoke rows

At minimum:

```text
strong_geom_anchor clean vs stress
oracle_future_shield clean vs stress
Gψ point-corrected fixed clean vs stress
best calibrated adaptive margin clean vs stress
```

If stress creates oracle headroom where clean EnvV2 did not, report:

```text
clean EnvV2 insufficiently stresses prediction;
future formal N4-U should focus on stress/noisy suite.
```

---

# 12. Full-pipeline parallelism and resource policy

The user confirmed prior maximum parallel settings still used only about:

```text
50% CPU during training
17% CPU during eval
```

This phase must target much higher utilization while preserving correctness.

## 12.1 Targets

```text
Eval CPU utilization target: >= 70%, ideally 80–90% if healthy.
Analysis CPU utilization target: high enough to avoid idle cores, while not exhausting RAM.
Training target if ensemble training is needed: >= 80%, ideally near 90%, while preserving individual job semantics.
```

## 12.2 Per-worker thread settings

For every eval/training worker:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

Use process-level parallelism rather than per-process thread oversubscription.

## 12.3 Eval worker scaling

Start aggressively:

```text
initial eval concurrency = 24 workers
```

After first health check:

```text
if CPU < 70%, RAM/disk healthy, logs advancing:
  increase to 32 workers

if CPU still < 80% and no IO/RAM pressure:
  increase to 40 workers

if CPU reaches 80–90% or per-worker throughput collapses:
  hold

if RAM/disk/IO/log stalls occur:
  reduce concurrency by 25–50%, optional/stress jobs first
```

If 40 workers is too high, watcher must back off and report why.

## 12.4 Ensemble training concurrency

If training a Gψ ensemble is required:

```text
run ensemble members in parallel if safe
target CPU >= 80%
preserve each training job’s semantics
reduce optional stress/eval jobs before mandatory ensemble jobs if resource pressure occurs
```

## 12.5 Analysis parallelism

For calibration and bootstrap:

```text
use multiprocessing/chunked map-reduce
cache intermediate aggregate tables
avoid repeated full CSV loads
```

Suggested workers:

```text
start 12
increase to 16–24 if CPU/RAM healthy
```

Bootstrap should be parallelized.

## 12.6 Adaptive heartbeat

Do not use fixed 5-minute heartbeat for long phases.

Use:

```text
initial heartbeat = 10 minutes
if progress per 10 minutes is small but stable:
  expand to 20 minutes
if jobs near completion:
  reduce to 10 minutes
if debugging/resource pressure/stale logs/finalization:
  reduce to 5 minutes
```

Heartbeat must include:

```text
current concurrency
CPU/RAM/disk
completed shards
running workers
stale workers
estimated remaining work
next heartbeat interval
```

---

# 13. Required scripts / code changes

Suggested files:

```text
utils/gpsi_risk_correction.py
utils/gpsi_uncertainty_calibration.py
utils/bootstrap_ci.py

scripts/audit_env_v2_cpa_ttc_sources_v2.py
scripts/eval_env_v2_n4u_oracle_headroom_v2.py
scripts/eval_env_v2_n4u_gpsi_corrected_shield_v2.py
scripts/analyze_env_v2_n4u_online_calibration_v2.py
scripts/eval_env_v2_n4u_adaptive_margin_smoke_v2.py
scripts/eval_env_v2_n4u_stress_smoke_v2.py
scripts/analyze_env_v2_phase_n4u_precheck_v2.py
scripts/watch_phase_n4u_precheck_oracle_calibration_v2.sh
```

Reuse existing scripts if clean. Avoid fragile copy-paste forks.

---

# 14. Preflight checks

Run:

```bash
python -m py_compile utils/gpsi_risk_correction.py
python -m py_compile utils/gpsi_uncertainty_calibration.py
python -m py_compile utils/bootstrap_ci.py
python -m py_compile scripts/audit_env_v2_cpa_ttc_sources_v2.py
python -m py_compile scripts/eval_env_v2_n4u_oracle_headroom_v2.py
python -m py_compile scripts/eval_env_v2_n4u_gpsi_corrected_shield_v2.py
python -m py_compile scripts/analyze_env_v2_n4u_online_calibration_v2.py
python -m py_compile scripts/eval_env_v2_n4u_adaptive_margin_smoke_v2.py
python -m py_compile scripts/eval_env_v2_n4u_stress_smoke_v2.py
python -m py_compile scripts/analyze_env_v2_phase_n4u_precheck_v2.py
bash -n scripts/watch_phase_n4u_precheck_oracle_calibration_v2.sh
```

Also verify:

```text
attention_full formal seed0 checkpoint exists
Gψ NLL checkpoint exists
ensemble checkpoints exist or can be trained
strong_geom_anchor implementation available
oracle future access valid
Gψ normalization/denormalization correct
candidate velocity set matches strong_geom_anchor
bootstrap CI code works
```

---

# 15. Stop conditions

Create stop flags as needed:

```text
STOP_RESOURCE_UNSAFE.flag
STOP_PREFLIGHT_FAILED.flag
STOP_CPA_TTC_AUDIT_FAILED.flag
STOP_PREMISE_INVALIDATED.flag
STOP_ORACLE_IMPL_INVALID.flag
STOP_GPSI_RISK_IMPL_INVALID.flag
STOP_CALIBRATION_INVALID.flag
STOP_ENSEMBLE_UNAVAILABLE.flag
STOP_EVAL_FAILED.flag
STOP_SELECTOR_CONTAMINATED.flag
STOP_ANALYSIS_FAILED.flag
```

Immediate stop:

```text
ground-truth/oracle-like planned CPA/TTC invalidates premise;
oracle implementation uses future labels incorrectly;
coordinate frame mismatch invalidates Gψ correction;
test seed contamination;
resource unsafe;
strong_geom_anchor cannot be reproduced or imported.
```

---

# 16. Complete condition

Write complete flag only when:

```text
CPA/TTC audit completed;
strong_geom_anchor locked;
oracle headroom evaluated against strong_geom_anchor with bootstrap CI;
Gψ point correction evaluated;
heteroscedastic and ensemble calibration analyzed;
fixed/adaptive margin smoke completed or blocked by a valid gate;
conditional stress smoke completed if clean EnvV2 headroom is insufficient;
report written;
decision table written.
```

Complete flag:

```text
results/env_v2_phase_n4u_precheck_oracle_calibration_v2/PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_COMPLETE.flag
```

---

# 17. Terminal decision labels

Use exactly one:

```text
phase_n4u_precheck_v2_complete_ready_for_n4u
phase_n4u_precheck_v2_complete_oracle_headroom_but_calibration_blocked
phase_n4u_precheck_v2_complete_oracle_headroom_but_gpsi_prediction_blocked
phase_n4u_precheck_v2_complete_no_clean_env_headroom_stress_promising
phase_n4u_precheck_v2_complete_no_clean_env_headroom_stress_not_promising
phase_n4u_precheck_v2_complete_adaptive_margin_failed
phase_n4u_precheck_v2_stopped_premise_invalidated
phase_n4u_precheck_v2_stopped_resource_unsafe
phase_n4u_precheck_v2_stopped_preflight_failed
phase_n4u_precheck_v2_stopped_cpa_ttc_audit_failed
phase_n4u_precheck_v2_stopped_oracle_impl_invalid
phase_n4u_precheck_v2_stopped_gpsi_risk_impl_invalid
phase_n4u_precheck_v2_stopped_ensemble_unavailable
phase_n4u_precheck_v2_stopped_eval_failed
phase_n4u_precheck_v2_stopped_selector_contaminated
phase_n4u_precheck_v2_stopped_analysis_failed
```

Meaning:

```text
ready_for_n4u:
  oracle beats strong_geom_anchor with CI,
  σ² calibrated or validation-calibrated,
  adaptive margin beats fixed margin with CI.

oracle_headroom_but_calibration_blocked:
  oracle headroom exists, but uncertainty is not usable.

oracle_headroom_but_gpsi_prediction_blocked:
  oracle headroom exists, but Gψ point correction is too far from oracle or harmful.

no_clean_env_headroom_stress_promising:
  clean EnvV2 has no headroom, but stress smoke creates headroom.

no_clean_env_headroom_stress_not_promising:
  clean EnvV2 and stress smoke both lack usable headroom.

adaptive_margin_failed:
  Gψ point/fixed may work, but σ² adaptive margin does not beat fixed margin.

premise_invalidated:
  planned CPA/TTC or shield CPA/TTC are already future/oracle-like, invalidating constant-velocity correction framing.
```

---

# 18. Final report requirements

The report must include:

```text
terminal_decision
GitHub sync status and commit
changed files
CPA/TTC audit
premise gate result
strong_geom_anchor manifest
ordinary VO / Phase B / N4-O parameter equivalence
statistical protocol and episode counts
bootstrap CI details
oracle headroom vs strong_geom_anchor
Gψ point correction vs strong_geom_anchor and oracle
heteroscedastic σ² online calibration
ensemble σ² online calibration
hetero vs ensemble reliability diagrams
temperature scaling values
fixed vs adaptive margin results
hetero adaptive vs fixed CI
ensemble adaptive vs fixed CI
adaptive margin failure-mode breakdown
stress smoke trigger and results if applicable
resource utilization and concurrency scaling
watcher heartbeat interval decisions
N4-U readiness decision
N4-O status
recommended next phase
```

Explicitly state:

```text
This phase is not allowed to claim N4-U success without CI-supported oracle headroom and adaptive-margin gains against the strongest deployable geometric baseline.
```
