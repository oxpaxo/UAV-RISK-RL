# Phase N3PF-STAB Guide

> Project: UAV / Dynamic Obstacle Avoidance / DRL / EnvV2 / Gψ-HeadA + PPO  
> Guide target: Codex executing inside the project root of `UAV-RISK-RL`  
> Phase purpose: stabilize P3 block_projected Gψ-PPO before any N4-U uncertainty-aware shield work.

---

## 0. Current terminal state and non-negotiable decision

The current project state is:

```text
Do not continue N4-U now.
Do not ignore seed2.
Enter Phase N3PF-STAB.
```

Reason:

```text
P3 block_projected has useful upside, but its training-seed stability is unresolved.

N3PF-MS:
  seed0: success=0.6167 / collision=0.3833
  seed1: success=0.6089 / collision=0.3911
  seed2: success=0.4222 / collision=0.5778

N3PF-MS-AB:
  seed2_rerunA at 1500k again success=0.4222 / collision=0.5778
  no hard config/path/checkpoint/Gψ feature engineering error found
  most likely cause: PPO seed-sensitive bad local optimum / training instability
```

N4-O ordinary shield is **not discarded**. Its correct status is:

```text
N4-O is a conditional positive result:
  good P3 seed0 + ordinary VO shield is strong.

But N4-O is not currently sufficient to proceed to N4-U because P3 backbone is not yet stable.
```

After P3-STAB succeeds, ordinary shield must be rerun as:

```text
P3-STAB + ordinary shield
attention_full + ordinary shield
no_z + ordinary shield
```

Only after that should N4-U / σ² uncertainty-aware shield resume.

---

## 1. GitHub sync requirement before any code changes

Before changing code, configs, scripts, or generated guide-related files, sync the current project to GitHub.

Repository:

```text
https://github.com/oxpaxo/UAV-RISK-RL
```

Run from the project root:

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

If push fails because authentication is unavailable, write the failure into the phase status log and continue only if local commit state is safe and user policy permits local work. Do not silently skip sync.

---

## 2. Known repo implementation facts

Use these as the current implementation ground truth.

### 2.1 P3 block_projected adapter

```text
File: models/gpsi_ppo_policy.py
Class: GpsiBlockProjectedNoZExtractor
```

P3 adapter structure:

```text
obs(12)    -> Linear -> LayerNorm -> Tanh -> 32
delta(9)  -> Linear -> LayerNorm -> Tanh -> 16
logvar(9) -> Linear -> LayerNorm -> Tanh -> 16
concat -> 64-d per-obstacle embedding
```

Then this obstacle embedding goes into ego-query / obstacle-key masked attention.

P3 config:

```text
configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml

delta_scale = 5.0
logvar_clamp = [-5, 3]
logvar_output_scale = 0.2
feature_adapter = block_projected_no_z
```

Scale / clip are handled in:

```text
envs/wrappers/gpsi_obs_wrapper.py
```

not hard-coded inside the adapter.

### 2.2 attention_full backbone

```text
File: policies/obstacle_set_extractor.py
Class: ObstacleSetExtractor
```

attention_full structure:

```text
obstacle obs: 12-d -> hidden 64
ego state:    10-d -> hidden 64
attention: W_q(ego) against W_k(obs_h), masked softmax
```

P3 does not directly reuse the same class, but it reuses the same conceptual interface:

```text
ego
obs
mask
global_risk
```

and the same kind of masked attention aggregation.

### 2.3 Gψ wrapper schema

PPO sees dict keys:

```text
ego
obs
mask
global_risk
```

The wrapper replaces / augments `obs`; delta_hat and logvar are flattened into each obstacle's `obs` block. They are not separate dict keys.

HeadA feature shape:

```text
num_horizons = 3
state_dim = 3
delta_hat: 3 horizons × xyz = 9 dims
logvar:    3 horizons × xyz = 9 dims
```

Important:

```text
logvar means log σ², not log σ.
```

### 2.4 Gψ checkpoint and normalization

```text
Gψ checkpoint:
  work_dirs/gpsi_heada_v1_nll/best.pth

Normalization stats:
  embedded inside the checkpoint normalization field

Wrapper std fix:
  degenerate_std_threshold = 1e-5
  degenerate_std_floor = 1.0
```

All STAB experiments must keep Gψ frozen unless a later guide explicitly says otherwise.

### 2.5 Existing result paths

Original multi-seed result:

```text
results/env_v2_phase_n3pf_ms_multiseed/
```

Key tables:

```text
tables/phase_n3pf_ms_episode_metrics.csv
tables/phase_n3pf_ms_scenario_breakdown.csv
tables/phase_n3pf_ms_motion_mode_breakdown.csv
tables/phase_n3pf_ms_threat_class_breakdown.csv
tables/phase_n3pf_ms_raw_unsafe_action_steps.csv
tables/phase_n3pf_ms_raw_unsafe_action_summary.csv
```

AB rerun result:

```text
results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/
```

Key rerun tables:

```text
step_b_rerun/tables/phase_n3pf_ms_seed2b_*_scenario_breakdown.csv
step_b_rerun/tables/phase_n3pf_ms_seed2b_*_motion_mode_breakdown.csv
step_b_rerun/tables/phase_n3pf_ms_seed2b_*_threat_class_breakdown.csv
step_b_rerun/tables/phase_n3pf_ms_seed2b_*_raw_unsafe_action_summary.csv
step_b_rerun/tables/phase_n3pf_ms_seed2b_*_behavior_diagnostics.csv
```

### 2.6 Phase B / N4-O ordinary shield

Phase B VO-like filter:

```text
scripts/run_env_v2_phase_b_geometry_filter_baselines.py
function: vo_like_filter
```

N4-O ordinary shield imports and calls the Phase B `vo_like_filter`.

Strong Phase B ordinary VO-like config:

```text
vo_like_filter_h45_cpa1p2_h16:
  horizon = 4.5
  cpa_safe = 1.2
  num_headings = 16
```

N4-O ordinary shield does not use σ²:

```text
ordinary_shield_uses_sigma2 = 0
directional_uncertainty_margin = 0
```

### 2.7 Current N4-U implementation status

There is no confirmed formal N4-U uncertainty-aware shield implementation. Do not start N4-U in this phase.

---

## 3. Phase N3PF-STAB objectives

Primary objective:

```text
Determine whether P3 block_projected seed2 collapse can be removed by lightweight PPO stabilization or by attention-preserving / attention-like gated Gψ fusion.
```

Secondary objectives:

```text
1. Prevent post-hoc checkpoint selection.
2. Prevent overfitting only to seed2.
3. Keep N4-O as paused positive evidence, not discarded evidence.
4. Prepare a clean route for later P3-STAB + ordinary shield and N4-U.
```

Do not optimize only for the highest single checkpoint. The target is stable multi-training-seed behavior.

---

## 4. Strict prohibitions

Do not:

```text
1. Modify EnvV2 core dynamics, reward definitions, termination, scenario definitions, or obstacle generation.
2. Fine-tune Gψ.
3. Start N4-U.
4. Claim S2 is "attention-preserving" unless the implementation actually preserves / loads / distills the trained attention_full capability.
5. Use test seeds for checkpoint selection, structure selection, hyperparameter selection, or deciding whether to continue a branch.
6. Judge lower-LR variants only by 500k performance.
7. Treat seed2 recovery alone as final proof.
8. Delete or overwrite previous result directories.
9. Use 60-second long-training watcher spam.
```

---

## 5. Experiment design overview

Phase N3PF-STAB has four parts:

```text
Part A: S1 lower-LR PPO stabilization
Part B: S2 gated Gψ fusion with strict attention fallback semantics
Part C: S3 validation/test checkpoint selector
Part D: seed2 diagnostics and multi-seed confirmation logic
```

Recommended result root:

```text
results/env_v2_phase_n3pf_stab/
```

Recommended checkpoint root:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab/
```

Recommended report:

```text
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_REPORT.md
```

Complete flag:

```text
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_COMPLETE.flag
```

Stop flags:

```text
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_STOP_HARD_ERROR.flag
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_STOP_VALIDATION_TEST_LEAKAGE.flag
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_STOP_TRAINING_BROKEN.flag
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_STOP_RESOURCE_SATURATION.flag
```

---

## 6. Part A — S1 lower-LR PPO stabilization

### 6.1 Purpose

S1 tests whether seed2 collapse is caused mainly by PPO step size / optimization instability.

Original P3 uses:

```text
learning_rate = 3e-4
```

Create lower-LR variants:

```text
S1-lr2e-4
S1-lr1e-4
```

Use new YAML configs. Do not modify core PPO code just to change LR.

Recommended config names:

```text
configs/env_v2_gpsi_heada_ppo_n3pf_stab_s1_lr2e4.yaml
configs/env_v2_gpsi_heada_ppo_n3pf_stab_s1_lr1e4.yaml
```

Each config should inherit or copy the current P3 block_projected config and change only the required stabilization fields. At minimum:

```text
learning_rate = 2e-4  # for lr2e4 variant
learning_rate = 1e-4  # for lr1e4 variant
method_name / out_dir updated
```

Keep unchanged unless explicitly documented:

```text
feature_adapter = block_projected_no_z
include_z = false
delta_scale = 5.0
logvar_clamp = [-5, 3]
logvar_output_scale = 0.2
Gψ checkpoint = work_dirs/gpsi_heada_v1_nll/best.pth
Gψ frozen = true
n_envs = 4
device = cpu
train_scenario = train_flow_mixed
no_shield = true
```

### 6.2 Lower-LR slow-warm rule

Do not kill lower-LR variants because they are weak at 500k.

Lower LR naturally learns more slowly. Therefore, evaluate the trend across:

```text
500k
750k
1000k
```

If a lower-LR run is still improving at 1000k and is approaching the acceptance gate, allow continuation to:

```text
1500k
```

Record training diagnostics at all checkpoints:

```text
train reward
entropy
approx_kl
clip_fraction
value_loss
policy_loss
explained_variance
action_delta
raw_unsafe_action_rate
progress
```

If logs do not currently expose these fields, add lightweight logging without changing PPO semantics.

### 6.3 S1 screening seed

Initial S1 screening seed:

```text
training_seed = 2
```

Reason: seed2 is the known reproducible failure seed.

But seed2 is only a screening seed. It is not the final proof seed.

### 6.4 S1 checkpoint/eval protocol

Candidate checkpoints:

```text
500k
750k
1000k
optional 1500k if 1000k trend justifies continuation
```

Validation eval seeds:

```text
900
901
```

Validation scenarios:

```text
eval_flow_id
eval_flow_high_density
eval_flow_high_speed
eval_flow_high_threat
eval_flow_mixed_ood
eval_flow_sudden_threat
```

Recommended validation scale for screening:

```text
30 to 50 episodes per scenario per eval seed
```

Do not use final test seeds for S1 branch selection.

---

## 7. Part B — S2 gated Gψ fusion

### 7.1 Correct interpretation

The intended idea is:

```text
base_feature = attention_full-like obs branch
gpsi_feature = delta/logvar branch
fused_feature = base_feature + gate * gpsi_feature
```

with `gate` initialized near zero.

However:

```text
gate ≈ 0 alone does not guarantee a trained attention_full fallback.
```

If the base obs branch is randomly initialized, then with `gate≈0` the policy falls back only to a randomly initialized attention-like obs branch, not to the trained attention_full policy.

Therefore, S2 must be implemented and named carefully.

### 7.2 Allowed S2 variants

Implement one of the following, in priority order.

#### S2-A: strict attention_full warm-start variant

This is the preferred variant if feasible.

Requirements:

```text
1. Base obs branch architecture is exactly isomorphic to attention_full ObstacleSetExtractor where relevant.
2. Load matching parameters from the trained attention_full checkpoint.
3. Initialize Gψ branch so that its contribution is near zero at the start.
4. Preserve or safely map actor/critic layers when dimensions permit.
5. Verify state_dict loading coverage and report all loaded / missing / unexpected keys.
```

Only this variant can be called:

```text
attention_preserving_gated_gpsi
```

Required verification:

```text
At initialization with gate≈0, compare S2-A behavior against attention_full on a fixed batch / fixed eval smoke set.

Report:
  action mean absolute difference
  action max absolute difference
  feature difference if available
  success/collision on smoke eval if feasible
```

If action parity is not close, do not claim strict attention preservation. Rename the method to an attention-like fallback variant.

#### S2-B: isomorphic obs branch load only

If full policy warm-start is not feasible, but the obs/ego/attention branch can load attention_full extractor parameters:

```text
1. Load the base obs/ego/attention extractor weights from attention_full.
2. Gψ branch gate starts near zero.
3. Actor/critic may be newly initialized if dimensions or architecture prevent safe loading.
4. Clearly report that only the feature extractor is warm-started.
```

Name:

```text
attention_extractor_warmstart_gated_gpsi
```

Do not claim this is a complete trained attention_full fallback.

#### S2-C: obs-only distillation / pretraining

If direct loading is not feasible:

```text
1. Train the base obs branch / policy to imitate attention_full actions or features on collected observations.
2. Then enable gated Gψ branch.
3. Report distillation loss, dataset source, and parity metrics.
```

Name:

```text
attention_distilled_gated_gpsi
```

#### S2-D: attention-like fallback only

If none of S2-A/B/C is feasible within this phase:

```text
1. Implement the gated structure.
2. Initialize gate near zero.
3. Explicitly label it as attention-like fallback, not attention-preserving fallback.
4. Do not use it to claim trained attention_full preservation.
```

Name:

```text
attention_like_gated_gpsi
```

This variant is allowed only as a screening ablation.

### 7.3 Required S2 implementation notes

Recommended new class:

```text
models/gpsi_ppo_policy.py
class: GpsiGatedResidualExtractor
```

Avoid editing the existing `GpsiBlockProjectedNoZExtractor` in a way that changes old P3 behavior.

Recommended high-level structure:

```text
base_obs_emb = base_obs_encoder(obs_12)
gpsi_emb = gpsi_encoder(delta_9, logvar_9)

gate = sigmoid(gate_logit) or scalar/tensor gate parameter
fused_obs_emb = base_obs_emb + gate * gpsi_emb
```

Gate initialization:

```text
gate should start near 0, e.g. sigmoid(-5) ≈ 0.0067
```

Record:

```text
gate value
base_obs_emb norm
gpsi_emb norm
gate * gpsi_emb norm
fused_obs_emb norm
```

Do not let gate initialization be the only evidence for fallback. Report warm-start / distillation status.

### 7.4 S2 screening protocol

Initial S2 screening seed:

```text
training_seed = 2
```

Checkpoints:

```text
500k
750k
1000k
optional 1500k if trend justifies continuation
```

Validation eval seeds:

```text
900
901
```

Do not use test seeds for branch selection.

---

## 8. Part C — S3 validation/test checkpoint selector

### 8.1 Why S3 is mandatory

Current `best_by_eval` is not a strict validation selector and may simply copy `final.zip` in some scripts. A strict selector is required to reduce post-hoc checkpoint selection risk.

Create a new selector script, for example:

```text
scripts/select_env_v2_phase_n3pf_stab_checkpoint.py
```

or integrate a clearly separated selection step in the analysis script.

### 8.2 Seed split policy

Use three seed groups:

```text
validation seeds:
  900, 901

test seeds:
  1000, 1001, 1002

final held-out seeds:
  1100, 1101, 1102
```

Rules:

```text
1. validation seeds may be used for checkpoint selection.
2. test seeds must not be used for structure, hyperparameter, or checkpoint selection.
3. if test seeds are used to identify a problem and change the method, those test seeds are no longer clean for final claims.
4. final held-out seeds should be reserved for the last candidate after the design is frozen.
```

### 8.3 Candidate checkpoints

Selector should handle:

```text
500k
750k
1000k
1250k
1500k
final
```

If some checkpoints are absent in screening runs, select among available checkpoints and report missing ones.

### 8.4 Pre-registered selection metric

Use a simple metric. Suggested:

```text
selection_score = success_rate - collision_rate
```

or:

```text
selection_score = success_rate - 2.0 * collision_rate
```

Choose one before running selection and write it to the report. Do not change the metric after seeing test results.

Recommended hard filters:

```text
collision_rate <= 0.45 on validation
progress >= 0.93 on validation
no diagnostic hard error
```

For seed2 recovery screening, suggested pass gate:

```text
validation success >= 0.58
validation collision <= 0.42
```

Do not treat this gate as a final paper claim. It is a screening gate.

---

## 9. Part D — Diagnostics

Generate diagnostics before and after STAB.

### 9.1 Existing seed2 collapse diagnostics

Use existing tables from:

```text
results/env_v2_phase_n3pf_ms_multiseed/
results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/
```

At minimum, produce:

```text
seed0 vs seed2:
  success/collision
  action_delta distribution
  raw_unsafe_action_rate
  raw_min_predicted_cpa
  raw_min_predicted_ttc
  progress
  scenario breakdown
  motion-mode breakdown
  threat-class breakdown

seed2 original vs seed2_rerunA:
  success/collision consistency
  raw_unsafe consistency
  action_delta consistency
  behavior diagnostics consistency
```

Recommended diagnostic output:

```text
results/env_v2_phase_n3pf_stab/tables/phase_n3pf_stab_seed2_collapse_diagnostics.csv
results/env_v2_phase_n3pf_stab/plots/
```

### 9.2 New STAB diagnostics

For S1 and S2 runs, record per-checkpoint:

```text
success_rate
collision_rate
near_miss_rate
progress
mean_min_distance
raw_unsafe_action_rate
action_delta
raw_min_predicted_cpa
raw_min_predicted_ttc
scenario breakdown
motion-mode breakdown
threat-class breakdown
```

For S2 additionally record:

```text
gate value trajectory
base branch norm
gpsi branch norm
gated contribution norm
state_dict warm-start coverage
attention parity smoke metrics
```

---

## 10. Screening and confirmation logic

### 10.1 Stage 1: seed2 screening

Run:

```text
S1-lr2e-4 seed2
S1-lr1e-4 seed2
S2 chosen variant seed2
```

Use validation seeds only for checkpoint selection.

Recommended CPU policy:

```text
Use device=cpu.
Keep n_envs=4 for PPO semantics consistency.
Use process-level parallelism cautiously.
Run at most 2 screening PPO jobs concurrently at first.
If stable and resource usage is safe, up to 3 short screening jobs may be acceptable.
Monitor fps, memory, IO, checkpoint writing, and logs.
```

### 10.2 Stage 2: candidate continuation

If a lower-LR run is slow but improving at 1000k, allow continuation to 1500k before rejecting it.

A seed2 screening candidate can continue if:

```text
validation trend is improving through 1000k
AND collision is moving toward <= 0.42
AND progress is not collapsing
AND diagnostics show no hard engineering error
```

### 10.3 Stage 3: multi-training-seed confirmation

Any candidate passing seed2 screening must be confirmed with:

```text
training seeds: 0, 1, 2
```

If resources allow, also run:

```text
training seed: 3
```

Seed2 recovery alone is not sufficient. Avoid overfitting the method to the known failure seed.

### 10.4 Stage 4: attention_full multi-seed requirement

For final paper-level claims, attention_full must also be evaluated under a matching training-seed protocol.

Minimum final comparison target:

```text
P3-STAB seed0/1/2
attention_full seed0/1/2
same eval protocol
same validation/test/final-heldout discipline
```

If attention_full seed1/2 are not available under the same formal 1.5M protocol, prepare a later phase to train them.

Until then, write claims conservatively:

```text
P3-STAB multi-seed mean exceeds the available attention_full seed0 reference.
```

Do not claim:

```text
P3-STAB is definitively multi-seed superior to attention_full.
```

unless attention_full multi-seed has been run.

---

## 11. Reward/cost policy for this phase

Do not add reward/cost changes in the first STAB pass.

Reason:

```text
Changing structure, LR, checkpoint selector, reward, and shield simultaneously would make attribution impossible.
```

However, do not permanently rule out reward/cost fixes.

If S1/S2 fail to resolve seed2 collapse, the next stabilization phase may revisit:

```text
dense CPA/TTC cost
risk-window response cost
near-miss cost shaping
PPO-Lagrangian or cost-penalty fallback
```

For this phase, only collect diagnostics that help decide whether reward-safety mismatch is likely.

---

## 12. Watcher requirement

Create a blocking watcher:

```text
scripts/watch_phase_n3pf_stab.sh
```

The watcher must:

```text
1. Run from project root.
2. Execute audit / config creation checks / training / eval / analysis in order.
3. Monitor each stage.
4. Write status to:
   results/env_v2_phase_n3pf_stab/phase_n3pf_stab_status.txt
5. Write watcher log to:
   results/env_v2_phase_n3pf_stab/phase_n3pf_stab_watcher.log
6. Exit only when a complete flag or a defined stop flag / stop condition exists.
7. Continue running otherwise.
```

Long-training heartbeat:

```text
Use about 5 minutes between heartbeat updates for long training.
Do not spam every 60 seconds.
Short smoke tests may use shorter intervals.
```

Terminal complete flag:

```text
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_COMPLETE.flag
```

The final watcher output must include:

```text
terminal_decision = phase_n3pf_stab_complete
```

or a stop decision such as:

```text
terminal_decision = phase_n3pf_stab_stop_hard_error
terminal_decision = phase_n3pf_stab_stop_validation_test_leakage
terminal_decision = phase_n3pf_stab_stop_training_broken
terminal_decision = phase_n3pf_stab_stop_resource_saturation
```

---

## 13. Required final report

Create:

```text
results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_REPORT.md
```

The report must include:

```text
1. Terminal decision.
2. GitHub sync status.
3. Repo implementation verification summary.
4. S1 configs and exact changed fields.
5. S2 implementation type:
   S2-A / S2-B / S2-C / S2-D
6. Explicit statement:
   whether S2 is strict attention-preserving, feature-warmstarted, distilled, or only attention-like.
7. Validation/test/final-heldout seed discipline.
8. S1 seed2 screening results across 500k / 750k / 1000k / optional 1500k.
9. S2 seed2 screening results across checkpoints.
10. Selector result and selected checkpoint per candidate.
11. Seed2 collapse diagnostics.
12. Multi-seed confirmation status:
    not run / seed0-1-2 run / seed0-1-2-3 run.
13. Comparison against:
    available attention_full reference
    no_z reference
    Phase B VO-like / CPA-TTC APF context
14. Decision:
    P3-STAB candidate found or not.
15. Whether N4-O can be rerun next.
16. Whether N4-U remains blocked.
```

Use conservative language.

Allowed conclusions:

```text
P3-STAB seed2 screening passed; proceed to multi-seed confirmation.
P3-STAB multi-seed candidate is stable enough to rerun N4-O.
P3-STAB failed to remove seed2 collapse; keep N4-U blocked and consider reward/cost stabilization next.
```

Do not write:

```text
P3-STAB proves the method is paper-ready.
P3-STAB definitively beats attention_full.
```

unless the required multi-seed evidence exists.

---

## 14. Suggested implementation steps

### Step 0 — audit and sync

```bash
git status --short
# perform sync commands from Section 1
```

Then verify existence of:

```text
models/gpsi_ppo_policy.py
policies/obstacle_set_extractor.py
envs/wrappers/gpsi_obs_wrapper.py
configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml
work_dirs/gpsi_heada_v1_nll/best.pth
results/env_v2_phase_n3pf_ms_multiseed/
results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/
```

### Step 1 — add configs

Add S1 configs:

```text
configs/env_v2_gpsi_heada_ppo_n3pf_stab_s1_lr2e4.yaml
configs/env_v2_gpsi_heada_ppo_n3pf_stab_s1_lr1e4.yaml
```

Add S2 config(s) according to the implemented variant:

```text
configs/env_v2_gpsi_heada_ppo_n3pf_stab_s2_<variant>.yaml
```

### Step 2 — implement S2 if feasible

Before coding S2, inspect attention_full checkpoint state_dict and current extractor keys.

If strict warm-start is feasible, implement S2-A or S2-B.

If not feasible, implement S2-D and clearly label it as attention-like only.

### Step 3 — add selector and diagnostics

Add:

```text
scripts/select_env_v2_phase_n3pf_stab_checkpoint.py
scripts/analyze_env_v2_phase_n3pf_stab_results.py
```

or equivalent.

Do not use test seeds inside the selector.

### Step 4 — run watcher

```bash
bash scripts/watch_phase_n3pf_stab.sh
```

The watcher must block until complete or stop.

---

## 15. Final decision policy

At phase end:

### If no candidate rescues seed2

```text
Keep N4-U blocked.
Report that lower-LR / gated fusion did not remove seed2 instability.
Recommend next phase: reward/cost stabilization or deeper PPO local-optimum audit.
```

### If seed2 is rescued but multi-seed confirmation is not done

```text
Do not resume N4-U.
Proceed to multi-training-seed confirmation.
```

### If seed0/1/2 are stable

```text
Allow next phase:
  rerun N4-O with P3-STAB + ordinary shield
  prepare attention_full multi-seed if not yet formalized
  only then consider N4-U
```

### If seed0/1/2/optional seed3 are stable and N4-O rerun is strong

```text
N4-U may be unblocked in a later guide.
```

