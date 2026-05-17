# Phase N3PF-DECOUPLE Guide

## 0. Phase purpose

This phase answers one specific question:

```text
Is the current Gψ-PPO failure mainly caused by Gψ features being unhelpful,
or by combining Gψ features with the attention-style obstacle aggregator?
```

The phase must not try to solve the whole project in one step. It is a **factorized decoupling experiment**.

Current evidence:

```text
Gψ HeadA offline prediction is still valid evidence:
  delta/logvar are learnable from history/current obstacle information.

attention_full is still a strong learning baseline:
  current obstacle profile + masked attention + PPO works reasonably well.

But:
  P3 / S2-D showed that Gψ features + current attention-compatible PPO
  are not stable enough across training seeds.
```

Therefore, the next experiment must test whether Gψ features become useful when the attention aggregator is removed.

---

## 1. Hard decisions from previous phases

Do not reinterpret the previous result as a temporary bad checkpoint.

Current status:

```text
S2-D attention-like gated Gψ:
  seed2 screening positive;
  multi-training-seed confirmation failed.

Mandatory seeds 0/1/2 test aggregate:
  mean success = 0.5122
  mean collision = 0.4878

Reference:
  attention_full seed0 = 0.6033 / 0.3967
  no_z = 0.5667 / 0.4333
```

Implications:

```text
Do not continue S2-D as N4 backbone.
Do not rerun N4-O on S2-D.
Do not enter N4-U.
Do not keep tuning S2-D gate/LR/checkpoint selection.
```

This phase is not S2-D repair. It is Gψ/attention decoupling.

---

## 2. Required GitHub sync before code changes

Before changing any code or config, sync the current repository to GitHub:

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

If push fails because credentials are unavailable, record the failure clearly in the report and continue only after local commit is created. Do not invent a successful remote sync.

---

## 3. Output locations

Create this guide in the repository:

```text
codex_guide/PHASE_N3PF_DECOUPLE_GUIDE.md
```

Use result root:

```text
results/env_v2_phase_n3pf_decouple/
```

Required artifacts:

```text
results/env_v2_phase_n3pf_decouple/PHASE_N3PF_DECOUPLE_REPORT.md
results/env_v2_phase_n3pf_decouple/PHASE_N3PF_DECOUPLE_COMPLETE.flag
results/env_v2_phase_n3pf_decouple/phase_n3pf_decouple_watcher.log
results/env_v2_phase_n3pf_decouple/phase_n3pf_decouple_status.txt
results/env_v2_phase_n3pf_decouple/tables/
results/env_v2_phase_n3pf_decouple/plots/
results/env_v2_phase_n3pf_decouple/logs/
```

---

## 4. What this phase must compare

Use a clean 2 × 2 decomposition:

```text
Aggregator axis:
  attention aggregator
  non-attention aggregator

Feature axis:
  obs-only
  obs + Gψ HeadA delta/logvar
```

The minimum new non-attention matrix:

```text
NK-obs-only
NK-Gψ
DeepSets-obs-only
DeepSets-Gψ
```

For context, import previous references into analysis tables:

```text
attention_full reference
no_z reference
P3 block_projected reference
S2-D failed confirmation reference
Phase B VO-like / CPA-TTC APF context
```

Do not retrain attention_full inside this phase unless explicitly needed for a preflight sanity check. Formal attention_full multiseed remains a separate infrastructure task.

---

# 5. Non-attention aggregator design

This section is the core design. Implement it carefully.

The goal is to remove the attention softmax / ego-query-key obstacle aggregation while keeping the problem still learnable.

The two required non-attention aggregators are:

```text
A. Nearest/Risk-K ordered MLP aggregator
B. DeepSets pooling aggregator
```

These are deliberately simple. Their purpose is not to be the final architecture; their purpose is to test whether Gψ features help when attention is removed.

---

## 5.1 Common observation schema

Existing Gψ wrapper behavior:

```text
PPO dict keys remain:
  ego
  obs
  mask
  global_risk

For Gψ variants:
  obs last dimension = 30
  obs_i = [base_obs_12, delta_hat_scaled_9, logvar_scaled_9]

For obs-only variants:
  use only base_obs_12
```

Do not rely on a separate `delta` or `logvar` dict key. The Gψ features are already flattened into the per-obstacle `obs` block.

Recommended internal split:

```python
base_obs = obs[..., :12]
delta_hat = obs[..., 12:21]
logvar = obs[..., 21:30]
```

Validate all dimensions at runtime. If dimensions differ, stop and write `STOP_SCHEMA_MISMATCH.flag`.

---

## 5.2 Aggregator A: NK ordered MLP

### Purpose

This aggregator removes attention entirely and gives PPO a deterministic, fixed-order set of the most relevant obstacles.

It answers:

```text
Does Gψ help a simple ordered top-K obstacle policy?
```

### Selection rule

For each timestep, select top-K active obstacles from `obs` using a deterministic score.

Use a configurable `rank_key`:

```text
risk_ttc_distance
risk
ttc
distance
```

Default:

```text
rank_key = risk_ttc_distance
K = 6
```

Recommended score:

```text
score_i =
  + w_risk * normalized_risk_i
  + w_close * normalized_inverse_distance_i
  + w_ttc * normalized_inverse_ttc_i
```

Where:

```text
active mask must be applied;
inactive obstacles must receive -inf score;
NaN/inf values must be sanitized and reported;
TTC invalid/large values must not dominate.
```

Suggested weights:

```text
w_risk = 1.0
w_close = 1.0
w_ttc = 1.0
```

Do not overfit these weights in this phase. They are only used to define a deterministic top-K order.

If obstacle obs field indexes for distance/risk/TTC are already centralized in existing code/config, use that source. If not, infer indexes from current EnvV2 observation schema and report them explicitly.

### Padding

If fewer than K active obstacles exist:

```text
pad obstacle feature vectors with zeros
pad mask indicator with 0
```

Append a binary active indicator per selected slot:

```text
selected_feature_i = [feature_i, active_i]
```

This avoids ambiguity between real zero features and padded slots.

### Feature variants

NK-obs-only:

```text
per_obstacle_feature = base_obs_12
selected flattened input = K × (12 + 1)
```

NK-Gψ:

```text
per_obstacle_feature = [base_obs_12, delta_hat_9, logvar_9]
selected flattened input = K × (30 + 1)
```

### Policy extractor

Build a feature extractor such as:

```text
GpsiNearestKExtractor
```

Architecture:

```text
ego encoder:
  ego_dim -> Linear(64) -> LayerNorm -> Tanh

selected obstacle MLP:
  flatten top-K selected features
  Linear(256) -> LayerNorm -> Tanh
  Linear(128) -> LayerNorm -> Tanh

global risk branch:
  global_risk_dim -> Linear(16) -> Tanh

concat:
  ego_emb + obstacle_emb + global_risk_emb

final:
  Linear(128) -> LayerNorm -> Tanh
```

Keep it reasonably comparable in parameter scale to prior feature extractors. Report parameter count.

### Mandatory diagnostics

For NK variants report:

```text
topK_rank_key
K
selected slot active rate
mean selected distance/risk/TTC if available
selected obstacle overlap with attention top weights if easy to compute, optional
feature nonfinite count
flattened feature L2 mean/p95
```

---

## 5.3 Aggregator B: DeepSets pooling

### Purpose

This aggregator removes attention but keeps permutation invariance.

It answers:

```text
Does Gψ help when obstacle features are encoded independently and pooled,
without softmax competition?
```

### Design

For each active obstacle:

```text
h_i = phi(per_obstacle_feature_i)
```

Then pool:

```text
mean_pool over active obstacles
max_pool over active obstacles
```

Concatenate both:

```text
pooled = [mean_pool, max_pool]
```

Do not use learned attention weights.

### Feature variants

DeepSets-obs-only:

```text
per_obstacle_feature = base_obs_12
```

DeepSets-Gψ:

```text
per_obstacle_feature = [base_obs_12, delta_hat_9, logvar_9]
```

### Suggested extractor

Build:

```text
GpsiDeepSetsExtractor
```

Architecture:

```text
phi:
  input_dim -> Linear(64) -> LayerNorm -> Tanh
  Linear(64) -> LayerNorm -> Tanh

pool:
  masked mean over active obstacles
  masked max over active obstacles

ego branch:
  ego_dim -> Linear(64) -> LayerNorm -> Tanh

global risk branch:
  global_risk_dim -> Linear(16) -> Tanh

rho:
  concat(ego_emb, mean_pool, max_pool, global_risk_emb)
  Linear(128) -> LayerNorm -> Tanh
  Linear(128) -> LayerNorm -> Tanh
```

### Masking rules

For mean pool:

```text
sum active h_i / max(active_count, 1)
```

For max pool:

```text
inactive entries should be replaced by a large negative value before max;
if active_count == 0, max_pool should be zeros.
```

### Mandatory diagnostics

For DeepSets variants report:

```text
active obstacle count distribution
mean_pool L2 mean/p95
max_pool L2 mean/p95
phi output L2 mean/p95
feature nonfinite count
obs/delta/logvar block L2 stats for Gψ variants
```

---

## 5.4 What not to implement

Do not add:

```text
new attention layer
transformer
cross-attention
recurrent policy
learned risk map
action-conditioned Gψ risk
shield
reward rewrite
Gψ fine-tuning
```

This phase must isolate aggregator choice.

---

# 6. Experiment design

## 6.1 Mandatory variants

Train these four variants:

```text
decouple_nk_obs
decouple_nk_gpsi
decouple_deepsets_obs
decouple_deepsets_gpsi
```

Use the same PPO settings unless a script requires minimal config naming changes.

Default PPO settings should match previous stable PPO jobs as much as possible:

```text
device = cpu
n_envs = 4
n_steps = 1024
batch_size = 256
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
```

Do not change reward, EnvV2 core, Gψ checkpoint, or Gψ normalization.

## 6.2 Training seeds

Use multi-seed from the start:

```text
mandatory seeds: 0, 1, 2
optional sanity seed: 3
```

Unlike the earlier seed2-only screen, this phase must not optimize around seed2.

## 6.3 Training budget

Use staged training:

```text
Stage A screening:
  train each variant/seed to 750k
  save 250k / 500k / 750k

Stage B continuation:
  continue only variants with evidence of Gψ benefit or strong obs-only baseline to 1.5M
  save 1000k / 1250k / 1500k / final
```

A variant should continue to Stage B if any of the following is true:

```text
Gψ variant beats its obs-only counterpart by >= +0.03 success and <= -0.03 collision on validation mean;
Gψ variant has clearly better high_speed/high_threat/high_density breakdown;
obs-only variant itself is competitive with no_z reference;
the trend is still improving at 750k and not obviously collapsed.
```

If all variants are poor by 750k, continue only the best Gψ and best obs-only variant to 1.0M for confirmation, then stop.

---

# 7. Validation/test/final-heldout discipline

Use:

```text
validation seeds: 900, 901
test seeds: 1000, 1001, 1002
final-heldout seeds: 1100, 1101, 1102
```

Rules:

```text
validation seeds may be used for checkpoint selection and Stage A/B continuation decisions.
test seeds must not be used for structure, hyperparameter, or checkpoint selection.
final-heldout seeds are reserved; only use them if a candidate passes test gate and the report explicitly freezes the candidate first.
```

If test results are used to modify structure/config/selection, mark:

```text
STOP_SELECTOR_CONTAMINATED.flag
```

---

# 8. Checkpoint selector

For every trained job, evaluate validation checkpoints:

Stage A:

```text
250k / 500k / 750k
```

Stage B:

```text
1000k / 1250k / 1500k / final
```

Selection score:

```text
selection_score = success_rate - 2 * collision_rate
```

Tie-breakers:

```text
1. lower collision
2. higher success
3. higher progress
4. lower raw_unsafe_action_rate
5. earlier checkpoint if nearly equal
```

Important:

```text
final.zip is only a candidate.
Do not automatically use final.zip.
Record checkpoint-vs-final parameter drift for selected candidates.
```

---

# 9. Evaluation scale

For validation:

```text
6 scenarios
50 episodes per scenario per eval seed if runtime allows
minimum 30 if resource pressure is high, but record it
```

For test:

```text
6 scenarios
50 episodes per scenario per eval seed
3 eval seeds
900 episodes per selected variant/seed if full scale
```

Use parallel evaluation aggressively, subject to health checks.

---

# 10. Required analysis questions

The report must answer these questions directly.

## Q1. Does Gψ help without attention?

Compare:

```text
NK-Gψ vs NK-obs-only
DeepSets-Gψ vs DeepSets-obs-only
```

Use:

```text
mean across training seeds
min seed
scenario breakdown
motion-mode breakdown
threat-class breakdown
raw unsafe / CPA / TTC / action_delta
```

## Q2. Is attention the likely incompatibility point?

Evidence for "attention may be the incompatibility point" requires:

```text
Gψ improves at least one non-attention aggregator consistently;
Gψ gain is not seed2-only;
Gψ gain appears in high_speed/high_density/high_threat or linear/accel modes;
feature diagnostics are clean.
```

Evidence against:

```text
Gψ fails to improve both NK and DeepSets;
Gψ variants are <= obs-only variants across seeds;
Gψ gain is seed-specific or checkpoint-specific only.
```

## Q3. Are non-attention aggregators too weak?

If all non-attention variants are far below attention_full/no_z, report:

```text
non-attention backbones may be too weak for final policy,
but Gψ benefit can still be assessed relative to their obs-only controls.
```

Do not claim "Gψ is useless" if the backbone itself is too weak. Claim only what the paired controls support.

## Q4. Should Gψ stay as PPO input?

Possible decisions:

```text
A. keep Gψ-PPO route:
   Gψ improves non-attention aggregator robustly and candidate is competitive.

B. use Gψ only with redesigned non-attention route:
   Gψ improves controls but absolute policy is not yet strong.

C. downgrade Gψ-as-PPO-input:
   Gψ does not improve obs-only controls or remains unstable.

D. pivot to Gψ uncertainty-aware shield:
   Gψ as policy input remains weak, but offline prediction and uncertainty remain useful for shield.
```

---

# 11. Full-pipeline parallelization and resource policy

The user has confirmed that the previous phase used less than 30% CPU and that expanding parallelism by 2–3× is acceptable if health remains good.

This phase must explicitly maximize full-pipeline efficiency while preserving safety.

## 11.1 Single-job PPO semantics

For each PPO job, preserve rollout semantics unless a specific change is justified:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

Recommended per PPO job:

```text
n_envs = 4
device = cpu
```

Do not silently increase per-job `n_envs` to accelerate training. Prefer process-level parallelism.

## 11.2 Training parallelism

Previous phase used 3–4 concurrent PPO jobs and underused CPU. This phase should use higher process-level parallelism.

Initial concurrency target:

```text
6 concurrent PPO jobs
```

If health is good, raise to:

```text
8 concurrent PPO jobs
```

If still healthy and memory/disk/fps remain stable, allow:

```text
up to 10 concurrent PPO jobs
```

Do not exceed this without writing a resource note in the report.

Priority scheduling:

```text
Priority 1:
  mandatory seeds 0/1/2 for all four Stage A variants

Priority 2:
  optional seed3 for all four Stage A variants

Priority 3:
  Stage B continuations selected by validation

Priority 4:
  optional extra diagnostics
```

If resources become unsafe, reduce optional jobs first:

```text
pause/queue seed3
pause/queue lower-priority Stage B jobs
keep mandatory seed0/1/2 jobs alive if possible
```

## 11.3 Evaluation parallelism

Evaluation was slow in prior phases. This phase must parallelize eval as well.

Required:

```text
Run validation eval jobs in parallel by variant/checkpoint/training_seed/eval_seed where safe.
Run test eval jobs in parallel by selected variant/training_seed/eval_seed.
Use process-level parallel eval workers.
```

Suggested evaluation concurrency:

```text
start with 8 eval workers
increase to 12 if CPU/RAM/IO healthy
cap at 16 unless report justifies more
```

Each eval worker should avoid internal thread oversubscription:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=""
```

## 11.4 Analysis parallelism

For CSV aggregation and plotting:

```text
Use multiprocessing or chunked processing for large raw unsafe/action-step CSVs.
Avoid loading huge CSVs repeatedly.
Prefer aggregated summaries when possible.
Write README files when raw per-step files are intentionally compressed or omitted.
```

## 11.5 Health checks

Watcher must monitor:

```text
CPU utilization
load average
RAM usage
disk free
per-job fps
log update time
checkpoint file creation time
non-empty checkpoint files
number of alive jobs
failed jobs
zombie processes if detectable
```

Health thresholds:

```text
RAM free < 8 GB: stop launching new jobs, queue optional jobs.
Disk free < 10 GB: stop and write STOP_RESOURCE_UNSAFE.flag.
Any mandatory job exits nonzero: write STOP_TRAINING_FAILED.flag unless retried by guide rule.
No log/checkpoint update for > 30 minutes while process alive: mark stale and inspect.
Mean fps collapse > 50% for > 20 minutes: reduce concurrency.
```

Long-run heartbeat:

```text
approximately every 5 minutes
```

Do not print status every 60 seconds during long training unless debugging a failure.

---

# 12. Required scripts / code changes

Prefer adding new code without breaking existing extractors.

Suggested files:

```text
models/gpsi_ppo_policy.py
  add GpsiNearestKExtractor
  add GpsiDeepSetsExtractor

configs/env_v2_gpsi_heada_ppo_n3pf_decouple_nk_obs.yaml
configs/env_v2_gpsi_heada_ppo_n3pf_decouple_nk_gpsi.yaml
configs/env_v2_gpsi_heada_ppo_n3pf_decouple_deepsets_obs.yaml
configs/env_v2_gpsi_heada_ppo_n3pf_decouple_deepsets_gpsi.yaml

scripts/train_env_v2_gpsi_ppo_n3pf_decouple.py
scripts/eval_env_v2_gpsi_ppo_n3pf_decouple.py
scripts/select_env_v2_phase_n3pf_decouple_checkpoints.py
scripts/analyze_env_v2_phase_n3pf_decouple.py
scripts/watch_phase_n3pf_decouple.sh
```

If existing train/eval scripts can support these variants with minimal extension, reuse them. But do not create fragile one-off hacks.

---

# 13. Preflight checks

Run:

```bash
python -m py_compile models/gpsi_ppo_policy.py
python -m py_compile scripts/train_env_v2_gpsi_ppo_n3pf_decouple.py
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3pf_decouple.py
python -m py_compile scripts/select_env_v2_phase_n3pf_decouple_checkpoints.py
python -m py_compile scripts/analyze_env_v2_phase_n3pf_decouple.py
bash -n scripts/watch_phase_n3pf_decouple.sh
```

Also verify:

```text
Gψ checkpoint exists.
Wrapper schema gives expected obs dim.
Non-attention extractors produce finite outputs.
Masking works with all inactive / partial active obstacle sets.
Config variants differ only in intended fields.
```

---

# 14. Stop conditions

Create these stop flags as needed:

```text
STOP_RESOURCE_UNSAFE.flag
STOP_PREFLIGHT_FAILED.flag
STOP_SCHEMA_MISMATCH.flag
STOP_TRAINING_FAILED.flag
STOP_EVAL_FAILED.flag
STOP_SELECTOR_CONTAMINATED.flag
STOP_ANALYSIS_FAILED.flag
STOP_NO_VALID_VARIANT.flag
```

Stop immediately for:

```text
schema mismatch that invalidates obs/Gψ splits;
selector contamination;
resource unsafe state;
repeated mandatory job failure;
nonfinite features that cannot be repaired without changing method.
```

---

# 15. Complete condition

Write complete flag only when:

```text
Stage A mandatory variants/seeds completed or valid stop condition reached;
validation selector completed;
at least the required selected variants were test-evaluated if any variant passed Stage A continuation;
analysis report written;
decision table written;
watcher log complete.
```

Complete flag:

```text
results/env_v2_phase_n3pf_decouple/PHASE_N3PF_DECOUPLE_COMPLETE.flag
```

---

# 16. Final decision labels

Use exactly one terminal decision:

```text
phase_n3pf_decouple_complete_gpsi_nonattention_positive
phase_n3pf_decouple_complete_gpsi_nonattention_promising_not_competitive
phase_n3pf_decouple_complete_gpsi_nonattention_failed
phase_n3pf_decouple_complete_backbones_too_weak
phase_n3pf_decouple_stopped_resource_unsafe
phase_n3pf_decouple_stopped_preflight_failed
phase_n3pf_decouple_stopped_schema_mismatch
phase_n3pf_decouple_stopped_training_failed
phase_n3pf_decouple_stopped_eval_failed
phase_n3pf_decouple_stopped_selector_contaminated
```

Meaning:

```text
positive:
  Gψ improves non-attention controls and absolute performance is competitive.

promising_not_competitive:
  Gψ improves paired controls but non-attention backbone is still below attention/no_z.

failed:
  Gψ does not improve paired controls or remains unstable.

backbones_too_weak:
  both obs-only and Gψ non-attention backbones are too poor to interpret strongly.
```

---

# 17. Final report requirements

The report must include:

```text
terminal_decision
GitHub sync status and commit
exact changed files
non-attention aggregator implementation details
top-K ranking formula and feature indexes
DeepSets pooling implementation
parallel training/eval strategy and resource usage
all trained variants/seeds/checkpoints
validation selector discipline
validation aggregate
test aggregate
scenario/motion/threat breakdown
raw unsafe / CPA / TTC / action_delta diagnostics
feature/Gψ diagnostics
PPO diagnostics
parameter drift selected-vs-final
paired Gψ-vs-obs-only comparisons
decision: attention incompatibility likely or not
decision: whether to keep Gψ as PPO input
recommendation for next phase
N4-O status
N4-U status
```

Explicitly state:

```text
This phase does not prove Gψ is useless unless Gψ fails relative to matched obs-only controls.
This phase does not prove attention is bad unless Gψ improves without attention and the evidence is multi-seed stable.
```

---

# 18. Expected interpretation

The intended interpretation is:

```text
If NK-Gψ / DeepSets-Gψ > matched obs-only controls:
  Gψ may be useful, and attention aggregation may be the mismatch.

If Gψ variants do not beat matched obs-only controls:
  Gψ as PPO input is not currently justified.

If non-attention variants are all weak:
  do not use them as final policy;
  use them only as diagnostic evidence.

If Gψ-as-PPO-input remains weak:
  pivot to stable backbone + Gψ uncertainty-aware shield.
```
