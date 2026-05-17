# Phase N3PF-STAB-CONFIRM Execution Guide

> Project: UAV-RISK-RL  
> Phase: `N3PF-STAB-CONFIRM`  
> Goal: confirm whether the S2-D gated Gψ fusion candidate from Phase N3PF-STAB is a stable multi-training-seed policy backbone.  
> Status before this phase: S2-D passed seed2 screening at `checkpoint_1000k`, but this is **not** yet a stable method-level conclusion. N4-U remains blocked.

---

## 0. Phase decision context

The previous Phase N3PF-STAB completed with:

```text
terminal_decision = phase_n3pf_stab_complete
selected_screening_variant = stab_s2d_gated
selected_screening_checkpoint = checkpoint_1000k
seed2 validation result at checkpoint_1000k:
  success = 0.5933
  collision = 0.4067
```

Key interpretation:

```text
S2-D gated is not rejected by the final.zip drop.
The final.zip drop is most likely terminal PPO drift / post-target update instability, not a decisive failure of S2-D.
```

Why:

```text
checkpoint_1000k and final.zip are different parameter states;
final.zip has additional post-1000k rollout/update steps;
feature diagnostics are clean;
validation seeds show the drop is systematic, not eval-seed noise;
the drop is consistent with PPO checkpoint sensitivity / last-update policy drift.
```

However, the correct conclusion is still conservative:

```text
S2-D is only a screening candidate.
It must pass multi-training-seed confirmation before it can become a stable P3-STAB candidate.
N4-U remains blocked.
N4-O is not discarded, but must be rerun only after a stable candidate is confirmed.
```

---

## 1. Hard rules

### 1.1 Do not proceed to N4-U

This phase must not implement or evaluate σ² uncertainty-aware shield / N4-U.

### 1.2 Do not use final.zip as the default candidate

`final.zip` is only one checkpoint candidate. It must not automatically replace validation-selected checkpoints.

Every training run must save and evaluate checkpoint candidates:

```text
500k / 750k / 1000k / 1250k / 1500k / final
```

The selected checkpoint must be chosen by validation seeds only.

### 1.3 Do not tune on test seeds

Use seed sets strictly:

```text
validation eval seeds: 900, 901
  - used only for checkpoint selection

test eval seeds: 1000, 1001, 1002
  - used only after checkpoint selection is frozen
  - must not be used for structure / hyperparameter / checkpoint selection

final held-out eval seeds: 1100, 1101, 1102
  - optional final sanity after the candidate passes test
  - must not be used for tuning
```

If any test result is used to modify structure, hyperparameters, checkpoint rule, or evaluation scope, explicitly mark the test set as contaminated in the report.

### 1.4 S2-D semantics

Current S2-D is:

```text
attention_like_gated_gpsi
```

It is **not** strict attention-preserving fusion.

Do not claim strict attention preservation unless one of the following is implemented and verified:

```text
A. warm-start from a trained attention_full checkpoint;
B. architecture-isomorphic base branch with loaded attention_full parameters;
C. obs-only / attention_full-like distillation or pretraining.
```

This phase should keep the S2-D candidate fixed as the screening-selected architecture. Do not add S2-B/S2-C unless explicitly required by a stop decision.

---

## 2. Mandatory GitHub pre-sync before any code or config changes

Before changing code, configs, scripts, or generated guides, synchronize the current project state to GitHub:

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

Remote repository:

```text
https://github.com/oxpaxo/UAV-RISK-RL
```

Record sync status, commit hash, and any push error in the final report.

---

## 3. Phase outputs

Create result root:

```text
results/env_v2_phase_n3pf_stab_confirm/
```

Required final artifacts:

```text
results/env_v2_phase_n3pf_stab_confirm/PHASE_N3PF_STAB_CONFIRM_REPORT.md
results/env_v2_phase_n3pf_stab_confirm/PHASE_N3PF_STAB_CONFIRM_COMPLETE.flag
results/env_v2_phase_n3pf_stab_confirm/phase_n3pf_stab_confirm_status.txt
results/env_v2_phase_n3pf_stab_confirm/phase_n3pf_stab_confirm_watcher.log
```

Required subdirectories:

```text
results/env_v2_phase_n3pf_stab_confirm/tables/
results/env_v2_phase_n3pf_stab_confirm/plots/
results/env_v2_phase_n3pf_stab_confirm/logs/
```

Recommended checkpoint roots:

```text
checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s0/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s1/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s2/
checkpoints/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_s3/   # if resource-stable optional sanity seed
```

Attention reference roots, if formal attention_full multiseed is run in this phase:

```text
checkpoints/env_v2_attention_full_formal_s1/
checkpoints/env_v2_attention_full_formal_s2/
```

---

## 4. Repo and implementation verification

Before training, verify and record:

```text
models/gpsi_ppo_policy.py
  - GpsiBlockProjectedNoZExtractor exists
  - GpsiGatedResidualExtractor exists

policies/obstacle_set_extractor.py
  - ObstacleSetExtractor exists

envs/wrappers/gpsi_obs_wrapper.py
  - Gψ wrapper exists
  - delta/logvar scale and clamp behavior is available

configs/env_v2_gpsi_heada_ppo_n3pf_stab_s2d_gated.yaml
  - feature_adapter = gated_residual_no_z or equivalent S2-D setting
  - no z_i
  - Gψ checkpoint = work_dirs/gpsi_heada_v1_nll/best.pth
  - Gψ frozen = true

work_dirs/gpsi_heada_v1_nll/best.pth exists
```

If the exact S2-D config name differs from the above, use the actual Phase N3PF-STAB S2-D config and record the path.

---

## 5. Required experiment design

### 5.1 Primary: S2-D multi-training-seed confirmation

Run S2-D gated training for:

```text
mandatory training seeds: 0, 1, 2
optional sanity seed: 3, if resources remain healthy
```

Training target:

```text
total training target: 1.5M timesteps per seed
checkpoints: 500k, 750k, 1000k, 1250k, 1500k, final
```

Important:

```text
Do not rely on final.zip.
Do not stop just because final.zip is bad.
Do not select checkpoint from test seeds.
```

For each seed, validation selector chooses the checkpoint using only validation seeds 900/901.

### 5.2 Secondary: attention_full formal multiseed reference

Because a decisive claim against attention_full requires a fair multi-training-seed reference, audit existing attention_full seed1/seed2 checkpoints first.

Step A: audit existing attention checkpoints:

```text
checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip
checkpoints/attention_seed1/...
checkpoints/attention_full_s1.zip
checkpoints/attention_full_s2.zip
```

Decide whether seed1/seed2 are valid formal 1.5M references under the same protocol. Record the decision.

If not valid and resources are healthy, run formal attention_full seed1/seed2 to 1.5M using the same evaluation protocol.

Priority rule:

```text
S2-D seed0/1/2 confirmation is mandatory and higher priority.
attention_full seed1/2 formal runs are strongly recommended if resources allow.
If resource pressure occurs, pause/queue attention_full before interrupting S2-D mandatory seeds.
```

---

## 6. Parallel / multi-process training requirement

The user explicitly wants training accelerated. Use process-level parallelism while keeping each PPO job's rollout semantics unchanged.

### 6.1 Do not change single-job PPO semantics

Unless a pre-existing config already does so, keep per-job settings stable:

```text
n_envs = 4
device = cpu
OMP_NUM_THREADS = 1
MKL_NUM_THREADS = 1
OPENBLAS_NUM_THREADS = 1
NUMEXPR_NUM_THREADS = 1
CUDA_VISIBLE_DEVICES = ""
```

Do not increase `n_envs` just to use more CPU. Prefer running multiple independent PPO jobs in parallel.

### 6.2 Required parallel schedule

Start the three mandatory S2-D jobs in parallel:

```text
S2-D seed0
S2-D seed1
S2-D seed2
```

If system health remains good after the first heartbeat window, start optional jobs in this priority order:

```text
1. S2-D seed3 sanity
2. attention_full formal seed1, if needed
3. attention_full formal seed2, if needed
```

Maximum concurrency guideline:

```text
initial max concurrent PPO jobs: 3
resource-healthy max concurrent PPO jobs: 4
allow 5 only if CPU/RAM/fps/logging remain clearly healthy
```

Health criteria:

```text
no OOM or swap pressure;
root disk has safe free space;
each PPO job still makes progress and writes checkpoints;
fps does not collapse persistently below a reasonable fraction of prior single-job speed;
no repeated failed writes / checkpoint corruption;
watcher remains responsive;
logs are updated;
no NaN/Inf feature diagnostics.
```

If degradation occurs, do not kill mandatory S2-D jobs first. Pause or queue optional jobs in this order:

```text
attention_full seed2
attention_full seed1
S2-D seed3
```

Keep S2-D seed0/1/2 running if safe.

### 6.3 Monitoring cadence

Long training heartbeat should be roughly every 5 minutes, not every 60 seconds.

Record per-job:

```text
pid
seed
variant
current timestep
latest checkpoint
fps if available
recent reward if available
CPU/RAM/disk summary
last log line timestamp
```

---

## 7. PPO training diagnostics to add or preserve

Because S2-D `checkpoint_1000k` and `final.zip` diverged strongly in Phase N3PF-STAB, this phase must log PPO optimization diagnostics whenever possible.

Required if available from SB3 logger:

```text
approx_kl
clip_fraction
entropy_loss
policy_gradient_loss
value_loss
explained_variance
std or log_std
learning_rate
loss
n_updates
```

If the current training script does not export these, add lightweight logging without changing PPO update logic.

Also compute parameter deltas:

```text
selected_checkpoint vs final.zip
1000k vs final.zip
feature_extractor parameter L2 delta
actor/action_net parameter L2 delta
critic/value_net parameter L2 delta
log_std delta if available
gate parameter delta if available
```

This is diagnostic only. Do not use these diagnostics to select checkpoints after test evaluation.

---

## 8. Validation selector

For every training seed, evaluate all candidate checkpoints on validation seeds 900/901.

Candidate checkpoints:

```text
500k
750k
1000k
1250k
1500k
final
```

Use a pre-registered selection score:

```text
selection_score = success_rate - 2 * collision_rate
```

Tie-breakers, in order:

```text
1. lower collision_rate
2. higher success_rate
3. higher progress
4. lower raw_unsafe_action_rate
5. earlier checkpoint, if all above are nearly equal
```

Record:

```text
validation_checkpoint_scores.csv
selector_decision.csv
selected checkpoint per training seed
whether final.zip was selected or rejected
selected-vs-final comparison
```

Do not inspect test results before selector decisions are frozen and written to disk.

---

## 9. Frozen test evaluation

After validation selector has frozen the selected checkpoint for each training seed, evaluate selected checkpoints on test seeds:

```text
test eval seeds: 1000, 1001, 1002
scenarios: all 6 standard EnvV2 eval scenarios
episodes per scenario per eval seed: use the established formal protocol, normally 50
```

Required summaries:

```text
test_eval_summary_aggregate.csv
test_eval_summary_by_seed.csv
test_scenario_breakdown.csv
test_motion_mode_breakdown.csv
test_threat_class_breakdown.csv
test_raw_unsafe_action_summary.csv
test_raw_unsafe_action_steps.csv or compressed aggregate if the raw file is too large
test_feature_block_stats.csv
test_gpsi_output_summary.csv
```

If a raw per-step CSV is too large, create a compressed aggregate that includes at minimum:

```text
variant
training_seed
eval_seed
checkpoint_label
scenario
motion_mode
threat_class
raw_unsafe_action_rate
action_delta_mean
action_delta_p50
action_delta_p95
raw_min_predicted_cpa_mean
raw_min_predicted_cpa_p05
raw_min_predicted_ttc_mean
raw_min_predicted_ttc_p05
progress_mean
collision_rate
success_rate
```

---

## 10. Optional final held-out evaluation

Only if S2-D passes the test gate, run final held-out seeds:

```text
1100, 1101, 1102
```

Do not use these seeds to tune anything. They are a final sanity check only.

---

## 11. Confirmation gates

### 11.1 Minimum pass gate

S2-D can be considered a stable candidate only if all are true:

```text
mandatory seeds 0/1/2 all trained successfully;
validation selector completed without using test seeds;
no training seed collapses to original seed2-level failure;
min selected-test success across training seeds >= 0.56;
mean selected-test success across training seeds >= 0.60 or close to attention_full reference with clearly lower collision;
mean selected-test collision <= 0.40 to 0.42;
feature diagnostics clean;
selected checkpoint is not chosen post-hoc from test results.
```

### 11.2 Strong pass gate

A stronger result suitable for restoring N4-O and preparing N4-U requires:

```text
S2-D seed0/1/2 mean success > available attention_full seed0 reference 0.6033;
S2-D seed0/1/2 mean collision < available attention_full seed0 reference 0.3967;
no seed below no_z reference by a large margin;
no high-speed or high-threat collapse;
checkpoint selector is clean;
if attention_full formal seed1/2 are available, S2-D is competitive or better under the same eval protocol.
```

### 11.3 Fail gate

Declare S2-D not yet stable if any of these occurs:

```text
any mandatory seed collapses near success <= 0.45 / collision >= 0.55;
selected checkpoint depends on test seeds;
feature/gate/Gψ diagnostics show nonfinite values or scale explosion;
1000k-vs-final drift repeats across most seeds and selector cannot identify a stable region;
S2-D mean remains below no_z or attention_full reference with no safety advantage.
```

---

## 12. Attention_full multiseed reporting rule

If attention_full seed1/2 are not run or not validated as formal references, the final report must say:

```text
S2-D multi-seed is compared against the available attention_full seed0 reference only.
A decisive claim that S2-D is robustly better than attention_full requires formal attention_full multi-training-seed evaluation.
```

If formal attention_full seed0/1/2 are available, report same-protocol comparison:

```text
S2-D selected checkpoints, train seeds 0/1/2[/3]
vs
attention_full train seeds 0/1/2
```

Metrics:

```text
mean success
std success across training seeds
min success
mean collision
std collision across training seeds
max collision
scenario / motion / threat breakdown
raw unsafe action rate
progress
```

---

## 13. N4-O / N4-U decision after this phase

At the end, set one of the following decisions:

```text
A. s2d_confirmed_stable_restore_n4o_next
   - S2-D passed multi-seed confirmation.
   - Next phase should rerun N4-O ordinary shield with S2-D selected checkpoints.
   - N4-U remains after N4-O rerun.

B. s2d_promising_but_needs_attention_warmstart_or_distillation
   - S2-D improves seed2 but is not robust enough.
   - Next phase should implement S2-B/S2-C strict attention-preserving fusion.

C. s2d_failed_stability_gate
   - S2-D does not solve training-seed instability.
   - Do not proceed to N4-O/N4-U.
   - Reconsider reward-safety mismatch, dense CPA/TTC cost, or risk-window response cost in a separate phase.
```

N4-U must remain blocked in this phase regardless of outcome.

N4-O may be marked as ready-to-rerun only if S2-D passes multi-seed confirmation.

---

## 14. Required scripts

Create or update scripts as needed, preferring small, phase-specific scripts:

```text
scripts/watch_phase_n3pf_stab_confirm.sh
scripts/eval_env_v2_gpsi_ppo_n3pf_stab_confirm.py
scripts/select_env_v2_phase_n3pf_stab_confirm_checkpoints.py
scripts/analyze_env_v2_phase_n3pf_stab_confirm.py
```

If current training script already supports S2-D config and required CLI arguments, reuse it. Do not duplicate training logic unnecessarily.

Preflight checks:

```bash
python -m py_compile scripts/eval_env_v2_gpsi_ppo_n3pf_stab_confirm.py
python -m py_compile scripts/select_env_v2_phase_n3pf_stab_confirm_checkpoints.py
python -m py_compile scripts/analyze_env_v2_phase_n3pf_stab_confirm.py
bash -n scripts/watch_phase_n3pf_stab_confirm.sh
```

---

## 15. Blocking watcher requirements

The watcher must be blocking and self-contained.

It must:

```text
1. run GitHub pre-sync or verify it was done;
2. run repo/config/checkpoint preflight;
3. launch mandatory S2-D seed0/1/2 jobs in parallel;
4. launch optional S2-D seed3 / attention_full jobs if resources remain healthy;
5. monitor all jobs until completion or stop condition;
6. run validation checkpoint evaluation;
7. freeze selector decisions before any test evaluation;
8. run test evaluation;
9. optionally run final held-out evaluation only if the gate passes;
10. run analysis;
11. write final report;
12. write complete flag only after all required artifacts are valid.
```

The watcher may end only when one of these appears:

```text
results/env_v2_phase_n3pf_stab_confirm/PHASE_N3PF_STAB_CONFIRM_COMPLETE.flag
```

or a stop flag / stop condition:

```text
results/env_v2_phase_n3pf_stab_confirm/STOP_RESOURCE_UNSAFE.flag
results/env_v2_phase_n3pf_stab_confirm/STOP_PREFLIGHT_FAILED.flag
results/env_v2_phase_n3pf_stab_confirm/STOP_TRAINING_FAILED.flag
results/env_v2_phase_n3pf_stab_confirm/STOP_SELECTOR_CONTAMINATED.flag
results/env_v2_phase_n3pf_stab_confirm/STOP_EVAL_FAILED.flag
```

If a stop flag is written, the report must explain the reason and include partial artifacts.

---

## 16. Final report required contents

The final report must include:

```text
terminal_decision
GitHub sync status and commit
resource/parallelism summary
repo verification
config manifest and config diff
training command manifest
checkpoint manifest
PPO diagnostics summary
validation checkpoint scores
selector decision
selected-vs-final comparison
1000k-vs-final drift analysis per seed
test aggregate results
test per-training-seed results
scenario breakdown
motion-mode breakdown
threat-class breakdown
raw unsafe / CPA / TTC / action_delta diagnostics
feature / Gψ / gate diagnostics
attention_full multiseed audit or formal result
final held-out result if run
N4-O readiness decision
N4-U blocked status
next recommended phase
```

Use precise language:

```text
Do say: S2-D is confirmed / not confirmed as a stable candidate under this protocol.
Do not say: S2-D is strictly attention-preserving unless attention weights are loaded or distillation/pretraining is performed.
Do not say: N4-U can proceed from screening-only evidence.
```

---

## 17. Expected terminal decision names

Use one of:

```text
phase_n3pf_stab_confirm_complete_s2d_confirmed
phase_n3pf_stab_confirm_complete_s2d_promising_not_confirmed
phase_n3pf_stab_confirm_complete_s2d_failed
phase_n3pf_stab_confirm_stopped_resource_unsafe
phase_n3pf_stab_confirm_stopped_preflight_failed
phase_n3pf_stab_confirm_stopped_training_failed
phase_n3pf_stab_confirm_stopped_selector_contaminated
phase_n3pf_stab_confirm_stopped_eval_failed
```

