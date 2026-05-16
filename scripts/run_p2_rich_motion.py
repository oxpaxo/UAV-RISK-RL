from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results/p2_rich_motion"
OOD_DIR = OUT_DIR / "ood_eval"
STAGE2_EVAL_DIR = OUT_DIR / "stage2_eval"
STAGE4_EVAL_DIR = OUT_DIR / "stage4_eval"
PLOTS_DIR = OUT_DIR / "plots"
CKPT_DIR = ROOT / "checkpoints/p2_rich_motion"
RUN_DIR = ROOT / "runs/p2_rich_motion"
LOG_DIR = ROOT / "runs/logs"

CHECKPOINT_STEPS = [250000, 500000, 750000, 1000000]
STAGE1_STEPS = [750000, 1000000]
NEW_SCENARIOS = [
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_mixed_v2",
    "eval_threat_validated_sudden",
]
STAGE2_SCENARIOS = [
    "eval_random_switch",
    "eval_sudden_turn",
    "eval_random_switch_hard",
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_mixed_v2",
    "eval_threat_validated_sudden",
]
PARETO_SCENARIOS = [
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_mixed_v2",
    "eval_threat_validated_sudden",
    "eval_sudden_turn",
    "eval_random_switch_hard",
]
SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "mixed_uncertainty": "mixed",
    "eval_sinusoidal": "sinusoidal",
    "eval_accel_decel": "accel",
    "eval_ar1": "ar1",
    "eval_mixed_v2": "mixed_v2",
    "eval_threat_validated_sudden": "threat_sudden",
}
EPS = 1e-10


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_method: str
    d_warning: float
    use_safety_cost: bool
    cost_type: str
    beta_cost: float = 5.0


STAGE1_METHODS = [
    MethodSpec("attention_full", "attention_full", 1.0, False, "none"),
    MethodSpec("attention_full_distance_penalty", "attention_full_distance_penalty_d1", 1.0, True, "distance_warning"),
    MethodSpec("attention_full_distance_penalty_wide_d2", "attention_full_distance_penalty_wide_d2", 2.0, True, "distance_warning"),
    MethodSpec("attention_full_risk_penalty", "attention_full_risk_penalty", 1.0, True, "risk_sum"),
]
STAGE2_METHODS = [
    MethodSpec("attention_full", "attention_full", 1.0, False, "none"),
    MethodSpec("attention_full_distance_penalty_wide_d2", "attention_full_distance_penalty_wide_d2", 2.0, True, "distance_warning"),
    MethodSpec("attention_full_risk_penalty", "attention_full_risk_penalty", 1.0, True, "risk_sum"),
]
STAGE4_METHODS = [
    MethodSpec("attention_full_distance_penalty_wide_d2", "attention_full_distance_penalty_wide_d2", 2.0, True, "distance_warning"),
    MethodSpec("attention_full_risk_penalty", "attention_full_risk_penalty", 1.0, True, "risk_sum"),
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, OOD_DIR, STAGE2_EVAL_DIR, STAGE4_EVAL_DIR, PLOTS_DIR, CKPT_DIR, RUN_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def run_command(cmd: list[str], log_path: Path, skip_marker: str | None = None) -> None:
    if skip_marker and log_path.exists() and skip_marker in log_path.read_text(encoding="utf-8", errors="ignore"):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP completed log={rel(log_path)}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def mean_col(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.mean())


def nan_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").isna().mean())


def unique_modes(df: pd.DataFrame) -> str:
    if "obstacle_motion_modes" not in df:
        return ""
    values = sorted({str(v) for v in df["obstacle_motion_modes"].dropna().unique()})
    return ";".join(values[:8])


def summarize_eval_csv(path: Path, method: str, seed: int, step: int, scenario: str, d_warning: float, checkpoint: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    cost_max = pd.to_numeric(df.get("distance_warning_cost_max", pd.Series(dtype=float)), errors="coerce")
    reaction_eval = pd.to_numeric(df.get("reaction_time_eval_style", pd.Series(dtype=float)), errors="coerce")
    reaction_nan = pd.to_numeric(df.get("reaction_time_nan_style", pd.Series(dtype=float)), errors="coerce")
    return {
        "method": method,
        "seed": int(seed),
        "step": int(step),
        "scenario": scenario,
        "d_warning": float(d_warning),
        "episodes": int(len(df)),
        "success_rate": mean_col(df, "success"),
        "collision_rate": mean_col(df, "collision"),
        "reaction_time_eval_style": float(reaction_eval.mean()) if not reaction_eval.dropna().empty else float("nan"),
        "reaction_time_nan_style": float(reaction_nan.mean()) if not reaction_nan.dropna().empty else float("nan"),
        "nan_reaction_rate": nan_rate(df, "reaction_time_nan_style"),
        "mean_min_distance": mean_col(df, "episode_min_distance"),
        "near_miss_rate": mean_col(df, "near_miss"),
        "mean_time": mean_col(df, "time_to_goal"),
        "mean_episode_reward": mean_col(df, "episode_reward"),
        "distance_warning_cost_nonzero_rate": float((cost_max > 0.0).mean()) if len(cost_max) else float("nan"),
        "distance_warning_cost_mean": mean_col(df, "distance_warning_cost_mean"),
        "distance_warning_cost_p90": mean_col(df, "distance_warning_cost_p90"),
        "distance_warning_cost_p95": mean_col(df, "distance_warning_cost_p95"),
        "distance_warning_cost_max": float(cost_max.max()) if not cost_max.dropna().empty else float("nan"),
        "risk_sum_mean": mean_col(df, "risk_sum_mean"),
        "risk_max_mean": mean_col(df, "risk_max_mean"),
        "scenario_valid_rate": mean_col(df, "scenario_valid"),
        "planned_threat_valid_rate": mean_col(df, "planned_threat_valid"),
        "threat_valid_rate": mean_col(df, "threat_valid"),
        "realized_near_miss_rate": mean_col(df, "realized_near_miss"),
        "predicted_cpa_to_nominal_path_mean": mean_col(df, "predicted_cpa_to_nominal_path"),
        "obstacle_motion_modes": unique_modes(df),
        "source_csv": rel(path),
        "checkpoint_path": rel(checkpoint),
    }


def stage1_checkpoint(spec: MethodSpec, step: int) -> Path:
    if spec.display_method == "attention_full":
        return ROOT / f"checkpoints/longtrain_baseline/attention_full_s0_step{step}.zip"
    if spec.display_method == "attention_full_distance_penalty_d1":
        return ROOT / f"checkpoints/gate2b/attention_full_distance_penalty_s0_step{step}.zip"
    if spec.display_method == "attention_full_distance_penalty_wide_d2":
        return ROOT / f"checkpoints/p0_5_distance_wide/attention_full_distance_penalty_wide_s0_step{step}.zip"
    if spec.display_method == "attention_full_risk_penalty":
        return ROOT / f"checkpoints/gate2b/attention_full_risk_penalty_s0_step{step}.zip"
    raise ValueError(spec.display_method)


def p2_checkpoint(spec: MethodSpec, seed: int, step: int) -> Path:
    return CKPT_DIR / f"{spec.method}_s{seed}_step{step}.zip"


def eval_csv_path(base_dir: Path, spec: MethodSpec, seed: int, step: int, scenario: str) -> Path:
    return base_dir / f"{spec.method}_s{seed}_step{step}_{SCENARIO_SUFFIX[scenario]}.csv"


def train_if_needed(spec: MethodSpec, seed: int, stage: str, total_steps: int = 1000000, n_envs: int = 16) -> None:
    checkpoint_steps = [step for step in CHECKPOINT_STEPS if step <= total_steps]
    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)
    checkpoint_steps = sorted(set(checkpoint_steps))
    if all(p2_checkpoint(spec, seed, step).exists() for step in checkpoint_steps):
        log_path = LOG_DIR / f"train_p2_{stage}_{spec.method}_s{seed}.log"
        if not log_path.exists() or "TRAIN_END" not in log_path.read_text(encoding="utf-8", errors="ignore"):
            log_path.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP existing checkpoints\nTRAIN_END reused_existing\n", encoding="utf-8")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP training existing method={spec.method} seed={seed}", flush=True)
        return
    cmd = [
        PYTHON,
        "train.py",
        "--method",
        spec.method,
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
        "--seed",
        str(seed),
        "--total_steps",
        str(total_steps),
        "--n_envs",
        str(n_envs),
        "--device",
        "cpu",
        "--scenario",
        "train_mixed_modes_v2",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        ",".join(str(step) for step in checkpoint_steps),
        "--checkpoint_dir",
        str(CKPT_DIR),
        "--log_dir",
        str(RUN_DIR / f"{spec.method}_s{seed}"),
        "--run_name",
        spec.method,
        "--save_path",
        str(CKPT_DIR / f"{spec.method}_s{seed}_step{total_steps}.zip"),
        "--heartbeat_seconds",
        "30",
    ]
    if spec.use_safety_cost:
        cmd += [
            "--use_safety_cost",
            "true",
            "--cost_type",
            spec.cost_type,
            "--fallback_penalty",
            "true",
            "--beta_cost",
            str(spec.beta_cost),
            "--d_warning",
            str(spec.d_warning),
        ]
    run_command(cmd, LOG_DIR / f"train_p2_{stage}_{spec.method}_s{seed}.log", skip_marker="TRAIN_END")


def eval_if_needed(
    spec: MethodSpec,
    seed: int,
    step: int,
    scenario: str,
    checkpoint: Path,
    out_dir: Path,
    log_prefix: str,
    episodes: int = 50,
) -> Path:
    out_csv = eval_csv_path(out_dir, spec, seed, step, scenario)
    if csv_row_count(out_csv) >= episodes:
        return out_csv
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(checkpoint),
        "--method",
        spec.method,
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
        "--seed",
        str(seed),
        "--eval_seed",
        "1000",
        "--episodes",
        str(episodes),
        "--scenario",
        scenario,
        "--device",
        "cpu",
        "--out_csv",
        str(out_csv),
        "--global_step",
        str(step),
        "--heartbeat_seconds",
        "15",
        "--d_warning",
        str(spec.d_warning),
    ]
    log_path = LOG_DIR / f"{log_prefix}_{spec.method}_s{seed}_step{step}_{SCENARIO_SUFFIX[scenario]}.log"
    run_command(cmd, log_path, skip_marker="EVAL_END")
    return out_csv


def stage0_random_policy() -> pd.DataFrame:
    from envs.dynamic_obstacle_env import DynamicObstacleEnv

    rows: list[dict[str, Any]] = []
    episodes = 10
    for scenario in NEW_SCENARIOS:
        success = []
        collision = []
        near_miss = []
        min_distance = []
        times = []
        scenario_valid = []
        threat_valid = []
        init_collision = []
        for episode_id in range(episodes):
            env = DynamicObstacleEnv(scenario=scenario)
            _, info = env.reset(seed=1000 + episode_id)
            init_collision.append(int(float(info.get("initial_min_distance", 999.0)) < env.d_collision))
            done = False
            steps = 0
            while not done:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                steps += 1
                done = terminated or truncated
            success.append(int(info["is_success"]))
            collision.append(int(info["is_collision"]))
            near_miss.append(int(float(info["episode_min_distance"]) < 1.0))
            min_distance.append(float(info["episode_min_distance"]))
            times.append(steps * env.dt)
            scenario_valid.append(int(bool(info.get("scenario_valid", True))))
            threat_valid.append(int(bool(info.get("threat_valid", True))))
        rows.append(
            {
                "scenario": scenario,
                "policy": "random",
                "episodes": episodes,
                "success_rate": float(np.mean(success)),
                "collision_rate": float(np.mean(collision)),
                "near_miss_rate": float(np.mean(near_miss)),
                "mean_min_distance": float(np.mean(min_distance)),
                "min_min_distance": float(np.min(min_distance)),
                "mean_time": float(np.mean(times)),
                "scenario_valid_rate": float(np.mean(scenario_valid)),
                "threat_valid_rate": float(np.mean(threat_valid)),
                "init_collision_rate": float(np.mean(init_collision)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "env_sanity_random_policy.csv", index=False)
    return out


def stage0_short_ppo(n_envs: int) -> pd.DataFrame:
    short_steps = int(os.environ.get("P2_SHORT_STEPS", "50000"))
    spec = MethodSpec("p2_short_attention_full", "p2_short_attention_full", 1.0, False, "none")
    if not p2_checkpoint(spec, 0, short_steps).exists():
        train_if_needed(spec, 0, "stage0_short", total_steps=short_steps, n_envs=n_envs)
    rows: list[dict[str, Any]] = []
    ckpt = p2_checkpoint(spec, 0, short_steps)
    for scenario in ["eval_sinusoidal", "eval_accel_decel", "eval_ar1", "eval_mixed_v2", "eval_random_switch"]:
        path = eval_if_needed(spec, 0, short_steps, scenario, ckpt, OUT_DIR / "short_ppo_eval", "eval_p2_stage0_short", episodes=20)
        rows.append(summarize_eval_csv(path, spec.display_method, 0, short_steps, scenario, spec.d_warning, ckpt))
    out = pd.DataFrame(rows).sort_values(["scenario"])
    out.to_csv(OUT_DIR / "env_sanity_short_ppo.csv", index=False)
    return out


def stage0_pass(random_df: pd.DataFrame, short_df: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if (random_df["scenario_valid_rate"] < 0.95).any() or (short_df["scenario_valid_rate"] < 0.95).any():
        reasons.append("scenario_valid_rate_below_0p95")
    threat_random = random_df[random_df["scenario"].isin(["eval_mixed_v2", "eval_threat_validated_sudden"])]
    threat_short = short_df[short_df["scenario"].isin(["eval_mixed_v2"])]
    if (threat_random["threat_valid_rate"] < 0.8).any() or (threat_short["threat_valid_rate"] < 0.8).any():
        reasons.append("threat_valid_rate_below_0p8")
    common = sorted(set(random_df["scenario"]) & set(short_df["scenario"]))
    if common:
        r = random_df[random_df["scenario"].isin(common)]
        s = short_df[short_df["scenario"].isin(common)]
        if (r["success_rate"].min() > 0.95 and s["success_rate"].min() > 0.95 and r["near_miss_rate"].max() < 0.05 and s["near_miss_rate"].max() < 0.05):
            reasons.append("environment_too_easy_random_and_short_ppo")
        if r["collision_rate"].min() > 0.95 and s["collision_rate"].min() > 0.95:
            reasons.append("environment_too_hard_random_and_short_ppo")
    if (random_df["init_collision_rate"] > 0.05).any():
        reasons.append("init_collision_rate_above_0p05")
    return not reasons, reasons


def write_stage0_report(random_df: pd.DataFrame, short_df: pd.DataFrame, passed: bool, reasons: list[str]) -> None:
    lines = [
        "# P2 Environment Sanity Report",
        "",
        "## Stage 0 Result",
        f"- Passed: {passed}",
        f"- Reasons: {', '.join(reasons) if reasons else 'none'}",
        "",
        "## Random Policy Rollout",
        "| scenario | success | collision | near_miss | min_distance | scenario_valid | threat_valid | init_collision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in random_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['mean_min_distance'])} | {fmt(row['scenario_valid_rate'])} | {fmt(row['threat_valid_rate'])} | {fmt(row['init_collision_rate'])} |"
        )
    lines += [
        "",
        "## Short PPO Sanity",
        "| scenario | success | collision | near_miss | mean_time | reward | scenario_valid | threat_valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in short_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['mean_time'])} | {fmt(row['mean_episode_reward'])} | {fmt(row['scenario_valid_rate'])} | {fmt(row['threat_valid_rate'])} |"
        )
    write_lines(ROOT / "P2_ENVIRONMENT_SANITY_REPORT.md", lines)


def run_stage1() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in STAGE1_METHODS:
        for step in STAGE1_STEPS:
            ckpt = stage1_checkpoint(spec, step)
            for scenario in NEW_SCENARIOS:
                path = eval_if_needed(spec, 0, step, scenario, ckpt, OOD_DIR, "eval_p2_stage1_ood", episodes=50)
                rows.append(summarize_eval_csv(path, spec.display_method, 0, step, scenario, spec.d_warning, ckpt))
    out = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    out.to_csv(OUT_DIR / "p2_stage1_ood_eval.csv", index=False)
    return out


def stage1_pass(df: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    primary = df[df["method"].isin(["attention_full", "attention_full_distance_penalty_wide_d2", "attention_full_risk_penalty"])]
    if primary.empty:
        return False, ["missing_primary_stage1_rows"]
    too_easy = primary["success_rate"].min() > 0.95 and primary["near_miss_rate"].max() < 0.05 and primary["collision_rate"].max() < 0.02
    too_hard = primary["collision_rate"].min() > 0.5 and primary["success_rate"].max() < 0.5
    if too_easy:
        reasons.append("new_modes_too_easy_all_methods")
    if too_hard:
        reasons.append("new_modes_too_hard_all_methods")
    baseline = primary[primary["method"] == "attention_full"]
    safety_issue = bool(
        (baseline["near_miss_rate"] >= 0.10).any()
        or (baseline["collision_rate"] >= 0.05).any()
        or (baseline["reaction_time_eval_style"].fillna(0.0) > 1.0).any()
    )
    efficiency_diff = False
    for (step, scenario), group in primary.groupby(["step", "scenario"]):
        risk = group[group["method"] == "attention_full_risk_penalty"]
        wide = group[group["method"] == "attention_full_distance_penalty_wide_d2"]
        if risk.empty or wide.empty:
            continue
        if abs(float(risk.iloc[0]["mean_time"]) - float(wide.iloc[0]["mean_time"])) > 0.25:
            efficiency_diff = True
    if not safety_issue and not efficiency_diff:
        reasons.append("no_safety_or_efficiency_discrimination")
    return not reasons, reasons


def write_stage1_report(df: pd.DataFrame, passed: bool, reasons: list[str]) -> None:
    lines = [
        "# P2 Stage 1 OOD Evaluation Report",
        "",
        f"- Passed Stage 1 gate: {passed}",
        f"- Reasons: {', '.join(reasons) if reasons else 'none'}",
        "",
        "## 750k / 1000k Existing Checkpoint OOD Summary",
        "| method | step | scenario | success | collision | reaction | near_miss | min_distance | mean_time | threat_valid |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.sort_values(["step", "scenario", "method"]).iterrows():
        lines.append(
            f"| {row['method']} | {int(row['step'])} | {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | "
            f"{fmt(row['reaction_time_eval_style'])} | {fmt(row['near_miss_rate'])} | {fmt(row['mean_min_distance'])} | "
            f"{fmt(row['mean_time'])} | {fmt(row['threat_valid_rate'])} |"
        )
    write_lines(ROOT / "P2_STAGE1_OOD_EVAL_REPORT.md", lines)
    plot_stage1(df)


def plot_stage1(df: pd.DataFrame) -> None:
    for metric in ["success_rate", "collision_rate", "near_miss_rate", "mean_time", "reaction_time_eval_style"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        subset = df[df["step"] == 750000]
        for method, group in subset.groupby("method", sort=True):
            ax.plot(group["scenario"], group[metric], marker="o", label=method)
        ax.set_title(f"Stage1 750k {metric}")
        ax.set_xlabel("scenario")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"stage1_{metric}.png", dpi=160)
        plt.close(fig)


def run_stage2_seed0(n_envs: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in STAGE2_METHODS:
        train_if_needed(spec, 0, "stage2", total_steps=1000000, n_envs=n_envs)
    for spec in STAGE2_METHODS:
        for step in CHECKPOINT_STEPS:
            ckpt = p2_checkpoint(spec, 0, step)
            for scenario in STAGE2_SCENARIOS:
                path = eval_if_needed(spec, 0, step, scenario, ckpt, STAGE2_EVAL_DIR, "eval_p2_stage2", episodes=50)
                rows.append(summarize_eval_csv(path, spec.display_method, 0, step, scenario, spec.d_warning, ckpt))
    out = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    out.to_csv(OUT_DIR / "p2_seed0_by_step_scenario.csv", index=False)
    write_stage2_report(out)
    return out


def write_stage2_report(df: pd.DataFrame) -> None:
    main = build_seed0_main(df)
    lines = [
        "# P2 Stage 2 Rich Training Seed0 Report",
        "",
        "## Scope",
        "- Trained attention_full, distance_penalty_wide_d2, and risk_penalty on train_mixed_modes_v2 with seed=0.",
        "- Evaluated 250k/500k/750k/1000k on legacy and new P2 scenarios.",
        "",
        "## 750k/1000k Main Table",
        "| method | step | scenario | success | collision | reaction | near_miss | min_distance | mean_time | threat_valid |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['step'])} | {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | "
            f"{fmt(row['reaction_time_eval_style'])} | {fmt(row['near_miss_rate'])} | {fmt(row['mean_min_distance'])} | "
            f"{fmt(row['mean_time'])} | {fmt(row['threat_valid_rate'])} |"
        )
    write_lines(ROOT / "P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md", lines)


def build_seed0_main(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "method",
        "step",
        "scenario",
        "success_rate",
        "collision_rate",
        "reaction_time_eval_style",
        "mean_min_distance",
        "near_miss_rate",
        "mean_time",
        "distance_warning_cost_nonzero_rate",
        "risk_sum_mean",
        "threat_valid_rate",
    ]
    main = df[df["step"].isin([750000, 1000000])][cols].copy().sort_values(["step", "scenario", "method"])
    main.to_csv(OUT_DIR / "p2_seed0_main_750k_1000k_table.csv", index=False)
    return main


def build_risk_adaptation_summary(df: pd.DataFrame) -> pd.DataFrame:
    use = df[df["step"].isin([750000, 1000000])].copy()
    rows: list[dict[str, Any]] = []
    for (scenario, method), group in use.groupby(["scenario", "method"], sort=True):
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "risk_sum_mean": float(group["risk_sum_mean"].mean()),
                "risk_max_mean": float(group["risk_max_mean"].mean()),
                "distance_warning_cost_nonzero_rate": float(group["distance_warning_cost_nonzero_rate"].mean()),
                "mean_min_distance": float(group["mean_min_distance"].mean()),
                "mean_time": float(group["mean_time"].mean()),
                "near_miss_rate": float(group["near_miss_rate"].mean()),
                "collision_rate": float(group["collision_rate"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["scenario", "method"])
    out.to_csv(OUT_DIR / "p2_risk_adaptation_summary.csv", index=False)
    return out


def plot_seed0_pareto(df: pd.DataFrame) -> None:
    use = df[df["step"].isin([750000, 1000000])].copy()
    metrics = [
        ("reaction_time_eval_style", "Reaction Time"),
        ("near_miss_rate", "Near Miss Rate"),
        ("collision_rate", "Collision Rate"),
        ("mean_min_distance", "Mean Min Distance"),
        ("success_rate", "Success Rate"),
    ]
    for scenario in PARETO_SCENARIOS:
        scenario_df = use[use["scenario"] == scenario]
        if scenario_df.empty:
            continue
        for metric, label in metrics:
            fig, ax = plt.subplots(figsize=(7, 5))
            for method, group in scenario_df.groupby("method", sort=True):
                ax.scatter(group["mean_time"], group[metric], s=70, label=method)
                for _, row in group.iterrows():
                    ax.annotate(str(int(row["step"] / 1000)), (row["mean_time"], row[metric]), fontsize=7)
            ax.set_title(f"{scenario}: {label} vs Mean Time")
            ax.set_xlabel("mean_time")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"seed0_pareto_{metric}_{scenario}.png", dpi=160)
            plt.close(fig)


def stage3_gate(df: pd.DataFrame) -> tuple[bool, list[str], pd.DataFrame]:
    use = df[df["step"].isin([750000, 1000000]) & df["scenario"].isin(PARETO_SCENARIOS)].copy()
    rows: list[dict[str, Any]] = []
    for scenario, group in use.groupby("scenario", sort=True):
        risk = group[group["method"] == "attention_full_risk_penalty"]
        wide = group[group["method"] == "attention_full_distance_penalty_wide_d2"]
        base = group[group["method"] == "attention_full"]
        if risk.empty or wide.empty:
            continue
        risk_mean = risk.mean(numeric_only=True)
        wide_mean = wide.mean(numeric_only=True)
        base_mean = base.mean(numeric_only=True) if not base.empty else pd.Series(dtype=float)
        safety_close = (
            float(risk_mean["collision_rate"]) <= float(wide_mean["collision_rate"]) + 0.05
            and float(risk_mean["near_miss_rate"]) <= float(wide_mean["near_miss_rate"]) + 0.10
            and float(risk_mean["mean_min_distance"]) >= float(wide_mean["mean_min_distance"]) - 0.25
        )
        if scenario in {"eval_sudden_turn", "eval_mixed_v2", "eval_threat_validated_sudden"}:
            r_react = float(risk_mean.get("reaction_time_eval_style", np.nan))
            w_react = float(wide_mean.get("reaction_time_eval_style", np.nan))
            if not math.isnan(r_react) and not math.isnan(w_react):
                safety_close = safety_close and r_react <= w_react + 0.50
        risk_faster = float(risk_mean["mean_time"]) <= float(wide_mean["mean_time"]) - 0.25
        risk_safer = (
            float(risk_mean["collision_rate"]) < float(wide_mean["collision_rate"]) - 0.02
            or float(risk_mean["near_miss_rate"]) < float(wide_mean["near_miss_rate"]) - 0.05
        )
        baseline_issue = False
        if not base_mean.empty:
            baseline_issue = (
                float(base_mean["near_miss_rate"]) >= 0.10
                or float(base_mean["collision_rate"]) >= 0.05
                or float(base_mean.get("reaction_time_eval_style", 0.0)) > 1.0
            )
        rows.append(
            {
                "scenario": scenario,
                "risk_safety_close_to_wide": bool(safety_close),
                "risk_faster_than_wide": bool(risk_faster),
                "risk_safer_than_wide": bool(risk_safer),
                "baseline_issue": bool(baseline_issue),
                "risk_mean_time": float(risk_mean["mean_time"]),
                "wide_mean_time": float(wide_mean["mean_time"]),
                "risk_collision": float(risk_mean["collision_rate"]),
                "wide_collision": float(wide_mean["collision_rate"]),
                "risk_near_miss": float(risk_mean["near_miss_rate"]),
                "wide_near_miss": float(wide_mean["near_miss_rate"]),
            }
        )
    gate_df = pd.DataFrame(rows)
    gate_df.to_csv(OUT_DIR / "p2_stage3_gate_by_scenario.csv", index=False)
    count = int((gate_df["risk_safety_close_to_wide"] & (gate_df["risk_faster_than_wide"] | gate_df["risk_safer_than_wide"])).sum()) if not gate_df.empty else 0
    reasons = [f"pareto_positive_scenarios={count}"]
    go = count >= 2
    if not go:
        reasons.append("stage3_threshold_not_met")
    return go, reasons, gate_df


def write_stage3_report(df: pd.DataFrame, adaptation: pd.DataFrame, gate_df: pd.DataFrame, go: bool, reasons: list[str]) -> None:
    lines = [
        "# P2 Stage 3 Seed0 Pareto Report",
        "",
        f"- Go to Stage 4: {go}",
        f"- Reasons: {', '.join(reasons)}",
        "",
        "## Gate By Scenario",
        "| scenario | close_safety | risk_faster | risk_safer | baseline_issue | risk_time | wide_time | risk_near | wide_near |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in gate_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {int(row['risk_safety_close_to_wide'])} | {int(row['risk_faster_than_wide'])} | "
            f"{int(row['risk_safer_than_wide'])} | {int(row['baseline_issue'])} | {fmt(row['risk_mean_time'])} | "
            f"{fmt(row['wide_mean_time'])} | {fmt(row['risk_near_miss'])} | {fmt(row['wide_near_miss'])} |"
        )
    lines += [
        "",
        "## Risk Adaptation Summary",
        "| scenario | method | risk_sum | risk_max | dist_cost_nonzero | min_distance | mean_time | near_miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in adaptation.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {fmt(row['risk_sum_mean'])} | {fmt(row['risk_max_mean'])} | "
            f"{fmt(row['distance_warning_cost_nonzero_rate'])} | {fmt(row['mean_min_distance'])} | {fmt(row['mean_time'])} | {fmt(row['near_miss_rate'])} |"
        )
    write_lines(ROOT / "P2_STAGE3_SEED0_PARETO_REPORT.md", lines)


def run_stage4(n_envs: int, seed0_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed0_keep = seed0_df[seed0_df["method"].isin([spec.display_method for spec in STAGE4_METHODS])].copy()
    rows.extend(seed0_keep.to_dict("records"))
    for spec in STAGE4_METHODS:
        for seed in [1, 2]:
            train_if_needed(spec, seed, "stage4", total_steps=1000000, n_envs=n_envs)
            for step in CHECKPOINT_STEPS:
                ckpt = p2_checkpoint(spec, seed, step)
                for scenario in STAGE2_SCENARIOS:
                    path = eval_if_needed(spec, seed, step, scenario, ckpt, STAGE4_EVAL_DIR, "eval_p2_stage4", episodes=50)
                    rows.append(summarize_eval_csv(path, spec.display_method, seed, step, scenario, spec.d_warning, ckpt))
    out = pd.DataFrame(rows).sort_values(["method", "seed", "step", "scenario"])
    out.to_csv(OUT_DIR / "p2_three_seed_summary.csv", index=False)
    write_stage4_report(out)
    return out


def write_stage4_report(df: pd.DataFrame) -> None:
    use = df[df["step"].isin([750000, 1000000])]
    rows: list[dict[str, Any]] = []
    for (method, scenario), group in use.groupby(["method", "scenario"], sort=True):
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seed_count": int(group["seed"].nunique()),
                "success_rate": float(group["success_rate"].mean()),
                "collision_rate": float(group["collision_rate"].mean()),
                "near_miss_rate": float(group["near_miss_rate"].mean()),
                "mean_time": float(group["mean_time"].mean()),
                "reaction_time_eval_style": float(group["reaction_time_eval_style"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    lines = [
        "# P2 Three-Seed Confirmation Report",
        "",
        "## 750k/1000k Mean Across Seeds",
        "| method | scenario | seeds | success | collision | reaction | near_miss | mean_time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['scenario']} | {int(row['seed_count'])} | {fmt(row['success_rate'])} | "
            f"{fmt(row['collision_rate'])} | {fmt(row['reaction_time_eval_style'])} | {fmt(row['near_miss_rate'])} | {fmt(row['mean_time'])} |"
        )
    write_lines(ROOT / "P2_THREE_SEED_CONFIRMATION_REPORT.md", lines)


def write_final_report(terminal_decision: str, stage0_reasons: list[str], stage1_reasons: list[str], stage3_reasons: list[str]) -> None:
    if terminal_decision == "stage0_no_go_environment":
        stage1_status = "not reached because Stage 0 environment sanity failed"
        stage2_status = "not reached"
        stage3_status = "not reached"
    elif terminal_decision == "stage1_no_go_ood":
        stage1_status = f"no-go: {', '.join(stage1_reasons) if stage1_reasons else 'unspecified'}"
        stage2_status = "not reached"
        stage3_status = "not reached"
    else:
        stage1_status = f"{', '.join(stage1_reasons) if stage1_reasons else 'passed'}"
        stage2_status = "reached"
        stage3_status = f"{', '.join(stage3_reasons) if stage3_reasons else 'passed'}"
    lines = [
        "# P2 Rich Motion Generalization Report",
        "",
        "## 1. Motivation",
        "P2 tests whether risk_penalty keeps a safety-efficiency Pareto advantage over distance_penalty_wide_d2 under richer obstacle motion, rather than re-proving risk over d_warning=1.0.",
        "",
        "## 2. New Motion Modes",
        "- Implemented sinusoidal lateral perturbation, accel_decel smooth speed changes, simple_ar1 stochastic velocity, train_mixed_modes_v2, eval_mixed_v2, and eval_threat_validated_sudden.",
        "- Each episode info/eval row records obstacle_motion_modes, scenario_valid, threat_valid, and threat_obstacle_id.",
        "",
        "## 3. Environment Sanity Check",
        f"- Stage 0 reasons: {', '.join(stage0_reasons) if stage0_reasons else 'passed'}",
        "- See P2_ENVIRONMENT_SANITY_REPORT.md.",
        "",
        "## 4. Existing Checkpoint OOD Evaluation",
        f"- Stage 1 status: {stage1_status}.",
        "- See P2_STAGE1_OOD_EVAL_REPORT.md and results/p2_rich_motion/p2_stage1_ood_eval.csv if Stage 1 was reached.",
        "",
        "## 5. Rich-Motion Training Seed0",
        f"- Stage 2 status: {stage2_status}.",
        "- See P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md when Stage 2 was reached.",
        "",
        "## 6. New Single-Mode Scenario Analysis",
        "- Single-mode sinusoidal / accel_decel / ar1 rows are included in p2_seed0_by_step_scenario.csv and seed0 Pareto plots when Stage 2 was reached.",
        "",
        "## 7. Risk vs Wide Distance Pareto",
        f"- Stage 3 status: {stage3_status}.",
        "",
        "## 8. Risk Adaptation Analysis",
        "- See results/p2_rich_motion/p2_risk_adaptation_summary.csv when Stage 3 was reached.",
        "",
        "## 9. Failure Cases",
        "- Mixed_v2 and threat_validated_sudden are reported separately; legacy mixed_uncertainty is not used as primary P2 evidence.",
        "",
        "## 10. Go/No-Go for Three Seeds",
        f"- Terminal decision: {terminal_decision}",
        "",
        "## 11. Final P2 Decision",
        "- If terminal decision is stage4_complete, risk remains a live Pareto-efficiency candidate against wide_d2.",
        "- If terminal decision is stage1_no_go or stage3_no_go, do not over-claim risk; revise environment or shift to safety-margin cost design as indicated.",
        "",
        "## Required Answers",
        "1. New motion modes implemented: yes.",
        "2. Random policy rollout: see env_sanity_random_policy.csv.",
        "3. Short PPO sanity: see env_sanity_short_ppo.csv.",
        "4. New eval scenario discrimination: not assessed if Stage 0 failed; otherwise see Stage 1 report.",
        "5. Existing checkpoint safety issues: not assessed if Stage 0 failed; otherwise see p2_stage1_ood_eval.csv.",
        "6. train_mixed_modes_v2 baseline drift: see p2_seed0_by_step_scenario.csv if Stage 2 reached.",
        "7. risk vs wide_d2 safety closeness: see p2_stage3_gate_by_scenario.csv if Stage 3 reached.",
        "8. risk efficiency vs wide_d2: see p2_stage3_gate_by_scenario.csv.",
        "9. Pareto front: see seed0_pareto plots and Stage 3 report.",
        "10. Single-mode results: see sinusoidal / accel_decel / ar1 rows in P2 tables.",
        "11. mixed_v2 stability: see eval_mixed_v2 rows.",
        "12. Risk adaptation: see p2_risk_adaptation_summary.csv.",
        "13. Three-seed confirmation: governed by Stage 3 gate.",
        "14. Risk mainline retention: governed by terminal decision.",
    ]
    write_lines(ROOT / "P2_RICH_MOTION_FINAL_REPORT.md", lines)


def write_flag(terminal_decision: str, extra: dict[str, Any]) -> None:
    manifest = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "terminal_decision": terminal_decision,
        "report": "P2_RICH_MOTION_FINAL_REPORT.md",
        **extra,
    }
    (OUT_DIR / "P2_COMPLETE.flag").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] P2 complete flag written terminal_decision={terminal_decision}", flush=True)


def main() -> None:
    ensure_dirs()
    n_envs = int(os.environ.get("P2_N_ENVS", "16"))

    random_df = stage0_random_policy()
    short_df = stage0_short_ppo(n_envs=n_envs)
    stage0_ok, stage0_reasons = stage0_pass(random_df, short_df)
    write_stage0_report(random_df, short_df, stage0_ok, stage0_reasons)
    if not stage0_ok:
        terminal = "stage0_no_go_environment"
        write_final_report(terminal, stage0_reasons, [], [])
        write_flag(
            terminal,
            {
                "env_sanity_random_rows": len(random_df),
                "env_sanity_short_rows": len(short_df),
            },
        )
        return

    stage1_df = run_stage1()
    stage1_ok, stage1_reasons = stage1_pass(stage1_df)
    write_stage1_report(stage1_df, stage1_ok, stage1_reasons)
    if not stage1_ok:
        terminal = "stage1_no_go_ood"
        write_final_report(terminal, stage0_reasons, stage1_reasons, [])
        write_flag(
            terminal,
            {
                "env_sanity_random_rows": len(random_df),
                "env_sanity_short_rows": len(short_df),
                "stage1_rows": len(stage1_df),
            },
        )
        return

    seed0_df = run_stage2_seed0(n_envs=n_envs)
    build_seed0_main(seed0_df)
    adaptation = build_risk_adaptation_summary(seed0_df)
    plot_seed0_pareto(seed0_df)
    stage3_go, stage3_reasons, gate_df = stage3_gate(seed0_df)
    write_stage3_report(seed0_df, adaptation, gate_df, stage3_go, stage3_reasons)
    if not stage3_go:
        terminal = "stage3_no_go_seed0"
        write_final_report(terminal, stage0_reasons, stage1_reasons, stage3_reasons)
        write_flag(
            terminal,
            {
                "env_sanity_random_rows": len(random_df),
                "env_sanity_short_rows": len(short_df),
                "stage1_rows": len(stage1_df),
                "seed0_rows": len(seed0_df),
                "seed0_main_rows": csv_row_count(OUT_DIR / "p2_seed0_main_750k_1000k_table.csv"),
                "risk_adaptation_rows": len(adaptation),
                "stage3_gate_rows": len(gate_df),
            },
        )
        return

    three_seed_df = run_stage4(n_envs=n_envs, seed0_df=seed0_df)
    terminal = "stage4_complete"
    write_final_report(terminal, stage0_reasons, stage1_reasons, stage3_reasons)
    write_flag(
        terminal,
        {
            "env_sanity_random_rows": len(random_df),
            "env_sanity_short_rows": len(short_df),
            "stage1_rows": len(stage1_df),
            "seed0_rows": len(seed0_df),
            "seed0_main_rows": csv_row_count(OUT_DIR / "p2_seed0_main_750k_1000k_table.csv"),
            "risk_adaptation_rows": len(adaptation),
            "stage3_gate_rows": len(gate_df),
            "three_seed_rows": len(three_seed_df),
        },
    )


if __name__ == "__main__":
    main()
