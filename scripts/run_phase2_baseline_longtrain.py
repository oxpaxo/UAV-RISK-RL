from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
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

OUT_DIR = ROOT / "results/env_v2_phase2"
EVAL_DIR = OUT_DIR / "eval"
TRACE_DIR = OUT_DIR / "traces"
PLOTS_DIR = OUT_DIR / "plots"
CKPT_DIR = ROOT / "checkpoints/env_v2_phase2/attention_full_s0"
RUN_DIR = ROOT / "runs/env_v2_phase2/attention_full_s0"
LOG_DIR = ROOT / "runs/logs"
STATUS_PATH = OUT_DIR / "phase2_status.txt"
LOCK_PATH = OUT_DIR / "phase2_pipeline.lock"

SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
TRACE_SCENARIOS = {"eval_flow_id", "eval_flow_high_threat", "eval_flow_mixed_ood", "eval_flow_sudden_threat"}
FIRST_STEPS = [250000, 500000, 750000, 1000000]
EXTENSION_STEPS = [1250000, 1500000]
REPRO_DECISIONS = {
    "phase2_strong_reproduction_go_phase3",
    "phase2_weak_reproduction_go_phase3_with_limited_claim",
}
NO_GO_DECISIONS = {
    "phase2_no_go_training_crash_or_nan",
    "phase2_no_go_env_incompatible_with_training",
    "phase2_no_go_baseline_cannot_learn_at_all",
    "phase2_no_go_eval_invalid_or_missing_metrics",
    "phase2_no_go_no_reproduction",
}


def ensure_dirs() -> None:
    for path in [OUT_DIR, EVAL_DIR, TRACE_DIR, PLOTS_DIR, CKPT_DIR, RUN_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)


def run_command(cmd: list[str], log_path: Path, skip_path: Path | None = None) -> None:
    if skip_path is not None and skip_path.exists() and skip_path.stat().st_size > 0:
        write_status(f"[skip] existing {rel(skip_path)}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_status(f"[run] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def checkpoint_path(step: int) -> Path:
    return CKPT_DIR / f"attention_full_s0_step{step}.zip"


def final_path(step: int) -> Path:
    return CKPT_DIR / f"attention_full_s0_final_step{step}.zip"


def train_to_1000k(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON,
        "train.py",
        "--env",
        "DynamicObstacleFlowEnv",
        "--scenario",
        "train_flow_mixed",
        "--method",
        "attention_full",
        "--run_name",
        "attention_full",
        "--agg",
        "attention",
        "--profile_mode",
        "full_12",
        "--seed",
        "0",
        "--total_steps",
        "1000000",
        "--n_envs",
        str(args.n_envs),
        "--device",
        "cpu",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        "250000,500000,750000,1000000",
        "--checkpoint_dir",
        rel(CKPT_DIR),
        "--log_dir",
        rel(RUN_DIR),
        "--save_path",
        rel(final_path(1000000)),
        "--heartbeat_seconds",
        str(args.train_heartbeat_seconds),
        "--use_safety_cost",
        "false",
        "--cost_type",
        "none",
        "--fallback_penalty",
        "false",
    ]
    run_command(cmd, LOG_DIR / "train_phase2_attention_full_s0_1000k.log", skip_path=checkpoint_path(1000000))


def train_to_1500k(args: argparse.Namespace) -> None:
    resume = checkpoint_path(1000000)
    if not resume.exists():
        raise FileNotFoundError(f"missing resume checkpoint: {resume}")
    cmd = [
        PYTHON,
        "train.py",
        "--env",
        "DynamicObstacleFlowEnv",
        "--scenario",
        "train_flow_mixed",
        "--method",
        "attention_full",
        "--run_name",
        "attention_full",
        "--agg",
        "attention",
        "--profile_mode",
        "full_12",
        "--seed",
        "0",
        "--total_steps",
        "1500000",
        "--remaining_steps",
        "500000",
        "--resume_from",
        rel(resume),
        "--resume_global_step",
        "1000000",
        "--reset_num_timesteps",
        "false",
        "--n_envs",
        str(args.n_envs),
        "--device",
        "cpu",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        "250000,500000",
        "--checkpoint_dir",
        rel(CKPT_DIR),
        "--log_dir",
        rel(ROOT / "runs/env_v2_phase2/attention_full_s0_resume"),
        "--save_path",
        rel(final_path(1500000)),
        "--heartbeat_seconds",
        str(args.train_heartbeat_seconds),
        "--use_safety_cost",
        "false",
        "--cost_type",
        "none",
        "--fallback_penalty",
        "false",
    ]
    run_command(cmd, LOG_DIR / "train_phase2_attention_full_s0_1500k.log", skip_path=checkpoint_path(1500000))


def eval_csv(step: int, scenario: str) -> Path:
    return EVAL_DIR / f"attention_full_s0_step{step}_{scenario}.csv"


def trace_csv(step: int, scenario: str) -> Path:
    return TRACE_DIR / f"attention_full_s0_step{step}_{scenario}_trace.csv"


def evaluate_steps(steps: list[int], args: argparse.Namespace) -> None:
    for step in steps:
        ckpt = checkpoint_path(step)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint for eval: {ckpt}")
        for scenario in SCENARIOS:
            out_csv = eval_csv(step, scenario)
            save_trace = scenario in TRACE_SCENARIOS
            cmd = [
                PYTHON,
                "scripts/eval_env_v2_attention_baseline.py",
                "--model_path",
                rel(ckpt),
                "--scenario",
                scenario,
                "--seed",
                "0",
                "--eval_seed",
                "1000",
                "--episodes",
                str(args.episodes),
                "--global_step",
                str(step),
                "--device",
                "cpu",
                "--out_csv",
                rel(out_csv),
                "--save_trace",
                "true" if save_trace else "false",
                "--trace_csv",
                rel(trace_csv(step, scenario)),
                "--heartbeat_seconds",
                str(args.eval_heartbeat_seconds),
            ]
            run_command(cmd, LOG_DIR / f"eval_phase2_attention_full_s0_step{step}_{scenario}.log", skip_path=out_csv)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.mean())


def load_episode_metrics(steps: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for step in steps:
        for scenario in SCENARIOS:
            path = eval_csv(step, scenario)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing eval csv: {path}")
            df = pd.read_csv(path)
            if len(df) == 0:
                raise ValueError(f"empty eval csv: {path}")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_episode_metrics(episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["checkpoint_step", "scenario"]
    summary = (
        episode_df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            near_miss_rate=("near_miss", "mean"),
            no_response_rate=("no_response", "mean"),
            reaction_time_eval_style=("reaction_time_eval_style", "mean"),
            reaction_time_nan_style=("reaction_time_nan_style", "mean"),
            conditional_reaction_time=("conditional_reaction_time", "mean"),
            nan_reaction_rate=("reaction_time_nan_style", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
            mean_min_distance=("mean_min_distance", "mean"),
            episode_min_distance=("episode_min_distance", "mean"),
            min_distance_after_threat=("min_distance_after_threat", "mean"),
            collision_or_near_miss_rate=("near_miss", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce")))),
            mean_time=("mean_time", "mean"),
            progress=("progress", "mean"),
            planned_cpa=("planned_cpa", "mean"),
            planned_ttc=("planned_ttc", "mean"),
            replacement_count=("replacement_count", "mean"),
            distance_warning_cost_nonzero_rate=("distance_warning_cost_nonzero_rate", "mean"),
            distance_warning_cost_mean=("distance_warning_cost_mean", "mean"),
            threat_valid_rate=("threat_valid_rate", "mean"),
            nan_or_crash=("nan_or_crash", "sum"),
        )
        .reset_index()
    )
    return summary


def threat_metrics(episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["checkpoint_step", "scenario", "threat_class"]
    return (
        episode_df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            planned_cpa=("planned_cpa", "mean"),
            planned_ttc=("planned_ttc", "mean"),
            no_response_rate=("no_response", "mean"),
            near_miss_rate=("near_miss", "mean"),
            collision_rate=("collision", "mean"),
            min_distance_after_threat=("min_distance_after_threat", "mean"),
            replacement_count=("replacement_count", "mean"),
        )
        .reset_index()
    )


def reaction_breakdown(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "checkpoint_step",
        "scenario",
        "episodes",
        "reaction_time_eval_style",
        "reaction_time_nan_style",
        "conditional_reaction_time",
        "nan_reaction_rate",
        "no_response_rate",
        "min_distance_after_threat",
    ]
    return summary_df[cols].copy()


def metric_delta(summary_df: pd.DataFrame, metric: str, start_step: int, end_step: int) -> pd.DataFrame:
    start = summary_df[summary_df["checkpoint_step"] == start_step][["scenario", metric]].rename(columns={metric: "start"})
    end = summary_df[summary_df["checkpoint_step"] == end_step][["scenario", metric]].rename(columns={metric: "end"})
    merged = start.merge(end, on="scenario", how="inner")
    merged["delta"] = merged["end"] - merged["start"]
    return merged


def decide_reproduction(summary_df: pd.DataFrame, max_step: int) -> tuple[str, dict[str, Any], list[str]]:
    first_step = int(summary_df["checkpoint_step"].min())
    reasons: list[str] = []
    if int(summary_df["nan_or_crash"].sum()) > 0:
        return "phase2_no_go_eval_invalid_or_missing_metrics", {"nan_or_crash": int(summary_df["nan_or_crash"].sum())}, reasons

    final = summary_df[summary_df["checkpoint_step"] == max_step]
    if final.empty:
        return "phase2_no_go_eval_invalid_or_missing_metrics", {"missing_final_step": max_step}, reasons
    if float(final["progress"].mean()) < 0.20 and float(final["success_rate"].mean()) < 0.05:
        return "phase2_no_go_baseline_cannot_learn_at_all", {
            "final_progress_mean": float(final["progress"].mean()),
            "final_success_mean": float(final["success_rate"].mean()),
        }, reasons

    no_resp = metric_delta(summary_df, "no_response_rate", first_step, max_step)
    near = metric_delta(summary_df, "near_miss_rate", first_step, max_step)
    min_dist = metric_delta(summary_df, "mean_min_distance", first_step, max_step)
    threat_dist = metric_delta(summary_df, "min_distance_after_threat", first_step, max_step)
    coll = metric_delta(summary_df, "collision_rate", first_step, max_step)

    scenario_hits: set[str] = set()
    for _, row in no_resp.iterrows():
        if float(row["delta"]) >= 0.20 or float(row["end"]) >= 0.20:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: no_response delta={row['delta']:.3f} end={row['end']:.3f}")
    for _, row in near.iterrows():
        if float(row["delta"]) >= 0.20:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: near_miss delta={row['delta']:.3f}")
    for _, row in min_dist.iterrows():
        if float(row["delta"]) <= -0.30:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: mean_min_distance drop={row['delta']:.3f}")
    for _, row in threat_dist.iterrows():
        if float(row["delta"]) <= -0.30:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: min_distance_after_threat drop={row['delta']:.3f}")
    for _, row in coll.iterrows():
        if float(row["delta"]) >= 0.05:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: collision delta={row['delta']:.3f}")

    peak_no_resp = summary_df.groupby("scenario")["no_response_rate"].max().reset_index()
    for _, row in peak_no_resp.iterrows():
        if float(row["no_response_rate"]) >= 0.20:
            scenario_hits.add(str(row["scenario"]))
            reasons.append(f"{row['scenario']}: no_response peak={row['no_response_rate']:.3f}")

    metrics = {
        "max_step": int(max_step),
        "first_step": int(first_step),
        "hit_scenarios": sorted(scenario_hits),
        "hit_scenario_count": len(scenario_hits),
        "mean_final_success_rate": float(final["success_rate"].mean()),
        "mean_final_progress": float(final["progress"].mean()),
        "peak_no_response_rate": float(summary_df["no_response_rate"].max()),
        "peak_near_miss_rate": float(summary_df["near_miss_rate"].max()),
        "min_mean_min_distance": float(summary_df["mean_min_distance"].min()),
    }
    if len(scenario_hits) >= 2 and not (scenario_hits <= {"eval_flow_sudden_threat"}):
        return "phase2_strong_reproduction_go_phase3", metrics, reasons
    if len(scenario_hits) >= 1:
        return "phase2_weak_reproduction_go_phase3_with_limited_claim", metrics, reasons
    if max_step >= 1500000:
        return "phase2_no_go_no_reproduction", metrics, reasons
    return "needs_extension", metrics, reasons


def plot_metric(summary_df: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for scenario, group in summary_df.groupby("scenario"):
        group = group.sort_values("checkpoint_step")
        plt.plot(group["checkpoint_step"], group[metric], marker="o", label=scenario)
    plt.xlabel("checkpoint step")
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_collision_success(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for scenario, group in summary_df.groupby("scenario"):
        group = group.sort_values("checkpoint_step")
        plt.plot(group["checkpoint_step"], group["success_rate"], marker="o", label=f"{scenario} success")
        plt.plot(group["checkpoint_step"], group["collision_rate"], marker="x", linestyle="--", label=f"{scenario} collision")
    plt.xlabel("checkpoint step")
    plt.ylabel("rate")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_reaction_breakdown(summary_df: pd.DataFrame, path: Path) -> None:
    aggregate = summary_df.groupby("checkpoint_step").agg(
        reaction_time_eval_style=("reaction_time_eval_style", "mean"),
        conditional_reaction_time=("conditional_reaction_time", "mean"),
        no_response_rate=("no_response_rate", "mean"),
    ).reset_index()
    plt.figure(figsize=(7, 4.5))
    plt.plot(aggregate["checkpoint_step"], aggregate["reaction_time_eval_style"], marker="o", label="eval-style reaction")
    plt.plot(aggregate["checkpoint_step"], aggregate["conditional_reaction_time"], marker="o", label="conditional reaction")
    plt.plot(aggregate["checkpoint_step"], aggregate["no_response_rate"], marker="x", label="no_response_rate")
    plt.xlabel("checkpoint step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_heatmap(summary_df: pd.DataFrame, path: Path) -> None:
    pivot = summary_df.pivot_table(index="scenario", columns="checkpoint_step", values="no_response_rate", aggfunc="mean")
    plt.figure(figsize=(7, 4))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=30)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="no_response_rate")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def generate_outputs(steps: list[int], decision: str, decision_metrics: dict[str, Any], reasons: list[str], args: argparse.Namespace) -> None:
    episode_df = load_episode_metrics(steps)
    summary_df = summarize_episode_metrics(episode_df)
    reaction_df = reaction_breakdown(summary_df)
    threat_df = threat_metrics(episode_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(OUT_DIR / "baseline_longtrain_episode_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "baseline_longtrain_by_checkpoint_scenario.csv", index=False)
    reaction_df.to_csv(OUT_DIR / "baseline_longtrain_reaction_breakdown.csv", index=False)
    threat_df.to_csv(OUT_DIR / "baseline_longtrain_threat_metrics.csv", index=False)

    plot_metric(summary_df, "no_response_rate", PLOTS_DIR / "no_response_curve.png", "no_response_rate")
    plot_metric(summary_df, "near_miss_rate", PLOTS_DIR / "near_miss_curve.png", "near_miss_rate")
    plot_metric(summary_df, "mean_min_distance", PLOTS_DIR / "min_distance_curve.png", "mean_min_distance")
    plot_collision_success(summary_df, PLOTS_DIR / "collision_success_curve.png")
    plot_reaction_breakdown(summary_df, PLOTS_DIR / "reaction_breakdown_curve.png")
    plot_heatmap(summary_df, PLOTS_DIR / "scenario_metric_heatmap.png")

    write_reports(summary_df, reaction_df, decision, decision_metrics, reasons, args, max(steps))
    write_flag(decision)


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def write_reports(
    summary_df: pd.DataFrame,
    reaction_df: pd.DataFrame,
    decision: str,
    metrics: dict[str, Any],
    reasons: list[str],
    args: argparse.Namespace,
    max_step: int,
) -> None:
    complete = decision in REPRO_DECISIONS
    if "checkpoint_step" in summary_df.columns and not summary_df.empty:
        final_rows = summary_df[summary_df["checkpoint_step"] == max_step]
        final_success = safe_mean(final_rows["success_rate"]) if "success_rate" in final_rows else float("nan")
        final_progress = safe_mean(final_rows["progress"]) if "progress" in final_rows else float("nan")
    else:
        final_success = float("nan")
        final_progress = float("nan")
    env_report = [
        "# Baseline Longtrain Env V2 Report",
        "",
        "## Setup",
        "",
        "- method: attention_full",
        "- seed: 0",
        "- env: DynamicObstacleFlowEnv",
        "- train scenario: train_flow_mixed",
        f"- trained steps: {max_step}",
        f"- n_envs: {args.n_envs}",
        "- safety cost: disabled",
        "- PPO-Lagrangian/adaptive lambda/new risk formula: not used",
        "",
        "## Metrics Note",
        "",
        "`reaction_time_eval_style` uses a timeout penalty for no-response episodes. It can rise because `no_response_rate` rises and must not be interpreted as every episode having slower physical reaction latency.",
        "`reaction_time_nan_style` / `conditional_reaction_time` only averages episodes where the action-based response criterion fired.",
    ]
    (ROOT / "BASELINE_LONGTRAIN_ENV_V2_REPORT.md").write_text("\n".join(env_report) + "\n", encoding="utf-8")

    lines = [
        "# Phase 2 Baseline Long-Training Reproduction Report",
        "",
        "## 1. Executive Summary",
        "",
        f"terminal_decision = {decision}",
        f"next_recommended_phase = {'Phase 3 failure localization' if complete else 'blocked'}",
        "",
        "## 2. Training Setup",
        "",
        f"attention_full seed=0 trained on DynamicObstacleFlowEnv/train_flow_mixed to {max_step} steps.",
        "Checkpoints are saved under `checkpoints/env_v2_phase2/attention_full_s0`.",
        "",
        "## 3. Evaluation Setup",
        "",
        f"Scenarios: {', '.join(SCENARIOS)}. Episodes per checkpoint/scenario: {args.episodes}. eval_seed=1000.",
        "",
        "## 4. Learning Sanity",
        "",
        f"Final mean success_rate={fmt(final_success)}; final mean progress={fmt(final_progress)}.",
        "",
        "## 5. Long-Training Curves",
        "",
        "| step | scenario | success | collision | near_miss | no_response | mean_min_distance | min_distance_after_threat |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not summary_df.empty and {"checkpoint_step", "scenario"}.issubset(summary_df.columns):
        summary_records = summary_df.sort_values(["checkpoint_step", "scenario"]).to_dict("records")
    else:
        summary_records = []
    for row in summary_records:
        lines.append(
            f"| {int(row['checkpoint_step'])} | {row['scenario']} | {fmt(row['success_rate'])} | "
            f"{fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | {fmt(row['no_response_rate'])} | "
            f"{fmt(row['mean_min_distance'])} | {fmt(row['min_distance_after_threat'])} |"
        )

    lines.extend(
        [
            "",
            "## 6. Reaction Metric Breakdown",
            "",
            "The no-response metric is action-based: after a planned threat window begins, the policy must emit continuous lateral/away-from-threat response. It is not physical completion latency.",
            "",
            "| step | scenario | eval_style | conditional | nan_reaction_rate | no_response_rate |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if not reaction_df.empty and {"checkpoint_step", "scenario"}.issubset(reaction_df.columns):
        reaction_records = reaction_df.sort_values(["checkpoint_step", "scenario"]).to_dict("records")
    else:
        reaction_records = []
    for row in reaction_records:
        lines.append(
            f"| {int(row['checkpoint_step'])} | {row['scenario']} | {fmt(row['reaction_time_eval_style'])} | "
            f"{fmt(row['conditional_reaction_time'])} | {fmt(row['nan_reaction_rate'])} | {fmt(row['no_response_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 7. Scenario-Wise Findings",
            "",
        ]
    )
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- No scenario met the configured reproduction thresholds.")

    lines.extend(
        [
            "",
            "## 8. Reproduction Decision",
            "",
        ]
    )
    if complete:
        lines.append(f"Decision: {decision}. Phase 3 may localize failure mechanisms.")
    else:
        lines.extend(
            [
                f"NO-GO triggered: {decision}",
                f"Triggering metrics: {metrics}",
                "Cannot enter Phase 3 because Phase 2 did not establish a reproducible baseline degradation signal.",
            ]
        )

    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            "If reproduction is strong or weak, the result supports investigating no-response / safety-margin erosion in Env V2. Claims remain bounded by seed=0 and the scenario coverage above.",
            "",
            "## 10. Next Step",
            "",
            "Enter Phase 3 failure localization if complete; otherwise fix training/eval/env or re-evaluate the direction according to the no-go reason.",
        ]
    )
    (ROOT / "PHASE2_BASELINE_LONGTRAIN_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flag(decision: str) -> None:
    complete_flag = ROOT / "PHASE2_BASELINE_LONGTRAIN_COMPLETE.flag"
    no_go_flag = ROOT / "PHASE2_BASELINE_LONGTRAIN_NO_GO.flag"
    if decision in REPRO_DECISIONS:
        if no_go_flag.exists():
            no_go_flag.unlink()
        complete_flag.write_text(
            f"terminal_decision={decision}\nnext_recommended_phase=Phase 3 failure localization\n",
            encoding="utf-8",
        )
    elif decision in NO_GO_DECISIONS:
        if complete_flag.exists():
            complete_flag.unlink()
        action = "fix training/eval/env or re-evaluate research direction"
        no_go_flag.write_text(
            f"terminal_decision={decision}\nnext_recommended_action={action}\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"cannot write terminal flag for non-terminal decision: {decision}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train_heartbeat_seconds", type=float, default=30.0)
    parser.add_argument("--eval_heartbeat_seconds", type=float, default=20.0)
    parser.add_argument("--force_1500k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, f"pid={os.getpid()}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode("utf-8"))
    os.close(lock_fd)
    try:
        required_phase1 = [
            ROOT / "PHASE0_PHASE1_COMPLETE.flag",
            ROOT / "PHASE0_PHASE1_FINAL_REPORT.md",
            ROOT / "ENV_V2_SANITY_REPORT.md",
            ROOT / "results/restart_phase0_phase1/env_v2/env_v2_sanity.csv",
        ]
        missing = [rel(path) for path in required_phase1 if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing Phase 1 artifacts: {missing}")

        train_to_1000k(args)
        evaluate_steps(FIRST_STEPS, args)
        episode_df = load_episode_metrics(FIRST_STEPS)
        summary_df = summarize_episode_metrics(episode_df)
        decision, metrics, reasons = decide_reproduction(summary_df, 1000000)
        if decision == "needs_extension" or args.force_1500k:
            write_status("[stage] 1000k did not meet reproduction threshold; extending to 1500k")
            train_to_1500k(args)
            evaluate_steps(EXTENSION_STEPS, args)
            all_steps = FIRST_STEPS + EXTENSION_STEPS
            episode_df = load_episode_metrics(all_steps)
            summary_df = summarize_episode_metrics(episode_df)
            decision, metrics, reasons = decide_reproduction(summary_df, 1500000)
            if decision == "needs_extension":
                decision = "phase2_no_go_no_reproduction"
        else:
            all_steps = FIRST_STEPS

        generate_outputs(all_steps, decision, metrics, reasons, args)
        write_status(f"[done] terminal_decision={decision}")
    except subprocess.CalledProcessError as exc:
        decision = "phase2_no_go_training_crash_or_nan"
        write_reports(pd.DataFrame(), pd.DataFrame(), decision, {"cmd": exc.cmd, "returncode": exc.returncode}, [], args, 0)
        write_flag(decision)
        raise
    except Exception as exc:
        decision = "phase2_no_go_env_incompatible_with_training"
        write_reports(pd.DataFrame(), pd.DataFrame(), decision, {"error": repr(exc)}, [], args, 0)
        write_flag(decision)
        raise
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
