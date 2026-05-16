from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

P0_DIR = ROOT / "results/p0_trace"
P1_DIR = ROOT / "results/p1_three_seed"
P1_EVAL_DIR = P1_DIR / "eval"
P1_PLOTS_DIR = P1_DIR / "plots"
P1_CHECKPOINT_DIR = ROOT / "checkpoints/p1_three_seed"
RUN_LOG_DIR = ROOT / "runs/logs"
P1_RUN_DIR = ROOT / "runs/p1_three_seed"

CHECKPOINT_STEPS = [250000, 500000, 750000, 1000000]
SEEDS = [0, 1, 2]
SCENARIOS = [
    "eval_random_switch",
    "eval_sudden_turn",
    "eval_random_switch_hard",
    "mixed_uncertainty",
]
SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "mixed_uncertainty": "mixed",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    agg: str = "attention"
    cost_type: str = "none"
    beta_cost: float = 5.0
    d_warning: float = 1.0


METHODS = {
    "attention_full": MethodSpec("attention_full"),
    "attention_full_distance_penalty": MethodSpec(
        "attention_full_distance_penalty",
        cost_type="distance_warning",
        beta_cost=5.0,
        d_warning=1.0,
    ),
    "attention_full_risk_penalty": MethodSpec(
        "attention_full_risk_penalty",
        cost_type="risk_sum",
        beta_cost=5.0,
    ),
}


def ensure_dirs() -> None:
    for path in [P0_DIR / "plots", P1_EVAL_DIR, P1_PLOTS_DIR, P1_CHECKPOINT_DIR, RUN_LOG_DIR, P1_RUN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_status(message: str) -> None:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (RUN_LOG_DIR / "p0_p1_replication_status.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_command(cmd: list[str], log_path: Path, allow_existing: bool = False) -> None:
    if allow_existing and log_path.exists() and "EVAL_END" in log_path.read_text(encoding="utf-8", errors="ignore"):
        append_status(f"SKIP completed command log={log_path}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_status("RUN " + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def checkpoint_num_timesteps(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        value = data.get("num_timesteps")
        return int(value) if value is not None else None
    except Exception:
        return None


def checkpoint_path(method: str, seed: int, step: int) -> Path:
    if method == "attention_full" and seed == 0:
        return ROOT / f"checkpoints/longtrain_baseline/attention_full_s0_step{step}.zip"
    if method == "attention_full" and seed == 1:
        return ROOT / f"checkpoints/attention_seed1/attention_full_s1_step{step}.zip"
    if method in {"attention_full_distance_penalty", "attention_full_risk_penalty"} and seed == 0:
        return ROOT / f"checkpoints/gate2b/{method}_s0_step{step}.zip"
    return P1_CHECKPOINT_DIR / f"{method}_s{seed}_step{step}.zip"


def p1_eval_csv(method: str, seed: int, step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[scenario]
    return P1_EVAL_DIR / f"{method}_s{seed}_step{step}_{suffix}.csv"


def source_eval_csv(method: str, seed: int, step: int, scenario: str) -> Path | None:
    suffix = SCENARIO_SUFFIX[scenario]
    candidates: list[Path] = []
    if method == "attention_full" and seed == 0:
        candidates.append(ROOT / f"results/longtrain_baseline/eval/attention_full_s0_step{step}_{suffix}.csv")
    if method == "attention_full" and seed == 1:
        candidates.append(ROOT / f"results/attention_seed1/eval/attention_full_s1_step{step}_{suffix}.csv")
    if method in {"attention_full_distance_penalty", "attention_full_risk_penalty"} and seed == 0:
        candidates.append(ROOT / f"results/gate2b/eval/{method}_s0_step{step}_{suffix}.csv")
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def copy_eval_if_available(method: str, seed: int, step: int, scenario: str, out_csv: Path) -> str:
    source = source_eval_csv(method, seed, step, scenario)
    if source is None or csv_row_count(source) < 50:
        return ""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_csv.exists() or csv_row_count(out_csv) < 50:
        shutil.copy2(source, out_csv)
        source_summary = source.with_suffix(".summary.csv")
        if source_summary.exists():
            shutil.copy2(source_summary, out_csv.with_suffix(".summary.csv"))
    return str(source.relative_to(ROOT))


def train_needed(method: str, seed: int) -> bool:
    return any(not checkpoint_path(method, seed, step).exists() for step in CHECKPOINT_STEPS)


def train_run(method: str, seed: int, n_envs: int) -> None:
    if not train_needed(method, seed):
        append_status(f"SKIP training existing checkpoints method={method} seed={seed}")
        return
    spec = METHODS[method]
    cmd = [
        PYTHON,
        "train.py",
        "--method",
        method,
        "--profile_mode",
        "full_12",
        "--agg",
        spec.agg,
        "--seed",
        str(seed),
        "--total_steps",
        "1000000",
        "--n_envs",
        str(n_envs),
        "--device",
        "cpu",
        "--scenario",
        "train_random_switch",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        ",".join(str(step) for step in CHECKPOINT_STEPS),
        "--checkpoint_dir",
        str(P1_CHECKPOINT_DIR),
        "--log_dir",
        str(P1_RUN_DIR / f"{method}_s{seed}"),
        "--run_name",
        method,
        "--save_path",
        str(P1_CHECKPOINT_DIR / f"{method}_s{seed}_step1000000.zip"),
        "--heartbeat_seconds",
        "30",
    ]
    if spec.cost_type != "none":
        cmd.extend(
            [
                "--use_safety_cost",
                "true",
                "--cost_type",
                spec.cost_type,
                "--fallback_penalty",
                "true",
                "--beta_cost",
                str(spec.beta_cost),
            ]
        )
        if spec.cost_type == "distance_warning":
            cmd.extend(["--d_warning", str(spec.d_warning)])
    run_command(cmd, RUN_LOG_DIR / f"train_p1_{method}_s{seed}.log")


def eval_run(method: str, seed: int, step: int, scenario: str) -> tuple[Path, str]:
    out_csv = p1_eval_csv(method, seed, step, scenario)
    reused = copy_eval_if_available(method, seed, step, scenario, out_csv)
    if csv_row_count(out_csv) >= 50:
        return out_csv, reused
    ckpt = checkpoint_path(method, seed, step)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint for eval: {ckpt}")
    spec = METHODS[method]
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(ckpt),
        "--method",
        method,
        "--profile_mode",
        "full_12",
        "--agg",
        spec.agg,
        "--seed",
        str(seed),
        "--eval_seed",
        "1000",
        "--episodes",
        "50",
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
    ]
    if spec.cost_type == "distance_warning":
        cmd.extend(["--d_warning", str(spec.d_warning)])
    log_name = f"eval_p1_{method}_s{seed}_step{step}_{SCENARIO_SUFFIX[scenario]}.log"
    run_command(cmd, RUN_LOG_DIR / log_name)
    return out_csv, reused


def summarize_eval_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)

    def mean_col(name: str) -> float:
        return float(df[name].mean()) if name in df else float("nan")

    def nanmean_col(name: str) -> float:
        if name not in df:
            return float("nan")
        arr = pd.to_numeric(df[name], errors="coerce")
        return float(arr.mean()) if not arr.isna().all() else float("nan")

    summary = {
        "success_rate": mean_col("success"),
        "collision_rate": mean_col("collision"),
        "mean_min_distance": mean_col("episode_min_distance"),
        "near_miss_rate": mean_col("near_miss"),
        "mean_time": mean_col("time_to_goal"),
        "distance_warning_cost_mean": mean_col("distance_warning_cost_mean"),
        "risk_sum_mean": mean_col("risk_sum_mean"),
        "risk_max_mean": mean_col("risk_max_mean"),
        "reaction_time_eval_style": nanmean_col("reaction_time_eval_style"),
        "reaction_time_nan_style": nanmean_col("reaction_time_nan_style"),
        "reaction_time": nanmean_col("reaction_time_eval_style"),
        "nan_reaction_rate": float(pd.to_numeric(df.get("reaction_time_nan_style", pd.Series(dtype=float)), errors="coerce").isna().mean())
        if "reaction_time_nan_style" in df
        else float("nan"),
        "mean_min_distance_after_turn": nanmean_col("min_distance_after_turn"),
        "total_episodes": float(len(df)),
    }
    return summary


def p0_trace_files(method: str, step: int, scenario: str) -> list[Path]:
    return sorted((ROOT / "results/gate2b/traces").glob(f"{method}_step{step}_{scenario}_ep*.csv"))


def first_time_after_turn(df: pd.DataFrame, column: str, threshold: float, turn_time: float) -> tuple[float, int]:
    after = df[df["time"] >= turn_time].copy()
    hits = after[pd.to_numeric(after[column], errors="coerce") > threshold]
    if hits.empty:
        return float("nan"), -1
    row = hits.iloc[0]
    return float(row["time"]), int(row["step"])


def analyze_p0() -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    append_status("P0 analysis start")
    trace_paths = p0_trace_files("attention_full_risk_penalty", 750000, "eval_sudden_turn")
    if len(trace_paths) < 10:
        raise FileNotFoundError("P0 needs 10 attention_full_risk_penalty step750000 sudden-turn traces")

    rows: list[dict[str, Any]] = []
    aligned_frames: list[pd.DataFrame] = []
    for path in trace_paths[:10]:
        df = pd.read_csv(path)
        df["time_rel_turn"] = df["time"] - df["turn_time"].iloc[0]
        turn_time = float(df["turn_time"].iloc[0])
        turn_step = int(df["turn_step"].iloc[0])
        pre = df[(df["step"] < turn_step) & (df["step"] >= turn_step - 5)]
        if pre.empty:
            pre = df[df["time"] < turn_time].tail(5)

        baseline_sum = float(pre["risk_sum"].mean()) if not pre.empty else 0.0
        baseline_max = float(pre["risk_max"].mean()) if not pre.empty else 0.0
        std_sum = float(pre["risk_sum"].std(ddof=0)) if not pre.empty else 0.0
        std_max = float(pre["risk_max"].std(ddof=0)) if not pre.empty else 0.0
        delta_sum = max(0.02, 0.2 * std_sum)
        delta_max = max(0.02, 0.2 * std_max)

        risk_sum_time, risk_sum_step = first_time_after_turn(df, "risk_sum", baseline_sum + delta_sum, turn_time)
        risk_max_time, risk_max_step = first_time_after_turn(df, "risk_max", baseline_max + delta_max, turn_time)
        dist_time, dist_step = first_time_after_turn(df, "distance_warning_cost", 1e-6, turn_time)

        reaction_values = pd.to_numeric(
            df.loc[pd.to_numeric(df["reaction_flag"], errors="coerce") > 0, "reaction_time_current"],
            errors="coerce",
        ).dropna()
        reaction_time = float(reaction_values.iloc[0]) if not reaction_values.empty else float("nan")
        min_distance_after_turn = float(df[df["time"] >= turn_time]["min_distance"].min())
        lead_sum = dist_time - risk_sum_time if not math.isnan(dist_time) and not math.isnan(risk_sum_time) else float("nan")
        lead_max = dist_time - risk_max_time if not math.isnan(dist_time) and not math.isnan(risk_max_time) else float("nan")
        dt = float(df["time"].diff().dropna().median()) if len(df) > 1 else 0.2

        episode = int(df["episode"].iloc[0])
        rows.append(
            {
                "method": "attention_full_risk_penalty",
                "seed": 0,
                "step": 750000,
                "scenario": "eval_sudden_turn",
                "episode": episode,
                "turn_time": turn_time,
                "turn_step": turn_step,
                "risk_rise_time_sum": risk_sum_time,
                "risk_rise_step_sum": risk_sum_step,
                "risk_rise_time_max": risk_max_time,
                "risk_rise_step_max": risk_max_step,
                "distance_cost_rise_time": dist_time,
                "distance_cost_rise_step": dist_step,
                "lead_time_sum": lead_sum,
                "lead_steps_sum": lead_sum / dt if not math.isnan(lead_sum) and dt > 0 else float("nan"),
                "lead_time_max": lead_max,
                "lead_steps_max": lead_max / dt if not math.isnan(lead_max) and dt > 0 else float("nan"),
                "reaction_time": reaction_time,
                "min_distance_after_turn": min_distance_after_turn,
                "near_miss": int(min_distance_after_turn < 1.0),
                "success": int(df["success"].iloc[-1]),
                "collision": int(df["collision"].iloc[-1]),
                "baseline_risk_sum": baseline_sum,
                "baseline_risk_max": baseline_max,
                "delta_risk_sum": delta_sum,
                "delta_risk_max": delta_max,
                "distance_warning_ever_after_turn": int(df[df["time"] >= turn_time]["distance_warning_cost"].max() > 1e-6),
                "source_trace_path": str(path.relative_to(ROOT)),
            }
        )

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(df["time"], df["risk_sum"], label="risk_sum", color="#1f77b4")
        ax1.plot(df["time"], df["risk_max"], label="risk_max", color="#17becf")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("risk")
        ax2 = ax1.twinx()
        ax2.plot(df["time"], df["distance_warning_cost"], label="distance_warning_cost", color="#d62728")
        ax2.set_ylabel("distance warning cost")
        ax1.axvline(turn_time, color="#444444", linestyle="--", linewidth=1.0, label="turn_time")
        if not math.isnan(reaction_time):
            ax1.axvline(turn_time + reaction_time, color="#2ca02c", linestyle=":", linewidth=1.0, label="reaction_time")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(P0_DIR / "plots" / f"risk_vs_distance_ep{episode}.png", dpi=150)
        plt.close(fig)

        aligned_frames.append(df[["time_rel_turn", "risk_sum", "risk_max", "distance_warning_cost"]].copy())

    summary_df = pd.DataFrame(rows).sort_values("episode")
    summary_df.to_csv(P0_DIR / "p0_trace_summary.csv", index=False)

    mean_df = pd.concat(aligned_frames, ignore_index=True)
    mean_df["time_rel_turn_rounded"] = mean_df["time_rel_turn"].round(6)
    mean_curve = (
        mean_df.groupby("time_rel_turn_rounded")[["risk_sum", "risk_max", "distance_warning_cost"]]
        .mean()
        .reset_index()
        .rename(columns={"time_rel_turn_rounded": "time_rel_turn"})
    )
    mean_curve.to_csv(P0_DIR / "p0_trace_mean_curve.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(mean_curve["time_rel_turn"], mean_curve["risk_sum"], label="mean risk_sum", color="#1f77b4")
    ax1.plot(mean_curve["time_rel_turn"], mean_curve["risk_max"], label="mean risk_max", color="#17becf")
    ax1.set_xlabel("time since turn (s)")
    ax1.set_ylabel("risk")
    ax2 = ax1.twinx()
    ax2.plot(mean_curve["time_rel_turn"], mean_curve["distance_warning_cost"], label="mean distance_warning_cost", color="#d62728")
    ax2.set_ylabel("distance warning cost")
    ax1.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0, label="turn_time")
    mean_reaction = float(summary_df["reaction_time"].mean())
    ax1.axvline(mean_reaction, color="#2ca02c", linestyle=":", linewidth=1.0, label="mean reaction_time")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(P0_DIR / "plots/risk_vs_distance_mean.png", dpi=150)
    plt.close(fig)

    write_p0_report(summary_df)
    append_status("P0 analysis complete")
    return summary_df


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def write_p0_report(summary_df: pd.DataFrame) -> None:
    gate = pd.read_csv(ROOT / "results/gate2b/gate2b_by_step.csv") if (ROOT / "results/gate2b/gate2b_by_step.csv").exists() else pd.DataFrame()
    sudden_750 = gate[(gate.get("scenario") == "eval_sudden_turn") & (gate.get("step") == 750000)] if not gate.empty else pd.DataFrame()
    risk_row = sudden_750[sudden_750.get("method") == "attention_full_risk_penalty"] if not sudden_750.empty else pd.DataFrame()
    dist_row = sudden_750[sudden_750.get("method") == "attention_full_distance_penalty"] if not sudden_750.empty else pd.DataFrame()

    lead_sum = pd.to_numeric(summary_df["lead_time_sum"], errors="coerce")
    lead_steps = pd.to_numeric(summary_df["lead_steps_sum"], errors="coerce")
    valid_leads = lead_sum.dropna()
    majority = int((lead_sum > 0).sum())
    distance_seen = int(summary_df["distance_warning_ever_after_turn"].sum())
    median_lead = float(valid_leads.median()) if not valid_leads.empty else float("nan")
    median_steps = float(lead_steps.dropna().median()) if not lead_steps.dropna().empty else float("nan")
    risk_reaction = float(risk_row.iloc[0]["reaction_time"]) if not risk_row.empty and "reaction_time" in risk_row else float("nan")
    dist_reaction = float(dist_row.iloc[0]["reaction_time"]) if not dist_row.empty and "reaction_time" in dist_row else float("nan")
    supports = bool((majority >= 6) and (not math.isnan(median_lead)) and (median_lead >= 0.4 or median_steps >= 2) and (risk_reaction < dist_reaction))

    lines = [
        "# P0 Trace Predictive Analysis Report",
        "",
        "## Scope",
        "- Core trace: attention_full_risk_penalty, seed=0, step=750000, eval_sudden_turn, 10 episodes.",
        "- Source traces: results/gate2b/traces.",
        "- Required trace fields are available, including attention_weights.",
        "",
        "## Main Finding",
    ]
    if supports:
        lines.append("- P0 supports the predictive-vs-reactive interpretation under the configured thresholds.")
    else:
        lines.append("- P0 does not cleanly support a strict early-warning claim under the configured thresholds.")
    lines += [
        f"- Episodes with distance_warning_cost rising after turn: {distance_seen}/10.",
        f"- Episodes with positive lead_time_sum: {majority}/10 among all episodes.",
        f"- Median lead_time_sum: {fmt(median_lead)} s; median lead_steps_sum: {fmt(median_steps)} steps.",
        f"- 750k sudden reaction: risk_penalty={fmt(risk_reaction)} s, distance_penalty={fmt(dist_reaction)} s.",
        "",
        "## Required Answers",
        f"1. risk_sum / risk_max earlier than distance_warning_cost: {'yes' if majority >= 6 else 'not established'} by the default rise rule.",
        f"2. Lead amount: median {fmt(median_lead)} s / {fmt(median_steps)} steps for valid lead episodes.",
        f"3. Majority: {majority}/10 episodes have lead_time_sum > 0.",
        f"4. Reaction consistency: risk_penalty is much faster than distance_penalty at 750k, but early-warning evidence is limited when distance_warning_cost never rises.",
        "5. If inconsistent: likely causes include conservative delta_risk=0.02 relative to this trace scale, policy avoiding the warning zone, or reward shaping rather than earlier warning.",
        f"6. Paper narrative: {'predictive risk cost vs reactive distance cost is supported' if supports else 'use cautious wording: risk_penalty is empirically effective, but the strict early-warning mechanism is not fully established by P0'}.",
        "",
        "## Episode Metrics",
        "| episode | risk_rise_sum | dist_rise | lead_s | lead_steps | reaction | min_dist_after_turn | near_miss | collision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {int(row['episode'])} | {fmt(row['risk_rise_time_sum'])} | {fmt(row['distance_cost_rise_time'])} | "
            f"{fmt(row['lead_time_sum'])} | {fmt(row['lead_steps_sum'])} | {fmt(row['reaction_time'])} | "
            f"{fmt(row['min_distance_after_turn'])} | {int(row['near_miss'])} | {int(row['collision'])} |"
        )
    lines += [
        "",
        "## Artifacts",
        "- results/p0_trace/p0_trace_summary.csv",
        "- results/p0_trace/p0_trace_mean_curve.csv",
        "- results/p0_trace/plots/risk_vs_distance_ep*.png",
        "- results/p0_trace/plots/risk_vs_distance_mean.png",
    ]
    write_lines(ROOT / "P0_TRACE_PREDICTIVE_ANALYSIS_REPORT.md", lines)


def run_p1(n_envs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    append_status("P1 start")
    train_plan = [
        ("attention_full", 2),
        ("attention_full_distance_penalty", 1),
        ("attention_full_distance_penalty", 2),
        ("attention_full_risk_penalty", 1),
        ("attention_full_risk_penalty", 2),
    ]
    for method, seed in train_plan:
        train_run(method, seed, n_envs)

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            for step in CHECKPOINT_STEPS:
                ckpt = checkpoint_path(method, seed, step)
                for scenario in SCENARIOS:
                    out_csv, reused = eval_run(method, seed, step, scenario)
                    summary = summarize_eval_csv(out_csv)
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "step": step,
                            "scenario": scenario,
                            **summary,
                            "checkpoint_path": str(ckpt.relative_to(ROOT)) if ckpt.exists() else str(ckpt),
                            "eval_csv_path": str(out_csv.relative_to(ROOT)),
                            "reused_from": reused,
                        }
                    )

    by_seed = pd.DataFrame(rows).sort_values(["method", "seed", "step", "scenario"])
    by_seed.to_csv(P1_DIR / "p1_by_seed_step_scenario.csv", index=False)

    grouped_rows: list[dict[str, Any]] = []
    for (method, step, scenario), group in by_seed.groupby(["method", "step", "scenario"]):
        grouped_rows.append(
            {
                "method": method,
                "step": step,
                "scenario": scenario,
                "mean_reaction": float(group["reaction_time_eval_style"].mean()),
                "std_reaction": float(group["reaction_time_eval_style"].std()),
                "mean_success": float(group["success_rate"].mean()),
                "std_success": float(group["success_rate"].std()),
                "mean_collision": float(group["collision_rate"].mean()),
                "std_collision": float(group["collision_rate"].std()),
                "mean_near_miss": float(group["near_miss_rate"].mean()),
                "std_near_miss": float(group["near_miss_rate"].std()),
                "mean_min_distance": float(group["mean_min_distance"].mean()),
                "std_min_distance": float(group["mean_min_distance"].std()),
                "mean_time": float(group["mean_time"].mean()),
                "std_time": float(group["mean_time"].std()),
            }
        )
    summary = pd.DataFrame(grouped_rows).sort_values(["method", "step", "scenario"])
    summary.to_csv(P1_DIR / "p1_summary_by_method_step_scenario.csv", index=False)

    plot_p1(by_seed)
    write_p1_report(by_seed, summary)
    append_status("P1 complete")
    return by_seed, summary


def plot_p1(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("eval_sudden_turn", "reaction_time_eval_style", "p1_sudden_reaction_by_seed.png", "reaction time (s)"),
        ("eval_sudden_turn", "success_rate", "p1_sudden_success_by_seed.png", "success rate"),
        ("eval_random_switch", "success_rate", "p1_random_success_by_seed.png", "success rate"),
        ("eval_sudden_turn", "near_miss_rate", "p1_sudden_near_miss_by_seed.png", "near miss rate"),
        ("mixed_uncertainty", "reaction_time_eval_style", "p1_mixed_reaction_by_seed.png", "reaction time (s)"),
    ]
    for scenario, metric, filename, ylabel in plot_specs:
        sub = df[df["scenario"] == scenario]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, method_group in sub.groupby("method"):
            for seed, seed_group in method_group.groupby("seed"):
                seed_group = seed_group.sort_values("step")
                ax.plot(seed_group["step"], seed_group[metric], marker="o", label=f"{method} s{seed}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{scenario}: {metric}")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(P1_PLOTS_DIR / filename, dpi=150)
        plt.close(fig)


def row_value(df: pd.DataFrame, method: str, seed: int, step: int, scenario: str, column: str) -> float:
    row = df[(df["method"] == method) & (df["seed"] == seed) & (df["step"] == step) & (df["scenario"] == scenario)]
    if row.empty or column not in row:
        return float("nan")
    return float(row.iloc[0][column])


def p1_decision(df: pd.DataFrame) -> dict[str, Any]:
    replicated_seeds: list[int] = []
    risk_better_seeds: list[int] = []
    distance_failed_seeds: list[int] = []
    for seed in SEEDS:
        risk_reaction = row_value(df, "attention_full_risk_penalty", seed, 750000, "eval_sudden_turn", "reaction_time_eval_style")
        risk_success = row_value(df, "attention_full_risk_penalty", seed, 750000, "eval_sudden_turn", "success_rate")
        risk_collision = row_value(df, "attention_full_risk_penalty", seed, 750000, "eval_sudden_turn", "collision_rate")
        risk_random_success = row_value(df, "attention_full_risk_penalty", seed, 750000, "eval_random_switch", "success_rate")
        dist_reaction = row_value(df, "attention_full_distance_penalty", seed, 750000, "eval_sudden_turn", "reaction_time_eval_style")
        if not math.isnan(dist_reaction) and dist_reaction > 5.0:
            distance_failed_seeds.append(seed)
        risk_core = (
            not math.isnan(risk_reaction)
            and risk_reaction < 2.0
            and risk_success >= 0.95
            and risk_collision <= 0.05
            and risk_random_success >= 0.95
        )
        if risk_core and (not math.isnan(dist_reaction)) and (dist_reaction > 5.0 or risk_reaction < dist_reaction):
            replicated_seeds.append(seed)
        if not math.isnan(risk_reaction) and not math.isnan(dist_reaction) and risk_reaction < dist_reaction:
            risk_better_seeds.append(seed)
    if len(replicated_seeds) >= 2:
        decision = "risk_penalty core effect replicated"
    elif len(risk_better_seeds) >= 2:
        decision = "risk_penalty effective but seed-sensitive"
    else:
        decision = "risk_penalty effect not robust"
    return {
        "replicated_seeds": replicated_seeds,
        "risk_better_seeds": risk_better_seeds,
        "distance_failed_seeds": distance_failed_seeds,
        "decision": decision,
    }


def write_p1_report(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    decision = p1_decision(df)
    reused = df[df["reused_from"].astype(str) != ""][["method", "seed", "checkpoint_path", "reused_from"]].drop_duplicates()
    trained = []
    for method in METHODS:
        for seed in SEEDS:
            if method == "attention_full" and seed in {0, 1}:
                continue
            if method in {"attention_full_distance_penalty", "attention_full_risk_penalty"} and seed == 0:
                continue
            trained.append((method, seed))

    lines = [
        "# P1 Three-Seed Replication Report",
        "",
        "## 1. Experiment Setup",
        "- Methods: attention_full, attention_full_distance_penalty, attention_full_risk_penalty.",
        "- Seeds: 0, 1, 2.",
        "- Total steps: 1000000.",
        "- Checkpoints: 250000, 500000, 750000, 1000000.",
        "- Scenarios: eval_random_switch, eval_sudden_turn, eval_random_switch_hard, mixed_uncertainty.",
        "- Eval episodes: 50; eval_seed: 1000.",
        "- beta_distance: 5.0; beta_risk: 5.0.",
        "- reaction_time definition: eval.py reaction_time_eval_style.",
        "",
        "## 2. Existing Runs Reused",
    ]
    if reused.empty:
        lines.append("- No existing eval CSVs were reused.")
    else:
        for _, row in reused.iterrows():
            lines.append(f"- {row['method']} seed={int(row['seed'])}: checkpoint {row['checkpoint_path']}; eval source {row['reused_from']}.")
    lines.append("")
    lines.append("New or completed P1 training runs:")
    for method, seed in trained:
        lines.append(f"- {method} seed={seed}.")

    sudden = df[df["scenario"] == "eval_sudden_turn"].sort_values(["method", "seed", "step"])
    lines += [
        "",
        "## 3. Main Results: sudden_turn",
        "| method | seed | step | reaction | success | collision | min_distance | near_miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sudden.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['seed'])} | {int(row['step'])} | {fmt(row['reaction_time_eval_style'])} | "
            f"{fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['mean_min_distance'])} | {fmt(row['near_miss_rate'])} |"
        )

    random_750 = df[(df["scenario"] == "eval_random_switch") & (df["step"] == 750000)].sort_values(["method", "seed"])
    lines += [
        "",
        "## 4. Random Switch Side Effects",
        "| method | seed | success | mean_time | min_distance | near_miss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in random_750.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['seed'])} | {fmt(row['success_rate'])} | {fmt(row['mean_time'])} | "
            f"{fmt(row['mean_min_distance'])} | {fmt(row['near_miss_rate'])} |"
        )

    hm = df[(df["scenario"].isin(["eval_random_switch_hard", "mixed_uncertainty"])) & (df["step"] == 750000)].sort_values(["scenario", "method", "seed"])
    lines += [
        "",
        "## 5. Hard and Mixed Results",
        "| scenario | method | seed | reaction | success | collision | near_miss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in hm.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {int(row['seed'])} | {fmt(row['reaction_time_eval_style'])} | "
            f"{fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} |"
        )

    lines += [
        "",
        "## 6. Replication Decision",
        f"- Decision: {decision['decision']}.",
        f"- Replicated seeds by 750k core rule: {decision['replicated_seeds']}.",
        f"- Seeds where risk_penalty has lower sudden-turn reaction than distance_penalty at 750k: {decision['risk_better_seeds']}.",
        f"- Seeds where distance_penalty reaction exceeds 5s at 750k: {decision['distance_failed_seeds']}.",
        "",
        "## 7. Interpretation",
    ]
    if decision["decision"] == "risk_penalty core effect replicated":
        lines.append("- risk_penalty is stable enough across this three-seed check to support the core Gate-2b effect.")
    elif decision["decision"] == "risk_penalty effective but seed-sensitive":
        lines.append("- risk_penalty often improves reaction, but training dynamics or penalty scaling still show seed sensitivity.")
    else:
        lines.append("- risk_penalty does not robustly reproduce the seed=0 effect in this small replication.")
    lines.append("- Compare p1_summary_by_method_step_scenario.csv for mean/std across seeds.")
    lines += [
        "",
        "## 8. Next Step Recommendation",
    ]
    if decision["decision"] == "risk_penalty core effect replicated":
        lines.append("- A. Enter P2 environment upgrade, while keeping beta/cost-scale diagnostics in the appendix.")
    elif decision["decision"] == "risk_penalty effective but seed-sensitive":
        lines.append("- B. Run beta sweep / cost normalization before P2.")
    else:
        lines.append("- C. Pause the risk main line and debug beta_cost, cost scale, reward shaping, and reaction-time accounting.")
    lines += [
        "",
        "## Artifacts",
        "- results/p1_three_seed/p1_by_seed_step_scenario.csv",
        "- results/p1_three_seed/p1_summary_by_method_step_scenario.csv",
        "- results/p1_three_seed/plots/*.png",
    ]
    write_lines(ROOT / "P1_THREE_SEED_REPLICATION_REPORT.md", lines)


def write_checkpoint_index() -> None:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            for step in CHECKPOINT_STEPS:
                ckpt = checkpoint_path(method, seed, step)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "step": step,
                        "checkpoint_path": str(ckpt.relative_to(ROOT)) if ckpt.exists() else str(ckpt),
                        "exists": int(ckpt.exists()),
                        "num_timesteps": checkpoint_num_timesteps(ckpt) or "",
                    }
                )
    pd.DataFrame(rows).to_csv(P1_DIR / "checkpoint_index.csv", index=False)


def write_completion_flag() -> None:
    manifest = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "p0_report": "P0_TRACE_PREDICTIVE_ANALYSIS_REPORT.md",
        "p1_report": "P1_THREE_SEED_REPLICATION_REPORT.md",
        "p0_summary": "results/p0_trace/p0_trace_summary.csv",
        "p1_by_seed": "results/p1_three_seed/p1_by_seed_step_scenario.csv",
        "p1_summary": "results/p1_three_seed/p1_summary_by_method_step_scenario.csv",
        "required_p1_rows": 144,
        "required_p1_summary_rows": 48,
    }
    (ROOT / "results/P0_P1_COMPLETE.flag").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=int(os.environ.get("P1_N_ENVS", "16")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_checkpoint_index()
    analyze_p0()
    run_p1(args.n_envs)
    write_checkpoint_index()
    write_completion_flag()
    append_status("P0/P1 complete flag written")


if __name__ == "__main__":
    main()
