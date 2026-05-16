from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

OUT_DIR = ROOT / "results/p1_5_distance_wide"
EVAL_DIR = OUT_DIR / "eval"
PLOTS_DIR = OUT_DIR / "plots"
SWEEP_PLOTS_DIR = OUT_DIR / "d_warning_sweep_seed0_plots"
CKPT_DIR = ROOT / "checkpoints/p1_5_distance_wide"
RUN_DIR = ROOT / "runs/p1_5_distance_wide"
LOG_DIR = ROOT / "runs/logs"

CHECKPOINT_STEPS = [250000, 500000, 750000, 1000000]
SCENARIOS = ["eval_random_switch", "eval_sudden_turn", "eval_random_switch_hard", "mixed_uncertainty"]
SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "mixed_uncertainty": "mixed",
}
EPS = 1e-10


@dataclass(frozen=True)
class MethodConfig:
    method: str
    display_method: str
    seeds: tuple[int, ...]
    d_warning: float
    train_new: bool
    source_kind: str


METHODS = [
    MethodConfig("attention_full", "attention_full", (0, 1, 2), 1.0, False, "p1"),
    MethodConfig("attention_full_distance_penalty", "attention_full_distance_penalty_d1", (0, 1, 2), 1.0, False, "p1"),
    MethodConfig("attention_full_distance_penalty_wide_d2", "attention_full_distance_penalty_wide_d2", (0, 1, 2), 2.0, True, "mixed_wide"),
    MethodConfig("attention_full_risk_penalty", "attention_full_risk_penalty", (0, 1, 2), 1.0, False, "p1"),
    MethodConfig("attention_full_distance_penalty_mid_d15", "attention_full_distance_penalty_mid_d15", (0,), 1.5, True, "p1_5"),
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, EVAL_DIR, PLOTS_DIR, SWEEP_PLOTS_DIR, CKPT_DIR, RUN_DIR, LOG_DIR]:
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


def checkpoint_path(config: MethodConfig, seed: int, step: int) -> Path:
    if config.display_method == "attention_full":
        if seed == 0:
            return ROOT / f"checkpoints/longtrain_baseline/attention_full_s0_step{step}.zip"
        if seed == 1:
            return ROOT / f"checkpoints/attention_seed1/attention_full_s1_step{step}.zip"
        return ROOT / f"checkpoints/p1_three_seed/attention_full_s2_step{step}.zip"
    if config.display_method == "attention_full_distance_penalty_d1":
        if seed == 0:
            return ROOT / f"checkpoints/gate2b/attention_full_distance_penalty_s0_step{step}.zip"
        return ROOT / f"checkpoints/p1_three_seed/attention_full_distance_penalty_s{seed}_step{step}.zip"
    if config.display_method == "attention_full_risk_penalty":
        if seed == 0:
            return ROOT / f"checkpoints/gate2b/attention_full_risk_penalty_s0_step{step}.zip"
        return ROOT / f"checkpoints/p1_three_seed/attention_full_risk_penalty_s{seed}_step{step}.zip"
    if config.display_method == "attention_full_distance_penalty_wide_d2" and seed == 0:
        return ROOT / f"checkpoints/p0_5_distance_wide/attention_full_distance_penalty_wide_s0_step{step}.zip"
    return CKPT_DIR / f"{config.method}_s{seed}_step{step}.zip"


def p1_eval_source(config: MethodConfig, seed: int, step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[scenario]
    if config.display_method == "attention_full":
        return ROOT / f"results/p1_three_seed/eval/attention_full_s{seed}_step{step}_{suffix}.csv"
    if config.display_method == "attention_full_distance_penalty_d1":
        return ROOT / f"results/p1_three_seed/eval/attention_full_distance_penalty_s{seed}_step{step}_{suffix}.csv"
    if config.display_method == "attention_full_risk_penalty":
        return ROOT / f"results/p1_three_seed/eval/attention_full_risk_penalty_s{seed}_step{step}_{suffix}.csv"
    if config.display_method == "attention_full_distance_penalty_wide_d2" and seed == 0:
        return ROOT / f"results/p0_5_distance_wide/eval/attention_full_distance_penalty_wide_s0_step{step}_{suffix}.csv"
    return EVAL_DIR / f"{config.method}_s{seed}_step{step}_{suffix}.csv"


def p1_5_eval_csv(config: MethodConfig, seed: int, step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[scenario]
    return EVAL_DIR / f"{config.method}_s{seed}_step{step}_{suffix}.csv"


def train_if_needed(config: MethodConfig, seed: int, n_envs: int) -> None:
    if not config.train_new:
        return
    if config.display_method == "attention_full_distance_penalty_wide_d2" and seed == 0:
        return
    if all(checkpoint_path(config, seed, step).exists() for step in CHECKPOINT_STEPS):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP training existing method={config.method} seed={seed}", flush=True)
        return
    cmd = [
        PYTHON,
        "train.py",
        "--method",
        config.method,
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
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
        "--use_safety_cost",
        "true",
        "--cost_type",
        "distance_warning",
        "--d_warning",
        str(config.d_warning),
        "--fallback_penalty",
        "true",
        "--beta_cost",
        "5.0",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        ",".join(str(step) for step in CHECKPOINT_STEPS),
        "--checkpoint_dir",
        str(CKPT_DIR),
        "--log_dir",
        str(RUN_DIR / f"{config.method}_s{seed}"),
        "--run_name",
        config.method,
        "--save_path",
        str(CKPT_DIR / f"{config.method}_s{seed}_step1000000.zip"),
        "--heartbeat_seconds",
        "30",
    ]
    run_command(cmd, LOG_DIR / f"train_p1_5_{config.method}_s{seed}.log", skip_marker="TRAIN_END")


def eval_if_needed(config: MethodConfig, seed: int, step: int, scenario: str) -> Path:
    source = p1_eval_source(config, seed, step, scenario)
    if source.exists() and csv_row_count(source) >= 50:
        return source
    out_csv = p1_5_eval_csv(config, seed, step, scenario)
    if csv_row_count(out_csv) >= 50:
        return out_csv
    ckpt = checkpoint_path(config, seed, step)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint for eval: {ckpt}")
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(ckpt),
        "--method",
        config.method,
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
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
        "--d_warning",
        str(config.d_warning),
    ]
    suffix = SCENARIO_SUFFIX[scenario]
    run_command(cmd, LOG_DIR / f"eval_p1_5_{config.method}_s{seed}_step{step}_{suffix}.log", skip_marker="EVAL_END")
    return out_csv


def percentile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(np.percentile(numeric.to_numpy(dtype=float), q))


def mean_col(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return float("nan")
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.dropna().empty:
        return float("nan")
    return float(numeric.mean())


def summarize_eval_csv(path: Path, config: MethodConfig, seed: int, step: int, scenario: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    cost_max = pd.to_numeric(df["distance_warning_cost_max"], errors="coerce")
    min_dist = pd.to_numeric(df["episode_min_distance"], errors="coerce")
    reaction_eval = pd.to_numeric(df.get("reaction_time_eval_style", pd.Series(dtype=float)), errors="coerce")
    reaction_nan = pd.to_numeric(df.get("reaction_time_nan_style", pd.Series(dtype=float)), errors="coerce")
    turn_based = scenario in {"eval_sudden_turn", "mixed_uncertainty"}
    return {
        "method": config.display_method,
        "raw_method": config.method,
        "seed": seed,
        "step": step,
        "scenario": scenario,
        "d_warning": config.d_warning,
        "episodes": int(len(df)),
        "success_rate": mean_col(df, "success"),
        "collision_rate": mean_col(df, "collision"),
        "reaction_time_eval_style": mean_col(df, "reaction_time_eval_style"),
        "reaction_time_nan_style": mean_col(df, "reaction_time_nan_style"),
        "nan_reaction_rate": float(reaction_nan.isna().mean()) if turn_based else float("nan"),
        "mean_min_distance": mean_col(df, "episode_min_distance"),
        "near_miss_rate": mean_col(df, "near_miss"),
        "distance_warning_cost_nonzero_rate": float((cost_max > EPS).mean()),
        "distance_warning_cost_mean": mean_col(df, "distance_warning_cost_mean"),
        "distance_warning_cost_p90": percentile(cost_max, 90),
        "distance_warning_cost_p95": percentile(cost_max, 95),
        "distance_warning_cost_max": float(cost_max.max()) if not cost_max.dropna().empty else float("nan"),
        "risk_sum_mean": mean_col(df, "risk_sum_mean"),
        "risk_max_mean": mean_col(df, "risk_max_mean"),
        "mean_time": mean_col(df, "time_to_goal"),
        "min_distance_p10": percentile(min_dist, 10),
        "min_distance_p25": percentile(min_dist, 25),
        "min_distance_min": float(min_dist.min()) if not min_dist.dropna().empty else float("nan"),
        "checkpoint_path": rel(checkpoint_path(config, seed, step)),
        "eval_csv_path": rel(path),
    }


def gather_rows(n_envs: int) -> pd.DataFrame:
    ensure_dirs()
    for config in METHODS:
        for seed in config.seeds:
            train_if_needed(config, seed, n_envs=n_envs)

    rows: list[dict[str, Any]] = []
    for config in METHODS:
        for seed in config.seeds:
            for step in CHECKPOINT_STEPS:
                for scenario in SCENARIOS:
                    path = eval_if_needed(config, seed, step, scenario)
                    rows.append(summarize_eval_csv(path, config, seed, step, scenario))

    by_seed = pd.DataFrame(rows).sort_values(["method", "seed", "step", "scenario"])
    by_seed.to_csv(OUT_DIR / "p1_5_by_seed_step_scenario.csv", index=False)
    return by_seed


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "success_rate",
        "collision_rate",
        "reaction_time_eval_style",
        "reaction_time_nan_style",
        "nan_reaction_rate",
        "mean_min_distance",
        "near_miss_rate",
        "distance_warning_cost_nonzero_rate",
        "distance_warning_cost_mean",
        "distance_warning_cost_p90",
        "distance_warning_cost_p95",
        "distance_warning_cost_max",
        "risk_sum_mean",
        "risk_max_mean",
        "mean_time",
    ]
    rows: list[dict[str, Any]] = []
    for (method, step, scenario), group in df.groupby(["method", "step", "scenario"], sort=True):
        row: dict[str, Any] = {
            "method": method,
            "step": int(step),
            "scenario": scenario,
            "seed_count": int(group["seed"].nunique()),
            "d_warning": float(group["d_warning"].dropna().iloc[0]) if not group["d_warning"].dropna().empty else float("nan"),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"mean_{col}"] = float(values.mean()) if not values.dropna().empty else float("nan")
            row[f"std_{col}"] = float(values.std()) if len(values.dropna()) > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    summary.to_csv(OUT_DIR / "p1_5_summary_by_method_step_scenario.csv", index=False)
    return summary


def build_main_750k_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, seed), group in df[df["step"] == 750000].groupby(["method", "seed"], sort=True):
        def metric(scenario: str, column: str) -> float:
            hit = group[group["scenario"] == scenario]
            if hit.empty:
                return float("nan")
            return float(hit.iloc[0][column])

        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "step": 750000,
                "sudden_reaction": metric("eval_sudden_turn", "reaction_time_eval_style"),
                "sudden_success": metric("eval_sudden_turn", "success_rate"),
                "sudden_collision": metric("eval_sudden_turn", "collision_rate"),
                "sudden_min_distance": metric("eval_sudden_turn", "mean_min_distance"),
                "sudden_near_miss": metric("eval_sudden_turn", "near_miss_rate"),
                "random_success": metric("eval_random_switch", "success_rate"),
                "random_mean_time": metric("eval_random_switch", "mean_time"),
                "random_min_distance": metric("eval_random_switch", "mean_min_distance"),
                "random_near_miss": metric("eval_random_switch", "near_miss_rate"),
                "hard_success": metric("eval_random_switch_hard", "success_rate"),
                "hard_collision": metric("eval_random_switch_hard", "collision_rate"),
                "hard_near_miss": metric("eval_random_switch_hard", "near_miss_rate"),
                "mixed_success": metric("mixed_uncertainty", "success_rate"),
                "mixed_collision": metric("mixed_uncertainty", "collision_rate"),
                "mixed_near_miss": metric("mixed_uncertainty", "near_miss_rate"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["method", "seed"])
    out.to_csv(OUT_DIR / "p1_5_main_750k_table.csv", index=False)
    return out


def build_pareto_summary(df: pd.DataFrame, *, write: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, seed, step), group in df.groupby(["method", "seed", "step"], sort=True):
        def metric(scenario: str, column: str) -> float:
            hit = group[group["scenario"] == scenario]
            if hit.empty:
                return float("nan")
            return float(hit.iloc[0][column])

        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "step": int(step),
                "d_warning": float(group["d_warning"].dropna().iloc[0]) if not group["d_warning"].dropna().empty else float("nan"),
                "sudden_reaction": metric("eval_sudden_turn", "reaction_time_eval_style"),
                "sudden_near_miss": metric("eval_sudden_turn", "near_miss_rate"),
                "sudden_collision": metric("eval_sudden_turn", "collision_rate"),
                "sudden_min_distance": metric("eval_sudden_turn", "mean_min_distance"),
                "sudden_mean_time": metric("eval_sudden_turn", "mean_time"),
                "random_success": metric("eval_random_switch", "success_rate"),
                "random_mean_time": metric("eval_random_switch", "mean_time"),
                "random_min_distance": metric("eval_random_switch", "mean_min_distance"),
                "random_near_miss": metric("eval_random_switch", "near_miss_rate"),
                "hard_success": metric("eval_random_switch_hard", "success_rate"),
                "hard_collision": metric("eval_random_switch_hard", "collision_rate"),
                "hard_near_miss": metric("eval_random_switch_hard", "near_miss_rate"),
                "hard_min_distance": metric("eval_random_switch_hard", "mean_min_distance"),
                "mixed_success": metric("mixed_uncertainty", "success_rate"),
                "mixed_collision": metric("mixed_uncertainty", "collision_rate"),
                "mixed_near_miss": metric("mixed_uncertainty", "near_miss_rate"),
                "mixed_min_distance": metric("mixed_uncertainty", "mean_min_distance"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["method", "seed", "step"])
    out["is_750k"] = (out["step"] == 750000).astype(int)
    if write:
        out.to_csv(OUT_DIR / "p1_5_pareto_summary.csv", index=False)
    return out


def build_d_warning_sweep(df: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "attention_full_distance_penalty_d1",
        "attention_full_distance_penalty_mid_d15",
        "attention_full_distance_penalty_wide_d2",
        "attention_full_risk_penalty",
    }
    sweep = df[(df["seed"] == 0) & (df["method"].isin(keep))].copy()
    sweep["sweep_label"] = sweep["method"].map(
        {
            "attention_full_distance_penalty_d1": "d1.0",
            "attention_full_distance_penalty_mid_d15": "d1.5",
            "attention_full_distance_penalty_wide_d2": "d2.0",
            "attention_full_risk_penalty": "risk",
        }
    )
    sweep.to_csv(OUT_DIR / "d_warning_sweep_seed0.csv", index=False)
    return sweep


def plot_outputs(pareto: pd.DataFrame, sweep: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("sudden_reaction", "random_mean_time", "sudden_reaction_vs_random_mean_time.png", "sudden reaction (s)", "random mean time (s)"),
        ("hard_near_miss", "random_mean_time", "hard_near_miss_vs_random_mean_time.png", "hard near miss", "random mean time (s)"),
        ("sudden_near_miss", "random_mean_time", "sudden_near_miss_vs_random_mean_time.png", "sudden near miss", "random mean time (s)"),
        ("mixed_collision", "random_mean_time", "mixed_collision_vs_random_mean_time.png", "mixed collision", "random mean time (s)"),
        ("sudden_min_distance", "sudden_mean_time", "min_distance_vs_mean_time.png", "sudden min distance", "sudden mean time (s)"),
    ]
    markers = {250000: "o", 500000: "s", 750000: "*", 1000000: "^"}
    for x_col, y_col, filename, xlabel, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in pareto.groupby("method", sort=True):
            for step, step_group in group.groupby("step"):
                size = 120 if step == 750000 else 45
                ax.scatter(step_group[x_col], step_group[y_col], s=size, marker=markers.get(int(step), "o"), label=f"{method} {int(step/1000)}k", alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    seed0_pareto = pareto[pareto["seed"] == 0].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in seed0_pareto.groupby("method", sort=True):
        group = group.sort_values("step")
        ax.plot(group["random_mean_time"], group["sudden_reaction"], marker="o", label=method)
    ax.set_xlabel("random mean time (s)")
    ax.set_ylabel("sudden reaction (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "seed0_tradeoff_curves.png", dpi=150)
    plt.close(fig)

    sweep_pareto = build_pareto_summary(sweep, write=False)
    step750 = sweep_pareto[sweep_pareto["step"] == 750000].sort_values("d_warning")
    ordered = ["attention_full_distance_penalty_d1", "attention_full_distance_penalty_mid_d15", "attention_full_distance_penalty_wide_d2", "attention_full_risk_penalty"]
    x_labels = ["d1.0", "d1.5", "d2.0", "risk"]
    metrics = [
        ("sudden_reaction", "d_warning_vs_sudden_reaction.png", "sudden reaction (s)"),
        ("random_mean_time", "d_warning_vs_random_mean_time.png", "random mean time (s)"),
        ("hard_near_miss", "d_warning_vs_hard_near_miss.png", "hard near miss"),
    ]
    for metric, filename, ylabel in metrics:
        values = []
        for method in ordered:
            row = step750[step750["method"] == method]
            values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x_labels, values, marker="o")
        ax.set_xlabel("distance margin setting")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(SWEEP_PLOTS_DIR / filename, dpi=150)
        plt.close(fig)


def method_mean(main750: pd.DataFrame, method: str, col: str) -> float:
    rows = main750[main750["method"] == method]
    if rows.empty:
        return float("nan")
    values = pd.to_numeric(rows[col], errors="coerce")
    return float(values.mean()) if not values.dropna().empty else float("nan")


def classify_decision(main750: pd.DataFrame) -> tuple[str, str]:
    wide_reaction = method_mean(main750, "attention_full_distance_penalty_wide_d2", "sudden_reaction")
    risk_reaction = method_mean(main750, "attention_full_risk_penalty", "sudden_reaction")
    wide_time = method_mean(main750, "attention_full_distance_penalty_wide_d2", "random_mean_time")
    risk_time = method_mean(main750, "attention_full_risk_penalty", "random_mean_time")
    wide_hard_near = method_mean(main750, "attention_full_distance_penalty_wide_d2", "hard_near_miss")
    risk_hard_near = method_mean(main750, "attention_full_risk_penalty", "hard_near_miss")
    wide_mixed_collision = method_mean(main750, "attention_full_distance_penalty_wide_d2", "mixed_collision")
    risk_mixed_collision = method_mean(main750, "attention_full_risk_penalty", "mixed_collision")

    safety_close = (
        abs(wide_reaction - risk_reaction) <= 1.0
        and abs(wide_hard_near - risk_hard_near) <= 0.15
        and abs(wide_mixed_collision - risk_mixed_collision) <= 0.15
    )
    wide_more_conservative = wide_time > risk_time + 0.25
    wide_better = (
        wide_reaction <= risk_reaction + 0.2
        and wide_time <= risk_time + 0.1
        and wide_hard_near <= risk_hard_near + 0.05
        and wide_mixed_collision <= risk_mixed_collision + 0.05
    )
    if safety_close and wide_more_conservative:
        return (
            "A",
            "wide_d2 is close to risk on safety, but is slower / more conservative on random-switch efficiency.",
        )
    if safety_close:
        return ("B", "wide_d2 and risk are broadly equivalent in this three-seed comparison.")
    if wide_better:
        return ("C", "wide_d2 is at least as safe and at least as efficient as risk.")
    return ("D", "wide_d2 does not stably match risk across seeds or scenarios.")


def write_report(df: pd.DataFrame, summary: pd.DataFrame, main750: pd.DataFrame, pareto: pd.DataFrame, sweep: pd.DataFrame) -> None:
    decision, decision_text = classify_decision(main750)
    wide = main750[main750["method"] == "attention_full_distance_penalty_wide_d2"]
    risk = main750[main750["method"] == "attention_full_risk_penalty"]

    def mean(method: str, col: str) -> str:
        return fmt(method_mean(main750, method, col))

    sweep750 = build_pareto_summary(sweep, write=False)
    sweep750 = sweep750[sweep750["step"] == 750000]
    risk_time = method_mean(main750, "attention_full_risk_penalty", "random_mean_time")
    risk_reaction = method_mean(main750, "attention_full_risk_penalty", "sudden_reaction")
    distance_candidates = sweep750[sweep750["method"].str.contains("distance_penalty", regex=False)]
    risk_on_front = True
    for _, row in distance_candidates.iterrows():
        if float(row["random_mean_time"]) <= risk_time and float(row["sudden_reaction"]) <= risk_reaction:
            if float(row["random_mean_time"]) < risk_time or float(row["sudden_reaction"]) < risk_reaction:
                risk_on_front = False

    lines = [
        "# P1.5 Distance Margin Sweep Report",
        "",
        "## 1. Purpose",
        "P1.5 tests whether the risk_penalty advantage is explained by motion-uncertainty risk itself or by a wider / denser distance safety margin. This must be settled before P2 environment upgrades, otherwise P2 could over-attribute a generic distance-margin effect to risk.",
        "",
        "## 2. Experiment Setup",
        "- Reused methods: attention_full, attention_full_distance_penalty_d1, attention_full_risk_penalty.",
        "- New required method: attention_full_distance_penalty_wide_d2, d_warning=2.0, seeds 1 and 2; seed 0 reused from P0.5-C.",
        "- Optional executed method: attention_full_distance_penalty_mid_d15, d_warning=1.5, seed 0.",
        "- beta_cost=5.0, fallback_penalty=true, profile_mode=full_12, train scenario=train_random_switch.",
        "- Checkpoints: 250000, 500000, 750000, 1000000.",
        "- Eval scenarios: eval_random_switch, eval_sudden_turn, eval_random_switch_hard, mixed_uncertainty.",
        "- Eval episodes: 50; eval_seed=1000.",
        "",
        "## 3. Reused Runs and New Runs",
        "- Reused P1 three-seed evals for baseline, d1 distance penalty, and risk penalty.",
        "- Reused P0.5-C wide_d2 seed=0 checkpoints/evals.",
        "- Newly trained/evaluated wide_d2 seeds 1 and 2.",
        "- Newly trained/evaluated mid_d15 seed 0.",
        "",
        "## 4. Main 750k Comparison",
        "| method | seed | sudden_reaction | sudden_success | sudden_collision | random_mean_time | hard_near_miss | mixed_collision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in main750.sort_values(["method", "seed"]).iterrows():
        lines.append(
            f"| {row['method']} | {int(row['seed'])} | {fmt(row['sudden_reaction'])} | "
            f"{fmt(row['sudden_success'], 3)} | {fmt(row['sudden_collision'], 3)} | "
            f"{fmt(row['random_mean_time'])} | {fmt(row['hard_near_miss'], 3)} | {fmt(row['mixed_collision'], 3)} |"
        )
    lines += [
        "",
        "## 5. Three-Seed Wide Distance Result",
        f"- wide_d2 mean sudden reaction at 750k: {mean('attention_full_distance_penalty_wide_d2', 'sudden_reaction')} s.",
        f"- risk_penalty mean sudden reaction at 750k: {mean('attention_full_risk_penalty', 'sudden_reaction')} s.",
        f"- wide_d2 mean hard near_miss at 750k: {mean('attention_full_distance_penalty_wide_d2', 'hard_near_miss')}.",
        f"- risk_penalty mean hard near_miss at 750k: {mean('attention_full_risk_penalty', 'hard_near_miss')}.",
        f"- wide_d2 seeds with sudden reaction <= 1s: {int((pd.to_numeric(wide['sudden_reaction'], errors='coerce') <= 1.0).sum())}/3.",
        f"- risk_penalty seeds with sudden reaction <= 1s: {int((pd.to_numeric(risk['sudden_reaction'], errors='coerce') <= 1.0).sum())}/3.",
        "",
        "## 6. Safety-Efficiency Trade-off",
        f"- wide_d2 mean random mean_time at 750k: {mean('attention_full_distance_penalty_wide_d2', 'random_mean_time')} s.",
        f"- risk_penalty mean random mean_time at 750k: {mean('attention_full_risk_penalty', 'random_mean_time')} s.",
        f"- d1 mean sudden reaction at 750k: {mean('attention_full_distance_penalty_d1', 'sudden_reaction')} s.",
        f"- baseline mean sudden reaction at 750k: {mean('attention_full', 'sudden_reaction')} s.",
        f"- Decision class: {decision}. {decision_text}",
        "",
        "## 7. Optional d_warning=1.5 Sweep",
        "- d15 seed=0 was executed.",
        "- The seed0 sweep is provided in d_warning_sweep_seed0.csv and d_warning_sweep_seed0_plots/*.png.",
        f"- Risk on the seed0 sudden_reaction/random_time Pareto front: {'yes' if risk_on_front else 'no'}.",
        "",
        "## 8. Decision",
        f"- Selected case: {decision}.",
        f"- Interpretation: {decision_text}",
        "",
        "## 9. Next Recommendation",
    ]
    if decision == "A":
        lines.append("- Enter P2, but compare risk_penalty against wide_d2 directly. The defensible risk claim is better safety-efficiency trade-off, not strict early warning.")
    elif decision == "B":
        lines.append("- Delay strong risk claims; treat risk and wide distance as competing dense safety-margin regularizers in P2.")
    elif decision == "C":
        lines.append("- Downgrade the risk main line and prioritize distance-margin / safety-cost schedule studies.")
    else:
        lines.append("- Keep risk as the primary candidate for P2 because wide_d2 did not reproduce stable risk-like behavior.")
    lines.append("- P2 should compare: attention_full baseline, attention_full_distance_penalty_wide_d2, attention_full_risk_penalty; optionally keep d1 as a reference.")
    lines += [
        "",
        "## Required Answers",
        f"1. wide_d2 seed=1/2 close to risk_penalty: compare table; mean wide_d2 sudden reaction {mean('attention_full_distance_penalty_wide_d2', 'sudden_reaction')} s vs risk {mean('attention_full_risk_penalty', 'sudden_reaction')} s.",
        f"2. Sudden reaction difference: {fmt(method_mean(main750, 'attention_full_distance_penalty_wide_d2', 'sudden_reaction') - method_mean(main750, 'attention_full_risk_penalty', 'sudden_reaction'))} s at 750k mean.",
        f"3. wide_d2 slower/more conservative: random mean_time wide={mean('attention_full_distance_penalty_wide_d2', 'random_mean_time')} s vs risk={mean('attention_full_risk_penalty', 'random_mean_time')} s.",
        f"4. risk safety-efficiency advantage over wide_d2: {'yes' if decision == 'A' else 'not clearly'}.",
        "5. d_warning=1.5 continuous trade-off: see d_warning_sweep_seed0.csv and plots.",
        f"6. risk on distance-margin sweep Pareto front: {'yes' if risk_on_front else 'no'}.",
        f"7. Enter P2: {'yes' if decision in {'A', 'D'} else 'only with risk downgraded / reframed'}.",
        "8. P2 methods: attention_full baseline, distance_penalty_wide_d2, risk_penalty, optional distance_penalty_d1.",
        "",
        "## Artifacts",
        "- results/p1_5_distance_wide/p1_5_by_seed_step_scenario.csv",
        "- results/p1_5_distance_wide/p1_5_summary_by_method_step_scenario.csv",
        "- results/p1_5_distance_wide/p1_5_main_750k_table.csv",
        "- results/p1_5_distance_wide/p1_5_pareto_summary.csv",
        "- results/p1_5_distance_wide/d_warning_sweep_seed0.csv",
        "- results/p1_5_distance_wide/plots/*.png",
        "- results/p1_5_distance_wide/d_warning_sweep_seed0_plots/*.png",
    ]
    write_lines(ROOT / "P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md", lines)


def write_completion_flag() -> None:
    manifest = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report": "P1_5_DISTANCE_MARGIN_SWEEP_REPORT.md",
        "by_seed_rows": 208,
        "summary_rows": 80,
        "main_750k_rows": 13,
        "pareto_rows": 52,
        "d_warning_sweep_seed0_rows": 64,
        "required_new_train_runs": [
            "attention_full_distance_penalty_wide_d2_s1",
            "attention_full_distance_penalty_wide_d2_s2",
            "attention_full_distance_penalty_mid_d15_s0",
        ],
    }
    (OUT_DIR / "P1_5_COMPLETE.flag").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=int(os.environ.get("P1_5_N_ENVS", "16")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    by_seed = gather_rows(n_envs=args.n_envs)
    summary = summarize_by_method(by_seed)
    main750 = build_main_750k_table(by_seed)
    pareto = build_pareto_summary(by_seed)
    sweep = build_d_warning_sweep(by_seed)
    plot_outputs(pareto, sweep)
    write_report(by_seed, summary, main750, pareto, sweep)
    write_completion_flag()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] P1.5 complete flag written", flush=True)


if __name__ == "__main__":
    main()
