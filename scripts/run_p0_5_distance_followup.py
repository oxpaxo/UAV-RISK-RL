from __future__ import annotations

import argparse
import math
import os
import re
import shutil
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

P05_SANITY_DIR = ROOT / "results/p0_5_distance_sanity"
P05_WIDE_DIR = ROOT / "results/p0_5_distance_wide"
P05_WIDE_EVAL_DIR = P05_WIDE_DIR / "eval"
P05_WIDE_CKPT_DIR = ROOT / "checkpoints/p0_5_distance_wide"
P05_WIDE_RUN_DIR = ROOT / "runs/p0_5_distance_wide"
RUN_LOG_DIR = ROOT / "runs/logs"

CHECKPOINT_STEPS = [250000, 500000, 750000, 1000000]
TRIGGER_STEPS = [500000, 750000, 1000000]
SCENARIOS_AB = ["eval_random_switch", "eval_sudden_turn"]
SCENARIOS_WIDE = ["eval_random_switch", "eval_sudden_turn", "eval_random_switch_hard", "mixed_uncertainty"]
SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "mixed_uncertainty": "mixed",
}
D_WARNING_NARROW = 1.0
D_WARNING_WIDE = 2.0
EPS = 1e-10


@dataclass(frozen=True)
class EvalSource:
    method: str
    display_method: str
    d_warning: float
    source_dir: Path


NARROW_EVAL_SOURCES = [
    EvalSource("attention_full", "attention_full", D_WARNING_NARROW, ROOT / "results/longtrain_baseline/eval"),
    EvalSource(
        "attention_full_distance_penalty",
        "attention_full_distance_penalty_d1",
        D_WARNING_NARROW,
        ROOT / "results/gate2b/eval",
    ),
    EvalSource(
        "attention_full_risk_penalty",
        "attention_full_risk_penalty",
        D_WARNING_NARROW,
        ROOT / "results/gate2b/eval",
    ),
]


def ensure_dirs() -> None:
    for path in [P05_SANITY_DIR, P05_WIDE_DIR, P05_WIDE_EVAL_DIR, P05_WIDE_CKPT_DIR, P05_WIDE_RUN_DIR, RUN_LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def pct(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(np.percentile(values.to_numpy(dtype=float), q))


def mean_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.mean())


def trace_episode_from_path(path: Path, df: pd.DataFrame) -> int:
    if "episode" in df.columns and not df.empty:
        try:
            return int(df["episode"].iloc[0])
        except Exception:
            pass
    match = re.search(r"_ep(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else -1


def sanity_judgment(
    min_distance: float,
    trace_max: float,
    recomputed_max: float,
    max_abs_diff: float,
    d_warning: float,
) -> str:
    if max_abs_diff > 1e-8:
        return "BUG_TRACE_RECOMPUTE_MISMATCH"
    if min_distance < d_warning and trace_max <= EPS:
        return "BUG_MIN_DISTANCE_BELOW_WARNING_BUT_TRACE_ZERO"
    if min_distance > d_warning and trace_max <= EPS and recomputed_max <= EPS:
        return "OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING"
    return "OK_TRIGGERED_CONSISTENT"


def run_distance_cost_sanity() -> pd.DataFrame:
    ensure_dirs()
    trace_dir = ROOT / "results/gate2b/traces"
    methods = [
        "attention_full",
        "attention_full_distance_penalty",
        "attention_full_risk_penalty",
    ]
    rows: list[dict[str, Any]] = []
    for method in methods:
        paths = sorted(trace_dir.glob(f"{method}_step750000_eval_sudden_turn_ep*.csv"))
        for path in paths[:10]:
            df = pd.read_csv(path)
            min_distance_values = pd.to_numeric(df["min_distance"], errors="coerce")
            trace_values = pd.to_numeric(df["distance_warning_cost"], errors="coerce")
            recomputed = np.maximum(0.0, D_WARNING_NARROW - min_distance_values.to_numpy(dtype=float)) ** 2
            trace_arr = trace_values.to_numpy(dtype=float)
            abs_diff = np.abs(np.nan_to_num(trace_arr, nan=0.0) - np.nan_to_num(recomputed, nan=0.0))
            min_distance = float(np.nanmin(min_distance_values.to_numpy(dtype=float)))
            trace_max = float(np.nanmax(trace_arr))
            recomputed_max = float(np.nanmax(recomputed))
            max_abs_diff = float(np.nanmax(abs_diff))
            rows.append(
                {
                    "method": method,
                    "seed": 0,
                    "step": 750000,
                    "scenario": "eval_sudden_turn",
                    "episode": trace_episode_from_path(path, df),
                    "d_warning": D_WARNING_NARROW,
                    "min_min_distance": min_distance,
                    "max_distance_warning_cost_trace": trace_max,
                    "count_min_distance_lt_d_warning": int((min_distance_values < D_WARNING_NARROW).sum()),
                    "count_distance_warning_cost_gt_0": int((trace_values > 0.0).sum()),
                    "max_recomputed_distance_warning_cost": recomputed_max,
                    "max_abs_diff_between_trace_and_recomputed": max_abs_diff,
                    "judgment": sanity_judgment(min_distance, trace_max, recomputed_max, max_abs_diff, D_WARNING_NARROW),
                    "source_trace_path": rel(path),
                }
            )
    out = pd.DataFrame(rows).sort_values(["method", "episode"])
    out_path = P05_SANITY_DIR / "distance_cost_sanity.csv"
    out.to_csv(out_path, index=False)
    write_distance_cost_sanity_report(out)
    return out


def write_distance_cost_sanity_report(df: pd.DataFrame) -> None:
    lines = [
        "# Distance Cost Sanity Report",
        "",
        "## Scope",
        "- d_warning = 1.0.",
        "- Recomputed formula: max(0, d_warning - min_distance) ** 2.",
        "- Covered existing step=750000 eval_sudden_turn traces for attention_full, attention_full_distance_penalty, and attention_full_risk_penalty.",
        "",
        "## Summary By Method",
        "| method | episodes | min(min_distance) | triggered episodes | mismatch rows | bug rows | zero-reasonable rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, group in df.groupby("method", sort=True):
        triggered = int((pd.to_numeric(group["max_distance_warning_cost_trace"], errors="coerce") > EPS).sum())
        mismatch = int((group["judgment"] == "BUG_TRACE_RECOMPUTE_MISMATCH").sum())
        bug = int(group["judgment"].astype(str).str.startswith("BUG").sum())
        zero_ok = int((group["judgment"] == "OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING").sum())
        lines.append(
            f"| {method} | {len(group)} | {fmt(group['min_min_distance'].min())} | {triggered} | {mismatch} | {bug} | {zero_ok} |"
        )
    lines += [
        "",
        "## Judgment",
    ]
    bug_rows = df[df["judgment"].astype(str).str.startswith("BUG")]
    if bug_rows.empty:
        lines.append("- No cost computation or trace recording mismatch was found in the covered traces.")
    else:
        lines.append(f"- Found {len(bug_rows)} suspicious rows; inspect distance_cost_sanity.csv for details.")
    zero_ok = int((df["judgment"] == "OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING").sum())
    lines.append(
        f"- {zero_ok}/{len(df)} episodes have min(min_distance) > 1.0 and both trace/recomputed distance_warning_cost equal to 0."
    )
    lines += [
        "",
        "## Episode Rows",
        "| method | episode | min_distance | trace_max | count_dist_lt_1 | recomputed_max | max_abs_diff | judgment |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['episode'])} | {fmt(row['min_min_distance'])} | "
            f"{fmt(row['max_distance_warning_cost_trace'])} | {int(row['count_min_distance_lt_d_warning'])} | "
            f"{fmt(row['max_recomputed_distance_warning_cost'])} | "
            f"{fmt(row['max_abs_diff_between_trace_and_recomputed'], 8)} | {row['judgment']} |"
        )
    lines += [
        "",
        "## Artifacts",
        "- results/p0_5_distance_sanity/distance_cost_sanity.csv",
    ]
    write_lines(P05_SANITY_DIR / "DISTANCE_COST_SANITY_REPORT.md", lines)


def source_eval_csv(source: EvalSource, step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[scenario]
    return source.source_dir / f"{source.method}_s0_step{step}_{suffix}.csv"


def summarize_eval_csv(path: Path, source: EvalSource, step: int, scenario: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    cost_max = pd.to_numeric(df["distance_warning_cost_max"], errors="coerce")
    min_dist = pd.to_numeric(df["episode_min_distance"], errors="coerce")
    turn_based = scenario in {"eval_sudden_turn", "mixed_uncertainty"}
    reaction_eval = pd.to_numeric(df.get("reaction_time_eval_style", pd.Series(dtype=float)), errors="coerce")
    reaction_nan = pd.to_numeric(df.get("reaction_time_nan_style", pd.Series(dtype=float)), errors="coerce")
    return {
        "method": source.display_method,
        "base_method": source.method,
        "seed": 0,
        "step": step,
        "scenario": scenario,
        "d_warning_eval": source.d_warning,
        "episodes": int(len(df)),
        "distance_warning_cost_nonzero_rate": float((cost_max > EPS).mean()),
        "distance_warning_cost_p50": pct(cost_max, 50),
        "distance_warning_cost_p90": pct(cost_max, 90),
        "distance_warning_cost_p95": pct(cost_max, 95),
        "distance_warning_cost_max": float(cost_max.max()) if not cost_max.dropna().empty else float("nan"),
        "min_distance_mean": mean_numeric(min_dist),
        "min_distance_p10": pct(min_dist, 10),
        "min_distance_p25": pct(min_dist, 25),
        "min_distance_min": float(min_dist.min()) if not min_dist.dropna().empty else float("nan"),
        "near_miss_rate": mean_numeric(df["near_miss"]),
        "success_rate": mean_numeric(df["success"]),
        "collision_rate": mean_numeric(df["collision"]),
        "mean_time": mean_numeric(df["time_to_goal"]),
        "reaction_time_eval_style": mean_numeric(reaction_eval),
        "reaction_time_nan_style": mean_numeric(reaction_nan),
        "nan_reaction_rate": float(reaction_nan.isna().mean()) if turn_based else float("nan"),
        "cost_stat_basis": "episode_distance_warning_cost_max",
        "source_eval_csv": rel(path),
    }


def run_distance_trigger_stats() -> pd.DataFrame:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in NARROW_EVAL_SOURCES:
        for step in TRIGGER_STEPS:
            for scenario in SCENARIOS_AB:
                path = source_eval_csv(source, step, scenario)
                if csv_row_count(path) < 50:
                    missing.append(rel(path))
                    continue
                rows.append(summarize_eval_csv(path, source, step, scenario))
    if missing:
        raise FileNotFoundError("Missing required P0.5-B eval CSVs:\n" + "\n".join(missing))
    out = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    out_path = P05_SANITY_DIR / "distance_trigger_stats.csv"
    out.to_csv(out_path, index=False)
    write_distance_trigger_stats_report(out)
    return out


def write_distance_trigger_stats_report(df: pd.DataFrame) -> None:
    lines = [
        "# Distance Trigger Stats Report",
        "",
        "## Scope",
        "- Existing seed=0 eval CSVs only; no new training was used for P0.5-B.",
        "- Methods: attention_full, attention_full_distance_penalty_d1, attention_full_risk_penalty.",
        "- Steps: 500000, 750000, 1000000.",
        "- Scenarios: eval_random_switch, eval_sudden_turn.",
        "- distance_warning_cost_* percentiles are computed over the 50 per-episode distance_warning_cost_max values.",
        "",
        "## Core Answer",
    ]
    max_rate_by_method = df.groupby("method")["distance_warning_cost_nonzero_rate"].max()
    for method, value in max_rate_by_method.items():
        lines.append(f"- {method}: max nonzero episode rate across requested rows = {fmt(value, 3)}.")
    if (max_rate_by_method < 0.1).all():
        lines.append(
            "- Baseline and distance_penalty both have distance_warning_cost_nonzero_rate near 0; d_warning=1.0 is a sparse trigger in these evals."
        )
    else:
        lines.append("- At least one method/scenario has a meaningful distance_warning_cost trigger rate; inspect the detailed table.")
    lines += [
        "",
        "## Detailed Rows",
        "| method | step | scenario | nonzero_rate | cost_p95 | cost_max | min_dist_mean | min_dist_min | near_miss | success | collision | reaction_eval | nan_reaction_rate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['step'])} | {row['scenario']} | "
            f"{fmt(row['distance_warning_cost_nonzero_rate'], 3)} | {fmt(row['distance_warning_cost_p95'])} | "
            f"{fmt(row['distance_warning_cost_max'])} | {fmt(row['min_distance_mean'])} | "
            f"{fmt(row['min_distance_min'])} | {fmt(row['near_miss_rate'], 3)} | "
            f"{fmt(row['success_rate'], 3)} | {fmt(row['collision_rate'], 3)} | "
            f"{fmt(row['reaction_time_eval_style'])} | {fmt(row['nan_reaction_rate'], 3)} |"
        )
    lines += [
        "",
        "## Artifacts",
        "- results/p0_5_distance_sanity/distance_trigger_stats.csv",
    ]
    write_lines(P05_SANITY_DIR / "DISTANCE_TRIGGER_STATS_REPORT.md", lines)


def run_ab() -> tuple[pd.DataFrame, pd.DataFrame]:
    sanity = run_distance_cost_sanity()
    trigger = run_distance_trigger_stats()
    return sanity, trigger


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


def wide_checkpoint_path(step: int) -> Path:
    return P05_WIDE_CKPT_DIR / f"attention_full_distance_penalty_wide_s0_step{step}.zip"


def train_wide_if_needed(n_envs: int) -> None:
    ensure_dirs()
    if all(wide_checkpoint_path(step).exists() for step in CHECKPOINT_STEPS):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP wide training existing checkpoints", flush=True)
        return
    cmd = [
        PYTHON,
        "train.py",
        "--method",
        "attention_full_distance_penalty_wide",
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
        "--seed",
        "0",
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
        str(P05_WIDE_CKPT_DIR),
        "--log_dir",
        str(P05_WIDE_RUN_DIR),
        "--run_name",
        "attention_full_distance_penalty_wide",
        "--save_path",
        str(P05_WIDE_CKPT_DIR / "attention_full_distance_penalty_wide_s0_step1000000.zip"),
        "--heartbeat_seconds",
        "30",
        "--use_safety_cost",
        "true",
        "--cost_type",
        "distance_warning",
        "--fallback_penalty",
        "true",
        "--beta_cost",
        "5.0",
        "--d_warning",
        str(D_WARNING_WIDE),
    ]
    run_command(cmd, RUN_LOG_DIR / "train_p0_5_attention_full_distance_penalty_wide_s0.log", skip_marker="TRAIN_END")


def wide_eval_csv(step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[scenario]
    return P05_WIDE_EVAL_DIR / f"attention_full_distance_penalty_wide_s0_step{step}_{suffix}.csv"


def eval_wide_if_needed(step: int, scenario: str) -> Path:
    out_csv = wide_eval_csv(step, scenario)
    if csv_row_count(out_csv) >= 50:
        return out_csv
    ckpt = wide_checkpoint_path(step)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing wide checkpoint: {ckpt}")
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(ckpt),
        "--method",
        "attention_full_distance_penalty_wide",
        "--profile_mode",
        "full_12",
        "--agg",
        "attention",
        "--seed",
        "0",
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
        str(D_WARNING_WIDE),
    ]
    suffix = SCENARIO_SUFFIX[scenario]
    run_command(cmd, RUN_LOG_DIR / f"eval_p0_5_attention_full_distance_penalty_wide_s0_step{step}_{suffix}.log", skip_marker="EVAL_END")
    return out_csv


def narrow_source_for_wide_report(source: EvalSource, step: int, scenario: str) -> Path | None:
    path = source_eval_csv(source, step, scenario)
    if csv_row_count(path) >= 50:
        return path
    return None


def run_wide(n_envs: int) -> pd.DataFrame:
    train_wide_if_needed(n_envs=n_envs)
    rows: list[dict[str, Any]] = []

    wide_source = EvalSource(
        "attention_full_distance_penalty_wide",
        "attention_full_distance_penalty_wide_d2",
        D_WARNING_WIDE,
        P05_WIDE_EVAL_DIR,
    )
    comparison_sources = [
        NARROW_EVAL_SOURCES[0],
        NARROW_EVAL_SOURCES[1],
        wide_source,
        NARROW_EVAL_SOURCES[2],
    ]

    for step in CHECKPOINT_STEPS:
        for scenario in SCENARIOS_WIDE:
            eval_wide_if_needed(step, scenario)

    for source in comparison_sources:
        for step in CHECKPOINT_STEPS:
            for scenario in SCENARIOS_WIDE:
                if source.display_method == "attention_full_distance_penalty_wide_d2":
                    path = wide_eval_csv(step, scenario)
                else:
                    path = narrow_source_for_wide_report(source, step, scenario)
                    if path is None:
                        continue
                rows.append(summarize_eval_csv(path, source, step, scenario))

    out = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    out.to_csv(P05_WIDE_DIR / "distance_wide_by_step_scenario.csv", index=False)
    write_distance_wide_report(out)
    update_p1_report_with_p05()
    (P05_WIDE_DIR / "P0_5_DISTANCE_WIDE_COMPLETE.flag").write_text(
        f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    return out


def write_distance_wide_report(df: pd.DataFrame) -> None:
    sudden_750 = df[(df["scenario"] == "eval_sudden_turn") & (df["step"] == 750000)]
    random_750 = df[(df["scenario"] == "eval_random_switch") & (df["step"] == 750000)]

    def row(method: str, frame: pd.DataFrame, col: str) -> float:
        hit = frame[frame["method"] == method]
        if hit.empty or col not in hit.columns:
            return float("nan")
        return float(hit.iloc[0][col])

    wide_reaction = row("attention_full_distance_penalty_wide_d2", sudden_750, "reaction_time_eval_style")
    risk_reaction = row("attention_full_risk_penalty", sudden_750, "reaction_time_eval_style")
    d1_reaction = row("attention_full_distance_penalty_d1", sudden_750, "reaction_time_eval_style")
    wide_random_success = row("attention_full_distance_penalty_wide_d2", random_750, "success_rate")
    risk_random_success = row("attention_full_risk_penalty", random_750, "success_rate")

    if not math.isnan(wide_reaction) and not math.isnan(risk_reaction) and wide_reaction <= risk_reaction + 1.0:
        judgment = "wide distance_penalty approaches risk_penalty on sudden-turn reaction."
    elif not math.isnan(wide_reaction) and not math.isnan(d1_reaction) and wide_reaction < d1_reaction:
        judgment = "wide distance_penalty improves over d_warning=1.0 but remains below risk_penalty."
    else:
        judgment = "wide distance_penalty does not clearly close the gap to risk_penalty."

    lines = [
        "# Distance Penalty Wide Ablation Report",
        "",
        "## Scope",
        "- New method: attention_full_distance_penalty_wide_d2.",
        "- Train seed: 0.",
        "- d_warning: 2.0.",
        "- Total steps: 1000000; checkpoints: 250000, 500000, 750000, 1000000.",
        "- Eval episodes: 50; eval_seed: 1000.",
        "",
        "## Main Judgment",
        f"- {judgment}",
        f"- At 750k sudden_turn: d1 reaction={fmt(d1_reaction)} s, wide_d2 reaction={fmt(wide_reaction)} s, risk reaction={fmt(risk_reaction)} s.",
        f"- At 750k random_switch: wide_d2 success={fmt(wide_random_success, 3)}, risk success={fmt(risk_random_success, 3)}.",
        "",
        "## 750k Comparison",
        "| scenario | method | reaction | success | collision | mean_time | min_dist_mean | near_miss | cost_nonzero |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows_750 = df[df["step"] == 750000].sort_values(["scenario", "method"])
    for _, row_data in rows_750.iterrows():
        lines.append(
            f"| {row_data['scenario']} | {row_data['method']} | {fmt(row_data['reaction_time_eval_style'])} | "
            f"{fmt(row_data['success_rate'], 3)} | {fmt(row_data['collision_rate'], 3)} | "
            f"{fmt(row_data['mean_time'])} | {fmt(row_data['min_distance_mean'])} | "
            f"{fmt(row_data['near_miss_rate'], 3)} | {fmt(row_data['distance_warning_cost_nonzero_rate'], 3)} |"
        )
    lines += [
        "",
        "## Artifacts",
        "- results/p0_5_distance_wide/distance_wide_by_step_scenario.csv",
        "- results/p0_5_distance_wide/eval/*.csv",
        "- checkpoints/p0_5_distance_wide/*.zip",
    ]
    write_lines(P05_WIDE_DIR / "DISTANCE_WIDE_REPORT.md", lines)


def p05_interaction_section() -> list[str]:
    sanity_path = P05_SANITY_DIR / "distance_cost_sanity.csv"
    trigger_path = P05_SANITY_DIR / "distance_trigger_stats.csv"
    wide_path = P05_WIDE_DIR / "distance_wide_by_step_scenario.csv"

    lines = [
        "## Interaction with P0.5 Distance Sanity / Wide Distance Ablation",
        "",
    ]
    if sanity_path.exists():
        sanity = pd.read_csv(sanity_path)
        bug_count = int(sanity["judgment"].astype(str).str.startswith("BUG").sum())
        zero_ok = int((sanity["judgment"] == "OK_ZERO_REASONABLE_MIN_DISTANCE_ABOVE_WARNING").sum())
        lines.append(
            f"1. distance_warning_cost=0 implementation check: {bug_count} bug-like rows found; "
            f"{zero_ok}/{len(sanity)} rows are zero because min_distance stayed above d_warning=1.0."
        )
    else:
        lines.append("1. distance_warning_cost=0 implementation check: P0.5-A not available yet.")

    if trigger_path.exists():
        trigger = pd.read_csv(trigger_path)
        max_rates = trigger.groupby("method")["distance_warning_cost_nonzero_rate"].max().to_dict()
        rate_text = ", ".join(f"{method} max={fmt(value, 3)}" for method, value in max_rates.items())
        sparse = all(float(value) < 0.1 for value in max_rates.values())
        lines.append(
            f"2. d_warning=1.0 trigger rate: {rate_text}. "
            + ("This supports the sparse-trigger interpretation." if sparse else "The trigger is not uniformly sparse.")
        )
    else:
        lines.append("2. d_warning=1.0 trigger rate: P0.5-B not available yet.")

    if wide_path.exists():
        wide = pd.read_csv(wide_path)
        sudden_750 = wide[(wide["scenario"] == "eval_sudden_turn") & (wide["step"] == 750000)]

        def get(method: str, col: str) -> float:
            hit = sudden_750[sudden_750["method"] == method]
            if hit.empty:
                return float("nan")
            return float(hit.iloc[0][col])

        d1 = get("attention_full_distance_penalty_d1", "reaction_time_eval_style")
        d2 = get("attention_full_distance_penalty_wide_d2", "reaction_time_eval_style")
        risk = get("attention_full_risk_penalty", "reaction_time_eval_style")
        lines.append(
            f"3. d_warning=2.0 ablation at 750k sudden_turn: d1={fmt(d1)} s, wide_d2={fmt(d2)} s, risk={fmt(risk)} s."
        )
        if not math.isnan(d2) and not math.isnan(risk) and d2 <= risk + 1.0:
            mechanism = "wider/dense safety-margin support is sufficient to approach risk_penalty in this ablation."
        elif not math.isnan(d2) and not math.isnan(d1) and d2 < d1:
            mechanism = "wider distance support helps, but risk_penalty retains additional value."
        else:
            mechanism = "risk_penalty appears to add value beyond simply widening the distance threshold."
        lines.append(f"4. Current mechanism interpretation: {mechanism}")
    else:
        lines.append("3. d_warning=2.0 ablation: P0.5-C not available yet.")
        lines.append(
            "4. Current mechanism interpretation: P0 trace does not support a strict post-turn early-warning mechanism; "
            "the working hypothesis is dense safety-margin regularization / wider cost support until P0.5-C finishes."
        )
    lines.append("")
    return lines


def update_p1_report_with_p05() -> None:
    section = "\n".join(p05_interaction_section()).rstrip() + "\n"
    report = ROOT / "P1_THREE_SEED_REPLICATION_REPORT.md"
    if not report.exists():
        write_lines(P05_SANITY_DIR / "P1_INTERACTION_SECTION_PENDING.md", section.splitlines())
        return
    text = report.read_text(encoding="utf-8")
    marker = "## Interaction with P0.5 Distance Sanity / Wide Distance Ablation"
    artifacts_marker = "\n## Artifacts"
    if marker in text:
        before = text.split(marker, 1)[0].rstrip()
        rest = text.split(marker, 1)[1]
        after = ""
        if artifacts_marker in rest:
            after = artifacts_marker + rest.split(artifacts_marker, 1)[1]
        text = before + "\n\n" + section + ("\n" + after.lstrip("\n") if after else "")
    elif artifacts_marker in text:
        before, after = text.split(artifacts_marker, 1)
        text = before.rstrip() + "\n\n" + section + "\n## Artifacts" + after
    else:
        text = text.rstrip() + "\n\n" + section
    report.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ab", "wide", "all", "update-p1"], default="ab")
    parser.add_argument("--n_envs", type=int, default=int(os.environ.get("P05_WIDE_N_ENVS", "16")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.mode in {"ab", "all"}:
        run_ab()
        update_p1_report_with_p05()
    if args.mode in {"wide", "all"}:
        run_wide(n_envs=args.n_envs)
    if args.mode == "update-p1":
        update_p1_report_with_p05()


if __name__ == "__main__":
    main()
