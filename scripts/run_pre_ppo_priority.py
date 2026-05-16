from __future__ import annotations

import json
import math
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_OUT = ROOT / "results/pre_ppo_priority"
P1_OUT = BASE_OUT / "pminus1_pareto_audit"
P0_OUT = BASE_OUT / "p0_adaptation"
P05_OUT = BASE_OUT / "p0_5_beta_sweep"
LOG_DIR = ROOT / "runs/logs/pre_ppo_priority"
STATUS_PATH = BASE_OUT / "status.json"
COMPLETE_FLAG = ROOT / "PRE_PPO_PRIORITY_COMPLETE.flag"
NO_GO_FLAG = ROOT / "PRE_PPO_PRIORITY_NO_GO.flag"

CHECKPOINT_STEPS = [250000, 500000, 750000]
P_MINUS_1_STEPS = [750000, 1000000]
EVAL_SEED = 1000
EPISODES = 50
N_ENVS = 16

SCENARIOS_BASE = [
    "eval_random_switch",
    "eval_sudden_turn",
    "eval_random_switch_hard",
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_mixed_v2",
    "eval_threat_validated_sudden",
]

SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "eval_sinusoidal": "sinusoidal",
    "eval_accel_decel": "accel",
    "eval_ar1": "ar1",
    "eval_mixed_v2": "mixed_v2",
    "eval_threat_validated_sudden": "threat_sudden",
}

CORE_METRICS = [
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "mean_min_distance",
    "mean_time",
]


@dataclass(frozen=True)
class TrainSpec:
    key: str
    method: str
    display_method: str
    agg: str
    scenario: str
    cost_type: str
    d_warning: float
    beta_cost: float
    checkpoint_dir: Path
    log_dir: Path
    run_name: str
    save_path: Path
    variant: str = "default"
    beta_label: str = "5.0"
    use_safety_cost: bool = True


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for path in [BASE_OUT, P1_OUT, P0_OUT, P05_OUT, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
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


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def update_status(
    stage: str,
    current_run: str = "",
    latest_checkpoint: str = "",
    completed_eval_count: int = 0,
    pending_eval_count: int = 0,
    watcher_status: str = "running",
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at": now(),
        "current_stage": stage,
        "current_run": current_run,
        "latest_checkpoint": latest_checkpoint,
        "completed_eval_count": int(completed_eval_count),
        "pending_eval_count": int(pending_eval_count),
        "watcher_status": watcher_status,
    }
    if extra:
        payload.update(extra)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], log_path: Path, skip_marker: str | None = None) -> None:
    if skip_marker and log_path.exists() and skip_marker in log_path.read_text(encoding="utf-8", errors="ignore"):
        print(f"[{now()}] SKIP completed log={rel(log_path)}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] RUN {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def read_p2_source() -> pd.DataFrame:
    candidates = [
        ROOT / "results/p2_rich_motion/p2_three_seed_summary.csv",
        ROOT / "results/p2_rich_motion/p2_seed0_by_step_scenario.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    raise FileNotFoundError("No P2 summary CSV found under results/p2_rich_motion")


def reaction_bad(delta: float) -> bool:
    return bool(np.isfinite(delta) and delta > 0.30)


def reaction_small(delta: float) -> bool:
    return bool((not np.isfinite(delta)) or abs(delta) < 0.10)


def classify_delta(row: pd.Series) -> str:
    dt = float(row["delta_mean_time"])
    dn = float(row["delta_near_miss"])
    dmin = float(row["delta_min_distance"])
    dc = float(row["delta_collision"])
    ds = float(row["delta_success"])
    dr = float(row["delta_reaction"]) if pd.notna(row["delta_reaction"]) else float("nan")

    if dt <= -0.20 and dc <= 0.01 and ds >= -0.02 and dn <= 0.05:
        return "risk_pareto_positive"
    if dt <= -0.20 and (dn > 0.05 or dc > 0.01 or reaction_bad(dr)):
        return "risk_efficiency_but_safety_gap"
    if abs(dt) < 0.10 and abs(dn) < 0.02 and abs(dc) < 0.01 and reaction_small(dr):
        return "no_meaningful_difference"
    if dt >= -0.10 and dn >= 0.0 and dc >= 0.0 and dmin <= 0.0:
        return "wide_d2_dominates"
    if dt < -0.10:
        return "risk_efficiency_but_safety_gap" if (dn > 0.05 or dc > 0.01 or reaction_bad(dr)) else "risk_pareto_positive"
    return "no_meaningful_difference"


def run_pminus1() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    update_status("Stage P-1", "P2 Pareto audit", "", 0, 0, "auditing")
    df = read_p2_source()
    required_methods = {
        "attention_full_distance_penalty_wide_d2",
        "attention_full_risk_penalty",
    }
    missing_methods = sorted(required_methods - set(df["method"].astype(str)))
    if missing_methods:
        raise RuntimeError(f"P2 source missing methods: {missing_methods}")
    use = df[
        df["method"].isin(required_methods)
        & df["step"].isin(P_MINUS_1_STEPS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (seed, step, scenario), group in use.groupby(["seed", "step", "scenario"], sort=True):
        wide = group[group["method"] == "attention_full_distance_penalty_wide_d2"]
        risk = group[group["method"] == "attention_full_risk_penalty"]
        if wide.empty or risk.empty:
            continue
        w = wide.iloc[0]
        r = risk.iloc[0]
        row = {
            "scenario": scenario,
            "seed": int(seed),
            "step": int(step),
            "risk_mean_time": float(r["mean_time"]),
            "wide_d2_mean_time": float(w["mean_time"]),
            "delta_mean_time": float(r["mean_time"]) - float(w["mean_time"]),
            "risk_near_miss": float(r["near_miss_rate"]),
            "wide_d2_near_miss": float(w["near_miss_rate"]),
            "delta_near_miss": float(r["near_miss_rate"]) - float(w["near_miss_rate"]),
            "risk_min_distance": float(r["mean_min_distance"]),
            "wide_d2_min_distance": float(w["mean_min_distance"]),
            "delta_min_distance": float(r["mean_min_distance"]) - float(w["mean_min_distance"]),
            "risk_collision": float(r["collision_rate"]),
            "wide_d2_collision": float(w["collision_rate"]),
            "delta_collision": float(r["collision_rate"]) - float(w["collision_rate"]),
            "risk_success": float(r["success_rate"]),
            "wide_d2_success": float(w["success_rate"]),
            "delta_success": float(r["success_rate"]) - float(w["success_rate"]),
            "risk_reaction": float(r["reaction_time_eval_style"]) if pd.notna(r["reaction_time_eval_style"]) else np.nan,
            "wide_d2_reaction": float(w["reaction_time_eval_style"]) if pd.notna(w["reaction_time_eval_style"]) else np.nan,
            "delta_reaction": (
                float(r["reaction_time_eval_style"]) - float(w["reaction_time_eval_style"])
                if pd.notna(r["reaction_time_eval_style"]) and pd.notna(w["reaction_time_eval_style"])
                else np.nan
            ),
            "risk_source_csv": str(r.get("source_csv", "")),
            "wide_d2_source_csv": str(w.get("source_csv", "")),
        }
        rows.append(row)

    delta = pd.DataFrame(rows).sort_values(["scenario", "seed", "step"])
    if delta.empty:
        raise RuntimeError("P-1 audit produced no risk-vs-wide rows")
    delta["classification"] = delta.apply(classify_delta, axis=1)
    delta_path = P1_OUT / "p2_delta_by_seed_step_scenario.csv"
    class_path = P1_OUT / "p2_pareto_classification.csv"
    delta.to_csv(delta_path, index=False)
    delta[
        [
            "scenario",
            "seed",
            "step",
            "classification",
            "delta_mean_time",
            "delta_near_miss",
            "delta_min_distance",
            "delta_collision",
            "delta_success",
            "delta_reaction",
        ]
    ].to_csv(class_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for scenario, group in delta.groupby("scenario", sort=True):
        row: dict[str, Any] = {
            "scenario": scenario,
            "rows": int(len(group)),
            "seed_count": int(group["seed"].nunique()),
            "step_count": int(group["step"].nunique()),
            "delta_mean_time_mean": float(group["delta_mean_time"].mean()),
            "delta_near_miss_mean": float(group["delta_near_miss"].mean()),
            "delta_min_distance_mean": float(group["delta_min_distance"].mean()),
            "delta_collision_mean": float(group["delta_collision"].mean()),
            "delta_success_mean": float(group["delta_success"].mean()),
            "delta_reaction_mean": float(group["delta_reaction"].mean()) if group["delta_reaction"].notna().any() else np.nan,
        }
        counts = group["classification"].value_counts().to_dict()
        for cls in [
            "risk_pareto_positive",
            "risk_efficiency_but_safety_gap",
            "wide_d2_dominates",
            "no_meaningful_difference",
        ]:
            row[f"{cls}_count"] = int(counts.get(cls, 0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("scenario")
    summary.to_csv(P1_OUT / "p2_delta_summary_by_scenario.csv", index=False)

    pareto_scenarios = sorted(summary.loc[summary["risk_pareto_positive_count"] > 0, "scenario"].astype(str))
    dominated_scenarios = sorted(summary.loc[summary["wide_d2_dominates_count"] > summary["rows"] / 2.0, "scenario"].astype(str))
    no_go = len(pareto_scenarios) == 0 and len(dominated_scenarios) > len(summary) / 2.0

    lines = [
        "# Stage P-1 P2 Pareto Audit Report",
        "",
        "## Scope",
        "- Source: results/p2_rich_motion/p2_three_seed_summary.csv when available.",
        "- Compared attention_full_risk_penalty against attention_full_distance_penalty_wide_d2.",
        "- Primary split: scenario / seed / checkpoint at 750k and 1000k.",
        "",
        "## Classification Summary",
        f"- Pareto-positive scenarios: {', '.join(pareto_scenarios) if pareto_scenarios else 'none'}",
        f"- Majority wide_d2-dominated scenarios: {', '.join(dominated_scenarios) if dominated_scenarios else 'none'}",
        f"- P-1 no-go triggered: {no_go}",
        "",
        "| scenario | rows | pareto_positive | efficiency_safety_gap | wide_d2_dominates | no_difference | d_time | d_near | d_collision | d_min_distance | d_reaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scenario']} | {int(row['rows'])} | {int(row['risk_pareto_positive_count'])} | "
            f"{int(row['risk_efficiency_but_safety_gap_count'])} | {int(row['wide_d2_dominates_count'])} | "
            f"{int(row['no_meaningful_difference_count'])} | {fmt(row['delta_mean_time_mean'])} | "
            f"{fmt(row['delta_near_miss_mean'])} | {fmt(row['delta_collision_mean'])} | "
            f"{fmt(row['delta_min_distance_mean'])} | {fmt(row['delta_reaction_mean'])} |"
        )
    lines += [
        "",
        "## Required Answers",
        f"1. risk Pareto-positive evidence appears in: {', '.join(pareto_scenarios) if pareto_scenarios else 'none'}.",
        f"2. wide_d2 majority dominance appears in: {', '.join(dominated_scenarios) if dominated_scenarios else 'none'}.",
        "3. Whether risk is merely faster with safety loss is shown by risk_efficiency_but_safety_gap counts above.",
        "4. 750k and 1000k consistency is available in p2_pareto_classification.csv.",
        "5. Seed-level stability is available in p2_delta_by_seed_step_scenario.csv.",
        f"6. Continue to P0/P0.5: {not no_go}.",
    ]
    write_text(ROOT / "P_MINUS_1_P2_PARETO_AUDIT_REPORT.md", lines)
    return delta, summary, no_go


def write_no_go(reason: str, stage: str, metrics: dict[str, Any], recommendation: str) -> None:
    payload = {
        "terminal_decision": reason,
        "stage": stage,
        "completed_at": now(),
        "metrics": metrics,
    }
    NO_GO_FLAG.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Pre-PPO Priority Experiments Final Report",
        "",
        "## No-Go",
        f"- Stage: {stage}",
        f"- terminal_decision: {reason}",
        "",
        "## Trigger Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Why Stop",
        "The guide allows stopping only on explicit no-go conditions. This run reached one of those conditions, so later stages must not be used as evidence in this cycle.",
        "",
        "## Recommendation",
        recommendation,
    ]
    write_text(ROOT / "PRE_PPO_PRIORITY_FINAL_REPORT.md", lines)
    update_status(stage, reason, "", 0, 0, "no_go", {"terminal_decision": reason})


def checkpoints_for(spec: TrainSpec) -> list[Path]:
    return [spec.checkpoint_dir / f"{spec.run_name}_s0_step{step}.zip" for step in CHECKPOINT_STEPS]


def train_if_needed(spec: TrainSpec, total_steps: int = 750000) -> None:
    expected = checkpoints_for(spec)
    if all(path.exists() and path.stat().st_size > 0 for path in expected):
        print(f"[{now()}] SKIP training existing run={spec.key}", flush=True)
        return
    cmd = [
        PYTHON,
        "train.py",
        "--method",
        spec.method,
        "--profile_mode",
        "full_12",
        "--agg",
        spec.agg,
        "--seed",
        "0",
        "--total_steps",
        str(total_steps),
        "--n_envs",
        str(N_ENVS),
        "--device",
        "cpu",
        "--scenario",
        spec.scenario,
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        ",".join(str(step) for step in CHECKPOINT_STEPS),
        "--checkpoint_dir",
        str(spec.checkpoint_dir),
        "--log_dir",
        str(spec.log_dir),
        "--run_name",
        spec.run_name,
        "--save_path",
        str(spec.save_path),
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
    update_status("training", spec.key, "", 0, 0, "training")
    run_command(cmd, LOG_DIR / f"train_{spec.key}.log", skip_marker="TRAIN_END")


def eval_csv_path(out_dir: Path, spec: TrainSpec, step: int, scenario: str) -> Path:
    suffix = SCENARIO_SUFFIX[base_scenario(scenario)]
    return out_dir / "eval" / f"{spec.run_name}_s0_step{step}_{suffix}.csv"


def base_scenario(scenario: str) -> str:
    if scenario.endswith("_high_speed"):
        return scenario[: -len("_high_speed")]
    if scenario.endswith("_small_space"):
        return scenario[: -len("_small_space")]
    return scenario


def summarize_eval_csv(path: Path, spec: TrainSpec, step: int, scenario: str, stage: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    reaction = pd.to_numeric(df.get("reaction_time_eval_style", pd.Series(dtype=float)), errors="coerce")
    reaction_nan = pd.to_numeric(df.get("reaction_time_nan_style", pd.Series(dtype=float)), errors="coerce")
    cost_max = pd.to_numeric(df.get("distance_warning_cost_max", pd.Series(dtype=float)), errors="coerce")

    def mean_col(name: str) -> float:
        if name not in df:
            return float("nan")
        values = pd.to_numeric(df[name], errors="coerce")
        return float(values.mean()) if values.notna().any() else float("nan")

    return {
        "stage": stage,
        "variant": spec.variant,
        "method": spec.display_method,
        "method_key": spec.key,
        "beta_cost": float(spec.beta_cost),
        "beta_label": spec.beta_label,
        "cost_type": spec.cost_type,
        "d_warning": float(spec.d_warning),
        "seed": 0,
        "step": int(step),
        "scenario": scenario,
        "base_scenario": base_scenario(scenario),
        "episodes": int(len(df)),
        "success_rate": mean_col("success"),
        "collision_rate": mean_col("collision"),
        "near_miss_rate": mean_col("near_miss"),
        "mean_min_distance": mean_col("episode_min_distance"),
        "mean_time": mean_col("time_to_goal"),
        "mean_episode_reward": mean_col("episode_reward"),
        "reaction_time_eval_style": float(reaction.mean()) if reaction.notna().any() else float("nan"),
        "reaction_time_nan_style": float(reaction_nan.mean()) if reaction_nan.notna().any() else float("nan"),
        "nan_reaction_rate": float(reaction_nan.isna().mean()) if len(reaction_nan) else float("nan"),
        "distance_warning_cost_nonzero_rate": float((cost_max > 0.0).mean()) if len(cost_max) else float("nan"),
        "distance_warning_cost_mean": mean_col("distance_warning_cost_mean"),
        "distance_warning_cost_p95": mean_col("distance_warning_cost_p95"),
        "distance_warning_cost_max": float(cost_max.max()) if cost_max.notna().any() else float("nan"),
        "risk_sum_mean": mean_col("risk_sum_mean"),
        "risk_sum_p95": mean_col("risk_sum_p95"),
        "risk_max_mean": mean_col("risk_max_mean"),
        "scenario_valid_rate": mean_col("scenario_valid"),
        "planned_threat_valid_rate": mean_col("planned_threat_valid"),
        "threat_valid_rate": mean_col("threat_valid"),
        "realized_near_miss_rate": mean_col("realized_near_miss"),
        "source_csv": rel(path),
        "checkpoint_path": rel(spec.checkpoint_dir / f"{spec.run_name}_s0_step{step}.zip"),
    }


def eval_if_needed(
    spec: TrainSpec,
    step: int,
    scenario: str,
    out_dir: Path,
    completed: int,
    pending: int,
    stage: str,
) -> Path:
    out_csv = eval_csv_path(out_dir, spec, step, scenario)
    if csv_row_count(out_csv) >= EPISODES:
        return out_csv
    checkpoint = spec.checkpoint_dir / f"{spec.run_name}_s0_step{step}.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(checkpoint),
        "--method",
        spec.display_method,
        "--profile_mode",
        "full_12",
        "--agg",
        spec.agg,
        "--seed",
        "0",
        "--eval_seed",
        str(EVAL_SEED),
        "--episodes",
        str(EPISODES),
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
    update_status(stage, spec.key, str(step), completed, pending, "evaluating")
    log_path = LOG_DIR / f"eval_{stage}_{spec.key}_step{step}_{SCENARIO_SUFFIX[base_scenario(scenario)]}.log"
    run_command(cmd, log_path, skip_marker="EVAL_END")
    return out_csv


def p0_specs() -> list[TrainSpec]:
    ckpt = ROOT / "checkpoints/pre_ppo_priority/p0_adaptation"
    run_root = ROOT / "runs/pre_ppo_priority/p0_adaptation"
    return [
        TrainSpec(
            key="high_speed_wide_d2",
            method="attention_full_distance_penalty_wide_d2",
            display_method="attention_full_distance_penalty_wide_d2",
            agg="attention",
            scenario="train_mixed_modes_v2_high_speed",
            cost_type="distance_warning",
            d_warning=2.0,
            beta_cost=5.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "high_speed_wide_d2_s0",
            run_name="high_speed_wide_d2",
            save_path=ckpt / "high_speed_wide_d2_s0_step750000.zip",
            variant="high_speed",
            beta_label="5.0",
        ),
        TrainSpec(
            key="high_speed_risk",
            method="attention_full_risk_penalty",
            display_method="attention_full_risk_penalty",
            agg="attention",
            scenario="train_mixed_modes_v2_high_speed",
            cost_type="risk_sum",
            d_warning=1.0,
            beta_cost=5.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "high_speed_risk_s0",
            run_name="high_speed_risk",
            save_path=ckpt / "high_speed_risk_s0_step750000.zip",
            variant="high_speed",
            beta_label="5.0",
        ),
        TrainSpec(
            key="small_space_wide_d2",
            method="attention_full_distance_penalty_wide_d2",
            display_method="attention_full_distance_penalty_wide_d2",
            agg="attention",
            scenario="train_mixed_modes_v2_small_space",
            cost_type="distance_warning",
            d_warning=2.0,
            beta_cost=5.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "small_space_wide_d2_s0",
            run_name="small_space_wide_d2",
            save_path=ckpt / "small_space_wide_d2_s0_step750000.zip",
            variant="small_space",
            beta_label="5.0",
        ),
        TrainSpec(
            key="small_space_risk",
            method="attention_full_risk_penalty",
            display_method="attention_full_risk_penalty",
            agg="attention",
            scenario="train_mixed_modes_v2_small_space",
            cost_type="risk_sum",
            d_warning=1.0,
            beta_cost=5.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "small_space_risk_s0",
            run_name="small_space_risk",
            save_path=ckpt / "small_space_risk_s0_step750000.zip",
            variant="small_space",
            beta_label="5.0",
        ),
    ]


def p05_specs() -> list[TrainSpec]:
    ckpt = ROOT / "checkpoints/pre_ppo_priority/p0_5_beta_sweep"
    run_root = ROOT / "runs/pre_ppo_priority/p0_5_beta_sweep"
    return [
        TrainSpec(
            key="wide_d2_beta2",
            method="attention_full_distance_penalty_wide_d2_beta2",
            display_method="attention_full_distance_penalty_wide_d2_beta2",
            agg="attention",
            scenario="train_mixed_modes_v2",
            cost_type="distance_warning",
            d_warning=2.0,
            beta_cost=2.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "wide_d2_beta2_s0",
            run_name="wide_d2_beta2",
            save_path=ckpt / "wide_d2_beta2_s0_step750000.zip",
            beta_label="2.0",
        ),
        TrainSpec(
            key="wide_d2_beta10",
            method="attention_full_distance_penalty_wide_d2_beta10",
            display_method="attention_full_distance_penalty_wide_d2_beta10",
            agg="attention",
            scenario="train_mixed_modes_v2",
            cost_type="distance_warning",
            d_warning=2.0,
            beta_cost=10.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "wide_d2_beta10_s0",
            run_name="wide_d2_beta10",
            save_path=ckpt / "wide_d2_beta10_s0_step750000.zip",
            beta_label="10.0",
        ),
        TrainSpec(
            key="risk_beta2",
            method="attention_full_risk_penalty_beta2",
            display_method="attention_full_risk_penalty_beta2",
            agg="attention",
            scenario="train_mixed_modes_v2",
            cost_type="risk_sum",
            d_warning=1.0,
            beta_cost=2.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "risk_beta2_s0",
            run_name="risk_beta2",
            save_path=ckpt / "risk_beta2_s0_step750000.zip",
            beta_label="2.0",
        ),
        TrainSpec(
            key="risk_beta10",
            method="attention_full_risk_penalty_beta10",
            display_method="attention_full_risk_penalty_beta10",
            agg="attention",
            scenario="train_mixed_modes_v2",
            cost_type="risk_sum",
            d_warning=1.0,
            beta_cost=10.0,
            checkpoint_dir=ckpt,
            log_dir=run_root / "risk_beta10_s0",
            run_name="risk_beta10",
            save_path=ckpt / "risk_beta10_s0_step750000.zip",
            beta_label="10.0",
        ),
    ]


def variant_eval_scenarios(variant: str) -> list[str]:
    return [f"{scenario}_{variant}" for scenario in SCENARIOS_BASE]


def compute_delta_table(
    df: pd.DataFrame,
    group_cols: list[str],
    risk_method: str,
    wide_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        risk = group[group["method"] == risk_method]
        wide = group[group["method"] == wide_method]
        if risk.empty or wide.empty:
            continue
        r = risk.iloc[0]
        w = wide.iloc[0]
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(
            {
                "risk_mean_time": float(r["mean_time"]),
                "wide_d2_mean_time": float(w["mean_time"]),
                "delta_mean_time": float(r["mean_time"]) - float(w["mean_time"]),
                "delta_near_miss": float(r["near_miss_rate"]) - float(w["near_miss_rate"]),
                "delta_min_distance": float(r["mean_min_distance"]) - float(w["mean_min_distance"]),
                "delta_collision": float(r["collision_rate"]) - float(w["collision_rate"]),
                "delta_success": float(r["success_rate"]) - float(w["success_rate"]),
                "delta_reaction": (
                    float(r["reaction_time_eval_style"]) - float(w["reaction_time_eval_style"])
                    if pd.notna(r["reaction_time_eval_style"]) and pd.notna(w["reaction_time_eval_style"])
                    else np.nan
                ),
                "risk_pareto_positive": bool(
                    float(r["mean_time"]) - float(w["mean_time"]) <= -0.20
                    and float(r["collision_rate"]) - float(w["collision_rate"]) <= 0.01
                    and float(r["success_rate"]) - float(w["success_rate"]) >= -0.02
                    and float(r["near_miss_rate"]) - float(w["near_miss_rate"]) <= 0.05
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_p0_environment_sanity() -> tuple[bool, pd.DataFrame, list[str]]:
    from envs.dynamic_obstacle_env import DynamicObstacleEnv

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for variant in ["high_speed", "small_space"]:
        for scenario in variant_eval_scenarios(variant):
            scenario_valid = []
            threat_valid = []
            init_collision = []
            for episode_id in range(20):
                env = DynamicObstacleEnv(scenario=scenario)
                _, info = env.reset(seed=7000 + episode_id)
                scenario_valid.append(float(bool(info.get("scenario_valid", True))))
                threat_valid.append(float(bool(info.get("threat_valid", True))))
                init_collision.append(float(float(info.get("initial_min_distance", 999.0)) < env.d_collision))
            row = {
                "variant": variant,
                "scenario": scenario,
                "scenario_valid_rate": float(np.mean(scenario_valid)),
                "threat_valid_rate": float(np.mean(threat_valid)),
                "init_collision_rate": float(np.mean(init_collision)),
            }
            rows.append(row)
            if row["scenario_valid_rate"] < 0.95:
                reasons.append(f"{scenario}:scenario_valid_rate={row['scenario_valid_rate']:.3f}")
            if base_scenario(scenario) in {"eval_mixed_v2", "eval_threat_validated_sudden"} and row["threat_valid_rate"] < 0.80:
                reasons.append(f"{scenario}:threat_valid_rate={row['threat_valid_rate']:.3f}")
            if row["init_collision_rate"] > 0.05:
                reasons.append(f"{scenario}:init_collision_rate={row['init_collision_rate']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(P0_OUT / "p0_environment_sanity.csv", index=False)
    return not reasons, out, reasons


def p0_judgments(df: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final = df[df["step"] == 750000].copy()
    for variant, group in final.groupby("variant", sort=True):
        wide = group[group["method"] == "attention_full_distance_penalty_wide_d2"]
        risk = group[group["method"] == "attention_full_risk_penalty"]
        d = delta[(delta["variant"] == variant) & (delta["step"] == 750000)]
        if wide.empty or risk.empty:
            continue
        risk_pareto_count = int(d["risk_pareto_positive"].sum()) if "risk_pareto_positive" in d else 0
        risk_safety_close_count = int(
            (
                (d["delta_collision"] <= 0.01)
                & (d["delta_success"] >= -0.02)
                & (d["delta_near_miss"] <= 0.05)
            ).sum()
        )
        wide_mean = wide.mean(numeric_only=True)
        risk_mean = risk.mean(numeric_only=True)
        wide_overconservative = bool(
            float(wide_mean["mean_time"]) - float(risk_mean["mean_time"]) >= 0.30
            and float(risk_mean["collision_rate"]) <= float(wide_mean["collision_rate"]) + 0.01
            and float(risk_mean["near_miss_rate"]) <= float(wide_mean["near_miss_rate"]) + 0.05
        )
        wide_failure = bool(
            float(wide_mean["collision_rate"]) >= float(risk_mean["collision_rate"]) + 0.05
            or float(wide_mean["near_miss_rate"]) >= float(risk_mean["near_miss_rate"]) + 0.10
            or float(wide_mean["success_rate"]) <= float(risk_mean["success_rate"]) - 0.10
        )
        both_fail = bool(float(wide_mean["success_rate"]) < 0.50 and float(risk_mean["success_rate"]) < 0.50)
        risk_adaptation_supported = bool((wide_failure or wide_overconservative) and risk_pareto_count > 0)
        both_work_but_risk_faster = bool(
            risk_pareto_count > 0
            and not wide_failure
            and float(wide_mean["success_rate"]) >= 0.80
            and float(risk_mean["success_rate"]) >= 0.80
        )
        fixed_margin_still_strong = bool(
            not wide_failure
            and not wide_overconservative
            and float(wide_mean["success_rate"]) >= float(risk_mean["success_rate"]) - 0.05
            and float(wide_mean["collision_rate"]) <= float(risk_mean["collision_rate"]) + 0.01
            and float(wide_mean["near_miss_rate"]) <= float(risk_mean["near_miss_rate"]) + 0.05
        )
        rows.append(
            {
                "variant": variant,
                "risk_pareto_positive_scenarios_750k": risk_pareto_count,
                "risk_safety_close_scenarios_750k": risk_safety_close_count,
                "wide_overconservative": wide_overconservative,
                "wide_failure": wide_failure,
                "both_fail": both_fail,
                "both_work_but_risk_faster": both_work_but_risk_faster,
                "fixed_margin_still_strong": fixed_margin_still_strong,
                "risk_adaptation_supported": risk_adaptation_supported,
                "wide_mean_time_750k": float(wide_mean["mean_time"]),
                "risk_mean_time_750k": float(risk_mean["mean_time"]),
                "wide_success_750k": float(wide_mean["success_rate"]),
                "risk_success_750k": float(risk_mean["success_rate"]),
                "wide_collision_750k": float(wide_mean["collision_rate"]),
                "risk_collision_750k": float(risk_mean["collision_rate"]),
                "wide_near_miss_750k": float(wide_mean["near_miss_rate"]),
                "risk_near_miss_750k": float(risk_mean["near_miss_rate"]),
            }
        )
    return pd.DataFrame(rows)


def run_p0() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    update_status("Stage P0", "environment_sanity", "", 0, 0, "sanity")
    ok, sanity, reasons = run_p0_environment_sanity()
    if not ok:
        write_no_go(
            "p0_no_go_environment_invalid",
            "Stage P0",
            {"reasons": "; ".join(reasons), "sanity_rows": len(sanity)},
            "Do not continue with these variant results. Fix the variant sampling/validity first, then rerun P0 before deciding whether to keep risk or shift to safety-margin cost design principles.",
        )
        raise SystemExit(0)

    specs = p0_specs()
    rows: list[dict[str, Any]] = []
    total_evals = len(specs) * len(CHECKPOINT_STEPS) * len(SCENARIOS_BASE)
    completed = 0
    for spec in specs:
        train_if_needed(spec)
        for step in CHECKPOINT_STEPS:
            for scenario in variant_eval_scenarios(spec.variant):
                path = eval_if_needed(spec, step, scenario, P0_OUT, completed, total_evals - completed, "Stage P0")
                rows.append(summarize_eval_csv(path, spec, step, scenario, "P0"))
                completed += 1

    df = pd.DataFrame(rows).sort_values(["variant", "method", "step", "base_scenario"])
    df.to_csv(P0_OUT / "p0_adaptation_by_variant_method_step_scenario.csv", index=False)
    delta = compute_delta_table(
        df,
        ["variant", "step", "base_scenario"],
        "attention_full_risk_penalty",
        "attention_full_distance_penalty_wide_d2",
    ).sort_values(["variant", "step", "base_scenario"])
    delta.to_csv(P0_OUT / "p0_adaptation_delta_risk_minus_wide_d2.csv", index=False)
    judgments = p0_judgments(df, delta)

    summary_rows: list[dict[str, Any]] = []
    for (variant, method, step), group in df.groupby(["variant", "method", "step"], sort=True):
        mean = group.mean(numeric_only=True)
        summary_rows.append(
            {
                "variant": variant,
                "method": method,
                "step": int(step),
                "scenario_count": int(group["base_scenario"].nunique()),
                "success_rate": float(mean["success_rate"]),
                "collision_rate": float(mean["collision_rate"]),
                "near_miss_rate": float(mean["near_miss_rate"]),
                "mean_min_distance": float(mean["mean_min_distance"]),
                "mean_time": float(mean["mean_time"]),
                "reaction_time_eval_style": float(mean["reaction_time_eval_style"]) if "reaction_time_eval_style" in mean else np.nan,
                "scenario_valid_rate": float(mean["scenario_valid_rate"]),
                "threat_valid_rate": float(mean["threat_valid_rate"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["variant", "method", "step"])
    if not judgments.empty:
        summary = summary.merge(judgments, on="variant", how="left")
    summary.to_csv(P0_OUT / "p0_adaptation_summary.csv", index=False)

    both_fail_all = bool(not judgments.empty and judgments["both_fail"].all())
    if both_fail_all:
        write_no_go(
            "p0_no_go_both_methods_fail_all_variants",
            "Stage P0",
            judgments.to_dict("records")[0] if len(judgments) == 1 else {"judgments": judgments.to_dict("records")},
            "Both fixed-margin and risk penalties failed every P0 variant. Keep neither result as a risk-mainline argument; revise environment difficulty or turn to safety-margin cost design principles.",
        )
        raise SystemExit(0)

    lines = [
        "# Stage P0 Adaptation Validation Report",
        "",
        "## Scope",
        "- Variants: high_speed_obstacles and small_space.",
        "- Methods: attention_full_distance_penalty_wide_d2 and attention_full_risk_penalty.",
        "- Seed 0, 750k training, checkpoints 250k/500k/750k, 50 eval episodes per scenario.",
        "",
        "## Variant Judgments at 750k",
        "| variant | risk_pareto_scenarios | wide_failure | wide_overconservative | fixed_margin_still_strong | risk_adaptation_supported | wide_time | risk_time | wide_near | risk_near |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in judgments.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['risk_pareto_positive_scenarios_750k'])} | "
            f"{int(row['wide_failure'])} | {int(row['wide_overconservative'])} | "
            f"{int(row['fixed_margin_still_strong'])} | {int(row['risk_adaptation_supported'])} | "
            f"{fmt(row['wide_mean_time_750k'])} | {fmt(row['risk_mean_time_750k'])} | "
            f"{fmt(row['wide_near_miss_750k'])} | {fmt(row['risk_near_miss_750k'])} |"
        )
    lines += [
        "",
        "## Required Answers",
    ]
    for variant in ["high_speed", "small_space"]:
        row = judgments[judgments["variant"] == variant]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"- {variant}: wide_d2 failure={bool(r['wide_failure'])}, overconservative={bool(r['wide_overconservative'])}, "
            f"risk Pareto scenarios={int(r['risk_pareto_positive_scenarios_750k'])}, "
            f"risk_adaptation_supported={bool(r['risk_adaptation_supported'])}."
        )
    write_text(ROOT / "P0_ADAPTATION_VALIDATION_REPORT.md", lines)
    return df, delta, summary, bool(judgments["risk_adaptation_supported"].any()) if not judgments.empty else False


def reference_beta5_rows() -> pd.DataFrame:
    path = ROOT / "results/p2_rich_motion/p2_seed0_by_step_scenario.csv"
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_csv(path)
    source = source[
        source["seed"].eq(0)
        & source["step"].isin(CHECKPOINT_STEPS)
        & source["method"].isin(
            [
                "attention_full_distance_penalty_wide_d2",
                "attention_full_risk_penalty",
            ]
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        method = str(row["method"])
        if method == "attention_full_distance_penalty_wide_d2":
            method_key = "wide_d2_beta5_reference"
            display = "attention_full_distance_penalty_wide_d2_beta5_reference"
            beta = 5.0
            cost_type = "distance_warning"
            d_warning = 2.0
        else:
            method_key = "risk_beta5_reference"
            display = "attention_full_risk_penalty_beta5_reference"
            beta = 5.0
            cost_type = "risk_sum"
            d_warning = 1.0
        out = {
            "stage": "P0.5",
            "variant": "default",
            "method": display,
            "method_key": method_key,
            "beta_cost": beta,
            "beta_label": "5.0",
            "cost_type": cost_type,
            "d_warning": d_warning,
            "seed": 0,
            "step": int(row["step"]),
            "scenario": str(row["scenario"]),
            "base_scenario": str(row["scenario"]),
            "episodes": int(row.get("episodes", EPISODES)),
            "source_csv": str(row.get("source_csv", "")),
            "checkpoint_path": str(row.get("checkpoint_path", "")),
        }
        for col in [
            "success_rate",
            "collision_rate",
            "near_miss_rate",
            "mean_min_distance",
            "mean_time",
            "mean_episode_reward",
            "reaction_time_eval_style",
            "reaction_time_nan_style",
            "nan_reaction_rate",
            "distance_warning_cost_nonzero_rate",
            "distance_warning_cost_mean",
            "distance_warning_cost_p95",
            "distance_warning_cost_max",
            "risk_sum_mean",
            "risk_max_mean",
            "scenario_valid_rate",
            "planned_threat_valid_rate",
            "threat_valid_rate",
            "realized_near_miss_rate",
        ]:
            out[col] = row[col] if col in row else np.nan
        out["risk_sum_p95"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def build_beta_delta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    wide_methods = [
        "attention_full_distance_penalty_wide_d2_beta2",
        "attention_full_distance_penalty_wide_d2_beta5_reference",
        "attention_full_distance_penalty_wide_d2_beta10",
    ]
    risk5_method = "attention_full_risk_penalty_beta5_reference"
    use = df[df["step"].isin(CHECKPOINT_STEPS)].copy()
    for (step, scenario), group in use.groupby(["step", "base_scenario"], sort=True):
        risk = group[group["method"] == risk5_method]
        if risk.empty:
            continue
        r = risk.iloc[0]
        wide_cover_any = False
        best_cover_method = ""
        for wide_method in wide_methods:
            wide = group[group["method"] == wide_method]
            if wide.empty:
                continue
            w = wide.iloc[0]
            delta = {
                "step": int(step),
                "scenario": scenario,
                "risk_reference_method": risk5_method,
                "wide_method": wide_method,
                "wide_beta": float(w["beta_cost"]),
                "delta_mean_time_risk5_minus_wide": float(r["mean_time"]) - float(w["mean_time"]),
                "delta_near_miss_risk5_minus_wide": float(r["near_miss_rate"]) - float(w["near_miss_rate"]),
                "delta_min_distance_risk5_minus_wide": float(r["mean_min_distance"]) - float(w["mean_min_distance"]),
                "delta_collision_risk5_minus_wide": float(r["collision_rate"]) - float(w["collision_rate"]),
                "delta_success_risk5_minus_wide": float(r["success_rate"]) - float(w["success_rate"]),
                "delta_reaction_risk5_minus_wide": (
                    float(r["reaction_time_eval_style"]) - float(w["reaction_time_eval_style"])
                    if pd.notna(r["reaction_time_eval_style"]) and pd.notna(w["reaction_time_eval_style"])
                    else np.nan
                ),
            }
            wide_covers = bool(
                delta["delta_mean_time_risk5_minus_wide"] >= -0.10
                and delta["delta_near_miss_risk5_minus_wide"] >= -0.02
                and delta["delta_collision_risk5_minus_wide"] >= -0.01
                and delta["delta_success_risk5_minus_wide"] <= 0.02
                and delta["delta_min_distance_risk5_minus_wide"] <= 0.05
            )
            delta["wide_covers_risk5"] = wide_covers
            if wide_covers and not wide_cover_any:
                wide_cover_any = True
                best_cover_method = wide_method
            rows.append(delta)
        if not wide_cover_any:
            rows.append(
                {
                    "step": int(step),
                    "scenario": scenario,
                    "risk_reference_method": risk5_method,
                    "wide_method": "best_wide_beta",
                    "wide_beta": np.nan,
                    "wide_covers_risk5": False,
                    "best_cover_method": "",
                }
            )
        else:
            rows.append(
                {
                    "step": int(step),
                    "scenario": scenario,
                    "risk_reference_method": risk5_method,
                    "wide_method": "best_wide_beta",
                    "wide_beta": np.nan,
                    "wide_covers_risk5": True,
                    "best_cover_method": best_cover_method,
                }
            )
    delta_df = pd.DataFrame(rows)
    primary = delta_df[(delta_df["wide_method"] == "best_wide_beta") & (delta_df["step"] == 750000)].copy()
    wide_cover_rate_750k = float(primary["wide_covers_risk5"].mean()) if not primary.empty else 0.0
    wide_covers_overall = bool(wide_cover_rate_750k >= 0.50)

    summary_rows: list[dict[str, Any]] = []
    for step, group in delta_df[delta_df["wide_method"] == "best_wide_beta"].groupby("step", sort=True):
        summary_rows.append(
            {
                "step": int(step),
                "scenario_count": int(group["scenario"].nunique()),
                "wide_cover_count": int(group["wide_covers_risk5"].sum()),
                "wide_cover_rate": float(group["wide_covers_risk5"].mean()),
                "risk_not_covered_count": int((~group["wide_covers_risk5"].astype(bool)).sum()),
                "distance_beta_covers_risk": bool(float(group["wide_covers_risk5"].mean()) >= 0.50),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return delta_df, summary, wide_covers_overall


def run_p05() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    specs = p05_specs()
    rows: list[dict[str, Any]] = []
    total_evals = len(specs) * len(CHECKPOINT_STEPS) * len(SCENARIOS_BASE)
    completed = 0
    try:
        for spec in specs:
            train_if_needed(spec)
            for step in CHECKPOINT_STEPS:
                for scenario in SCENARIOS_BASE:
                    path = eval_if_needed(spec, step, scenario, P05_OUT, completed, total_evals - completed, "Stage P0.5")
                    rows.append(summarize_eval_csv(path, spec, step, scenario, "P0.5"))
                    completed += 1
    except subprocess.CalledProcessError as exc:
        write_no_go(
            "p05_no_go_training_unstable_or_nan",
            "Stage P0.5",
            {"returncode": exc.returncode, "command": " ".join(exc.cmd)},
            "A beta-sweep training or evaluation command failed. Do not proceed to PPO-Lagrangian on unstable fixed-beta evidence; inspect logs and stabilize the fixed cost design first.",
        )
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    refs = reference_beta5_rows()
    if not refs.empty:
        df = pd.concat([df, refs], ignore_index=True, sort=False)
    df = df.sort_values(["method", "step", "base_scenario"])

    for metric in CORE_METRICS:
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(dtype=float)).any():
            write_no_go(
                "p05_no_go_training_unstable_or_nan",
                "Stage P0.5",
                {"metric": metric, "nan_count": int(values.isna().sum())},
                "The beta sweep produced invalid core metrics. Stabilize fixed-beta training/evaluation before entering adaptive-lambda PPO.",
            )
            raise SystemExit(0)

    df.to_csv(P05_OUT / "p0_5_beta_sweep_by_method_step_scenario.csv", index=False)
    delta, pareto_summary, wide_covers = build_beta_delta(df)
    delta.to_csv(P05_OUT / "p0_5_beta_delta_table.csv", index=False)
    pareto_summary.to_csv(P05_OUT / "p0_5_beta_pareto_summary.csv", index=False)

    summary_750 = pareto_summary[pareto_summary["step"] == 750000]
    cover_rate = float(summary_750["wide_cover_rate"].iloc[0]) if not summary_750.empty else 0.0
    lines = [
        "# Stage P0.5 Beta / Cost-Scale Sweep Report",
        "",
        "## Scope",
        "- Trained wide_d2 beta=2/10 and risk beta=2/10.",
        "- Included existing P2 seed0 beta=5 reference rows for wide_d2 and risk.",
        "- Evaluated 250k/500k/750k across the eight P2 scenarios.",
        "",
        "## Pareto Coverage",
        f"- 750k wide_d2 beta sweep cover rate over risk beta=5: {cover_rate:.4f}",
        f"- distance_beta_covers_risk: {wide_covers}",
        "",
        "| step | scenarios | wide_cover_count | wide_cover_rate | risk_not_covered_count |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in pareto_summary.iterrows():
        lines.append(
            f"| {int(row['step'])} | {int(row['scenario_count'])} | {int(row['wide_cover_count'])} | "
            f"{fmt(row['wide_cover_rate'])} | {int(row['risk_not_covered_count'])} |"
        )
    lines += [
        "",
        "## Required Answers",
        f"1. wide_d2 beta sweep covers risk beta=5 at 750k: {wide_covers}.",
        f"2. risk beta=5 being pure tuning luck is {'supported' if wide_covers else 'not supported'} by this coverage rule.",
        "3. The adaptive-lambda / PPO-Lagrangian decision is deferred to the integrated final report.",
    ]
    write_text(ROOT / "P0_5_BETA_COST_SCALE_SWEEP_REPORT.md", lines)
    return df, delta, pareto_summary, wide_covers


def pminus1_has_pareto(p1_summary: pd.DataFrame) -> bool:
    return bool((p1_summary["risk_pareto_positive_count"] > 0).any())


def final_decision(p1_summary: pd.DataFrame, p0_summary: pd.DataFrame, p05_wide_covers: bool) -> str:
    p1_live = pminus1_has_pareto(p1_summary)
    p0_adaptive = bool("risk_adaptation_supported" in p0_summary and p0_summary["risk_adaptation_supported"].fillna(False).any())
    p0_risk_faster = bool("both_work_but_risk_faster" in p0_summary and p0_summary["both_work_but_risk_faster"].fillna(False).any())
    if p0_adaptive and not p05_wide_covers:
        return "risk_adaptation_supported"
    if p1_live and (p0_risk_faster or not p0_adaptive) and not p05_wide_covers:
        return "risk_pareto_but_not_adaptive"
    if p05_wide_covers:
        return "distance_margin_explains_risk"
    return "risk_mainline_downgrade"


def write_final_report(
    terminal_decision: str,
    p1_summary: pd.DataFrame,
    p0_summary: pd.DataFrame,
    p05_summary: pd.DataFrame,
    p05_wide_covers: bool,
) -> None:
    pareto_scenarios = sorted(p1_summary.loc[p1_summary["risk_pareto_positive_count"] > 0, "scenario"].astype(str))
    dominated_scenarios = sorted(p1_summary.loc[p1_summary["wide_d2_dominates_count"] > p1_summary["rows"] / 2.0, "scenario"].astype(str))
    p0_rows = p0_summary.drop_duplicates("variant") if "variant" in p0_summary else pd.DataFrame()
    high = p0_rows[p0_rows["variant"] == "high_speed"] if not p0_rows.empty else pd.DataFrame()
    small = p0_rows[p0_rows["variant"] == "small_space"] if not p0_rows.empty else pd.DataFrame()
    cover_750 = p05_summary[p05_summary["step"] == 750000]
    cover_rate = float(cover_750["wide_cover_rate"].iloc[0]) if not cover_750.empty else float("nan")

    def variant_answer(row: pd.DataFrame, field: str, default: Any = "unknown") -> Any:
        if row.empty or field not in row:
            return default
        return row.iloc[0][field]

    lines = [
        "# Pre-PPO Priority Experiments Final Report",
        "",
        "## 1. Motivation",
        "This run resolves the three pre-PPO-Lagrangian questions: P2 Pareto scope, fixed-margin adaptation under two variants, and beta/cost-scale confounding.",
        "",
        "## 2. P-1 Pareto Audit",
        f"- Pareto-positive scenarios: {', '.join(pareto_scenarios) if pareto_scenarios else 'none'}.",
        f"- Majority wide_d2-dominated scenarios: {', '.join(dominated_scenarios) if dominated_scenarios else 'none'}.",
        "",
        "## 3. P0 Adaptation Validation",
        f"- high_speed wide failure: {bool(variant_answer(high, 'wide_failure', False))}; overconservative: {bool(variant_answer(high, 'wide_overconservative', False))}; risk adaptation supported: {bool(variant_answer(high, 'risk_adaptation_supported', False))}.",
        f"- small_space wide failure: {bool(variant_answer(small, 'wide_failure', False))}; overconservative: {bool(variant_answer(small, 'wide_overconservative', False))}; risk adaptation supported: {bool(variant_answer(small, 'risk_adaptation_supported', False))}.",
        "",
        "## 4. P0.5 Beta / Cost-Scale Sweep",
        f"- 750k wide_d2 beta-sweep cover rate over risk beta=5: {fmt(cover_rate)}.",
        f"- wide_d2 beta sweep covers risk beta=5 under the configured rule: {p05_wide_covers}.",
        "",
        "## 5. Integrated Decision",
        f"terminal_decision = {terminal_decision}",
        "",
        "## 6. Next Recommendation",
    ]
    if terminal_decision == "risk_adaptation_supported":
        lines.append("- Keep risk as the mainline and use these results as justification to enter adaptive-lambda PPO / PPO-Lagrangian next.")
    elif terminal_decision == "risk_pareto_but_not_adaptive":
        lines.append("- Keep risk as a Pareto-efficient safety cost, but do not claim strong adaptation; adaptive-lambda PPO is worth a scoped follow-up.")
    elif terminal_decision == "distance_margin_explains_risk":
        lines.append("- Downgrade risk as a main innovation and pivot toward safety-margin cost design principles and beta/margin tuning.")
    else:
        lines.append("- Downgrade the risk mainline; fixed safety-margin design is the more defensible next direction.")
    lines += [
        "",
        "## Required Answers",
        f"1. P2 Pareto audit: risk advantage holds in {', '.join(pareto_scenarios) if pareto_scenarios else 'no'} scenarios under the classification rule.",
        f"2. risk is {'not globally dominated' if pareto_scenarios else 'dominated or not meaningfully better'} by wide_d2; majority-dominated scenarios: {', '.join(dominated_scenarios) if dominated_scenarios else 'none'}.",
        f"3. high_speed wide_d2 failure/overconservative: failure={bool(variant_answer(high, 'wide_failure', False))}, overconservative={bool(variant_answer(high, 'wide_overconservative', False))}.",
        f"4. high_speed risk Pareto scenarios at 750k: {int(variant_answer(high, 'risk_pareto_positive_scenarios_750k', 0))}.",
        f"5. small_space wide_d2 failure/overconservative: failure={bool(variant_answer(small, 'wide_failure', False))}, overconservative={bool(variant_answer(small, 'wide_overconservative', False))}.",
        f"6. small_space risk Pareto scenarios at 750k: {int(variant_answer(small, 'risk_pareto_positive_scenarios_750k', 0))}.",
        f"7. beta sweep: wide_d2 covers risk beta=5 = {p05_wide_covers} with 750k cover_rate={fmt(cover_rate)}.",
        f"8. risk beta=5 tuning-only explanation: {'supported' if p05_wide_covers else 'not fully supported'}.",
        f"9. adaptive-lambda / PPO-Lagrangian now: {'yes' if terminal_decision in {'risk_adaptation_supported', 'risk_pareto_but_not_adaptive'} else 'no'}.",
        f"10. risk mainline: {terminal_decision}.",
    ]
    write_text(ROOT / "PRE_PPO_PRIORITY_FINAL_REPORT.md", lines)
    COMPLETE_FLAG.write_text(f"terminal_decision={terminal_decision}\ncompleted_at={now()}\n", encoding="utf-8")
    update_status("complete", terminal_decision, "750000", 0, 0, "complete", {"terminal_decision": terminal_decision})


def main() -> None:
    ensure_dirs()
    update_status("starting", "", "", 0, 0, "starting")
    if COMPLETE_FLAG.exists() or NO_GO_FLAG.exists():
        COMPLETE_FLAG.unlink(missing_ok=True)
        NO_GO_FLAG.unlink(missing_ok=True)

    p1_delta, p1_summary, p1_no_go = run_pminus1()
    if p1_no_go:
        write_no_go(
            "pminus1_no_go_risk_not_pareto",
            "Stage P-1",
            {
                "pareto_positive_rows": int((p1_delta["classification"] == "risk_pareto_positive").sum()),
                "wide_d2_dominates_rows": int((p1_delta["classification"] == "wide_d2_dominates").sum()),
            },
            "Downgrade risk to a safety-cost design reference and pivot to safety-margin cost design principles.",
        )
        return

    _, _, p0_summary, _ = run_p0()
    _, _, p05_summary, p05_wide_covers = run_p05()
    terminal_decision = final_decision(p1_summary, p0_summary, p05_wide_covers)
    write_final_report(terminal_decision, p1_summary, p0_summary, p05_summary, p05_wide_covers)
    print(f"[{now()}] COMPLETE terminal_decision={terminal_decision}", flush=True)


if __name__ == "__main__":
    main()
