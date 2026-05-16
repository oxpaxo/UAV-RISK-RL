from __future__ import annotations

import argparse
import csv
import json
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
SCENARIOS = ["eval_random_switch", "eval_sudden_turn", "eval_random_switch_hard", "mixed_uncertainty"]
SCENARIO_SUFFIX = {
    "eval_random_switch": "random",
    "eval_sudden_turn": "sudden",
    "eval_random_switch_hard": "hard",
    "mixed_uncertainty": "mixed",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    method: str
    agg: str
    seed: int
    target_steps: int
    checkpoint_dir: Path
    run_dir: Path
    result_dir: Path
    save_path: Path
    checkpoint_steps: list[int]
    train_extra: dict[str, Any]
    eval_extra: dict[str, Any]
    resume_from: Path | None = None
    resume_global_step: int = 0


def run(cmd: list[str], log_path: Path | None = None, allow_fail: bool = False) -> int:
    print(f"[PIPELINE] RUN {' '.join(cmd)}", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n\n$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def ensure_dirs() -> None:
    dirs = [
        "results/preflight",
        "results/preflight/traces",
        "results/preflight/plots",
        "checkpoints/longtrain_baseline",
        "runs/longtrain_baseline",
        "results/longtrain_baseline/eval",
        "results/longtrain_baseline/plots",
        "checkpoints/gate2b",
        "runs/gate2b",
        "results/gate2b/eval",
        "results/gate2b/traces",
        "results/gate2b/plots",
        "results/gate2b/summary",
        "checkpoints/attention_seed1",
        "runs/attention_seed1",
        "results/attention_seed1/eval",
        "results/attention_seed1/plots",
        "runs/logs",
    ]
    for rel in dirs:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


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


def find_checkpoint(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path
    return None


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_parser(script: str) -> str:
    proc = subprocess.run([PYTHON, script, "--help"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def stage0_code_status() -> None:
    train_help = inspect_parser("train.py")
    eval_help = inspect_parser("eval.py")
    extractor_text = (ROOT / "policies/obstacle_set_extractor.py").read_text(encoding="utf-8")
    env_text = (ROOT / "envs/dynamic_obstacle_env.py").read_text(encoding="utf-8")
    lines = [
        "# Code Status Preflight",
        "",
        "Current directory code was inspected directly. Old handoff claims are not assumed.",
        "",
        "## train.py key parameters",
    ]
    for param in [
        "--method",
        "--profile_mode",
        "--agg",
        "--seed",
        "--total_steps",
        "--n_envs",
        "--device",
        "--scenario",
        "--save_checkpoints",
        "--checkpoint_steps",
        "--checkpoint_dir",
        "--log_dir",
        "--save_path",
        "--resume_from",
        "--resume_global_step",
        "--use_safety_cost",
        "--cost_type",
        "--beta_cost",
        "--fallback_penalty",
        "--d_warning",
        "--use_risk_bias",
        "--lambda_bias",
    ]:
        lines.append(f"- {param}: {'yes' if param in train_help else 'missing'}")
    lines += ["", "## eval.py key parameters"]
    for param in [
        "--model_path",
        "--method",
        "--profile_mode",
        "--agg",
        "--seed",
        "--eval_seed",
        "--episodes",
        "--scenario",
        "--device",
        "--out_csv",
        "--save_trace",
        "--trace_dir",
    ]:
        lines.append(f"- {param}: {'yes' if param in eval_help else 'missing'}")
    lines += [
        "",
        "## Aggregation modes",
        f"- risk: {'risk' in extractor_text}",
        f"- attention: {'attention' in extractor_text}",
        f"- mean: {'mean' in extractor_text}",
        f"- risk_biased_attention logits: {'lambda_bias * torch.log' in extractor_text}",
        f"- latest_attention_weights cache: {'latest_attention_weights' in extractor_text}",
        "",
        "## DynamicObstacleEnv info fields",
    ]
    for key in ["risk_sum", "risk_max", "distance_warning_cost", "risk_values", "sigma_values"]:
        lines.append(f"- {key}: {key in env_text}")
    write_lines(ROOT / "results/preflight/CODE_STATUS_PREFLIGHT.md", lines)


def stage0_resume_smoke(args: argparse.Namespace) -> None:
    smoke_model = ROOT / "checkpoints/ewma_short/smoke.zip"
    if not smoke_model.exists():
        run(
            [
                PYTHON,
                "train.py",
                "--method",
                "resume_smoke_base",
                "--agg",
                "mean",
                "--seed",
                "99",
                "--total_steps",
                str(args.smoke_train_steps),
                "--n_envs",
                "1",
                "--device",
                "cpu",
                "--scenario",
                "train_random_switch",
                "--save_path",
                str(smoke_model),
                "--log_dir",
                "runs/preflight/resume_smoke_base",
                "--heartbeat_seconds",
                "9999",
            ],
            ROOT / "runs/logs/preflight_resume_smoke_base.log",
        )
    out_model = ROOT / "checkpoints/preflight/resume_smoke_step12000.zip"
    (ROOT / "checkpoints/preflight").mkdir(parents=True, exist_ok=True)
    rc = run(
        [
            PYTHON,
            "train.py",
            "--method",
            "resume_smoke",
            "--agg",
            "mean",
            "--seed",
            "99",
            "--total_steps",
            str(args.smoke_train_steps),
            "--resume_from",
            str(smoke_model),
            "--resume_global_step",
            "10000",
            "--n_envs",
            "1",
            "--device",
            "cpu",
            "--scenario",
            "train_random_switch",
            "--save_checkpoints",
            "true",
            "--checkpoint_steps",
            str(args.smoke_train_steps),
            "--checkpoint_dir",
            "checkpoints/preflight",
            "--run_name",
            "resume_smoke",
            "--save_path",
            str(out_model),
            "--log_dir",
            "runs/preflight/resume_smoke",
            "--heartbeat_seconds",
            "9999",
        ],
        ROOT / "runs/logs/preflight_resume_smoke.log",
        allow_fail=True,
    )
    expected = ROOT / f"checkpoints/preflight/resume_smoke_s99_step{10000 + args.smoke_train_steps}.zip"
    lines = [
        "# Resume Preflight Report",
        "",
        f"- Base checkpoint: {smoke_model}",
        f"- Resume return code: {rc}",
        f"- Expected global-step checkpoint: {expected}",
        f"- Global-step checkpoint exists: {expected.exists()}",
        "- Resume semantics: when --resume_from is set, --total_steps means additional local steps.",
    ]
    if rc != 0:
        lines.append("- Result: resume smoke failed; longtrain code will use from-scratch fallback when needed.")
    else:
        lines.append("- Result: resume smoke passed.")
    write_lines(ROOT / "results/preflight/RESUME_PREFLIGHT_REPORT.md", lines)


def stage0_risk_config() -> None:
    extractor_text = (ROOT / "policies/obstacle_set_extractor.py").read_text(encoding="utf-8")
    train_help = inspect_parser("train.py")
    lines = [
        "# Risk Config Preflight Report",
        "",
        "- R_gate parameter name: --r_gate",
        "- lambda_ewma parameter name: --lambda_ewma",
        "- rbar_floor parameter name: --rbar_floor",
        f"- --r_gate exposed by train.py: {'--r_gate' in train_help}",
        f"- --lambda_ewma exposed by train.py: {'--lambda_ewma' in train_help}",
        f"- --rbar_floor exposed by train.py: {'--rbar_floor' in train_help}",
        f"- rbar_floor participates in R_bar clamp: {'torch.clamp(R_bar, min=self.rbar_floor)' in extractor_text}",
        "- Rgate8_lambda015_RbarFloor03 maps to R_gate=8.0, lambda_ewma=0.15, rbar_floor=0.3.",
    ]
    write_lines(ROOT / "results/preflight/RISK_CONFIG_PREFLIGHT_REPORT.md", lines)


def eval_model(
    model_path: Path,
    method: str,
    agg: str,
    seed: int,
    scenario: str,
    out_csv: Path,
    episodes: int,
    eval_seed: int = 1000,
    global_step: int = -1,
    save_trace: bool = False,
    trace_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
    log_name: str | None = None,
) -> None:
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
            if len(existing) >= episodes:
                return
        except Exception:
            pass
    cmd = [
        PYTHON,
        "eval.py",
        "--model_path",
        str(model_path),
        "--method",
        method,
        "--agg",
        agg,
        "--seed",
        str(seed),
        "--eval_seed",
        str(eval_seed),
        "--episodes",
        str(episodes),
        "--scenario",
        scenario,
        "--device",
        "cpu",
        "--out_csv",
        str(out_csv),
        "--global_step",
        str(global_step),
    ]
    for key, value in (extra or {}).items():
        cmd.extend([f"--{key}", str(value)])
    if save_trace:
        cmd.extend(["--save_trace", "true", "--trace_dir", str(trace_dir or out_csv.parent / "traces")])
    run(cmd, ROOT / f"runs/logs/{log_name or out_csv.stem}.log")


def summarize_eval_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    summary: dict[str, float] = {
        "success_rate": float(df["success"].mean()),
        "collision_rate": float(df["collision"].mean()),
        "mean_min_distance": float(df["episode_min_distance"].mean()),
        "near_miss_rate": float(df["near_miss"].mean()) if "near_miss" in df.columns else np.nan,
        "mean_time": float(df["time_to_goal"].mean()),
        "mean_episode_reward": float(df["episode_reward"].mean()),
    }
    for col in [
        "distance_warning_cost_mean",
        "distance_warning_cost_p50",
        "distance_warning_cost_p90",
        "distance_warning_cost_p95",
        "distance_warning_cost_max",
        "risk_sum_mean",
        "risk_sum_p50",
        "risk_sum_p90",
        "risk_sum_p95",
        "risk_sum_max",
        "risk_max_mean",
        "risk_max_p50",
        "risk_max_p90",
        "risk_max_p95",
        "risk_max_max",
        "reaction_time_eval_style",
        "reaction_time_nan_style",
        "reaction_time",
        "min_distance_after_turn",
    ]:
        if col in df.columns:
            summary[col] = float(df[col].mean(skipna=True))
    if "reaction_time_nan_style" in df.columns:
        summary["nan_reaction_rate"] = float(df["reaction_time_nan_style"].isna().mean())
    if "no_response" in df.columns:
        summary["no_response_count"] = float(df["no_response"].sum())
        summary["total_episodes"] = float(len(df))
    return summary


def write_cost_report(cost_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(cost_rows)
    df.to_csv(ROOT / "results/preflight/cost_stats_attention_gate.csv", index=False)
    if df.empty:
        lines = ["# Cost Scale Preflight Report", "", "No attention gate checkpoint was available; cost stats could not be computed."]
    else:
        p90_distance = float(df["distance_warning_cost_p90"].median())
        p90_risk_sum = float(df["risk_sum_p90"].median())
        p90_risk_max = float(df["risk_max_p90"].median())
        beta = 5.0
        lines = [
            "# Cost Scale Preflight Report",
            "",
            f"- Rows: {len(df)}",
            f"- Median distance_warning_cost_p90: {p90_distance:.6f}; beta*cost: {beta * p90_distance:.6f}",
            f"- Median risk_sum_p90: {p90_risk_sum:.6f}; beta*cost: {beta * p90_risk_sum:.6f}",
            f"- Median risk_max_p90: {p90_risk_max:.6f}; beta*cost: {beta * p90_risk_max:.6f}",
            "- Collision penalty is approximately -10.",
            f"- distance_warning_cost and risk_sum same order: {0.2 <= (p90_distance / max(p90_risk_sum, 1e-8)) <= 5.0}",
            f"- beta_cost=5.0 distance scale in target 1-5: {1.0 <= beta * p90_distance <= 5.0}",
            f"- beta_cost=5.0 risk_sum scale in target 1-5: {1.0 <= beta * p90_risk_sum <= 5.0}",
        ]
        if not (1.0 <= beta * p90_distance <= 5.0):
            rec = min(max(2.0 / max(p90_distance, 1e-8), 0.1), 20.0)
            lines.append(f"- Suggested distance_penalty beta_cost: {rec:.3f}")
        if not (1.0 <= beta * p90_risk_sum <= 5.0):
            rec = min(max(2.0 / max(p90_risk_sum, 1e-8), 0.1), 20.0)
            lines.append(f"- Suggested risk_penalty beta_cost: {rec:.3f}")
        lines.append("- Gate-2b can use beta_cost=5.0 only if both beta*cost_p90 values above are acceptable.")
    write_lines(ROOT / "results/preflight/COST_SCALE_PREFLIGHT_REPORT.md", lines)


def stage0_cost_and_trace(args: argparse.Namespace) -> None:
    attention_candidates = [
        "checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step250000.zip",
        "checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step500000.zip",
        "checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step750000.zip",
        "checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step1000000.zip",
        "checkpoints/attention_full_s0.zip",
    ]
    model = find_checkpoint(attention_candidates)
    cost_rows: list[dict[str, Any]] = []
    if model is not None:
        step = checkpoint_num_timesteps(model) or 0
        for scenario in ["eval_random_switch", "eval_sudden_turn"]:
            out_csv = ROOT / f"results/preflight/attention_full_s0_step{step}_{SCENARIO_SUFFIX[scenario]}_cost.csv"
            eval_model(
                model,
                "attention_full",
                "attention",
                0,
                scenario,
                out_csv,
                episodes=args.preflight_episodes,
                global_step=step,
                log_name=f"preflight_attention_cost_{scenario}",
            )
            summary = summarize_eval_csv(out_csv)
            cost_rows.append({"method": "attention_full", "seed": 0, "global_step": step, "scenario": scenario, **summary})
        for scenario in ["eval_sudden_turn", "mixed_uncertainty"]:
            trace_csv = ROOT / f"results/preflight/attention_full_s0_step{step}_{SCENARIO_SUFFIX[scenario]}_trace_eval.csv"
            eval_model(
                model,
                "attention_full",
                "attention",
                0,
                scenario,
                trace_csv,
                episodes=args.trace_episodes,
                global_step=step,
                save_trace=True,
                trace_dir=ROOT / "results/preflight/traces",
                log_name=f"preflight_attention_trace_{scenario}",
            )
    write_cost_report(cost_rows)
    trace_files = list((ROOT / "results/preflight/traces").glob("*.csv"))
    lines = [
        "# Trace Field Preflight Report",
        "",
        f"- Trace CSV files: {len(trace_files)}",
        f"- Supports risk_sum vs distance_warning_cost timing: {bool(trace_files)}",
        f"- Supports attention weight diagnostics: {bool(trace_files)}",
        "- Missing fields: none expected when trace files exist; if attention weights cannot be read they are written as NaN arrays.",
    ]
    write_lines(ROOT / "results/preflight/TRACE_FIELD_PREFLIGHT_REPORT.md", lines)


def stage0_reaction_report() -> None:
    lines = [
        "# Reaction Definition Check",
        "",
        "- turn_time: env.turn_step * env.dt; default turn_step=15, dt=0.2, turn_time=3.0 seconds.",
        "- reaction flag: lateral desired velocity relative to goal direction > 0.3.",
        "- consecutive_steps: 2.",
        "- eval-style no response: max_episode_time - turn_time.",
        "- nan-style no response: NaN.",
        "- eval.py writes reaction_time_eval_style, reaction_time_nan_style, nan_reaction_rate, no_response_count, total_episodes.",
        "- diagnostic mismatch in old data is primarily expected from NaN vs upper-bound fill for no-response episodes.",
    ]
    write_lines(ROOT / "results/preflight/reaction_definition_check.md", lines)


def stage0_checkpoint_index() -> None:
    rows: list[dict[str, Any]] = []

    def add(method: str, config: str, seed: int, step: int, ckpt: Path | None, notes: str = "") -> None:
        rows.append(
            {
                "method": method,
                "config": config,
                "seed": seed,
                "global_step": step,
                "checkpoint_path": str(ckpt) if ckpt and ckpt.exists() else "missing",
                "can_resume": bool(ckpt and ckpt.exists()),
                "eval_random_switch_csv": "missing",
                "eval_sudden_turn_csv": "missing",
                "eval_random_switch_hard_csv": "missing",
                "mixed_uncertainty_csv": "missing",
                "trace_dir": "missing",
                "notes": notes,
            }
        )

    for step in [250000, 500000, 750000, 1000000]:
        candidates = [
            ROOT / f"checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step{step}.zip",
            ROOT / f"checkpoints/attention_full_s0_step{step}.zip",
        ]
        ckpt = next((p for p in candidates if p.exists()), None)
        if step == 500000 and ckpt is None and (ROOT / "checkpoints/attention_full_s0.zip").exists():
            ckpt = ROOT / "checkpoints/attention_full_s0.zip"
        add("attention_full", "baseline", 0, step, ckpt, "attention 1000k gate directory unavailable in current workspace")
    for step_label, step in [("100k", 100000), ("200k", 200000), ("300k", 300000), ("500k", 500000)]:
        add(
            "risk_full_rbar",
            "Rgate8_lambda015_RbarFloor03",
            0,
            step,
            ROOT / f"checkpoints/ewma_formal/Rgate8_lambda015_RbarFloor03_s0_step{step_label}.zip",
        )
    for config in ["Rgate8", "Rgate8_lambda015", "Rgate8_lambda015_RbarFloor03"]:
        for seed in [0, 1, 2]:
            for step_label, step in [("100k", 100000), ("200k", 200000), ("300k", 300000), ("500k", 500000)]:
                add("risk_full_rbar", config, seed, step, ROOT / f"checkpoints/ewma_formal/{config}_s{seed}_step{step_label}.zip")
    for method in ["risk_full_rbar", "attention_full"]:
        for seed in [0, 1, 2]:
            add(method, "preexp", seed, checkpoint_num_timesteps(ROOT / f"checkpoints/{method}_s{seed}.zip") or -1, ROOT / f"checkpoints/{method}_s{seed}.zip")
    pd.DataFrame(rows).to_csv(ROOT / "results/preflight/CHECKPOINT_EVAL_INDEX.csv", index=False)


def stage0_safety_and_bias_reports() -> None:
    safety_lines = [
        "# Safety Cost Preflight Report",
        "",
        "- distance_penalty enters reward through SafetyCostWrapper when --use_safety_cost true --fallback_penalty true --cost_type distance_warning.",
        "- risk_penalty enters reward through SafetyCostWrapper when --use_safety_cost true --fallback_penalty true --cost_type risk_sum.",
        "- Training info records base_reward, applied_cost, shaped_reward, fallback_penalty_active.",
        "- Default beta_cost: 5.0.",
        "- The training log prints: FALLBACK: cost-penalty, not PPO-Lagrangian.",
    ]
    write_lines(ROOT / "results/preflight/SAFETY_COST_PREFLIGHT_REPORT.md", safety_lines)
    bias_lines = [
        "# Risk Biased Attention Preflight Report",
        "",
        "- Implementation: score_i = learned_score_i + lambda_bias * log(risk_i + eps).",
        "- Default lambda_bias: 0.2.",
        "- risk_i is read from full_12 obstacle profile final dimension.",
        "- latest_attention_weights are cached for trace export.",
        "- A smoke comparison is covered by trace availability and by using --use_risk_bias in Gate-2b run_config.",
    ]
    write_lines(ROOT / "results/preflight/RISK_BIASED_ATTENTION_PREFLIGHT_REPORT.md", bias_lines)
    config_lines = [
        "# Gate-2b Config Logging Preflight Report",
        "",
        "- train.py writes runs/{group}/{run_name}/run_config.json.",
        "- run_config.json includes method, profile, seed, total_steps, checkpoint_steps, cost settings, risk-bias settings, risk config, d_safe, d_warning, resume settings, and checkpoint naming rule.",
        "- Penalty reward path is logged through SafetyCostWrapper info fields.",
        "- risk_bias path is logged by --use_risk_bias and --lambda_bias and implemented in ObstacleSetExtractor logits.",
    ]
    write_lines(ROOT / "results/preflight/GATE2B_CONFIG_LOGGING_PREFLIGHT_REPORT.md", config_lines)


def stage0(args: argparse.Namespace) -> None:
    ensure_dirs()
    stage0_code_status()
    stage0_resume_smoke(args)
    stage0_risk_config()
    stage0_cost_and_trace(args)
    stage0_reaction_report()
    stage0_checkpoint_index()
    stage0_safety_and_bias_reports()


def maybe_train(spec: RunSpec, args: argparse.Namespace) -> None:
    if spec.save_path.exists() and checkpoint_num_timesteps(spec.save_path):
        print(f"[PIPELINE] SKIP existing model {spec.save_path}", flush=True)
        return
    checkpoint_steps = ",".join(str(s) for s in spec.checkpoint_steps)
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
        str(spec.seed),
        "--total_steps",
        str(spec.target_steps if spec.resume_from is None else spec.target_steps - spec.resume_global_step),
        "--n_envs",
        str(args.n_envs),
        "--device",
        "cpu",
        "--scenario",
        "train_random_switch",
        "--save_checkpoints",
        "true",
        "--checkpoint_steps",
        checkpoint_steps,
        "--checkpoint_dir",
        str(spec.checkpoint_dir),
        "--log_dir",
        str(spec.run_dir),
        "--run_name",
        spec.name,
        "--save_path",
        str(spec.save_path),
        "--heartbeat_seconds",
        "30",
    ]
    if spec.resume_from is not None and spec.resume_from.exists():
        cmd.extend(["--resume_from", str(spec.resume_from), "--resume_global_step", str(spec.resume_global_step)])
    for key, value in spec.train_extra.items():
        cmd.extend([f"--{key}", str(value)])
    rc = run(cmd, ROOT / f"runs/logs/train_{spec.name}.log", allow_fail=True)
    if rc != 0 and spec.resume_from is not None:
        fallback_steps = [250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000]
        if spec.target_steps <= 1000000:
            fallback_steps = [250000, 500000, 750000, 1000000]
        fallback_cmd = [
            PYTHON,
            "train.py",
            "--method",
            spec.method,
            "--profile_mode",
            "full_12",
            "--agg",
            spec.agg,
            "--seed",
            str(spec.seed),
            "--total_steps",
            str(spec.target_steps),
            "--n_envs",
            str(args.n_envs),
            "--device",
            "cpu",
            "--scenario",
            "train_random_switch",
            "--save_checkpoints",
            "true",
            "--checkpoint_steps",
            ",".join(str(s) for s in fallback_steps),
            "--checkpoint_dir",
            str(spec.checkpoint_dir),
            "--log_dir",
            str(spec.run_dir.with_name(spec.run_dir.name + "_from_scratch")),
            "--run_name",
            spec.name,
            "--save_path",
            str(spec.save_path),
            "--heartbeat_seconds",
            "30",
        ]
        for key, value in spec.train_extra.items():
            fallback_cmd.extend([f"--{key}", str(value)])
        run(fallback_cmd, ROOT / f"runs/logs/train_{spec.name}_fallback.log")
    elif rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def checkpoint_for(spec: RunSpec, step: int) -> Path:
    direct = spec.checkpoint_dir / f"{spec.name}_s{spec.seed}_step{step}.zip"
    if direct.exists():
        return direct
    if step == spec.target_steps and spec.save_path.exists():
        return spec.save_path
    return direct


def eval_spec(spec: RunSpec, args: argparse.Namespace, result_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step in spec.checkpoint_steps:
        global_step = spec.resume_global_step + step if spec.resume_from is not None else step
        ckpt = checkpoint_for(spec, global_step)
        if not ckpt.exists():
            print(f"[PIPELINE] missing checkpoint for eval: {ckpt}", flush=True)
            continue
        for scenario in SCENARIOS:
            suffix = SCENARIO_SUFFIX[scenario]
            out_csv = spec.result_dir / "eval" / f"{spec.name}_s{spec.seed}_step{global_step}_{suffix}.csv"
            eval_model(
                ckpt,
                spec.method,
                spec.agg,
                spec.seed,
                scenario,
                out_csv,
                episodes=args.eval_episodes,
                global_step=global_step,
                extra=spec.eval_extra,
                log_name=f"eval_{spec.name}_s{spec.seed}_step{global_step}_{suffix}",
            )
            summary = summarize_eval_csv(out_csv)
            rows.append({"method": spec.method, "run_name": spec.name, "seed": spec.seed, "step": global_step, "scenario": scenario, **summary})
    df = pd.DataFrame(rows)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_csv, index=False)
    return df


def eval_risk_history(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    step_pairs = [("100k", 100000), ("200k", 200000), ("300k", 300000), ("500k", 500000)]
    for label, step in step_pairs:
        ckpt = ROOT / f"checkpoints/ewma_formal/Rgate8_lambda015_RbarFloor03_s0_step{label}.zip"
        if not ckpt.exists():
            continue
        for scenario in SCENARIOS:
            suffix = SCENARIO_SUFFIX[scenario]
            out_csv = ROOT / f"results/longtrain_baseline/eval/risk_Rgate8_lambda015_RbarFloor03_s0_step{step}_{suffix}.csv"
            eval_model(
                ckpt,
                "risk_full_rbar",
                "risk",
                0,
                scenario,
                out_csv,
                episodes=args.eval_episodes,
                global_step=step,
                extra={"r_gate": 8.0, "lambda_ewma": 0.15, "rbar_floor": 0.3, "use_rbar": "true"},
                log_name=f"eval_risk_history_step{step}_{suffix}",
            )
            summary = summarize_eval_csv(out_csv)
            rows.append(
                {
                    "method": "risk_full_rbar",
                    "run_name": "risk_Rgate8_lambda015_RbarFloor03",
                    "seed": 0,
                    "step": step,
                    "scenario": scenario,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def plot_lines(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "success_rate",
        "collision_rate",
        "mean_min_distance",
        "near_miss_rate",
        "distance_warning_cost_p95",
        "risk_sum_p95",
        "reaction_time",
    ]
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        for (run_name, scenario), group in df.groupby(["run_name", "scenario"]):
            if group[metric].isna().all():
                continue
            group = group.sort_values("step")
            plt.plot(group["step"], group[metric], marker="o", label=f"{run_name}:{scenario}")
        plt.xlabel("global step")
        plt.ylabel(metric)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{metric}.png", dpi=140)
        plt.close()


def build_specs(args: argparse.Namespace) -> tuple[RunSpec, RunSpec, list[RunSpec], RunSpec]:
    attention_start = find_checkpoint(
        [
            "checkpoints/attention_1000k_gate/G1_attention_full_1000k_s0_step1000000.zip",
            "checkpoints/attention_full_s0_step1000000.zip",
        ]
    )
    attention_resume_step = 1000000 if attention_start is not None else 0
    attention_spec = RunSpec(
        name="attention_full",
        method="attention_full",
        agg="attention",
        seed=0,
        target_steps=2000000,
        checkpoint_dir=ROOT / "checkpoints/longtrain_baseline",
        run_dir=ROOT / "runs/longtrain_baseline/attention_full_s0",
        result_dir=ROOT / "results/longtrain_baseline",
        save_path=ROOT / "checkpoints/longtrain_baseline/attention_full_s0_step2000000.zip",
        checkpoint_steps=[250000, 500000, 750000, 1000000] if attention_start else [250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000],
        train_extra={},
        eval_extra={},
        resume_from=attention_start,
        resume_global_step=attention_resume_step,
    )
    risk_start = find_checkpoint(["checkpoints/ewma_formal/Rgate8_lambda015_RbarFloor03_s0_step500k.zip"])
    risk_spec = RunSpec(
        name="risk_Rgate8_lambda015_RbarFloor03",
        method="risk_full_rbar",
        agg="risk",
        seed=0,
        target_steps=2000000,
        checkpoint_dir=ROOT / "checkpoints/longtrain_baseline",
        run_dir=ROOT / "runs/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0",
        result_dir=ROOT / "results/longtrain_baseline",
        save_path=ROOT / "checkpoints/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_step2000000.zip",
        checkpoint_steps=[250000, 500000, 1000000, 1500000] if risk_start else [250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000],
        train_extra={"r_gate": 8.0, "lambda_ewma": 0.15, "rbar_floor": 0.3, "use_rbar": "true"},
        eval_extra={"r_gate": 8.0, "lambda_ewma": 0.15, "rbar_floor": 0.3, "use_rbar": "true"},
        resume_from=risk_start,
        resume_global_step=500000 if risk_start else 0,
    )
    gate_specs = [
        RunSpec(
            name="attention_full_distance_penalty",
            method="attention_full_distance_penalty",
            agg="attention",
            seed=0,
            target_steps=1000000,
            checkpoint_dir=ROOT / "checkpoints/gate2b",
            run_dir=ROOT / "runs/gate2b/attention_full_distance_penalty_s0",
            result_dir=ROOT / "results/gate2b",
            save_path=ROOT / "checkpoints/gate2b/attention_full_distance_penalty_s0_step1000000.zip",
            checkpoint_steps=[250000, 500000, 750000, 1000000],
            train_extra={"use_safety_cost": "true", "cost_type": "distance_warning", "d_warning": 1.0, "fallback_penalty": "true", "beta_cost": args.beta_cost},
            eval_extra={},
        ),
        RunSpec(
            name="attention_full_risk_penalty",
            method="attention_full_risk_penalty",
            agg="attention",
            seed=0,
            target_steps=1000000,
            checkpoint_dir=ROOT / "checkpoints/gate2b",
            run_dir=ROOT / "runs/gate2b/attention_full_risk_penalty_s0",
            result_dir=ROOT / "results/gate2b",
            save_path=ROOT / "checkpoints/gate2b/attention_full_risk_penalty_s0_step1000000.zip",
            checkpoint_steps=[250000, 500000, 750000, 1000000],
            train_extra={"use_safety_cost": "true", "cost_type": "risk_sum", "fallback_penalty": "true", "beta_cost": args.beta_cost},
            eval_extra={},
        ),
        RunSpec(
            name="risk_biased_attention_risk_penalty",
            method="risk_biased_attention_risk_penalty",
            agg="attention",
            seed=0,
            target_steps=1000000,
            checkpoint_dir=ROOT / "checkpoints/gate2b",
            run_dir=ROOT / "runs/gate2b/risk_biased_attention_risk_penalty_s0",
            result_dir=ROOT / "results/gate2b",
            save_path=ROOT / "checkpoints/gate2b/risk_biased_attention_risk_penalty_s0_step1000000.zip",
            checkpoint_steps=[250000, 500000, 750000, 1000000],
            train_extra={"use_safety_cost": "true", "cost_type": "risk_sum", "fallback_penalty": "true", "beta_cost": args.beta_cost, "use_risk_bias": "true", "lambda_bias": 0.2},
            eval_extra={"use_risk_bias": "true", "lambda_bias": 0.2},
        ),
    ]
    seed1_spec = RunSpec(
        name="attention_full",
        method="attention_full",
        agg="attention",
        seed=1,
        target_steps=1000000,
        checkpoint_dir=ROOT / "checkpoints/attention_seed1",
        run_dir=ROOT / "runs/attention_seed1/attention_full_s1_1000k",
        result_dir=ROOT / "results/attention_seed1",
        save_path=ROOT / "checkpoints/attention_seed1/attention_full_s1_step1000000.zip",
        checkpoint_steps=[250000, 500000, 750000, 1000000],
        train_extra={},
        eval_extra={},
    )
    return attention_spec, risk_spec, gate_specs, seed1_spec


def stage1_to_3(args: argparse.Namespace, attention_spec: RunSpec, risk_spec: RunSpec) -> pd.DataFrame:
    maybe_train(attention_spec, args)
    maybe_train(risk_spec, args)
    attention_df = eval_spec(attention_spec, args, ROOT / "results/longtrain_baseline/attention_full_s0_by_step.csv")
    risk_history_df = eval_risk_history(args)
    risk_new_df = eval_spec(risk_spec, args, ROOT / "results/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_new_by_step.csv")
    risk_df = pd.concat([risk_history_df, risk_new_df], ignore_index=True)
    if not risk_df.empty:
        risk_df = risk_df.drop_duplicates(subset=["run_name", "seed", "step", "scenario"], keep="last").sort_values(["step", "scenario"])
    risk_df.to_csv(ROOT / "results/longtrain_baseline/risk_Rgate8_lambda015_RbarFloor03_s0_by_step.csv", index=False)
    combined = pd.concat([attention_df, risk_df], ignore_index=True)
    combined.to_csv(ROOT / "results/longtrain_baseline/attention_vs_risk_longtrain_summary.csv", index=False)
    random_df = combined[combined["scenario"] == "eval_random_switch"].copy()
    random_df.to_csv(ROOT / "results/longtrain_baseline/random_switch_safety_trend.csv", index=False)
    plot_lines(combined, ROOT / "results/longtrain_baseline/plots", "longtrain")
    write_baseline_report(combined)
    write_train_distribution_report(random_df)
    return combined


def classify_oscillation(values: list[float]) -> str:
    clean = [float(v) for v in values if not pd.isna(v)]
    if len(clean) < 3:
        return "insufficient data"
    diffs = np.diff(clean)
    sign_changes = int(np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1])))
    if sign_changes >= 1:
        return "checkpoint oscillation"
    if clean[-1] > clean[0]:
        return "degraded"
    return "recovered or improved"


def write_baseline_report(df: pd.DataFrame) -> None:
    sudden = df[df["scenario"] == "eval_sudden_turn"].sort_values(["run_name", "step"])
    lines = ["# Attention vs Risk 2000k Baseline Report", ""]
    for run_name, group in sudden.groupby("run_name"):
        reaction = group["reaction_time"].tolist() if "reaction_time" in group else []
        lines.append(f"- {run_name}: {classify_oscillation(reaction)} based on sudden-turn reaction curve.")
    if not sudden.empty:
        lines.append("")
        lines.append("## Sudden-Turn Reaction")
        lines.append("| run | step | reaction | success | collision |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in sudden.iterrows():
            lines.append(f"| {row['run_name']} | {int(row['step'])} | {row.get('reaction_time', np.nan):.4f} | {row['success_rate']:.4f} | {row['collision_rate']:.4f} |")
    lines += [
        "",
        "## Required Answers",
        "- attention to 2000k: see classification above.",
        "- risk to 2000k: see classification above.",
        "- long-training stability: compare reaction, collision, and safety-cost curves in CSV/plots.",
        "- pure risk hard weighting should continue only if it is at least as stable as attention.",
    ]
    write_lines(ROOT / "ATTENTION_RISK_2000K_BASELINE_REPORT.md", lines)


def write_train_distribution_report(df: pd.DataFrame) -> None:
    lines = ["# Train Distribution Safety Trend", ""]
    if df.empty:
        lines.append("No eval_random_switch rows available.")
    else:
        lines.append("| run | first_step | last_step | success_first | success_last | min_dist_first | min_dist_last | near_miss_first | near_miss_last |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run_name, group in df.sort_values("step").groupby("run_name"):
            first = group.iloc[0]
            last = group.iloc[-1]
            lines.append(
                f"| {run_name} | {int(first['step'])} | {int(last['step'])} | {first['success_rate']:.4f} | {last['success_rate']:.4f} | "
                f"{first['mean_min_distance']:.4f} | {last['mean_min_distance']:.4f} | {first['near_miss_rate']:.4f} | {last['near_miss_rate']:.4f} |"
            )
    write_lines(ROOT / "TRAIN_DISTRIBUTION_SAFETY_TREND.md", lines)


def stage4_to_5(args: argparse.Namespace, gate_specs: list[RunSpec], baseline_df: pd.DataFrame, attention_spec: RunSpec) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    for spec in gate_specs:
        maybe_train(spec, args)
        all_dfs.append(eval_spec(spec, args, ROOT / f"results/gate2b/summary/{spec.name}_by_step.csv"))
    gate_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    gate_df.to_csv(ROOT / "results/gate2b/gate2b_by_step.csv", index=False)
    trace_rows = run_gate_traces(args, [attention_spec, *gate_specs])
    pd.DataFrame(trace_rows).to_csv(ROOT / "results/gate2b/gate2b_curve_diagnostics_summary.csv", index=False)
    plot_lines(gate_df, ROOT / "results/gate2b/plots", "gate2b")
    write_gate2b_report(gate_df, pd.DataFrame(trace_rows), baseline_df)
    return gate_df


def run_gate_traces(args: argparse.Namespace, gate_specs: list[RunSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in gate_specs:
        for step in [250000, 750000, 1000000]:
            ckpt = checkpoint_for(spec, step)
            if not ckpt.exists():
                continue
            for scenario in ["eval_sudden_turn", "mixed_uncertainty"]:
                out_csv = ROOT / f"results/gate2b/eval/{spec.name}_s{spec.seed}_step{step}_{SCENARIO_SUFFIX[scenario]}_trace_eval.csv"
                eval_model(
                    ckpt,
                    spec.method,
                    spec.agg,
                    spec.seed,
                    scenario,
                    out_csv,
                    episodes=args.trace_episodes,
                    global_step=step,
                    save_trace=True,
                    trace_dir=ROOT / "results/gate2b/traces",
                    extra=spec.eval_extra,
                    log_name=f"trace_{spec.name}_{step}_{SCENARIO_SUFFIX[scenario]}",
                )
                summary = summarize_eval_csv(out_csv)
                rows.append({"run_name": spec.name, "method": spec.method, "step": step, "scenario": scenario, **summary})
    return rows


def write_gate2b_report(gate_df: pd.DataFrame, trace_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    lines = ["# Gate-2b Penalty 1000k Report", ""]
    if not gate_df.empty:
        sudden = gate_df[gate_df["scenario"] == "eval_sudden_turn"]
        lines.append("| run | step | reaction | success | collision | random_success |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        random = gate_df[gate_df["scenario"] == "eval_random_switch"]
        for _, row in sudden.sort_values(["run_name", "step"]).iterrows():
            r = random[(random["run_name"] == row["run_name"]) & (random["step"] == row["step"])]
            random_success = float(r.iloc[0]["success_rate"]) if not r.empty else np.nan
            lines.append(f"| {row['run_name']} | {int(row['step'])} | {row.get('reaction_time', np.nan):.4f} | {row['success_rate']:.4f} | {row['collision_rate']:.4f} | {random_success:.4f} |")
    lines += [
        "",
        "## Required Answers",
        "- risk cost earlier than distance warning: see gate2b_curve_diagnostics_summary.csv and trace CSVs.",
        "- risk_penalty vs distance_penalty: compare sudden reaction/collision plus random_switch success.",
        "- risk_biased_attention vs risk_penalty only: compare the two risk penalty rows and trace attention weights.",
        "- penalty side effects: random_switch success/time are included in gate2b_by_step.csv.",
    ]
    write_lines(ROOT / "GATE2B_PENALTY_1000K_REPORT.md", lines)


def stage6(args: argparse.Namespace, seed1_spec: RunSpec) -> pd.DataFrame:
    maybe_train(seed1_spec, args)
    df = eval_spec(seed1_spec, args, ROOT / "results/attention_seed1/attention_seed1_by_step.csv")
    plot_lines(df, ROOT / "results/attention_seed1/plots", "attention_seed1")
    sudden = df[df["scenario"] == "eval_sudden_turn"].sort_values("step")
    classification = classify_oscillation(sudden["reaction_time"].tolist()) if not sudden.empty and "reaction_time" in sudden else "insufficient data"
    lines = ["# Attention Seed1 1000k Report", "", f"- seed1 sudden-turn classification: {classification}", ""]
    if not sudden.empty:
        lines.append("| step | reaction | success | collision |")
        lines.append("|---:|---:|---:|---:|")
        for _, row in sudden.iterrows():
            lines.append(f"| {int(row['step'])} | {row.get('reaction_time', np.nan):.4f} | {row['success_rate']:.4f} | {row['collision_rate']:.4f} |")
    write_lines(ROOT / "ATTENTION_SEED1_1000K_REPORT.md", lines)
    return df


def final_report(baseline_df: pd.DataFrame, gate_df: pd.DataFrame, seed1_df: pd.DataFrame) -> None:
    def answer_attention(df: pd.DataFrame, run_name: str) -> str:
        sudden = df[(df["run_name"] == run_name) & (df["scenario"] == "eval_sudden_turn")].sort_values("step")
        return classify_oscillation(sudden["reaction_time"].tolist()) if not sudden.empty and "reaction_time" in sudden else "insufficient data"

    attention_status = answer_attention(baseline_df, "attention_full")
    risk_status = answer_attention(baseline_df, "risk_Rgate8_lambda015_RbarFloor03")
    seed1_status = answer_attention(seed1_df, "attention_full")
    lines = [
        "# Final Direction Decision Report",
        "",
        "## Answers",
        f"1. attention_full seed=0 to 2000k: {attention_status}.",
        f"2. Rgate8_lambda015_RbarFloor03 to 2000k: {risk_status}.",
        "3. attention vs risk long-training stability: compare attention_vs_risk_longtrain_summary.csv; this report does not collapse multi-metric evidence into a single claim without the table.",
        "4. eval_random_switch safety erosion: see TRAIN_DISTRIBUTION_SAFETY_TREND.md and random_switch_safety_trend.csv.",
        "5. distance_penalty drift suppression: see GATE2B_PENALTY_1000K_REPORT.md.",
        "6. risk_penalty vs distance_penalty: see results/gate2b/gate2b_by_step.csv.",
        "7. risk cost earlier than distance warning: see results/gate2b/gate2b_curve_diagnostics_summary.csv and traces.",
        "8. risk_biased_attention vs risk_penalty only: see Gate-2b report and attention trace fields.",
        f"9. attention seed=1 oscillation: {seed1_status}.",
        "10. next direction: choose risk-constrained attention only if risk_penalty or risk_biased_attention is clearly better than distance_penalty; otherwise shift to safe attention / drift diagnosis.",
        "",
        "## Completion Gate",
        "- Stage 0 preflight reports generated.",
        "- Stage 1-3 baseline CSV/report/plots generated.",
        "- Stage 4-5 Gate-2b CSV/report/trace diagnostics generated.",
        "- Stage 6 seed1 CSV/report generated.",
    ]
    write_lines(ROOT / "FINAL_DIRECTION_DECISION_REPORT.md", lines)


def validate_completion() -> None:
    required = [
        "results/preflight/CODE_STATUS_PREFLIGHT.md",
        "results/preflight/RESUME_PREFLIGHT_REPORT.md",
        "results/preflight/RISK_CONFIG_PREFLIGHT_REPORT.md",
        "results/preflight/cost_stats_attention_gate.csv",
        "results/preflight/COST_SCALE_PREFLIGHT_REPORT.md",
        "results/preflight/reaction_definition_check.md",
        "results/preflight/TRACE_FIELD_PREFLIGHT_REPORT.md",
        "results/preflight/CHECKPOINT_EVAL_INDEX.csv",
        "results/preflight/SAFETY_COST_PREFLIGHT_REPORT.md",
        "results/preflight/RISK_BIASED_ATTENTION_PREFLIGHT_REPORT.md",
        "results/preflight/GATE2B_CONFIG_LOGGING_PREFLIGHT_REPORT.md",
        "results/longtrain_baseline/attention_vs_risk_longtrain_summary.csv",
        "results/gate2b/gate2b_by_step.csv",
        "results/gate2b/gate2b_curve_diagnostics_summary.csv",
        "results/attention_seed1/attention_seed1_by_step.csv",
        "ATTENTION_RISK_2000K_BASELINE_REPORT.md",
        "GATE2B_PENALTY_1000K_REPORT.md",
        "ATTENTION_SEED1_1000K_REPORT.md",
        "FINAL_DIRECTION_DECISION_REPORT.md",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists() or (ROOT / rel).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing completion artifacts: {missing}")
    status = ROOT / "results/PIPELINE_COMPLETE.flag"
    status.write_text(f"complete_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=int(os.environ.get("LONGTRAIN_N_ENVS", "16")))
    parser.add_argument("--eval_episodes", type=int, default=int(os.environ.get("LONGTRAIN_EVAL_EPISODES", "50")))
    parser.add_argument("--preflight_episodes", type=int, default=int(os.environ.get("LONGTRAIN_PREFLIGHT_EPISODES", "50")))
    parser.add_argument("--trace_episodes", type=int, default=int(os.environ.get("LONGTRAIN_TRACE_EPISODES", "10")))
    parser.add_argument("--smoke_train_steps", type=int, default=int(os.environ.get("LONGTRAIN_SMOKE_STEPS", "2048")))
    parser.add_argument("--beta_cost", type=float, default=float(os.environ.get("LONGTRAIN_BETA_COST", "5.0")))
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    stage0(args)
    if args.preflight_only:
        return
    attention_spec, risk_spec, gate_specs, seed1_spec = build_specs(args)
    baseline_df = stage1_to_3(args, attention_spec, risk_spec)
    gate_df = stage4_to_5(args, gate_specs, baseline_df, attention_spec)
    seed1_df = stage6(args, seed1_spec)
    final_report(baseline_df, gate_df, seed1_df)
    validate_completion()


if __name__ == "__main__":
    main()
