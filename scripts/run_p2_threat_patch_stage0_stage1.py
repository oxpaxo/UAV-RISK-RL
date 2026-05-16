from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_env import DynamicObstacleEnv
from scripts.run_p2_rich_motion import (
    LOG_DIR,
    NEW_SCENARIOS,
    OOD_DIR,
    OUT_DIR,
    PLOTS_DIR,
    SCENARIO_SUFFIX,
    STAGE1_METHODS,
    STAGE1_STEPS,
    eval_if_needed,
    fmt,
    rel,
    stage1_checkpoint,
    stage1_pass,
    summarize_eval_csv,
    write_lines,
    write_stage1_report,
)

PATCH_SCENARIOS = ["eval_mixed_v2", "eval_threat_validated_sudden"]
PATCH_RANDOM_EPISODES = 50
PATCH_SHORT_EPISODES = 50
SHORT_MODEL = ROOT / "checkpoints/p2_rich_motion/p2_short_attention_full_s0_step50000.zip"
PATCH_OUT = OUT_DIR


def ensure_dirs() -> None:
    for path in [PATCH_OUT, OOD_DIR, PLOTS_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def random_policy_patched() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for scenario in PATCH_SCENARIOS:
        stats: dict[str, list[float]] = {
            "success": [],
            "collision": [],
            "near_miss": [],
            "realized_near_miss": [],
            "min_distance": [],
            "time": [],
            "scenario_valid": [],
            "planned_threat_valid": [],
            "init_collision": [],
            "predicted_cpa": [],
        }
        invalid = Counter()
        for episode_id in range(PATCH_RANDOM_EPISODES):
            env = DynamicObstacleEnv(scenario=scenario)
            _, info = env.reset(seed=1000 + episode_id)
            init_collision = int(float(info.get("initial_min_distance", 999.0)) < env.d_collision)
            done = False
            steps = 0
            while not done:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                steps += 1
                done = terminated or truncated
            reason = str(info.get("invalid_reason", "none"))
            invalid[reason] += 1
            stats["success"].append(float(info["is_success"]))
            stats["collision"].append(float(info["is_collision"]))
            stats["near_miss"].append(float(float(info["episode_min_distance"]) < 1.0))
            stats["realized_near_miss"].append(float(info.get("realized_near_miss", False)))
            stats["min_distance"].append(float(info["episode_min_distance"]))
            stats["time"].append(float(steps * env.dt))
            stats["scenario_valid"].append(float(info.get("scenario_valid", True)))
            stats["planned_threat_valid"].append(float(info.get("planned_threat_valid", info.get("threat_valid", True))))
            stats["init_collision"].append(float(init_collision))
            stats["predicted_cpa"].append(float(info.get("predicted_cpa_to_nominal_path", np.nan)))
        rows.append(
            {
                "scenario": scenario,
                "policy": "random",
                "episodes": PATCH_RANDOM_EPISODES,
                "success_rate": float(np.mean(stats["success"])),
                "collision_rate": float(np.mean(stats["collision"])),
                "near_miss_rate": float(np.mean(stats["near_miss"])),
                "realized_near_miss_rate": float(np.mean(stats["realized_near_miss"])),
                "mean_min_distance": float(np.mean(stats["min_distance"])),
                "min_min_distance": float(np.min(stats["min_distance"])),
                "mean_time": float(np.mean(stats["time"])),
                "scenario_valid_rate": float(np.mean(stats["scenario_valid"])),
                "planned_threat_valid_rate": float(np.mean(stats["planned_threat_valid"])),
                "init_collision_rate": float(np.mean(stats["init_collision"])),
                "predicted_cpa_to_nominal_path_mean": float(np.nanmean(stats["predicted_cpa"])),
            }
        )
        for reason, count in sorted(invalid.items()):
            invalid_rows.append({"stage": "patched_random_policy", "scenario": scenario, "invalid_reason": reason, "count": int(count), "rate": count / PATCH_RANDOM_EPISODES})
    return pd.DataFrame(rows), pd.DataFrame(invalid_rows)


def eval_short_patched() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    if not SHORT_MODEL.exists():
        raise FileNotFoundError(f"missing short PPO checkpoint: {SHORT_MODEL}")
    for scenario in PATCH_SCENARIOS:
        out_csv = PATCH_OUT / "short_ppo_eval_patched" / f"p2_short_attention_full_s0_step50000_{SCENARIO_SUFFIX[scenario]}.csv"
        if len(pd.read_csv(out_csv)) < PATCH_SHORT_EPISODES if out_csv.exists() else True:
            cmd = [
                PYTHON,
                "eval.py",
                "--model_path",
                str(SHORT_MODEL),
                "--method",
                "p2_short_attention_full",
                "--profile_mode",
                "full_12",
                "--agg",
                "attention",
                "--seed",
                "0",
                "--eval_seed",
                "1000",
                "--episodes",
                str(PATCH_SHORT_EPISODES),
                "--scenario",
                scenario,
                "--device",
                "cpu",
                "--out_csv",
                str(out_csv),
                "--global_step",
                "50000",
                "--heartbeat_seconds",
                "15",
                "--d_warning",
                "1.0",
            ]
            run_command(cmd, LOG_DIR / f"eval_p2_stage0_short_patched_{scenario}.log", skip_marker="EVAL_END")
        df = pd.read_csv(out_csv)
        rows.append(
            {
                "scenario": scenario,
                "policy": "short_ppo",
                "episodes": len(df),
                "success_rate": float(pd.to_numeric(df["success"], errors="coerce").mean()),
                "collision_rate": float(pd.to_numeric(df["collision"], errors="coerce").mean()),
                "near_miss_rate": float(pd.to_numeric(df["near_miss"], errors="coerce").mean()),
                "realized_near_miss_rate": float(pd.to_numeric(df["realized_near_miss"], errors="coerce").mean()),
                "mean_min_distance": float(pd.to_numeric(df["episode_min_distance"], errors="coerce").mean()),
                "min_min_distance": float(pd.to_numeric(df["episode_min_distance"], errors="coerce").min()),
                "mean_time": float(pd.to_numeric(df["time_to_goal"], errors="coerce").mean()),
                "mean_episode_reward": float(pd.to_numeric(df["episode_reward"], errors="coerce").mean()),
                "scenario_valid_rate": float(pd.to_numeric(df["scenario_valid"], errors="coerce").mean()),
                "planned_threat_valid_rate": float(pd.to_numeric(df["planned_threat_valid"], errors="coerce").mean()),
                "init_collision_rate": float((pd.to_numeric(df["initial_min_distance"], errors="coerce") < 0.55).mean()),
                "predicted_cpa_to_nominal_path_mean": float(pd.to_numeric(df["predicted_cpa_to_nominal_path"], errors="coerce").mean()),
            }
        )
        invalid = df["invalid_reason"].fillna("none").astype(str).value_counts()
        for reason, count in invalid.items():
            invalid_rows.append({"stage": "patched_short_ppo", "scenario": scenario, "invalid_reason": reason, "count": int(count), "rate": float(count / len(df))})
    return pd.DataFrame(rows), pd.DataFrame(invalid_rows)


def patched_stage0_pass(random_df: pd.DataFrame, short_df: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for label, df in [("random", random_df), ("short_ppo", short_df)]:
        if (df["scenario_valid_rate"] < 1.0).any():
            reasons.append(f"{label}_scenario_valid_rate_below_1")
        if (df["planned_threat_valid_rate"] < 0.8).any():
            reasons.append(f"{label}_planned_threat_valid_rate_below_0p8")
        if (df["init_collision_rate"] > 0.0).any():
            reasons.append(f"{label}_init_collision_rate_gt_0")
        if (df["collision_rate"] >= 1.0).any():
            reasons.append(f"{label}_all_collision")
        if (df["realized_near_miss_rate"] <= 0.0).any():
            reasons.append(f"{label}_no_realized_near_miss")
    return not reasons, reasons


def write_patched_report(random_df: pd.DataFrame, short_df: pd.DataFrame, passed: bool, reasons: list[str]) -> None:
    lines = [
        "# P2 Environment Sanity Report Patched",
        "",
        f"- Passed patched Stage 0 gate: {passed}",
        f"- Reasons: {', '.join(reasons) if reasons else 'none'}",
        "- Gate uses planned_threat_valid_rate, not realized_near_miss_rate.",
        "",
        "## Random Policy 50 Episodes",
        "| scenario | success | collision | near_miss | realized_near | planned_valid | scenario_valid | init_collision | cpa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in random_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['realized_near_miss_rate'])} | {fmt(row['planned_threat_valid_rate'])} | {fmt(row['scenario_valid_rate'])} | "
            f"{fmt(row['init_collision_rate'])} | {fmt(row['predicted_cpa_to_nominal_path_mean'])} |"
        )
    lines += [
        "",
        "## Short PPO 50 Episodes",
        "| scenario | success | collision | near_miss | realized_near | planned_valid | scenario_valid | init_collision | cpa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in short_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['realized_near_miss_rate'])} | {fmt(row['planned_threat_valid_rate'])} | {fmt(row['scenario_valid_rate'])} | "
            f"{fmt(row['init_collision_rate'])} | {fmt(row['predicted_cpa_to_nominal_path_mean'])} |"
        )
    write_lines(ROOT / "P2_ENVIRONMENT_SANITY_REPORT_PATCHED.md", lines)


def run_stage1_after_patch() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in STAGE1_METHODS:
        for step in STAGE1_STEPS:
            ckpt = stage1_checkpoint(spec, step)
            for scenario in NEW_SCENARIOS:
                path = eval_if_needed(spec, 0, step, scenario, ckpt, OOD_DIR, "eval_p2_stage1_ood", episodes=50)
                rows.append(summarize_eval_csv(path, spec.display_method, 0, step, scenario, spec.d_warning, ckpt))
    out = pd.DataFrame(rows).sort_values(["method", "step", "scenario"])
    out.to_csv(OUT_DIR / "p2_stage1_ood_eval.csv", index=False)
    ok, reasons = stage1_pass(out)
    write_stage1_report(out, ok, reasons)
    return out


def write_flag(terminal_decision: str, random_df: pd.DataFrame, short_df: pd.DataFrame, stage1_df: pd.DataFrame | None, reasons: list[str]) -> None:
    flag = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "terminal_decision": terminal_decision,
        "patched_random_rows": len(random_df),
        "patched_short_rows": len(short_df),
        "stage1_rows": 0 if stage1_df is None else len(stage1_df),
        "reasons": reasons,
    }
    (OUT_DIR / "P2_PATCHED_STAGE1_COMPLETE.flag").write_text(json.dumps(flag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] patched flag written terminal_decision={terminal_decision}", flush=True)


def main() -> None:
    ensure_dirs()
    random_df, invalid_random = random_policy_patched()
    short_df, invalid_short = eval_short_patched()
    random_df.to_csv(OUT_DIR / "env_sanity_random_policy_patched.csv", index=False)
    short_df.to_csv(OUT_DIR / "env_sanity_short_ppo_patched.csv", index=False)
    invalid_df = pd.concat([invalid_random, invalid_short], ignore_index=True)
    invalid_df.to_csv(OUT_DIR / "threat_generation_invalid_reasons.csv", index=False)
    passed, reasons = patched_stage0_pass(random_df, short_df)
    write_patched_report(random_df, short_df, passed, reasons)
    if not passed:
        write_flag("patched_stage0_no_go", random_df, short_df, None, reasons)
        return
    stage1_df = run_stage1_after_patch()
    write_flag("patched_stage1_complete", random_df, short_df, stage1_df, reasons)


if __name__ == "__main__":
    main()
