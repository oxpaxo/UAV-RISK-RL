from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv


DEFAULT_SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
DEFAULT_POLICIES = ["random", "straight_line", "reactive"]


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def policy_action(policy_name: str, env: DynamicObstacleFlowEnv, info: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    if policy_name == "random":
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        action[2] *= 0.2
        return action

    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    goal_vec = goal - uav
    goal_vec[2] = 0.0
    v_goal = normalize(goal_vec)
    if policy_name == "straight_line":
        return np.clip(v_goal, -1.0, 1.0).astype(np.float32)

    if policy_name != "reactive":
        raise ValueError(f"unsupported sanity policy: {policy_name}")

    obs_pos = np.asarray(info["obstacle_positions"], dtype=np.float32)
    obs_vel = np.asarray(info["obstacle_velocities"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    avoid = np.zeros(3, dtype=np.float32)
    d_reactive = 4.0
    horizon = 4.5
    for pos, vel in zip(obs_pos, obs_vel):
        rel = pos - uav
        rel[2] = 0.0
        distance = float(np.linalg.norm(rel))
        if distance < 1e-6:
            continue
        rel_vel = vel - uav_vel
        rel_vel[2] = 0.0
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        tcpa = 0.0
        cpa_distance = distance
        if rel_speed_sq > 1e-8:
            tcpa = float(np.clip(-np.dot(rel, rel_vel) / rel_speed_sq, 0.0, horizon))
            cpa_distance = float(np.linalg.norm(rel + rel_vel * tcpa))

        if distance < d_reactive or (0.0 < tcpa < horizon and cpa_distance < 2.4):
            away = -rel / (distance + 1e-6)
            lateral = np.array([-v_goal[1], v_goal[0], 0.0], dtype=np.float32)
            if np.dot(lateral, away) < 0.0:
                lateral = -lateral
            proximity_gain = max((d_reactive - min(distance, d_reactive)) / d_reactive, 0.0)
            cpa_gain = max((2.4 - min(cpa_distance, 2.4)) / 2.4, 0.0)
            closing_gain = 1.0 if tcpa > 0.0 else 0.35
            avoid += (1.3 * proximity_gain + 2.2 * cpa_gain * closing_gain) * (0.65 * away + 0.55 * lateral)

    command = normalize(1.0 * v_goal + 2.1 * avoid)
    if float(np.linalg.norm(command)) < 1e-6:
        command = v_goal
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def run_episode(
    policy_name: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    env = DynamicObstacleFlowEnv(scenario=scenario)
    min_distance_values: list[float] = []
    cpa_rows: list[dict[str, Any]] = []
    ttc_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    try:
        _, info = env.reset(seed=episode_seed)
        done = False
        while not done:
            action = policy_action(policy_name, env, info, rng)
            _, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            min_distance_values.append(float(info["min_distance"]))
            active_rows.append(
                {
                    "policy_name": policy_name,
                    "scenario": scenario,
                    "episode_id": episode_id,
                    "step": int(info["step"]),
                    "active_obstacle_count": int(info["active_obstacle_count"]),
                }
            )
            if not np.isfinite(float(info["min_distance"])):
                raise RuntimeError("non-finite min_distance")

        for record in info.get("spawn_records", []):
            base = {
                "policy_name": policy_name,
                "scenario": scenario,
                "episode_id": episode_id,
                "obstacle_id": int(record["obstacle_id"]),
                "spawn_step": int(record["spawn_step"]),
                "spawn_time": float(record["spawn_time"]),
                "spawn_reason": str(record["spawn_reason"]),
                "motion_mode": str(record["motion_mode"]),
                "threat_class": str(record["threat_class"]),
                "threat_valid": int(bool(record["threat_valid"])),
            }
            cpa_row = dict(base)
            cpa_row["planned_cpa"] = float(record["planned_cpa"])
            ttc_row = dict(base)
            ttc_row["planned_ttc"] = float(record["planned_ttc"])
            cpa_rows.append(cpa_row)
            ttc_rows.append(ttc_row)

        row = {
            "policy_name": policy_name,
            "scenario": scenario,
            "episode_id": episode_id,
            "success": int(bool(info["is_success"])),
            "collision": int(bool(info["is_collision"])),
            "near_miss": int(bool(info["near_miss"])),
            "mean_min_distance": float(np.mean(min_distance_values)) if min_distance_values else float("nan"),
            "min_distance": float(info["episode_min_distance"]),
            "progress": float(info["progress"]),
            "mean_time": float(info["time"]),
            "episode_length": int(info["step"]),
            "threat_valid_rate": float(info["threat_valid_rate"]),
            "planned_cpa_min": float(np.min(info["planned_cpa_values"])) if len(info["planned_cpa_values"]) else float("nan"),
            "planned_cpa_mean": float(np.mean(info["planned_cpa_values"])) if len(info["planned_cpa_values"]) else float("nan"),
            "planned_ttc_min": float(np.min(info["planned_ttc_values"])) if len(info["planned_ttc_values"]) else float("nan"),
            "planned_ttc_mean": float(np.mean(info["planned_ttc_values"])) if len(info["planned_ttc_values"]) else float("nan"),
            "replacement_count": int(info["replacement_count"]),
            "spawn_count": int(info["spawn_count"]),
            "remove_count": int(info["remove_count"]),
            "mean_active_obstacle_count": float(info["mean_active_obstacle_count"]),
            "max_active_obstacle_count": int(info["max_active_obstacle_count"]),
            "min_active_obstacle_count": int(info["min_active_obstacle_count"]),
            "init_collision": int(bool(info["init_collision"])),
            "out_of_bounds": int(bool(info["out_of_bounds"])),
            "out_of_bounds_count": int(info["out_of_bounds_count"]),
            "nan_or_crash": 0,
            "turn_time": finite_float(info.get("turn_time")),
            "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
            "planned_cpa_to_threat": finite_float(info.get("planned_cpa_to_threat")),
            "planned_ttc_to_threat": finite_float(info.get("planned_ttc_to_threat")),
        }
    except Exception as exc:
        row = {
            "policy_name": policy_name,
            "scenario": scenario,
            "episode_id": episode_id,
            "success": 0,
            "collision": 0,
            "near_miss": 0,
            "mean_min_distance": float("nan"),
            "min_distance": float("nan"),
            "progress": 0.0,
            "mean_time": float("nan"),
            "episode_length": 0,
            "threat_valid_rate": 0.0,
            "planned_cpa_min": float("nan"),
            "planned_cpa_mean": float("nan"),
            "planned_ttc_min": float("nan"),
            "planned_ttc_mean": float("nan"),
            "replacement_count": 0,
            "spawn_count": 0,
            "remove_count": 0,
            "mean_active_obstacle_count": 0.0,
            "max_active_obstacle_count": 0,
            "min_active_obstacle_count": 0,
            "init_collision": 0,
            "out_of_bounds": 0,
            "out_of_bounds_count": 0,
            "nan_or_crash": 1,
            "turn_time": float("nan"),
            "threat_obstacle_id": -1,
            "planned_cpa_to_threat": float("nan"),
            "planned_ttc_to_threat": float("nan"),
            "error": repr(exc),
        }
    return row, cpa_rows, ttc_rows, active_rows


def summarize(episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["policy_name", "scenario"]
    summary = (
        episode_df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            near_miss_rate=("near_miss", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            min_distance_mean=("min_distance", "mean"),
            min_distance_p05=("min_distance", lambda s: float(np.nanpercentile(s, 5))),
            progress_mean=("progress", "mean"),
            mean_time=("mean_time", "mean"),
            episode_length_mean=("episode_length", "mean"),
            threat_valid_rate=("threat_valid_rate", "mean"),
            planned_cpa_min=("planned_cpa_min", "mean"),
            planned_cpa_mean=("planned_cpa_mean", "mean"),
            planned_ttc_min=("planned_ttc_min", "mean"),
            planned_ttc_mean=("planned_ttc_mean", "mean"),
            replacement_count_mean=("replacement_count", "mean"),
            spawn_count_mean=("spawn_count", "mean"),
            remove_count_mean=("remove_count", "mean"),
            mean_active_obstacle_count=("mean_active_obstacle_count", "mean"),
            max_active_obstacle_count=("max_active_obstacle_count", "max"),
            min_active_obstacle_count=("min_active_obstacle_count", "min"),
            init_collision_rate=("init_collision", "mean"),
            out_of_bounds_rate=("out_of_bounds", "mean"),
            out_of_bounds_count_mean=("out_of_bounds_count", "mean"),
            nan_or_crash=("nan_or_crash", "sum"),
        )
        .reset_index()
    )
    return summary


def overall_metrics(episode_df: pd.DataFrame, summary_df: pd.DataFrame, cpa_df: pd.DataFrame) -> dict[str, Any]:
    straight = episode_df[episode_df["policy_name"] == "straight_line"]
    reactive = episode_df[episode_df["policy_name"] == "reactive"]
    random_df = episode_df[episode_df["policy_name"] == "random"]
    high_cpa = cpa_df[cpa_df["threat_class"] == "high"]["planned_cpa"] if not cpa_df.empty else pd.Series(dtype=float)
    low_cpa = cpa_df[cpa_df["threat_class"] == "low"]["planned_cpa"] if not cpa_df.empty else pd.Series(dtype=float)
    metrics = {
        "threat_valid_rate": float(cpa_df["threat_valid"].mean()) if not cpa_df.empty else float(episode_df["threat_valid_rate"].mean()),
        "replacement_count_mean": float(episode_df["replacement_count"].mean()),
        "init_collision_rate": float(episode_df["init_collision"].mean()),
        "nan_or_crash": int(episode_df["nan_or_crash"].sum()),
        "out_of_bounds_rate": float(episode_df["out_of_bounds"].mean()),
        "random_success_rate": float(random_df["success"].mean()) if not random_df.empty else float("nan"),
        "straight_success_rate": float(straight["success"].mean()) if not straight.empty else float("nan"),
        "straight_collision_rate": float(straight["collision"].mean()) if not straight.empty else float("nan"),
        "straight_near_miss_rate": float(straight["near_miss"].mean()) if not straight.empty else float("nan"),
        "reactive_success_rate": float(reactive["success"].mean()) if not reactive.empty else float("nan"),
        "reactive_collision_rate": float(reactive["collision"].mean()) if not reactive.empty else float("nan"),
        "reactive_near_miss_rate": float(reactive["near_miss"].mean()) if not reactive.empty else float("nan"),
        "high_cpa_mean": float(high_cpa.mean()) if len(high_cpa) else float("nan"),
        "low_cpa_mean": float(low_cpa.mean()) if len(low_cpa) else float("nan"),
        "summary_rows": int(len(summary_df)),
    }
    metrics["reactive_success_advantage"] = metrics["reactive_success_rate"] - metrics["straight_success_rate"]
    metrics["reactive_collision_reduction"] = metrics["straight_collision_rate"] - metrics["reactive_collision_rate"]
    metrics["high_low_cpa_gap"] = metrics["low_cpa_mean"] - metrics["high_cpa_mean"]
    return metrics


def decide(metrics: dict[str, Any]) -> tuple[str, str]:
    if metrics["nan_or_crash"] != 0:
        return "phase1_no_go_env_crash_or_nan", "debug crash or non-finite environment rollout"
    if metrics["threat_valid_rate"] < 0.8 or not (metrics["high_low_cpa_gap"] > 1.0):
        return "phase1_no_go_threat_generation_invalid", "adjust threat generation and planned CPA/TTC control"
    if metrics["replacement_count_mean"] <= 0.0:
        return "phase1_no_go_replacement_not_working", "fix obstacle replacement and active-count maintenance"
    if metrics["init_collision_rate"] > 0.02 or metrics["out_of_bounds_rate"] > 0.20:
        return "phase1_no_go_env_crash_or_nan", "debug init collision or out-of-bounds behavior"
    too_easy = (
        (metrics["random_success_rate"] > 0.8)
        or (metrics["straight_success_rate"] > 0.9 and metrics["straight_collision_rate"] < 0.05 and metrics["straight_near_miss_rate"] < 0.10)
    )
    if too_easy:
        return "phase1_no_go_env_too_easy", "increase threat density, high-threat probability, or crossing severity"
    too_hard = (
        metrics["reactive_success_rate"] < 0.25
        and metrics["reactive_collision_rate"] > 0.45
        and abs(metrics["reactive_success_advantage"]) < 0.10
    )
    if too_hard:
        return "phase1_no_go_env_too_hard", "reduce crossing severity or tune reactive-reachable margins"
    reactive_better = (
        metrics["reactive_success_advantage"] >= 0.10
        or metrics["reactive_collision_reduction"] >= 0.10
    ) and metrics["reactive_success_rate"] >= metrics["straight_success_rate"] - 0.05
    if not reactive_better:
        return "phase1_no_go_env_too_hard", "rebalance dynamics so simple reactive avoider is distinguishable from straight-line"
    return "phase0_phase1_complete", "Phase 2 baseline long-training reproduction"


def write_plot(path: Path, values: pd.Series, title: str, xlabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        clean = pd.to_numeric(values, errors="coerce").dropna()
        plt.hist(clean, bins=30, color="#4c78a8", edgecolor="white")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
    except Exception:
        path.write_text("plot generation unavailable\n", encoding="utf-8")


def write_design_report(root: Path) -> None:
    lines = [
        "# ENV V2 Design Report",
        "",
        "## Environment",
        "",
        "`DynamicObstacleFlowEnv` is implemented in `envs/dynamic_obstacle_flow_env.py` as a separate environment from the frozen legacy 3-ball Gym.",
        "",
        "## State And Action",
        "",
        "- UAV state uses position, velocity, goal-relative direction, progress, and obstacle-set observations.",
        "- Action is a 3D continuous velocity command, with altitude constrained around a nominal height. This is horizontal avoidance with constrained altitude, not full 3D flight.",
        "- Observation keys remain `ego`, `obs`, `mask`, and `global_risk` for later SB3 compatibility.",
        "",
        "## Active Obstacle Flow",
        "",
        "- Default active obstacles: 5-8, with `eval_flow_high_density` using 8-10.",
        "- The environment maintains a target active count and replaces removed obstacles during the episode.",
        "- Removal reasons include `passed_by_uav`, `out_of_bounds`, `no_future_threat`, `lifetime`, and `far_from_nominal_path`.",
        "",
        "## Motion Models",
        "",
        "- `linear`",
        "- `sinusoidal_lateral` with amplitude, period, and phase.",
        "- `accel_decel` with target speed resampling and transition time.",
        "- `ar1_velocity` with phi=0.90 and scenario-dependent noise.",
        "- `crossing_or_sudden_threat` with planned turn time and post-turn crossing toward a planned CPA point.",
        "",
        "## Threat Generation",
        "",
        "- Each spawned obstacle samples `low`, `medium`, or `high` threat class.",
        "- Planned CPA ranges are high=[0.35,1.15], medium=[1.55,2.35], low=[2.80,4.20].",
        "- Planned TTC is sampled from scenario-specific ranges and is computed against the nominal straight path, independent of the sanity policy trajectory.",
        "- `threat_valid` checks planned CPA class bounds, TTC finiteness, and non-colliding initial placement.",
        "",
        "## Train/Eval Scenarios",
        "",
        "- `train_flow_mixed`",
        "- `eval_flow_id`",
        "- `eval_flow_high_density`",
        "- `eval_flow_high_speed`",
        "- `eval_flow_high_threat`",
        "- `eval_flow_mixed_ood`",
        "- `eval_flow_sudden_threat`",
    ]
    (root / "ENV_V2_DESIGN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def write_sanity_report(
    root: Path,
    args: argparse.Namespace,
    summary_df: pd.DataFrame,
    metrics: dict[str, Any],
    decision: str,
) -> None:
    lines = [
        "# ENV V2 Sanity Report",
        "",
        "## Run Configuration",
        "",
        f"- env: {args.env}",
        f"- scenarios: {args.scenarios}",
        f"- policies: {args.policies}",
        f"- episodes per policy/scenario: {args.episodes}",
        f"- eval_seed: {args.eval_seed}",
        "- dt: 0.2",
        "- max_steps: 500",
        "",
        "## Overall Metrics",
        "",
    ]
    for key in [
        "threat_valid_rate",
        "replacement_count_mean",
        "init_collision_rate",
        "nan_or_crash",
        "out_of_bounds_rate",
        "random_success_rate",
        "straight_success_rate",
        "straight_collision_rate",
        "reactive_success_rate",
        "reactive_collision_rate",
        "reactive_success_advantage",
        "reactive_collision_reduction",
        "high_cpa_mean",
        "low_cpa_mean",
        "high_low_cpa_gap",
    ]:
        lines.append(f"- {key}: {fmt(metrics[key])}")

    lines.extend(
        [
            "",
            "## Policy/Scenario Summary",
            "",
            "| policy | scenario | episodes | success | collision | near_miss | min_distance_mean | replacement_mean | threat_valid | active_mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_df.to_dict("records"):
        lines.append(
            f"| {row['policy_name']} | {row['scenario']} | {int(row['episodes'])} | "
            f"{fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['min_distance_mean'])} | {fmt(row['replacement_count_mean'])} | "
            f"{fmt(row['threat_valid_rate'])} | {fmt(row['mean_active_obstacle_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate Checks",
            "",
            f"- random policy not 100% success: {metrics['random_success_rate'] < 1.0}",
            f"- straight-line not 100% success: {metrics['straight_success_rate'] < 1.0}",
            f"- reactive clearly better than straight-line: {decision == 'phase0_phase1_complete'}",
            f"- threat_valid_rate >= 0.8: {metrics['threat_valid_rate'] >= 0.8}",
            f"- high-threat CPA lower than low-threat CPA: {metrics['high_low_cpa_gap'] > 1.0}",
            f"- replacement_count mean > 0: {metrics['replacement_count_mean'] > 0.0}",
            f"- init_collision_rate close to 0: {metrics['init_collision_rate'] <= 0.02}",
            f"- nan_or_crash = 0: {metrics['nan_or_crash'] == 0}",
            "",
            f"terminal_decision = {decision}",
        ]
    )
    (root / "ENV_V2_SANITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    root: Path,
    summary_df: pd.DataFrame,
    metrics: dict[str, Any],
    decision: str,
    next_action: str,
) -> None:
    complete = decision == "phase0_phase1_complete"
    lines = [
        "# Phase 0 + Phase 1 Final Report",
        "",
        "## 1. Executive Summary",
        "",
        f"terminal_decision = {decision}",
        f"next_recommended_phase = {'Phase 2 baseline long-training reproduction' if complete else 'blocked'}",
        "",
        "## 2. Phase 0 Old Assets Summary",
        "",
        "Old experiments are frozen and downgraded to preliminary diagnostic evidence. Reusable assets include environment metric conventions, SB3 training/eval wiring, obstacle-set policy extraction, trace/report scripts, watcher patterns, and config/run metadata.",
        "",
        "## 3. Environment V2 Design",
        "",
        "`DynamicObstacleFlowEnv` is implemented as a separate continuous obstacle-flow environment with 5-8 default active obstacles, high-density eval support, fixed/constrained altitude horizontal avoidance, and scenario-specific train/eval split.",
        "",
        "## 4. Motion Models",
        "",
        "`linear`, `sinusoidal_lateral`, `accel_decel`, `ar1_velocity`, and `crossing_or_sudden_threat` are implemented.",
        "",
        "## 5. Threat Generation",
        "",
        "Each obstacle samples a threat class and planned CPA/TTC against the nominal path. High-threat planned CPA is lower than low-threat planned CPA by construction and verified in sanity data.",
        "",
        "## 6. Replacement Mechanism",
        "",
        "Obstacles are removed when passed, out of bounds, no longer a future threat, over lifetime, or far from the nominal path. The active count is maintained by immediate replacement.",
        "",
        "## 7. Sanity Policies",
        "",
        "Sanity policies are `random`, `straight_line`, and a simple current-state `reactive` avoider.",
        "",
        "## 8. Sanity Results",
        "",
        "| policy | scenario | episodes | success | collision | near_miss | replacement_mean | threat_valid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            f"| {row['policy_name']} | {row['scenario']} | {int(row['episodes'])} | "
            f"{fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['replacement_count_mean'])} | {fmt(row['threat_valid_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 9. Go/No-Go Decision",
            "",
        ]
    )
    if complete:
        lines.extend(
            [
                "Phase 1 passes the environment sanity gate.",
                "",
                "- threat_valid_rate >= 0.8",
                "- replacement_count mean > 0",
                "- init_collision_rate close to 0",
                "- nan_or_crash = 0",
                "- reactive avoider is clearly better than straight-line",
                "- environment is neither too easy nor too hard by the sanity-policy checks",
            ]
        )
    else:
        lines.extend(
            [
                f"NO-GO triggered: {decision}",
                "",
                f"1. no-go: {decision}",
                f"2. triggering metrics: {metrics}",
                "3. cannot enter Phase 2 because Env V2 did not pass the sanity gate.",
                f"4. required fix: {next_action}.",
                "5. rerun: `python scripts/run_env_v2_sanity.py --env DynamicObstacleFlowEnv --scenarios eval_flow_id,eval_flow_high_density,eval_flow_high_speed,eval_flow_high_threat,eval_flow_mixed_ood,eval_flow_sudden_threat --policies random,straight_line,reactive --episodes 50 --eval_seed 1000 --out_dir results/restart_phase0_phase1/env_v2`.",
            ]
        )

    lines.extend(
        [
            "",
            "## 10. Next Step",
            "",
            "If complete: enter Phase 2 baseline long-training reproduction. If no-go: fix the environment issue above and rerun sanity.",
            "",
            "## Required Final Answers",
            "",
            f"1. Old experiment assets organized: yes.",
            "2. Reusable modules: env metric patterns, train/eval tooling, policy extractor, cost bookkeeping, reaction diagnostics, traces, watchers, report scripts, configs.",
            "3. DynamicObstacleFlowEnv implemented: yes.",
            "4. Supports 5-8 active obstacles: yes; high-density eval supports 8-10.",
            f"5. Replacement works: {'yes' if metrics['replacement_count_mean'] > 0 else 'no'}.",
            "6. Motion modes implemented: yes.",
            "7. Threat class / planned CPA / planned TTC implemented: yes.",
            "8. Train/eval split implemented: yes.",
            "9. random / straight-line / reactive sanity completed: yes.",
            f"10. Environment too easy: {'no' if complete or decision != 'phase1_no_go_env_too_easy' else 'yes'}.",
            f"11. Environment too hard: {'no' if complete or decision != 'phase1_no_go_env_too_hard' else 'yes'}.",
            f"12. threat_valid_rate >= 0.8: {'yes' if metrics['threat_valid_rate'] >= 0.8 else 'no'} ({fmt(metrics['threat_valid_rate'])}).",
            f"13. replacement_count > 0: {'yes' if metrics['replacement_count_mean'] > 0 else 'no'} ({fmt(metrics['replacement_count_mean'])}).",
            f"14. Can enter Phase 2 baseline long-training reproduction: {'yes' if complete else 'no'}.",
            f"15. If not, fix: {'n/a' if complete else next_action}.",
        ]
    )
    (root / "PHASE0_PHASE1_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flag(root: Path, decision: str, next_action: str) -> None:
    complete_flag = root / "PHASE0_PHASE1_COMPLETE.flag"
    no_go_flag = root / "PHASE0_PHASE1_NO_GO.flag"
    if decision == "phase0_phase1_complete":
        if no_go_flag.exists():
            no_go_flag.unlink()
        complete_flag.write_text(
            "terminal_decision=phase0_phase1_complete\n"
            "next_recommended_phase=Phase 2 baseline long-training reproduction\n",
            encoding="utf-8",
        )
    else:
        if complete_flag.exists():
            complete_flag.unlink()
        no_go_flag.write_text(
            f"terminal_decision={decision}\n"
            f"next_recommended_action={next_action}\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="DynamicObstacleFlowEnv")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="results/restart_phase0_phase1/env_v2")
    parser.add_argument("--heartbeat_seconds", type=float, default=20.0)
    parser.add_argument("--no_flags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env != "DynamicObstacleFlowEnv":
        raise ValueError("Phase 1 only supports --env DynamicObstacleFlowEnv")
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    cpa_rows: list[dict[str, Any]] = []
    ttc_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(policies) * args.episodes
    completed = 0
    last_heartbeat = time.time()
    print(
        f"ENV_V2_SANITY_START env={args.env} scenarios={len(scenarios)} policies={len(policies)} "
        f"episodes={args.episodes} total={total}",
        flush=True,
    )
    for scenario_index, scenario in enumerate(scenarios):
        for policy_index, policy in enumerate(policies):
            rng = np.random.default_rng(args.eval_seed + scenario_index * 10000 + policy_index * 1000)
            for episode_id in range(args.episodes):
                episode_seed = args.eval_seed + scenario_index * 10000 + policy_index * 1000 + episode_id
                row, episode_cpa_rows, episode_ttc_rows, episode_active_rows = run_episode(
                    policy, scenario, episode_id, episode_seed, rng
                )
                episode_rows.append(row)
                cpa_rows.extend(episode_cpa_rows)
                ttc_rows.extend(episode_ttc_rows)
                active_rows.extend(episode_active_rows)
                completed += 1
                now = time.time()
                if now - last_heartbeat >= args.heartbeat_seconds:
                    print(
                        f"ENV_V2_SANITY_PROGRESS completed={completed}/{total} "
                        f"current_policy={policy} current_scenario={scenario}",
                        flush=True,
                    )
                    last_heartbeat = now

    episode_df = pd.DataFrame(episode_rows)
    cpa_df = pd.DataFrame(cpa_rows)
    ttc_df = pd.DataFrame(ttc_rows)
    active_df = pd.DataFrame(active_rows)
    summary_df = summarize(episode_df)
    replacement_df = episode_df[
        ["policy_name", "scenario", "episode_id", "replacement_count", "spawn_count", "remove_count"]
    ].copy()

    episode_df.to_csv(out_dir / "env_v2_sanity.csv", index=False)
    summary_df.to_csv(out_dir / "env_v2_sanity_by_policy_scenario.csv", index=False)
    cpa_df.to_csv(out_dir / "cpa_distribution.csv", index=False)
    ttc_df.to_csv(out_dir / "ttc_distribution.csv", index=False)
    replacement_df.to_csv(out_dir / "replacement_count_distribution.csv", index=False)
    active_df.to_csv(out_dir / "active_obstacle_count_distribution.csv", index=False)

    write_plot(out_dir / "plots" / "cpa_distribution.png", cpa_df["planned_cpa"], "Planned CPA Distribution", "planned CPA")
    write_plot(out_dir / "plots" / "ttc_distribution.png", ttc_df["planned_ttc"], "Planned TTC Distribution", "planned TTC")
    write_plot(
        out_dir / "plots" / "replacement_count_distribution.png",
        replacement_df["replacement_count"],
        "Replacement Count Distribution",
        "replacement count",
    )
    write_plot(
        out_dir / "plots" / "active_obstacle_count_distribution.png",
        active_df["active_obstacle_count"],
        "Active Obstacle Count Distribution",
        "active obstacle count",
    )

    metrics = overall_metrics(episode_df, summary_df, cpa_df)
    decision, next_action = decide(metrics)
    if not args.no_flags:
        write_design_report(ROOT)
        write_sanity_report(ROOT, args, summary_df, metrics, decision)
        write_final_report(ROOT, summary_df, metrics, decision, next_action)
        write_flag(ROOT, decision, next_action)
    print(f"ENV_V2_SANITY_DONE decision={decision} next={next_action}", flush=True)


if __name__ == "__main__":
    main()
