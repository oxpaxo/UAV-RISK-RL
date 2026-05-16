from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from policies.obstacle_set_extractor import ObstacleSetExtractor


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def json_array(value: Any) -> str:
    arr = np.asarray(value)
    return json.dumps(arr.tolist(), separators=(",", ":"))


def policy_kwargs(hidden_dim: int) -> dict[str, Any]:
    return dict(
        features_extractor_class=ObstacleSetExtractor,
        features_extractor_kwargs=dict(
            agg_mode="attention",
            hidden_dim=hidden_dim,
            beta=1.0,
            r_ref=1.0,
            use_rbar=False,
            rbar_floor=0.0,
            use_risk_bias=False,
            lambda_bias=0.0,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )


def attention_snapshot(model: PPO, max_obs: int) -> tuple[np.ndarray, float]:
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    weights = getattr(extractor, "latest_attention_weights", None)
    entropy = getattr(extractor, "latest_attention_entropy", None)
    if weights is None:
        arr = np.full(max_obs, np.nan, dtype=np.float32)
    else:
        arr = weights.detach().cpu().numpy()
        if arr.ndim >= 2:
            arr = arr[0]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] < max_obs:
            arr = np.pad(arr, (0, max_obs - arr.shape[0]), constant_values=np.nan)
    if entropy is None:
        entropy_value = float("nan")
    else:
        entropy_arr = entropy.detach().cpu().numpy()
        entropy_value = float(np.ravel(entropy_arr)[0]) if entropy_arr.size else float("nan")
    return arr[:max_obs], entropy_value


def lateral_deviation(action: np.ndarray, goal: np.ndarray, uav: np.ndarray, v_uav_max: float) -> float:
    goal_vec = goal - uav
    goal_vec[2] = 0.0
    goal_dir = normalize(goal_vec)
    v_des = np.asarray(action, dtype=np.float32) * v_uav_max
    v_des[2] = 0.0
    parallel = np.dot(v_des, goal_dir) * goal_dir
    return float(np.linalg.norm(v_des - parallel))


def threat_geometry(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    threat_index = int(info.get("threat_obstacle_index", -1))
    if threat_index < 0 or threat_index >= len(positions):
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), -1
    return positions[threat_index], velocities[threat_index], threat_index


def evaluate_episode(
    model: PPO,
    env: DynamicObstacleFlowEnv,
    args: argparse.Namespace,
    scenario: str,
    episode_id: int,
    episode_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, info = env.reset(seed=episode_seed)
    done = False
    steps = 0
    episode_reward = 0.0
    max_episode_time = env.max_steps * env.dt
    threat_window_enter_time = np.nan
    reaction_time_nan_style = np.nan
    reaction_hits = 0
    min_distance_after_threat = np.inf
    min_distance_values: list[float] = []
    distance_warning_values: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    last_info = info

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        episode_reward += float(reward)
        last_info = info

        time_now = float(info["time"])
        uav = np.asarray(info["uav_position"], dtype=np.float32)
        goal = np.asarray(info["goal_position"], dtype=np.float32)
        uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
        threat_pos, threat_vel, threat_index = threat_geometry(info)
        min_distance = float(info["min_distance"])
        min_distance_values.append(min_distance)
        distance_warning_cost = float(info.get("distance_warning_cost", max(0.0, 2.0 - min_distance) ** 2))
        distance_warning_values.append(distance_warning_cost)

        ttc_remaining = float(info.get("planned_ttc_remaining_to_threat", np.nan))
        threat_active = bool(np.isfinite(ttc_remaining) and 0.0 <= ttc_remaining <= args.threat_window_seconds)
        if threat_active:
            if np.isnan(threat_window_enter_time):
                threat_window_enter_time = time_now
            min_distance_after_threat = min(min_distance_after_threat, min_distance)
            deviation = lateral_deviation(action, goal, uav, env.v_uav_max)
            rel = threat_pos - uav
            rel[2] = 0.0
            away = normalize(-rel)
            lateral_away_speed = float(np.dot(uav_vel, away))
            reaction_flag = bool(deviation > args.reaction_lateral_threshold or lateral_away_speed > args.away_velocity_threshold)
            if reaction_flag:
                reaction_hits += 1
                if reaction_hits >= args.reaction_consecutive_steps and np.isnan(reaction_time_nan_style):
                    reaction_time_nan_style = max(time_now - threat_window_enter_time, 0.0)
            else:
                reaction_hits = 0
        else:
            deviation = lateral_deviation(action, goal, uav, env.v_uav_max)
            rel = threat_pos - uav
            rel[2] = 0.0
            away = normalize(-rel)
            lateral_away_speed = float(np.dot(uav_vel, away))
            reaction_flag = False

        if args.save_trace and scenario in set(args.trace_scenarios.split(",")):
            weights, entropy = attention_snapshot(model, env.max_obs)
            threat_weight = float(weights[threat_index]) if 0 <= threat_index < len(weights) else float("nan")
            rank = float("nan")
            if 0 <= threat_index < len(weights):
                order = np.argsort(-np.nan_to_num(weights, nan=-np.inf))
                match = np.where(order == threat_index)[0]
                if match.size:
                    rank = float(match[0] + 1)
            trace_rows.append(
                {
                    "checkpoint_step": int(args.global_step),
                    "scenario": scenario,
                    "episode_id": int(episode_id),
                    "step": int(info["step"]),
                    "time": time_now,
                    "uav_pos": json_array(uav),
                    "uav_vel": json_array(uav_vel),
                    "action": json_array(action),
                    "lateral_deviation": float(deviation),
                    "away_from_threat_velocity": float(lateral_away_speed),
                    "goal_directed_velocity": float(np.dot(uav_vel, normalize(goal - uav))),
                    "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
                    "threat_obstacle_index": int(threat_index),
                    "threat_class": str(info.get("threat_class", "none")),
                    "planned_cpa": float(info.get("planned_cpa_to_threat", np.nan)),
                    "planned_ttc": float(info.get("planned_ttc_to_threat", np.nan)),
                    "planned_ttc_remaining": ttc_remaining,
                    "min_distance": min_distance,
                    "min_distance_after_threat": float(min_distance_after_threat) if np.isfinite(min_distance_after_threat) else np.nan,
                    "attention_weights": json_array(weights),
                    "attention_entropy": entropy,
                    "threat_obstacle_attention_weight": threat_weight,
                    "threat_obstacle_attention_rank": rank,
                    "no_response_flag": int(threat_active and np.isnan(reaction_time_nan_style)),
                    "reaction_flag": int(reaction_flag),
                }
            )

    if np.isinf(min_distance_after_threat):
        min_distance_after_threat = float(last_info["episode_min_distance"])
    no_response = int(not np.isnan(threat_window_enter_time) and np.isnan(reaction_time_nan_style))
    if np.isnan(threat_window_enter_time):
        reaction_time_eval_style = np.nan
    elif np.isnan(reaction_time_nan_style):
        reaction_time_eval_style = max_episode_time - float(threat_window_enter_time)
    else:
        reaction_time_eval_style = float(reaction_time_nan_style)

    row = {
        "method": "attention_full",
        "seed": int(args.seed),
        "checkpoint_step": int(args.global_step),
        "scenario": scenario,
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "success": int(bool(last_info["is_success"])),
        "collision": int(bool(last_info["is_collision"])),
        "near_miss": int(float(last_info["episode_min_distance"]) < args.near_miss_distance),
        "mean_min_distance": float(np.mean(min_distance_values)) if min_distance_values else np.nan,
        "episode_min_distance": float(last_info["episode_min_distance"]),
        "min_distance_after_threat": float(min_distance_after_threat),
        "reaction_time_eval_style": float(reaction_time_eval_style) if np.isfinite(reaction_time_eval_style) else np.nan,
        "reaction_time_nan_style": float(reaction_time_nan_style) if np.isfinite(reaction_time_nan_style) else np.nan,
        "conditional_reaction_time": float(reaction_time_nan_style) if np.isfinite(reaction_time_nan_style) else np.nan,
        "nan_reaction": int(np.isnan(reaction_time_nan_style)),
        "no_response": int(no_response),
        "threat_window_enter_time": float(threat_window_enter_time) if np.isfinite(threat_window_enter_time) else np.nan,
        "mean_time": float(steps * env.dt),
        "episode_length": int(steps),
        "progress": float(last_info["progress"]),
        "planned_cpa": float(last_info.get("planned_cpa_to_threat", np.nan)),
        "planned_ttc": float(last_info.get("planned_ttc_to_threat", np.nan)),
        "threat_class": str(last_info.get("threat_class", "none")),
        "threat_motion_mode": str(last_info.get("threat_motion_mode", "none")),
        "threat_valid_rate": float(last_info.get("threat_valid_rate", np.nan)),
        "replacement_count": int(last_info.get("replacement_count", 0)),
        "distance_warning_cost_nonzero_rate": float(np.mean(np.asarray(distance_warning_values) > 0.0)) if distance_warning_values else np.nan,
        "distance_warning_cost_mean": float(np.mean(distance_warning_values)) if distance_warning_values else np.nan,
        "episode_reward": float(episode_reward),
        "nan_or_crash": 0,
    }
    return row, trace_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--global_step", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--save_trace", type=str2bool, default=False)
    parser.add_argument("--trace_csv", type=str, default="")
    parser.add_argument("--trace_scenarios", type=str, default="eval_flow_id,eval_flow_high_threat,eval_flow_mixed_ood,eval_flow_sudden_threat")
    parser.add_argument("--heartbeat_seconds", type=float, default=20.0)
    parser.add_argument("--near_miss_distance", type=float, default=1.5)
    parser.add_argument("--threat_window_seconds", type=float, default=4.0)
    parser.add_argument("--reaction_lateral_threshold", type=float, default=0.30)
    parser.add_argument("--away_velocity_threshold", type=float, default=0.15)
    parser.add_argument("--reaction_consecutive_steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = DynamicObstacleFlowEnv(scenario=args.scenario)
    model = PPO.load(
        args.model_path,
        device=args.device,
        custom_objects={"policy_kwargs": policy_kwargs(args.hidden_dim)},
    )
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ENV_V2_EVAL_START "
        f"model={args.model_path} scenario={args.scenario} step={args.global_step} episodes={args.episodes}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    start = time.time()
    last = start
    for episode_id in range(args.episodes):
        seed = args.eval_seed + args.seed * 10000 + episode_id
        row, trace_rows = evaluate_episode(model, env, args, args.scenario, episode_id, seed)
        rows.append(row)
        traces.extend(trace_rows)
        now = time.time()
        if now - last >= args.heartbeat_seconds or episode_id == args.episodes - 1:
            done = episode_id + 1
            rate = done / max(now - start, 1e-6)
            eta = (args.episodes - done) / max(rate, 1e-6)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ENV_V2_EVAL_HEARTBEAT "
                f"episodes={done}/{args.episodes} rate={rate:.2f} ep/s eta={eta/60.0:.2f} min",
                flush=True,
            )
            last = now
    write_csv(Path(args.out_csv), rows)
    if args.save_trace and args.trace_csv:
        write_csv(Path(args.trace_csv), traces)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ENV_V2_EVAL_END out_csv={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
