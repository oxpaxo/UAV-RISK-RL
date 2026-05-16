from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.dynamic_obstacle_env import DynamicObstacleEnv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--profile_mode", type=str, default="full_12")
    parser.add_argument("--agg", choices=["risk", "attention", "mean"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--use_rbar", type=str2bool, default=True)
    parser.add_argument("--rbar_floor", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--r_gate", type=float, default=5.0)
    parser.add_argument("--lambda_ewma", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--d_warning", type=float, default=1.0)
    parser.add_argument("--near_miss_distance", type=float, default=1.0)
    parser.add_argument("--use_risk_bias", type=str2bool, default=False)
    parser.add_argument("--lambda_bias", type=float, default=0.2)
    parser.add_argument("--save_trace", type=str2bool, default=False)
    parser.add_argument("--trace_dir", type=str, default="")
    parser.add_argument("--global_step", type=int, default=-1)
    parser.add_argument("--heartbeat_seconds", type=float, default=15.0)
    return parser.parse_args()


def policy_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    use_rbar = args.use_rbar if args.agg == "risk" else False
    return dict(
        features_extractor_class=ObstacleSetExtractor,
        features_extractor_kwargs=dict(
            agg_mode=args.agg,
            hidden_dim=args.hidden_dim,
            beta=1.0,
            r_ref=1.0,
            use_rbar=use_rbar,
            rbar_floor=args.rbar_floor,
            use_risk_bias=args.use_risk_bias,
            lambda_bias=args.lambda_bias,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def lateral_deviation(
    action: np.ndarray,
    goal_pos: np.ndarray,
    uav_pos: np.ndarray,
    v_uav_max: float,
) -> float:
    goal_dir = normalize(goal_pos - uav_pos)
    v_des = np.asarray(action, dtype=np.float32) * v_uav_max
    v_parallel = np.dot(v_des, goal_dir) * goal_dir
    v_lateral = v_des - v_parallel
    return float(np.linalg.norm(v_lateral))


def reaction_flag_from_deviation(deviation: float, threshold: float = 0.3) -> bool:
    return bool(deviation > threshold)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def get_feature_extractor(model: PPO) -> Any:
    policy = getattr(model, "policy", None)
    extractor = getattr(policy, "features_extractor", None)
    return extractor


def tensor_to_first_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim >= 2:
        return np.asarray(arr[0])
    return np.asarray(arr)


def attention_snapshot(model: PPO, max_obs: int) -> tuple[np.ndarray, float]:
    extractor = get_feature_extractor(model)
    weights = tensor_to_first_numpy(getattr(extractor, "latest_attention_weights", None))
    entropy = tensor_to_first_numpy(getattr(extractor, "latest_attention_entropy", None))
    if weights is None:
        weights = np.full(max_obs, np.nan, dtype=np.float32)
    if weights.shape[0] < max_obs:
        weights = np.pad(weights, (0, max_obs - weights.shape[0]), constant_values=np.nan)
    if entropy is None or entropy.size == 0:
        entropy_value = float("nan")
    else:
        entropy_value = float(entropy[0])
    return weights.astype(np.float32), entropy_value


def write_trace(trace_rows: list[dict[str, Any]], trace_dir: str, method: str, global_step: int, scenario: str, episode_id: int) -> None:
    if not trace_rows:
        return
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_label = f"step{global_step}" if global_step >= 0 else "stepunknown"
    path = out_dir / f"{method}_{step_label}_{scenario}_ep{episode_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)


def json_array(value: np.ndarray | list[float]) -> str:
    arr = np.asarray(value, dtype=np.float32)
    return json.dumps(arr.tolist(), separators=(",", ":"))


def json_value(value: Any) -> str:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        return obj

    return json.dumps(convert(value), separators=(",", ":"))


def is_turn_reaction_scenario(scenario: str) -> bool:
    if scenario.endswith("_high_speed"):
        scenario = scenario[: -len("_high_speed")]
    elif scenario.endswith("_small_space"):
        scenario = scenario[: -len("_small_space")]
    return scenario in {
        "eval_sudden_turn",
        "mixed_uncertainty",
        "eval_mixed_v2",
        "eval_threat_validated_sudden",
    }


def evaluate_episode(
    model: PPO,
    env: DynamicObstacleEnv,
    episode_seed: int,
    scenario: str,
    args: argparse.Namespace,
    episode_id: int,
) -> dict[str, Any]:
    obs, info = env.reset(seed=episode_seed)
    done = False
    episode_reward = 0.0
    steps = 0

    turn_step = int(info.get("turn_step", env.turn_step))
    turn_time = turn_step * env.dt
    ever_turned = False
    turning_obstacle_id = -1
    reaction_time_nan_style = np.nan
    risk_rise_time = np.nan
    min_distance_after_turn = np.inf
    consecutive_reaction_hits = 0
    trace_rows: list[dict[str, Any]] = []

    distance_cost_values: list[float] = []
    risk_sum_values: list[float] = []
    risk_max_values: list[float] = []
    min_distance_values: list[float] = []
    base_reward_values: list[float] = []
    shaped_reward_values: list[float] = []
    applied_cost_values: list[float] = []
    scenario_valid_values: list[float] = []
    threat_valid_values: list[float] = []
    planned_threat_valid_values: list[float] = []
    realized_near_miss_values: list[float] = []

    method = args.method or ("risk_full_rbar" if args.agg == "risk" and args.use_rbar else "attention_full" if args.agg == "attention" else args.agg)
    max_episode_time = env.max_steps * env.dt

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated

        risk_values = np.asarray(info["risk_values"], dtype=np.float32)
        sigma_values = np.asarray(info["sigma_values"], dtype=np.float32)
        uav_position = np.asarray(info["uav_position"], dtype=np.float32)
        uav_velocity = np.asarray(info["uav_velocity"], dtype=np.float32)
        goal_position = np.asarray(info["goal_position"], dtype=np.float32)
        min_distance = float(info["min_distance"])
        distance_warning_cost = float(info.get("distance_warning_cost", max(0.0, args.d_warning - min_distance) ** 2))
        risk_sum = float(info.get("risk_sum", np.sum(risk_values)))
        risk_max = float(info.get("risk_max", np.max(risk_values) if risk_values.size else 0.0))
        turning_obstacle_id = int(info.get("turning_obstacle_id", -1))
        has_turned = bool(info.get("has_turned", False))
        deviation = lateral_deviation(action, goal_position, uav_position, env.v_uav_max)
        reaction_flag = False
        reaction_time_current = np.nan

        distance_cost_values.append(distance_warning_cost)
        risk_sum_values.append(risk_sum)
        risk_max_values.append(risk_max)
        min_distance_values.append(min_distance)
        base_reward_values.append(float(info.get("base_reward", reward)))
        shaped_reward_values.append(float(info.get("shaped_reward", reward)))
        applied_cost_values.append(float(info.get("applied_cost", 0.0)))

        scenario_valid_values.append(float(bool(info.get("scenario_valid", True))))
        threat_valid_values.append(float(bool(info.get("threat_valid", True))))
        planned_threat_valid_values.append(float(bool(info.get("planned_threat_valid", info.get("threat_valid", True)))))
        realized_near_miss_values.append(float(bool(info.get("realized_near_miss", False))))

        if is_turn_reaction_scenario(scenario):
            if has_turned:
                ever_turned = True
                min_distance_after_turn = min(min_distance_after_turn, min_distance)
                reaction_flag = reaction_flag_from_deviation(deviation)
                if reaction_flag:
                    consecutive_reaction_hits += 1
                    if consecutive_reaction_hits >= 2 and np.isnan(reaction_time_nan_style):
                        reaction_time_nan_style = max(float(info["time"]) - turn_time, 0.0)
                else:
                    consecutive_reaction_hits = 0

                if turning_obstacle_id >= 0 and np.isnan(risk_rise_time):
                    if turning_obstacle_id < risk_values.shape[0] and risk_values[turning_obstacle_id] > 0.5:
                        risk_rise_time = max(float(info["time"]) - turn_time, 0.0)
                if not np.isnan(reaction_time_nan_style):
                    reaction_time_current = reaction_time_nan_style

        weights, entropy = attention_snapshot(model, env.max_obs)
        turning_weight = np.nan
        turning_rank = np.nan
        turning_risk = np.nan
        if 0 <= turning_obstacle_id < len(weights):
            turning_weight = float(weights[turning_obstacle_id])
            order = np.argsort(-np.nan_to_num(weights, nan=-np.inf))
            match = np.where(order == turning_obstacle_id)[0]
            if match.size:
                turning_rank = float(match[0] + 1)
        if 0 <= turning_obstacle_id < len(risk_values):
            turning_risk = float(risk_values[turning_obstacle_id])

        if args.save_trace:
            trace_rows.append(
                {
                    "episode": episode_id,
                    "step": int(info["step"]),
                    "time": float(info["time"]),
                    "turn_time": float(turn_time),
                    "turn_step": int(turn_step),
                    "scenario": scenario,
                    "success": int(info["is_success"]),
                    "collision": int(info["is_collision"]),
                    "uav_position": json_array(uav_position),
                    "uav_velocity": json_array(uav_velocity),
                    "goal_position": json_array(goal_position),
                    "action": json_array(np.asarray(action, dtype=np.float32)),
                    "min_distance": min_distance,
                    "distance_warning_cost": distance_warning_cost,
                    "risk_sum": risk_sum,
                    "risk_max": risk_max,
                    "risk_values": json_array(risk_values),
                    "sigma_values": json_array(sigma_values.reshape(-1)),
                    "reaction_flag": int(reaction_flag),
                    "reaction_time_current": float(reaction_time_current) if not np.isnan(reaction_time_current) else np.nan,
                    "attention_weights": json_array(weights),
                    "attention_entropy": entropy,
                    "turning_obstacle_id": turning_obstacle_id,
                    "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
                    "obstacle_motion_modes": json_value(info.get("obstacle_motion_modes", [])),
                    "scenario_valid": int(bool(info.get("scenario_valid", True))),
                    "planned_threat_valid": int(bool(info.get("planned_threat_valid", info.get("threat_valid", True)))),
                    "threat_valid": int(bool(info.get("threat_valid", True))),
                    "realized_near_miss": int(bool(info.get("realized_near_miss", False))),
                    "predicted_cpa_to_nominal_path": float(info.get("predicted_cpa_to_nominal_path", np.nan)),
                    "invalid_reason": str(info.get("invalid_reason", "none")),
                    "planned_threat_target_point": json_value(info.get("planned_threat_target_point", [])),
                    "current_target_speed": json_value(info.get("current_target_speed", [])),
                    "next_resample_time": json_value(info.get("next_resample_time", [])),
                    "ar1_phi": json_value(info.get("ar1_phi", [])),
                    "ar1_sigma": json_value(info.get("ar1_sigma", [])),
                    "speed_clip_applied": json_value(info.get("speed_clip_applied", [])),
                    "turning_obstacle_attention_weight": turning_weight,
                    "turning_obstacle_attention_rank": turning_rank,
                    "turning_obstacle_risk": turning_risk,
                    "R_sum": risk_sum,
                    "R_max": risk_max,
                    "lateral_deviation": deviation,
                    "base_reward": float(info.get("base_reward", reward)),
                    "applied_cost": float(info.get("applied_cost", 0.0)),
                    "shaped_reward": float(info.get("shaped_reward", reward)),
                    "fallback_penalty_active": int(bool(info.get("fallback_penalty_active", False))),
                }
            )

    if is_turn_reaction_scenario(scenario):
        if not ever_turned:
            reaction_time_eval_style = np.nan
            reaction_time_nan_style = np.nan
            risk_rise_time = np.nan
        else:
            reaction_time_eval_style = (
                float(reaction_time_nan_style)
                if not np.isnan(reaction_time_nan_style)
                else float(max_episode_time - turn_time)
            )
            if np.isnan(risk_rise_time):
                risk_rise_time = float(max_episode_time - turn_time)
        if np.isinf(min_distance_after_turn):
            min_distance_after_turn = float(info["episode_min_distance"])
    else:
        reaction_time_eval_style = np.nan
        reaction_time_nan_style = np.nan
        risk_rise_time = np.nan
        min_distance_after_turn = np.nan

    if args.save_trace:
        write_trace(
            trace_rows,
            args.trace_dir,
            method=method,
            global_step=args.global_step,
            scenario=scenario,
            episode_id=episode_id,
        )

    no_response = int(is_turn_reaction_scenario(scenario) and ever_turned and np.isnan(reaction_time_nan_style))
    return {
        "method": method,
        "global_step": int(args.global_step) if args.global_step >= 0 else np.nan,
        "scenario": scenario,
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "success": int(info["is_success"]),
        "collision": int(info["is_collision"]),
        "episode_min_distance": float(info["episode_min_distance"]),
        "mean_min_distance": mean(min_distance_values),
        "near_miss": int(np.min(min_distance_values) < args.near_miss_distance) if min_distance_values else 0,
        "episode_reward": float(episode_reward),
        "steps": int(steps),
        "time_to_goal": float(steps * env.dt),
        "reaction_time": float(reaction_time_eval_style) if not np.isnan(reaction_time_eval_style) else np.nan,
        "reaction_time_eval_style": float(reaction_time_eval_style) if not np.isnan(reaction_time_eval_style) else np.nan,
        "reaction_time_nan_style": float(reaction_time_nan_style) if not np.isnan(reaction_time_nan_style) else np.nan,
        "no_response": int(no_response),
        "risk_rise_time": float(risk_rise_time) if not np.isnan(risk_rise_time) else np.nan,
        "min_distance_after_turn": float(min_distance_after_turn) if not np.isnan(min_distance_after_turn) else np.nan,
        "turn_time": float(turn_time),
        "turn_step": int(turn_step),
        "turning_obstacle_id": int(turning_obstacle_id),
        "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
        "obstacle_motion_modes": json_value(info.get("obstacle_motion_modes", [])),
        "scenario_valid": int(bool(info.get("scenario_valid", True))),
        "planned_threat_valid": int(bool(info.get("planned_threat_valid", info.get("threat_valid", True)))),
        "threat_valid": int(bool(info.get("threat_valid", True))),
        "realized_near_miss": int(bool(info.get("realized_near_miss", False))),
        "predicted_cpa_to_nominal_path": float(info.get("predicted_cpa_to_nominal_path", np.nan)),
        "invalid_reason": str(info.get("invalid_reason", "none")),
        "initial_min_distance": float(info.get("initial_min_distance", np.nan)),
        "scenario_valid_mean": mean(scenario_valid_values),
        "planned_threat_valid_mean": mean(planned_threat_valid_values),
        "threat_valid_mean": mean(threat_valid_values),
        "realized_near_miss_mean": mean(realized_near_miss_values),
        "distance_warning_cost_mean": mean(distance_cost_values),
        "distance_warning_cost_p50": percentile(distance_cost_values, 50),
        "distance_warning_cost_p90": percentile(distance_cost_values, 90),
        "distance_warning_cost_p95": percentile(distance_cost_values, 95),
        "distance_warning_cost_max": max(distance_cost_values) if distance_cost_values else np.nan,
        "risk_sum_mean": mean(risk_sum_values),
        "risk_sum_p50": percentile(risk_sum_values, 50),
        "risk_sum_p90": percentile(risk_sum_values, 90),
        "risk_sum_p95": percentile(risk_sum_values, 95),
        "risk_sum_max": max(risk_sum_values) if risk_sum_values else np.nan,
        "risk_max_mean": mean(risk_max_values),
        "risk_max_p50": percentile(risk_max_values, 50),
        "risk_max_p90": percentile(risk_max_values, 90),
        "risk_max_p95": percentile(risk_max_values, 95),
        "risk_max_max": max(risk_max_values) if risk_max_values else np.nan,
        "base_reward_mean": mean(base_reward_values),
        "applied_cost_mean": mean(applied_cost_values),
        "shaped_reward_mean": mean(shaped_reward_values),
    }


def summarize(rows: list[dict[str, Any]], scenario: str) -> dict[str, float]:
    def col(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float32)

    summary = {
        "success_rate": float(np.mean(col("success"))),
        "collision_rate": float(np.mean(col("collision"))),
        "mean_min_distance": float(np.mean(col("episode_min_distance"))),
        "near_miss_rate": float(np.mean(col("near_miss"))),
        "mean_time": float(np.mean(col("time_to_goal"))),
        "mean_episode_reward": float(np.mean(col("episode_reward"))),
        "distance_warning_cost_mean": float(np.mean(col("distance_warning_cost_mean"))),
        "distance_warning_cost_p50": float(np.mean(col("distance_warning_cost_p50"))),
        "distance_warning_cost_p90": float(np.mean(col("distance_warning_cost_p90"))),
        "distance_warning_cost_p95": float(np.mean(col("distance_warning_cost_p95"))),
        "distance_warning_cost_max": float(np.max(col("distance_warning_cost_max"))),
        "risk_sum_mean": float(np.mean(col("risk_sum_mean"))),
        "risk_sum_p50": float(np.mean(col("risk_sum_p50"))),
        "risk_sum_p90": float(np.mean(col("risk_sum_p90"))),
        "risk_sum_p95": float(np.mean(col("risk_sum_p95"))),
        "risk_sum_max": float(np.max(col("risk_sum_max"))),
        "risk_max_mean": float(np.mean(col("risk_max_mean"))),
        "risk_max_p50": float(np.mean(col("risk_max_p50"))),
        "risk_max_p90": float(np.mean(col("risk_max_p90"))),
        "risk_max_p95": float(np.mean(col("risk_max_p95"))),
        "risk_max_max": float(np.max(col("risk_max_max"))),
        "applied_cost_mean": float(np.mean(col("applied_cost_mean"))),
        "distance_warning_cost_nonzero_rate": float(np.mean(col("distance_warning_cost_max") > 0.0)),
        "scenario_valid_rate": float(np.mean(col("scenario_valid"))),
        "planned_threat_valid_rate": float(np.mean(col("planned_threat_valid"))),
        "threat_valid_rate": float(np.mean(col("threat_valid"))),
        "realized_near_miss_rate": float(np.mean(col("realized_near_miss"))),
        "predicted_cpa_to_nominal_path_mean": float(np.nanmean(col("predicted_cpa_to_nominal_path"))),
    }
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVAL_SUMMARY "
        f"scenario={scenario} success_rate={summary['success_rate']:.4f} "
        f"collision_rate={summary['collision_rate']:.4f} "
        f"mean_min_distance={summary['mean_min_distance']:.4f} "
        f"near_miss_rate={summary['near_miss_rate']:.4f} "
        f"mean_time={summary['mean_time']:.4f}",
        flush=True,
    )
    if is_turn_reaction_scenario(scenario):
        eval_style = col("reaction_time_eval_style")
        nan_style = col("reaction_time_nan_style")
        no_response = col("no_response")
        summary.update(
            {
                "mean_reaction_eval_style": float(np.nanmean(eval_style)),
                "mean_reaction_nan_style": float(np.nanmean(nan_style)) if not np.all(np.isnan(nan_style)) else np.nan,
                "nan_reaction_rate": float(np.isnan(nan_style).mean()),
                "no_response_count": float(np.sum(no_response)),
                "total_episodes": float(len(rows)),
                "reaction_time": float(np.nanmean(eval_style)),
                "mean_risk_rise_time": float(np.nanmean(col("risk_rise_time"))),
                "mean_min_distance_after_turn": float(np.nanmean(col("min_distance_after_turn"))),
            }
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVAL_REACTION "
            f"mean_reaction_eval_style={summary['mean_reaction_eval_style']:.4f} "
            f"mean_reaction_nan_style={summary['mean_reaction_nan_style']:.4f} "
            f"nan_reaction_rate={summary['nan_reaction_rate']:.4f} "
            f"no_response_count={int(summary['no_response_count'])}/{len(rows)}",
            flush=True,
        )
    return summary


def write_csv(rows: list[dict[str, Any]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary: dict[str, float], out_csv: str) -> None:
    summary_path = Path(out_csv).with_suffix(".summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVAL_START model={args.model_path} "
        f"method={args.method or 'none'} agg={args.agg} train_seed={args.seed} "
        f"eval_seed={args.eval_seed} scenario={args.scenario} episodes={args.episodes}",
        flush=True,
    )

    env = DynamicObstacleEnv(
        scenario=args.scenario,
        R_gate=args.r_gate,
        lambda_ewma=args.lambda_ewma,
        sigma_min=args.sigma_min,
        d_warning=args.d_warning,
    )
    model = PPO.load(
        args.model_path,
        device=args.device,
        custom_objects={
            "policy_kwargs": policy_kwargs_from_args(args),
        },
    )

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    last_heartbeat = start_time

    for episode_id in range(args.episodes):
        episode_seed = args.eval_seed + args.seed * 10000 + episode_id
        row = evaluate_episode(model, env, episode_seed=episode_seed, scenario=args.scenario, args=args, episode_id=episode_id)
        rows.append(row)

        now = time.time()
        if now - last_heartbeat >= args.heartbeat_seconds or episode_id == args.episodes - 1:
            done = episode_id + 1
            percent = 100.0 * done / max(args.episodes, 1)
            elapsed = now - start_time
            rate = done / max(elapsed, 1e-6)
            eta_seconds = max(args.episodes - done, 0) / max(rate, 1e-6)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVAL_HEARTBEAT "
                f"episodes={done}/{args.episodes} percent={percent:6.2f}% "
                f"rate={rate:6.2f} ep/s eta={eta_seconds/60.0:6.2f} min",
                flush=True,
            )
            last_heartbeat = now

    write_csv(rows, args.out_csv)
    summary = summarize(rows, args.scenario)
    write_summary_csv(summary, args.out_csv)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVAL_END out_csv={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
