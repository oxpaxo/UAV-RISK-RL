from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_env import DynamicObstacleEnv
from policies.obstacle_set_extractor import ObstacleSetExtractor


def str2bool(value: str) -> bool:
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
    parser.add_argument("--agg", choices=["risk", "attention", "mean"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_rbar", type=str2bool, default=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--r_ref", type=float, default=1.0)
    parser.add_argument("--rbar_floor", type=float, default=0.0)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--lambda_ewma", type=float, default=0.1)
    parser.add_argument("--r_gate", type=float, default=5.0)
    parser.add_argument("--scenario", type=str, default="eval_sudden_turn")
    return parser.parse_args()


def policy_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    use_rbar = args.use_rbar if args.agg == "risk" else False
    return dict(
        features_extractor_class=ObstacleSetExtractor,
        features_extractor_kwargs=dict(
            agg_mode=args.agg,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            r_ref=args.r_ref,
            use_rbar=use_rbar,
            rbar_floor=args.rbar_floor,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def theoretical_risk_weights(risk_values: np.ndarray, beta: float) -> np.ndarray:
    raw = np.clip(risk_values, 0.0, None) ** beta
    denom = float(np.sum(raw))
    if denom < 1e-8:
        return np.full_like(risk_values, 1.0 / max(len(risk_values), 1))
    return raw / denom


def risk_rank(risk_values: np.ndarray, obstacle_id: int) -> float:
    if obstacle_id < 0 or obstacle_id >= len(risk_values):
        return np.nan
    order = np.argsort(-risk_values)
    return float(np.where(order == obstacle_id)[0][0] + 1)


def compute_deviation_lateral(action: np.ndarray, goal_position: np.ndarray, uav_position: np.ndarray, v_uav_max: float) -> float:
    goal_dir = normalize(goal_position - uav_position)
    v_des = np.asarray(action, dtype=np.float32) * v_uav_max
    v_parallel = np.dot(v_des, goal_dir) * goal_dir
    v_lateral = v_des - v_parallel
    return float(np.linalg.norm(v_lateral))


def first_rise_time(times: list[float], values: list[float], threshold: float, turn_time: float) -> float:
    for t, v in zip(times, values):
        if t >= turn_time and v >= threshold:
            return float(t - turn_time)
    return np.nan


def first_sigma_2x_time(times: list[float], values: list[float], baseline: float, turn_time: float) -> float:
    if np.isnan(baseline):
        return np.nan
    target = 2.0 * baseline
    for t, v in zip(times, values):
        if t >= turn_time and v >= target:
            return float(t - turn_time)
    return np.nan


def reaction_time_from_trace(rows: list[dict[str, Any]], turn_time: float) -> float:
    consecutive = 0
    for row in rows:
        if float(row["time"]) < turn_time:
            continue
        if float(row["deviation_lateral"]) > 0.3:
            consecutive += 1
            if consecutive >= 2:
                return float(row["time"] - turn_time)
        else:
            consecutive = 0
    return np.nan


def evaluate_episode(
    model: PPO,
    env: DynamicObstacleEnv,
    episode_seed: int,
    args: argparse.Namespace,
    episode_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, info = env.reset(seed=episode_seed)
    done = False
    episode_rows: list[dict[str, Any]] = []
    episode_reward = 0.0
    steps = 0

    turn_step = int(info["turn_step"])
    turn_time = turn_step * env.dt
    turn_step_row: dict[str, Any] | None = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated

        risk_values = np.asarray(info["risk_values"], dtype=np.float32)
        sigma_values = np.asarray(info["sigma_values"], dtype=np.float32)
        mu_values = np.asarray(info["mu_values"], dtype=np.float32)
        obstacle_positions = np.asarray(info["obstacle_positions"], dtype=np.float32)
        obstacle_velocities = np.asarray(info["obstacle_velocities"], dtype=np.float32)
        uav_position = np.asarray(info["uav_position"], dtype=np.float32)
        uav_velocity = np.asarray(info["uav_velocity"], dtype=np.float32)
        goal_position = np.asarray(info["goal_position"], dtype=np.float32)
        turning_obstacle_id = int(info["turning_obstacle_id"])

        R_max = float(np.max(risk_values))
        R_sum = float(np.sum(risk_values))
        R_bar = float(np.tanh(R_sum / args.r_ref))
        weights = theoretical_risk_weights(risk_values, beta=args.beta)

        if 0 <= turning_obstacle_id < len(risk_values):
            risk_turn = float(risk_values[turning_obstacle_id])
            sigma_turn = sigma_values[turning_obstacle_id]
            sigma_turn_trace = float(np.sum(sigma_turn))
            mu_turn = mu_values[turning_obstacle_id]
            dist_turn = float(np.linalg.norm(obstacle_positions[turning_obstacle_id] - uav_position))
            rel_speed_turn = float(np.linalg.norm(obstacle_velocities[turning_obstacle_id] - uav_velocity))
            w_risk_turn = float(weights[turning_obstacle_id])
            risk_rank_turn = risk_rank(risk_values, turning_obstacle_id)
        else:
            risk_turn = np.nan
            sigma_turn = np.full(3, np.nan, dtype=np.float32)
            sigma_turn_trace = np.nan
            mu_turn = np.full(3, np.nan, dtype=np.float32)
            dist_turn = np.nan
            rel_speed_turn = np.nan
            w_risk_turn = np.nan
            risk_rank_turn = np.nan

        action = np.asarray(action, dtype=np.float32)
        v_des = action * env.v_uav_max
        row = {
            "method": "risk_full_rbar" if args.agg == "risk" and args.use_rbar else "attention_full" if args.agg == "attention" else args.agg,
            "seed": args.seed,
            "episode": episode_id,
            "episode_seed": episode_seed,
            "step": int(info["step"]),
            "time": float(info["time"]),
            "turn_step": turn_step,
            "has_turned": bool(info["has_turned"]),
            "turning_obstacle_id": turning_obstacle_id,
            "success": int(info["is_success"]),
            "collision": int(info["is_collision"]),
            "min_distance": float(info["min_distance"]),
            "episode_min_distance": float(info["episode_min_distance"]),
            "R_max": R_max,
            "R_sum": R_sum,
            "R_bar": R_bar,
            "risk_turn": risk_turn,
            "sigma_turn_x": float(sigma_turn[0]),
            "sigma_turn_y": float(sigma_turn[1]),
            "sigma_turn_z": float(sigma_turn[2]),
            "sigma_turn_trace": sigma_turn_trace,
            "mu_turn_x": float(mu_turn[0]),
            "mu_turn_y": float(mu_turn[1]),
            "mu_turn_z": float(mu_turn[2]),
            "dist_turn": dist_turn,
            "rel_speed_turn": rel_speed_turn,
            "w_risk_turn": w_risk_turn,
            "risk_rank_turn": risk_rank_turn,
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "action_z": float(action[2]),
            "v_des_norm": float(np.linalg.norm(v_des)),
            "deviation_lateral": compute_deviation_lateral(action, goal_position, uav_position, env.v_uav_max),
        }
        episode_rows.append(row)

        if int(info["step"]) == turn_step:
            turn_step_row = row

    times = [float(row["time"]) for row in episode_rows]
    risk_trace = [float(row["risk_turn"]) for row in episode_rows]
    sigma_trace = [float(row["sigma_turn_trace"]) for row in episode_rows]
    R_sum_trace = [float(row["R_sum"]) for row in episode_rows]
    R_bar_trace = [float(row["R_bar"]) for row in episode_rows]
    w_risk_trace = [float(row["w_risk_turn"]) for row in episode_rows]
    risk_rank_trace = [float(row["risk_rank_turn"]) for row in episode_rows]

    risk_at_turn = float(turn_step_row["risk_turn"]) if turn_step_row is not None else np.nan
    sigma_trace_at_turn = float(turn_step_row["sigma_turn_trace"]) if turn_step_row is not None else np.nan
    R_sum_at_turn = float(turn_step_row["R_sum"]) if turn_step_row is not None else np.nan
    R_bar_at_turn = float(turn_step_row["R_bar"]) if turn_step_row is not None else np.nan
    w_risk_at_turn = float(turn_step_row["w_risk_turn"]) if turn_step_row is not None else np.nan
    risk_rank_at_turn = float(turn_step_row["risk_rank_turn"]) if turn_step_row is not None else np.nan

    summary = {
        "method": episode_rows[-1]["method"] if episode_rows else args.agg,
        "seed": args.seed,
        "episode": episode_id,
        "success": int(episode_rows[-1]["success"]) if episode_rows else 0,
        "collision": int(episode_rows[-1]["collision"]) if episode_rows else 0,
        "episode_min_distance": float(episode_rows[-1]["episode_min_distance"]) if episode_rows else np.nan,
        "steps": steps,
        "turn_step": turn_step,
        "turning_obstacle_id": int(episode_rows[-1]["turning_obstacle_id"]) if episode_rows else -1,
        "risk_at_turn": risk_at_turn,
        "risk_max_after_turn": float(np.nanmax(risk_trace)) if risk_trace else np.nan,
        "risk_rise_time_0p3": first_rise_time(times, risk_trace, 0.3, turn_time),
        "risk_rise_time_0p5": first_rise_time(times, risk_trace, 0.5, turn_time),
        "risk_rise_time_0p7": first_rise_time(times, risk_trace, 0.7, turn_time),
        "sigma_trace_at_turn": sigma_trace_at_turn,
        "sigma_trace_max_after_turn": float(np.nanmax(sigma_trace)) if sigma_trace else np.nan,
        "sigma_rise_time_2x": first_sigma_2x_time(times, sigma_trace, sigma_trace_at_turn, turn_time),
        "R_sum_at_turn": R_sum_at_turn,
        "R_sum_max_after_turn": float(np.nanmax(R_sum_trace)) if R_sum_trace else np.nan,
        "R_bar_at_turn": R_bar_at_turn,
        "R_bar_max_after_turn": float(np.nanmax(R_bar_trace)) if R_bar_trace else np.nan,
        "w_risk_at_turn": w_risk_at_turn,
        "w_risk_max_after_turn": float(np.nanmax(w_risk_trace)) if w_risk_trace else np.nan,
        "risk_rank_at_turn": risk_rank_at_turn,
        "risk_rank_best_after_turn": float(np.nanmin(risk_rank_trace)) if risk_rank_trace else np.nan,
        "reaction_time": reaction_time_from_trace(episode_rows, turn_time),
    }
    return episode_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    traces_dir = Path(args.out_dir) / "traces"
    summary_dir = Path(args.out_dir) / "summary"
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    env = DynamicObstacleEnv(
        scenario=args.scenario,
        sigma_min=args.sigma_min,
        lambda_ewma=args.lambda_ewma,
        R_gate=args.r_gate,
    )
    model = PPO.load(
        args.model_path,
        device=args.device,
        custom_objects={"policy_kwargs": policy_kwargs_from_args(args)},
    )

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DIAG_START model={args.model_path} seed={args.seed} episodes={args.episodes}",
        flush=True,
    )

    episode_summaries: list[dict[str, Any]] = []
    eval_base_seed = 1000 + args.seed * 10000
    start_time = time.time()
    last_heartbeat = start_time

    for episode_id in range(args.episodes):
        episode_seed = eval_base_seed + episode_id
        episode_rows, summary = evaluate_episode(model, env, episode_seed, args, episode_id)
        trace_path = traces_dir / f"{summary['method']}_s{args.seed}_ep{episode_id}_trace.csv"
        write_csv(trace_path, episode_rows)
        episode_summaries.append(summary)

        now = time.time()
        if now - last_heartbeat >= 10.0 or episode_id == args.episodes - 1:
            done = episode_id + 1
            rate = done / max(now - start_time, 1e-6)
            eta = max(args.episodes - done, 0) / max(rate, 1e-6)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DIAG_HEARTBEAT "
                f"episodes={done}/{args.episodes} rate={rate:.2f} ep/s eta={eta/60.0:.2f} min",
                flush=True,
            )
            last_heartbeat = now

    summary_path = summary_dir / f"{episode_summaries[0]['method']}_s{args.seed}_episode_summary.csv"
    write_csv(summary_path, episode_summaries)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DIAG_END summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
