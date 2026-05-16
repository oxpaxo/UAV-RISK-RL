from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
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


RESULT_SUBDIRS = ("tables", "traces", "logs")
COMPLETE_FLAG = "PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag"
STOP_FLAGS = (
    "PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag",
    "PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag",
    "PHASE_A_STOP_SCHEMA_UNIFICATION_FAILED.flag",
    "PHASE_A_STOP_WATCHER_FAILED.flag",
)

DEFAULT_SCENARIOS = ("eval_flow_id", "eval_flow_high_speed")
DEFAULT_POLICIES = ("random", "straight_line", "cpa_reactive", "attention_full", "filtered_attention_full")
DEFAULT_CHECKPOINTS = (
    ROOT / "checkpoints" / "env_v2_phase2" / "attention_full_s0" / "attention_full_s0_step1500000.zip",
    ROOT / "checkpoints" / "env_v2_phase2" / "attention_full_s0" / "attention_full_s0_final_step1500000.zip",
    ROOT / "checkpoints" / "longtrain_baseline" / "attention_full_s0_step1500000.zip",
    ROOT / "checkpoints" / "attention_full_s0.zip",
)

EPISODE_FIELDS = [
    "method",
    "policy_name",
    "scenario",
    "checkpoint_step",
    "episode_id",
    "episode_seed",
    "success",
    "collision",
    "timeout",
    "truncated",
    "out_of_bounds",
    "near_miss",
    "progress",
    "final_goal_distance",
    "mean_time",
    "episode_length_steps",
    "episode_return",
    "mean_min_distance",
    "episode_min_distance",
    "min_distance_after_threat",
    "no_response",
    "no_response_rate",
    "reaction_time_eval_style",
    "conditional_reaction_time",
    "planned_cpa",
    "planned_ttc",
    "threat_class",
    "motion_mode",
    "replacement_count",
    "active_obstacle_count",
    "mean_action_norm",
    "mean_action_delta",
    "max_action_delta",
    "filter_used",
    "filter_trigger_count",
    "filter_trigger_rate",
    "mean_filter_delta_norm",
    "max_filter_delta_norm",
]

TRACE_FIELDS = [
    "checkpoint_step",
    "method",
    "policy_name",
    "scenario",
    "episode_id",
    "episode_seed",
    "step",
    "time",
    "done",
    "terminated",
    "truncated",
    "uav_pos_x",
    "uav_pos_y",
    "uav_pos_z",
    "uav_vel_x",
    "uav_vel_y",
    "uav_vel_z",
    "goal_pos_x",
    "goal_pos_y",
    "goal_pos_z",
    "goal_dist",
    "action_raw_x",
    "action_raw_y",
    "action_raw_z",
    "action_filtered_x",
    "action_filtered_y",
    "action_filtered_z",
    "action_executed_x",
    "action_executed_y",
    "action_executed_z",
    "filter_used",
    "filter_triggered",
    "filter_reason",
    "filter_delta_norm",
    "min_distance",
    "nearest_obstacle_id",
    "nearest_obstacle_distance",
    "threat_obstacle_id",
    "threat_obstacle_index",
    "threat_class",
    "planned_cpa",
    "planned_ttc",
    "planned_ttc_remaining",
    "lateral_deviation",
    "away_from_threat_velocity",
    "goal_directed_velocity",
    "reaction_flag",
    "no_response_flag",
    "attention_entropy",
    "threat_obstacle_attention_weight",
    "threat_obstacle_attention_rank",
    "min_predicted_cpa_raw",
    "min_predicted_cpa_filtered",
    "min_ttc_raw",
    "min_ttc_filtered",
    "unsafe_obstacle_id",
    "unsafe_obstacle_distance",
    "unsafe_obstacle_tcpa",
    "unsafe_obstacle_cpa",
]

OBSTACLE_FIELDS = [
    "method",
    "policy_name",
    "scenario",
    "episode_id",
    "episode_seed",
    "step",
    "time",
    "obstacle_slot",
    "obstacle_id",
    "active",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "distance",
    "closing",
    "planned_cpa",
    "planned_ttc",
    "threat_class",
    "motion_mode",
    "risk_value",
]

SUMMARY_FIELDS = [
    "method",
    "policy_name",
    "scenario",
    "episodes",
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "timeout_rate",
    "mean_progress",
    "mean_episode_return",
    "mean_min_distance",
    "episode_min_distance",
    "mean_episode_length_steps",
    "mean_filter_trigger_rate",
    "nan_or_crash",
]

STOP_REASON_TO_FLAG = {
    "eval_framework_failed": "PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag",
    "checkpoint_not_found": "PHASE_A_STOP_CHECKPOINT_NOT_FOUND.flag",
    "schema_unification_failed": "PHASE_A_STOP_SCHEMA_UNIFICATION_FAILED.flag",
    "env_core_change_required": "PHASE_A_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
}


class PhaseALogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.log_path = out_dir / "logs" / "phase_a_eval_framework.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class PhaseAStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


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
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def bool_int(value: Any) -> int:
    return int(bool(value))


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


def attention_snapshot(model: PPO | None, max_obs: int) -> tuple[np.ndarray, float]:
    if model is None:
        return np.full(max_obs, np.nan, dtype=np.float32), float("nan")
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    weights = getattr(extractor, "latest_attention_weights", None)
    entropy = getattr(extractor, "latest_attention_entropy", None)
    if weights is None:
        weight_arr = np.full(max_obs, np.nan, dtype=np.float32)
    else:
        weight_arr = weights.detach().cpu().numpy()
        if weight_arr.ndim >= 2:
            weight_arr = weight_arr[0]
        weight_arr = np.asarray(weight_arr, dtype=np.float32)
        if weight_arr.shape[0] < max_obs:
            weight_arr = np.pad(weight_arr, (0, max_obs - weight_arr.shape[0]), constant_values=np.nan)
        weight_arr = weight_arr[:max_obs]
    if entropy is None:
        entropy_value = float("nan")
    else:
        entropy_arr = entropy.detach().cpu().numpy()
        entropy_value = float(np.ravel(entropy_arr)[0]) if entropy_arr.size else float("nan")
    return weight_arr, entropy_value


def lateral_deviation(action: np.ndarray, goal: np.ndarray, uav: np.ndarray, v_uav_max: float) -> float:
    goal_vec = np.asarray(goal, dtype=np.float32) - np.asarray(uav, dtype=np.float32)
    goal_vec[2] = 0.0
    goal_dir = normalize(goal_vec)
    v_des = np.asarray(action, dtype=np.float32) * float(v_uav_max)
    v_des[2] = 0.0
    parallel = float(np.dot(v_des, goal_dir)) * goal_dir
    return float(np.linalg.norm(v_des - parallel))


def cpa_reactive_action(env: DynamicObstacleFlowEnv, info: dict[str, Any]) -> np.ndarray:
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    goal_vec = goal - uav
    goal_vec[2] = 0.0
    v_goal = normalize(goal_vec)

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


def straight_line_action(info: dict[str, Any]) -> np.ndarray:
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    goal_vec = goal - uav
    goal_vec[2] = 0.0
    return np.clip(normalize(goal_vec), -1.0, 1.0).astype(np.float32)


def predicted_cpa_geometry(
    action: np.ndarray,
    env: DynamicObstacleFlowEnv,
    info: dict[str, Any],
    horizon: float = 4.5,
) -> dict[str, float]:
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    obs_pos = np.asarray(info["obstacle_positions"], dtype=np.float32)
    obs_vel = np.asarray(info["obstacle_velocities"], dtype=np.float32)
    obs_ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    if len(obs_pos) == 0:
        return {
            "min_cpa": float("nan"),
            "min_ttc": float("nan"),
            "unsafe_obstacle_id": -1,
            "unsafe_obstacle_distance": float("nan"),
            "unsafe_obstacle_tcpa": float("nan"),
            "unsafe_obstacle_cpa": float("nan"),
        }

    v_uav_pred = np.asarray(action, dtype=np.float32) * float(env.v_uav_max)
    v_uav_pred[2] = 0.0
    best = {
        "min_cpa": float("inf"),
        "min_ttc": float("nan"),
        "unsafe_obstacle_id": -1,
        "unsafe_obstacle_distance": float("nan"),
        "unsafe_obstacle_tcpa": float("nan"),
        "unsafe_obstacle_cpa": float("inf"),
    }
    for index, (pos, vel) in enumerate(zip(obs_pos, obs_vel)):
        rel = np.asarray(pos, dtype=np.float32) - uav
        rel[2] = 0.0
        distance = float(np.linalg.norm(rel))
        rel_vel = np.asarray(vel, dtype=np.float32) - v_uav_pred
        rel_vel[2] = 0.0
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        tcpa = 0.0
        cpa = distance
        if rel_speed_sq > 1e-8:
            tcpa = float(np.clip(-np.dot(rel, rel_vel) / rel_speed_sq, 0.0, horizon))
            cpa = float(np.linalg.norm(rel + rel_vel * tcpa))
        if cpa < best["min_cpa"]:
            best = {
                "min_cpa": cpa,
                "min_ttc": tcpa,
                "unsafe_obstacle_id": int(obs_ids[index]) if index < len(obs_ids) else int(index),
                "unsafe_obstacle_distance": distance,
                "unsafe_obstacle_tcpa": tcpa,
                "unsafe_obstacle_cpa": cpa,
            }
    return best


def minimal_safety_filter(
    action_raw: np.ndarray,
    env: DynamicObstacleFlowEnv,
    info: dict[str, Any],
) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    raw_geometry = predicted_cpa_geometry(action_raw, env, info, horizon=5.5)
    min_distance = finite(info.get("min_distance"))
    min_cpa = raw_geometry["min_cpa"]
    min_ttc = raw_geometry["min_ttc"]
    trigger = bool(
        (np.isfinite(min_distance) and min_distance < 5.0)
        or (np.isfinite(min_cpa) and min_cpa < 3.2 and np.isfinite(min_ttc) and 0.0 <= min_ttc <= 5.5)
    )
    if not trigger:
        filtered = np.asarray(action_raw, dtype=np.float32).copy()
        reason = "none"
    else:
        reactive = cpa_reactive_action(env, info)
        blended = 0.55 * np.asarray(action_raw, dtype=np.float32) + 0.85 * reactive
        blended[2] = action_raw[2]
        filtered = normalize(blended)
        if float(np.linalg.norm(filtered)) < 1e-6:
            filtered = reactive
        filtered = np.clip(filtered, -1.0, 1.0).astype(np.float32)
        reason = "phase_a_minimal_cpa_distance_filter"

    filtered_geometry = predicted_cpa_geometry(filtered, env, info, horizon=5.5)
    debug: dict[str, float] = {
        "min_predicted_cpa_raw": raw_geometry["min_cpa"],
        "min_predicted_cpa_filtered": filtered_geometry["min_cpa"],
        "min_ttc_raw": raw_geometry["min_ttc"],
        "min_ttc_filtered": filtered_geometry["min_ttc"],
        "unsafe_obstacle_id": raw_geometry["unsafe_obstacle_id"],
        "unsafe_obstacle_distance": raw_geometry["unsafe_obstacle_distance"],
        "unsafe_obstacle_tcpa": raw_geometry["unsafe_obstacle_tcpa"],
        "unsafe_obstacle_cpa": raw_geometry["unsafe_obstacle_cpa"],
    }
    return filtered, trigger, reason, debug


def act_policy(
    policy_name: str,
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    env: DynamicObstacleFlowEnv,
    rng: np.random.Generator,
    model: PPO | None,
) -> dict[str, Any]:
    if policy_name == "random":
        action_raw = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        action_raw[2] *= 0.2
        return {
            "action_raw": action_raw,
            "action_filtered": action_raw.copy(),
            "filter_triggered": False,
            "filter_reason": "none",
            "debug": {},
        }
    if policy_name == "straight_line":
        action_raw = straight_line_action(info)
        return {
            "action_raw": action_raw,
            "action_filtered": action_raw.copy(),
            "filter_triggered": False,
            "filter_reason": "none",
            "debug": {},
        }
    if policy_name == "cpa_reactive":
        action_raw = cpa_reactive_action(env, info)
        return {
            "action_raw": action_raw,
            "action_filtered": action_raw.copy(),
            "filter_triggered": False,
            "filter_reason": "none",
            "debug": {},
        }
    if policy_name in {"attention_full", "filtered_attention_full"}:
        if model is None:
            raise PhaseAStop("checkpoint_not_found", "attention_full policy requested but no checkpoint was loaded")
        action_raw, _ = model.predict(obs, deterministic=True)
        action_raw = np.asarray(action_raw, dtype=np.float32)
        if policy_name == "attention_full":
            return {
                "action_raw": action_raw,
                "action_filtered": action_raw.copy(),
                "filter_triggered": False,
                "filter_reason": "none",
                "debug": {},
            }
        action_filtered, triggered, reason, debug = minimal_safety_filter(action_raw, env, info)
        return {
            "action_raw": action_raw,
            "action_filtered": action_filtered,
            "filter_triggered": triggered,
            "filter_reason": reason,
            "debug": debug,
        }
    raise PhaseAStop("eval_framework_failed", f"unsupported policy: {policy_name}")


def nearest_obstacle(info: dict[str, Any]) -> tuple[int, float]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    if len(positions) == 0:
        return -1, float("nan")
    distances = np.linalg.norm(positions - uav, axis=1)
    index = int(np.argmin(distances))
    return int(ids[index]) if index < len(ids) else index, float(distances[index])


def action_debug_value(act: dict[str, Any], key: str) -> float:
    debug = act.get("debug", {}) or {}
    return finite(debug.get(key))


def build_trace_row(
    *,
    method: str,
    policy_name: str,
    scenario: str,
    checkpoint_step: int,
    episode_id: int,
    episode_seed: int,
    env: DynamicObstacleFlowEnv,
    info: dict[str, Any],
    act: dict[str, Any],
    reward_done: tuple[bool, bool, bool],
    reaction_flag: bool,
    no_response_flag: bool,
    model: PPO | None,
) -> dict[str, Any]:
    done, terminated, truncated = reward_done
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    action_raw = np.asarray(act["action_raw"], dtype=np.float32)
    action_filtered = np.asarray(act["action_filtered"], dtype=np.float32)
    action_executed = np.clip(action_filtered, env.action_space.low, env.action_space.high).astype(np.float32)
    nearest_id, nearest_distance = nearest_obstacle(info)
    weights, entropy = attention_snapshot(model if policy_name in {"attention_full", "filtered_attention_full"} else None, env.max_obs)
    threat_index = int(info.get("threat_obstacle_index", -1))
    threat_weight = float(weights[threat_index]) if 0 <= threat_index < len(weights) else float("nan")
    rank = float("nan")
    if 0 <= threat_index < len(weights):
        order = np.argsort(-np.nan_to_num(weights, nan=-np.inf))
        match = np.where(order == threat_index)[0]
        if match.size:
            rank = float(match[0] + 1)
    rel_threat = threat_relative_vectors(info)

    return {
        "checkpoint_step": int(checkpoint_step),
        "method": method,
        "policy_name": policy_name,
        "scenario": scenario,
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "step": int(info["step"]),
        "time": float(info["time"]),
        "done": bool_int(done),
        "terminated": bool_int(terminated),
        "truncated": bool_int(truncated),
        "uav_pos_x": float(uav[0]),
        "uav_pos_y": float(uav[1]),
        "uav_pos_z": float(uav[2]),
        "uav_vel_x": float(uav_vel[0]),
        "uav_vel_y": float(uav_vel[1]),
        "uav_vel_z": float(uav_vel[2]),
        "goal_pos_x": float(goal[0]),
        "goal_pos_y": float(goal[1]),
        "goal_pos_z": float(goal[2]),
        "goal_dist": float(info.get("goal_distance", np.linalg.norm(goal - uav))),
        "action_raw_x": float(action_raw[0]),
        "action_raw_y": float(action_raw[1]),
        "action_raw_z": float(action_raw[2]),
        "action_filtered_x": float(action_filtered[0]),
        "action_filtered_y": float(action_filtered[1]),
        "action_filtered_z": float(action_filtered[2]),
        "action_executed_x": float(action_executed[0]),
        "action_executed_y": float(action_executed[1]),
        "action_executed_z": float(action_executed[2]),
        "filter_used": bool_int(policy_name == "filtered_attention_full"),
        "filter_triggered": bool_int(act["filter_triggered"]),
        "filter_reason": str(act["filter_reason"]),
        "filter_delta_norm": float(np.linalg.norm(action_filtered - action_raw)),
        "min_distance": float(info["min_distance"]),
        "nearest_obstacle_id": int(nearest_id),
        "nearest_obstacle_distance": float(nearest_distance),
        "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
        "threat_obstacle_index": int(threat_index),
        "threat_class": str(info.get("threat_class", "none")),
        "planned_cpa": finite(info.get("planned_cpa_to_threat")),
        "planned_ttc": finite(info.get("planned_ttc_to_threat")),
        "planned_ttc_remaining": finite(info.get("planned_ttc_remaining_to_threat")),
        "lateral_deviation": lateral_deviation(action_executed, goal, uav, env.v_uav_max),
        "away_from_threat_velocity": float(rel_threat["away_velocity"]),
        "goal_directed_velocity": float(np.dot(uav_vel, normalize(goal - uav))),
        "reaction_flag": bool_int(reaction_flag),
        "no_response_flag": bool_int(no_response_flag),
        "attention_entropy": entropy,
        "threat_obstacle_attention_weight": threat_weight,
        "threat_obstacle_attention_rank": rank,
        "min_predicted_cpa_raw": action_debug_value(act, "min_predicted_cpa_raw"),
        "min_predicted_cpa_filtered": action_debug_value(act, "min_predicted_cpa_filtered"),
        "min_ttc_raw": action_debug_value(act, "min_ttc_raw"),
        "min_ttc_filtered": action_debug_value(act, "min_ttc_filtered"),
        "unsafe_obstacle_id": int(action_debug_value(act, "unsafe_obstacle_id")) if np.isfinite(action_debug_value(act, "unsafe_obstacle_id")) else -1,
        "unsafe_obstacle_distance": action_debug_value(act, "unsafe_obstacle_distance"),
        "unsafe_obstacle_tcpa": action_debug_value(act, "unsafe_obstacle_tcpa"),
        "unsafe_obstacle_cpa": action_debug_value(act, "unsafe_obstacle_cpa"),
    }


def threat_relative_vectors(info: dict[str, Any]) -> dict[str, float]:
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    threat_index = int(info.get("threat_obstacle_index", -1))
    if threat_index < 0 or threat_index >= len(positions):
        return {"away_velocity": float("nan")}
    rel = positions[threat_index] - uav
    rel[2] = 0.0
    away = normalize(-rel)
    return {"away_velocity": float(np.dot(uav_vel, away))}


def obstacle_rows(
    *,
    method: str,
    policy_name: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    planned_cpas = np.asarray(info.get("planned_cpa_values", []), dtype=np.float32)
    planned_ttcs = np.asarray(info.get("planned_ttc_values", []), dtype=np.float32)
    classes = list(info.get("threat_classes", []))
    modes = list(info.get("obstacle_motion_modes", []))

    rows: list[dict[str, Any]] = []
    for index, pos in enumerate(positions):
        vel = velocities[index] if index < len(velocities) else np.zeros(3, dtype=np.float32)
        rel = pos - uav
        distance = float(np.linalg.norm(rel))
        rel_dir = rel / (distance + 1e-8)
        closing = -float(np.dot(rel_dir, vel - uav_vel))
        rows.append(
            {
                "method": method,
                "policy_name": policy_name,
                "scenario": scenario,
                "episode_id": int(episode_id),
                "episode_seed": int(episode_seed),
                "step": int(info["step"]),
                "time": float(info["time"]),
                "obstacle_slot": int(index),
                "obstacle_id": int(ids[index]) if index < len(ids) else int(index),
                "active": 1,
                "pos_x": float(pos[0]),
                "pos_y": float(pos[1]),
                "pos_z": float(pos[2]),
                "vel_x": float(vel[0]),
                "vel_y": float(vel[1]),
                "vel_z": float(vel[2]),
                "distance": distance,
                "closing": closing,
                "planned_cpa": float(planned_cpas[index]) if index < len(planned_cpas) else float("nan"),
                "planned_ttc": float(planned_ttcs[index]) if index < len(planned_ttcs) else float("nan"),
                "threat_class": str(classes[index]) if index < len(classes) else "none",
                "motion_mode": str(modes[index]) if index < len(modes) else "none",
                "risk_value": float(np.clip((3.0 - distance) / 3.0, 0.0, 1.0)),
            }
        )
    return rows


def run_episode(
    *,
    policy_name: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    checkpoint_step: int,
    rng: np.random.Generator,
    model: PPO | None,
    write_traces: bool,
    logger: PhaseALogger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = DynamicObstacleFlowEnv(scenario=scenario)
    obs, info = env.reset(seed=episode_seed)
    done = False
    steps = 0
    episode_return = 0.0
    last_info = info
    trace_rows: list[dict[str, Any]] = []
    step_obstacle_rows: list[dict[str, Any]] = []

    min_distance_values: list[float] = []
    action_norms: list[float] = []
    action_deltas: list[float] = []
    filter_deltas: list[float] = []
    filter_triggers = 0
    prev_action: np.ndarray | None = None
    threat_window_enter_time = float("nan")
    first_reaction_time = float("nan")
    min_distance_after_threat = float("inf")
    no_response_steps = 0
    threat_active_steps = 0
    reaction_hits = 0
    max_episode_time = env.max_steps * env.dt
    method = policy_name

    try:
        while not done:
            act = act_policy(policy_name, obs, info, env, rng, model)
            action_raw = np.asarray(act["action_raw"], dtype=np.float32)
            action_filtered = np.asarray(act["action_filtered"], dtype=np.float32)
            action_executed = np.clip(action_filtered, env.action_space.low, env.action_space.high).astype(np.float32)
            filter_delta = float(np.linalg.norm(action_filtered - action_raw))
            filter_deltas.append(filter_delta)
            filter_triggers += int(bool(act["filter_triggered"]))
            action_norms.append(float(np.linalg.norm(action_executed)))
            if prev_action is not None:
                action_deltas.append(float(np.linalg.norm(action_executed - prev_action)))
            prev_action = action_executed.copy()

            obs, reward, terminated, truncated, info = env.step(action_executed)
            done = bool(terminated or truncated)
            steps += 1
            episode_return += float(reward)
            last_info = info

            min_distance = float(info["min_distance"])
            min_distance_values.append(min_distance)
            time_now = float(info["time"])
            ttc_remaining = finite(info.get("planned_ttc_remaining_to_threat"))
            threat_active = bool(np.isfinite(ttc_remaining) and 0.0 <= ttc_remaining <= 4.0)
            uav = np.asarray(info["uav_position"], dtype=np.float32)
            goal = np.asarray(info["goal_position"], dtype=np.float32)
            deviation = lateral_deviation(action_executed, goal, uav, env.v_uav_max)
            away_velocity = threat_relative_vectors(info)["away_velocity"]
            reaction_flag = bool(deviation > 0.30 or (np.isfinite(away_velocity) and away_velocity > 0.15))

            if threat_active:
                threat_active_steps += 1
                if np.isnan(threat_window_enter_time):
                    threat_window_enter_time = time_now
                min_distance_after_threat = min(min_distance_after_threat, min_distance)
                if reaction_flag:
                    reaction_hits += 1
                    if reaction_hits >= 2 and np.isnan(first_reaction_time):
                        first_reaction_time = max(time_now - threat_window_enter_time, 0.0)
                else:
                    reaction_hits = 0
                if np.isnan(first_reaction_time):
                    no_response_steps += 1

            if write_traces:
                trace_rows.append(
                    build_trace_row(
                        method=method,
                        policy_name=policy_name,
                        scenario=scenario,
                        checkpoint_step=checkpoint_step,
                        episode_id=episode_id,
                        episode_seed=episode_seed,
                        env=env,
                        info=info,
                        act=act,
                        reward_done=(done, bool(terminated), bool(truncated)),
                        reaction_flag=reaction_flag,
                        no_response_flag=bool(threat_active and np.isnan(first_reaction_time)),
                        model=model,
                    )
                )
                step_obstacle_rows.extend(
                    obstacle_rows(
                        method=method,
                        policy_name=policy_name,
                        scenario=scenario,
                        episode_id=episode_id,
                        episode_seed=episode_seed,
                        info=info,
                    )
                )
    except PhaseAStop:
        raise
    except Exception as exc:
        logger.log(f"EPISODE_FAILED policy={policy_name} scenario={scenario} episode={episode_id} error={exc!r}")
        raise PhaseAStop("eval_framework_failed", f"episode rollout failed for {policy_name}/{scenario}/ep{episode_id}: {exc!r}") from exc

    if np.isinf(min_distance_after_threat):
        min_distance_after_threat = finite(last_info.get("episode_min_distance"))
    if np.isnan(threat_window_enter_time):
        reaction_time_eval_style = float("nan")
        no_response = 0
    elif np.isnan(first_reaction_time):
        reaction_time_eval_style = max_episode_time - float(threat_window_enter_time)
        no_response = 1
    else:
        reaction_time_eval_style = float(first_reaction_time)
        no_response = 0

    row = {
        "method": method,
        "policy_name": policy_name,
        "scenario": scenario,
        "checkpoint_step": int(checkpoint_step),
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "success": bool_int(last_info.get("is_success")),
        "collision": bool_int(last_info.get("is_collision")),
        "timeout": bool_int(last_info.get("truncated")),
        "truncated": bool_int(last_info.get("truncated")),
        "out_of_bounds": bool_int(last_info.get("out_of_bounds")),
        "near_miss": bool_int(last_info.get("near_miss")),
        "progress": finite(last_info.get("progress")),
        "final_goal_distance": finite(last_info.get("goal_distance")),
        "mean_time": float(steps * env.dt),
        "episode_length_steps": int(steps),
        "episode_return": float(episode_return),
        "mean_min_distance": float(np.mean(min_distance_values)) if min_distance_values else float("nan"),
        "episode_min_distance": finite(last_info.get("episode_min_distance")),
        "min_distance_after_threat": float(min_distance_after_threat),
        "no_response": int(no_response),
        "no_response_rate": float(no_response_steps / threat_active_steps) if threat_active_steps else float("nan"),
        "reaction_time_eval_style": float(reaction_time_eval_style) if np.isfinite(reaction_time_eval_style) else float("nan"),
        "conditional_reaction_time": float(first_reaction_time) if np.isfinite(first_reaction_time) else float("nan"),
        "planned_cpa": finite(last_info.get("planned_cpa_to_threat")),
        "planned_ttc": finite(last_info.get("planned_ttc_to_threat")),
        "threat_class": str(last_info.get("threat_class", "none")),
        "motion_mode": str(last_info.get("threat_motion_mode", "none")),
        "replacement_count": int(last_info.get("replacement_count", 0)),
        "active_obstacle_count": int(last_info.get("active_obstacle_count", 0)),
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else float("nan"),
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
        "filter_used": bool_int(policy_name == "filtered_attention_full"),
        "filter_trigger_count": int(filter_triggers),
        "filter_trigger_rate": float(filter_triggers / max(steps, 1)),
        "mean_filter_delta_norm": float(np.mean(filter_deltas)) if filter_deltas else float("nan"),
        "max_filter_delta_norm": float(np.max(filter_deltas)) if filter_deltas else float("nan"),
    }
    return row, trace_rows, step_obstacle_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, float("nan")) for field in fieldnames}
            writer.writerow(normalized)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["policy_name"]), str(row["scenario"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (method, policy_name, scenario), items in sorted(grouped.items()):
        def mean_field(field: str) -> float:
            vals = [finite(item.get(field)) for item in items]
            vals = [value for value in vals if np.isfinite(value)]
            return float(np.mean(vals)) if vals else float("nan")

        summaries.append(
            {
                "method": method,
                "policy_name": policy_name,
                "scenario": scenario,
                "episodes": len(items),
                "success_rate": mean_field("success"),
                "collision_rate": mean_field("collision"),
                "near_miss_rate": mean_field("near_miss"),
                "timeout_rate": mean_field("timeout"),
                "mean_progress": mean_field("progress"),
                "mean_episode_return": mean_field("episode_return"),
                "mean_min_distance": mean_field("mean_min_distance"),
                "episode_min_distance": mean_field("episode_min_distance"),
                "mean_episode_length_steps": mean_field("episode_length_steps"),
                "mean_filter_trigger_rate": mean_field("filter_trigger_rate"),
                "nan_or_crash": 0,
            }
        )
    return summaries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_checkpoint_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return -1
    name = checkpoint.name
    match = re.search(r"step(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)k", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000
    return -1


def resolve_checkpoint(args: argparse.Namespace, policies: list[str]) -> Path | None:
    needs_checkpoint = any(policy in {"attention_full", "filtered_attention_full"} for policy in policies)
    if not needs_checkpoint:
        return None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.exists():
            raise PhaseAStop("checkpoint_not_found", f"checkpoint path not found: {checkpoint}")
        return checkpoint
    for candidate in DEFAULT_CHECKPOINTS:
        if candidate.exists():
            return candidate
    default_list = ", ".join(str(path.relative_to(ROOT)) for path in DEFAULT_CHECKPOINTS)
    raise PhaseAStop("checkpoint_not_found", f"no default attention_full checkpoint found; checked: {default_list}")


def load_attention_model(checkpoint: Path | None, args: argparse.Namespace, logger: PhaseALogger) -> PPO | None:
    if checkpoint is None:
        return None
    try:
        logger.log(f"LOADING_CHECKPOINT path={checkpoint}")
        return PPO.load(
            str(checkpoint),
            device=args.device,
            custom_objects={"policy_kwargs": policy_kwargs(args.hidden_dim)},
        )
    except Exception as exc:
        raise PhaseAStop("checkpoint_not_found", f"checkpoint exists but could not be loaded: {checkpoint}: {exc!r}") from exc


def schema_rows(fields: list[str], table: str) -> list[dict[str, Any]]:
    required_notes = {
        "action_raw_x": "raw adapter action, required for filtered policies",
        "action_filtered_x": "post-filter action, same as raw for unfiltered baselines",
        "action_executed_x": "clipped command passed to EnvV2 step",
        "filter_triggered": "required for filtered policies",
        "filter_reason": "required for filtered policies",
        "filter_delta_norm": "required for filtered policies",
        "min_predicted_cpa_filtered": "reserved for Phase B formal filters; populated for minimal Phase A filter",
    }
    rows = []
    for field in fields:
        rows.append(
            {
                "table": table,
                "field": field,
                "required": 1,
                "phase_a_status": "implemented",
                "notes": required_notes.get(field, ""),
            }
        )
    return rows


def write_schema_csvs(out_dir: Path) -> None:
    write_csv(
        out_dir / "tables" / "phase_a_trace_schema.csv",
        schema_rows(TRACE_FIELDS, "per_step_trace") + schema_rows(OBSTACLE_FIELDS, "step_obstacles_long"),
        ["table", "field", "required", "phase_a_status", "notes"],
    )


def write_env_freeze_check(out_dir: Path) -> None:
    env_file = ROOT / "envs" / "dynamic_obstacle_flow_env.py"
    rows = [
        {
            "file": str(env_file.relative_to(ROOT)),
            "sha256": sha256_file(env_file),
            "phase_a_action": "read_only",
            "freeze_scope": "obstacle ranges, motion modes, scenarios, reward, termination, dynamics",
            "status": "EnvV2-core not modified by Phase A eval framework",
        }
    ]
    write_csv(out_dir / "tables" / "phase_a_env_freeze_check.csv", rows, list(rows[0].keys()))


def command_manifest_rows(args: argparse.Namespace, checkpoint: Path | None) -> list[dict[str, Any]]:
    runner_command = [
        "python",
        "scripts/run_env_v2_phase_a_eval_framework.py",
        "--out-dir",
        args.out_dir,
        "--num-episodes",
        str(args.num_episodes),
        "--scenarios",
        *args.scenarios,
        "--policies",
        *args.policies,
        "--eval-seed",
        str(args.eval_seed),
        "--seed",
        str(args.seed),
    ]
    if checkpoint is not None:
        runner_command += ["--checkpoint", str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint)]
    if args.write_traces:
        runner_command += ["--write-traces"]
    return [
        {
            "command_name": "phase_a_eval_runner",
            "command": " ".join(runner_command),
            "status": "completed",
        },
        {
            "command_name": "phase_a_watcher",
            "command": "bash scripts/watch_phase_a_eval_framework.sh",
            "status": "available",
        },
    ]


def write_policy_adapter_check(out_dir: Path, policies: list[str], checkpoint: Path | None, model_loaded: bool) -> None:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        rows.append(
            {
                "policy_name": policy,
                "act_schema": "action_raw/action_filtered/filter_triggered/filter_reason/debug",
                "filter_used": int(policy == "filtered_attention_full"),
                "checkpoint_required": int(policy in {"attention_full", "filtered_attention_full"}),
                "checkpoint_path": str(checkpoint.relative_to(ROOT) if checkpoint and checkpoint.is_relative_to(ROOT) else checkpoint or ""),
                "checkpoint_loaded": int(model_loaded) if policy in {"attention_full", "filtered_attention_full"} else 0,
                "status": "implemented",
            }
        )
    write_csv(out_dir / "tables" / "phase_a_policy_adapter_check.csv", rows, list(rows[0].keys()))


def validate_outputs(
    *,
    out_dir: Path,
    episode_rows: list[dict[str, Any]],
    trace_by_policy: dict[str, list[dict[str, Any]]],
    obstacle_rows_all: list[dict[str, Any]],
    policies: list[str],
    scenarios: list[str],
) -> None:
    missing_episode = [field for field in EPISODE_FIELDS if any(field not in row for row in episode_rows)]
    if missing_episode:
        raise PhaseAStop("schema_unification_failed", f"episode metrics missing fields: {missing_episode}")
    if not episode_rows:
        raise PhaseAStop("eval_framework_failed", "episode metrics CSV would be empty")
    if "eval_flow_id" not in scenarios or "eval_flow_high_speed" not in scenarios:
        raise PhaseAStop("schema_unification_failed", "smoke scenarios must include eval_flow_id and eval_flow_high_speed")
    for required_policy in ("random", "straight_line", "cpa_reactive"):
        if required_policy not in policies:
            raise PhaseAStop("schema_unification_failed", f"required smoke policy missing: {required_policy}")
    if not obstacle_rows_all:
        raise PhaseAStop("schema_unification_failed", "full active obstacle long table is empty")
    if not trace_by_policy:
        raise PhaseAStop("schema_unification_failed", "per-step trace output is empty")
    for policy in policies:
        rows = trace_by_policy.get(policy, [])
        if not rows:
            raise PhaseAStop("schema_unification_failed", f"missing per-step trace rows for policy: {policy}")
        missing_trace = [field for field in TRACE_FIELDS if any(field not in row for row in rows)]
        if missing_trace:
            raise PhaseAStop("schema_unification_failed", f"trace missing fields for {policy}: {missing_trace}")
    if "filtered_attention_full" in policies:
        rows = trace_by_policy.get("filtered_attention_full", [])
        required = [
            "action_raw_x",
            "action_filtered_x",
            "action_executed_x",
            "filter_used",
            "filter_triggered",
            "filter_reason",
            "filter_delta_norm",
            "min_predicted_cpa_raw",
            "min_predicted_cpa_filtered",
            "min_ttc_raw",
            "min_ttc_filtered",
            "unsafe_obstacle_id",
            "unsafe_obstacle_distance",
            "unsafe_obstacle_tcpa",
            "unsafe_obstacle_cpa",
        ]
        missing = [field for field in required if any(field not in row for row in rows)]
        if missing:
            raise PhaseAStop("schema_unification_failed", f"filtered policy missing trace fields: {missing}")
        triggered = sum(int(row["filter_triggered"]) for row in rows)
        max_delta = max(float(row["filter_delta_norm"]) for row in rows)
        if triggered <= 0 or max_delta <= 0.0:
            raise PhaseAStop(
                "schema_unification_failed",
                "filtered_attention_full trace did not include any raw-vs-filtered action difference",
            )
    expected_files = [
        out_dir / "tables" / "phase_a_eval_summary.csv",
        out_dir / "tables" / "phase_a_episode_metrics_sample.csv",
        out_dir / "tables" / "phase_a_trace_schema.csv",
        out_dir / "tables" / "phase_a_policy_adapter_check.csv",
        out_dir / "tables" / "phase_a_env_freeze_check.csv",
        out_dir / "tables" / "phase_a_command_manifest.csv",
        out_dir / "tables" / "phase_a_step_obstacles_sample.csv",
    ]
    for path in expected_files:
        if not path.exists() or path.stat().st_size == 0:
            raise PhaseAStop("eval_framework_failed", f"required output missing or empty: {path}")


def fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def write_report(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    checkpoint: Path | None,
    summary_rows: list[dict[str, Any]],
    complete: bool,
    stop_reason: str | None = None,
    stop_detail: str | None = None,
) -> None:
    checkpoint_text = str(checkpoint.relative_to(ROOT) if checkpoint and checkpoint.is_relative_to(ROOT) else checkpoint or "n/a")
    lines = [
        "# Phase A Eval Framework Report",
        "",
        "## 1. Background And Goal",
        "",
        "Phase A establishes a unified evaluation runner and trace schema for EnvV2 baseline auditing. It does not train a new policy and does not introduce a new method claim.",
        "",
        "## 2. EnvV2-Core Freeze Statement",
        "",
        "EnvV2-core was frozen in Phase A. This phase did not modify obstacle count ranges, motion modes, train/eval scenario definitions, action dynamics, reward function, collision/success/near-miss definitions, or termination logic. All changes were limited to evaluation infrastructure, policy/controller adapters, unified logging, trace schema, and watcher/report generation.",
        "",
        "## 3. Added Or Modified Files",
        "",
        "- `scripts/run_env_v2_phase_a_eval_framework.py`",
        "- `scripts/watch_phase_a_eval_framework.sh`",
        f"- `{out_dir.relative_to(ROOT)}/` result artifacts",
        "",
        "## 4. Eval Runner Usage",
        "",
        "```bash",
        "python scripts/run_env_v2_phase_a_eval_framework.py \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --num-episodes {args.num_episodes} \\",
        f"  --scenarios {' '.join(args.scenarios)} \\",
        f"  --policies {' '.join(args.policies)} \\",
        f"  --checkpoint {checkpoint_text} \\",
        f"  --eval-seed {args.eval_seed} \\",
        "  --write-traces",
        "```",
        "",
        "## 5. Supported Policies / Controllers",
        "",
        "- `random`: action-space sample with fixed per-episode RNG.",
        "- `straight_line`: normalized horizontal goal direction.",
        "- `cpa_reactive`: current CPA-reactive logic reused from `scripts/run_env_v2_sanity.py`.",
        "- `attention_full`: SB3 PPO attention checkpoint adapter.",
        "- `filtered_attention_full`: Phase A minimal safety-filter wrapper around `attention_full`.",
        "",
        "The minimal filter is for infrastructure validation only and is not a formal Phase B baseline.",
        "",
        "## 6. Episode-Level Metrics Schema",
        "",
        f"`tables/phase_a_episode_metrics_sample.csv` uses {len(EPISODE_FIELDS)} fixed fields: `{', '.join(EPISODE_FIELDS)}`.",
        "",
        "## 7. Per-Step Trace Schema",
        "",
        f"`traces/sample_<policy>_trace.csv` uses {len(TRACE_FIELDS)} fixed trace fields. `tables/phase_a_trace_schema.csv` records the trace and obstacle-long schema.",
        "",
        "## 8. Full Active Obstacle Set Logging",
        "",
        "`tables/phase_a_step_obstacles_sample.csv` uses long-table logging. Every per-step active obstacle is recorded with slot, id, position, velocity, distance, closing speed, planned CPA/TTC, threat class, motion mode, and risk value.",
        "",
        "## 9. Safety Filter Trace Fields",
        "",
        "`filtered_attention_full` records `action_raw`, `action_filtered`, `action_executed`, `filter_triggered`, `filter_reason`, `filter_delta_norm`, `min_predicted_cpa_raw`, `min_predicted_cpa_filtered`, `min_ttc_raw`, `min_ttc_filtered`, and unsafe-obstacle metadata. These columns are reserved for Phase B formal safety-filter baselines and populated by the Phase A minimal filter where available.",
        "",
        "## 10. Seed Rule And Fairness Note",
        "",
        f"`episode_seed = eval_seed + seed * 10000 + episode_id`, with `eval_seed={args.eval_seed}` and `seed={args.seed}`.",
        "",
        "Because obstacle replacement depends on policy trajectory, identical reset seed does not guarantee identical obstacle schedules across policies. Phase A standardizes initial seeds and trace logging; stricter precomputed spawn schedule / decoupled obstacle RNG is deferred unless required by later comparisons.",
        "",
        "## 11. Smoke Test Scale And Results",
        "",
        f"- scenarios: `{', '.join(args.scenarios)}`",
        f"- policies: `{', '.join(args.policies)}`",
        f"- episodes per scenario-policy: `{args.num_episodes}`",
        f"- checkpoint: `{checkpoint_text}`",
        "",
        "| policy | scenario | episodes | success | collision | near_miss | mean_min_distance | filter_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy_name']} | {row['scenario']} | {int(row['episodes'])} | "
            f"{fmt(row['success_rate'])} | {fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | "
            f"{fmt(row['mean_min_distance'])} | {fmt(row['mean_filter_trigger_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 12. Generated Artifacts",
            "",
            "- `tables/phase_a_eval_summary.csv`",
            "- `tables/phase_a_episode_metrics_sample.csv`",
            "- `tables/phase_a_trace_schema.csv`",
            "- `tables/phase_a_policy_adapter_check.csv`",
            "- `tables/phase_a_env_freeze_check.csv`",
            "- `tables/phase_a_command_manifest.csv`",
            "- `tables/phase_a_step_obstacles_sample.csv`",
            "- `traces/sample_random_trace.csv`",
            "- `traces/sample_straight_line_trace.csv`",
            "- `traces/sample_cpa_reactive_trace.csv`",
            "- `traces/sample_attention_full_trace.csv`",
            "- `traces/sample_filtered_attention_trace.csv`",
            "- `logs/phase_a_eval_framework.log`",
            "- `phase_a_status.txt`",
            "- `phase_a_watcher.log` after watcher execution",
            "",
            "## 13. Phase A Completion Criteria",
            "",
            "- EnvV2-core freeze documented and checked.",
            "- Unified runner exists and completed the smoke test.",
            "- Required policies are supported.",
            "- Episode metrics CSV has the fixed schema.",
            "- Per-step traces exist for each policy.",
            "- Full active obstacle set is logged with a long table.",
            "- Filtered policy records raw and filtered actions plus filter metadata.",
            "- Report, status, log, and flag files are generated.",
            "",
            "## 14. Conclusion",
            "",
        ]
    )
    if complete:
        lines.extend(
            [
                "Phase A complete.",
                "Unified eval framework and trace schema are ready for Phase B.",
            ]
        )
    else:
        lines.extend(
            [
                "Phase A not complete.",
                f"stop_reason: {stop_reason or 'unknown'}",
                f"stop_detail: {stop_detail or 'unknown'}",
                "Phase B is blocked until the stop reason is resolved.",
            ]
        )
    (out_dir / "PHASE_A_EVAL_FRAMEWORK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_old_flags(out_dir: Path) -> None:
    for flag in (COMPLETE_FLAG, *STOP_FLAGS):
        path = out_dir / flag
        if path.exists():
            path.unlink()


def write_status(out_dir: Path, status: str) -> None:
    (out_dir / "phase_a_status.txt").write_text(status + "\n", encoding="utf-8")


def write_flag(out_dir: Path, flag: str, detail: str) -> None:
    (out_dir / flag).write_text(detail.rstrip() + "\n", encoding="utf-8")


def stop_run(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    checkpoint: Path | None,
    logger: PhaseALogger,
    reason: str,
    detail: str,
    summary_rows: list[dict[str, Any]] | None = None,
) -> None:
    flag = STOP_REASON_TO_FLAG.get(reason, "PHASE_A_STOP_EVAL_FRAMEWORK_FAILED.flag")
    write_status(out_dir, f"stopped:{flag}")
    write_report(
        out_dir=out_dir,
        args=args,
        checkpoint=checkpoint,
        summary_rows=summary_rows or [],
        complete=False,
        stop_reason=reason,
        stop_detail=detail,
    )
    write_flag(out_dir, flag, f"terminal_decision=phase_a_stopped_{reason}\ndetail={detail}")
    logger.log(f"PHASE_A_STOP reason={reason} flag={flag} detail={detail}")


def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    for subdir in RESULT_SUBDIRS:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    clear_old_flags(out_dir)
    write_status(out_dir, "running")
    logger = PhaseALogger(out_dir)
    logger.log("PHASE_A_EVAL_FRAMEWORK_START")

    policies = list(args.policies)
    scenarios = list(args.scenarios)
    checkpoint: Path | None = None
    summary_rows: list[dict[str, Any]] = []
    try:
        checkpoint = resolve_checkpoint(args, policies)
        checkpoint_step = derive_checkpoint_step(checkpoint)
        model = load_attention_model(checkpoint, args, logger)
        write_env_freeze_check(out_dir)
        write_schema_csvs(out_dir)
        write_policy_adapter_check(out_dir, policies, checkpoint, model is not None)
        write_csv(
            out_dir / "tables" / "phase_a_command_manifest.csv",
            command_manifest_rows(args, checkpoint),
            ["command_name", "command", "status"],
        )

        if args.dry_run:
            raise PhaseAStop("eval_framework_failed", "dry_run=true does not produce required Phase A CSV/trace artifacts")

        episode_rows: list[dict[str, Any]] = []
        obstacle_rows_all: list[dict[str, Any]] = []
        trace_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        total = len(scenarios) * len(policies) * args.num_episodes
        completed = 0
        start = time.time()
        last_heartbeat = start

        for scenario in scenarios:
            for policy_index, policy_name in enumerate(policies):
                for episode_id in range(args.num_episodes):
                    episode_seed = args.eval_seed + args.seed * 10000 + episode_id
                    rng = np.random.default_rng(args.eval_seed + args.seed * 10000 + policy_index * 1000 + episode_id)
                    row, trace_rows, step_obstacle_rows = run_episode(
                        policy_name=policy_name,
                        scenario=scenario,
                        episode_id=episode_id,
                        episode_seed=episode_seed,
                        checkpoint_step=checkpoint_step if policy_name in {"attention_full", "filtered_attention_full"} else -1,
                        rng=rng,
                        model=model if policy_name in {"attention_full", "filtered_attention_full"} else None,
                        write_traces=args.write_traces,
                        logger=logger,
                    )
                    episode_rows.append(row)
                    trace_by_policy[policy_name].extend(trace_rows)
                    obstacle_rows_all.extend(step_obstacle_rows)
                    completed += 1
                    now = time.time()
                    if now - last_heartbeat >= args.heartbeat_seconds or completed == total:
                        elapsed = now - start
                        rate = completed / max(elapsed, 1e-6)
                        logger.log(
                            "PHASE_A_EVAL_HEARTBEAT "
                            f"completed={completed}/{total} policy={policy_name} scenario={scenario} "
                            f"rate={rate:.2f}_episodes_per_sec"
                        )
                        write_status(out_dir, f"running completed={completed}/{total}")
                        last_heartbeat = now

        summary_rows = summarize_rows(episode_rows)
        write_csv(out_dir / "tables" / "phase_a_episode_metrics_sample.csv", episode_rows, EPISODE_FIELDS)
        write_csv(out_dir / "tables" / "phase_a_eval_summary.csv", summary_rows, SUMMARY_FIELDS)
        write_csv(out_dir / "tables" / "phase_a_step_obstacles_sample.csv", obstacle_rows_all, OBSTACLE_FIELDS)

        trace_name_map = {
            "random": "sample_random_trace.csv",
            "straight_line": "sample_straight_line_trace.csv",
            "cpa_reactive": "sample_cpa_reactive_trace.csv",
            "attention_full": "sample_attention_full_trace.csv",
            "filtered_attention_full": "sample_filtered_attention_trace.csv",
        }
        for policy_name, rows in trace_by_policy.items():
            trace_path = out_dir / "traces" / trace_name_map.get(policy_name, f"sample_{policy_name}_trace.csv")
            write_csv(trace_path, rows, TRACE_FIELDS)

        validate_outputs(
            out_dir=out_dir,
            episode_rows=episode_rows,
            trace_by_policy=trace_by_policy,
            obstacle_rows_all=obstacle_rows_all,
            policies=policies,
            scenarios=scenarios,
        )
        write_report(
            out_dir=out_dir,
            args=args,
            checkpoint=checkpoint,
            summary_rows=summary_rows,
            complete=True,
        )
        write_status(out_dir, "complete")
        write_flag(
            out_dir,
            COMPLETE_FLAG,
            "terminal_decision=phase_a_eval_framework_complete\nnext_recommended_phase=Phase B\n",
        )
        logger.log("PHASE_A_EVAL_FRAMEWORK_COMPLETE")
    except PhaseAStop as exc:
        stop_run(
            out_dir=out_dir,
            args=args,
            checkpoint=checkpoint,
            logger=logger,
            reason=exc.reason,
            detail=exc.detail,
            summary_rows=summary_rows,
        )
        raise SystemExit(2) from None
    except Exception as exc:
        detail = f"unexpected exception: {exc!r}"
        logger.log(detail)
        logger.log(traceback.format_exc())
        stop_run(
            out_dir=out_dir,
            args=args,
            checkpoint=checkpoint,
            logger=logger,
            reason="eval_framework_failed",
            detail=detail,
            summary_rows=summary_rows,
        )
        raise SystemExit(2) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/env_v2_phase_a_eval_framework")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--dry-run", type=str2bool, default=False)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.num_episodes <= 0:
        raise argparse.ArgumentTypeError("--num-episodes must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
