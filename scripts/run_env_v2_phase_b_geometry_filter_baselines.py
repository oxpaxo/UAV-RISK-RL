from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from policies.obstacle_set_extractor import ObstacleSetExtractor
from scripts.run_env_v2_phase_a_eval_framework import EPISODE_FIELDS as PHASE_A_EPISODE_FIELDS
from scripts.run_env_v2_phase_a_eval_framework import OBSTACLE_FIELDS, TRACE_FIELDS


ALL_SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
SMOKE_SCENARIOS = ["eval_flow_id", "eval_flow_high_speed"]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip"
PHASE_A_DIR = ROOT / "results/env_v2_phase_a_eval_framework"
PHASE_A_COMPLETE = PHASE_A_DIR / "PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag"
PHASE_A_TRACE_SCHEMA = PHASE_A_DIR / "tables/phase_a_trace_schema.csv"
PHASE_A_EPISODE_SCHEMA_SAMPLE = PHASE_A_DIR / "tables/phase_a_episode_metrics_sample.csv"
PHASE_A_ENV_FREEZE = PHASE_A_DIR / "tables/phase_a_env_freeze_check.csv"
ENV_FILE = ROOT / "envs/dynamic_obstacle_flow_env.py"

COMPLETE_FLAG = "PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag"
STOP_FLAGS = (
    "PHASE_B_STOP_PHASE_A_MISSING.flag",
    "PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag",
    "PHASE_B_STOP_BASELINE_IMPL_FAILED.flag",
    "PHASE_B_STOP_EVAL_FAILED.flag",
    "PHASE_B_STOP_SCHEMA_MISMATCH.flag",
    "PHASE_B_STOP_RESOURCE_LIMIT.flag",
    "PHASE_B_STOP_WATCHER_FAILED.flag",
)
STOP_REASON_TO_FLAG = {
    "phase_a_missing": "PHASE_B_STOP_PHASE_A_MISSING.flag",
    "env_core_change_required": "PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "checkpoint_not_found": "PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag",
    "baseline_impl_failed": "PHASE_B_STOP_BASELINE_IMPL_FAILED.flag",
    "eval_failed": "PHASE_B_STOP_EVAL_FAILED.flag",
    "schema_mismatch": "PHASE_B_STOP_SCHEMA_MISMATCH.flag",
    "resource_limit": "PHASE_B_STOP_RESOURCE_LIMIT.flag",
}

PHASE_B_EXTRA_EPISODE_FIELDS = [
    "stage",
    "baseline_name",
    "config_name",
    "baseline_category",
    "config_params",
    "episode_filter_triggered",
    "mean_min_predicted_cpa_raw",
    "mean_min_predicted_cpa_filtered",
    "min_min_predicted_cpa_raw",
    "min_min_predicted_cpa_filtered",
    "mean_min_ttc_raw",
    "mean_min_ttc_filtered",
    "min_min_ttc_raw",
    "min_min_ttc_filtered",
    "failure_type",
    "nan_or_crash",
]
EPISODE_FIELDS = PHASE_B_EXTRA_EPISODE_FIELDS + PHASE_A_EPISODE_FIELDS
PHASE_B_TRACE_FIELDS = TRACE_FIELDS + ["stage", "baseline_name", "config_name", "baseline_category"]
PHASE_B_OBSTACLE_FIELDS = OBSTACLE_FIELDS + ["stage", "baseline_name", "config_name", "baseline_category"]


class PhaseBStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "logs/phase_b_geometry_filter_eval.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class BaselineConfig:
    baseline_name: str
    config_name: str
    category: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        if fallback is None:
            return np.zeros_like(arr, dtype=np.float32)
        return normalize(fallback)
    return (arr / norm).astype(np.float32)


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def bool_int(value: Any) -> int:
    return int(bool(value))


def json_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        arr = np.full(max_obs, np.nan, dtype=np.float32)
    else:
        arr = weights.detach().cpu().numpy()
        if arr.ndim >= 2:
            arr = arr[0]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] < max_obs:
            arr = np.pad(arr, (0, max_obs - arr.shape[0]), constant_values=np.nan)
        arr = arr[:max_obs]
    if entropy is None:
        entropy_value = float("nan")
    else:
        entropy_arr = entropy.detach().cpu().numpy()
        entropy_value = float(np.ravel(entropy_arr)[0]) if entropy_arr.size else float("nan")
    return arr, entropy_value


def goal_action(info: dict[str, Any]) -> np.ndarray:
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    vec = goal - uav
    vec[2] = 0.0
    return np.clip(normalize(vec, np.array([1.0, 0.0, 0.0], dtype=np.float32)), -1.0, 1.0)


def nearest_obstacle_detail(info: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    if len(positions) == 0:
        return {"index": -1, "id": -1, "distance": float("nan"), "position": np.zeros(3), "velocity": np.zeros(3)}
    distances = np.linalg.norm(positions - uav, axis=1)
    idx = int(np.argmin(distances))
    return {
        "index": idx,
        "id": int(ids[idx]) if idx < len(ids) else idx,
        "distance": float(distances[idx]),
        "position": positions[idx],
        "velocity": velocities[idx] if idx < len(velocities) else np.zeros(3, dtype=np.float32),
    }


def cpa_for_obstacle(action: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], index: int, horizon: float) -> dict[str, float]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    if index < 0 or index >= len(positions):
        return {"id": -1, "distance": float("nan"), "tcpa": float("nan"), "cpa": float("nan")}
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    v_cmd = np.asarray(action, dtype=np.float32) * float(env.v_uav_max)
    v_cmd[2] = 0.0
    rel = positions[index] - uav
    rel[2] = 0.0
    distance = float(np.linalg.norm(rel))
    rel_vel = velocities[index] - v_cmd
    rel_vel[2] = 0.0
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    tcpa = 0.0
    cpa = distance
    if rel_speed_sq > 1e-8:
        tcpa = float(np.clip(-np.dot(rel, rel_vel) / rel_speed_sq, 0.0, horizon))
        cpa = float(np.linalg.norm(rel + rel_vel * tcpa))
    return {
        "id": int(ids[index]) if index < len(ids) else int(index),
        "distance": distance,
        "tcpa": tcpa,
        "cpa": cpa,
    }


def predicted_cpa_geometry(action: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], horizon: float = 4.5) -> dict[str, float]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    if len(positions) == 0:
        return {
            "min_cpa": float("nan"),
            "min_ttc": float("nan"),
            "unsafe_obstacle_id": -1,
            "unsafe_obstacle_distance": float("nan"),
            "unsafe_obstacle_tcpa": float("nan"),
            "unsafe_obstacle_cpa": float("nan"),
            "unsafe_obstacle_index": -1,
        }
    best = {
        "min_cpa": float("inf"),
        "min_ttc": float("nan"),
        "unsafe_obstacle_id": -1,
        "unsafe_obstacle_distance": float("nan"),
        "unsafe_obstacle_tcpa": float("nan"),
        "unsafe_obstacle_cpa": float("inf"),
        "unsafe_obstacle_index": -1,
    }
    for index in range(len(positions)):
        geom = cpa_for_obstacle(action, env, info, index, horizon)
        if geom["cpa"] < best["min_cpa"]:
            best = {
                "min_cpa": geom["cpa"],
                "min_ttc": geom["tcpa"],
                "unsafe_obstacle_id": int(geom["id"]),
                "unsafe_obstacle_distance": geom["distance"],
                "unsafe_obstacle_tcpa": geom["tcpa"],
                "unsafe_obstacle_cpa": geom["cpa"],
                "unsafe_obstacle_index": index,
            }
    return best


def away_lateral_action(info: dict[str, Any], obstacle_index: int) -> np.ndarray:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    goal = goal_action(info)
    if obstacle_index < 0 or obstacle_index >= len(positions):
        return goal
    rel = positions[obstacle_index] - uav
    rel[2] = 0.0
    away = normalize(-rel, -goal)
    lateral = np.array([-goal[1], goal[0], 0.0], dtype=np.float32)
    if np.dot(lateral, away) < 0.0:
        lateral = -lateral
    return normalize(0.75 * away + 0.55 * lateral, away)


def cpa_reactive_action(env: DynamicObstacleFlowEnv, info: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    v_goal = goal_action(info)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    obs_pos = np.asarray(info["obstacle_positions"], dtype=np.float32)
    obs_vel = np.asarray(info["obstacle_velocities"], dtype=np.float32)
    d_reactive = float(params.get("d_reactive", 4.0))
    horizon = float(params.get("horizon", 4.5))
    cpa_trigger = float(params.get("cpa_trigger", 2.4))
    avoid_weight = float(params.get("avoid_weight", 2.1))
    avoid = np.zeros(3, dtype=np.float32)
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
        if distance < d_reactive or (0.0 < tcpa < horizon and cpa_distance < cpa_trigger):
            away = -rel / (distance + 1e-6)
            lateral = np.array([-v_goal[1], v_goal[0], 0.0], dtype=np.float32)
            if np.dot(lateral, away) < 0.0:
                lateral = -lateral
            proximity_gain = max((d_reactive - min(distance, d_reactive)) / d_reactive, 0.0)
            cpa_gain = max((cpa_trigger - min(cpa_distance, cpa_trigger)) / cpa_trigger, 0.0)
            closing_gain = 1.0 if tcpa > 0.0 else 0.35
            avoid += (1.3 * proximity_gain + 2.2 * cpa_gain * closing_gain) * (0.65 * away + 0.55 * lateral)
    command = normalize(v_goal + avoid_weight * avoid, v_goal)
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def apf_action(env: DynamicObstacleFlowEnv, info: dict[str, Any], cfg: BaselineConfig) -> np.ndarray:
    del env
    v_goal = goal_action(info)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    obs_pos = np.asarray(info["obstacle_positions"], dtype=np.float32)
    obs_vel = np.asarray(info["obstacle_velocities"], dtype=np.float32)
    d0 = float(cfg.params.get("d0", 4.0))
    w_goal = float(cfg.params.get("w_goal", 1.0))
    w_rep = float(cfg.params.get("w_rep", 1.0))
    avoid = np.zeros(3, dtype=np.float32)
    for pos, vel in zip(obs_pos, obs_vel):
        rel = pos - uav
        rel[2] = 0.0
        d = float(np.linalg.norm(rel))
        if d < 1e-6 or d >= d0:
            continue
        away = -rel / (d + 1e-6)
        base_rep = w_rep * (1.0 / (d + 1e-6) - 1.0 / d0) / (d * d + 1e-6)
        gain = base_rep
        if cfg.kind in {"velocity_aware_apf", "cpa_ttc_weighted_apf"}:
            rel_vel = vel - uav_vel
            rel_vel[2] = 0.0
            closing = -float(np.dot(rel, rel_vel)) / (d + 1e-6)
            closing_gain = float(np.clip(closing / 2.0, 0.0, 1.5))
            gain *= 1.0 + float(cfg.params.get("alpha_closing", 1.0)) * closing_gain
        if cfg.kind == "cpa_ttc_weighted_apf":
            rel_vel = vel - uav_vel
            rel_vel[2] = 0.0
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))
            horizon = float(cfg.params.get("horizon", 4.5))
            cpa_threshold = float(cfg.params.get("cpa_threshold", 2.4))
            tcpa = 0.0
            cpa_dist = d
            if rel_speed_sq > 1e-8:
                tcpa = float(np.clip(-np.dot(rel, rel_vel) / rel_speed_sq, 0.0, horizon))
                cpa_dist = float(np.linalg.norm(rel + rel_vel * tcpa))
            cpa_gain = float(np.clip((cpa_threshold - cpa_dist) / cpa_threshold, 0.0, 1.0))
            tcpa_gain = float(np.clip((horizon - tcpa) / horizon, 0.0, 1.0))
            gain *= 1.0 + float(cfg.params.get("alpha_cpa", 2.0)) * cpa_gain * tcpa_gain
        avoid += gain * away
    command = normalize(w_goal * v_goal + avoid, v_goal)
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def filter_debug(action_raw: np.ndarray, action_filtered: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], horizon: float) -> dict[str, float]:
    raw = predicted_cpa_geometry(action_raw, env, info, horizon)
    filtered = predicted_cpa_geometry(action_filtered, env, info, horizon)
    return {
        "min_predicted_cpa_raw": raw["min_cpa"],
        "min_predicted_cpa_filtered": filtered["min_cpa"],
        "min_ttc_raw": raw["min_ttc"],
        "min_ttc_filtered": filtered["min_ttc"],
        "unsafe_obstacle_id": raw["unsafe_obstacle_id"],
        "unsafe_obstacle_distance": raw["unsafe_obstacle_distance"],
        "unsafe_obstacle_tcpa": raw["unsafe_obstacle_tcpa"],
        "unsafe_obstacle_cpa": raw["unsafe_obstacle_cpa"],
    }


def distance_filter(action_raw: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], cfg: BaselineConfig) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    nearest = nearest_obstacle_detail(info)
    d_filter = float(cfg.params["d_filter"])
    beta = float(cfg.params["beta"])
    action_filtered = np.asarray(action_raw, dtype=np.float32).copy()
    triggered = False
    reason = "none"
    if nearest["index"] >= 0 and finite(nearest["distance"]) < d_filter:
        uav = np.asarray(info["uav_position"], dtype=np.float32)
        rel = np.asarray(nearest["position"], dtype=np.float32) - uav
        rel[2] = 0.0
        rel_dir = normalize(rel)
        v_raw = np.asarray(action_raw, dtype=np.float32) * env.v_uav_max
        v_raw[2] = 0.0
        obs_vel = np.asarray(nearest["velocity"], dtype=np.float32)
        closing_raw = -float(np.dot(rel_dir, obs_vel - v_raw))
        if closing_raw > 0.0:
            avoid = away_lateral_action(info, int(nearest["index"]))
            blended = (1.0 - beta) * np.asarray(action_raw, dtype=np.float32) + beta * avoid
            blended[2] = action_raw[2]
            action_filtered = np.clip(normalize(blended, avoid), -1.0, 1.0)
            triggered = True
            reason = f"distance_filter_d{d_filter:g}_beta{beta:g}"
    return action_filtered, triggered, reason, filter_debug(action_raw, action_filtered, env, info, horizon=4.5)


def cpa_ttc_filter(action_raw: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], cfg: BaselineConfig) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    horizon = float(cfg.params["horizon"])
    cpa_safe = float(cfg.params["cpa_safe"])
    beta = float(cfg.params.get("beta", 0.8))
    raw = predicted_cpa_geometry(action_raw, env, info, horizon)
    triggered = bool(np.isfinite(raw["min_cpa"]) and 0.0 < raw["min_ttc"] < horizon and raw["min_cpa"] < cpa_safe)
    if not triggered:
        action_filtered = np.asarray(action_raw, dtype=np.float32).copy()
        reason = "none"
    else:
        avoid = away_lateral_action(info, int(raw["unsafe_obstacle_index"]))
        blended = (1.0 - beta) * np.asarray(action_raw, dtype=np.float32) + beta * avoid
        blended[2] = action_raw[2]
        action_filtered = np.clip(normalize(blended, avoid), -1.0, 1.0)
        reason = f"cpa_ttc_filter_h{horizon:g}_cpa{cpa_safe:g}"
    return action_filtered, triggered, reason, filter_debug(action_raw, action_filtered, env, info, horizon)


def vo_like_filter(action_raw: np.ndarray, env: DynamicObstacleFlowEnv, info: dict[str, Any], cfg: BaselineConfig) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    horizon = float(cfg.params["horizon"])
    cpa_safe = float(cfg.params["cpa_safe"])
    headings = int(cfg.params.get("num_headings", 16))
    raw = predicted_cpa_geometry(action_raw, env, info, horizon)
    triggered = bool(np.isfinite(raw["min_cpa"]) and 0.0 <= raw["min_ttc"] <= horizon and raw["min_cpa"] < cpa_safe)
    if not triggered:
        action_filtered = np.asarray(action_raw, dtype=np.float32).copy()
        return action_filtered, False, "none", filter_debug(action_raw, action_filtered, env, info, horizon)

    goal = goal_action(info)
    current_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    current_action = current_vel / max(float(env.v_uav_max), 1e-6)
    current_action[2] = 0.0
    candidates: list[np.ndarray] = [
        np.asarray(action_raw, dtype=np.float32).copy(),
        goal.copy(),
        normalize(current_action, goal),
        away_lateral_action(info, int(raw["unsafe_obstacle_index"])),
    ]
    angles = np.linspace(0.0, 2.0 * math.pi, headings, endpoint=False)
    for speed in (0.4, 0.7, 1.0):
        for angle in angles:
            candidates.append(np.array([math.cos(angle) * speed, math.sin(angle) * speed, action_raw[2]], dtype=np.float32))

    best_safe: tuple[float, np.ndarray] | None = None
    best_any: tuple[float, np.ndarray] | None = None
    for cand in candidates:
        cand = np.clip(cand.astype(np.float32), -1.0, 1.0)
        cand[2] = action_raw[2]
        geom = predicted_cpa_geometry(cand, env, info, horizon)
        min_cpa = finite(geom["min_cpa"], 0.0)
        unsafe = bool(np.isfinite(geom["min_ttc"]) and 0.0 <= geom["min_ttc"] <= horizon and min_cpa < cpa_safe)
        progress_alignment = float(np.dot(normalize(cand, goal), goal))
        score = (
            -1.2 * float(np.linalg.norm(cand - action_raw))
            + 0.8 * progress_alignment
            + 0.35 * min_cpa
            - 0.2 * float(np.linalg.norm(cand - current_action))
        )
        if best_any is None or score > best_any[0]:
            best_any = (score, cand.copy())
        if not unsafe and (best_safe is None or score > best_safe[0]):
            best_safe = (score, cand.copy())
    action_filtered = (best_safe or best_any or (0.0, np.asarray(action_raw, dtype=np.float32)))[1]
    reason = f"vo_like_filter_h{horizon:g}_cpa{cpa_safe:g}_h{headings}"
    return action_filtered, True, reason, filter_debug(action_raw, action_filtered, env, info, horizon)


def act(cfg: BaselineConfig, obs: dict[str, np.ndarray], info: dict[str, Any], env: DynamicObstacleFlowEnv, rng: np.random.Generator, model: PPO | None) -> dict[str, Any]:
    if cfg.kind == "random":
        action_raw = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        action_raw[2] *= 0.2
    elif cfg.kind == "straight_line":
        action_raw = goal_action(info)
    elif cfg.kind == "attention_full":
        if model is None:
            raise PhaseBStop("checkpoint_not_found", "attention_full_1500k requested but model is not loaded")
        action_raw, _ = model.predict(obs, deterministic=True)
        action_raw = np.asarray(action_raw, dtype=np.float32)
    elif cfg.kind == "cpa_reactive":
        action_raw = cpa_reactive_action(env, info, cfg.params)
    elif cfg.kind in {"naive_apf", "velocity_aware_apf", "cpa_ttc_weighted_apf"}:
        action_raw = apf_action(env, info, cfg)
    elif cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"}:
        if model is None:
            raise PhaseBStop("checkpoint_not_found", f"{cfg.kind} requested but attention_full model is not loaded")
        action_raw, _ = model.predict(obs, deterministic=True)
        action_raw = np.asarray(action_raw, dtype=np.float32)
    else:
        raise PhaseBStop("baseline_impl_failed", f"unsupported baseline kind: {cfg.kind}")

    action_raw = np.clip(np.asarray(action_raw, dtype=np.float32), -1.0, 1.0)
    if cfg.kind == "distance_filter":
        action_filtered, triggered, reason, debug = distance_filter(action_raw, env, info, cfg)
    elif cfg.kind == "cpa_ttc_filter":
        action_filtered, triggered, reason, debug = cpa_ttc_filter(action_raw, env, info, cfg)
    elif cfg.kind == "vo_like_filter":
        action_filtered, triggered, reason, debug = vo_like_filter(action_raw, env, info, cfg)
    else:
        action_filtered = action_raw.copy()
        triggered = False
        reason = "none"
        debug = filter_debug(action_raw, action_filtered, env, info, horizon=4.5)
    return {
        "action_raw": action_raw,
        "action_filtered": np.clip(action_filtered, -1.0, 1.0).astype(np.float32),
        "filter_used": cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"},
        "filter_triggered": bool(triggered),
        "filter_reason": reason,
        "debug": debug,
    }


def lateral_deviation(action: np.ndarray, goal: np.ndarray, uav: np.ndarray, v_uav_max: float) -> float:
    goal_vec = np.asarray(goal, dtype=np.float32) - np.asarray(uav, dtype=np.float32)
    goal_vec[2] = 0.0
    goal_dir = normalize(goal_vec)
    v_des = np.asarray(action, dtype=np.float32) * float(v_uav_max)
    v_des[2] = 0.0
    parallel = float(np.dot(v_des, goal_dir)) * goal_dir
    return float(np.linalg.norm(v_des - parallel))


def threat_away_velocity(info: dict[str, Any]) -> float:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    idx = int(info.get("threat_obstacle_index", -1))
    if idx < 0 or idx >= len(positions):
        return float("nan")
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    rel = positions[idx] - uav
    rel[2] = 0.0
    return float(np.dot(uav_vel, normalize(-rel)))


def debug_value(act_result: dict[str, Any], key: str) -> float:
    return finite((act_result.get("debug") or {}).get(key))


def build_trace_row(
    cfg: BaselineConfig,
    stage: str,
    scenario: str,
    checkpoint_step: int,
    episode_id: int,
    episode_seed: int,
    env: DynamicObstacleFlowEnv,
    info: dict[str, Any],
    act_result: dict[str, Any],
    done: bool,
    terminated: bool,
    truncated: bool,
    reaction_flag: bool,
    no_response_flag: bool,
    model: PPO | None,
) -> dict[str, Any]:
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    action_raw = np.asarray(act_result["action_raw"], dtype=np.float32)
    action_filtered = np.asarray(act_result["action_filtered"], dtype=np.float32)
    action_executed = np.clip(action_filtered, env.action_space.low, env.action_space.high).astype(np.float32)
    nearest = nearest_obstacle_detail(info)
    weights, entropy = attention_snapshot(model if cfg.kind in {"attention_full", "distance_filter", "cpa_ttc_filter", "vo_like_filter"} else None, env.max_obs)
    threat_idx = int(info.get("threat_obstacle_index", -1))
    threat_weight = float(weights[threat_idx]) if 0 <= threat_idx < len(weights) else float("nan")
    rank = float("nan")
    if 0 <= threat_idx < len(weights):
        order = np.argsort(-np.nan_to_num(weights, nan=-np.inf))
        match = np.where(order == threat_idx)[0]
        if match.size:
            rank = float(match[0] + 1)
    row = {
        "checkpoint_step": int(checkpoint_step),
        "method": cfg.baseline_name,
        "policy_name": cfg.config_name,
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
        "goal_dist": finite(info.get("goal_distance", np.linalg.norm(goal - uav))),
        "action_raw_x": float(action_raw[0]),
        "action_raw_y": float(action_raw[1]),
        "action_raw_z": float(action_raw[2]),
        "action_filtered_x": float(action_filtered[0]),
        "action_filtered_y": float(action_filtered[1]),
        "action_filtered_z": float(action_filtered[2]),
        "action_executed_x": float(action_executed[0]),
        "action_executed_y": float(action_executed[1]),
        "action_executed_z": float(action_executed[2]),
        "filter_used": bool_int(act_result["filter_used"]),
        "filter_triggered": bool_int(act_result["filter_triggered"]),
        "filter_reason": str(act_result["filter_reason"]),
        "filter_delta_norm": float(np.linalg.norm(action_filtered - action_raw)),
        "min_distance": finite(info.get("min_distance")),
        "nearest_obstacle_id": int(nearest["id"]),
        "nearest_obstacle_distance": finite(nearest["distance"]),
        "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
        "threat_obstacle_index": int(threat_idx),
        "threat_class": str(info.get("threat_class", "none")),
        "planned_cpa": finite(info.get("planned_cpa_to_threat")),
        "planned_ttc": finite(info.get("planned_ttc_to_threat")),
        "planned_ttc_remaining": finite(info.get("planned_ttc_remaining_to_threat")),
        "lateral_deviation": lateral_deviation(action_executed, goal, uav, env.v_uav_max),
        "away_from_threat_velocity": threat_away_velocity(info),
        "goal_directed_velocity": float(np.dot(uav_vel, normalize(goal - uav))),
        "reaction_flag": bool_int(reaction_flag),
        "no_response_flag": bool_int(no_response_flag),
        "attention_entropy": entropy,
        "threat_obstacle_attention_weight": threat_weight,
        "threat_obstacle_attention_rank": rank,
        "min_predicted_cpa_raw": debug_value(act_result, "min_predicted_cpa_raw"),
        "min_predicted_cpa_filtered": debug_value(act_result, "min_predicted_cpa_filtered"),
        "min_ttc_raw": debug_value(act_result, "min_ttc_raw"),
        "min_ttc_filtered": debug_value(act_result, "min_ttc_filtered"),
        "unsafe_obstacle_id": int(debug_value(act_result, "unsafe_obstacle_id")) if np.isfinite(debug_value(act_result, "unsafe_obstacle_id")) else -1,
        "unsafe_obstacle_distance": debug_value(act_result, "unsafe_obstacle_distance"),
        "unsafe_obstacle_tcpa": debug_value(act_result, "unsafe_obstacle_tcpa"),
        "unsafe_obstacle_cpa": debug_value(act_result, "unsafe_obstacle_cpa"),
        "stage": stage,
        "baseline_name": cfg.baseline_name,
        "config_name": cfg.config_name,
        "baseline_category": cfg.category,
    }
    return row


def obstacle_rows(cfg: BaselineConfig, stage: str, scenario: str, episode_id: int, episode_seed: int, info: dict[str, Any]) -> list[dict[str, Any]]:
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    cpas = np.asarray(info.get("planned_cpa_values", []), dtype=np.float32)
    ttcs = np.asarray(info.get("planned_ttc_values", []), dtype=np.float32)
    classes = list(info.get("threat_classes", []))
    modes = list(info.get("obstacle_motion_modes", []))
    rows: list[dict[str, Any]] = []
    for idx, pos in enumerate(positions):
        vel = velocities[idx] if idx < len(velocities) else np.zeros(3, dtype=np.float32)
        rel = pos - uav
        distance = float(np.linalg.norm(rel))
        closing = -float(np.dot(rel / (distance + 1e-8), vel - uav_vel))
        row = {
            "method": cfg.baseline_name,
            "policy_name": cfg.config_name,
            "scenario": scenario,
            "episode_id": int(episode_id),
            "episode_seed": int(episode_seed),
            "step": int(info["step"]),
            "time": float(info["time"]),
            "obstacle_slot": int(idx),
            "obstacle_id": int(ids[idx]) if idx < len(ids) else int(idx),
            "active": 1,
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "pos_z": float(pos[2]),
            "vel_x": float(vel[0]),
            "vel_y": float(vel[1]),
            "vel_z": float(vel[2]),
            "distance": distance,
            "closing": closing,
            "planned_cpa": float(cpas[idx]) if idx < len(cpas) else float("nan"),
            "planned_ttc": float(ttcs[idx]) if idx < len(ttcs) else float("nan"),
            "threat_class": str(classes[idx]) if idx < len(classes) else "none",
            "motion_mode": str(modes[idx]) if idx < len(modes) else "none",
            "risk_value": float(np.clip((3.0 - distance) / 3.0, 0.0, 1.0)),
            "stage": stage,
            "baseline_name": cfg.baseline_name,
            "config_name": cfg.config_name,
            "baseline_category": cfg.category,
        }
        rows.append(row)
    return rows


def run_episode(
    cfg: BaselineConfig,
    stage: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    checkpoint_step: int,
    rng: np.random.Generator,
    model: PPO | None,
    collect_trace: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = DynamicObstacleFlowEnv(scenario=scenario)
    obs, info = env.reset(seed=episode_seed)
    done = False
    steps = 0
    episode_return = 0.0
    last_info = info
    trace_rows: list[dict[str, Any]] = []
    step_obstacle_rows: list[dict[str, Any]] = []
    min_distances: list[float] = []
    action_norms: list[float] = []
    action_deltas: list[float] = []
    filter_deltas: list[float] = []
    filter_triggers = 0
    filter_triggered_any = False
    prev_action: np.ndarray | None = None
    raw_cpas: list[float] = []
    filtered_cpas: list[float] = []
    raw_ttcs: list[float] = []
    filtered_ttcs: list[float] = []
    threat_window_enter_time = float("nan")
    first_reaction_time = float("nan")
    no_response_steps = 0
    threat_active_steps = 0
    reaction_hits = 0
    min_distance_after_threat = float("inf")
    max_episode_time = env.max_steps * env.dt

    while not done:
        act_result = act(cfg, obs, info, env, rng, model)
        action_raw = np.asarray(act_result["action_raw"], dtype=np.float32)
        action_filtered = np.asarray(act_result["action_filtered"], dtype=np.float32)
        action_executed = np.clip(action_filtered, env.action_space.low, env.action_space.high).astype(np.float32)
        filter_delta = float(np.linalg.norm(action_filtered - action_raw))
        filter_deltas.append(filter_delta)
        filter_triggers += int(bool(act_result["filter_triggered"]))
        filter_triggered_any = filter_triggered_any or bool(act_result["filter_triggered"])
        raw_cpas.append(debug_value(act_result, "min_predicted_cpa_raw"))
        filtered_cpas.append(debug_value(act_result, "min_predicted_cpa_filtered"))
        raw_ttcs.append(debug_value(act_result, "min_ttc_raw"))
        filtered_ttcs.append(debug_value(act_result, "min_ttc_filtered"))
        action_norms.append(float(np.linalg.norm(action_executed)))
        if prev_action is not None:
            action_deltas.append(float(np.linalg.norm(action_executed - prev_action)))
        prev_action = action_executed.copy()

        obs, reward, terminated, truncated, info = env.step(action_executed)
        done = bool(terminated or truncated)
        steps += 1
        episode_return += float(reward)
        last_info = info
        min_distance = finite(info.get("min_distance"))
        min_distances.append(min_distance)
        time_now = finite(info.get("time"))
        ttc_remaining = finite(info.get("planned_ttc_remaining_to_threat"))
        threat_active = bool(np.isfinite(ttc_remaining) and 0.0 <= ttc_remaining <= 4.0)
        uav = np.asarray(info["uav_position"], dtype=np.float32)
        goal = np.asarray(info["goal_position"], dtype=np.float32)
        deviation = lateral_deviation(action_executed, goal, uav, env.v_uav_max)
        away_velocity = threat_away_velocity(info)
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
        if collect_trace:
            trace_rows.append(
                build_trace_row(
                    cfg,
                    stage,
                    scenario,
                    checkpoint_step,
                    episode_id,
                    episode_seed,
                    env,
                    info,
                    act_result,
                    done,
                    bool(terminated),
                    bool(truncated),
                    reaction_flag,
                    bool(threat_active and np.isnan(first_reaction_time)),
                    model,
                )
            )
            step_obstacle_rows.extend(obstacle_rows(cfg, stage, scenario, episode_id, episode_seed, info))

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
    failure_type = "none"
    if bool(last_info.get("is_collision")):
        failure_type = "collision"
    elif bool(last_info.get("truncated")):
        failure_type = "timeout"
    elif bool(last_info.get("near_miss")):
        failure_type = "near_miss"

    def mean_finite(values: list[float]) -> float:
        clean = [float(v) for v in values if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    def min_finite(values: list[float]) -> float:
        clean = [float(v) for v in values if np.isfinite(v)]
        return float(np.min(clean)) if clean else float("nan")

    row = {
        "stage": stage,
        "baseline_name": cfg.baseline_name,
        "config_name": cfg.config_name,
        "baseline_category": cfg.category,
        "config_params": json_params(cfg.params),
        "episode_filter_triggered": bool_int(filter_triggered_any),
        "mean_min_predicted_cpa_raw": mean_finite(raw_cpas),
        "mean_min_predicted_cpa_filtered": mean_finite(filtered_cpas),
        "min_min_predicted_cpa_raw": min_finite(raw_cpas),
        "min_min_predicted_cpa_filtered": min_finite(filtered_cpas),
        "mean_min_ttc_raw": mean_finite(raw_ttcs),
        "mean_min_ttc_filtered": mean_finite(filtered_ttcs),
        "min_min_ttc_raw": min_finite(raw_ttcs),
        "min_min_ttc_filtered": min_finite(filtered_ttcs),
        "failure_type": failure_type,
        "nan_or_crash": 0,
        "method": cfg.baseline_name,
        "policy_name": cfg.config_name,
        "scenario": scenario,
        "checkpoint_step": int(checkpoint_step) if cfg.kind in {"attention_full", "distance_filter", "cpa_ttc_filter", "vo_like_filter"} else -1,
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
        "mean_min_distance": float(np.mean(min_distances)) if min_distances else float("nan"),
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
        "filter_used": bool_int(cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"}),
        "filter_trigger_count": int(filter_triggers),
        "filter_trigger_rate": float(filter_triggers / max(steps, 1)),
        "mean_filter_delta_norm": float(np.mean(filter_deltas)) if filter_deltas else float("nan"),
        "max_filter_delta_norm": float(np.max(filter_deltas)) if filter_deltas else float("nan"),
    }
    return row, trace_rows, step_obstacle_rows


def make_baselines() -> list[BaselineConfig]:
    configs: list[BaselineConfig] = [
        BaselineConfig("random", "random", "reference", "random"),
        BaselineConfig("straight_line", "straight_line", "reference", "straight_line"),
        BaselineConfig("attention_full_1500k", "attention_full_1500k", "reference", "attention_full"),
        BaselineConfig(
            "current_cpa_reactive",
            "cpa_reactive_current",
            "reference,cpa_reactive_sweep",
            "cpa_reactive",
            {"d_reactive": 4.0, "horizon": 4.5, "cpa_trigger": 2.4, "avoid_weight": 2.1},
        ),
    ]
    for d0 in (3.0, 4.0, 5.0):
        for w_rep in (0.6, 1.0, 1.6):
            configs.append(BaselineConfig("naive_apf", f"naive_apf_d{d0:g}_w{w_rep:g}".replace(".", "p"), "apf", "naive_apf", {"d0": d0, "w_goal": 1.0, "w_rep": w_rep}))
    for alpha in (0.5, 1.0, 2.0):
        configs.append(BaselineConfig("velocity_aware_apf", f"velocity_aware_apf_alpha{alpha:g}".replace(".", "p"), "apf", "velocity_aware_apf", {"d0": 4.0, "w_goal": 1.0, "w_rep": 1.0, "alpha_closing": alpha}))
    for alpha in (1.0, 2.0, 3.0):
        configs.append(BaselineConfig("cpa_ttc_weighted_apf", f"cpa_ttc_weighted_apf_alpha{alpha:g}".replace(".", "p"), "apf", "cpa_ttc_weighted_apf", {"d0": 4.0, "w_goal": 1.0, "w_rep": 1.0, "alpha_closing": 1.0, "horizon": 4.5, "cpa_threshold": 2.4, "alpha_cpa": alpha}))
    current = {"d_reactive": 4.0, "horizon": 4.5, "cpa_trigger": 2.4, "avoid_weight": 2.1}
    variants = [
        ("cpa_reactive_d3", {"d_reactive": 3.0}),
        ("cpa_reactive_d5", {"d_reactive": 5.0}),
        ("cpa_reactive_cpa18", {"cpa_trigger": 1.8}),
        ("cpa_reactive_cpa30", {"cpa_trigger": 3.0}),
        ("cpa_reactive_h3", {"horizon": 3.0}),
        ("cpa_reactive_h6", {"horizon": 6.0}),
        ("cpa_reactive_w14", {"avoid_weight": 1.4}),
        ("cpa_reactive_w28", {"avoid_weight": 2.8}),
    ]
    for name, override in variants:
        params = dict(current)
        params.update(override)
        configs.append(BaselineConfig("cpa_reactive_sweep", name, "cpa_reactive_sweep", "cpa_reactive", params))
    for d_filter in (1.5, 2.0, 2.5):
        for beta in (0.5, 0.8):
            configs.append(BaselineConfig("distance_filter", f"distance_filter_d{d_filter:g}_beta{beta:g}".replace(".", "p"), "safety_filter", "distance_filter", {"d_filter": d_filter, "beta": beta}))
    for horizon in (3.0, 4.5):
        for cpa_safe in (1.2, 1.5, 2.0):
            configs.append(BaselineConfig("cpa_ttc_filter", f"cpa_ttc_filter_h{horizon:g}_cpa{cpa_safe:g}".replace(".", "p"), "safety_filter", "cpa_ttc_filter", {"horizon": horizon, "cpa_safe": cpa_safe, "beta": 0.8}))
    for cpa_safe in (1.2, 1.5, 2.0):
        configs.append(BaselineConfig("vo_like_filter", f"vo_like_filter_h45_cpa{cpa_safe:g}_h16".replace(".", "p"), "safety_filter", "vo_like_filter", {"horizon": 4.5, "cpa_safe": cpa_safe, "num_headings": 16}))
    return configs


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, float("nan")) for field in fields})


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, float("nan")) for field in fields})


def clear_flags(out_dir: Path) -> None:
    for flag in (COMPLETE_FLAG, *STOP_FLAGS):
        path = out_dir / flag
        if path.exists():
            path.unlink()


def write_status(out_dir: Path, message: str) -> None:
    (out_dir / "phase_b_status.txt").write_text(message + "\n", encoding="utf-8")


def write_stop(out_dir: Path, args: argparse.Namespace, reason: str, detail: str) -> None:
    flag = STOP_REASON_TO_FLAG.get(reason, "PHASE_B_STOP_EVAL_FAILED.flag")
    write_status(out_dir, f"stopped:{flag}")
    (out_dir / flag).write_text(f"terminal_decision=phase_b_stopped_{reason}\ndetail={detail}\n", encoding="utf-8")
    write_partial_report(out_dir, args, reason, detail)


def write_partial_report(out_dir: Path, args: argparse.Namespace, reason: str, detail: str) -> None:
    lines = [
        "# Phase B Geometry Filter Baseline Report",
        "",
        "Phase B not complete.",
        "",
        f"terminal_decision = phase_b_stopped_{reason}",
        f"stop_detail = {detail}",
        "",
        "## Partial Context",
        "",
        f"- out_dir: `{args.out_dir}`",
        f"- stage: `{args.stage}`",
        f"- checkpoint: `{args.checkpoint}`",
        "",
        "Phase C is blocked until this stop reason is resolved.",
    ]
    (out_dir / "PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace, out_dir: Path, logger: Logger) -> Path:
    if not PHASE_A_COMPLETE.exists():
        raise PhaseBStop("phase_a_missing", f"missing Phase A complete flag: {PHASE_A_COMPLETE}")
    if not PHASE_A_TRACE_SCHEMA.exists() or not PHASE_A_EPISODE_SCHEMA_SAMPLE.exists():
        raise PhaseBStop("phase_a_missing", "missing Phase A schema CSVs")
    if not PHASE_A_ENV_FREEZE.exists():
        raise PhaseBStop("phase_a_missing", "missing Phase A env freeze check")
    phase_a_freeze = pd.read_csv(PHASE_A_ENV_FREEZE)
    current_hash = sha256_file(ENV_FILE)
    phase_a_hash = str(phase_a_freeze.iloc[0]["sha256"])
    if current_hash != phase_a_hash:
        raise PhaseBStop("env_core_change_required", "EnvV2-core hash differs from Phase A freeze check")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise PhaseBStop("checkpoint_not_found", f"checkpoint not found: {checkpoint}")
    trace_schema = pd.read_csv(PHASE_A_TRACE_SCHEMA)
    phase_a_trace_fields = set(trace_schema.loc[trace_schema["table"] == "per_step_trace", "field"].astype(str))
    phase_a_obstacle_fields = set(trace_schema.loc[trace_schema["table"] == "step_obstacles_long", "field"].astype(str))
    if not phase_a_trace_fields.issubset(set(PHASE_B_TRACE_FIELDS)):
        raise PhaseBStop("schema_mismatch", "Phase B trace schema does not include all Phase A trace fields")
    if not phase_a_obstacle_fields.issubset(set(PHASE_B_OBSTACLE_FIELDS)):
        raise PhaseBStop("schema_mismatch", "Phase B obstacle schema does not include all Phase A obstacle fields")
    phase_a_episode_columns = set(pd.read_csv(PHASE_A_EPISODE_SCHEMA_SAMPLE, nrows=0).columns.astype(str))
    if not phase_a_episode_columns.issubset(set(EPISODE_FIELDS)):
        raise PhaseBStop("schema_mismatch", "Phase B episode schema does not include all Phase A episode fields")
    logger.log("PREFLIGHT_OK phase_a_complete=true env_hash_match=true checkpoint_exists=true schema_compatible=true")
    return checkpoint


def load_model(checkpoint: Path, args: argparse.Namespace, logger: Logger) -> PPO:
    try:
        logger.log(f"LOADING_CHECKPOINT path={checkpoint}")
        return PPO.load(str(checkpoint), device=args.device, custom_objects={"policy_kwargs": policy_kwargs(args.hidden_dim)})
    except Exception as exc:
        raise PhaseBStop("checkpoint_not_found", f"checkpoint exists but could not be loaded: {exc!r}") from exc


def derive_checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"step(\d+)", checkpoint.name)
    return int(match.group(1)) if match else 1500000


def manifest_rows(configs: list[BaselineConfig]) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        rows.append(
            {
                "baseline_name": cfg.baseline_name,
                "config_name": cfg.config_name,
                "baseline_category": cfg.category,
                "kind": cfg.kind,
                "config_params": json_params(cfg.params),
                "implemented": 1,
                "uses_attention_checkpoint": int(cfg.kind in {"attention_full", "distance_filter", "cpa_ttc_filter", "vo_like_filter"}),
                "filter_used": int(cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"}),
            }
        )
    return rows


def command_manifest_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    runner = [
        "python",
        "scripts/run_env_v2_phase_b_geometry_filter_baselines.py",
        "--out-dir",
        args.out_dir,
        "--checkpoint",
        args.checkpoint,
        "--eval-seed",
        str(args.eval_seed),
        "--stage",
        args.stage,
    ]
    if args.write_traces:
        runner.append("--write-traces")
    return [
        {"command_name": "phase_b_runner", "command": " ".join(runner), "status": "started"},
        {"command_name": "phase_b_analysis", "command": f"python scripts/analyze_env_v2_phase_b_results.py --result-dir {args.out_dir}", "status": "pending"},
        {"command_name": "phase_b_watcher", "command": "bash scripts/watch_phase_b_geometry_filter_baselines.sh", "status": "available"},
    ]


def schema_check_rows() -> list[dict[str, Any]]:
    rows = []
    for field_name in PHASE_A_EPISODE_FIELDS:
        rows.append({"schema": "episode_metrics", "field": field_name, "source": "phase_a", "phase_b_status": "preserved"})
    for field_name in PHASE_B_EXTRA_EPISODE_FIELDS:
        rows.append({"schema": "episode_metrics", "field": field_name, "source": "phase_b", "phase_b_status": "added"})
    for field_name in TRACE_FIELDS:
        rows.append({"schema": "trace", "field": field_name, "source": "phase_a", "phase_b_status": "preserved"})
    for field_name in OBSTACLE_FIELDS:
        rows.append({"schema": "step_obstacles", "field": field_name, "source": "phase_a", "phase_b_status": "preserved"})
    return rows


def summarize_for_top_configs(out_dir: Path) -> pd.DataFrame:
    episode_path = out_dir / "tables/phase_b_episode_metrics.csv"
    df = pd.read_csv(episode_path)
    b1 = df[df["stage"] == "b1_coarse"].copy()
    if b1.empty:
        raise PhaseBStop("eval_failed", "B1 coarse data missing; cannot select B2 configs")
    summary = (
        b1.groupby(["baseline_name", "config_name", "baseline_category"], dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            near_miss_rate=("near_miss", "mean"),
            progress_mean=("progress", "mean"),
            episode_min_distance_mean=("episode_min_distance", "mean"),
            mean_action_delta=("mean_action_delta", "mean"),
            filter_trigger_rate=("filter_trigger_rate", "mean"),
        )
        .reset_index()
    )
    summary["rank_score"] = (
        summary["success_rate"]
        - 2.0 * summary["collision_rate"]
        - 0.5 * summary["near_miss_rate"]
        + 0.2 * summary["progress_mean"]
    )
    return summary.sort_values("rank_score", ascending=False)


def select_b2_configs(all_configs: list[BaselineConfig], out_dir: Path) -> list[BaselineConfig]:
    by_name = {cfg.config_name: cfg for cfg in all_configs}
    summary = summarize_for_top_configs(out_dir)
    selected_names = ["random", "straight_line", "attention_full_1500k", "cpa_reactive_current"]
    cpa = summary[
        summary["baseline_category"].astype(str).str.contains("cpa_reactive_sweep")
        & (summary["config_name"] != "cpa_reactive_current")
    ].head(3)
    apf = summary[summary["baseline_category"].astype(str).str.contains("apf")].head(2)
    filters = summary[summary["baseline_category"].astype(str).str.contains("safety_filter")].head(2)
    selected_names.extend(cpa["config_name"].tolist())
    selected_names.extend(apf["config_name"].tolist())
    selected_names.extend(filters["config_name"].tolist())
    deduped: list[str] = []
    for name in selected_names:
        if name not in deduped:
            deduped.append(name)
    rows = []
    for rank, name in enumerate(deduped, start=1):
        row = summary[summary["config_name"] == name]
        rows.append(
            {
                "selection_rank": rank,
                "config_name": name,
                "baseline_name": by_name[name].baseline_name,
                "baseline_category": by_name[name].category,
                "selected_for_b2": 1,
                "b1_rank_score": float(row["rank_score"].iloc[0]) if not row.empty else float("nan"),
                "b1_success_rate": float(row["success_rate"].iloc[0]) if not row.empty else float("nan"),
                "b1_collision_rate": float(row["collision_rate"].iloc[0]) if not row.empty else float("nan"),
            }
        )
    write_csv(out_dir / "tables/phase_b_top_configs.csv", rows, list(rows[0].keys()))
    return [by_name[name] for name in deduped]


def should_collect_trace(stage: str, cfg: BaselineConfig, scenario: str, episode_id: int) -> bool:
    if stage == "b0_smoke":
        return episode_id == 0 and scenario in SMOKE_SCENARIOS
    if stage == "b1_coarse":
        return episode_id == 0
    if stage == "b2_formal":
        return episode_id in {0, 1}
    return False


def trace_path(out_dir: Path, stage: str, cfg: BaselineConfig, scenario: str, episode_id: int, failure_type: str, filter_triggered: int) -> Path:
    sample_dir = out_dir / "traces/sample_traces"
    failure_dir = out_dir / "traces/failure_traces"
    formal_dir = out_dir / "traces/formal_traces"
    safe_name = cfg.config_name.replace("/", "_")
    if failure_type == "collision":
        return failure_dir / f"{stage}_{safe_name}_{scenario}_ep{episode_id}_collision_trace.csv"
    if filter_triggered and cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"}:
        return sample_dir / f"{stage}_{safe_name}_{scenario}_ep{episode_id}_filter_triggered_trace.csv"
    if stage == "b2_formal":
        return formal_dir / f"{stage}_{safe_name}_{scenario}_ep{episode_id}_trace.csv"
    return sample_dir / f"{stage}_{safe_name}_{scenario}_ep{episode_id}_trace.csv"


def run_stage(
    *,
    stage: str,
    configs: list[BaselineConfig],
    scenarios: list[str],
    episodes: int,
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_step: int,
    model: PPO,
    logger: Logger,
) -> None:
    total = len(configs) * len(scenarios) * episodes
    completed = 0
    start = time.time()
    last_heartbeat = start
    logger.log(f"PHASE_B_STAGE_START stage={stage} configs={len(configs)} scenarios={len(scenarios)} episodes={episodes} total={total}")
    for scenario_index, scenario in enumerate(scenarios):
        for cfg_index, cfg in enumerate(configs):
            first_collision_saved = False
            first_filter_saved = False
            for episode_id in range(episodes):
                episode_seed = args.eval_seed + args.seed * 10000 + episode_id
                rng = np.random.default_rng(args.eval_seed + args.seed * 10000 + scenario_index * 1000 + cfg_index * 100000 + episode_id)
                collect_trace = bool(args.write_traces and should_collect_trace(stage, cfg, scenario, episode_id))
                try:
                    row, rows_trace, rows_obstacles = run_episode(
                        cfg,
                        stage,
                        scenario,
                        episode_id,
                        episode_seed,
                        checkpoint_step,
                        rng,
                        model,
                        collect_trace=True if args.write_traces else False,
                    )
                except PhaseBStop:
                    raise
                except Exception as exc:
                    raise PhaseBStop("eval_failed", f"rollout failed stage={stage} config={cfg.config_name} scenario={scenario} episode={episode_id}: {exc!r}") from exc

                append_csv(out_dir / "tables/phase_b_episode_metrics.csv", [row], EPISODE_FIELDS)
                save_trace = collect_trace
                if args.write_traces and row["failure_type"] == "collision" and not first_collision_saved:
                    save_trace = True
                    first_collision_saved = True
                if args.write_traces and row["episode_filter_triggered"] and cfg.kind in {"distance_filter", "cpa_ttc_filter", "vo_like_filter"} and not first_filter_saved:
                    save_trace = True
                    first_filter_saved = True
                if save_trace and rows_trace:
                    path = trace_path(out_dir, stage, cfg, scenario, episode_id, str(row["failure_type"]), int(row["episode_filter_triggered"]))
                    write_csv(path, rows_trace, PHASE_B_TRACE_FIELDS)
                    append_csv(out_dir / "tables/phase_b_step_obstacles_sample.csv", rows_obstacles, PHASE_B_OBSTACLE_FIELDS)
                completed += 1
                now = time.time()
                if now - last_heartbeat >= args.heartbeat_seconds or completed == total:
                    elapsed = now - start
                    rate = completed / max(elapsed, 1e-6)
                    eta = (total - completed) / max(rate, 1e-6)
                    logger.log(
                        f"PHASE_B_STAGE_HEARTBEAT stage={stage} completed={completed}/{total} "
                        f"config={cfg.config_name} scenario={scenario} rate={rate:.2f}_eps_sec eta_min={eta/60.0:.1f}"
                    )
                    write_status(out_dir, f"running:{stage}:completed={completed}/{total}")
                    last_heartbeat = now
    logger.log(f"PHASE_B_STAGE_DONE stage={stage} completed={completed}/{total}")


def validate_stage_outputs(out_dir: Path, stage: str) -> None:
    path = out_dir / "tables/phase_b_episode_metrics.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise PhaseBStop("eval_failed", f"episode metrics missing after {stage}")
    df = pd.read_csv(path)
    if df[df["stage"] == stage].empty:
        raise PhaseBStop("eval_failed", f"no rows written for stage {stage}")
    missing = [field for field in PHASE_A_EPISODE_FIELDS if field not in df.columns]
    if missing:
        raise PhaseBStop("schema_mismatch", f"episode metrics missing Phase A fields: {missing}")


def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    for sub in ["tables", "plots", "traces/sample_traces", "traces/failure_traces", "traces/formal_traces", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    clear_flags(out_dir)
    write_status(out_dir, "running:preflight")
    logger = Logger(out_dir)
    logger.log("PHASE_B_GEOMETRY_FILTER_BASELINE_RUNNER_START")
    checkpoint: Path | None = None
    try:
        checkpoint = preflight(args, out_dir, logger)
        model = load_model(checkpoint, args, logger)
        checkpoint_step = derive_checkpoint_step(checkpoint)
        configs = make_baselines()
        write_csv(out_dir / "tables/phase_b_baseline_manifest.csv", manifest_rows(configs), ["baseline_name", "config_name", "baseline_category", "kind", "config_params", "implemented", "uses_attention_checkpoint", "filter_used"])
        write_csv(out_dir / "tables/phase_b_command_manifest.csv", command_manifest_rows(args), ["command_name", "command", "status"])
        write_csv(out_dir / "tables/phase_b_schema_check.csv", schema_check_rows(), ["schema", "field", "source", "phase_b_status"])
        if args.stage == "smoke":
            run_stage(
                stage="b0_smoke",
                configs=configs,
                scenarios=args.scenarios or SMOKE_SCENARIOS,
                episodes=args.num_episodes,
                args=args,
                out_dir=out_dir,
                checkpoint_step=checkpoint_step,
                model=model,
                logger=logger,
            )
            validate_stage_outputs(out_dir, "b0_smoke")
            write_status(out_dir, "smoke_complete")
            logger.log("PHASE_B_SMOKE_COMPLETE")
            return
        if args.stage != "full":
            raise PhaseBStop("baseline_impl_failed", f"unsupported --stage: {args.stage}")

        run_stage(
            stage="b0_smoke",
            configs=configs,
            scenarios=SMOKE_SCENARIOS,
            episodes=args.b0_episodes,
            args=args,
            out_dir=out_dir,
            checkpoint_step=checkpoint_step,
            model=model,
            logger=logger,
        )
        validate_stage_outputs(out_dir, "b0_smoke")
        run_stage(
            stage="b1_coarse",
            configs=configs,
            scenarios=ALL_SCENARIOS,
            episodes=args.b1_episodes,
            args=args,
            out_dir=out_dir,
            checkpoint_step=checkpoint_step,
            model=model,
            logger=logger,
        )
        validate_stage_outputs(out_dir, "b1_coarse")
        selected = select_b2_configs(configs, out_dir)
        logger.log("PHASE_B_B2_SELECTED configs=" + ",".join(cfg.config_name for cfg in selected))
        run_stage(
            stage="b2_formal",
            configs=selected,
            scenarios=ALL_SCENARIOS,
            episodes=args.b2_episodes,
            args=args,
            out_dir=out_dir,
            checkpoint_step=checkpoint_step,
            model=model,
            logger=logger,
        )
        validate_stage_outputs(out_dir, "b2_formal")
        write_status(out_dir, "runner_complete:analysis_pending")
        logger.log("PHASE_B_GEOMETRY_FILTER_BASELINE_RUNNER_COMPLETE")
    except PhaseBStop as exc:
        write_stop(out_dir, args, exc.reason, exc.detail)
        logger.log(f"PHASE_B_STOP reason={exc.reason} detail={exc.detail}")
        raise SystemExit(2) from None
    except Exception as exc:
        detail = f"unexpected exception: {exc!r}"
        logger.log(detail)
        logger.log(traceback.format_exc())
        write_stop(out_dir, args, "eval_failed", detail)
        raise SystemExit(2) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/env_v2_phase_b_geometry_filter_baselines")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--b0-episodes", type=int, default=3)
    parser.add_argument("--b1-episodes", type=int, default=20)
    parser.add_argument("--b2-episodes", type=int, default=50)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.stage == "smoke" and args.num_episodes <= 0:
        raise argparse.ArgumentTypeError("--num-episodes must be positive")
    for name in ("b0_episodes", "b1_episodes", "b2_episodes"):
        if getattr(args, name) <= 0:
            raise argparse.ArgumentTypeError(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
