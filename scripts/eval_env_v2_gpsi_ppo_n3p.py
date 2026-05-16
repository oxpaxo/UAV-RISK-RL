from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_env_v2_gpsi_ppo_n3fz as base_eval
from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from envs.wrappers.gpsi_obs_wrapper import GpsiObsWrapper


STOP_FLAGS = {
    "gpsi_checkpoint_missing": "PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "baseline_artifacts_missing": "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag",
    "config_invalid": "PHASE_N3P_STOP_CONFIG_INVALID.flag",
    "train_failed": "PHASE_N3P_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3P_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag",
    "feature_scale_invalid": "PHASE_N3P_STOP_FEATURE_SCALE_INVALID.flag",
    "watcher_failed": "PHASE_N3P_STOP_WATCHER_FAILED.flag",
}

DEFAULT_CONFIGS = {
    "obs_delta_only": {
        "config": "configs/env_v2_gpsi_heada_ppo_n3p_obs_delta_only.yaml",
        "checkpoint_dir": "checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0",
    },
    "logvar_scaled": {
        "config": "configs/env_v2_gpsi_heada_ppo_n3p_logvar_scaled.yaml",
        "checkpoint_dir": "checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0",
    },
    "block_projected": {
        "config": "configs/env_v2_gpsi_heada_ppo_n3p_block_projected.yaml",
        "checkpoint_dir": "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0",
    },
}


class PhaseN3PEvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase N3P no-z representation ablations.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3p_noz_representation_ablation")
    parser.add_argument("--configs", nargs="+", default=["obs_delta_only", "logvar_scaled", "block_projected"])
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--near-miss-distance", type=float, default=1.5)
    parser.add_argument("--raw-cpa-horizon", type=float, default=4.5)
    parser.add_argument("--raw-cpa-threshold", type=float, default=1.2)
    parser.add_argument("--raw-cpa-safe-threshold", type=float, default=1.5)
    parser.add_argument("--no-response-action-norm", type=float, default=0.05)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(result_dir: Path) -> None:
    for path in [
        result_dir,
        result_dir / "logs",
        result_dir / "tables",
        result_dir / "plots",
        result_dir / "traces",
        result_dir / "traces/sampled_success_traces",
        result_dir / "traces/sampled_collision_traces",
        result_dir / "traces/sampled_near_miss_traces",
        result_dir / "traces/high_raw_unsafe_rate_traces",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    write_csv(path, old + rows)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    ensure_dirs(result_dir)
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3p_status.txt", f"stopped:{flag_name}\n")
    write_text(
        result_dir / "PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3P No-Z Representation Ablation Report",
                "",
                f"`terminal_decision = phase_n3p_stopped_{reason}`",
                "",
                "Partial report generated by the N3P eval script.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "can_enter_N4: no",
            ]
        )
        + "\n",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PhaseN3PEvalStop("config_invalid", f"missing config: {rel(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PhaseN3PEvalStop("config_invalid", f"config is not a mapping: {rel(path)}")
    return payload


def json_array(value: Any) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def make_gpsi_env(cfg: dict[str, Any], scenario: str, args: argparse.Namespace) -> GpsiObsWrapper:
    gpsi_cfg = cfg.get("gpsi", {})
    checkpoint = ROOT / str(gpsi_cfg.get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    if not checkpoint.exists():
        raise PhaseN3PEvalStop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")
    wrapped = GpsiObsWrapper(
        DynamicObstacleFlowEnv(scenario=scenario),
        gpsi_checkpoint=checkpoint,
        device=args.device,
        history_steps=int(gpsi_cfg.get("history_steps", 20)),
        delta_scale=float(gpsi_cfg.get("delta_scale", 5.0)),
        logvar_clamp=tuple(gpsi_cfg.get("logvar_clamp", [-5.0, 3.0])),
        normalize_z=bool(gpsi_cfg.get("normalize_z", False)),
        z_stats_path=(ROOT / str(gpsi_cfg.get("z_stats_path"))) if gpsi_cfg.get("z_stats_path") else None,
        z_std_floor=float(gpsi_cfg.get("z_std_floor", 1.0e-3)),
        include_z=bool(gpsi_cfg.get("include_z", False)),
        z_transform=gpsi_cfg.get("z_transform"),
        z_l2_target_norm=float(gpsi_cfg.get("z_l2_target_norm", 4.0)),
        z_l2_eps=float(gpsi_cfg.get("z_l2_eps", 1.0e-6)),
        z_layernorm_alpha=float(gpsi_cfg.get("z_layernorm_alpha", 0.5)),
        z_layernorm_eps=float(gpsi_cfg.get("z_layernorm_eps", 1.0e-5)),
        include_logvar=bool(gpsi_cfg.get("include_logvar", True)),
        logvar_output_scale=float(gpsi_cfg.get("logvar_output_scale", 1.0)),
        degenerate_std_threshold=float(gpsi_cfg.get("degenerate_std_threshold", 1.0e-5)),
        degenerate_std_floor=float(gpsi_cfg.get("degenerate_std_floor", 1.0)),
    )
    freeze = wrapped.freeze_check
    if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
        raise PhaseN3PEvalStop("diagnostics_failed", f"Gpsi freeze check failed in eval: {freeze}")
    expected = int(gpsi_cfg.get("obs_aug_dim", 30))
    actual = int(wrapped.observation_space["obs"].shape[-1])
    if actual != expected:
        raise PhaseN3PEvalStop("config_invalid", f"expected aug obs dim {expected}, got {actual}")
    return wrapped


def flow_env_from(env: DynamicObstacleFlowEnv | GpsiObsWrapper) -> DynamicObstacleFlowEnv:
    return env.env if isinstance(env, GpsiObsWrapper) else env


def feature_key(prefix: dict[str, Any], info: dict[str, Any]) -> tuple[Any, ...]:
    return (
        prefix["method_key"],
        prefix["method"],
        prefix["checkpoint"],
        int(prefix["checkpoint_step"]),
        prefix["checkpoint_label"],
        prefix["scenario"],
        str(info.get("threat_motion_mode", "none")),
        str(info.get("threat_class", "none")),
    )


def add_adapter_output(
    *,
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
    model: PPO,
    cfg: dict[str, Any],
    env: GpsiObsWrapper,
    info: dict[str, Any],
    prefix: dict[str, Any],
) -> None:
    if str(cfg.get("method_key", "")) != "block_projected":
        return
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    adapter = getattr(extractor, "latest_adapter_output", None)
    if adapter is None:
        return
    arr = adapter.detach().cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] < 1:
        return
    debug = env.latest_gpsi_debug
    active = np.asarray(debug.get("active_slots", []), dtype=np.int64)
    if active.size == 0:
        return
    key = feature_key(prefix, info)
    if key not in feature_accum:
        feature_accum[key] = defaultdict(base_eval.BlockAccumulator)
    feature_accum[key]["adapter_output_64"].add(arr[0, active])


def add_gpsi_diagnostics(
    *,
    accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
    cfg: dict[str, Any],
    env: GpsiObsWrapper,
    info: dict[str, Any],
    prefix: dict[str, Any],
) -> dict[str, Any]:
    debug = env.latest_gpsi_debug
    active = np.asarray(debug.get("active_slots", []), dtype=np.int64)
    key = feature_key(prefix, info)
    if key not in accum:
        accum[key] = defaultdict(base_eval.ScalarAccumulator)
    if key not in feature_accum:
        feature_accum[key] = defaultdict(base_eval.BlockAccumulator)
    group = accum[key]
    blocks = feature_accum[key]
    base_obs = np.asarray(debug.get("gpsi_input_obs_current", np.zeros((env.max_obs, 12))), dtype=np.float32)
    delta_raw = np.asarray(debug.get("delta_hat_raw", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    delta_scaled = np.asarray(debug.get("delta_hat_norm", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    logvar_raw = np.asarray(debug.get("logvar_hat", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    logvar_policy = np.asarray(debug.get("logvar_hat_policy", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    include_logvar = bool(cfg.get("gpsi", {}).get("include_logvar", True))
    if active.size:
        blocks["obs_i_12"].add(base_obs[active])
        blocks["delta_hat_9_after_scale"].add(delta_scaled[active].reshape(active.size, -1))
        blocks["logvar_raw_9_clamped"].add(logvar_raw[active].reshape(active.size, -1))
        if include_logvar:
            blocks["logvar_scaled_9_policy"].add(logvar_policy[active].reshape(active.size, -1))
        parts = [base_obs[active], delta_scaled[active].reshape(active.size, -1)]
        if include_logvar:
            parts.append(logvar_policy[active].reshape(active.size, -1))
        blocks["full_aug_obs"].add(np.concatenate(parts, axis=1))
        group["history_valid_ratio"].add(np.asarray(debug.get("history_valid_ratio", np.zeros(env.max_obs)), dtype=np.float32)[active])
        inactive_delta = delta_raw[[slot for slot in range(env.max_obs) if slot not in set(active.tolist())]]
        group["inactive_forwarded_count"].add(np.asarray([int(np.any(np.abs(inactive_delta) > 1.0e-8))], dtype=np.float32))
        for horizon_idx, suffix in enumerate(["1s", "2s", "4s"]):
            group[f"delta_norm_{suffix}"].add(np.linalg.norm(delta_raw[active, horizon_idx, :], axis=1))
            group[f"logvar_xy_{suffix}"].add(logvar_raw[active, horizon_idx, :2].reshape(-1))
    else:
        blocks["obs_i_12"].add(np.zeros((0, 12), dtype=np.float32))
        blocks["delta_hat_9_after_scale"].add(np.zeros((0, env.delta_dim), dtype=np.float32))
        blocks["logvar_raw_9_clamped"].add(np.zeros((0, env.logvar_dim), dtype=np.float32))
        blocks["full_aug_obs"].add(np.zeros((0, env.aug_obs_dim), dtype=np.float32))

    nearest_slot = -1
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    if len(positions):
        nearest_slot = int(np.argmin(np.linalg.norm(positions[:, :3] - uav[:3], axis=1)))
    if 0 <= nearest_slot < env.max_obs:
        rel = positions[nearest_slot] - uav
        radial = base_eval.normalize(rel[:2])
        velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
        uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
        relvel_xy = velocities[nearest_slot, :2] - uav_vel[:2] if nearest_slot < len(velocities) else np.zeros(2, dtype=np.float32)
        relvel_dir = base_eval.normalize(relvel_xy)
        sigma2_xy = np.exp(np.clip(logvar_raw[nearest_slot, 0, :2], -5.0, 3.0))
        group["projected_std_radial"].add(np.sqrt(np.sum((radial**2) * sigma2_xy)))
        group["projected_std_relvel"].add(np.sqrt(np.sum((relvel_dir**2) * sigma2_xy)))

    step_summary: dict[str, Any] = {}
    for suffix in ["1s", "2s", "4s"]:
        values = np.asarray(group[f"delta_norm_{suffix}"].values[-active.size:] if active.size else [], dtype=np.float32)
        step_summary[f"mean_delta_norm_{suffix}"] = float(np.mean(values)) if values.size else np.nan
    step_summary["gpsi_forward_ms"] = float(debug.get("gpsi_forward_ms", 0.0))
    step_summary["gpsi_forward_batch_size"] = int(debug.get("gpsi_forward_batch_size", 0))
    step_summary["aug_obs_dim"] = int(env.aug_obs_dim)
    step_summary["include_logvar"] = int(include_logvar)
    step_summary["logvar_output_scale"] = float(cfg.get("gpsi", {}).get("logvar_output_scale", 1.0))
    return step_summary


def evaluate_episode(
    *,
    model: PPO,
    env: DynamicObstacleFlowEnv | GpsiObsWrapper,
    cfg: dict[str, Any],
    method_key: str,
    method: str,
    checkpoint_name: str,
    checkpoint_step_value: int,
    checkpoint_label: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    args: argparse.Namespace,
    gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[tuple[Path, dict[str, Any]]]]:
    obs, info = env.reset(seed=episode_seed)
    done = False
    steps = 0
    episode_reward = 0.0
    min_distance_values: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    last_info = info
    last_action = np.zeros(3, dtype=np.float32)
    raw_unsafe_count = 0
    raw_safe_unsafe_count = 0
    action_norms: list[float] = []
    action_deltas: list[float] = []
    no_response_values: list[float] = []
    cpa_values: list[float] = []

    while not done:
        flow_env = flow_env_from(env)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        prefix = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint_name,
            "checkpoint_step": int(checkpoint_step_value),
            "checkpoint_label": checkpoint_label,
            "scenario": scenario,
        }
        if isinstance(env, GpsiObsWrapper):
            add_adapter_output(feature_accum=feature_accum, model=model, cfg=cfg, env=env, info=info, prefix=prefix)
        raw = base_eval.raw_action_diagnostics(action, info, flow_env, args)
        action_norm = float(np.linalg.norm(action))
        action_delta = float(np.linalg.norm(action - last_action))
        no_response = int(action_norm < float(args.no_response_action_norm))
        action_norms.append(action_norm)
        action_deltas.append(action_delta)
        no_response_values.append(float(no_response))
        cpa_values.append(float(raw["raw_min_predicted_cpa"]))
        raw_unsafe_count += int(raw["raw_unsafe_action"])
        raw_safe_unsafe_count += int(raw["raw_safe_margin_unsafe_action"])
        raw_rows.append(
            {
                **prefix,
                "episode_id": int(episode_id),
                "step": int(info.get("step", steps)),
                "time": float(info.get("time", steps * flow_env.dt)),
                "motion_mode": raw["raw_unsafe_motion_mode"],
                "threat_class": raw["raw_unsafe_threat_class"],
                "action_norm": action_norm,
                "action_delta": action_delta,
                "no_response": int(no_response),
                **raw,
            }
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        episode_reward += float(reward)
        min_distance = float(info["min_distance"])
        min_distance_values.append(min_distance)
        last_info = info

        weights, entropy = base_eval.attention_snapshot(model, flow_env.max_obs)
        threat_index = int(info.get("threat_obstacle_index", -1))
        threat_weight = float(weights[threat_index]) if 0 <= threat_index < len(weights) else float("nan")
        gpsi_step = add_gpsi_diagnostics(
            accum=gpsi_accum,
            feature_accum=feature_accum,
            cfg=cfg,
            env=env,
            info=info,
            prefix=prefix,
        )

        if args.write_traces:
            uav = np.asarray(info["uav_position"], dtype=np.float32)
            goal = np.asarray(info["goal_position"], dtype=np.float32)
            uav_vel = np.asarray(info["uav_velocity"], dtype=np.float32)
            trace_rows.append(
                {
                    **prefix,
                    "episode_id": int(episode_id),
                    "step": int(info["step"]),
                    "time": float(info["time"]),
                    "uav_pos": json_array(uav),
                    "uav_vel": json_array(uav_vel),
                    "action_raw": json_array(action),
                    "action_executed": json_array(action),
                    "filter_used": False,
                    "filter_triggered": False,
                    "filter_delta_norm": 0.0,
                    "goal_directed_velocity": float(np.dot(uav_vel, base_eval.normalize(goal - uav))),
                    "threat_obstacle_id": int(info.get("threat_obstacle_id", -1)),
                    "threat_obstacle_index": int(threat_index),
                    "threat_class": str(info.get("threat_class", "none")),
                    "threat_motion_mode": str(info.get("threat_motion_mode", "none")),
                    "planned_cpa": float(info.get("planned_cpa_to_threat", np.nan)),
                    "planned_ttc": float(info.get("planned_ttc_to_threat", np.nan)),
                    "planned_ttc_remaining": float(info.get("planned_ttc_remaining_to_threat", np.nan)),
                    "min_distance": min_distance,
                    "episode_min_distance_so_far": float(info.get("episode_min_distance", np.nan)),
                    "attention_weights": json_array(weights),
                    "attention_entropy": entropy,
                    "threat_obstacle_attention_weight": threat_weight,
                    "action_norm": action_norm,
                    "action_delta": action_delta,
                    "no_response": int(no_response),
                    **raw,
                    **gpsi_step,
                }
            )
        last_action = action.copy()

    near_miss = int(float(last_info["episode_min_distance"]) < args.near_miss_distance and not bool(last_info["is_collision"]))
    episode_row = {
        "method_key": method_key,
        "method": method,
        "checkpoint": checkpoint_name,
        "checkpoint_path": rel(ROOT / DEFAULT_CONFIGS[method_key]["checkpoint_dir"] / ("final.zip" if checkpoint_label == "final" else "best_by_eval.zip" if checkpoint_label == "best_by_eval" else f"checkpoint_{checkpoint_step_value // 1000}k.zip")),
        "checkpoint_step": int(checkpoint_step_value),
        "checkpoint_label": checkpoint_label,
        "seed": int(args.seed),
        "scenario": scenario,
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "success": int(bool(last_info["is_success"])),
        "collision": int(bool(last_info["is_collision"])),
        "near_miss": near_miss,
        "mean_min_distance": float(np.mean(min_distance_values)) if min_distance_values else np.nan,
        "episode_min_distance": float(last_info["episode_min_distance"]),
        "episode_length": int(steps),
        "progress": float(last_info["progress"]),
        "planned_cpa": float(last_info.get("planned_cpa_to_threat", np.nan)),
        "planned_ttc": float(last_info.get("planned_ttc_to_threat", np.nan)),
        "threat_class": str(last_info.get("threat_class", "none")),
        "threat_motion_mode": str(last_info.get("threat_motion_mode", "none")),
        "threat_valid_rate": float(last_info.get("threat_valid_rate", np.nan)),
        "replacement_count": int(last_info.get("replacement_count", 0)),
        "episode_reward": float(episode_reward),
        "raw_unsafe_action_rate": float(raw_unsafe_count / max(steps, 1)),
        "raw_safe_margin_unsafe_action_rate": float(raw_safe_unsafe_count / max(steps, 1)),
        "raw_min_predicted_cpa": float(np.nanmean(cpa_values)) if cpa_values else np.nan,
        "action_norm": float(np.mean(action_norms)) if action_norms else np.nan,
        "action_delta": float(np.mean(action_deltas)) if action_deltas else np.nan,
        "no_response_rate": float(np.mean(no_response_values)) if no_response_values else np.nan,
        "nan_or_crash": 0,
    }
    trace_files: list[tuple[Path, dict[str, Any]]] = []
    if args.write_traces and trace_rows:
        trace_path = ROOT / args.result_dir / "traces" / f"{method_key}_{checkpoint_label}_{scenario}_ep{episode_id}.csv"
        write_csv(trace_path, trace_rows)
        trace_files.append((trace_path, episode_row))
    return episode_row, raw_rows, trace_rows, trace_files


def checkpoint_specs(config_key: str, cfg: dict[str, Any]) -> list[tuple[str, Path, int, str]]:
    train_steps = int(cfg.get("training", {}).get("total_steps", 500_000))
    ckpt_dir = ROOT / DEFAULT_CONFIGS[config_key]["checkpoint_dir"]
    steps = [int(step) for step in cfg.get("training", {}).get("checkpoint_steps", [250_000, train_steps])]
    if train_steps not in steps:
        steps.append(train_steps)
    ckpts: list[tuple[str, Path, int, str]] = []
    seen: set[int] = set()
    for step in sorted(step for step in steps if 0 < step <= train_steps):
        if step in seen:
            continue
        seen.add(step)
        label = f"{step // 1000}k"
        ckpts.append((f"{config_key}_{label}", ckpt_dir / f"checkpoint_{step // 1000}k.zip", step, label))
    ckpts.append((f"{config_key}_final", ckpt_dir / "final.zip", train_steps, "final"))
    best = ckpt_dir / "best_by_eval.zip"
    if best.exists():
        ckpts.append((f"{config_key}_best_by_eval", best, train_steps, "best_by_eval"))
    return ckpts


def feature_summary_rows(
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
    cfg_by_method: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_order = [
        "obs_i_12",
        "delta_hat_9_after_scale",
        "logvar_raw_9_clamped",
        "logvar_scaled_9_policy",
        "full_aug_obs",
        "adapter_output_64",
    ]
    for key, groups in feature_accum.items():
        method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        cfg = cfg_by_method.get(str(method_key), {})
        gpsi_cfg = cfg.get("gpsi", {})
        ppo_cfg = cfg.get("ppo", {})
        prefix = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
            "include_logvar": int(bool(gpsi_cfg.get("include_logvar", True))),
            "logvar_output_scale": float(gpsi_cfg.get("logvar_output_scale", 1.0)),
            "feature_adapter": str(ppo_cfg.get("feature_adapter", "raw_concat")),
        }
        for block in block_order:
            not_applicable = (block == "logvar_scaled_9_policy" and not bool(gpsi_cfg.get("include_logvar", True))) or (
                block == "adapter_output_64" and str(ppo_cfg.get("feature_adapter", "")) != "block_projected_no_z"
            )
            if block in groups:
                rows.append(groups[block].row(prefix, block))
            else:
                rows.append(base_eval.BlockAccumulator().row(prefix, block, not_applicable=not_applicable))
    return rows


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    ensure_dirs(result_dir)
    try:
        append_csv(
            result_dir / "tables/phase_n3p_command_manifest.csv",
            [
                {
                    "stage": "eval",
                    "method_key": "all",
                    "method": "all",
                    "command": " ".join(["python", *sys.argv]),
                    "episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "eval_seed": int(args.eval_seed),
                }
            ],
        )
        cfg_by_method: dict[str, dict[str, Any]] = {}
        config_rows: list[dict[str, Any]] = []
        for key in args.configs:
            if key not in DEFAULT_CONFIGS:
                raise PhaseN3PEvalStop("config_invalid", f"unknown config key: {key}")
            cfg = load_yaml(ROOT / DEFAULT_CONFIGS[key]["config"])
            cfg_by_method[key] = cfg
            config_rows.append(
                {
                    "method_key": key,
                    "method": cfg.get("method_name", key),
                    "config": DEFAULT_CONFIGS[key]["config"],
                    "checkpoint_dir": DEFAULT_CONFIGS[key]["checkpoint_dir"],
                    "obs_aug_dim": int(cfg.get("gpsi", {}).get("obs_aug_dim", -1)),
                    "include_z": int(bool(cfg.get("gpsi", {}).get("include_z", False))),
                    "include_logvar": int(bool(cfg.get("gpsi", {}).get("include_logvar", True))),
                    "logvar_output_scale": float(cfg.get("gpsi", {}).get("logvar_output_scale", 1.0)),
                    "feature_adapter": str(cfg.get("ppo", {}).get("feature_adapter", "raw_concat")),
                    "train_steps": int(cfg.get("training", {}).get("total_steps", 500_000)),
                    "seed": int(cfg.get("training", {}).get("seed", 0)),
                    "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
                    "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
                }
            )
        write_csv(result_dir / "tables/phase_n3p_config_manifest.csv", config_rows)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        all_episode_rows: list[dict[str, Any]] = []
        all_raw_rows: list[dict[str, Any]] = []
        all_trace_files: list[tuple[Path, dict[str, Any]]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}

        jobs: list[tuple[str, str, dict[str, Any], str, Path, int, str]] = []
        for key in args.configs:
            cfg = cfg_by_method[key]
            for checkpoint_name, checkpoint_path, fallback_step, label in checkpoint_specs(key, cfg):
                if not checkpoint_path.exists():
                    raise PhaseN3PEvalStop("train_failed", f"missing checkpoint for {key}: {rel(checkpoint_path)}")
                jobs.append((key, str(cfg.get("method_name", key)), cfg, checkpoint_name, checkpoint_path, base_eval.checkpoint_step(checkpoint_path, fallback_step), label))

        for method_key, method, cfg, checkpoint_name, checkpoint_path, step_value, checkpoint_label in jobs:
            model = PPO.load(str(checkpoint_path), device=args.device)
            for scenario in args.scenarios:
                env = make_gpsi_env(cfg, scenario, args)
                start = time.time()
                last = start
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3P_EVAL_START "
                    f"method={method_key} checkpoint={checkpoint_label} scenario={scenario} episodes={args.num_episodes}",
                    flush=True,
                )
                for episode_id in range(args.num_episodes):
                    seed = int(args.eval_seed) + int(args.seed) * 10000 + episode_id
                    ep_row, raw_rows, _trace_rows, trace_files = evaluate_episode(
                        model=model,
                        env=env,
                        cfg=cfg,
                        method_key=method_key,
                        method=method,
                        checkpoint_name=checkpoint_name,
                        checkpoint_step_value=step_value,
                        checkpoint_label=checkpoint_label,
                        scenario=scenario,
                        episode_id=episode_id,
                        episode_seed=seed,
                        args=args,
                        gpsi_accum=gpsi_accum,
                        feature_accum=feature_accum,
                    )
                    all_episode_rows.append(ep_row)
                    all_raw_rows.extend(raw_rows)
                    all_trace_files.extend(trace_files)
                    now = time.time()
                    if now - last >= args.heartbeat_seconds or episode_id == args.num_episodes - 1:
                        done = episode_id + 1
                        rate = done / max(now - start, 1e-6)
                        eta = (args.num_episodes - done) / max(rate, 1e-6)
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3P_EVAL_HEARTBEAT "
                            f"method={method_key} checkpoint={checkpoint_label} scenario={scenario} "
                            f"episodes={done}/{args.num_episodes} rate={rate:.2f} ep/s eta={eta/60.0:.2f} min",
                            flush=True,
                        )
                        last = now
                env.close()

        checkpoint_summary = base_eval.summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "scenario"])
        write_csv(result_dir / "tables/phase_n3p_episode_metrics.csv", all_episode_rows)
        write_csv(result_dir / "tables/phase_n3p_checkpoint_eval_summary.csv", checkpoint_summary)
        write_csv(result_dir / "tables/phase_n3p_eval_summary.csv", checkpoint_summary)
        write_csv(result_dir / "tables/phase_n3p_scenario_breakdown.csv", checkpoint_summary)
        write_csv(
            result_dir / "tables/phase_n3p_motion_mode_breakdown.csv",
            base_eval.summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "threat_motion_mode"]),
        )
        write_csv(
            result_dir / "tables/phase_n3p_threat_class_breakdown.csv",
            base_eval.summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "threat_class"]),
        )
        write_csv(result_dir / "tables/phase_n3p_raw_unsafe_action_steps.csv", all_raw_rows)
        write_csv(result_dir / "tables/phase_n3p_raw_unsafe_action_summary.csv", base_eval.summarize_raw(all_raw_rows))
        write_csv(result_dir / "tables/phase_n3p_gpsi_output_summary.csv", base_eval.gpsi_summary_rows(gpsi_accum))
        write_csv(result_dir / "tables/phase_n3p_feature_block_stats.csv", feature_summary_rows(feature_accum, cfg_by_method))
        base_eval.copy_sample_traces(result_dir, all_trace_files)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3P_EVAL_END "
            f"episodes={len(all_episode_rows)} raw_steps={len(all_raw_rows)} gpsi_groups={len(gpsi_accum)} traces={len(all_trace_files)}",
            flush=True,
        )
    except PhaseN3PEvalStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "eval_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
