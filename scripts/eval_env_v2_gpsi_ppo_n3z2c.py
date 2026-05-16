from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
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

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from envs.wrappers.gpsi_obs_wrapper import GpsiObsWrapper
from policies.obstacle_set_extractor import ObstacleSetExtractor


STOP_FLAGS = {
    "gpsi_checkpoint_missing": "PHASE_N3Z2C_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "checkpoint_missing": "PHASE_N3Z2C_STOP_TRAIN_FAILED.flag",
    "gpsi_wrapper_failed": "PHASE_N3Z2C_STOP_GPSI_WRAPPER_FAILED.flag",
    "gpsi_not_frozen": "PHASE_N3Z2C_STOP_GPSI_NOT_FROZEN.flag",
    "schema_mismatch": "PHASE_N3Z2C_STOP_SCHEMA_MISMATCH.flag",
    "eval_failed": "PHASE_N3Z2C_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3Z2C_STOP_DIAGNOSTICS_FAILED.flag",
}

DEFAULT_CONFIGS = {
    "n3f_no_z_full": {
        "config": "configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml",
        "checkpoint_dir": "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0",
    },
    "z_layernorm_alpha_0p5_cont_1p5m": {
        "config": "configs/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_1p5m.yaml",
        "checkpoint_dir": "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0",
    },
}


class PhaseN3Z2CEvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class ScalarAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.nan_count = 0
        self.inf_count = 0

    def add(self, value: Any) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        self.nan_count += int(np.isnan(arr).sum())
        self.inf_count += int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        self.values.extend(float(v) for v in finite)

    def stats(self, prefix: str) -> dict[str, Any]:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.size == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mean": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_p95": np.nan,
                f"{prefix}_max": np.nan,
                f"{prefix}_min": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_nan_count": int(self.nan_count),
                f"{prefix}_inf_count": int(self.inf_count),
            }
        return {
            f"{prefix}_count": int(arr.size),
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_p95": float(np.percentile(arr, 95)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_nan_count": int(self.nan_count),
            f"{prefix}_inf_count": int(self.inf_count),
        }


class BlockAccumulator:
    def __init__(self) -> None:
        self.l2 = ScalarAccumulator()
        self.max_abs = ScalarAccumulator()
        self.element_count = 0
        self.nan_count = 0
        self.inf_count = 0

    def add(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        self.element_count += int(arr.size)
        self.nan_count += int(np.isnan(arr).sum())
        self.inf_count += int(np.isinf(arr).sum())
        if arr.size == 0:
            return
        reshaped = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(1, -1)
        self.l2.add(np.linalg.norm(reshaped, axis=1))
        self.max_abs.add(np.max(np.abs(reshaped), axis=1))

    def row(self, prefix: dict[str, Any], block: str, not_applicable: bool = False) -> dict[str, Any]:
        row = dict(prefix)
        row["block"] = block
        row["not_applicable"] = int(not_applicable)
        if not_applicable:
            row.update(
                {
                    "samples": 0,
                    "element_count": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                    "l2_norm_mean": np.nan,
                    "l2_norm_median": np.nan,
                    "l2_norm_p95": np.nan,
                    "l2_norm_max": np.nan,
                    "max_abs_mean": np.nan,
                    "max_abs_p95": np.nan,
                    "max_abs_max": np.nan,
                }
            )
            return row
        l2_stats = self.l2.stats("l2")
        max_stats = self.max_abs.stats("max_abs")
        row.update(
            {
                "samples": int(l2_stats["l2_count"]),
                "element_count": int(self.element_count),
                "nan_count": int(self.nan_count),
                "inf_count": int(self.inf_count),
                "l2_norm_mean": l2_stats["l2_mean"],
                "l2_norm_median": l2_stats["l2_median"],
                "l2_norm_p95": l2_stats["l2_p95"],
                "l2_norm_max": l2_stats["l2_max"],
                "max_abs_mean": max_stats["max_abs_mean"],
                "max_abs_p95": max_stats["max_abs_p95"],
                "max_abs_max": max_stats["max_abs_max"],
            }
        )
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase N3Z2C repaired Gpsi-PPO configs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3z2c_z2_continuation")
    parser.add_argument("--configs", nargs="+", default=["n3f_no_z_full", "z_layernorm_alpha_0p5_cont_1p5m"])
    parser.add_argument("--z2-checkpoint-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0")
    parser.add_argument("--noz-reference-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0")
    parser.add_argument("--attention-checkpoint", default="checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--near-miss-distance", type=float, default=1.5)
    parser.add_argument("--raw-cpa-horizon", type=float, default=4.5)
    parser.add_argument("--raw-cpa-threshold", type=float, default=1.2)
    parser.add_argument("--raw-cpa-safe-threshold", type=float, default=1.5)
    parser.add_argument("--no-response-action-norm", type=float, default=0.05)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
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


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    ensure_dirs(result_dir)
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3z2c_status.txt", f"stopped:{flag_name}\n")
    write_text(
        result_dir / "PHASE_N3Z2C_Z2_CONTINUATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3Z2C Z2 Continuation Report",
                "",
                f"`terminal_decision = phase_n3z2c_stopped_{reason}`",
                "",
                "Partial report generated by the N3Z2C eval script.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "Can enter N4: no.",
            ]
        )
        + "\n",
    )


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
                seen.add(key)
                fields.append(key)
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


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PhaseN3Z2CEvalStop("schema_mismatch", f"missing config: {rel(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PhaseN3Z2CEvalStop("schema_mismatch", f"config is not a mapping: {rel(path)}")
    return payload


def json_array(value: Any) -> str:
    arr = np.asarray(value)
    return json.dumps(arr.tolist(), separators=(",", ":"))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def policy_kwargs_attention(hidden_dim: int) -> dict[str, Any]:
    return {
        "features_extractor_class": ObstacleSetExtractor,
        "features_extractor_kwargs": {
            "agg_mode": "attention",
            "hidden_dim": hidden_dim,
            "beta": 1.0,
            "r_ref": 1.0,
            "use_rbar": False,
            "rbar_floor": 0.0,
            "use_risk_bias": False,
            "lambda_bias": 0.0,
        },
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation_fn": torch.nn.Tanh,
    }


def checkpoint_step(path: Path, fallback: int) -> int:
    patterns = [r"checkpoint_(\d+)k", r"step(\d+)", r"(\d+)k"]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            value = int(match.group(1))
            return value * 1000 if "k" in match.group(0) else value
    return int(fallback)


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


def raw_action_diagnostics(action: np.ndarray, info: dict[str, Any], env: DynamicObstacleFlowEnv, args: argparse.Namespace) -> dict[str, Any]:
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int64)
    modes = list(info.get("obstacle_motion_modes", []))
    classes = list(info.get("threat_classes", []))
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    v_cmd = np.asarray(action, dtype=np.float32) * float(env.v_uav_max)
    v_cmd[2] = 0.0
    best = {
        "raw_min_predicted_cpa": float("inf"),
        "raw_min_predicted_ttc": float("nan"),
        "raw_unsafe_action": 0,
        "raw_safe_margin_unsafe_action": 0,
        "raw_unsafe_obstacle_id": -1,
        "raw_unsafe_motion_mode": "none",
        "raw_unsafe_threat_class": "none",
    }
    for slot, pos in enumerate(positions):
        rel = np.asarray(pos[:2] - uav[:2], dtype=np.float32)
        obs_vel = velocities[slot] if slot < len(velocities) else np.zeros(3, dtype=np.float32)
        rel_vel_pred = np.asarray(obs_vel[:2] - v_cmd[:2], dtype=np.float32)
        denom = float(np.dot(rel_vel_pred, rel_vel_pred)) + 1e-8
        tcpa = float(np.clip(-float(np.dot(rel, rel_vel_pred)) / denom, 0.0, args.raw_cpa_horizon))
        cpa = float(np.linalg.norm(rel + rel_vel_pred * tcpa))
        if cpa < float(best["raw_min_predicted_cpa"]):
            best.update(
                {
                    "raw_min_predicted_cpa": cpa,
                    "raw_min_predicted_ttc": tcpa,
                    "raw_unsafe_obstacle_id": int(ids[slot]) if slot < len(ids) else int(slot),
                    "raw_unsafe_motion_mode": str(modes[slot]) if slot < len(modes) else "none",
                    "raw_unsafe_threat_class": str(classes[slot]) if slot < len(classes) else "none",
                }
            )
    if math.isfinite(float(best["raw_min_predicted_cpa"])):
        unsafe = 0.0 < float(best["raw_min_predicted_ttc"]) < args.raw_cpa_horizon and float(best["raw_min_predicted_cpa"]) < args.raw_cpa_threshold
        unsafe_safe = 0.0 < float(best["raw_min_predicted_ttc"]) < args.raw_cpa_horizon and float(best["raw_min_predicted_cpa"]) < args.raw_cpa_safe_threshold
        best["raw_unsafe_action"] = int(unsafe)
        best["raw_safe_margin_unsafe_action"] = int(unsafe_safe)
    else:
        best["raw_min_predicted_cpa"] = float("nan")
    return best


def make_gpsi_env(cfg: dict[str, Any], scenario: str, args: argparse.Namespace) -> GpsiObsWrapper:
    gpsi_cfg = cfg.get("gpsi", {})
    checkpoint = ROOT / str(gpsi_cfg.get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    if not checkpoint.exists():
        raise PhaseN3Z2CEvalStop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")
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
        include_z=bool(gpsi_cfg.get("include_z", True)),
        z_transform=gpsi_cfg.get("z_transform"),
        z_l2_target_norm=float(gpsi_cfg.get("z_l2_target_norm", 4.0)),
        z_l2_eps=float(gpsi_cfg.get("z_l2_eps", 1.0e-6)),
        z_layernorm_alpha=float(gpsi_cfg.get("z_layernorm_alpha", 0.5)),
        z_layernorm_eps=float(gpsi_cfg.get("z_layernorm_eps", 1.0e-5)),
        degenerate_std_threshold=float(gpsi_cfg.get("degenerate_std_threshold", 1.0e-5)),
        degenerate_std_floor=float(gpsi_cfg.get("degenerate_std_floor", 1.0)),
    )
    freeze = wrapped.freeze_check
    if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
        raise PhaseN3Z2CEvalStop("gpsi_not_frozen", f"Gpsi freeze check failed in eval: {freeze}")
    expected = int(gpsi_cfg.get("obs_aug_dim", 94))
    actual = int(wrapped.observation_space["obs"].shape[-1])
    if actual != expected:
        raise PhaseN3Z2CEvalStop("schema_mismatch", f"expected aug obs dim {expected}, got {actual}")
    return wrapped


def flow_env_from(env: DynamicObstacleFlowEnv | GpsiObsWrapper) -> DynamicObstacleFlowEnv:
    return env.env if isinstance(env, GpsiObsWrapper) else env


def add_gpsi_diagnostics(
    *,
    accum: dict[tuple[Any, ...], dict[str, ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, BlockAccumulator]],
    cfg: dict[str, Any],
    env: GpsiObsWrapper,
    info: dict[str, Any],
    prefix: dict[str, Any],
) -> dict[str, Any]:
    debug = env.latest_gpsi_debug
    active = np.asarray(debug.get("active_slots", []), dtype=np.int64)
    key = (
        prefix["method_key"],
        prefix["method"],
        prefix["checkpoint"],
        int(prefix["checkpoint_step"]),
        prefix["checkpoint_label"],
        prefix["scenario"],
        str(info.get("threat_motion_mode", "none")),
        str(info.get("threat_class", "none")),
    )
    if key not in accum:
        accum[key] = defaultdict(ScalarAccumulator)
    if key not in feature_accum:
        feature_accum[key] = defaultdict(BlockAccumulator)
    group = accum[key]
    blocks = feature_accum[key]
    base_obs = np.asarray(debug.get("gpsi_input_obs_current", np.zeros((env.max_obs, 12))), dtype=np.float32)
    z_raw = np.asarray(debug.get("z_raw", np.zeros((env.max_obs, env.z_dim))), dtype=np.float32)
    z_after = np.asarray(debug.get("z_after_norm", np.zeros((env.max_obs, env.z_dim))), dtype=np.float32)
    delta_raw = np.asarray(debug.get("delta_hat_raw", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    delta_scaled = np.asarray(debug.get("delta_hat_norm", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    logvar = np.asarray(debug.get("logvar_hat", np.zeros((env.max_obs, env.num_horizons, env.state_dim))), dtype=np.float32)
    full = np.asarray(env.latest_gpsi_debug.get("aug_obs_dim", 0))
    del full
    obs_aug = None
    include_z = bool(cfg.get("gpsi", {}).get("include_z", True))
    if active.size:
        blocks["obs_i_12"].add(base_obs[active])
        blocks["z_i_64_raw"].add(z_raw[active])
        if include_z:
            blocks["z_i_64_after_constraint"].add(z_after[active])
        blocks["delta_hat_9_after_scale"].add(delta_scaled[active].reshape(active.size, -1))
        blocks["logvar_hat_9_clamped"].add(logvar[active].reshape(active.size, -1))
        block_parts = [base_obs[active]]
        if include_z:
            block_parts.append(z_after[active])
        block_parts.extend([delta_scaled[active].reshape(active.size, -1), logvar[active].reshape(active.size, -1)])
        obs_aug = np.concatenate(block_parts, axis=1)
        blocks["full_aug_obs"].add(obs_aug)
        group["z_norm_raw"].add(np.linalg.norm(z_raw[active], axis=1))
        group["z_norm_after"].add(np.linalg.norm(z_after[active], axis=1))
        group["z_zero_norm_count"].add(np.asarray([float(debug.get("z_zero_norm_count", 0))], dtype=np.float32))
        group["history_valid_ratio"].add(np.asarray(debug.get("history_valid_ratio", np.zeros(env.max_obs)), dtype=np.float32)[active])
        inactive_delta = delta_raw[[slot for slot in range(env.max_obs) if slot not in set(active.tolist())]]
        group["inactive_forwarded_count"].add(np.asarray([int(np.any(np.abs(inactive_delta) > 1.0e-8))], dtype=np.float32))
        for horizon_idx, suffix in enumerate(["1s", "2s", "4s"]):
            group[f"delta_norm_{suffix}"].add(np.linalg.norm(delta_raw[active, horizon_idx, :], axis=1))
            group[f"logvar_xy_{suffix}"].add(logvar[active, horizon_idx, :2].reshape(-1))
    else:
        blocks["obs_i_12"].add(np.zeros((0, 12), dtype=np.float32))
        blocks["z_i_64_raw"].add(np.zeros((0, env.z_dim), dtype=np.float32))
        blocks["delta_hat_9_after_scale"].add(np.zeros((0, env.delta_dim), dtype=np.float32))
        blocks["logvar_hat_9_clamped"].add(np.zeros((0, env.logvar_dim), dtype=np.float32))
        blocks["full_aug_obs"].add(np.zeros((0, env.aug_obs_dim), dtype=np.float32))

    nearest_slot = -1
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    if len(positions):
        nearest_slot = int(np.argmin(np.linalg.norm(positions[:, :3] - uav[:3], axis=1)))
    if 0 <= nearest_slot < env.max_obs:
        rel = positions[nearest_slot] - uav
        radial = normalize(rel[:2])
        velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
        uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
        relvel_xy = velocities[nearest_slot, :2] - uav_vel[:2] if nearest_slot < len(velocities) else np.zeros(2, dtype=np.float32)
        relvel_dir = normalize(relvel_xy)
        sigma2_xy = np.exp(np.clip(logvar[nearest_slot, 0, :2], -5.0, 3.0))
        group["projected_std_radial"].add(np.sqrt(np.sum((radial**2) * sigma2_xy)))
        group["projected_std_relvel"].add(np.sqrt(np.sum((relvel_dir**2) * sigma2_xy)))

    step_summary: dict[str, Any] = {}
    for suffix in ["1s", "2s", "4s"]:
        values = np.asarray(group[f"delta_norm_{suffix}"].values[-active.size:] if active.size else [], dtype=np.float32)
        step_summary[f"mean_delta_norm_{suffix}"] = float(np.mean(values)) if values.size else np.nan
    step_summary["gpsi_forward_ms"] = float(debug.get("gpsi_forward_ms", 0.0))
    step_summary["gpsi_forward_batch_size"] = int(debug.get("gpsi_forward_batch_size", 0))
    step_summary["aug_obs_dim"] = int(env.aug_obs_dim)
    step_summary["include_z"] = int(include_z)
    step_summary["normalize_z"] = int(bool(cfg.get("gpsi", {}).get("normalize_z", False)))
    step_summary["z_transform"] = str(cfg.get("gpsi", {}).get("z_transform", "standardize" if cfg.get("gpsi", {}).get("normalize_z", False) else "raw"))
    return step_summary


def write_trace(trace_rows: list[dict[str, Any]], path: Path) -> None:
    if trace_rows:
        write_csv(path, trace_rows)


def summarize_episode_rows(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    import pandas as pd

    df = pd.DataFrame(rows)
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            near_miss_rate=("near_miss", "mean"),
            progress=("progress", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            episode_min_distance=("episode_min_distance", "mean"),
            episode_length=("episode_length", "mean"),
            episode_reward=("episode_reward", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            raw_safe_margin_unsafe_action_rate=("raw_safe_margin_unsafe_action_rate", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response_rate", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            nan_or_crash=("nan_or_crash", "sum"),
        )
        .reset_index()
    )
    return out.to_dict("records")


def summarize_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    import pandas as pd

    df = pd.DataFrame(rows)
    group_cols = ["method_key", "method", "checkpoint", "checkpoint_step", "checkpoint_label", "scenario", "motion_mode", "threat_class"]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            steps=("step", "count"),
            raw_unsafe_rate=("raw_unsafe_action", "mean"),
            raw_safe_margin_unsafe_rate=("raw_safe_margin_unsafe_action", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            raw_min_predicted_ttc=("raw_min_predicted_ttc", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response", "mean"),
        )
        .reset_index()
    )
    return out.to_dict("records")


def gpsi_summary_rows(accum: dict[tuple[Any, ...], dict[str, ScalarAccumulator]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, groups in accum.items():
        method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        row: dict[str, Any] = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
        }
        for name, acc in groups.items():
            stats = acc.stats(name)
            row.update(stats)
            if name.startswith("logvar_xy_"):
                values = np.asarray(acc.values, dtype=np.float64)
                row[f"{name}_span"] = float(np.max(values) - np.min(values)) if values.size else np.nan
        rows.append(row)
    return rows


def feature_summary_rows(
    feature_accum: dict[tuple[Any, ...], dict[str, BlockAccumulator]],
    cfg_by_method: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, groups in feature_accum.items():
        method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        prefix = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
        }
        gpsi_cfg = cfg_by_method.get(str(method_key), {}).get("gpsi", {})
        include_z = bool(gpsi_cfg.get("include_z", True))
        prefix.update(
            {
                "z_transform": str(gpsi_cfg.get("z_transform", "standardize" if gpsi_cfg.get("normalize_z", False) else "raw")),
                "z_l2_target_norm": float(gpsi_cfg.get("z_l2_target_norm", np.nan)),
                "z_layernorm_alpha": float(gpsi_cfg.get("z_layernorm_alpha", np.nan)),
                "z_layernorm_eps": float(gpsi_cfg.get("z_layernorm_eps", np.nan)),
            }
        )
        for block in ["obs_i_12", "z_i_64_raw", "z_i_64_after_constraint", "delta_hat_9_after_scale", "logvar_hat_9_clamped", "full_aug_obs"]:
            if block == "z_i_64_after_constraint" and not include_z:
                rows.append(BlockAccumulator().row(prefix, block, not_applicable=True))
            elif block in groups:
                rows.append(groups[block].row(prefix, block))
            else:
                rows.append(BlockAccumulator().row(prefix, block, not_applicable=(block.startswith("z_i") and not include_z)))
    return rows


def copy_sample_traces(result_dir: Path, trace_files: list[tuple[Path, dict[str, Any]]]) -> None:
    success = collision = near = unsafe = 0
    for path, row in trace_files:
        if not path.exists():
            continue
        if int(row.get("success", 0)) and success < 9:
            shutil.copyfile(path, result_dir / "traces/sampled_success_traces" / path.name)
            success += 1
        if int(row.get("collision", 0)) and collision < 9:
            shutil.copyfile(path, result_dir / "traces/sampled_collision_traces" / path.name)
            collision += 1
        if int(row.get("near_miss", 0)) and not int(row.get("collision", 0)) and near < 9:
            shutil.copyfile(path, result_dir / "traces/sampled_near_miss_traces" / path.name)
            near += 1
        if float(row.get("raw_unsafe_action_rate", 0.0)) >= 0.10 and unsafe < 9:
            shutil.copyfile(path, result_dir / "traces/high_raw_unsafe_rate_traces" / path.name)
            unsafe += 1


def evaluate_episode(
    *,
    model: PPO,
    env: DynamicObstacleFlowEnv | GpsiObsWrapper,
    cfg: dict[str, Any] | None,
    method_key: str,
    method: str,
    checkpoint_name: str,
    checkpoint_step_value: int,
    checkpoint_label: str,
    scenario: str,
    episode_id: int,
    episode_seed: int,
    args: argparse.Namespace,
    gpsi_accum: dict[tuple[Any, ...], dict[str, ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, BlockAccumulator]],
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
        raw = raw_action_diagnostics(action, info, flow_env, args)
        action_norm = float(np.linalg.norm(action))
        action_delta = float(np.linalg.norm(action - last_action))
        no_response = int(action_norm < float(args.no_response_action_norm))
        action_norms.append(action_norm)
        action_deltas.append(action_delta)
        no_response_values.append(float(no_response))
        cpa_values.append(float(raw["raw_min_predicted_cpa"]))
        raw_unsafe_count += int(raw["raw_unsafe_action"])
        raw_safe_unsafe_count += int(raw["raw_safe_margin_unsafe_action"])

        prefix = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint_name,
            "checkpoint_step": int(checkpoint_step_value),
            "checkpoint_label": checkpoint_label,
            "scenario": scenario,
        }
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

        weights, entropy = attention_snapshot(model, flow_env.max_obs)
        threat_index = int(info.get("threat_obstacle_index", -1))
        threat_weight = float(weights[threat_index]) if 0 <= threat_index < len(weights) else float("nan")
        gpsi_step: dict[str, Any] = {}
        if isinstance(env, GpsiObsWrapper) and cfg is not None:
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
            trace_row = {
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
                "goal_directed_velocity": float(np.dot(uav_vel, normalize(goal - uav))),
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
            trace_rows.append(trace_row)
        last_action = action.copy()

    near_miss = int(float(last_info["episode_min_distance"]) < args.near_miss_distance and not bool(last_info["is_collision"]))
    episode_row = {
        "method_key": method_key,
        "method": method,
        "checkpoint": checkpoint_name,
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
        write_trace(trace_rows, trace_path)
        trace_files.append((trace_path, episode_row))
    return episode_row, raw_rows, trace_rows, trace_files


def checkpoint_specs(config_key: str, cfg: dict[str, Any]) -> list[tuple[str, Path, int, str]]:
    train_steps = int(cfg.get("training", {}).get("total_steps", 500_000))
    ckpt_dir = ROOT / DEFAULT_CONFIGS[config_key]["checkpoint_dir"]
    if config_key == "z_layernorm_alpha_0p5_cont_1p5m":
        labels: list[tuple[str, str, int]] = [
            ("parent_500k", "parent_500k.zip", 500_000),
            ("750k", "checkpoint_750k.zip", 750_000),
            ("1000k", "checkpoint_1000k.zip", 1_000_000),
            ("1250k", "checkpoint_1250k.zip", 1_250_000),
            ("1500k", "checkpoint_1500k.zip", 1_500_000),
            ("final", "final.zip", 1_500_000),
            ("best_by_eval", "best_by_eval.zip", 1_500_000),
        ]
        return [
            (f"{config_key}_{label}", ckpt_dir / filename, step, label)
            for label, filename, step in labels
        ]
    if config_key == "n3f_no_z_full":
        return [
            (f"{config_key}_final", ckpt_dir / "final.zip", 1_500_000, "final"),
            (f"{config_key}_best_by_eval", ckpt_dir / "best_by_eval.zip", 1_500_000, "best_by_eval"),
        ]
    steps = [int(step) for step in cfg.get("training", {}).get("checkpoint_steps", [250_000, train_steps])]
    if train_steps not in steps:
        steps.append(train_steps)
    ckpts: list[tuple[str, Path, int, str]] = []
    seen_steps: set[int] = set()
    for step in sorted(step for step in steps if step > 0 and step <= train_steps):
        if step in seen_steps:
            continue
        seen_steps.add(step)
        label = f"{step // 1000}k"
        ckpts.append((f"{config_key}_{label}", ckpt_dir / f"checkpoint_{step // 1000}k.zip", step, label))
    ckpts.append((f"{config_key}_final", ckpt_dir / "final.zip", train_steps, "final"))
    best = ckpt_dir / "best_by_eval.zip"
    if best.exists():
        ckpts.append((f"{config_key}_best_by_eval", best, train_steps, "best_by_eval"))
    return ckpts


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    ensure_dirs(result_dir)
    try:
        DEFAULT_CONFIGS["z_layernorm_alpha_0p5_cont_1p5m"]["checkpoint_dir"] = args.z2_checkpoint_dir
        DEFAULT_CONFIGS["n3f_no_z_full"]["checkpoint_dir"] = args.noz_reference_dir
        attention_path = ROOT / args.attention_checkpoint
        if not attention_path.exists():
            raise PhaseN3Z2CEvalStop("checkpoint_missing", f"missing attention reference checkpoint: {rel(attention_path)}")
        append_csv(
            result_dir / "tables/phase_n3z2c_command_manifest.csv",
            [
                {
                    "stage": "eval",
                    "method_key": "all",
                    "method": "all",
                    "command": " ".join(["python", *sys.argv]),
                    "episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "eval_seed": int(args.eval_seed),
                    "attention_checkpoint": rel(attention_path),
                }
            ],
        )

        cfg_by_method: dict[str, dict[str, Any]] = {}
        config_rows: list[dict[str, Any]] = []
        for key in args.configs:
            if key not in DEFAULT_CONFIGS:
                raise PhaseN3Z2CEvalStop("schema_mismatch", f"unknown config key: {key}")
            cfg = load_yaml(ROOT / DEFAULT_CONFIGS[key]["config"])
            cfg_by_method[key] = cfg
            config_rows.append(
                {
                    "method_key": key,
                    "method": cfg.get("method_name", key),
                    "config": DEFAULT_CONFIGS[key]["config"],
                    "checkpoint_dir": DEFAULT_CONFIGS[key]["checkpoint_dir"],
                    "obs_aug_dim": int(cfg.get("gpsi", {}).get("obs_aug_dim", -1)),
                    "include_z": int(bool(cfg.get("gpsi", {}).get("include_z", True))),
                    "normalize_z": int(bool(cfg.get("gpsi", {}).get("normalize_z", False))),
                    "z_transform": str(cfg.get("gpsi", {}).get("z_transform", "standardize" if cfg.get("gpsi", {}).get("normalize_z", False) else "raw")),
                    "z_l2_target_norm": cfg.get("gpsi", {}).get("z_l2_target_norm", ""),
                    "z_layernorm_alpha": cfg.get("gpsi", {}).get("z_layernorm_alpha", ""),
                    "z_layernorm_eps": cfg.get("gpsi", {}).get("z_layernorm_eps", ""),
                    "logvar_clamp": json.dumps(cfg.get("gpsi", {}).get("logvar_clamp", [-5.0, 3.0])),
                    "z_stats_path": cfg.get("gpsi", {}).get("z_stats_path", ""),
                    "train_steps": int(cfg.get("training", {}).get("total_steps", 500_000)),
                    "seed": int(cfg.get("training", {}).get("seed", 0)),
                    "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
                    "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
                }
            )
        write_csv(result_dir / "tables/phase_n3z2c_config_manifest.csv", config_rows)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        all_episode_rows: list[dict[str, Any]] = []
        all_raw_rows: list[dict[str, Any]] = []
        all_trace_files: list[tuple[Path, dict[str, Any]]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, BlockAccumulator]] = {}

        method_jobs: list[tuple[str, str, dict[str, Any] | None, str, Path, int, str]] = []
        method_jobs.append(("attention_full", "attention_full_1500k", None, "attention_full_1500k", attention_path, 1_500_000, "attention_full_1500k"))
        for key in args.configs:
            cfg = cfg_by_method[key]
            for checkpoint_name, checkpoint_path, fallback_step, label in checkpoint_specs(key, cfg):
                if not checkpoint_path.exists():
                    raise PhaseN3Z2CEvalStop("checkpoint_missing", f"missing checkpoint for {key}: {rel(checkpoint_path)}")
                method_jobs.append((key, str(cfg.get("method_name", key)), cfg, checkpoint_name, checkpoint_path, checkpoint_step(checkpoint_path, fallback_step), label))

        for method_key, method, cfg, checkpoint_name, checkpoint_path, step_value, checkpoint_label in method_jobs:
            if method_key == "attention_full":
                model = PPO.load(
                    str(checkpoint_path),
                    device=args.device,
                    custom_objects={"policy_kwargs": policy_kwargs_attention(args.hidden_dim)},
                )
            else:
                model = PPO.load(str(checkpoint_path), device=args.device)

            for scenario in args.scenarios:
                env: DynamicObstacleFlowEnv | GpsiObsWrapper
                env = DynamicObstacleFlowEnv(scenario=scenario) if cfg is None else make_gpsi_env(cfg, scenario, args)
                start = time.time()
                last = start
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3Z2C_EVAL_START "
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
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3Z2C_EVAL_HEARTBEAT "
                            f"method={method_key} checkpoint={checkpoint_label} scenario={scenario} "
                            f"episodes={done}/{args.num_episodes} rate={rate:.2f} ep/s eta={eta/60.0:.2f} min",
                            flush=True,
                        )
                        last = now
                env.close()

        eval_summary = summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_step", "checkpoint_label", "scenario"])
        write_csv(result_dir / "tables/phase_n3z2c_episode_metrics.csv", all_episode_rows)
        write_csv(result_dir / "tables/phase_n3z2c_checkpoint_eval_summary.csv", eval_summary)
        write_csv(result_dir / "tables/phase_n3z2c_eval_summary.csv", eval_summary)
        write_csv(result_dir / "tables/phase_n3z2c_scenario_breakdown.csv", eval_summary)
        write_csv(
            result_dir / "tables/phase_n3z2c_motion_mode_breakdown.csv",
            summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_step", "checkpoint_label", "threat_motion_mode"]),
        )
        write_csv(
            result_dir / "tables/phase_n3z2c_threat_class_breakdown.csv",
            summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_step", "checkpoint_label", "threat_class"]),
        )
        write_csv(result_dir / "tables/phase_n3z2c_raw_unsafe_action_steps.csv", all_raw_rows)
        write_csv(result_dir / "tables/phase_n3z2c_raw_unsafe_action_summary.csv", summarize_raw(all_raw_rows))
        write_csv(result_dir / "tables/phase_n3z2c_gpsi_output_summary.csv", gpsi_summary_rows(gpsi_accum))
        write_csv(result_dir / "tables/phase_n3z2c_aug_feature_block_stats.csv", feature_summary_rows(feature_accum, cfg_by_method))
        copy_sample_traces(result_dir, all_trace_files)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3Z2C_EVAL_END "
            f"episodes={len(all_episode_rows)} raw_steps={len(all_raw_rows)} gpsi_groups={len(gpsi_accum)} traces={len(all_trace_files)}",
            flush=True,
        )
    except PhaseN3Z2CEvalStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "eval_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
