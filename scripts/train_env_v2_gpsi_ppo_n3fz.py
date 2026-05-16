from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from envs.wrappers.gpsi_obs_wrapper import GpsiObsWrapper
from models.gpsi_ppo_policy import GpsiObstacleSetExtractor


RESULT_DIR = ROOT / "results/env_v2_phase_n3fz_noz_full_z_screen"
N3R_FLAG = ROOT / "results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag"
N35_FLAG = ROOT / "results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n3r_missing": "PHASE_N3FZ_STOP_PHASE_N3R_MISSING.flag",
    "phase_n3_5_missing": "PHASE_N3FZ_STOP_PHASE_N3_5_MISSING.flag",
    "gpsi_checkpoint_missing": "PHASE_N3FZ_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "z_stats_missing": "PHASE_N3FZ_STOP_Z_STATS_MISSING.flag",
    "gpsi_wrapper_failed": "PHASE_N3FZ_STOP_GPSI_WRAPPER_FAILED.flag",
    "gpsi_not_frozen": "PHASE_N3FZ_STOP_GPSI_NOT_FROZEN.flag",
    "schema_mismatch": "PHASE_N3FZ_STOP_SCHEMA_MISMATCH.flag",
    "output_scale_invalid": "PHASE_N3FZ_STOP_OUTPUT_SCALE_INVALID.flag",
    "train_failed": "PHASE_N3FZ_STOP_TRAIN_FAILED.flag",
}


class PhaseN3FZTrainStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class ProgressState:
    total_steps: int
    start_time: float
    last_heartbeat: float


class N3FZTrainCallback(BaseCallback):
    def __init__(self, total_steps: int, method_key: str, method_name: str, heartbeat_seconds: float = 30.0) -> None:
        super().__init__()
        now = time.time()
        self.state = ProgressState(total_steps=total_steps, start_time=now, last_heartbeat=now)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.method_key = method_key
        self.method_name = method_name
        self.start_num_timesteps = 0
        self.heartbeat_rows: list[dict[str, Any]] = []
        self.episode_rows: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.start_num_timesteps = int(self.model.num_timesteps)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_TRAIN_START "
            f"config={self.method_key} total_steps={self.state.total_steps} "
            f"start_num_timesteps={self.start_num_timesteps}",
            flush=True,
        )

    def _on_step(self) -> bool:
        current_steps = max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)
        infos = self.locals.get("infos", []) if isinstance(self.locals, dict) else []
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode:
                self.episode_rows.append(
                    {
                        "method_key": self.method_key,
                        "method": self.method_name,
                        "steps": int(current_steps),
                        "episode_reward": float(episode.get("r", np.nan)),
                        "episode_length": int(episode.get("l", 0)),
                        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        now = time.time()
        if now - self.state.last_heartbeat >= self.heartbeat_seconds:
            elapsed = now - self.state.start_time
            percent = 100.0 * current_steps / max(self.state.total_steps, 1)
            rate = current_steps / max(elapsed, 1e-6)
            eta_seconds = max(self.state.total_steps - current_steps, 0) / max(rate, 1e-6)
            recent_rewards = [row["episode_reward"] for row in self.episode_rows[-50:]]
            row = {
                "method_key": self.method_key,
                "method": self.method_name,
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "steps": int(current_steps),
                "target_steps": int(self.state.total_steps),
                "percent": float(percent),
                "steps_per_second": float(rate),
                "eta_minutes": float(eta_seconds / 60.0),
                "recent_episode_reward_mean": float(np.mean(recent_rewards)) if recent_rewards else np.nan,
            }
            self.heartbeat_rows.append(row)
            print(
                f"[{row['wall_time']}] N3FZ_TRAIN_HEARTBEAT config={self.method_key} "
                f"steps={current_steps}/{self.state.total_steps} percent={percent:6.2f}% "
                f"rate={rate:8.1f} step/s eta={eta_seconds/60.0:7.2f} min "
                f"recent_reward={row['recent_episode_reward_mean']:.3f}",
                flush=True,
            )
            self.state.last_heartbeat = now
        return True


class FixedNameCheckpointCallback(BaseCallback):
    def __init__(self, target_steps: list[int], checkpoint_dir: Path) -> None:
        super().__init__()
        self.target_steps = sorted(int(step) for step in target_steps)
        self.checkpoint_dir = checkpoint_dir
        self.saved_steps: set[int] = set()
        self.start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_num_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        current_steps = max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)
        for step in self.target_steps:
            if step in self.saved_steps or current_steps < step:
                continue
            path = self.checkpoint_dir / f"checkpoint_{step // 1000}k.zip"
            self.model.save(str(path))
            self.saved_steps.add(step)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_CHECKPOINT_SAVED step={step} path={rel(path)}", flush=True)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Phase N3FZ repaired Gpsi-PPO no-shield config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-steps", type=int, default=500_000)
    parser.add_argument("--checkpoint-steps", nargs="*", type=int, default=[250_000, 500_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(out_dir: Path) -> None:
    for path in [out_dir, RESULT_DIR, RESULT_DIR / "logs", RESULT_DIR / "tables", RESULT_DIR / "plots"]:
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


def checkpoint_for_steps(out_dir: Path, train_steps: int) -> Path:
    return out_dir / f"checkpoint_{int(train_steps) // 1000}k.zip"


def file_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def target_artifacts_ready(out_dir: Path, train_steps: int) -> bool:
    return file_ready(out_dir / "final.zip") and file_ready(checkpoint_for_steps(out_dir, train_steps))


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_path(out_dir: Path) -> Path:
    return out_dir / "TRAIN_RUNNING.lock"


def complete_path(out_dir: Path) -> Path:
    return out_dir / "TRAIN_COMPLETE.flag"


def status_path(out_dir: Path) -> Path:
    return out_dir / "TRAIN_STATUS.json"


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_train_status(out_dir: Path, status: str, cfg: dict[str, Any], args: argparse.Namespace, detail: str = "") -> None:
    payload = {
        "status": status,
        "detail": detail,
        "pid": int(os.getpid()),
        "config": str(args.config),
        "method_key": str(cfg.get("method_key", "")),
        "method_name": str(cfg.get("method_name", cfg.get("method_key", ""))),
        "out_dir": rel(out_dir),
        "requested_train_steps": int(args.train_steps),
        "expected_checkpoint": rel(checkpoint_for_steps(out_dir, int(args.train_steps))),
        "final_zip": rel(out_dir / "final.zip"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
    }
    status_path(out_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_train_lock(out_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "pid": int(os.getpid()),
        "config": str(args.config),
        "method_key": str(cfg.get("method_key", "")),
        "method_name": str(cfg.get("method_name", cfg.get("method_key", ""))),
        "out_dir": rel(out_dir),
        "requested_train_steps": int(args.train_steps),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "command": " ".join(["python", *sys.argv]),
    }
    lock_path(out_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wait_for_running_training(out_dir: Path, cfg: dict[str, Any], args: argparse.Namespace, lock_payload: dict[str, Any]) -> None:
    running_pid = int(lock_payload.get("pid", -1))
    write_train_status(out_dir, "waiting_on_running_lock", cfg, args, f"waiting for pid={running_pid}")
    last_log = 0.0
    while True:
        if target_artifacts_ready(out_dir, int(args.train_steps)) or (complete_path(out_dir).exists() and target_artifacts_ready(out_dir, int(args.train_steps))):
            complete_path(out_dir).write_text(f"completed_wait\npid={running_pid}\n", encoding="utf-8")
            write_train_status(out_dir, "completed_wait", cfg, args, f"observed completed training from pid={running_pid}")
            return
        if not pid_is_running(running_pid):
            write_train_status(out_dir, "stale_lock_unfinished", cfg, args, f"pid={running_pid} exited before target artifacts were ready")
            raise PhaseN3FZTrainStop(
                "train_failed",
                f"TRAIN_RUNNING.lock pid={running_pid} is no longer running and requested artifacts are incomplete in {rel(out_dir)}; refusing duplicate training",
            )
        now = time.time()
        if now - last_log >= 30.0:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_TRAIN_WAIT "
                f"out_dir={rel(out_dir)} pid={running_pid} target_steps={args.train_steps}",
                flush=True,
            )
            last_log = now
        time.sleep(10.0)


def prepare_training_guard(out_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_ready = target_artifacts_ready(out_dir, int(args.train_steps))
    lock = lock_path(out_dir)

    if expected_ready:
        complete_path(out_dir).write_text(f"completed_skip\ntrain_steps={int(args.train_steps)}\n", encoding="utf-8")
        write_train_status(out_dir, "completed_skip", cfg, args, "target checkpoint and final.zip already exist")
        if lock.exists():
            payload = read_json_file(lock)
            pid = int(payload.get("pid", -1))
            if not pid_is_running(pid):
                lock.unlink(missing_ok=True)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_TRAIN_SKIP "
            f"out_dir={rel(out_dir)} target_steps={args.train_steps}",
            flush=True,
        )
        return "skip"

    if lock.exists():
        payload = read_json_file(lock)
        pid = int(payload.get("pid", -1))
        if pid_is_running(pid):
            wait_for_running_training(out_dir, cfg, args, payload)
            return "wait"
        existing = sorted(path.name for path in out_dir.glob("*.zip"))
        if existing:
            write_train_status(out_dir, "stale_lock_partial_artifacts", cfg, args, f"stale pid={pid}; existing={existing}")
            raise PhaseN3FZTrainStop(
                "train_failed",
                f"stale TRAIN_RUNNING.lock in {rel(out_dir)} and partial checkpoints exist: {existing}; resume is not implemented, refusing overwrite",
            )
        write_train_status(out_dir, "stale_lock_recovered", cfg, args, f"removed stale pid={pid}; no checkpoints existed")
        lock.unlink(missing_ok=True)

    existing = sorted(path.name for path in out_dir.glob("*.zip"))
    if existing:
        write_train_status(out_dir, "partial_artifacts_no_lock", cfg, args, f"existing={existing}")
        raise PhaseN3FZTrainStop(
            "train_failed",
            f"partial checkpoint artifacts exist in {rel(out_dir)} without TRAIN_COMPLETE.flag for requested steps={args.train_steps}: {existing}; refusing overwrite",
        )

    write_train_lock(out_dir, cfg, args)
    write_train_status(out_dir, "running", cfg, args, "lock acquired")
    return "start"


def write_stop(reason: str, detail: str) -> None:
    ensure_dirs(RESULT_DIR)
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["train_failed"])
    write_text(RESULT_DIR / flag_name, f"{reason}\n{detail.strip()}\n")
    write_text(RESULT_DIR / "phase_n3fz_status.txt", f"stopped:{flag_name}\n")
    write_text(
        RESULT_DIR / "PHASE_N3FZ_NOZ_FULL_Z_SCREEN_REPORT.md",
        "\n".join(
            [
                "# Phase N3F/Z No-Z Full + Z Screen Report",
                "",
                f"`terminal_decision = phase_n3fz_stopped_{reason}`",
                "",
                "Partial report generated by the N3FZ training script.",
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise PhaseN3FZTrainStop("schema_mismatch", f"config is not a mapping: {rel(path)}")
    return cfg


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_prerequisites(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if not N35_FLAG.exists():
        raise PhaseN3FZTrainStop("phase_n3_5_missing", f"missing Phase N3.5 complete flag: {rel(N35_FLAG)}")
    if not N3R_FLAG.exists():
        raise PhaseN3FZTrainStop("phase_n3r_missing", f"missing Phase N3R complete flag: {rel(N3R_FLAG)}")
    method_key = str(cfg.get("method_key", ""))
    if method_key != "n3f_no_z_full" and int(args.train_steps) > 500_000:
        raise PhaseN3FZTrainStop("schema_mismatch", f"N3FZ z screening may not train above 500k steps: {args.train_steps}")
    checkpoint = ROOT / str(cfg.get("gpsi", {}).get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    if not checkpoint.exists():
        raise PhaseN3FZTrainStop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")
    gpsi_cfg = cfg.get("gpsi", {})
    if bool(gpsi_cfg.get("normalize_z", False)):
        z_stats = ROOT / str(gpsi_cfg.get("z_stats_path", ""))
        if not z_stats.exists():
            raise PhaseN3FZTrainStop("z_stats_missing", f"missing train-split z stats file: {rel(z_stats)}")


def make_wrapper(cfg: dict[str, Any], scenario: str, device: str) -> GpsiObsWrapper:
    gpsi_cfg = cfg.get("gpsi", {})
    return GpsiObsWrapper(
        DynamicObstacleFlowEnv(scenario=scenario),
        gpsi_checkpoint=ROOT / str(gpsi_cfg.get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth")),
        device=device,
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


def make_env_factory(cfg: dict[str, Any], args: argparse.Namespace, rank: int):
    env_cfg = cfg.get("env", {})
    scenario = str(env_cfg.get("train_scenario", "train_flow_mixed"))
    device = args.device or str(env_cfg.get("device", "cpu"))
    expected_dim = int(cfg.get("gpsi", {}).get("obs_aug_dim", 94))

    def _factory() -> gym.Env:
        env = make_wrapper(cfg, scenario, device)
        freeze = env.freeze_check
        if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
            raise PhaseN3FZTrainStop("gpsi_not_frozen", f"Gpsi freeze check failed: {freeze}")
        actual_dim = int(env.observation_space["obs"].shape[-1])
        if actual_dim != expected_dim:
            raise PhaseN3FZTrainStop("schema_mismatch", f"aug obs dim mismatch: env={actual_dim} config={expected_dim}")
        wrapped = Monitor(env)
        wrapped.reset(seed=int(args.seed) + rank)
        return wrapped

    return _factory


def make_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo_cfg = cfg.get("ppo", {})
    gpsi_cfg = cfg.get("gpsi", {})
    return {
        "features_extractor_class": GpsiObstacleSetExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": int(ppo_cfg.get("hidden_dim", 64)),
            "obs_dim": int(gpsi_cfg.get("obs_aug_dim", 94)),
            "use_risk_bias": bool(ppo_cfg.get("use_risk_bias", False)),
            "lambda_bias": float(ppo_cfg.get("lambda_bias", 0.0)),
        },
        "net_arch": ppo_cfg.get("net_arch", {"pi": [128, 128], "vf": [128, 128]}),
        "activation_fn": torch.nn.Tanh,
    }


def finite_l2(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)


def validate_wrapper(cfg: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> None:
    try:
        device = args.device or str(cfg.get("env", {}).get("device", "cpu"))
        env = make_wrapper(cfg, "eval_flow_id", device)
        freeze = env.freeze_check
        obs, info = env.reset(seed=int(args.seed))
        expected_dim = int(cfg.get("gpsi", {}).get("obs_aug_dim", 94))
        rows = [
            {"method_key": cfg.get("method_key", ""), "check": "phase_n3_5_complete_flag", "value": str(N35_FLAG.exists()), "detail": rel(N35_FLAG)},
            {"method_key": cfg.get("method_key", ""), "check": "phase_n3r_complete_flag", "value": str(N3R_FLAG.exists()), "detail": rel(N3R_FLAG)},
            {
                "method_key": cfg.get("method_key", ""),
                "check": "gpsi_frozen",
                "value": str(not freeze.training and not freeze.requires_grad_any and freeze.trainable_parameters == 0),
                "detail": json.dumps(freeze.__dict__, sort_keys=True),
            },
            {
                "method_key": cfg.get("method_key", ""),
                "check": "aug_obs_dim",
                "value": str(int(obs["obs"].shape[-1])),
                "detail": f"expected={expected_dim}",
            },
            {"method_key": cfg.get("method_key", ""), "check": "no_shield", "value": "true", "detail": "observation-only Gpsi wrapper"},
            {"method_key": cfg.get("method_key", ""), "check": "no_action_filtering", "value": "true", "detail": "raw PPO velocity command only"},
            {"method_key": cfg.get("method_key", ""), "check": "no_safety_cost", "value": "true", "detail": "EnvV2 original reward"},
            {
                "method_key": cfg.get("method_key", ""),
                "check": "logvar_clip_sanity",
                "value": "already_bounded" if tuple(cfg.get("gpsi", {}).get("logvar_clamp", [-5.0, 3.0]))[0] >= -5.0 and tuple(cfg.get("gpsi", {}).get("logvar_clamp", [-5.0, 3.0]))[1] <= 5.0 else "needs_review",
                "detail": f"logvar_clamp={tuple(cfg.get('gpsi', {}).get('logvar_clamp', [-5.0, 3.0]))}",
            },
        ]
        if int(obs["obs"].shape[-1]) != expected_dim:
            raise PhaseN3FZTrainStop("schema_mismatch", f"obs dim mismatch: {obs['obs'].shape[-1]} vs {expected_dim}")
        if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
            raise PhaseN3FZTrainStop("gpsi_not_frozen", f"Gpsi freeze check failed: {freeze}")

        delta_1s: list[float] = []
        logvar_1s: list[float] = []
        z_raw_l2: list[float] = []
        z_after_l2: list[float] = []
        for _ in range(80):
            goal = np.asarray(info["goal_position"], dtype=np.float32)
            uav = np.asarray(info["uav_position"], dtype=np.float32)
            vec = goal - uav
            vec[2] = 0.0
            norm = float(np.linalg.norm(vec))
            action = vec / norm if norm > 1e-8 else env.action_space.sample()
            action = np.asarray(action, dtype=np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            debug = env.latest_gpsi_debug
            active = np.asarray(debug.get("active_slots", []), dtype=np.int64)
            if active.size:
                delta = np.asarray(debug["delta_hat_raw"], dtype=np.float32)[active, 0, :]
                logvar = np.asarray(debug["logvar_hat"], dtype=np.float32)[active, 0, :2]
                delta_1s.extend(finite_l2(delta).tolist())
                logvar_1s.extend(logvar.reshape(-1).tolist())
                z_raw_l2.extend(finite_l2(np.asarray(debug["z_raw"], dtype=np.float32)[active]).tolist())
                z_after_l2.extend(finite_l2(np.asarray(debug["z_after_norm"], dtype=np.float32)[active]).tolist())
            if terminated or truncated:
                obs, info = env.reset(seed=int(args.seed) + 1)
        env.close()
        delta_p95 = float(np.percentile(delta_1s, 95)) if delta_1s else float("nan")
        delta_max = float(np.max(delta_1s)) if delta_1s else float("nan")
        logvar_span = float(np.max(logvar_1s) - np.min(logvar_1s)) if logvar_1s else float("nan")
        rows.extend(
            [
                {
                    "method_key": cfg.get("method_key", ""),
                    "check": "delta_norm_1s_p95_pretrain",
                    "value": delta_p95,
                    "detail": f"max={delta_max}",
                },
                {
                    "method_key": cfg.get("method_key", ""),
                    "check": "logvar_xy_1s_span_pretrain",
                    "value": logvar_span,
                    "detail": f"min={np.min(logvar_1s) if logvar_1s else np.nan} max={np.max(logvar_1s) if logvar_1s else np.nan}",
                },
                {
                    "method_key": cfg.get("method_key", ""),
                    "check": "z_l2_p95_raw_pretrain",
                    "value": float(np.percentile(z_raw_l2, 95)) if z_raw_l2 else np.nan,
                    "detail": "raw frozen-Gpsi z before optional z normalization",
                },
                {
                    "method_key": cfg.get("method_key", ""),
                    "check": "z_l2_p95_after_pretrain",
                    "value": float(np.percentile(z_after_l2, 95)) if z_after_l2 else np.nan,
                    "detail": "z block supplied to PPO when include_z=true",
                },
            ]
        )
        append_csv(RESULT_DIR / "tables/phase_n3fz_schema_check.csv", rows)
        if not np.isfinite(delta_p95) or delta_p95 > 100.0 or delta_max > 1000.0:
            raise PhaseN3FZTrainStop("output_scale_invalid", f"pretrain Gpsi output scale invalid: delta_1s_p95={delta_p95} max={delta_max}")
    except PhaseN3FZTrainStop:
        raise
    except Exception as exc:
        raise PhaseN3FZTrainStop("gpsi_wrapper_failed", traceback.format_exc()) from exc


def write_command_manifest(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    checkpoint = ROOT / str(cfg.get("gpsi", {}).get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    row = {
        "stage": "train",
        "method_key": cfg.get("method_key", ""),
        "method": cfg.get("method_name", cfg.get("method_key", "")),
        "command": " ".join(["python", *sys.argv]),
        "config": rel(ROOT / args.config),
        "gpsi_checkpoint": rel(checkpoint),
        "gpsi_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else "missing",
        "out_dir": rel(out_dir),
        "train_steps": int(args.train_steps),
        "seed": int(args.seed),
        "n_envs": int(args.n_envs or cfg.get("env", {}).get("n_envs", 4)),
    }
    append_csv(RESULT_DIR / "tables/phase_n3fz_command_manifest.csv", [row])


def train() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    try:
        cfg = load_config(ROOT / args.config)
        if args.n_envs is not None:
            cfg.setdefault("env", {})["n_envs"] = int(args.n_envs)
        if args.device is not None:
            cfg.setdefault("env", {})["device"] = args.device
        cfg.setdefault("training", {})["seed"] = int(args.seed)
        cfg["training"]["total_steps"] = int(args.train_steps)
        cfg["training"]["checkpoint_steps"] = [int(step) for step in args.checkpoint_steps]
        check_prerequisites(cfg, args)
        guard_action = prepare_training_guard(out_dir, cfg, args)
        if guard_action in {"skip", "wait"}:
            return
        save_yaml(out_dir / "config_resolved.yaml", cfg)
        write_command_manifest(args, cfg, out_dir)
        set_seed(args.seed)
        validate_wrapper(cfg, args, out_dir)

        n_envs = int(cfg.get("env", {}).get("n_envs", 4))
        device = args.device or str(cfg.get("env", {}).get("device", "cpu"))
        env_fns = [make_env_factory(cfg, args, rank) for rank in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)
        ppo_cfg = cfg.get("ppo", {})
        method_key = str(cfg.get("method_key", Path(args.config).stem))
        method_name = str(cfg.get("method_name", method_key))
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_TRAIN_CONFIG "
            f"config={method_key} steps={args.train_steps} seed={args.seed} n_envs={n_envs} device={device} "
            f"out_dir={rel(out_dir)} no_shield=true action_filtering=false safety_cost=false",
            flush=True,
        )
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            seed=int(args.seed),
            device=device,
            verbose=1,
            tensorboard_log=str(out_dir / "tensorboard"),
            policy_kwargs=make_policy_kwargs(cfg),
            n_steps=int(ppo_cfg.get("n_steps", 1024)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
            vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        )
        callback = N3FZTrainCallback(args.train_steps, method_key, method_name, args.heartbeat_seconds)
        targets = [step for step in args.checkpoint_steps if step <= args.train_steps]
        checkpoint_cb = FixedNameCheckpointCallback(targets, out_dir)
        model.learn(
            total_timesteps=int(args.train_steps),
            callback=[callback, checkpoint_cb],
            progress_bar=False,
            reset_num_timesteps=True,
        )
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        if not (out_dir / "best_by_eval.zip").exists():
            shutil.copyfile(final_path, out_dir / "best_by_eval.zip")
        final_step_checkpoint = out_dir / f"checkpoint_{int(args.train_steps) // 1000}k.zip"
        if not final_step_checkpoint.exists():
            shutil.copyfile(final_path, final_step_checkpoint)
        append_csv(RESULT_DIR / "tables/phase_n3fz_train_curve.csv", callback.episode_rows)
        append_csv(RESULT_DIR / "tables/phase_n3fz_train_heartbeat.csv", callback.heartbeat_rows)
        vec_env.close()
        complete_path(out_dir).write_text(f"completed\ntrain_steps={int(args.train_steps)}\n", encoding="utf-8")
        write_train_status(out_dir, "completed", cfg, args, "training completed successfully")
        lock_path(out_dir).unlink(missing_ok=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3FZ_TRAIN_END config={method_key} final={rel(final_path)}", flush=True)
    except PhaseN3FZTrainStop as exc:
        write_stop(exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        try:
            if "out_dir" in locals() and "cfg" in locals() and "args" in locals():
                write_train_status(out_dir, "failed", cfg, args, detail)
        except Exception:
            pass
        write_stop("train_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    train()
