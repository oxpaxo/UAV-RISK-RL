from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


RESULT_DIR = ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield"
N2_FLAG = ROOT / "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n2_missing": "PHASE_N3_STOP_PHASE_N2_MISSING.flag",
    "gpsi_checkpoint_missing": "PHASE_N3_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "gpsi_wrapper_failed": "PHASE_N3_STOP_GPSI_WRAPPER_FAILED.flag",
    "gpsi_not_frozen": "PHASE_N3_STOP_GPSI_NOT_FROZEN.flag",
    "schema_mismatch": "PHASE_N3_STOP_SCHEMA_MISMATCH.flag",
    "train_failed": "PHASE_N3_STOP_TRAIN_FAILED.flag",
}


class PhaseN3TrainStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class ProgressState:
    total_steps: int
    start_time: float
    last_heartbeat: float


class HeartbeatCallback(BaseCallback):
    def __init__(self, total_steps: int, heartbeat_seconds: float = 30.0) -> None:
        super().__init__()
        now = time.time()
        self.state = ProgressState(total_steps=total_steps, start_time=now, last_heartbeat=now)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.start_num_timesteps = 0
        self.rows: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.start_num_timesteps = int(self.model.num_timesteps)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3_TRAIN_START "
            f"total_steps={self.state.total_steps} start_num_timesteps={self.start_num_timesteps}",
            flush=True,
        )

    def _on_step(self) -> bool:
        now = time.time()
        if now - self.state.last_heartbeat >= self.heartbeat_seconds:
            elapsed = now - self.state.start_time
            steps_done = max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)
            percent = 100.0 * steps_done / max(self.state.total_steps, 1)
            rate = steps_done / max(elapsed, 1e-6)
            eta_seconds = max(self.state.total_steps - steps_done, 0) / max(rate, 1e-6)
            row = {
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "steps": int(steps_done),
                "target_steps": int(self.state.total_steps),
                "percent": float(percent),
                "steps_per_second": float(rate),
                "eta_minutes": float(eta_seconds / 60.0),
            }
            self.rows.append(row)
            print(
                f"[{row['wall_time']}] N3_TRAIN_HEARTBEAT "
                f"steps={steps_done}/{self.state.total_steps} percent={percent:6.2f}% "
                f"rate={rate:8.1f} step/s eta={eta_seconds/60.0:7.2f} min",
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
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3_CHECKPOINT_SAVED step={step} path={path}", flush=True)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase N3 frozen-Gpsi PPO no-shield policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpsi-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-steps", type=int, default=1_500_000)
    parser.add_argument("--checkpoint-steps", nargs="*", type=int, default=[250_000, 500_000, 1_000_000, 1_500_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
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
    for path in [
        out_dir,
        RESULT_DIR,
        RESULT_DIR / "logs",
        RESULT_DIR / "tables",
        RESULT_DIR / "plots",
        RESULT_DIR / "traces",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stop(reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["train_failed"])
    write_text(RESULT_DIR / flag_name, f"{reason}\n{detail}\n")
    write_text(RESULT_DIR / "phase_n3_status.txt", f"stopped:{flag_name}\n")
    report = [
        "# Phase N3 Gpsi-PPO No-Shield Report",
        "",
        f"`terminal_decision = phase_n3_stopped_{reason}`",
        "",
        "Partial report generated by training script.",
        "",
        "```text",
        detail.strip(),
        "```",
    ]
    write_text(RESULT_DIR / "PHASE_N3_GPSI_PPO_NO_SHIELD_REPORT.md", "\n".join(report) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PhaseN3TrainStop("schema_mismatch", f"config is not a mapping: {path}")
    return payload


def save_json_or_yaml(path: Path, payload: dict[str, Any]) -> None:
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


def check_prerequisites(args: argparse.Namespace) -> None:
    if not N2_FLAG.exists():
        raise PhaseN3TrainStop("phase_n2_missing", f"missing Phase N2 complete flag: {rel(N2_FLAG)}")
    checkpoint = ROOT / args.gpsi_checkpoint
    if not checkpoint.exists():
        raise PhaseN3TrainStop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")


def make_env_factory(cfg: dict[str, Any], args: argparse.Namespace, rank: int):
    env_cfg = cfg.get("env", {})
    gpsi_cfg = cfg.get("gpsi", {})
    scenario = str(env_cfg.get("train_scenario", "train_flow_mixed"))
    device = args.device or str(env_cfg.get("device", "cpu"))
    checkpoint = ROOT / args.gpsi_checkpoint

    def _factory():
        env = DynamicObstacleFlowEnv(scenario=scenario)
        env = GpsiObsWrapper(
            env,
            gpsi_checkpoint=checkpoint,
            device=device,
            history_steps=int(gpsi_cfg.get("history_steps", 20)),
            delta_scale=float(gpsi_cfg.get("delta_scale", 5.0)),
            logvar_clamp=tuple(gpsi_cfg.get("logvar_clamp", [-5.0, 3.0])),
            normalize_z=bool(gpsi_cfg.get("normalize_z", False)),
        )
        freeze = env.freeze_check
        if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
            raise PhaseN3TrainStop("gpsi_not_frozen", f"Gpsi freeze check failed: {freeze}")
        if int(env.observation_space["obs"].shape[-1]) != int(gpsi_cfg.get("obs_aug_dim", 94)):
            raise PhaseN3TrainStop(
                "schema_mismatch",
                f"augmented obs dim mismatch: env={env.observation_space['obs'].shape[-1]} config={gpsi_cfg.get('obs_aug_dim', 94)}",
            )
        env = Monitor(env)
        env.reset(seed=args.seed + rank)
        return env

    return _factory


def make_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo_cfg = cfg.get("ppo", {})
    gpsi_cfg = cfg.get("gpsi", {})
    net_arch = ppo_cfg.get("net_arch", {"pi": [128, 128], "vf": [128, 128]})
    return {
        "features_extractor_class": GpsiObstacleSetExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": int(ppo_cfg.get("hidden_dim", 64)),
            "obs_dim": int(gpsi_cfg.get("obs_aug_dim", 94)),
            "use_risk_bias": bool(ppo_cfg.get("use_risk_bias", False)),
            "lambda_bias": float(ppo_cfg.get("lambda_bias", 0.0)),
        },
        "net_arch": net_arch,
        "activation_fn": torch.nn.Tanh,
    }


def validate_wrapper(cfg: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> None:
    try:
        env = make_env_factory(cfg, args, 0)()
        obs, info = env.reset(seed=args.seed)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        unwrapped = env.env if isinstance(env, Monitor) else env
        freeze = unwrapped.freeze_check
        rows = [
            {
                "check": "n2_complete_flag",
                "value": str(N2_FLAG.exists()),
                "detail": rel(N2_FLAG),
            },
            {
                "check": "gpsi_checkpoint_exists",
                "value": str((ROOT / args.gpsi_checkpoint).exists()),
                "detail": rel(ROOT / args.gpsi_checkpoint),
            },
            {
                "check": "gpsi_frozen",
                "value": str(not freeze.training and not freeze.requires_grad_any and freeze.trainable_parameters == 0),
                "detail": json.dumps(freeze.__dict__, sort_keys=True),
            },
            {
                "check": "aug_obs_dim",
                "value": str(int(obs["obs"].shape[-1])),
                "detail": f"next={next_obs['obs'].shape[-1]} reward={float(reward):.6f} done={bool(terminated or truncated)}",
            },
            {
                "check": "history_ratio_available",
                "value": str("gpsi_history_valid_ratio" in next_info),
                "detail": json.dumps(np.asarray(next_info.get("gpsi_history_valid_ratio", [])).tolist()),
            },
            {
                "check": "no_shield",
                "value": "true",
                "detail": "wrapper only augments observation; action_executed remains raw PPO action",
            },
        ]
        write_csv(RESULT_DIR / "tables/phase_n3_schema_check.csv", rows)
        if int(obs["obs"].shape[-1]) != int(cfg.get("gpsi", {}).get("obs_aug_dim", 94)):
            raise PhaseN3TrainStop("schema_mismatch", f"obs dim mismatch: {obs['obs'].shape[-1]}")
        env.close()
    except PhaseN3TrainStop:
        raise
    except Exception as exc:
        raise PhaseN3TrainStop("gpsi_wrapper_failed", traceback.format_exc()) from exc


def write_command_manifest(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    checkpoint = ROOT / args.gpsi_checkpoint
    rows = [
        {
            "stage": "train_smoke" if args.smoke_test else "train_formal",
            "command": " ".join(["python", *sys.argv]),
            "config": rel(ROOT / args.config),
            "gpsi_checkpoint": rel(checkpoint),
            "gpsi_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else "missing",
            "out_dir": rel(out_dir),
            "train_steps": int(args.train_steps),
            "seed": int(args.seed),
            "n_envs": int(args.n_envs or cfg.get("env", {}).get("n_envs", 8)),
        }
    ]
    manifest = RESULT_DIR / "tables/phase_n3_command_manifest.csv"
    old_rows: list[dict[str, Any]] = []
    if manifest.exists() and manifest.stat().st_size > 0:
        try:
            with manifest.open("r", newline="", encoding="utf-8") as handle:
                old_rows = list(csv.DictReader(handle))
        except Exception:
            old_rows = []
    write_csv(manifest, old_rows + rows)


def train() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    try:
        check_prerequisites(args)
        cfg = load_config(ROOT / args.config)
        if args.n_envs is not None:
            cfg.setdefault("env", {})["n_envs"] = int(args.n_envs)
        if args.device is not None:
            cfg.setdefault("env", {})["device"] = args.device
        cfg.setdefault("training", {})["seed"] = int(args.seed)
        cfg["training"]["total_steps"] = int(args.train_steps)
        cfg["training"]["checkpoint_steps"] = [int(step) for step in args.checkpoint_steps]
        cfg["training"]["smoke_test"] = bool(args.smoke_test)
        cfg["gpsi"]["checkpoint"] = args.gpsi_checkpoint
        save_json_or_yaml(out_dir / "config_resolved.yaml", cfg)
        write_command_manifest(args, cfg, out_dir)
        set_seed(args.seed)
        validate_wrapper(cfg, args, out_dir)

        n_envs = int(cfg.get("env", {}).get("n_envs", 8))
        if args.smoke_test:
            n_envs = min(n_envs, 2)
        env_fns = [make_env_factory(cfg, args, rank) for rank in range(n_envs)]
        # Gpsi inference uses torch inside the environment wrapper. DummyVecEnv
        # avoids fork-time torch deadlocks and keeps Gpsi frozen in-process.
        vec_env = DummyVecEnv(env_fns)
        ppo_cfg = cfg.get("ppo", {})
        device = args.device or str(cfg.get("env", {}).get("device", "cpu"))
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3_TRAIN_CONFIG "
            f"steps={args.train_steps} seed={args.seed} n_envs={n_envs} device={device} "
            f"out_dir={rel(out_dir)} no_shield=true safety_cost=false",
            flush=True,
        )
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            seed=args.seed,
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
        heartbeat = HeartbeatCallback(args.train_steps, heartbeat_seconds=args.heartbeat_seconds)
        targets = [step for step in args.checkpoint_steps if step <= args.train_steps]
        checkpoint_cb = FixedNameCheckpointCallback(targets, out_dir)
        model.learn(
            total_timesteps=int(args.train_steps),
            callback=[heartbeat, checkpoint_cb],
            progress_bar=False,
            reset_num_timesteps=True,
        )
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        if not (out_dir / "best_by_eval.zip").exists():
            shutil.copyfile(final_path, out_dir / "best_by_eval.zip")
        if args.train_steps in args.checkpoint_steps:
            expected = out_dir / f"checkpoint_{args.train_steps // 1000}k.zip"
            if not expected.exists():
                shutil.copyfile(final_path, expected)
        write_csv(RESULT_DIR / "tables/phase_n3_train_curve.csv", heartbeat.rows)
        vec_env.close()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3_TRAIN_END final={rel(final_path)}", flush=True)
    except PhaseN3TrainStop as exc:
        write_stop(exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception as exc:
        detail = traceback.format_exc()
        write_stop("train_failed", detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    train()
