from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.dynamic_obstacle_env import DynamicObstacleEnv
from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
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
        self.heartbeat_seconds = heartbeat_seconds
        self.start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self.start_num_timesteps = int(self.model.num_timesteps)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TRAIN_START total_steps={self.state.total_steps} "
            f"start_num_timesteps={self.start_num_timesteps}",
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
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HEARTBEAT "
                f"steps={steps_done}/{self.state.total_steps} "
                f"percent={percent:6.2f}% rate={rate:8.1f} step/s "
                f"eta={eta_seconds/60.0:7.2f} min",
                flush=True,
            )
            self.state.last_heartbeat = now
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.state.start_time
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TRAIN_END elapsed={elapsed/60.0:.2f} min "
            f"local_steps={max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)} "
            f"model_num_timesteps={self.model.num_timesteps}",
            flush=True,
        )


class TargetCheckpointCallback(BaseCallback):
    def __init__(
        self,
        target_steps: list[int],
        checkpoint_dir: str,
        save_path_prefix: str,
        resume_global_step: int = 0,
    ) -> None:
        super().__init__()
        self.target_steps = sorted(target_steps)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_path_prefix = save_path_prefix
        self.resume_global_step = int(resume_global_step)
        self.saved_steps: set[int] = set()
        self.start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_num_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        current_steps = max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)
        for target_step in self.target_steps:
            if target_step in self.saved_steps:
                continue
            if current_steps >= target_step:
                global_step = self.resume_global_step + target_step
                save_path = self.checkpoint_dir / f"{self.save_path_prefix}_step{global_step}.zip"
                self.model.save(str(save_path))
                self.saved_steps.add(target_step)
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CHECKPOINT_SAVED "
                    f"local_step={target_step} global_step={global_step} path={save_path}",
                    flush=True,
                )
        return True


class SafetyCostWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        use_safety_cost: bool = False,
        cost_type: str = "none",
        beta_cost: float = 5.0,
        fallback_penalty: bool = False,
    ) -> None:
        super().__init__(env)
        if cost_type not in {"none", "distance_warning", "risk_sum"}:
            raise ValueError(f"unsupported cost_type: {cost_type}")
        self.use_safety_cost = bool(use_safety_cost)
        self.cost_type = cost_type
        self.beta_cost = float(beta_cost)
        self.fallback_penalty = bool(fallback_penalty)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_reward = float(reward)
        applied_cost = 0.0
        active = self.use_safety_cost and self.fallback_penalty and self.cost_type != "none"
        if active:
            if self.cost_type == "distance_warning":
                applied_cost = float(info.get("distance_warning_cost", 0.0))
            elif self.cost_type == "risk_sum":
                applied_cost = float(info.get("risk_sum", 0.0))
            reward = base_reward - self.beta_cost * applied_cost
        shaped_reward = float(reward)
        info = dict(info)
        info["base_reward"] = base_reward
        info["applied_cost"] = float(applied_cost)
        info["shaped_reward"] = shaped_reward
        info["fallback_penalty_active"] = bool(active)
        return obs, shaped_reward, terminated, truncated, info


def build_env(
    seed: int,
    env_name: str,
    scenario: str,
    rank: int,
    r_gate: float,
    lambda_ewma: float,
    sigma_min: float,
    d_warning: float,
    use_safety_cost: bool,
    cost_type: str,
    beta_cost: float,
    fallback_penalty: bool,
):
    def _factory():
        if env_name == "DynamicObstacleFlowEnv":
            env = DynamicObstacleFlowEnv(scenario=scenario)
        elif env_name == "DynamicObstacleEnv":
            env = DynamicObstacleEnv(
                scenario=scenario,
                R_gate=r_gate,
                lambda_ewma=lambda_ewma,
                sigma_min=sigma_min,
                d_warning=d_warning,
            )
        else:
            raise ValueError(f"unsupported env: {env_name}")
        env = SafetyCostWrapper(
            env,
            use_safety_cost=use_safety_cost,
            cost_type=cost_type,
            beta_cost=beta_cost,
            fallback_penalty=fallback_penalty,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["DynamicObstacleEnv", "DynamicObstacleFlowEnv"], default="DynamicObstacleEnv")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--profile_mode", type=str, default="full_12")
    parser.add_argument("--agg", choices=["risk", "attention", "mean"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="runs/debug")
    parser.add_argument("--save_path", type=str, default="checkpoints/model.zip")
    parser.add_argument("--scenario", type=str, default="train_random_switch")
    parser.add_argument("--use_rbar", type=str2bool, default=True)
    parser.add_argument("--rbar_floor", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--r_gate", type=float, default=5.0)
    parser.add_argument("--lambda_ewma", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_checkpoints", type=str2bool, default=False)
    parser.add_argument("--checkpoint_steps", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--heartbeat_seconds", type=float, default=30.0)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--resume_global_step", type=int, default=0)
    parser.add_argument("--remaining_steps", type=int, default=0)
    parser.add_argument("--reset_num_timesteps", type=str2bool, default=False)
    parser.add_argument("--use_safety_cost", type=str2bool, default=False)
    parser.add_argument("--cost_type", choices=["none", "distance_warning", "risk_sum"], default="none")
    parser.add_argument("--beta_cost", type=float, default=5.0)
    parser.add_argument("--fallback_penalty", type=str2bool, default=False)
    parser.add_argument("--d_warning", type=float, default=1.0)
    parser.add_argument("--use_risk_bias", type=str2bool, default=False)
    parser.add_argument("--lambda_bias", type=float, default=0.2)
    return parser.parse_args()


def make_policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return dict(
        features_extractor_class=ObstacleSetExtractor,
        features_extractor_kwargs=dict(
            agg_mode=args.agg,
            hidden_dim=args.hidden_dim,
            beta=1.0,
            r_ref=1.0,
            use_rbar=args.use_rbar,
            rbar_floor=args.rbar_floor,
            use_risk_bias=args.use_risk_bias,
            lambda_bias=args.lambda_bias,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )


def write_run_config(args: argparse.Namespace) -> None:
    run_dir = Path(args.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method or args.run_name or Path(args.save_path).stem,
        "profile_mode": args.profile_mode,
        "profile_dim": 12 if args.profile_mode == "full_12" else "unknown",
        "env": args.env,
        "agg": args.agg,
        "seed": args.seed,
        "train_seed": args.seed,
        "eval_seed": 1000,
        "episode_seed_rule": "eval_seed + seed*10000 + episode_id",
        "total_steps": args.total_steps,
        "learn_steps": args.remaining_steps if args.remaining_steps > 0 else args.total_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "use_safety_cost": args.use_safety_cost,
        "cost_type": args.cost_type,
        "fallback_penalty": args.fallback_penalty,
        "beta_cost": args.beta_cost,
        "use_risk_bias": args.use_risk_bias,
        "lambda_bias": args.lambda_bias,
        "R_gate": args.r_gate,
        "lambda_ewma": args.lambda_ewma,
        "rbar_floor": args.rbar_floor,
        "d_safe": 0.8,
        "d_warning": args.d_warning,
        "resume_from": args.resume_from,
        "resume_global_step": args.resume_global_step,
        "remaining_steps": args.remaining_steps,
        "reset_num_timesteps": args.reset_num_timesteps,
        "checkpoint_naming_rule": "checkpoint global_step = resume_global_step + local_step",
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.agg != "risk":
        args.use_rbar = False

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CONFIG "
        f"env={args.env} agg={args.agg} seed={args.seed} total_steps={args.total_steps} "
        f"learn_steps={args.remaining_steps if args.remaining_steps > 0 else args.total_steps} "
        f"n_envs={args.n_envs} device={args.device} scenario={args.scenario} "
        f"use_rbar={args.use_rbar} rbar_floor={args.rbar_floor} hidden_dim={args.hidden_dim} "
        f"r_gate={args.r_gate} lambda_ewma={args.lambda_ewma} sigma_min={args.sigma_min} "
        f"run_name={args.run_name or 'none'} method={args.method or 'none'} "
        f"resume_from={args.resume_from or 'none'} resume_global_step={args.resume_global_step} "
        f"remaining_steps={args.remaining_steps} reset_num_timesteps={args.reset_num_timesteps} "
        f"use_safety_cost={args.use_safety_cost} cost_type={args.cost_type} "
        f"fallback_penalty={args.fallback_penalty} beta_cost={args.beta_cost} "
        f"use_risk_bias={args.use_risk_bias} lambda_bias={args.lambda_bias}",
        flush=True,
    )
    if args.use_safety_cost and args.fallback_penalty:
        print("FALLBACK: cost-penalty, not PPO-Lagrangian.", flush=True)
    write_run_config(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_fns = [
        build_env(
            args.seed,
            args.env,
            args.scenario,
            rank,
            r_gate=args.r_gate,
            lambda_ewma=args.lambda_ewma,
            sigma_min=args.sigma_min,
            d_warning=args.d_warning,
            use_safety_cost=args.use_safety_cost,
            cost_type=args.cost_type,
            beta_cost=args.beta_cost,
            fallback_penalty=args.fallback_penalty,
        )
        for rank in range(args.n_envs)
    ]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        env = DummyVecEnv(env_fns)

    policy_kwargs = make_policy_kwargs(args)
    if args.resume_from:
        learn_steps = args.remaining_steps if args.remaining_steps > 0 else args.total_steps
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RESUME_LOAD path={args.resume_from} "
            f"learn_steps={learn_steps}",
            flush=True,
        )
        model = PPO.load(args.resume_from, env=env, device=args.device)
        reset_num_timesteps = bool(args.reset_num_timesteps)
    else:
        learn_steps = args.total_steps
        model = PPO(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        reset_num_timesteps = True

    callbacks: list[BaseCallback] = [HeartbeatCallback(total_steps=learn_steps, heartbeat_seconds=args.heartbeat_seconds)]
    if args.save_checkpoints and args.checkpoint_steps:
        target_steps = [int(step.strip()) for step in args.checkpoint_steps.split(",") if step.strip()]
        checkpoint_prefix = args.run_name or Path(args.save_path).stem.replace("_final", "")
        if checkpoint_prefix.endswith(f"_s{args.seed}"):
            save_prefix = checkpoint_prefix
        else:
            save_prefix = f"{checkpoint_prefix}_s{args.seed}"
        callbacks.append(
            TargetCheckpointCallback(
                target_steps=target_steps,
                checkpoint_dir=args.checkpoint_dir or os.path.dirname(args.save_path) or ".",
                save_path_prefix=save_prefix,
                resume_global_step=args.resume_global_step,
            )
        )
    callback = callbacks[0] if len(callbacks) == 1 else callbacks
    model.learn(
        total_timesteps=learn_steps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(args.save_path)
    env.close()

    total_batches = math.ceil(learn_steps / max(args.n_envs * 1024, 1))
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SAVED model={args.save_path} approx_updates={total_batches}",
        flush=True,
    )


if __name__ == "__main__":
    main()
