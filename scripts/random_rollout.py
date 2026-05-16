from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_env import DynamicObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scenario", type=str, default="train_random_switch")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DynamicObstacleEnv(scenario=args.scenario)

    success_list = []
    collision_list = []
    min_distance_list = []

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RANDOM_ROLLOUT_START "
        f"episodes={args.episodes} scenario={args.scenario} seed={args.seed}",
        flush=True,
    )

    for episode_id in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode_id)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            del obs
            total_reward += reward
            steps += 1
            done = terminated or truncated

        success = int(info["is_success"])
        collision = int(info["is_collision"])
        episode_min_distance = float(info["episode_min_distance"])

        success_list.append(success)
        collision_list.append(collision)
        min_distance_list.append(episode_min_distance)

        print(
            f"episode_id={episode_id} episode_reward={total_reward:.4f} "
            f"success={success} collision={collision} "
            f"episode_min_distance={episode_min_distance:.4f} steps={steps}",
            flush=True,
        )

    print(
        f"random_success_rate={float(np.mean(success_list)):.4f}\n"
        f"random_collision_rate={float(np.mean(collision_list)):.4f}\n"
        f"mean_episode_min_distance={float(np.mean(min_distance_list)):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
