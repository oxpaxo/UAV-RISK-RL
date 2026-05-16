from pathlib import Path
import sys

from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_env import DynamicObstacleEnv


def main() -> None:
    env = DynamicObstacleEnv(
        n_obstacles=3,
        max_obs=3,
        scenario="train_random_switch",
    )
    check_env(env, warn=True)
    print("custom env check OK", flush=True)


if __name__ == "__main__":
    main()
