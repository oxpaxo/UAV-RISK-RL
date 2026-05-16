from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DynamicObstacleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        space_x: tuple[float, float] = (-10.0, 10.0),
        space_y: tuple[float, float] = (-10.0, 10.0),
        space_z: tuple[float, float] = (0.0, 5.0),
        n_obstacles: int = 3,
        max_obs: int = 3,
        scenario: str = "train_random_switch",
        r_uav: float = 0.2,
        r_obs: float = 0.3,
        dt: float = 0.2,
        max_steps: int = 200,
        v_uav_max: float = 2.0,
        tau_v: float = 0.1,
        v_obs_min: float = 0.5,
        v_obs_max: float = 1.5,
        p_switch: float = 0.08,
        lambda_ewma: float = 0.1,
        sigma0: float = 0.5,
        sigma_min: float = 0.05,
        d_warning: float = 1.0,
        risk_K: int = 3,
        rho: float = 0.3,
        R_gate: float = 5.0,
        sigma_obs: float = 0.05,
        interaction_episode_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        requested_scenario = scenario
        scenario_variant = "default"
        if scenario.endswith("_high_speed"):
            scenario = scenario[: -len("_high_speed")]
            scenario_variant = "high_speed"
            v_obs_min = 1.5
            v_obs_max = 3.0
        elif scenario.endswith("_small_space"):
            scenario = scenario[: -len("_small_space")]
            scenario_variant = "small_space"
            space_x = (-5.0, 5.0)
            space_y = (-5.0, 5.0)
        if n_obstacles > max_obs:
            raise ValueError("n_obstacles must be <= max_obs")
        if scenario not in {
            "train_random_switch",
            "train_mixed_modes_v2",
            "eval_random_switch",
            "eval_random_switch_hard",
            "mixed_uncertainty",
            "eval_sudden_turn",
            "eval_sinusoidal",
            "eval_accel_decel",
            "eval_ar1",
            "eval_mixed_v2",
            "eval_threat_validated_sudden",
        }:
            raise ValueError(f"unsupported scenario: {scenario}")

        self.space_x = space_x
        self.space_y = space_y
        self.space_z = space_z
        self.low_bounds = np.array([space_x[0], space_y[0], space_z[0]], dtype=np.float32)
        self.high_bounds = np.array([space_x[1], space_y[1], space_z[1]], dtype=np.float32)

        self.n_obstacles = n_obstacles
        self.max_obs = max_obs
        self.requested_scenario = requested_scenario
        self.scenario_variant = scenario_variant
        self.scenario = scenario
        self.r_uav = float(r_uav)
        self.r_obs = float(r_obs)
        self.d_collision = self.r_uav + self.r_obs + 0.05
        self.d_safe = self.r_uav + self.r_obs + 0.30

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.v_uav_max = float(v_uav_max)
        self.tau_v = float(tau_v)
        self.v_obs_min = float(v_obs_min)
        self.v_obs_max = float(v_obs_max)
        self.p_switch = float(p_switch)

        self.lambda_ewma = float(lambda_ewma)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.d_warning = float(d_warning)
        self.risk_K = int(risk_K)
        self.rho = float(rho)
        self.R_gate = float(R_gate)
        self.sigma_obs = float(sigma_obs)
        self.interaction_episode_ratio = float(interaction_episode_ratio)

        self.turn_time = 3.0
        self.turn_step = int(self.turn_time / self.dt)
        self.eps = 1e-8

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "ego": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs, 12), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_obs,), dtype=np.float32),
                "global_risk": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )

        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_min_distance = float("inf")
        self.current_episode_interactive = False
        self.has_turned = False
        self.turning_obstacle_id: int | None = None
        self.threat_obstacle_id: int | None = None
        self.planned_threat_valid = False
        self.threat_valid = False
        self.realized_near_miss = False
        self.scenario_valid = True
        self.initial_min_distance = float("inf")
        self.predicted_cpa_to_nominal_path = float("nan")
        self.threat_invalid_reason = "none"
        self.planned_threat_target_point = np.full(3, np.nan, dtype=np.float32)

        self.start_pos = np.zeros(3, dtype=np.float32)
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.p_uav = np.zeros(3, dtype=np.float32)
        self.v_uav = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)

        self.p_obs = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.v_obs = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.mu_obs = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.sigma_diag = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.risk_values = np.zeros(self.n_obstacles, dtype=np.float32)
        self.obstacle_motion_modes = ["random_switch" for _ in range(self.n_obstacles)]
        self.motion_params: list[dict[str, Any]] = [{} for _ in range(self.n_obstacles)]

    def _variant_speed_min(self, default_min: float) -> float:
        if self.scenario_variant == "high_speed":
            return max(float(default_min), self.v_obs_min)
        return float(default_min)

    def _variant_speed_max(self, default_max: float) -> float:
        if self.scenario_variant == "high_speed":
            return max(float(default_max), self.v_obs_max)
        return float(default_max)

    def _threat_speed_min(self) -> float:
        return self._variant_speed_min(0.8)

    def _threat_speed_max(self) -> float:
        return self._variant_speed_max(1.5)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.step_count = 0
        self.episode_reward = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.v_uav = np.zeros(3, dtype=np.float32)
        self.has_turned = False
        self.turning_obstacle_id = None
        self.threat_obstacle_id = None
        self.planned_threat_valid = False
        self.threat_valid = False
        self.realized_near_miss = False
        self.scenario_valid = True
        self.predicted_cpa_to_nominal_path = float("nan")
        self.threat_invalid_reason = "none"
        self.planned_threat_target_point = np.full(3, np.nan, dtype=np.float32)

        self.current_episode_interactive = self._should_use_interactive_episode()
        self.start_pos, self.goal_pos = self._sample_start_and_goal()
        self.p_uav = self.start_pos.copy()
        self.p_obs, self.v_obs = self._spawn_obstacles(self.current_episode_interactive)
        self.initial_min_distance = self._compute_min_distance()
        self.scenario_valid = bool(np.isfinite(self.initial_min_distance) and self.initial_min_distance > self.d_collision)
        self.planned_threat_valid = self._compute_planned_threat_valid()
        self.threat_valid = self.planned_threat_valid
        self.mu_obs = self.v_obs.copy()
        self.sigma_diag = np.full(
            (self.n_obstacles, 3),
            self.sigma0**2,
            dtype=np.float32,
        )
        self.risk_values = self._compute_risk_values()
        self.episode_min_distance = self._compute_min_distance()

        observation = self._get_observation()
        info = self._build_info(is_collision=False, is_success=False)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        v_des = action * self.v_uav_max
        v_des_norm = np.linalg.norm(v_des)
        if v_des_norm > self.v_uav_max:
            v_des = v_des / (v_des_norm + self.eps) * self.v_uav_max

        prev_goal_dist = self._goal_distance()
        smooth_penalty = -0.05 * float(np.sum((action - self.last_action) ** 2))

        kappa = 1.0 - math.exp(-self.dt / self.tau_v)
        self.v_uav = self.v_uav + kappa * (v_des - self.v_uav)
        self.p_uav = self.p_uav + self.v_uav * self.dt

        self._step_obstacles()
        self._update_ewma()
        self.risk_values = self._compute_risk_values()
        self.step_count += 1

        curr_goal_dist = self._goal_distance()
        min_distance = self._compute_min_distance()
        self.episode_min_distance = min(self.episode_min_distance, min_distance)
        self.realized_near_miss = bool(self.episode_min_distance < 1.5)

        is_collision = min_distance < self.d_collision
        is_success = (curr_goal_dist < 0.5) and not is_collision
        terminated = bool(is_collision or is_success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        progress_reward = prev_goal_dist - curr_goal_dist
        goal_reward = 10.0 if is_success else 0.0
        collision_penalty = -10.0 if is_collision else 0.0
        step_penalty = -0.01
        reward = progress_reward + goal_reward + collision_penalty + smooth_penalty + step_penalty

        self.episode_reward += reward
        self.last_action = action.astype(np.float32)

        observation = self._get_observation()
        info = self._build_info(is_collision=is_collision, is_success=is_success)
        return observation, float(reward), terminated, truncated, info

    def _should_use_interactive_episode(self) -> bool:
        if self.scenario in {
            "eval_sudden_turn",
            "eval_threat_validated_sudden",
            "eval_mixed_v2",
            "eval_sinusoidal",
            "eval_accel_decel",
            "eval_ar1",
        }:
            return True
        if self.scenario == "eval_random_switch_hard":
            return True
        if self.scenario == "mixed_uncertainty":
            return True
        return bool(self.np_random.random() < self.interaction_episode_ratio)

    def _scenario_obstacle_speed_max(self) -> float:
        if self.scenario == "eval_random_switch_hard":
            return max(self.v_obs_max, 1.8)
        return self.v_obs_max

    def _scenario_switch_probability(self) -> float:
        if self.scenario == "eval_random_switch_hard":
            return max(self.p_switch, 0.12)
        return self.p_switch

    def _sample_start_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        direction_sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        x_margin = 0.7
        max_abs_x = max(min(abs(self.space_x[0]), abs(self.space_x[1])) - x_margin, 1.5)
        if max_abs_x < 5.5:
            start_low = max(2.5, 0.72 * max_abs_x)
            start_high = max(start_low + 0.1, max_abs_x)
        else:
            start_low = 5.5
            start_high = min(7.0, max_abs_x)
        start_x = self.np_random.uniform(start_low, start_high)
        goal_x = self.np_random.uniform(start_low, start_high)
        lateral_limit = max((self.space_y[1] - self.space_y[0]) * 0.5 - 0.8, 0.5)
        lateral_goal_span = min(3.2 if self.scenario == "eval_random_switch_hard" else 4.5, lateral_limit)

        if direction_sign > 0:
            start = np.array(
                [-start_x, self.np_random.uniform(-lateral_goal_span, lateral_goal_span), self.np_random.uniform(1.0, 4.0)],
                dtype=np.float32,
            )
            goal = np.array(
                [goal_x, self.np_random.uniform(-lateral_goal_span, lateral_goal_span), self.np_random.uniform(1.0, 4.0)],
                dtype=np.float32,
            )
        else:
            start = np.array(
                [start_x, self.np_random.uniform(-lateral_goal_span, lateral_goal_span), self.np_random.uniform(1.0, 4.0)],
                dtype=np.float32,
            )
            goal = np.array(
                [-goal_x, self.np_random.uniform(-lateral_goal_span, lateral_goal_span), self.np_random.uniform(1.0, 4.0)],
                dtype=np.float32,
            )

        if np.linalg.norm(goal - start) < 8.0:
            goal[1] = np.clip(goal[1] + 4.0 * direction_sign, self.space_y[0] + 1.0, self.space_y[1] - 1.0)
        return start, goal

    def _spawn_obstacles(self, interactive: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.scenario == "mixed_uncertainty":
            return self._spawn_mixed_uncertainty_obstacles()
        if self.scenario in {
            "train_mixed_modes_v2",
            "eval_sinusoidal",
            "eval_accel_decel",
            "eval_ar1",
            "eval_mixed_v2",
            "eval_threat_validated_sudden",
        }:
            return self._spawn_rich_motion_obstacles(interactive)
        positions = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        velocities = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.obstacle_motion_modes = ["random_switch" for _ in range(self.n_obstacles)]
        self.motion_params = [{} for _ in range(self.n_obstacles)]

        main_vec = self.goal_pos - self.start_pos
        main_dir = self._normalize(main_vec)
        perp_1, perp_2 = self._orthonormal_basis(main_dir)

        for obstacle_id in range(self.n_obstacles):
            if interactive:
                position, velocity = self._sample_interactive_obstacle(
                    obstacle_id=obstacle_id,
                    existing_positions=positions[:obstacle_id],
                    main_dir=main_dir,
                    perp_1=perp_1,
                    perp_2=perp_2,
                )
            else:
                position, velocity = self._sample_random_obstacle(positions[:obstacle_id])
            positions[obstacle_id] = position
            velocities[obstacle_id] = velocity

        return positions, velocities

    def _spawn_rich_motion_obstacles(self, interactive: bool) -> tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        velocities = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.obstacle_motion_modes = self._select_rich_motion_modes()
        self.motion_params = [{} for _ in range(self.n_obstacles)]

        main_vec = self.goal_pos - self.start_pos
        main_dir = self._normalize(main_vec)
        perp_1, perp_2 = self._orthonormal_basis(main_dir)

        for obstacle_id, mode in enumerate(self.obstacle_motion_modes):
            if mode == "threat_validated_sudden":
                position, velocity, params = self._sample_planned_threat_obstacle(
                    obstacle_id=obstacle_id,
                    existing_positions=positions[:obstacle_id],
                    main_dir=main_dir,
                    perp_1=perp_1,
                    perp_2=perp_2,
                )
                positions[obstacle_id] = position
                velocities[obstacle_id] = velocity
                self.motion_params[obstacle_id] = params
                continue
            if interactive:
                position, velocity = self._sample_interactive_obstacle(
                    obstacle_id=obstacle_id,
                    existing_positions=positions[:obstacle_id],
                    main_dir=main_dir,
                    perp_1=perp_1,
                    perp_2=perp_2,
                )
            else:
                position, velocity = self._sample_random_obstacle(positions[:obstacle_id])
            positions[obstacle_id] = position
            velocities[obstacle_id] = velocity
            self._init_motion_params(obstacle_id, mode, position, velocity, main_dir, perp_1, perp_2)

        if "threat_validated_sudden" in self.obstacle_motion_modes:
            self.threat_obstacle_id = int(self.obstacle_motion_modes.index("threat_validated_sudden"))
        elif self.scenario == "eval_mixed_v2":
            self.threat_obstacle_id = 0
        return positions, velocities

    def _nominal_path_point(self, time_seconds: float) -> np.ndarray:
        line = self.goal_pos - self.start_pos
        distance = float(np.linalg.norm(line))
        if distance < self.eps:
            return self.start_pos.copy()
        progress = float(np.clip((self.v_uav_max * time_seconds) / distance, 0.0, 1.0))
        return (self.start_pos + progress * line).astype(np.float32)

    def _sample_planned_threat_obstacle(
        self,
        obstacle_id: int,
        existing_positions: np.ndarray,
        main_dir: np.ndarray,
        perp_1: np.ndarray,
        perp_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        d_threat = 1.5
        d_init_min = 2.0
        d_goal_min = 1.5
        max_attempts = 100
        invalid_counts = {
            "init_too_close": 0,
            "goal_too_close": 0,
            "out_of_bounds": 0,
            "cpa_too_large": 0,
            "speed_clip_failed": 0,
            "max_attempts_exceeded": 0,
        }

        for _ in range(max_attempts):
            intercept_after_turn = float(self.np_random.uniform(1.0, 2.5))
            target_time = self.turn_time + intercept_after_turn
            target_point = self._nominal_path_point(target_time)
            lateral_axis = perp_1 if self.np_random.random() < 0.5 else perp_2
            side = float(self.np_random.choice([-1.0, 1.0]))
            lateral_distance = float(self.np_random.uniform(1.8, 2.6))
            vertical_offset = float(self.np_random.uniform(-0.25, 0.25))
            turn_position = target_point + side * lateral_distance * lateral_axis + vertical_offset * perp_2
            speed = float(np.clip(lateral_distance / intercept_after_turn, self._threat_speed_min(), self._threat_speed_max()))
            if speed < self._threat_speed_min() or speed > self._threat_speed_max():
                invalid_counts["speed_clip_failed"] += 1
                continue
            pre_velocity = self._normalize(0.35 * main_dir - 0.65 * side * lateral_axis) * float(
                self.np_random.uniform(self._variant_speed_min(0.6), self._variant_speed_max(1.0))
            )
            position = turn_position - pre_velocity * self.turn_time
            if np.any(position < self.low_bounds + 0.4) or np.any(position > self.high_bounds - 0.4):
                invalid_counts["out_of_bounds"] += 1
                continue
            if float(np.linalg.norm(position - self.start_pos)) <= d_init_min:
                invalid_counts["init_too_close"] += 1
                continue
            if float(np.linalg.norm(position - self.goal_pos)) <= d_goal_min:
                invalid_counts["goal_too_close"] += 1
                continue
            if not self._valid_obstacle_position(position.astype(np.float32), existing_positions):
                invalid_counts["init_too_close"] += 1
                continue
            predicted_cpa = self._distance_to_nominal_segment(target_point)
            if predicted_cpa >= d_threat:
                invalid_counts["cpa_too_large"] += 1
                continue
            post_direction = self._normalize(target_point - turn_position)
            params = {
                "pre_turn_velocity": pre_velocity.astype(np.float32),
                "post_turn_speed": speed,
                "target_point": target_point.astype(np.float32),
                "turn_position": turn_position.astype(np.float32),
                "predicted_cpa_to_nominal_path": predicted_cpa,
                "invalid_reason": "none",
                "invalid_counts": invalid_counts.copy(),
                "speed_clip_applied": False,
            }
            self.threat_obstacle_id = int(obstacle_id)
            self.predicted_cpa_to_nominal_path = float(predicted_cpa)
            self.threat_invalid_reason = "none"
            self.planned_threat_target_point = target_point.astype(np.float32)
            return position.astype(np.float32), pre_velocity.astype(np.float32), params

        invalid_counts["max_attempts_exceeded"] += 1
        position, velocity, params = self._deterministic_crossing_threat(
            obstacle_id=obstacle_id,
            existing_positions=existing_positions,
            main_dir=main_dir,
            perp_1=perp_1,
            perp_2=perp_2,
            invalid_counts=invalid_counts,
        )
        return position, velocity, params

    def _deterministic_crossing_threat(
        self,
        obstacle_id: int,
        existing_positions: np.ndarray,
        main_dir: np.ndarray,
        perp_1: np.ndarray,
        perp_2: np.ndarray,
        invalid_counts: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        intercept_after_turn = 1.6
        target_time = self.turn_time + intercept_after_turn
        target_point = self._nominal_path_point(target_time)
        side = 1.0 if obstacle_id % 2 == 0 else -1.0
        lateral_distance = 2.0
        turn_position = target_point + side * lateral_distance * perp_1
        speed = float(np.clip(lateral_distance / intercept_after_turn, self._threat_speed_min(), self._threat_speed_max()))
        pre_velocity = self._normalize(0.25 * main_dir - 0.75 * side * perp_1) * self._variant_speed_min(0.8)
        position = turn_position - pre_velocity * self.turn_time
        position = np.clip(position, self.low_bounds + 0.6, self.high_bounds - 0.6).astype(np.float32)
        if not self._valid_obstacle_position(position, existing_positions):
            position = np.clip(target_point + side * 2.6 * perp_1 - pre_velocity * self.turn_time, self.low_bounds + 0.6, self.high_bounds - 0.6).astype(np.float32)
        predicted_cpa = self._distance_to_nominal_segment(target_point)
        params = {
            "pre_turn_velocity": pre_velocity.astype(np.float32),
            "post_turn_speed": speed,
            "target_point": target_point.astype(np.float32),
            "turn_position": turn_position.astype(np.float32),
            "predicted_cpa_to_nominal_path": predicted_cpa,
            "invalid_reason": "max_attempts_exceeded",
            "invalid_counts": invalid_counts.copy(),
            "speed_clip_applied": False,
        }
        self.threat_obstacle_id = int(obstacle_id)
        self.predicted_cpa_to_nominal_path = float(predicted_cpa)
        self.threat_invalid_reason = "max_attempts_exceeded"
        self.planned_threat_target_point = target_point.astype(np.float32)
        return position.astype(np.float32), pre_velocity.astype(np.float32), params

    def _distance_to_nominal_segment(self, point: np.ndarray) -> float:
        line = self.goal_pos - self.start_pos
        line_norm_sq = float(np.dot(line, line))
        if line_norm_sq < self.eps:
            return float(np.linalg.norm(point - self.start_pos))
        t = float(np.clip(np.dot(point - self.start_pos, line) / line_norm_sq, 0.0, 1.0))
        closest = self.start_pos + t * line
        return float(np.linalg.norm(point - closest))

    def _select_rich_motion_modes(self) -> list[str]:
        if self.scenario == "eval_sinusoidal":
            return ["sinusoidal" for _ in range(self.n_obstacles)]
        if self.scenario == "eval_accel_decel":
            return ["accel_decel" for _ in range(self.n_obstacles)]
        if self.scenario == "eval_ar1":
            return ["simple_ar1" for _ in range(self.n_obstacles)]
        if self.scenario == "eval_mixed_v2":
            base = ["threat_validated_sudden", "sinusoidal", "simple_ar1"]
            return base[: self.n_obstacles]
        if self.scenario == "eval_threat_validated_sudden":
            base = ["threat_validated_sudden", "random_switch", "random_switch"]
            return base[: self.n_obstacles]
        modes = ["random_switch", "sinusoidal", "accel_decel", "simple_ar1"]
        return [str(self.np_random.choice(modes)) for _ in range(self.n_obstacles)]

    def _init_motion_params(
        self,
        obstacle_id: int,
        mode: str,
        position: np.ndarray,
        velocity: np.ndarray,
        main_dir: np.ndarray,
        perp_1: np.ndarray,
        perp_2: np.ndarray,
    ) -> None:
        speed = float(np.linalg.norm(velocity))
        if speed < self.eps:
            speed = float(self.np_random.uniform(self.v_obs_min, self._scenario_obstacle_speed_max()))
            velocity = self._random_unit_vector() * speed
        direction = self._normalize(velocity)
        params: dict[str, Any] = {
            "direction": direction.astype(np.float32),
            "speed": speed,
            "speed_clip_applied": False,
        }
        if mode == "sinusoidal":
            base_speed = float(np.clip(speed, self._variant_speed_min(0.3), self._variant_speed_max(1.2)))
            lateral = perp_1 if self.np_random.random() < 0.5 else perp_2
            params.update(
                {
                    "base_velocity": direction.astype(np.float32) * base_speed,
                    "amplitude": float(self.np_random.uniform(0.3, 0.8)),
                    "period": float(self.np_random.uniform(3.0, 6.0)),
                    "phase": float(self.np_random.uniform(0.0, 2.0 * np.pi)),
                    "u_perp": lateral.astype(np.float32),
                }
            )
        elif mode == "accel_decel":
            now = 0.0
            params.update(
                {
                    "direction": direction.astype(np.float32),
                    "source_speed": speed,
                    "target_speed": float(self.np_random.uniform(self._variant_speed_min(0.3), self._variant_speed_max(1.5))),
                    "transition_start_time": now,
                    "transition_duration": float(self.np_random.uniform(0.5, 1.0)),
                    "next_resample_time": float(self.np_random.uniform(1.0, 3.0)),
                }
            )
        elif mode == "simple_ar1":
            mean_speed = float(np.clip(speed, self._variant_speed_min(0.3), self._variant_speed_max(1.2)))
            params.update(
                {
                    "v_mean": direction.astype(np.float32) * mean_speed,
                    "phi": 0.90,
                    "sigma_ar": float(self.np_random.uniform(0.05, 0.20)),
                    "speed_min": self._variant_speed_min(0.2),
                    "speed_max": self._variant_speed_max(1.5),
                }
            )
        elif mode == "threat_validated_sudden":
            params.update(
                {
                    "pre_turn_velocity": self._normalize(0.4 * main_dir + 0.6 * perp_1).astype(np.float32)
                    * float(np.clip(speed, self._variant_speed_min(0.6), self._variant_speed_max(1.0))),
                    "post_turn_speed": float(self.np_random.uniform(self._threat_speed_min(), self._threat_speed_max())),
                    "target_t": float(self.np_random.uniform(0.20, 0.55)),
                }
            )
        self.motion_params[obstacle_id] = params

    def _spawn_mixed_uncertainty_obstacles(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        velocities = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        self.obstacle_motion_modes = ["random_switch", "sinusoidal", "threat_validated_sudden"][: self.n_obstacles]
        self.motion_params = [{} for _ in range(self.n_obstacles)]
        line = self.goal_pos - self.start_pos
        main_dir = self._normalize(line)
        perp_1, perp_2 = self._orthonormal_basis(main_dir)

        obstacle_specs = [
            (0.22, 0.45, 0.40, 0.30, 0.6, 1.0),
            (0.38, 0.62, 0.55, 0.35, 0.7, 1.1),
            (0.18, 0.42, 0.20, 0.14, 1.0, 1.4),
        ]
        for obstacle_id, (t_low, t_high, lat1, lat2, speed_low, speed_high) in enumerate(obstacle_specs):
            for _ in range(128):
                t = self.np_random.uniform(t_low, t_high)
                base = self.start_pos + t * line
                candidate = np.clip(
                    base
                    + self.np_random.uniform(-lat1, lat1) * perp_1
                    + self.np_random.uniform(-lat2, lat2) * perp_2,
                    self.low_bounds + 0.4,
                    self.high_bounds - 0.4,
                ).astype(np.float32)
                if (
                    np.linalg.norm(candidate - self.start_pos) > 2.0
                    and np.linalg.norm(candidate - self.goal_pos) > 1.5
                    and self._valid_obstacle_position(candidate, positions[:obstacle_id])
                ):
                    positions[obstacle_id] = candidate
                    break
            speed = self.np_random.uniform(speed_low, speed_high)
            if obstacle_id == 0:
                direction = self._normalize(0.9 * main_dir + 0.1 * perp_1)
            elif obstacle_id == 1:
                direction = self._normalize(0.6 * main_dir + 0.4 * perp_2)
            else:
                target = self.start_pos + 0.3 * line
                direction = self._normalize(target - positions[obstacle_id])
            velocities[obstacle_id] = direction * speed

        return positions, velocities

    def _sample_interactive_obstacle(
        self,
        obstacle_id: int,
        existing_positions: np.ndarray,
        main_dir: np.ndarray,
        perp_1: np.ndarray,
        perp_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        line = self.goal_pos - self.start_pos
        position = np.zeros(3, dtype=np.float32)
        is_hard = self.scenario == "eval_random_switch_hard"
        for _ in range(128):
            if obstacle_id == 0 and is_hard:
                t = self.np_random.uniform(0.08, 0.26)
                lat_1 = self.np_random.uniform(-0.14, 0.14)
                lat_2 = self.np_random.uniform(-0.10, 0.10)
            elif obstacle_id == 1 and is_hard:
                t = self.np_random.uniform(0.18, 0.42)
                lat_1 = self.np_random.uniform(-0.22, 0.22)
                lat_2 = self.np_random.uniform(-0.14, 0.14)
            elif obstacle_id == 2 and is_hard:
                t = self.np_random.uniform(0.30, 0.56)
                lat_1 = self.np_random.uniform(-0.32, 0.32)
                lat_2 = self.np_random.uniform(-0.18, 0.18)
            elif obstacle_id == 0:
                t = self.np_random.uniform(0.12, 0.35)
                lat_1 = self.np_random.uniform(-0.25, 0.25)
                lat_2 = self.np_random.uniform(-0.20, 0.20)
            elif obstacle_id == 1:
                t = self.np_random.uniform(0.28, 0.55)
                lat_1 = self.np_random.uniform(-0.40, 0.40)
                lat_2 = self.np_random.uniform(-0.25, 0.25)
            else:
                t = self.np_random.uniform(0.45, 0.72)
                lat_1 = self.np_random.uniform(-0.55, 0.55)
                lat_2 = self.np_random.uniform(-0.35, 0.35)
            base = self.start_pos + t * line
            lateral = lat_1 * perp_1 + lat_2 * perp_2
            candidate = np.clip(base + lateral, self.low_bounds + 0.4, self.high_bounds - 0.4).astype(np.float32)
            if self._valid_obstacle_position(candidate, existing_positions):
                position = candidate
                break
        else:
            position = np.clip(base, self.low_bounds + 0.4, self.high_bounds - 0.4).astype(np.float32)

        if obstacle_id == 0 and is_hard:
            target_t = self.np_random.uniform(0.02, 0.10)
        elif obstacle_id == 1 and is_hard:
            target_t = self.np_random.uniform(0.10, 0.26)
        elif obstacle_id == 2 and is_hard:
            target_t = self.np_random.uniform(0.20, 0.40)
        elif obstacle_id == 0:
            target_t = self.np_random.uniform(0.05, 0.20)
        elif obstacle_id == 1:
            target_t = self.np_random.uniform(0.18, 0.40)
        else:
            target_t = self.np_random.uniform(0.32, 0.58)
        target = self.start_pos + target_t * line
        intercept_dir = self._normalize(target - position)
        cross_dir = self._normalize(
            self.np_random.choice([-1.0, 1.0]) * perp_1
            + self.np_random.uniform(-0.3, 0.3) * perp_2
            + self.np_random.uniform(-0.2, 0.2) * main_dir
        )
        mix = 0.96 if is_hard and self.np_random.random() < 0.95 else 0.84 if is_hard else 0.92 if self.np_random.random() < 0.9 else 0.75
        direction = self._normalize(mix * intercept_dir + (1.0 - mix) * cross_dir)
        speed = self.np_random.uniform(
            max(1.1 if is_hard else 0.9, self.v_obs_min),
            self._scenario_obstacle_speed_max(),
        )
        return position, (direction * speed).astype(np.float32)

    def _sample_random_obstacle(self, existing_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(128):
            candidate = np.array(
                [
                    self.np_random.uniform(self.space_x[0] + 0.5, self.space_x[1] - 0.5),
                    self.np_random.uniform(self.space_y[0] + 0.5, self.space_y[1] - 0.5),
                    self.np_random.uniform(self.space_z[0] + 0.5, self.space_z[1] - 0.5),
                ],
                dtype=np.float32,
            )
            if self._valid_obstacle_position(candidate, existing_positions):
                position = candidate
                break
        else:
            position = candidate

        direction = self._random_unit_vector()
        speed = self.np_random.uniform(self.v_obs_min, self.v_obs_max)
        return position, (direction * speed).astype(np.float32)

    def _valid_obstacle_position(self, candidate: np.ndarray, existing_positions: np.ndarray) -> bool:
        if np.linalg.norm(candidate - self.start_pos) < 1.3:
            return False
        if np.linalg.norm(candidate - self.goal_pos) < 1.3:
            return False
        if existing_positions.size > 0:
            distances = np.linalg.norm(existing_positions - candidate, axis=1)
            if np.any(distances < 1.0):
                return False
        return True

    def _step_obstacles(self) -> None:
        if self.scenario == "mixed_uncertainty":
            self._step_mixed_uncertainty_obstacles()
            return
        if self.scenario in {
            "train_mixed_modes_v2",
            "eval_sinusoidal",
            "eval_accel_decel",
            "eval_ar1",
            "eval_mixed_v2",
            "eval_threat_validated_sudden",
        }:
            self._step_rich_motion_obstacles()
            return
        for obstacle_id in range(self.n_obstacles):
            if self.scenario == "eval_sudden_turn" and self.has_turned and obstacle_id == self.turning_obstacle_id:
                pass
            elif self.np_random.random() < self._scenario_switch_probability():
                self._randomize_obstacle_direction(obstacle_id)

        self.p_obs = self.p_obs + self.v_obs * self.dt
        self._reflect_obstacles()

        if (
            self.scenario == "eval_sudden_turn"
            and not self.has_turned
            and (self.step_count + 1) >= self.turn_step
        ):
            distances = np.linalg.norm(self.p_obs - self.p_uav, axis=1)
            self.turning_obstacle_id = int(np.argmin(distances))
            direction = self._normalize(self.p_uav - self.p_obs[self.turning_obstacle_id])
            self.v_obs[self.turning_obstacle_id] = direction * self.v_obs_max
            self.has_turned = True

    def _step_rich_motion_obstacles(self) -> None:
        time_now = (self.step_count + 1) * self.dt
        for obstacle_id, mode in enumerate(self.obstacle_motion_modes):
            if mode == "random_switch":
                if self.np_random.random() < self._scenario_switch_probability():
                    self._randomize_obstacle_direction(obstacle_id)
            elif mode == "sinusoidal":
                self._step_sinusoidal_obstacle(obstacle_id, time_now)
            elif mode == "accel_decel":
                self._step_accel_decel_obstacle(obstacle_id, time_now)
            elif mode == "simple_ar1":
                self._step_ar1_obstacle(obstacle_id)
            elif mode == "threat_validated_sudden":
                self._step_threat_validated_sudden_obstacle(obstacle_id, time_now)

        self.p_obs = self.p_obs + self.v_obs * self.dt
        self._reflect_obstacles()

    def _step_sinusoidal_obstacle(self, obstacle_id: int, time_now: float) -> None:
        params = self.motion_params[obstacle_id]
        base_velocity = np.asarray(params.get("base_velocity", self.v_obs[obstacle_id]), dtype=np.float32)
        u_perp = np.asarray(params.get("u_perp", np.array([0.0, 1.0, 0.0], dtype=np.float32)), dtype=np.float32)
        amplitude = float(params.get("amplitude", 0.5))
        period = max(float(params.get("period", 4.5)), self.dt)
        phase = float(params.get("phase", 0.0))
        omega = 2.0 * np.pi / period
        lateral_velocity = amplitude * omega * math.cos(omega * time_now + phase) * u_perp
        self.v_obs[obstacle_id] = (base_velocity + lateral_velocity).astype(np.float32)
        self._clip_obstacle_speed(obstacle_id, self._variant_speed_min(0.2), self._variant_speed_max(1.7))

    def _step_accel_decel_obstacle(self, obstacle_id: int, time_now: float) -> None:
        params = self.motion_params[obstacle_id]
        direction = np.asarray(params.get("direction", self._normalize(self.v_obs[obstacle_id])), dtype=np.float32)
        current_speed = float(np.linalg.norm(self.v_obs[obstacle_id]))
        next_resample_time = float(params.get("next_resample_time", 1.0))
        if time_now >= next_resample_time:
            params["source_speed"] = current_speed
            params["target_speed"] = float(self.np_random.uniform(self._variant_speed_min(0.3), self._variant_speed_max(1.5)))
            params["transition_start_time"] = time_now
            params["transition_duration"] = float(self.np_random.uniform(0.5, 1.0))
            params["next_resample_time"] = time_now + float(self.np_random.uniform(1.0, 3.0))
            if self.np_random.random() < 0.25:
                perturb = 0.12 * self._random_unit_vector()
                direction = self._normalize(direction + perturb)
                params["direction"] = direction.astype(np.float32)

        start_time = float(params.get("transition_start_time", time_now))
        duration = max(float(params.get("transition_duration", 0.8)), self.dt)
        alpha = float(np.clip((time_now - start_time) / duration, 0.0, 1.0))
        source_speed = float(params.get("source_speed", current_speed))
        target_speed = float(params.get("target_speed", current_speed))
        speed = (1.0 - alpha) * source_speed + alpha * target_speed
        self.v_obs[obstacle_id] = direction * speed
        self._clip_obstacle_speed(obstacle_id, self._variant_speed_min(0.2), self._variant_speed_max(1.6))

    def _step_ar1_obstacle(self, obstacle_id: int) -> None:
        params = self.motion_params[obstacle_id]
        phi = float(params.get("phi", 0.90))
        sigma_ar = float(params.get("sigma_ar", 0.1))
        v_mean = np.asarray(params.get("v_mean", self.v_obs[obstacle_id]), dtype=np.float32)
        noise = self.np_random.normal(0.0, sigma_ar, size=3).astype(np.float32)
        self.v_obs[obstacle_id] = (phi * self.v_obs[obstacle_id] + (1.0 - phi) * v_mean + noise).astype(np.float32)
        clipped = self._clip_obstacle_speed(obstacle_id, float(params.get("speed_min", 0.2)), float(params.get("speed_max", 1.5)))
        params["speed_clip_applied"] = bool(params.get("speed_clip_applied", False) or clipped)

    def _step_threat_validated_sudden_obstacle(self, obstacle_id: int, time_now: float) -> None:
        params = self.motion_params[obstacle_id]
        if not self.has_turned and time_now >= self.turn_time:
            self.turning_obstacle_id = int(obstacle_id)
            self.threat_obstacle_id = int(obstacle_id)
            target = np.asarray(params.get("target_point", self._nominal_path_point(self.turn_time + 1.5)), dtype=np.float32)
            direction = self._normalize(target - self.p_obs[obstacle_id])
            self.v_obs[obstacle_id] = direction * float(params.get("post_turn_speed", 1.2))
            self.has_turned = True
        elif not self.has_turned:
            self.v_obs[obstacle_id] = np.asarray(params.get("pre_turn_velocity", self.v_obs[obstacle_id]), dtype=np.float32)

    def _clip_obstacle_speed(self, obstacle_id: int, speed_min: float, speed_max: float) -> bool:
        speed = float(np.linalg.norm(self.v_obs[obstacle_id]))
        clipped = False
        if speed < self.eps:
            self.v_obs[obstacle_id] = self._random_unit_vector() * speed_min
            return True
        new_speed = float(np.clip(speed, speed_min, speed_max))
        if abs(new_speed - speed) > 1e-6:
            clipped = True
        self.v_obs[obstacle_id] = self._normalize(self.v_obs[obstacle_id]) * new_speed
        return clipped

    def _step_mixed_uncertainty_obstacles(self) -> None:
        time_now = (self.step_count + 1) * self.dt
        linear_speed = np.linalg.norm(self.v_obs[0])
        if linear_speed < self.eps:
            linear_speed = 0.8
        self.v_obs[0] = self._normalize(self.v_obs[0]) * linear_speed

        smooth_speed = np.linalg.norm(self.v_obs[1])
        if smooth_speed < self.eps:
            smooth_speed = 0.9
        base_dir = self._normalize(self.goal_pos - self.start_pos)
        perp_1, perp_2 = self._orthonormal_basis(base_dir)
        omega = 2.0 * np.pi / 4.0
        smooth_dir = self._normalize(base_dir + 0.7 * np.cos(omega * time_now) * perp_1 + 0.4 * np.sin(omega * time_now) * perp_2)
        self.v_obs[1] = smooth_dir * min(smooth_speed, 1.2)

        threat_speed = np.linalg.norm(self.v_obs[2])
        if threat_speed < self.eps:
            threat_speed = 1.2
        if not self.has_turned and time_now >= self.turn_time:
            self.turning_obstacle_id = 2
            direction = self._normalize(self.p_uav - self.p_obs[2])
            self.v_obs[2] = direction * min(max(threat_speed, 1.2), 1.6)
            self.has_turned = True
        elif not self.has_turned:
            self.v_obs[2] = self._normalize(0.2 * base_dir + 0.8 * perp_1) * min(threat_speed, 1.0)

        self.p_obs = self.p_obs + self.v_obs * self.dt
        self._reflect_obstacles()

    def _randomize_obstacle_direction(self, obstacle_id: int) -> None:
        speed = np.linalg.norm(self.v_obs[obstacle_id])
        if speed < self.eps:
            speed = self.np_random.uniform(self.v_obs_min, self._scenario_obstacle_speed_max())

        is_hard = self.scenario == "eval_random_switch_hard"
        interactive_retarget_prob = 0.9 if is_hard else 0.8
        if self.current_episode_interactive and self.np_random.random() < interactive_retarget_prob:
            line = self.goal_pos - self.start_pos
            main_dir = self._normalize(line)
            perp_1, perp_2 = self._orthonormal_basis(main_dir)
            target_t = self.np_random.uniform(0.03, 0.55) if is_hard else self.np_random.uniform(0.05, 0.65)
            target = self.start_pos + target_t * line
            target_dir = self._normalize(target - self.p_obs[obstacle_id])
            cross_dir = self._normalize(
                self.np_random.choice([-1.0, 1.0]) * perp_1
                + self.np_random.uniform(-0.12, 0.12) * perp_2
                if is_hard
                else self.np_random.choice([-1.0, 1.0]) * perp_1 + self.np_random.uniform(-0.20, 0.20) * perp_2
            )
            chase_dir = self._normalize(self.p_uav - self.p_obs[obstacle_id])
            if is_hard:
                direction = self._normalize(0.30 * target_dir + 0.55 * chase_dir + 0.15 * cross_dir)
            else:
                direction = self._normalize(0.45 * target_dir + 0.35 * chase_dir + 0.20 * cross_dir)
        else:
            direction = self._random_unit_vector()

        self.v_obs[obstacle_id] = direction * speed

    def _reflect_obstacles(self) -> None:
        for obstacle_id in range(self.n_obstacles):
            for axis in range(3):
                low = self.low_bounds[axis]
                high = self.high_bounds[axis]
                if self.p_obs[obstacle_id, axis] < low:
                    self.p_obs[obstacle_id, axis] = low + (low - self.p_obs[obstacle_id, axis])
                    self.v_obs[obstacle_id, axis] *= -1.0
                elif self.p_obs[obstacle_id, axis] > high:
                    self.p_obs[obstacle_id, axis] = high - (self.p_obs[obstacle_id, axis] - high)
                    self.v_obs[obstacle_id, axis] *= -1.0
            self.p_obs[obstacle_id] = np.clip(self.p_obs[obstacle_id], self.low_bounds, self.high_bounds)

    def _update_ewma(self) -> None:
        innovation = self.v_obs - self.mu_obs
        self.mu_obs = self.mu_obs + self.lambda_ewma * innovation
        self.sigma_diag = (1.0 - self.lambda_ewma) * self.sigma_diag + self.lambda_ewma * (innovation**2)
        self.sigma_diag = np.maximum(self.sigma_diag, self.sigma_min**2).astype(np.float32)

    def _compute_risk_values(self) -> np.ndarray:
        risk_values = np.zeros(self.n_obstacles, dtype=np.float32)
        for obstacle_id in range(self.n_obstacles):
            p_rel = self.p_obs[obstacle_id] - self.p_uav
            mu_rel = self.mu_obs[obstacle_id] - self.v_uav
            base_distance = np.linalg.norm(p_rel)
            gate = math.exp(-base_distance / self.R_gate)

            max_risk = 0.0
            for k in range(1, self.risk_K + 1):
                dt_k = k * self.dt
                p_rel_k = p_rel + mu_rel * dt_k
                p_rel_norm = np.linalg.norm(p_rel_k)
                margin = p_rel_norm - self.d_safe
                sigma_pred = self.sigma_diag[obstacle_id] * (dt_k**2)
                direction = p_rel_k / (p_rel_norm + self.eps)

                sigma_rad = float(np.sum((direction**2) * sigma_pred))
                sigma_lat = float(np.sum(sigma_pred) - sigma_rad)
                sigma_eff = sigma_rad + self.rho * max(sigma_lat, 0.0) + self.sigma_obs**2 + self.eps
                risk_k = math.exp(-0.5 * max(margin, 0.0) ** 2 / sigma_eff)
                max_risk = max(max_risk, risk_k)

            risk_values[obstacle_id] = float(np.clip(gate * max_risk, 0.0, 1.0))
        return risk_values

    def _compute_min_distance(self) -> float:
        distances = np.linalg.norm(self.p_obs - self.p_uav, axis=1)
        return float(np.min(distances))

    def _compute_planned_threat_valid(self) -> bool:
        if self.scenario not in {"eval_mixed_v2", "eval_threat_validated_sudden", "mixed_uncertainty"}:
            return True
        threat_id = self.threat_obstacle_id
        if threat_id is None:
            if "threat_validated_sudden" in self.obstacle_motion_modes:
                threat_id = int(self.obstacle_motion_modes.index("threat_validated_sudden"))
            else:
                threat_id = 0
            self.threat_obstacle_id = int(threat_id)
        if threat_id < 0 or threat_id >= self.n_obstacles:
            self.threat_invalid_reason = "max_attempts_exceeded"
            return False
        if self.initial_min_distance <= self.d_collision:
            self.threat_invalid_reason = "init_too_close"
            return False
        init_distance = float(np.linalg.norm(self.p_obs[threat_id] - self.start_pos))
        goal_distance = float(np.linalg.norm(self.p_obs[threat_id] - self.goal_pos))
        if init_distance <= 2.0:
            self.threat_invalid_reason = "init_too_close"
            return False
        if goal_distance <= 1.5:
            self.threat_invalid_reason = "goal_too_close"
            return False
        cpa = float(self.predicted_cpa_to_nominal_path)
        if not np.isfinite(cpa):
            cpa = float(self.motion_params[threat_id].get("predicted_cpa_to_nominal_path", np.nan))
            self.predicted_cpa_to_nominal_path = cpa
        if not np.isfinite(cpa) or cpa >= 1.5:
            self.threat_invalid_reason = "cpa_too_large"
            return False
        params = self.motion_params[threat_id]
        invalid_reason = str(params.get("invalid_reason", "none"))
        if invalid_reason != "none":
            self.threat_invalid_reason = invalid_reason
            return False
        speed = float(params.get("post_turn_speed", np.nan))
        if not np.isfinite(speed) or speed < self._threat_speed_min() or speed > self._threat_speed_max():
            self.threat_invalid_reason = "speed_clip_failed"
            return False
        self.threat_invalid_reason = "none"
        return True

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_pos - self.p_uav))

    def _get_observation(self) -> dict[str, np.ndarray]:
        goal_vec = self.goal_pos - self.p_uav
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + self.eps)

        ego = np.concatenate(
            [
                self.v_uav / self.v_uav_max,
                goal_dir,
                np.array([goal_dist / 20.0], dtype=np.float32),
                self.last_action,
            ]
        ).astype(np.float32)

        obs = np.zeros((self.max_obs, 12), dtype=np.float32)
        mask = np.zeros(self.max_obs, dtype=np.float32)
        for obstacle_id in range(self.n_obstacles):
            rel_pos = self.p_obs[obstacle_id] - self.p_uav
            mu_rel = self.mu_obs[obstacle_id] - self.v_uav
            distance = np.linalg.norm(rel_pos)
            d_dot = float(np.dot(rel_pos / (distance + self.eps), mu_rel))

            obs[obstacle_id] = np.array(
                [
                    rel_pos[0] / 10.0,
                    rel_pos[1] / 10.0,
                    rel_pos[2] / 10.0,
                    mu_rel[0] / 2.0,
                    mu_rel[1] / 2.0,
                    mu_rel[2] / 2.0,
                    self.sigma_diag[obstacle_id, 0] / 1.0,
                    self.sigma_diag[obstacle_id, 1] / 1.0,
                    self.sigma_diag[obstacle_id, 2] / 1.0,
                    distance / 20.0,
                    d_dot / 2.0,
                    self.risk_values[obstacle_id],
                ],
                dtype=np.float32,
            )
            mask[obstacle_id] = 1.0

        global_risk = np.array(
            [
                float(np.max(self.risk_values)) if self.risk_values.size else 0.0,
                float(np.sum(self.risk_values)),
            ],
            dtype=np.float32,
        )
        return {
            "ego": ego,
            "obs": obs,
            "mask": mask,
            "global_risk": global_risk,
        }

    def _build_info(self, is_collision: bool, is_success: bool) -> dict[str, Any]:
        info: dict[str, Any] = {
            "min_distance": float(self._compute_min_distance()),
            "requested_scenario": self.requested_scenario,
            "scenario_variant": self.scenario_variant,
            "episode_min_distance": float(self.episode_min_distance),
            "is_collision": bool(is_collision),
            "is_success": bool(is_success),
            "risk_values": self.risk_values.copy(),
            "sigma_values": self.sigma_diag.copy(),
            "mu_values": self.mu_obs.copy(),
            "step": int(self.step_count),
            "time": float(self.step_count * self.dt),
            "episode_reward": float(self.episode_reward),
            "goal_distance": float(self._goal_distance()),
            "uav_position": self.p_uav.copy(),
            "goal_position": self.goal_pos.copy(),
            "uav_velocity": self.v_uav.copy(),
            "obstacle_positions": self.p_obs.copy(),
            "obstacle_velocities": self.v_obs.copy(),
            "turn_step": int(self.turn_step),
            "has_turned": bool(self.has_turned),
            "turning_obstacle_id": int(self.turning_obstacle_id) if self.turning_obstacle_id is not None else -1,
            "is_interactive_episode": bool(self.current_episode_interactive),
            "obstacle_motion_modes": list(self.obstacle_motion_modes),
            "scenario_valid": bool(self.scenario_valid),
            "planned_threat_valid": bool(self.planned_threat_valid),
            "threat_valid": bool(self.threat_valid),
            "realized_near_miss": bool(self.realized_near_miss),
            "threat_obstacle_id": int(self.threat_obstacle_id) if self.threat_obstacle_id is not None else -1,
            "threat_valid_rate": float(bool(self.threat_valid)),
            "planned_threat_valid_rate": float(bool(self.planned_threat_valid)),
            "realized_near_miss_rate": float(bool(self.realized_near_miss)),
            "predicted_cpa_to_nominal_path": float(self.predicted_cpa_to_nominal_path),
            "invalid_reason": str(self.threat_invalid_reason),
            "threat_invalid_reason": str(self.threat_invalid_reason),
            "planned_threat_target_point": self.planned_threat_target_point.copy(),
            "initial_min_distance": float(self.initial_min_distance),
            "current_target_speed": [
                float(params.get("target_speed", np.nan)) if isinstance(params, dict) else np.nan
                for params in self.motion_params
            ],
            "next_resample_time": [
                float(params.get("next_resample_time", np.nan)) if isinstance(params, dict) else np.nan
                for params in self.motion_params
            ],
            "ar1_phi": [
                float(params.get("phi", np.nan)) if isinstance(params, dict) else np.nan
                for params in self.motion_params
            ],
            "ar1_sigma": [
                float(params.get("sigma_ar", np.nan)) if isinstance(params, dict) else np.nan
                for params in self.motion_params
            ],
            "speed_clip_applied": [
                bool(params.get("speed_clip_applied", False)) if isinstance(params, dict) else False
                for params in self.motion_params
            ],
            "distance_warning_cost": float(max(0.0, self.d_warning - self._compute_min_distance()) ** 2),
            "risk_sum": float(np.sum(self.risk_values)),
            "risk_max": float(np.max(self.risk_values)) if self.risk_values.size else 0.0,
        }
        return info

    def _orthonormal_basis(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp_1 = np.cross(direction, ref)
        perp_1 = self._normalize(perp_1)
        perp_2 = np.cross(direction, perp_1)
        perp_2 = self._normalize(perp_2)
        return perp_1.astype(np.float32), perp_2.astype(np.float32)

    def _random_unit_vector(self) -> np.ndarray:
        vec = self.np_random.normal(0.0, 1.0, size=3).astype(np.float32)
        return self._normalize(vec)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < self.eps:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)
