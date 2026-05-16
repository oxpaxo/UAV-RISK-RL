from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


THREAT_CLASS_ID = {"low": 0.0, "medium": 0.5, "high": 1.0}


@dataclass
class FlowObstacle:
    obstacle_id: int
    position: np.ndarray
    velocity: np.ndarray
    base_velocity: np.ndarray
    motion_mode: str
    threat_class: str
    planned_cpa: float
    planned_ttc: float
    planned_cpa_point: np.ndarray
    planned_nominal_point: np.ndarray
    spawn_step: int
    spawn_time: float
    spawn_reason: str
    lifetime: float
    threat_valid: bool
    params: dict[str, Any] = field(default_factory=dict)


class DynamicObstacleFlowEnv(gym.Env):
    """Continuous dynamic-obstacle flow environment for Phase 1 sanity checks.

    The environment keeps a fixed-size active obstacle flow, replaces obstacles
    during each episode, and records planned CPA/TTC metadata independent of the
    policy trajectory. It is intentionally separate from the legacy 3-obstacle
    environment so old experiments remain frozen.
    """

    metadata = {"render_modes": []}

    SUPPORTED_SCENARIOS = {
        "train_flow_mixed",
        "eval_flow_id",
        "eval_flow_high_density",
        "eval_flow_high_speed",
        "eval_flow_high_threat",
        "eval_flow_mixed_ood",
        "eval_flow_sudden_threat",
    }

    BASE_MOTION_PROBS = {
        "linear": 0.15,
        "sinusoidal_lateral": 0.25,
        "accel_decel": 0.20,
        "ar1_velocity": 0.25,
        "crossing_or_sudden_threat": 0.15,
    }
    BASE_THREAT_PROBS = {"low": 0.30, "medium": 0.45, "high": 0.25}
    CPA_RANGES = {
        "high": (0.35, 1.15),
        "medium": (1.55, 2.35),
        "low": (2.80, 4.20),
    }

    def __init__(
        self,
        scenario: str = "train_flow_mixed",
        space_x: tuple[float, float] = (-18.0, 18.0),
        space_y: tuple[float, float] = (-14.0, 14.0),
        space_z: tuple[float, float] = (0.5, 2.5),
        dt: float = 0.2,
        max_steps: int = 500,
        n_active_min: int | None = None,
        n_active_max: int | None = None,
        n_active_default: int = 6,
        max_obs: int = 10,
        v_uav_max: float = 2.0,
        tau_v: float = 0.25,
        r_uav: float = 0.25,
        r_obs: float = 0.35,
        near_miss_distance: float = 1.5,
        goal_tolerance: float = 0.75,
        d_warning: float = 2.0,
    ) -> None:
        super().__init__()
        if scenario not in self.SUPPORTED_SCENARIOS:
            raise ValueError(f"unsupported DynamicObstacleFlowEnv scenario: {scenario}")

        self.scenario = scenario
        self.space_x = space_x
        self.space_y = space_y
        self.space_z = space_z
        self.low_bounds = np.array([space_x[0], space_y[0], space_z[0]], dtype=np.float32)
        self.high_bounds = np.array([space_x[1], space_y[1], space_z[1]], dtype=np.float32)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.max_obs = int(max_obs)
        self.v_uav_max = float(v_uav_max)
        self.tau_v = float(tau_v)
        self.r_uav = float(r_uav)
        self.r_obs = float(r_obs)
        self.d_collision = self.r_uav + self.r_obs + 0.05
        self.near_miss_distance = float(near_miss_distance)
        self.goal_tolerance = float(goal_tolerance)
        self.d_warning = float(d_warning)
        self.z_nominal = 1.5
        self.eps = 1e-8

        cfg = self._scenario_config(scenario)
        active_min, active_max = cfg["active_range"]
        if n_active_min is not None:
            active_min = int(n_active_min)
        if n_active_max is not None:
            active_max = int(n_active_max)
        if active_min > active_max:
            raise ValueError("n_active_min must be <= n_active_max")
        if active_max > self.max_obs:
            raise ValueError("n_active_max must be <= max_obs")
        self.n_active_min = int(active_min)
        self.n_active_max = int(active_max)
        self.n_active_default = int(np.clip(n_active_default, self.n_active_min, self.n_active_max))
        self.speed_range = tuple(float(v) for v in cfg["speed_range"])
        self.motion_mode_probs = dict(cfg["motion_probs"])
        self.threat_class_probs = dict(cfg["threat_probs"])
        self.ttc_range = tuple(float(v) for v in cfg["ttc_range"])
        self.split = "train" if scenario == "train_flow_mixed" else "eval"

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "ego": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs, 12), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_obs,), dtype=np.float32),
                "global_risk": spaces.Box(low=0.0, high=float(self.max_obs), shape=(2,), dtype=np.float32),
            }
        )

        self.step_count = 0
        self.episode_reward = 0.0
        self.start_pos = np.zeros(3, dtype=np.float32)
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.p_uav = np.zeros(3, dtype=np.float32)
        self.v_uav = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.path_vec = np.zeros(3, dtype=np.float32)
        self.path_len = 1.0
        self.path_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.path_perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.target_active_count = self.n_active_default
        self.obstacles: list[FlowObstacle] = []
        self.next_obstacle_id = 0
        self.spawn_count = 0
        self.remove_count = 0
        self.replacement_count = 0
        self.remove_reason_counts: dict[str, int] = {}
        self.spawn_reason_counts: dict[str, int] = {}
        self.active_count_history: list[int] = []
        self.min_distance_history: list[float] = []
        self.out_of_bounds_count = 0
        self.init_collision = False
        self.episode_min_distance = float("inf")
        self.latest_remove_reason = "none"
        self.latest_spawn_reason = "none"
        self.spawn_records: list[dict[str, Any]] = []

    def _scenario_config(self, scenario: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "active_range": (5, 8),
            "speed_range": (0.3, 2.5),
            "motion_probs": self.BASE_MOTION_PROBS,
            "threat_probs": self.BASE_THREAT_PROBS,
            "ttc_range": (3.2, 9.0),
        }
        if scenario == "eval_flow_high_density":
            cfg["active_range"] = (8, 10)
            cfg["ttc_range"] = (2.8, 8.0)
        elif scenario == "eval_flow_high_speed":
            cfg["speed_range"] = (1.5, 3.0)
            cfg["ttc_range"] = (2.4, 5.2)
        elif scenario == "eval_flow_high_threat":
            cfg["threat_probs"] = {"low": 0.15, "medium": 0.35, "high": 0.50}
            cfg["ttc_range"] = (2.8, 7.0)
        elif scenario == "eval_flow_mixed_ood":
            cfg["speed_range"] = (0.4, 2.8)
            cfg["motion_probs"] = {
                "linear": 0.05,
                "sinusoidal_lateral": 0.25,
                "accel_decel": 0.20,
                "ar1_velocity": 0.25,
                "crossing_or_sudden_threat": 0.25,
            }
            cfg["threat_probs"] = {"low": 0.20, "medium": 0.40, "high": 0.40}
            cfg["ttc_range"] = (2.5, 7.0)
        elif scenario == "eval_flow_sudden_threat":
            cfg["motion_probs"] = {
                "linear": 0.05,
                "sinusoidal_lateral": 0.15,
                "accel_decel": 0.15,
                "ar1_velocity": 0.20,
                "crossing_or_sudden_threat": 0.45,
            }
            cfg["threat_probs"] = {"low": 0.20, "medium": 0.35, "high": 0.45}
            cfg["ttc_range"] = (3.0, 7.5)
        return cfg

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
        self.v_uav = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.obstacles = []
        self.next_obstacle_id = 0
        self.spawn_count = 0
        self.remove_count = 0
        self.replacement_count = 0
        self.remove_reason_counts = {}
        self.spawn_reason_counts = {}
        self.active_count_history = []
        self.min_distance_history = []
        self.out_of_bounds_count = 0
        self.episode_min_distance = float("inf")
        self.latest_remove_reason = "none"
        self.latest_spawn_reason = "none"
        self.spawn_records = []

        self.start_pos, self.goal_pos = self._sample_start_goal()
        self.p_uav = self.start_pos.copy()
        self.path_vec = (self.goal_pos - self.start_pos).astype(np.float32)
        self.path_len = max(float(np.linalg.norm(self.path_vec)), self.eps)
        self.path_dir = self._normalize(self.path_vec)
        self.path_perp = np.array([-self.path_dir[1], self.path_dir[0], 0.0], dtype=np.float32)
        if np.linalg.norm(self.path_perp) < self.eps:
            self.path_perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.path_perp = self._normalize(self.path_perp)

        self.target_active_count = int(self.np_random.integers(self.n_active_min, self.n_active_max + 1))
        for _ in range(self.target_active_count):
            self.obstacles.append(self._spawn_obstacle("reset"))

        initial_min_distance = self._compute_min_distance()
        self.init_collision = bool(initial_min_distance < self.d_collision)
        self.episode_min_distance = initial_min_distance
        self._record_step_stats(initial_min_distance)

        observation = self._get_observation()
        info = self._build_info(is_collision=False, is_success=False, truncated=False)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_for_dynamics = action.copy()
        action_for_dynamics[2] = np.clip(0.8 * (self.z_nominal - self.p_uav[2]) / self.v_uav_max, -0.5, 0.5)

        prev_goal_dist = self._goal_distance()
        v_des = action_for_dynamics * self.v_uav_max
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > self.v_uav_max:
            v_des = v_des / (v_norm + self.eps) * self.v_uav_max
        kappa = 1.0 - math.exp(-self.dt / self.tau_v)
        self.v_uav = (self.v_uav + kappa * (v_des - self.v_uav)).astype(np.float32)
        self.v_uav = self._geofence_velocity(self.p_uav, self.v_uav)
        candidate_uav = (self.p_uav + self.v_uav * self.dt).astype(np.float32)
        out_of_bounds_now = bool(np.any(candidate_uav < self.low_bounds - 1e-5) or np.any(candidate_uav > self.high_bounds + 1e-5))
        if out_of_bounds_now:
            self.out_of_bounds_count += 1
        self.p_uav = np.clip(candidate_uav, self.low_bounds, self.high_bounds).astype(np.float32)

        self._step_obstacles()
        min_distance = self._compute_min_distance()
        self.episode_min_distance = min(self.episode_min_distance, min_distance)
        self.min_distance_history.append(min_distance)

        self._replace_obstacles_if_needed()
        self._maintain_active_count()
        self.active_count_history.append(len(self.obstacles))
        self.step_count += 1

        curr_goal_dist = self._goal_distance()
        progress_reward = prev_goal_dist - curr_goal_dist
        smooth_penalty = -0.02 * float(np.sum((action - self.last_action) ** 2))
        is_collision = bool(min_distance < self.d_collision)
        is_success = bool(curr_goal_dist < self.goal_tolerance and not is_collision)
        terminated = bool(is_collision or is_success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        reward = progress_reward - 0.005 + smooth_penalty
        if is_success:
            reward += 10.0
        if is_collision:
            reward -= 10.0
        if out_of_bounds_now:
            reward -= 0.5
        self.episode_reward += float(reward)
        self.last_action = action.astype(np.float32)

        observation = self._get_observation()
        info = self._build_info(is_collision=is_collision, is_success=is_success, truncated=truncated)
        return observation, float(reward), terminated, truncated, info

    def _sample_start_goal(self) -> tuple[np.ndarray, np.ndarray]:
        direction = 1.0 if self.np_random.random() < 0.5 else -1.0
        y0 = float(self.np_random.uniform(-1.2, 1.2))
        y1 = float(np.clip(y0 + self.np_random.uniform(-1.0, 1.0), self.space_y[0] + 2.0, self.space_y[1] - 2.0))
        if direction > 0.0:
            start = np.array([self.space_x[0] + 1.0, y0, self.z_nominal], dtype=np.float32)
            goal = np.array([self.space_x[1] - 1.0, y1, self.z_nominal], dtype=np.float32)
        else:
            start = np.array([self.space_x[1] - 1.0, y0, self.z_nominal], dtype=np.float32)
            goal = np.array([self.space_x[0] + 1.0, y1, self.z_nominal], dtype=np.float32)
        return start, goal

    def _spawn_obstacle(self, reason: str) -> FlowObstacle:
        obstacle = self._sample_obstacle_with_attempts(reason)
        self.spawn_count += 1
        self.latest_spawn_reason = reason
        self.spawn_reason_counts[reason] = self.spawn_reason_counts.get(reason, 0) + 1
        self.spawn_records.append(self._obstacle_record(obstacle))
        return obstacle

    def _sample_obstacle_with_attempts(self, reason: str) -> FlowObstacle:
        for _ in range(96):
            obstacle = self._sample_obstacle_candidate(reason)
            if self._valid_spawn(obstacle):
                return obstacle
        obstacle = self._sample_obstacle_candidate(reason)
        obstacle.position = np.clip(obstacle.position, self.low_bounds + 0.5, self.high_bounds - 0.5).astype(np.float32)
        obstacle.threat_valid = self._planned_threat_valid(obstacle)
        return obstacle

    def _sample_obstacle_candidate(self, reason: str) -> FlowObstacle:
        threat_class = self._weighted_choice(self.threat_class_probs)
        cpa_low, cpa_high = self.CPA_RANGES[threat_class]
        planned_cpa = float(self.np_random.uniform(cpa_low, cpa_high))
        side = float(self.np_random.choice([-1.0, 1.0]))
        speed = float(self.np_random.uniform(self.speed_range[0], self.speed_range[1]))
        ttc_low, ttc_high = self.ttc_range
        planned_ttc = float(self.np_random.uniform(ttc_low, ttc_high))

        current_s = self._path_progress_scalar(self.p_uav)
        target_s = min(current_s + self.v_uav_max * planned_ttc, self.path_len - 0.75)
        if target_s <= current_s + 1.5:
            target_s = min(self.path_len - 0.5, current_s + 1.5)
            planned_ttc = max((target_s - current_s) / self.v_uav_max, 0.8)
        nominal_point = (self.start_pos + target_s * self.path_dir).astype(np.float32)
        cpa_point = (nominal_point + side * planned_cpa * self.path_perp).astype(np.float32)

        lateral_distance = float(np.clip(speed * planned_ttc, 2.4, 8.5))
        start_pos = cpa_point + side * lateral_distance * self.path_perp
        start_pos += self.np_random.uniform(-0.6, 0.6) * self.path_dir
        start_pos[2] = self.z_nominal + float(self.np_random.uniform(-0.15, 0.15))

        base_velocity = (cpa_point - start_pos) / max(planned_ttc, self.dt)
        base_speed = float(np.linalg.norm(base_velocity))
        if base_speed < self.speed_range[0] or base_speed > self.speed_range[1]:
            base_velocity = self._normalize(base_velocity) * float(np.clip(base_speed, self.speed_range[0], self.speed_range[1]))

        motion_mode = self._weighted_choice(self.motion_mode_probs)
        params = self._motion_params(motion_mode, base_velocity, side, cpa_point, planned_ttc)
        velocity = base_velocity.copy()
        if motion_mode == "crossing_or_sudden_threat":
            velocity = np.asarray(params["pre_velocity"], dtype=np.float32)

        obstacle = FlowObstacle(
            obstacle_id=self.next_obstacle_id,
            position=start_pos.astype(np.float32),
            velocity=velocity.astype(np.float32),
            base_velocity=base_velocity.astype(np.float32),
            motion_mode=motion_mode,
            threat_class=threat_class,
            planned_cpa=planned_cpa,
            planned_ttc=planned_ttc,
            planned_cpa_point=cpa_point.astype(np.float32),
            planned_nominal_point=nominal_point.astype(np.float32),
            spawn_step=int(self.step_count),
            spawn_time=float(self.step_count * self.dt),
            spawn_reason=reason,
            lifetime=float(self.np_random.uniform(16.0, 28.0)),
            threat_valid=True,
            params=params,
        )
        self.next_obstacle_id += 1
        obstacle.threat_valid = self._planned_threat_valid(obstacle)
        return obstacle

    def _motion_params(
        self,
        motion_mode: str,
        base_velocity: np.ndarray,
        side: float,
        cpa_point: np.ndarray,
        planned_ttc: float,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"side": side}
        if motion_mode == "sinusoidal_lateral":
            amp_scale = 1.25 if self.scenario == "eval_flow_mixed_ood" else 1.0
            params.update(
                {
                    "amplitude": float(self.np_random.uniform(0.3, 0.8) * amp_scale),
                    "period": float(self.np_random.uniform(3.0, 6.0)),
                    "phase": float(self.np_random.uniform(0.0, 2.0 * math.pi)),
                    "axis": self.path_dir.copy(),
                }
            )
        elif motion_mode == "accel_decel":
            interval_scale = 0.65 if self.scenario == "eval_flow_mixed_ood" else 1.0
            speed = float(np.linalg.norm(base_velocity))
            params.update(
                {
                    "direction": self._normalize(base_velocity),
                    "source_speed": speed,
                    "target_speed": float(self.np_random.uniform(self.speed_range[0], self.speed_range[1])),
                    "transition_start_time": float(self.step_count * self.dt),
                    "transition_duration": float(self.np_random.uniform(0.5, 1.0)),
                    "next_resample_time": float(self.step_count * self.dt + self.np_random.uniform(1.0, 3.0) * interval_scale),
                    "interval_scale": interval_scale,
                }
            )
        elif motion_mode == "ar1_velocity":
            sigma_hi = 0.30 if self.scenario == "eval_flow_mixed_ood" else 0.20
            params.update(
                {
                    "v_mean": base_velocity.copy(),
                    "phi": 0.90,
                    "sigma_ar": float(self.np_random.uniform(0.05, sigma_hi)),
                    "speed_min": self.speed_range[0],
                    "speed_max": self.speed_range[1],
                }
            )
        elif motion_mode == "crossing_or_sudden_threat":
            turn_delay = float(min(max(planned_ttc * self.np_random.uniform(0.25, 0.45), 0.8), planned_ttc - 0.4))
            pre_velocity = self._normalize(0.85 * self.path_dir + 0.25 * side * self.path_perp) * min(
                max(float(np.linalg.norm(base_velocity)) * 0.55, self.speed_range[0]), self.speed_range[1]
            )
            params.update(
                {
                    "turn_delay": turn_delay,
                    "turn_time": float(self.step_count * self.dt + turn_delay),
                    "pre_velocity": pre_velocity.astype(np.float32),
                    "post_turn_speed": float(np.clip(np.linalg.norm(base_velocity), self.speed_range[0], self.speed_range[1])),
                    "target_point": cpa_point.astype(np.float32),
                    "has_turned": False,
                }
            )
        return params

    def _valid_spawn(self, obstacle: FlowObstacle) -> bool:
        if not self._planned_threat_valid(obstacle):
            return False
        if np.any(obstacle.position < self.low_bounds + 0.25) or np.any(obstacle.position > self.high_bounds - 0.25):
            return False
        if float(np.linalg.norm(obstacle.position - self.p_uav)) <= 2.0:
            return False
        if float(np.linalg.norm(obstacle.position - self.goal_pos)) <= 1.2:
            return False
        for other in self.obstacles:
            if float(np.linalg.norm(other.position - obstacle.position)) < 0.9:
                return False
        return True

    def _planned_threat_valid(self, obstacle: FlowObstacle) -> bool:
        if not np.isfinite(obstacle.planned_cpa) or not np.isfinite(obstacle.planned_ttc):
            return False
        cpa_low, cpa_high = self.CPA_RANGES[obstacle.threat_class]
        if obstacle.planned_cpa < cpa_low - 1e-6 or obstacle.planned_cpa > cpa_high + 1e-6:
            return False
        if obstacle.planned_ttc < 0.5 or obstacle.planned_ttc > 20.0:
            return False
        if float(np.linalg.norm(obstacle.position - self.p_uav)) < self.d_collision + 1.0:
            return False
        return True

    def _step_obstacles(self) -> None:
        time_now = float((self.step_count + 1) * self.dt)
        for obstacle in self.obstacles:
            if obstacle.motion_mode == "linear":
                obstacle.velocity = obstacle.base_velocity.copy()
            elif obstacle.motion_mode == "sinusoidal_lateral":
                params = obstacle.params
                amp = float(params.get("amplitude", 0.5))
                period = max(float(params.get("period", 4.5)), self.dt)
                phase = float(params.get("phase", 0.0))
                axis = np.asarray(params.get("axis", self.path_dir), dtype=np.float32)
                omega = 2.0 * math.pi / period
                obstacle.velocity = (obstacle.base_velocity + amp * omega * math.cos(omega * time_now + phase) * axis).astype(np.float32)
                obstacle.velocity = self._clip_speed(obstacle.velocity)
            elif obstacle.motion_mode == "accel_decel":
                self._step_accel_obstacle(obstacle, time_now)
            elif obstacle.motion_mode == "ar1_velocity":
                params = obstacle.params
                noise = self.np_random.normal(0.0, float(params.get("sigma_ar", 0.1)), size=3).astype(np.float32)
                v_mean = np.asarray(params.get("v_mean", obstacle.base_velocity), dtype=np.float32)
                phi = float(params.get("phi", 0.90))
                obstacle.velocity = (phi * obstacle.velocity + (1.0 - phi) * v_mean + noise).astype(np.float32)
                obstacle.velocity = self._clip_speed(obstacle.velocity)
            elif obstacle.motion_mode == "crossing_or_sudden_threat":
                params = obstacle.params
                if not bool(params.get("has_turned", False)) and time_now >= float(params.get("turn_time", time_now)):
                    target = np.asarray(params.get("target_point", obstacle.planned_cpa_point), dtype=np.float32)
                    speed = float(params.get("post_turn_speed", np.linalg.norm(obstacle.base_velocity)))
                    obstacle.velocity = self._normalize(target - obstacle.position) * speed
                    params["has_turned"] = True
                elif not bool(params.get("has_turned", False)):
                    obstacle.velocity = np.asarray(params.get("pre_velocity", obstacle.velocity), dtype=np.float32)

            obstacle.position = (obstacle.position + obstacle.velocity * self.dt).astype(np.float32)

    def _step_accel_obstacle(self, obstacle: FlowObstacle, time_now: float) -> None:
        params = obstacle.params
        direction = np.asarray(params.get("direction", self._normalize(obstacle.velocity)), dtype=np.float32)
        if time_now >= float(params.get("next_resample_time", time_now + 1.0)):
            current_speed = float(np.linalg.norm(obstacle.velocity))
            params["source_speed"] = current_speed
            params["target_speed"] = float(self.np_random.uniform(self.speed_range[0], self.speed_range[1]))
            params["transition_start_time"] = time_now
            params["transition_duration"] = float(self.np_random.uniform(0.5, 1.0))
            params["next_resample_time"] = time_now + float(self.np_random.uniform(1.0, 3.0)) * float(params.get("interval_scale", 1.0))
            if self.np_random.random() < 0.20:
                direction = self._normalize(direction + 0.12 * self._random_horizontal_unit())
                params["direction"] = direction.astype(np.float32)
        start_time = float(params.get("transition_start_time", time_now))
        duration = max(float(params.get("transition_duration", 0.8)), self.dt)
        alpha = float(np.clip((time_now - start_time) / duration, 0.0, 1.0))
        source_speed = float(params.get("source_speed", np.linalg.norm(obstacle.velocity)))
        target_speed = float(params.get("target_speed", source_speed))
        speed = (1.0 - alpha) * source_speed + alpha * target_speed
        obstacle.velocity = self._clip_speed(direction * speed)

    def _replace_obstacles_if_needed(self) -> None:
        kept: list[FlowObstacle] = []
        for obstacle in self.obstacles:
            remove_reason = self._remove_reason(obstacle)
            if remove_reason is None:
                kept.append(obstacle)
                continue
            self.remove_count += 1
            self.replacement_count += 1
            self.latest_remove_reason = remove_reason
            self.remove_reason_counts[remove_reason] = self.remove_reason_counts.get(remove_reason, 0) + 1
            kept.append(self._spawn_obstacle(f"replace_{remove_reason}"))
        self.obstacles = kept[: self.max_obs]

    def _maintain_active_count(self) -> None:
        while len(self.obstacles) < self.target_active_count:
            self.replacement_count += 1
            self.obstacles.append(self._spawn_obstacle("maintain_active_count"))
        if len(self.obstacles) > self.target_active_count:
            self.obstacles = self.obstacles[: self.target_active_count]

    def _remove_reason(self, obstacle: FlowObstacle) -> str | None:
        age = float(self.step_count * self.dt - obstacle.spawn_time)
        obs_s = self._path_progress_scalar(obstacle.position)
        uav_s = self._path_progress_scalar(self.p_uav)
        dist_to_path = self._distance_to_nominal_line(obstacle.position)
        if np.any(obstacle.position < self.low_bounds - 0.5) or np.any(obstacle.position > self.high_bounds + 0.5):
            return "out_of_bounds"
        if obs_s < uav_s - 2.0:
            return "passed_by_uav"
        if age > obstacle.planned_ttc + 2.0 and obs_s < uav_s + 1.0:
            return "no_future_threat"
        if age > obstacle.lifetime:
            return "lifetime"
        if dist_to_path > 9.0:
            return "far_from_nominal_path"
        return None

    def _compute_min_distance(self) -> float:
        if not self.obstacles:
            return float("inf")
        positions = np.asarray([obs.position for obs in self.obstacles], dtype=np.float32)
        distances = np.linalg.norm(positions - self.p_uav, axis=1)
        return float(np.min(distances))

    def _record_step_stats(self, min_distance: float) -> None:
        self.min_distance_history.append(float(min_distance))
        self.active_count_history.append(len(self.obstacles))

    def _get_observation(self) -> dict[str, np.ndarray]:
        goal_vec = self.goal_pos - self.p_uav
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = self._normalize(goal_vec)
        ego = np.concatenate(
            [
                self.v_uav / self.v_uav_max,
                goal_dir,
                np.array([goal_dist / max(self.path_len, 1.0)], dtype=np.float32),
                self.last_action,
            ]
        ).astype(np.float32)

        obs_array = np.zeros((self.max_obs, 12), dtype=np.float32)
        mask = np.zeros(self.max_obs, dtype=np.float32)
        risk_values = self._proximity_values()
        for idx, obstacle in enumerate(self.obstacles[: self.max_obs]):
            rel_pos = obstacle.position - self.p_uav
            rel_vel = obstacle.velocity - self.v_uav
            distance = float(np.linalg.norm(rel_pos))
            rel_dir = rel_pos / (distance + self.eps)
            closing = -float(np.dot(rel_dir, rel_vel))
            obs_array[idx] = np.array(
                [
                    rel_pos[0] / 20.0,
                    rel_pos[1] / 20.0,
                    rel_pos[2] / 5.0,
                    rel_vel[0] / 3.0,
                    rel_vel[1] / 3.0,
                    rel_vel[2] / 3.0,
                    obstacle.planned_cpa / 5.0,
                    obstacle.planned_ttc / 20.0,
                    distance / 30.0,
                    closing / 3.0,
                    THREAT_CLASS_ID[obstacle.threat_class],
                    risk_values[idx],
                ],
                dtype=np.float32,
            )
            mask[idx] = 1.0
        global_risk = np.array(
            [
                float(np.max(risk_values)) if risk_values else 0.0,
                float(np.sum(risk_values)) if risk_values else 0.0,
            ],
            dtype=np.float32,
        )
        return {"ego": ego, "obs": obs_array, "mask": mask, "global_risk": global_risk}

    def _build_info(self, is_collision: bool, is_success: bool, truncated: bool) -> dict[str, Any]:
        min_distance = self._compute_min_distance()
        active_counts = self.active_count_history or [len(self.obstacles)]
        planned_cpas = [obs.planned_cpa for obs in self.obstacles]
        planned_ttcs = [max(obs.planned_ttc - (self.step_count * self.dt - obs.spawn_time), 0.0) for obs in self.obstacles]
        threat_valid_values = [float(obs.threat_valid) for obs in self.obstacles]
        threat_obstacle = min(self.obstacles, key=lambda item: item.planned_cpa) if self.obstacles else None
        threat_id = int(threat_obstacle.obstacle_id) if threat_obstacle is not None else -1
        threat_index = self.obstacles.index(threat_obstacle) if threat_obstacle is not None else -1
        threat_age = float(self.step_count * self.dt - threat_obstacle.spawn_time) if threat_obstacle is not None else float("nan")
        threat_ttc_remaining = (
            max(float(threat_obstacle.planned_ttc) - threat_age, 0.0) if threat_obstacle is not None else float("nan")
        )
        turn_time = float("nan")
        if threat_obstacle is not None and threat_obstacle.motion_mode == "crossing_or_sudden_threat":
            turn_time = float(threat_obstacle.params.get("turn_time", np.nan))
        info: dict[str, Any] = {
            "scenario": self.scenario,
            "split": self.split,
            "step": int(self.step_count),
            "time": float(self.step_count * self.dt),
            "dt": self.dt,
            "max_steps": self.max_steps,
            "uav_position": self.p_uav.copy(),
            "uav_velocity": self.v_uav.copy(),
            "start_position": self.start_pos.copy(),
            "goal_position": self.goal_pos.copy(),
            "obstacle_positions": np.asarray([obs.position for obs in self.obstacles], dtype=np.float32),
            "obstacle_velocities": np.asarray([obs.velocity for obs in self.obstacles], dtype=np.float32),
            "obstacle_ids": np.asarray([obs.obstacle_id for obs in self.obstacles], dtype=np.int32),
            "obstacle_motion_modes": [obs.motion_mode for obs in self.obstacles],
            "threat_classes": [obs.threat_class for obs in self.obstacles],
            "planned_cpa_values": np.asarray(planned_cpas, dtype=np.float32),
            "planned_ttc_values": np.asarray(planned_ttcs, dtype=np.float32),
            "threat_valid_values": np.asarray(threat_valid_values, dtype=np.float32),
            "threat_valid_rate": float(np.mean(threat_valid_values)) if threat_valid_values else 0.0,
            "threat_obstacle_id": threat_id,
            "threat_obstacle_index": int(threat_index),
            "threat_class": str(threat_obstacle.threat_class) if threat_obstacle is not None else "none",
            "threat_motion_mode": str(threat_obstacle.motion_mode) if threat_obstacle is not None else "none",
            "planned_cpa_to_threat": float(threat_obstacle.planned_cpa) if threat_obstacle is not None else float("nan"),
            "planned_ttc_to_threat": float(threat_obstacle.planned_ttc) if threat_obstacle is not None else float("nan"),
            "planned_ttc_remaining_to_threat": float(threat_ttc_remaining),
            "turn_time": turn_time,
            "min_distance": float(min_distance),
            "episode_min_distance": float(self.episode_min_distance),
            "near_miss": bool(self.episode_min_distance < self.near_miss_distance and not is_collision),
            "is_collision": bool(is_collision),
            "is_success": bool(is_success),
            "truncated": bool(truncated),
            "progress": float(np.clip(self._path_progress_scalar(self.p_uav) / self.path_len, 0.0, 1.0)),
            "goal_distance": float(self._goal_distance()),
            "episode_reward": float(self.episode_reward),
            "active_obstacle_count": int(len(self.obstacles)),
            "target_active_count": int(self.target_active_count),
            "episode_replacement_count": int(self.replacement_count),
            "replacement_count": int(self.replacement_count),
            "spawn_count": int(self.spawn_count),
            "remove_count": int(self.remove_count),
            "spawn_reason": self.latest_spawn_reason,
            "remove_reason": self.latest_remove_reason,
            "spawn_reason_counts": dict(self.spawn_reason_counts),
            "remove_reason_counts": dict(self.remove_reason_counts),
            "obstacle_lifetime": [float(self.step_count * self.dt - obs.spawn_time) for obs in self.obstacles],
            "mean_active_obstacle_count": float(np.mean(active_counts)),
            "max_active_obstacle_count": int(np.max(active_counts)),
            "min_active_obstacle_count": int(np.min(active_counts)),
            "init_collision": bool(self.init_collision),
            "initial_min_distance": float(self.min_distance_history[0]) if self.min_distance_history else float("nan"),
            "out_of_bounds": bool(self.out_of_bounds_count > 0),
            "out_of_bounds_count": int(self.out_of_bounds_count),
            "nan_or_crash": 0,
            "distance_warning_cost": float(max(0.0, self.d_warning - min_distance) ** 2),
            "spawn_records": list(self.spawn_records),
            "motion_mode_prob": dict(self.motion_mode_probs),
            "threat_class_prob": dict(self.threat_class_probs),
            "speed_range": tuple(self.speed_range),
            "active_range": (self.n_active_min, self.n_active_max),
            "fixed_altitude": True,
        }
        return info

    def _proximity_values(self) -> list[float]:
        values: list[float] = []
        for obstacle in self.obstacles[: self.max_obs]:
            distance = float(np.linalg.norm(obstacle.position - self.p_uav))
            values.append(float(np.clip((3.0 - distance) / 3.0, 0.0, 1.0)))
        return values

    def _obstacle_record(self, obstacle: FlowObstacle) -> dict[str, Any]:
        return {
            "obstacle_id": int(obstacle.obstacle_id),
            "spawn_step": int(obstacle.spawn_step),
            "spawn_time": float(obstacle.spawn_time),
            "spawn_reason": obstacle.spawn_reason,
            "motion_mode": obstacle.motion_mode,
            "threat_class": obstacle.threat_class,
            "planned_cpa": float(obstacle.planned_cpa),
            "planned_ttc": float(obstacle.planned_ttc),
            "threat_valid": bool(obstacle.threat_valid),
        }

    def _weighted_choice(self, probabilities: dict[str, float]) -> str:
        keys = list(probabilities.keys())
        probs = np.asarray([probabilities[key] for key in keys], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.np_random.choice(keys, p=probs))

    def _clip_speed(self, velocity: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed < self.eps:
            return self._random_horizontal_unit() * self.speed_range[0]
        clipped = float(np.clip(speed, self.speed_range[0], self.speed_range[1]))
        return (velocity / speed * clipped).astype(np.float32)

    def _geofence_velocity(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        adjusted = velocity.astype(np.float32).copy()
        for axis in range(3):
            next_value = float(position[axis] + adjusted[axis] * self.dt)
            if next_value < float(self.low_bounds[axis]):
                adjusted[axis] = float(self.low_bounds[axis] - position[axis]) / self.dt
            elif next_value > float(self.high_bounds[axis]):
                adjusted[axis] = float(self.high_bounds[axis] - position[axis]) / self.dt
        return adjusted.astype(np.float32)

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_pos - self.p_uav))

    def _path_progress_scalar(self, point: np.ndarray) -> float:
        return float(np.dot(point - self.start_pos, self.path_dir))

    def _distance_to_nominal_line(self, point: np.ndarray) -> float:
        s = float(np.clip(self._path_progress_scalar(point), 0.0, self.path_len))
        closest = self.start_pos + s * self.path_dir
        return float(np.linalg.norm(point - closest))

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < self.eps:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def _random_horizontal_unit(self) -> np.ndarray:
        angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)
