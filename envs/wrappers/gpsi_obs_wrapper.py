from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from models.gpsi_head_a import GpsiHeadA


@dataclass(frozen=True)
class GpsiFreezeCheck:
    training: bool
    trainable_parameters: int
    total_parameters: int
    requires_grad_any: bool


class GpsiObsWrapper(gym.Wrapper):
    """Append frozen Gpsi-HeadA features to each active obstacle profile.

    The wrapper owns obstacle history because EnvV2 exposes obstacle ids and
    physical state only through reset/step infos. Histories are keyed by
    obstacle_id, so a replaced obstacle starts a new left-padded sequence.
    """

    def __init__(
        self,
        env: gym.Env,
        gpsi_checkpoint: str | Path,
        device: str | torch.device = "cpu",
        history_steps: int = 20,
        delta_scale: float = 5.0,
        logvar_clamp: tuple[float, float] = (-5.0, 3.0),
        normalize_z: bool = False,
        z_stats_path: str | Path | None = None,
        z_std_floor: float = 1.0e-3,
        include_z: bool = True,
        z_transform: str | None = None,
        z_l2_target_norm: float = 4.0,
        z_l2_eps: float = 1.0e-6,
        z_layernorm_alpha: float = 0.5,
        z_layernorm_eps: float = 1.0e-5,
        include_logvar: bool = True,
        logvar_output_scale: float = 1.0,
        degenerate_std_threshold: float = 1.0e-5,
        degenerate_std_floor: float = 1.0,
        expose_debug: bool = True,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("GpsiObsWrapper requires a Dict observation space")
        obs_space = env.observation_space["obs"]
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 2:
            raise TypeError("GpsiObsWrapper requires obs space shape [max_obs, obs_dim]")
        if int(obs_space.shape[1]) != 12:
            raise ValueError(f"GpsiObsWrapper expected base obs_dim=12, got {obs_space.shape[1]}")

        self.device = torch.device(device)
        self.history_steps = int(history_steps)
        self.delta_scale = float(delta_scale)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.normalize_z = bool(normalize_z)
        self.z_stats_path = str(z_stats_path) if z_stats_path is not None else ""
        self.z_std_floor = float(z_std_floor)
        self.include_z = bool(include_z)
        if z_transform is None:
            z_transform = "standardize" if self.normalize_z else "raw"
        self.z_transform = str(z_transform)
        if self.z_transform == "none":
            self.z_transform = "raw"
        valid_z_transforms = {"raw", "standardize", "l2_scale", "layernorm"}
        if self.z_transform not in valid_z_transforms:
            raise ValueError(f"unsupported z_transform={self.z_transform!r}; expected one of {sorted(valid_z_transforms)}")
        if self.normalize_z and self.z_transform not in {"standardize"}:
            raise ValueError("normalize_z=True is only compatible with z_transform='standardize'")
        if self.z_transform == "standardize":
            self.normalize_z = True
        self.z_l2_target_norm = float(z_l2_target_norm)
        self.z_l2_eps = float(z_l2_eps)
        self.z_layernorm_alpha = float(z_layernorm_alpha)
        self.z_layernorm_eps = float(z_layernorm_eps)
        self.include_logvar = bool(include_logvar)
        self.logvar_output_scale = float(logvar_output_scale)
        self.degenerate_std_threshold = float(degenerate_std_threshold)
        self.degenerate_std_floor = float(degenerate_std_floor)
        self.expose_debug = bool(expose_debug)
        self.gpsi_checkpoint = str(gpsi_checkpoint)
        self.gpsi, self.norm, self.gpsi_config = self._load_gpsi(Path(gpsi_checkpoint))
        self.gpsi.to(self.device)
        self.gpsi.eval()
        for parameter in self.gpsi.parameters():
            parameter.requires_grad_(False)

        self.max_obs = int(obs_space.shape[0])
        self.base_obs_dim = int(obs_space.shape[1])
        self.z_dim = int(self.gpsi.config.z_dim)
        self.num_horizons = int(self.gpsi.config.num_horizons)
        self.state_dim = int(self.gpsi.config.state_dim)
        self.delta_dim = self.num_horizons * self.state_dim
        self.logvar_dim = self.num_horizons * self.state_dim
        self.z_mean, self.z_std = self._load_z_stats(Path(z_stats_path)) if self.z_transform == "standardize" else (None, None)
        self.aug_obs_dim = (
            self.base_obs_dim
            + (self.z_dim if self.include_z else 0)
            + self.delta_dim
            + (self.logvar_dim if self.include_logvar else 0)
        )

        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs, self.aug_obs_dim), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        self._history: dict[int, deque[tuple[np.ndarray, np.ndarray]]] = {}
        self._last_info: dict[str, Any] | None = None
        self._last_debug: dict[str, Any] = {}

    @property
    def freeze_check(self) -> GpsiFreezeCheck:
        total = sum(parameter.numel() for parameter in self.gpsi.parameters())
        trainable = sum(parameter.numel() for parameter in self.gpsi.parameters() if parameter.requires_grad)
        return GpsiFreezeCheck(
            training=bool(self.gpsi.training),
            trainable_parameters=int(trainable),
            total_parameters=int(total),
            requires_grad_any=bool(any(parameter.requires_grad for parameter in self.gpsi.parameters())),
        )

    @property
    def latest_gpsi_debug(self) -> dict[str, Any]:
        return dict(self._last_debug)

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._history.clear()
        self._sync_history(info)
        aug = self._augment_observation(obs, info)
        info = self._augment_info(info)
        return aug, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._sync_history(info)
        aug = self._augment_observation(obs, info)
        info = self._augment_info(info)
        return aug, reward, terminated, truncated, info

    def _load_gpsi(self, checkpoint_path: Path) -> tuple[GpsiHeadA, dict[str, torch.Tensor], dict[str, Any]]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing Gpsi checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(f"invalid Gpsi checkpoint schema: {checkpoint_path}")
        cfg = checkpoint.get("config", {})
        model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
        model = GpsiHeadA(model_cfg)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        norm_raw = checkpoint.get("normalization", {})
        required_norm = [
            "ego_current_mean",
            "ego_current_std",
            "obs_current_mean",
            "obs_current_std",
            "history_rel_pos_mean",
            "history_rel_pos_std",
            "history_rel_vel_mean",
            "history_rel_vel_std",
        ]
        missing = [key for key in required_norm if key not in norm_raw]
        if missing:
            raise ValueError(f"Gpsi checkpoint missing normalization stats: {missing}")
        norm = {key: torch.as_tensor(norm_raw[key], dtype=torch.float32) for key in required_norm}
        return model, norm, cfg if isinstance(cfg, dict) else {}

    def _load_z_stats(self, stats_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
        if stats_path is None or str(stats_path) == "":
            raise ValueError("normalize_z=True requires z_stats_path")
        if not stats_path.exists():
            raise FileNotFoundError(f"missing z stats file: {stats_path}")
        with np.load(stats_path, allow_pickle=False) as payload:
            if "z_mean" not in payload:
                raise ValueError(f"z stats missing z_mean: {stats_path}")
            std_key = "z_std_effective" if "z_std_effective" in payload else "z_std"
            if std_key not in payload:
                raise ValueError(f"z stats missing z_std/z_std_effective: {stats_path}")
            z_mean = np.asarray(payload["z_mean"], dtype=np.float32)
            z_std = np.asarray(payload[std_key], dtype=np.float32)
        if z_mean.shape != (self.z_dim,) or z_std.shape != (self.z_dim,):
            raise ValueError(f"z stats shape mismatch: mean={z_mean.shape} std={z_std.shape} expected={(self.z_dim,)}")
        if not np.isfinite(z_mean).all() or not np.isfinite(z_std).all():
            raise ValueError(f"z stats contain non-finite values: {stats_path}")
        z_std = np.maximum(z_std, max(self.z_std_floor, 1.0e-8)).astype(np.float32)
        return z_mean.astype(np.float32), z_std

    def _sync_history(self, info: dict[str, Any]) -> None:
        ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int64)
        positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
        velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
        uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
        uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)

        active_ids: set[int] = set()
        for slot, obstacle_id_raw in enumerate(ids[: self.max_obs]):
            if slot >= len(positions):
                continue
            obstacle_id = int(obstacle_id_raw)
            active_ids.add(obstacle_id)
            rel_pos = np.asarray(positions[slot], dtype=np.float32) - uav
            rel_vel = (np.asarray(velocities[slot], dtype=np.float32) if slot < len(velocities) else np.zeros(3, dtype=np.float32)) - uav_vel
            history = self._history.get(obstacle_id)
            if history is None:
                history = deque(maxlen=self.history_steps)
                self._history[obstacle_id] = history
            history.append((rel_pos.astype(np.float32), rel_vel.astype(np.float32)))

        stale = [obstacle_id for obstacle_id in self._history if obstacle_id not in active_ids]
        for obstacle_id in stale:
            del self._history[obstacle_id]

    def _augment_observation(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, np.ndarray]:
        base_obs = np.asarray(obs["obs"], dtype=np.float32)
        if base_obs.shape != (self.max_obs, self.base_obs_dim):
            raise ValueError(f"base obs shape mismatch: expected {(self.max_obs, self.base_obs_dim)}, got {base_obs.shape}")

        mask = np.asarray(obs["mask"], dtype=np.float32)
        ego = np.asarray(obs["ego"], dtype=np.float32)
        ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int64)

        z_raw = np.zeros((self.max_obs, self.z_dim), dtype=np.float32)
        z_after = np.zeros((self.max_obs, self.z_dim), dtype=np.float32)
        delta = np.zeros((self.max_obs, self.num_horizons, self.state_dim), dtype=np.float32)
        logvar = np.zeros((self.max_obs, self.num_horizons, self.state_dim), dtype=np.float32)
        logvar_policy = np.zeros((self.max_obs, self.num_horizons, self.state_dim), dtype=np.float32)
        logvar_raw = np.zeros((self.max_obs, self.num_horizons, self.state_dim), dtype=np.float32)
        history_mask_out = np.zeros((self.max_obs, self.history_steps), dtype=np.float32)
        gpsi_ego_input = np.zeros((self.max_obs, ego.shape[0]), dtype=np.float32)
        gpsi_obs_input = np.zeros((self.max_obs, self.base_obs_dim), dtype=np.float32)
        gpsi_hist_pos_input = np.zeros((self.max_obs, self.history_steps, 3), dtype=np.float32)
        gpsi_hist_vel_input = np.zeros((self.max_obs, self.history_steps, 3), dtype=np.float32)

        active_slots = [slot for slot in range(min(self.max_obs, len(ids))) if slot < len(mask) and mask[slot] > 0.0]
        forward_ms = 0.0
        output: dict[str, np.ndarray] = {
            "z": np.zeros((0, self.z_dim), dtype=np.float32),
            "delta_hat": np.zeros((0, self.num_horizons, self.state_dim), dtype=np.float32),
            "logvar_hat": np.zeros((0, self.num_horizons, self.state_dim), dtype=np.float32),
        }
        if active_slots:
            ego_batch = np.repeat(ego[None, :], len(active_slots), axis=0).astype(np.float32)
            obs_batch = base_obs[active_slots].astype(np.float32)
            hist_pos = np.zeros((len(active_slots), self.history_steps, 3), dtype=np.float32)
            hist_vel = np.zeros((len(active_slots), self.history_steps, 3), dtype=np.float32)
            hist_mask = np.zeros((len(active_slots), self.history_steps), dtype=np.float32)
            for batch_idx, slot in enumerate(active_slots):
                obstacle_id = int(ids[slot])
                values = list(self._history.get(obstacle_id, []))
                start = max(self.history_steps - len(values), 0)
                for offset, (rel_pos, rel_vel) in enumerate(values[-self.history_steps :]):
                    dst = start + offset
                    hist_pos[batch_idx, dst] = rel_pos
                    hist_vel[batch_idx, dst] = rel_vel
                    hist_mask[batch_idx, dst] = 1.0
                history_mask_out[slot] = hist_mask[batch_idx]
                gpsi_ego_input[slot] = ego_batch[batch_idx]
                gpsi_obs_input[slot] = obs_batch[batch_idx]
                gpsi_hist_pos_input[slot] = hist_pos[batch_idx]
                gpsi_hist_vel_input[slot] = hist_vel[batch_idx]

            forward_start = time.perf_counter()
            output = self._forward_gpsi(ego_batch, obs_batch, hist_pos, hist_vel, hist_mask)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            z_active = output["z"]
            z_active_after, z_aux = self._transform_z_numpy(z_active)
            delta_active = output["delta_hat"] / max(self.delta_scale, 1.0e-6)
            logvar_active = np.clip(output["logvar_hat"], self.logvar_clamp[0], self.logvar_clamp[1])
            logvar_policy_active = logvar_active * self.logvar_output_scale
            for batch_idx, slot in enumerate(active_slots):
                z_raw[slot] = z_active[batch_idx]
                z_after[slot] = z_active_after[batch_idx]
                delta[slot] = delta_active[batch_idx]
                logvar[slot] = logvar_active[batch_idx]
                logvar_policy[slot] = logvar_policy_active[batch_idx]
                logvar_raw[slot] = output["logvar_hat"][batch_idx]

        blocks = [base_obs]
        if self.include_z:
            blocks.append(z_after)
        blocks.append(delta.reshape(self.max_obs, -1))
        if self.include_logvar:
            blocks.append(logvar_policy.reshape(self.max_obs, -1))
        aug_obs_array = np.concatenate(blocks, axis=-1)
        aug = dict(obs)
        aug["obs"] = aug_obs_array.astype(np.float32)
        self._last_info = info
        self._last_debug = {
            "z": z_after,
            "z_raw": z_raw,
            "z_after_norm": z_after,
            "delta_hat_norm": delta,
            "delta_hat_raw": delta * self.delta_scale,
            "logvar_hat": logvar,
            "logvar_hat_policy": logvar_policy,
            "logvar_hat_raw": logvar_raw,
            "history_valid_mask": history_mask_out,
            "history_valid_ratio": history_mask_out.mean(axis=1),
            "obstacle_ids": ids[: self.max_obs].astype(np.int64),
            "active_slots": np.asarray(active_slots, dtype=np.int64),
            "gpsi_input_ego_current": gpsi_ego_input,
            "gpsi_input_obs_current": gpsi_obs_input,
            "gpsi_input_history_rel_pos": gpsi_hist_pos_input,
            "gpsi_input_history_rel_vel": gpsi_hist_vel_input,
            "gpsi_input_history_valid_mask": history_mask_out,
            "gpsi_norm_ego_current": self._normalize_numpy(gpsi_ego_input, "ego_current"),
            "gpsi_norm_obs_current": self._normalize_numpy(gpsi_obs_input, "obs_current"),
            "gpsi_norm_history_rel_pos": self._normalize_numpy(gpsi_hist_pos_input, "history_rel_pos"),
            "gpsi_norm_history_rel_vel": self._normalize_numpy(gpsi_hist_vel_input, "history_rel_vel"),
            "aug_obs_dim": int(self.aug_obs_dim),
            "delta_scale": float(self.delta_scale),
            "logvar_clamp": self.logvar_clamp,
            "normalize_z": bool(self.normalize_z),
            "include_z": bool(self.include_z),
            "include_logvar": bool(self.include_logvar),
            "logvar_output_scale": float(self.logvar_output_scale),
                "z_stats_path": self.z_stats_path,
                "z_std_floor": float(self.z_std_floor),
                "z_transform": self.z_transform,
                "z_l2_target_norm": float(self.z_l2_target_norm),
                "z_l2_eps": float(self.z_l2_eps),
                "z_layernorm_alpha": float(self.z_layernorm_alpha),
                "z_layernorm_eps": float(self.z_layernorm_eps),
                "z_zero_norm_count": int(z_aux.get("zero_norm_count", 0)) if active_slots else 0,
                "degenerate_std_threshold": float(self.degenerate_std_threshold),
                "degenerate_std_floor": float(self.degenerate_std_floor),
                "gpsi_forward_ms": float(forward_ms),
                "gpsi_forward_batch_size": int(len(active_slots)),
            }
        return aug

    def _transform_z_numpy(self, z: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z = np.asarray(z, dtype=np.float32)
        aux: dict[str, Any] = {"zero_norm_count": 0}
        if self.z_transform == "raw":
            return z.astype(np.float32), aux
        if self.z_transform == "standardize":
            if self.z_mean is None or self.z_std is None:
                raise RuntimeError("z normalization requested before z stats were loaded")
            return ((z - self.z_mean[None, :]) / self.z_std[None, :]).astype(np.float32), aux
        if self.z_transform == "l2_scale":
            norm = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32)
            aux["zero_norm_count"] = int(np.sum(norm <= self.z_l2_eps))
            scaled = z / np.maximum(norm, self.z_l2_eps) * self.z_l2_target_norm
            return scaled.astype(np.float32), aux
        if self.z_transform == "layernorm":
            mean = np.mean(z, axis=1, keepdims=True)
            var = np.mean((z - mean) ** 2, axis=1, keepdims=True)
            normalized = (z - mean) / np.sqrt(var + self.z_layernorm_eps)
            return (self.z_layernorm_alpha * normalized).astype(np.float32), aux
        raise RuntimeError(f"unsupported z_transform={self.z_transform!r}")

    def _normalize_z_numpy(self, z: np.ndarray) -> np.ndarray:
        transformed, _aux = self._transform_z_numpy(z)
        return transformed

    def _forward_gpsi(
        self,
        ego_current: np.ndarray,
        obs_current: np.ndarray,
        history_rel_pos: np.ndarray,
        history_rel_vel: np.ndarray,
        history_valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.gpsi.eval()
        ego_t = self._normalize(torch.from_numpy(ego_current).float(), "ego_current").to(self.device)
        obs_t = self._normalize(torch.from_numpy(obs_current).float(), "obs_current").to(self.device)
        hist_pos_t = self._normalize(torch.from_numpy(history_rel_pos).float(), "history_rel_pos").to(self.device)
        hist_vel_t = self._normalize(torch.from_numpy(history_rel_vel).float(), "history_rel_vel").to(self.device)
        hist_mask_t = torch.from_numpy(history_valid_mask).float().to(self.device)
        with torch.no_grad():
            output = self.gpsi(ego_t, obs_t, hist_pos_t, hist_vel_t, hist_mask_t)
        return {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in output.items()
        }

    def _normalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        mean = self.norm[f"{key}_mean"].to(value.device)
        std = self._effective_std(self.norm[f"{key}_std"].to(value.device))
        return (value - mean) / std

    def _effective_std(self, std: torch.Tensor) -> torch.Tensor:
        std = std.clamp(min=1.0e-6)
        if self.degenerate_std_threshold <= 0.0:
            return std
        floor = max(float(self.degenerate_std_floor), 1.0e-6)
        return torch.where(std <= self.degenerate_std_threshold, torch.full_like(std, floor), std)

    def _normalize_numpy(self, value: np.ndarray, key: str) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).float()
        return self._normalize(tensor, key).detach().cpu().numpy().astype(np.float32)

    def normalization_debug_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in ["ego_current", "obs_current", "history_rel_pos", "history_rel_vel"]:
            raw_std = self.norm[f"{key}_std"].detach().cpu().float()
            effective = self._effective_std(raw_std).detach().cpu()
            flat_raw = raw_std.reshape(-1).numpy()
            flat_eff = effective.reshape(-1).numpy()
            for idx, (original, repaired) in enumerate(zip(flat_raw, flat_eff)):
                rows.append(
                    {
                        "field": key,
                        "dim": int(idx),
                        "checkpoint_std": float(original),
                        "effective_std": float(repaired),
                        "std_repaired": int(abs(float(original) - float(repaired)) > 1.0e-12),
                        "degenerate_std_threshold": float(self.degenerate_std_threshold),
                        "degenerate_std_floor": float(self.degenerate_std_floor),
                    }
                )
        return rows

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        if not self.expose_debug:
            return info
        out = dict(info)
        debug = self._last_debug
        out["gpsi_aug_obs_dim"] = int(self.aug_obs_dim)
        out["gpsi_delta_scale"] = float(self.delta_scale)
        out["gpsi_logvar_clamp"] = tuple(self.logvar_clamp)
        out["gpsi_history_valid_ratio"] = np.asarray(debug.get("history_valid_ratio", np.zeros(self.max_obs)), dtype=np.float32)
        out["gpsi_delta_hat_norm"] = np.asarray(
            debug.get("delta_hat_norm", np.zeros((self.max_obs, self.num_horizons, self.state_dim))),
            dtype=np.float32,
        )
        out["gpsi_delta_hat_raw"] = np.asarray(
            debug.get("delta_hat_raw", np.zeros((self.max_obs, self.num_horizons, self.state_dim))),
            dtype=np.float32,
        )
        out["gpsi_z_raw"] = np.asarray(debug.get("z_raw", np.zeros((self.max_obs, self.z_dim))), dtype=np.float32)
        out["gpsi_z_after_norm"] = np.asarray(debug.get("z_after_norm", np.zeros((self.max_obs, self.z_dim))), dtype=np.float32)
        out["gpsi_z_transform"] = str(debug.get("z_transform", self.z_transform))
        out["gpsi_z_zero_norm_count"] = int(debug.get("z_zero_norm_count", 0))
        out["gpsi_logvar_hat"] = np.asarray(
            debug.get("logvar_hat", np.zeros((self.max_obs, self.num_horizons, self.state_dim))),
            dtype=np.float32,
        )
        out["gpsi_logvar_hat_policy"] = np.asarray(
            debug.get("logvar_hat_policy", np.zeros((self.max_obs, self.num_horizons, self.state_dim))),
            dtype=np.float32,
        )
        out["gpsi_logvar_hat_raw"] = np.asarray(
            debug.get("logvar_hat_raw", np.zeros((0, self.num_horizons, self.state_dim))),
            dtype=np.float32,
        )
        out["gpsi_forward_ms"] = float(debug.get("gpsi_forward_ms", 0.0))
        out["gpsi_forward_batch_size"] = int(debug.get("gpsi_forward_batch_size", 0))
        return out
