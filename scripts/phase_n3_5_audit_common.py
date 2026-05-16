from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from envs.wrappers.gpsi_obs_wrapper import GpsiObsWrapper
from models.gpsi_head_a import GpsiHeadA


N2_FLAG = ROOT / "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag"
N3_REPORT = ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield/PHASE_N3_GPSI_PPO_NO_SHIELD_REPORT.md"
N3_COMPLETE_FLAG = ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield/PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag"
N3_REQUIRED_TABLES = [
    ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_eval_summary.csv",
    ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_gpsi_output_summary.csv",
    ROOT / "results/env_v2_phase_n3_gpsi_ppo_no_shield/tables/phase_n3_gpsi_forward_profile.csv",
]

COMPLETE_FLAG = "PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n2_missing": "PHASE_N3_5_STOP_PHASE_N2_MISSING.flag",
    "gpsi_checkpoint_missing": "PHASE_N3_5_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "n3_artifacts_missing": "PHASE_N3_5_STOP_N3_ARTIFACTS_MISSING.flag",
    "offline_online_mismatch": "PHASE_N3_5_STOP_OFFLINE_ONLINE_MISMATCH.flag",
    "input_distribution_invalid": "PHASE_N3_5_STOP_INPUT_DISTRIBUTION_INVALID.flag",
    "output_scale_invalid": "PHASE_N3_5_STOP_OUTPUT_SCALE_INVALID.flag",
    "feature_scale_invalid": "PHASE_N3_5_STOP_FEATURE_SCALE_INVALID.flag",
    "wrapper_repair_failed": "PHASE_N3_5_STOP_WRAPPER_REPAIR_FAILED.flag",
    "watcher_failed": "PHASE_N3_5_STOP_WATCHER_FAILED.flag",
}

MOTION_MODE_NAME = {
    0: "linear",
    1: "sinusoidal_lateral",
    2: "accel_decel",
    3: "ar1_velocity",
    4: "crossing_or_sudden_threat",
}
THREAT_CLASS_NAME = {0: "low", 1: "medium", 2: "high"}
MOTION_MODE_ID = {value: key for key, value in MOTION_MODE_NAME.items()}
THREAT_CLASS_ID = {value: key for key, value in THREAT_CLASS_NAME.items()}
HORIZON_SUFFIXES = ["1s", "2s", "4s"]


class PhaseN35Stop(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(out_dir: Path) -> None:
    for rel_dir in ["tables", "plots", "logs"]:
        (out_dir / rel_dir).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    if fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stop(out_dir: Path, reason: str, detail: str, *, terminal_reason: str | None = None) -> None:
    ensure_dirs(out_dir)
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["watcher_failed"])
    write_text(out_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(out_dir / "phase_n3_5_status.txt", f"stopped:{flag}\n")
    terminal = terminal_reason or reason
    report = [
        "# Phase N3.5 Gpsi Wrapper Audit Report",
        "",
        f"`terminal_decision = phase_n3_5_stopped_{terminal}`",
        "",
        "Partial report generated because a stop condition was reached.",
        "",
        "## Stop Condition",
        "",
        f"- reason: `{reason}`",
        f"- flag: `{flag}`",
        "",
        "```text",
        detail.strip(),
        "```",
        "",
        "Can enter N4: no.",
    ]
    write_text(out_dir / "PHASE_N3_5_GPSI_WRAPPER_AUDIT_REPORT.md", "\n".join(report) + "\n")


def check_prerequisites(out_dir: Path, checkpoint: Path) -> None:
    missing_n3 = [path for path in [N3_REPORT, N3_COMPLETE_FLAG, *N3_REQUIRED_TABLES] if not path.exists()]
    if not N2_FLAG.exists():
        raise PhaseN35Stop("phase_n2_missing", f"missing Phase N2 complete flag: {rel(N2_FLAG)}")
    if missing_n3:
        raise PhaseN35Stop("n3_artifacts_missing", "missing N3 artifacts:\n" + "\n".join(rel(path) for path in missing_n3))
    if not checkpoint.exists():
        raise PhaseN35Stop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")
    ensure_dirs(out_dir)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_gpsi_checkpoint(path: Path, device: torch.device) -> tuple[GpsiHeadA, dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise PhaseN35Stop("gpsi_checkpoint_missing", f"invalid checkpoint schema: {rel(path)}")
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model = GpsiHeadA(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    norm_payload = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in checkpoint.get("normalization", {}).items()}
    required = [
        "ego_current_mean",
        "ego_current_std",
        "obs_current_mean",
        "obs_current_std",
        "history_rel_pos_mean",
        "history_rel_pos_std",
        "history_rel_vel_mean",
        "history_rel_vel_std",
    ]
    missing = [key for key in required if key not in norm_payload]
    if missing:
        raise PhaseN35Stop("gpsi_checkpoint_missing", f"checkpoint missing normalization stats: {missing}")
    return model, norm_payload, checkpoint


def effective_std(std: torch.Tensor, threshold: float, floor: float) -> torch.Tensor:
    std = std.float().clamp(min=1.0e-6)
    if threshold <= 0.0:
        return std
    return torch.where(std <= threshold, torch.full_like(std, max(float(floor), 1.0e-6)), std)


def normalize_array(
    array: np.ndarray,
    norm_payload: dict[str, torch.Tensor],
    key: str,
    *,
    threshold: float,
    floor: float,
) -> np.ndarray:
    value = torch.from_numpy(np.asarray(array, dtype=np.float32)).float()
    mean = norm_payload[f"{key}_mean"].float()
    std = effective_std(norm_payload[f"{key}_std"], threshold, floor)
    return ((value - mean) / std).numpy().astype(np.float32)


def forward_gpsi_np(
    model: GpsiHeadA,
    inputs: dict[str, np.ndarray],
    norm_payload: dict[str, torch.Tensor],
    device: torch.device,
    *,
    threshold: float,
    floor: float,
    batch_size: int = 4096,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = int(inputs["ego_current"].shape[0])
    normalized = {
        "ego_current": normalize_array(inputs["ego_current"], norm_payload, "ego_current", threshold=threshold, floor=floor),
        "obs_current": normalize_array(inputs["obs_current"], norm_payload, "obs_current", threshold=threshold, floor=floor),
        "history_rel_pos": normalize_array(inputs["history_rel_pos"], norm_payload, "history_rel_pos", threshold=threshold, floor=floor),
        "history_rel_vel": normalize_array(inputs["history_rel_vel"], norm_payload, "history_rel_vel", threshold=threshold, floor=floor),
        "history_valid_mask": np.asarray(inputs["history_valid_mask"], dtype=np.float32),
    }
    parts: dict[str, list[np.ndarray]] = {"z": [], "delta_hat": [], "logvar_hat": []}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out = model(
                torch.from_numpy(normalized["ego_current"][start:end]).float().to(device),
                torch.from_numpy(normalized["obs_current"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_rel_pos"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_rel_vel"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_valid_mask"][start:end]).float().to(device),
            )
            for key in parts:
                parts[key].append(out[key].detach().cpu().numpy().astype(np.float32))
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}, normalized


def make_wrapper(
    checkpoint: Path,
    scenario: str,
    device: str,
    *,
    threshold: float,
    floor: float,
    delta_scale: float = 5.0,
    logvar_clamp: tuple[float, float] = (-5.0, 3.0),
) -> GpsiObsWrapper:
    env = DynamicObstacleFlowEnv(scenario=scenario)
    wrapper = GpsiObsWrapper(
        env,
        gpsi_checkpoint=checkpoint,
        device=device,
        history_steps=20,
        delta_scale=delta_scale,
        logvar_clamp=logvar_clamp,
        degenerate_std_threshold=threshold,
        degenerate_std_floor=floor,
    )
    freeze = wrapper.freeze_check
    if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
        raise PhaseN35Stop("wrapper_repair_failed", f"Gpsi freeze check failed: {freeze}")
    return wrapper


def action_from_policy(info: dict[str, Any], policy: str, rng: np.random.Generator) -> np.ndarray:
    if policy in {"straight_line", "random_or_straight_line"}:
        goal = np.asarray(info["goal_position"], dtype=np.float32)
        uav = np.asarray(info["uav_position"], dtype=np.float32)
        vec = goal - uav
        vec[2] = 0.0
        norm = float(np.linalg.norm(vec))
        if norm < 1.0e-8:
            return np.zeros(3, dtype=np.float32)
        return np.clip(vec / norm, -1.0, 1.0).astype(np.float32)
    if policy == "hold_position":
        return np.zeros(3, dtype=np.float32)
    if policy == "random":
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        action[2] = 0.0
        return action
    raise ValueError(f"unsupported audit policy: {policy}")


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    row = {
        "count": int(arr.size),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
    }
    if finite.size == 0:
        row.update({key: float("nan") for key in ["mean", "std", "min", "max", "p01", "p05", "p50", "p95", "p99"]})
        return row
    finite = finite.astype(np.float64)
    row.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p01": float(np.percentile(finite, 1)),
            "p05": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
        }
    )
    return row


def l2_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    stats = finite_stats(arr)
    norm_stats = finite_stats(norms)
    stats.update(
        {
            "l2_norm_mean": norm_stats["mean"],
            "l2_norm_median": norm_stats["p50"],
            "l2_norm_p95": norm_stats["p95"],
        }
    )
    return stats


def block_stats_row(prefix: dict[str, Any], block: str, values: np.ndarray) -> dict[str, Any]:
    row = dict(prefix)
    row["block"] = block
    row.update(l2_stats(values))
    return row


def metric_diff_rows(
    component: str,
    offline: np.ndarray,
    online: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    a = np.asarray(offline, dtype=np.float64).reshape(-1)
    b = np.asarray(online, dtype=np.float64).reshape(-1)
    diff = a - b
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() >= 2 and np.std(a[finite]) > 0.0 and np.std(b[finite]) > 0.0:
        corr = float(np.corrcoef(a[finite], b[finite])[0, 1])
    else:
        corr = float("nan")
    max_abs = float(np.max(np.abs(diff[finite]))) if finite.any() else float("nan")
    mean_abs = float(np.mean(np.abs(diff[finite]))) if finite.any() else float("nan")
    rmse = float(np.sqrt(np.mean(diff[finite] ** 2))) if finite.any() else float("nan")
    return {
        "component": component,
        "shape": "x".join(str(v) for v in np.asarray(offline).shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rmse_diff": rmse,
        "corr": corr,
        "allclose_pass": int(bool(np.allclose(offline, online, atol=tolerance, rtol=tolerance))),
        "tolerance": float(tolerance),
        "nan_count_offline": int(np.isnan(a).sum()),
        "nan_count_online": int(np.isnan(b).sum()),
        "inf_count_offline": int(np.isinf(a).sum()),
        "inf_count_online": int(np.isinf(b).sum()),
    }


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def projected_std(logvar_xy: np.ndarray, direction_xy: np.ndarray) -> np.ndarray:
    lv = np.asarray(logvar_xy, dtype=np.float32)
    direction = np.asarray(direction_xy, dtype=np.float32)
    sigma2 = np.exp(np.clip(lv, -5.0, 3.0))
    return np.sqrt(np.sum((direction**2) * sigma2, axis=-1))


def history_ratio_bin(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value < 0.25:
        return "0_0.25"
    if value < 0.5:
        return "0.25_0.5"
    if value < 0.75:
        return "0.5_0.75"
    if value < 1.0:
        return "0.75_1.0"
    return "1.0"


def values_by_keys(rows: list[dict[str, Any]], key_fields: list[str], value_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    prefixes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        prefixes.setdefault(key, {field: row.get(field) for field in key_fields})
        bucket = groups.setdefault(key, {field: [] for field in value_fields})
        for field in value_fields:
            value = row.get(field)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                value_float = float("nan")
            bucket[field].append(value_float)
    out: list[dict[str, Any]] = []
    for key, bucket in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        prefix = dict(prefixes[key])
        for field, values in bucket.items():
            stats = finite_stats(np.asarray(values, dtype=np.float64))
            for stat_name, stat_value in stats.items():
                prefix[f"{field}_{stat_name}"] = stat_value
        out.append(prefix)
    return out


def save_histogram(path: Path, datasets: list[tuple[str, np.ndarray]], title: str, xlabel: str, *, bins: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in datasets:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            plt.hist(arr, bins=bins, alpha=0.45, density=True, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, *, limit: int = 20000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size > limit:
        idx = np.linspace(0, x_arr.size - 1, num=limit, dtype=np.int64)
        x_arr = x_arr[idx]
        y_arr = y_arr[idx]
    plt.figure(figsize=(6, 6))
    if x_arr.size:
        plt.scatter(x_arr, y_arr, s=3, alpha=0.35)
        lo = float(min(np.min(x_arr), np.min(y_arr)))
        hi = float(max(np.max(x_arr), np.max(y_arr)))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar(path: Path, rows: list[dict[str, Any]], label_field: str, value_field: str, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(label_field, "")) for row in rows]
    values = [float(row.get(value_field, float("nan"))) for row in rows]
    plt.figure(figsize=(max(7, len(labels) * 0.6), 5))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def command_manifest_row(name: str, command: str, status: str) -> dict[str, Any]:
    return {"command_name": name, "command": command, "status": status}


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")
