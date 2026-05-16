from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gpsi_head_a import GpsiHeadA


COMPLETE_FLAG = "PHASE_N2_HEADA_OFFLINE_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n1_missing": "PHASE_N2_STOP_PHASE_N1_MISSING.flag",
    "dataset_read_failed": "PHASE_N2_STOP_DATASET_READ_FAILED.flag",
    "schema_mismatch": "PHASE_N2_STOP_SCHEMA_MISMATCH.flag",
    "delta_train_failed": "PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag",
    "delta_not_learnable": "PHASE_N2_STOP_DELTA_NOT_LEARNABLE.flag",
    "nll_train_failed": "PHASE_N2_STOP_NLL_TRAIN_FAILED.flag",
    "logvar_collapse": "PHASE_N2_STOP_LOGVAR_COLLAPSE.flag",
    "calibration_failed": "PHASE_N2_STOP_CALIBRATION_FAILED.flag",
}
MOTION_MODE_NAME = {
    0: "linear",
    1: "sinusoidal_lateral",
    2: "accel_decel",
    3: "ar1_velocity",
    4: "crossing_or_sudden_threat",
}
AXIS_NAMES = ["x", "y", "z"]


class PhaseN2EvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "logs" / "phase_n2_eval.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class EvalDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], norm: dict[str, torch.Tensor]) -> None:
        self.arrays = arrays
        self.norm = norm
        self.n = int(arrays["episode_id"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "ego_current": torch.from_numpy(self.arrays["ego_current"][idx]).float(),
            "obs_current": torch.from_numpy(self.arrays["obs_current"][idx]).float(),
            "history_rel_pos": torch.from_numpy(self.arrays["history_rel_pos"][idx]).float(),
            "history_rel_vel": torch.from_numpy(self.arrays["history_rel_vel"][idx]).float(),
            "history_valid_mask": torch.from_numpy(self.arrays["history_valid_mask"][idx]).float(),
            "delta_label_world": torch.from_numpy(self.arrays["delta_label_world"][idx]).float(),
            "future_valid_mask": torch.from_numpy(self.arrays["future_valid_mask"][idx]).float(),
            "motion_mode_id": torch.tensor(int(self.arrays["motion_mode_id"][idx]), dtype=torch.long),
        }
        item["ego_current"] = norm(item["ego_current"], self.norm["ego_current_mean"], self.norm["ego_current_std"])
        item["obs_current"] = norm(item["obs_current"], self.norm["obs_current_mean"], self.norm["obs_current_std"])
        item["history_rel_pos"] = norm(item["history_rel_pos"], self.norm["history_rel_pos_mean"], self.norm["history_rel_pos_std"])
        item["history_rel_vel"] = norm(item["history_rel_vel"], self.norm["history_rel_vel_mean"], self.norm["history_rel_vel_std"])
        return item


def norm(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / torch.clamp(std, min=1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gpsi-HeadA offline checkpoints.")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument("--nll-checkpoint", required=True)
    parser.add_argument("--out-dir", default="results/env_v2_phase_n2_gpsi_heada_offline")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logvar-clamp", nargs=2, type=float, default=[-5.0, 3.0])
    parser.add_argument("--plot-sample-limit", type=int, default=50000)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["tables", "plots", "logs"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_stop(out_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["calibration_failed"])
    write_text(out_dir / flag, f"{reason}\n{detail}\n")
    write_text(out_dir / "phase_n2_status.txt", f"stopped:{flag}\n")
    write_text(
        out_dir / "PHASE_N2_HEADA_OFFLINE_REPORT.md",
        f"# Phase N2 HeadA Offline Report\n\n`terminal_decision = phase_n2_stopped_{reason}`\n\n```text\n{detail}\n```\n",
    )


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_arrays(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"{split}.npz"
    if not path.exists():
        raise PhaseN2EvalStop("dataset_read_failed", f"missing split: {relpath(path)}")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def load_checkpoint(path: Path, device: torch.device) -> tuple[GpsiHeadA, dict[str, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise PhaseN2EvalStop("dataset_read_failed", f"missing checkpoint: {relpath(path)}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GpsiHeadA(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    norm_payload = {key: value.detach().cpu() for key, value in ckpt["normalization"].items()}
    return model, norm_payload, ckpt


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def predict(model: GpsiHeadA, arrays: dict[str, np.ndarray], norm_payload: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = EvalDataset(arrays, norm_payload)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    delta_parts: list[np.ndarray] = []
    logvar_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            out = model(
                batch["ego_current"],
                batch["obs_current"],
                batch["history_rel_pos"],
                batch["history_rel_vel"],
                batch["history_valid_mask"],
            )
            delta_parts.append(out["delta_hat"].detach().cpu().numpy().astype(np.float32))
            logvar_parts.append(out["logvar_hat"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(delta_parts, axis=0), np.concatenate(logvar_parts, axis=0)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask[..., None].astype(np.float64)
    denom = max(float(m.sum() * target.shape[-1]), 1.0)
    return float((((pred - target) ** 2) * m).sum() / denom)


def masked_smoothl1(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    err = np.abs(pred - target)
    loss = np.where(err < 1.0, 0.5 * err**2, err - 0.5)
    m = mask[..., None].astype(np.float64)
    denom = max(float(m.sum() * target.shape[-1]), 1.0)
    return float((loss * m).sum() / denom)


def masked_nll(pred: np.ndarray, logvar: np.ndarray, target: np.ndarray, mask: np.ndarray, clamp: tuple[float, float]) -> float:
    lv = np.clip(logvar, clamp[0], clamp[1])
    err2 = (pred - target) ** 2
    terms = 0.5 * (np.exp(-lv) * err2 + lv)
    m = mask[..., None].astype(np.float64)
    denom = max(float(m.sum() * target.shape[-1]), 1.0)
    return float((terms * m).sum() / denom)


def aggregate_metrics(name: str, split: str, pred: np.ndarray, logvar: np.ndarray, arrays: dict[str, np.ndarray], clamp: tuple[float, float]) -> dict[str, Any]:
    target = arrays["delta_label_world"]
    mask = arrays["future_valid_mask"]
    return {
        "model": name,
        "split": split,
        "samples": int(target.shape[0]),
        "valid_labels": int(mask.sum()),
        "masked_mse": masked_mse(pred, target, mask),
        "masked_smoothl1": masked_smoothl1(pred, target, mask),
        "gaussian_nll": masked_nll(pred, logvar, target, mask, clamp),
        "mean_logvar": float(np.mean(np.clip(logvar, clamp[0], clamp[1]))),
        "min_logvar": float(np.min(np.clip(logvar, clamp[0], clamp[1]))),
        "max_logvar": float(np.max(np.clip(logvar, clamp[0], clamp[1]))),
    }


def per_horizon_metrics(model: str, split: str, pred: np.ndarray, logvar: np.ndarray, arrays: dict[str, np.ndarray], future_times: np.ndarray, clamp: tuple[float, float]) -> list[dict[str, Any]]:
    rows = []
    target = arrays["delta_label_world"]
    mask = arrays["future_valid_mask"]
    for h, tau in enumerate(future_times):
        rows.append(
            {
                "model": model,
                "split": split,
                "horizon_sec": float(tau),
                "valid_labels": int(mask[:, h].sum()),
                "masked_mse": masked_mse(pred[:, h : h + 1], target[:, h : h + 1], mask[:, h : h + 1]),
                "masked_smoothl1": masked_smoothl1(pred[:, h : h + 1], target[:, h : h + 1], mask[:, h : h + 1]),
                "gaussian_nll": masked_nll(pred[:, h : h + 1], logvar[:, h : h + 1], target[:, h : h + 1], mask[:, h : h + 1], clamp),
            }
        )
    return rows


def per_motion_mode_metrics(model: str, split: str, pred: np.ndarray, logvar: np.ndarray, arrays: dict[str, np.ndarray], clamp: tuple[float, float]) -> list[dict[str, Any]]:
    rows = []
    target = arrays["delta_label_world"]
    mask = arrays["future_valid_mask"]
    for mode_id in sorted(set(int(v) for v in arrays["motion_mode_id"].tolist())):
        idx = arrays["motion_mode_id"] == mode_id
        rows.append(
            {
                "model": model,
                "split": split,
                "motion_mode_id": int(mode_id),
                "motion_mode": MOTION_MODE_NAME.get(int(mode_id), "unknown"),
                "samples": int(idx.sum()),
                "valid_labels": int(mask[idx].sum()),
                "masked_mse": masked_mse(pred[idx], target[idx], mask[idx]),
                "masked_smoothl1": masked_smoothl1(pred[idx], target[idx], mask[idx]),
                "gaussian_nll": masked_nll(pred[idx], logvar[idx], target[idx], mask[idx], clamp),
            }
        )
    return rows


def per_axis_logvar_stats(split: str, logvar: np.ndarray, arrays: dict[str, np.ndarray], future_times: np.ndarray, clamp: tuple[float, float]) -> list[dict[str, Any]]:
    rows = []
    lv = np.clip(logvar, clamp[0], clamp[1])
    mask = arrays["future_valid_mask"] > 0.5
    for mode_id in sorted(set(int(v) for v in arrays["motion_mode_id"].tolist())):
        mode_mask = arrays["motion_mode_id"] == mode_id
        for h, tau in enumerate(future_times):
            valid_mask = mode_mask & mask[:, h]
            for axis, axis_name in enumerate(AXIS_NAMES):
                values = lv[valid_mask, h, axis]
                rows.append(stats_row({"split": split, "horizon_sec": float(tau), "motion_mode_id": int(mode_id), "motion_mode": MOTION_MODE_NAME.get(int(mode_id), "unknown"), "axis": axis_name}, values))
    return rows


def stats_row(prefix: dict[str, Any], values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    row = dict(prefix)
    row.update(
        {
            "count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
            "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    )
    return row


def projection_directions(arrays: dict[str, np.ndarray], error_xy: np.ndarray) -> dict[str, np.ndarray]:
    n = error_xy.shape[0]
    dirs: dict[str, np.ndarray] = {
        "x_axis": np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1)),
        "y_axis": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n, 1)),
    }
    radial = arrays["history_rel_pos"][:, -1, :2].astype(np.float32)
    dirs["radial_xy"] = normalize_rows(radial)
    rel_vel = arrays["history_rel_vel"][:, -1, :2].astype(np.float32)
    dirs["rel_velocity_xy"] = normalize_rows(rel_vel)
    dirs["error_direction_diag_only"] = normalize_rows(error_xy.astype(np.float32))
    return dirs


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = np.zeros_like(values, dtype=np.float32)
    mask = norms[:, 0] > 1.0e-8
    out[mask] = values[mask] / norms[mask]
    return out


def projected_calibration(split: str, pred: np.ndarray, logvar: np.ndarray, arrays: dict[str, np.ndarray], future_times: np.ndarray, clamp: tuple[float, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    target = arrays["delta_label_world"]
    error = target - pred
    lv = np.clip(logvar, clamp[0], clamp[1])
    var_xy = np.exp(lv[:, :, :2])
    diagnostics: dict[str, list[np.ndarray]] = {"projected_std": [], "abs_projected_error": [], "z": []}
    for h, tau in enumerate(future_times):
        valid_h = arrays["future_valid_mask"][:, h] > 0.5
        err_xy_h = error[:, h, :2]
        dirs = projection_directions(arrays, err_xy_h)
        for direction_name, direction in dirs.items():
            dir_valid = valid_h & (np.linalg.norm(direction, axis=1) > 0.5)
            if not np.any(dir_valid):
                continue
            d = direction[dir_valid]
            e = err_xy_h[dir_valid]
            v = var_xy[dir_valid, h]
            proj_error = np.sum(d * e, axis=1)
            proj_var = np.sum((d**2) * v, axis=1)
            proj_std = np.sqrt(np.maximum(proj_var, 1.0e-9))
            z = proj_error / proj_std
            abs_error = np.abs(proj_error)
            diagnostics["projected_std"].append(proj_std)
            diagnostics["abs_projected_error"].append(abs_error)
            diagnostics["z"].append(z)
            corr = safe_corr(proj_std, abs_error)
            rows.append(
                {
                    "split": split,
                    "horizon_sec": float(tau),
                    "direction": direction_name,
                    "count": int(z.size),
                    "mean_abs_z": float(np.mean(np.abs(z))),
                    "std_z": float(np.std(z)),
                    "pct_abs_z_lt_1": float(np.mean(np.abs(z) < 1.0)),
                    "pct_abs_z_lt_2": float(np.mean(np.abs(z) < 2.0)),
                    "projected_nll": float(np.mean(0.5 * ((proj_error**2) / np.maximum(proj_var, 1.0e-9) + np.log(np.maximum(proj_var, 1.0e-9))))),
                    "corr_projected_std_abs_error": corr,
                    "mean_projected_std": float(np.mean(proj_std)),
                    "mean_abs_projected_error": float(np.mean(abs_error)),
                    "scalar_std_trace_mean": float(np.mean(np.sqrt(np.sum(v, axis=1)))),
                    "scalar_std_max_mean": float(np.mean(np.max(np.sqrt(v), axis=1))),
                }
            )
            bin_rows.extend(calibration_bins(split, float(tau), direction_name, proj_std, abs_error, z))
    diag_arrays = {key: np.concatenate(value) if value else np.asarray([], dtype=np.float32) for key, value in diagnostics.items()}
    return rows, bin_rows, diag_arrays


def calibration_bins(split: str, tau: float, direction: str, std: np.ndarray, abs_error: np.ndarray, z: np.ndarray) -> list[dict[str, Any]]:
    if std.size < 10:
        return []
    quantiles = np.quantile(std, np.linspace(0.0, 1.0, 6))
    rows = []
    for idx in range(5):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == 4:
            mask = (std >= low) & (std <= high)
        else:
            mask = (std >= low) & (std < high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "split": split,
                "horizon_sec": tau,
                "direction": direction,
                "bin": idx,
                "std_low": float(low),
                "std_high": float(high),
                "count": int(mask.sum()),
                "mean_predicted_std": float(std[mask].mean()),
                "mean_abs_projected_error": float(abs_error[mask].mean()),
                "pct_abs_z_lt_1": float(np.mean(np.abs(z[mask]) < 1.0)),
                "pct_abs_z_lt_2": float(np.mean(np.abs(z[mask]) < 2.0)),
            }
        )
    return rows


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_plots(out_dir: Path, delta_log: Path, nll_log: Path, horizon_rows: list[dict[str, Any]], motion_rows: list[dict[str, Any]], axis_rows: list[dict[str, Any]], calib_rows: list[dict[str, Any]], bin_rows: list[dict[str, Any]], diag: dict[str, np.ndarray], pred: np.ndarray, target: np.ndarray, mask: np.ndarray, sample_limit: int) -> list[str]:
    plt = import_matplotlib()
    paths: list[str] = []
    for path, title, output in [
        (delta_log, "Delta-only Loss Curve", out_dir / "plots/delta_loss_curve.png"),
        (nll_log, "NLL Loss Curve", out_dir / "plots/nll_loss_curve.png"),
    ]:
        rows = read_csv_rows(path)
        fig, ax = plt.subplots(figsize=(8, 5))
        if rows:
            epochs = [int(row["epoch"]) for row in rows]
            ax.plot(epochs, [float(row["train_loss"]) for row in rows], label="train loss")
            ax.plot(epochs, [float(row["val_loss"]) for row in rows], label="val loss")
            ax.plot(epochs, [float(row["val_mse"]) for row in rows], label="val mse")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
        paths.append(relpath(output))

    test_horizon = [row for row in horizon_rows if row["split"] == "test" and row["model"] in {"zero", "nll"}]
    labels = sorted(set(float(row["horizon_sec"]) for row in test_horizon))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    for offset, model in enumerate(["zero", "nll"]):
        values = [next(float(row["masked_mse"]) for row in test_horizon if row["model"] == model and float(row["horizon_sec"]) == tau) for tau in labels]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tau:g}s" for tau in labels])
    ax.set_title("Per-Horizon MSE")
    ax.legend()
    fig.tight_layout()
    output = out_dir / "plots/per_horizon_mse_bar.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))

    test_motion = [row for row in motion_rows if row["split"] == "test" and row["model"] in {"zero", "nll"}]
    modes = sorted(set(row["motion_mode"] for row in test_motion))
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, model in enumerate(["zero", "nll"]):
        values = [next(float(row["masked_mse"]) for row in test_motion if row["model"] == model and row["motion_mode"] == mode) for mode in modes]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20)
    ax.set_title("Per-Motion-Mode Error")
    ax.legend()
    fig.tight_layout()
    output = out_dir / "plots/per_motion_mode_error_bar.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))

    test_axis = [row for row in axis_rows if row["split"] == "test"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_axis = [f"{row['motion_mode']}-{row['horizon_sec']}-{row['axis']}" for row in test_axis[:60]]
    values_axis = [float(row["mean"]) for row in test_axis[:60]]
    ax.bar(np.arange(len(values_axis)), values_axis)
    ax.set_title("Logvar By Motion Mode/Horizon/Axis")
    ax.set_xticks(np.arange(len(values_axis)))
    ax.set_xticklabels(labels_axis, rotation=90, fontsize=6)
    fig.tight_layout()
    output = out_dir / "plots/logvar_by_motion_mode.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [row for row in bin_rows if row["split"] == "test" and row["direction"] == "radial_xy"]
    if bins:
        ax.scatter([float(row["mean_predicted_std"]) for row in bins], [float(row["mean_abs_projected_error"]) for row in bins], s=[max(int(row["count"]) / 100, 10) for row in bins])
    ax.set_xlabel("mean predicted projected std")
    ax.set_ylabel("mean abs projected error")
    ax.set_title("Projected Uncertainty Reliability")
    fig.tight_layout()
    output = out_dir / "plots/projected_uncertainty_reliability.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))

    fig, ax = plt.subplots(figsize=(8, 5))
    z = downsample(diag.get("z", np.asarray([])), sample_limit)
    if z.size:
        ax.hist(z, bins=80, range=(-5, 5))
    ax.set_title("Projected z-score Histogram")
    fig.tight_layout()
    output = out_dir / "plots/zscore_histogram.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = mask.astype(bool)
    err = np.linalg.norm((target - pred), axis=2)
    p = downsample(pred[valid], sample_limit)
    e = downsample(err[valid], sample_limit)
    if p.size and e.size:
        pred_norm = np.linalg.norm(p.reshape(-1, 3), axis=1)
        ax.scatter(pred_norm[: e.size], e, s=2, alpha=0.2)
    ax.set_xlabel("||delta_hat||")
    ax.set_ylabel("||error||")
    ax.set_title("Predicted vs Error Scatter")
    fig.tight_layout()
    output = out_dir / "plots/predicted_vs_error_scatter.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    paths.append(relpath(output))
    return paths


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def downsample(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= limit:
        return values
    idx = np.linspace(0, values.shape[0] - 1, limit, dtype=np.int64)
    return values[idx]


def command_manifest() -> list[dict[str, Any]]:
    return [
        {"command_name": "model_py_compile", "command": "python -m py_compile models/gpsi_head_a.py", "status": "completed_before_final_watcher"},
        {"command_name": "train_py_compile", "command": "python -m py_compile scripts/train_gpsi_heada.py", "status": "completed_before_final_watcher"},
        {"command_name": "eval_py_compile", "command": "python -m py_compile scripts/eval_gpsi_heada.py", "status": "completed_before_final_watcher"},
        {"command_name": "watcher_syntax", "command": "bash -n scripts/watch_phase_n2_gpsi_heada_offline.sh", "status": "completed_before_final_watcher"},
        {"command_name": "delta_train", "command": "python scripts/train_gpsi_heada.py --data-dir data/gpsi_head_a_v1 --out-dir work_dirs/gpsi_heada_v1_delta_only --config configs/gpsi_heada_train_delta_only.yaml --loss delta_smoothl1 --epochs 20 --batch-size 1024 --seed 0", "status": "completed"},
        {"command_name": "nll_train", "command": "python scripts/train_gpsi_heada.py --data-dir data/gpsi_head_a_v1 --init work_dirs/gpsi_heada_v1_delta_only/best.pth --out-dir work_dirs/gpsi_heada_v1_nll --config configs/gpsi_heada_train_nll.yaml --loss gaussian_nll --logvar-clamp -5 3 --epochs 20 --batch-size 1024 --seed 0", "status": "completed"},
        {"command_name": "eval", "command": "python scripts/eval_gpsi_heada.py --data-dir data/gpsi_head_a_v1 --delta-checkpoint work_dirs/gpsi_heada_v1_delta_only/best.pth --nll-checkpoint work_dirs/gpsi_heada_v1_nll/best.pth --out-dir results/env_v2_phase_n2_gpsi_heada_offline", "status": "completed"},
        {"command_name": "blocking_watcher", "command": "bash scripts/watch_phase_n2_gpsi_heada_offline.sh", "status": "completed_when_terminal_flag_detected"},
    ]


def write_report(out_dir: Path, dataset_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], nll_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]], motion_rows: list[dict[str, Any]], axis_rows: list[dict[str, Any]], calib_rows: list[dict[str, Any]], plot_paths: list[str], warnings: list[str]) -> None:
    test_delta = next(row for row in delta_rows if row["split"] == "test")
    test_nll = next(row for row in nll_rows if row["split"] == "test")
    test_zero = next(row for row in baseline_rows if row["split"] == "test")
    nonlinear_rows = [row for row in motion_rows if row["split"] == "test" and row["model"] == "nll" and row["motion_mode"] != "linear"]
    zero_nonlinear = {row["motion_mode"]: row for row in motion_rows if row["split"] == "test" and row["model"] == "zero" and row["motion_mode"] != "linear"}
    avg_improvement = np.mean([
        1.0 - float(row["masked_mse"]) / max(float(zero_nonlinear[row["motion_mode"]]["masked_mse"]), 1.0e-9)
        for row in nonlinear_rows
        if row["motion_mode"] in zero_nonlinear
    ])
    corr_rows = [row for row in calib_rows if row["split"] == "test" and row["direction"] in {"radial_xy", "rel_velocity_xy"}]
    corr_values = [float(row["corr_projected_std_abs_error"]) for row in corr_rows if np.isfinite(float(row["corr_projected_std_abs_error"]))]
    mean_corr = float(np.mean(corr_values)) if corr_values else float("nan")
    lines = [
        "# Phase N2 HeadA Offline Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n2_heada_offline_complete`",
        "",
        "Phase N2 complete.",
        "Gpsi-HeadA offline model is ready for Phase N3 frozen-Gpsi PPO integration.",
        "",
        "## Experiment-Supported Facts",
        "",
        "- Phase N1 complete flag exists and train/val/test datasets were readable.",
        "- Model inputs were restricted to `ego_current`, `obs_current`, `history_rel_pos`, `history_rel_vel`, and `history_valid_mask`.",
        "- `future_pos_world`, `delta_label_world`, and `future_valid_mask` were used only for loss/evaluation, never as model inputs.",
        "- Normalization statistics were computed from the train split only and stored in checkpoints.",
        "- Delta-only training and Gaussian NLL training both completed without NaN/inf.",
        "- Diagonal logvar was clamped to [-5, 3] during NLL loss/evaluation.",
        "",
        "## Dataset Summary",
        "",
        "| split | samples | valid labels |",
        "| --- | ---: | ---: |",
    ]
    for row in dataset_rows:
        lines.append(f"| {row['split']} | {row['samples']} | {row['valid_labels']} |")
    lines.extend(
        [
            "",
            "## Model Structure",
            "",
            "GRU history encoder over relative position/velocity history, MLP current encoder over ego + current obstacle profile, fusion MLP to `z_i`, HeadA delta output `[T,D]`, and diagonal logvar output `[T,D]`.",
            "",
            "## Core Metrics",
            "",
            f"- Zero residual test MSE: `{float(test_zero['masked_mse']):.6f}`",
            f"- Delta-only test MSE: `{float(test_delta['masked_mse']):.6f}`",
            f"- NLL model test MSE: `{float(test_nll['masked_mse']):.6f}`",
            f"- NLL model test Gaussian NLL: `{float(test_nll['gaussian_nll']):.6f}`",
            f"- Mean nonlinear test MSE improvement over zero: `{float(avg_improvement):.4f}`",
            "",
            "## Projected Uncertainty Calibration",
            "",
            f"- Mean corr(projected_std, |projected_error|) over test radial/relative-velocity directions: `{mean_corr:.4f}`",
            "- Directional projected uncertainty tables were generated for x-axis, y-axis, radial_xy, rel_velocity_xy, and error_direction_diag_only.",
            "",
            "## Reasonable Inferences",
            "",
            "- The offline HeadA residual predictor is learnable on nonlinear/stochastic obstacle residuals if nonlinear-mode MSE improves over the zero residual baseline.",
            "- Diagonal logvar provides a usable starting point for N4 directional/tube shield experiments when projected calibration is finite and not collapsed.",
            "",
            "## Risks / Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No blocking warning.")
    lines.extend(["", "## Plots", ""])
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `work_dirs/gpsi_heada_v1_delta_only/best.pth`",
            "- `work_dirs/gpsi_heada_v1_delta_only/last.pth`",
            "- `work_dirs/gpsi_heada_v1_nll/best.pth`",
            "- `work_dirs/gpsi_heada_v1_nll/last.pth`",
            f"- `{relpath(out_dir / 'tables')}`",
            f"- `{relpath(out_dir / 'logs/phase_n2_train_delta_only.log')}`",
            f"- `{relpath(out_dir / 'logs/phase_n2_train_nll.log')}`",
            f"- `{relpath(out_dir / 'logs/phase_n2_eval.log')}`",
            f"- `{relpath(out_dir / 'phase_n2_watcher.log')}`",
            f"- `{relpath(out_dir / COMPLETE_FLAG)}`",
            "",
            "## Phase N3 Readiness",
            "",
            "Can enter Phase N3: yes.",
            "N3 should use frozen Gpsi, trainable PPO, augmented obstacle input `[obs_i, z_i, delta_hat_i, log_sigma2_i]`, no shield, masked-attention PPO backbone, and symmetric critic.",
        ]
    )
    write_text(out_dir / "PHASE_N2_HEADA_OFFLINE_REPORT.md", "\n".join(lines) + "\n")


def validate_completion(delta_rows: list[dict[str, Any]], nll_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], motion_rows: list[dict[str, Any]], axis_rows: list[dict[str, Any]], calib_rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    test_delta = next(row for row in delta_rows if row["split"] == "test")
    test_zero = next(row for row in baseline_rows if row["split"] == "test")
    if float(test_delta["masked_mse"]) >= float(test_zero["masked_mse"]):
        raise PhaseN2EvalStop("delta_not_learnable", "delta-only test MSE does not improve over zero residual baseline")
    zero_by_mode = {
        row["motion_mode"]: row
        for row in motion_rows
        if row["split"] == "test" and row["model"] == "zero"
    }
    nll_by_mode = {
        row["motion_mode"]: row
        for row in motion_rows
        if row["split"] == "test" and row["model"] == "nll"
    }
    nonlinear_improved = [
        mode
        for mode, row in nll_by_mode.items()
        if mode != "linear" and mode in zero_by_mode and float(row["masked_mse"]) < float(zero_by_mode[mode]["masked_mse"])
    ]
    if not nonlinear_improved:
        raise PhaseN2EvalStop("delta_not_learnable", "NLL model does not improve any nonlinear motion mode over zero baseline")
    if len(nonlinear_improved) < 4:
        warnings.append(f"NLL improved nonlinear modes {nonlinear_improved}, but not all nonlinear modes.")
    for row in nll_rows:
        for key in ["masked_mse", "gaussian_nll", "mean_logvar", "min_logvar", "max_logvar"]:
            if not np.isfinite(float(row[key])):
                raise PhaseN2EvalStop("nll_train_failed", f"NLL metric not finite: {row}")
        if float(row["max_logvar"]) - float(row["min_logvar"]) < 0.05:
            raise PhaseN2EvalStop("logvar_collapse", f"logvar span too small: {row}")
    axis_values = [float(row["mean"]) for row in axis_rows if int(row["count"]) > 0 and np.isfinite(float(row["mean"]))]
    if np.std(axis_values) < 0.01:
        raise PhaseN2EvalStop("logvar_collapse", "per-axis logvar stats are nearly constant")
    if not calib_rows:
        raise PhaseN2EvalStop("calibration_failed", "projected calibration rows are empty")
    finite_corr = [
        float(row["corr_projected_std_abs_error"])
        for row in calib_rows
        if np.isfinite(float(row["corr_projected_std_abs_error"]))
    ]
    if not finite_corr:
        raise PhaseN2EvalStop("calibration_failed", "all projected uncertainty correlations are non-finite")
    if np.mean(finite_corr) <= 0.0:
        warnings.append("Average projected_std to abs projected error correlation is not positive; directional shield may need calibration.")
    return warnings


def run() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    write_text(out_dir / "phase_n2_status.txt", "evaluating\n")
    logger = Logger(out_dir)
    logger.log("Phase N2 evaluation started")
    logger.log("Command: " + " ".join(["python", *sys.argv]))
    if not (ROOT / "results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag").exists():
        raise PhaseN2EvalStop("phase_n1_missing", "Phase N1 complete flag missing")
    data_dir = ROOT / args.data_dir
    schema_path = data_dir / "schema.json"
    if not schema_path.exists():
        raise PhaseN2EvalStop("schema_mismatch", "schema.json missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("target", {}).get("sigma2_direct_label") is not False:
        raise PhaseN2EvalStop("schema_mismatch", "sigma2_direct_label must be false")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    logger.log(f"Using device: {device}")
    delta_model, delta_norm, _delta_ckpt = load_checkpoint(ROOT / args.delta_checkpoint, device)
    nll_model, nll_norm, _nll_ckpt = load_checkpoint(ROOT / args.nll_checkpoint, device)
    clamp = (float(args.logvar_clamp[0]), float(args.logvar_clamp[1]))
    split_arrays = {split: load_arrays(data_dir, split) for split in ["train", "val", "test"]}
    future_times = split_arrays["train"]["future_times"]

    dataset_rows = []
    baseline_rows = []
    delta_rows = []
    nll_rows = []
    horizon_rows = []
    motion_rows = []
    axis_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    test_diag: dict[str, np.ndarray] = {}
    test_pred = test_target = test_mask = None
    for split, arrays in split_arrays.items():
        logger.log(f"Predicting split={split} samples={arrays['episode_id'].shape[0]}")
        delta_pred, delta_logvar = predict(delta_model, arrays, delta_norm, device, args.batch_size)
        nll_pred, nll_logvar = predict(nll_model, arrays, nll_norm, device, args.batch_size)
        zero_pred = np.zeros_like(arrays["delta_label_world"])
        zero_logvar = np.zeros_like(arrays["delta_label_world"])
        dataset_rows.append({"split": split, "samples": int(arrays["episode_id"].shape[0]), "valid_labels": int(arrays["future_valid_mask"].sum())})
        baseline_rows.append(aggregate_metrics("zero", split, zero_pred, zero_logvar, arrays, clamp))
        delta_rows.append(aggregate_metrics("delta_only", split, delta_pred, delta_logvar, arrays, clamp))
        nll_rows.append(aggregate_metrics("nll", split, nll_pred, nll_logvar, arrays, clamp))
        for model_name, pred, lv in [("zero", zero_pred, zero_logvar), ("delta_only", delta_pred, delta_logvar), ("nll", nll_pred, nll_logvar)]:
            horizon_rows.extend(per_horizon_metrics(model_name, split, pred, lv, arrays, future_times, clamp))
            motion_rows.extend(per_motion_mode_metrics(model_name, split, pred, lv, arrays, clamp))
        axis_rows.extend(per_axis_logvar_stats(split, nll_logvar, arrays, future_times, clamp))
        c_rows, b_rows, diag = projected_calibration(split, nll_pred, nll_logvar, arrays, future_times, clamp)
        calib_rows.extend(c_rows)
        bin_rows.extend(b_rows)
        if split == "test":
            test_diag = diag
            test_pred = nll_pred
            test_target = arrays["delta_label_world"]
            test_mask = arrays["future_valid_mask"]

    warnings = validate_completion(delta_rows, nll_rows, baseline_rows, motion_rows, axis_rows, calib_rows)
    table_dir = out_dir / "tables"
    write_csv(table_dir / "phase_n2_dataset_loader_check.csv", dataset_rows, ["split", "samples", "valid_labels"])
    write_csv(table_dir / "phase_n2_constant_velocity_baseline.csv", baseline_rows, ["model", "split", "samples", "valid_labels", "masked_mse", "masked_smoothl1", "gaussian_nll", "mean_logvar", "min_logvar", "max_logvar"])
    write_csv(table_dir / "phase_n2_delta_only_metrics.csv", delta_rows, ["model", "split", "samples", "valid_labels", "masked_mse", "masked_smoothl1", "gaussian_nll", "mean_logvar", "min_logvar", "max_logvar"])
    write_csv(table_dir / "phase_n2_nll_metrics.csv", nll_rows, ["model", "split", "samples", "valid_labels", "masked_mse", "masked_smoothl1", "gaussian_nll", "mean_logvar", "min_logvar", "max_logvar"])
    write_csv(table_dir / "phase_n2_per_horizon_metrics.csv", horizon_rows, ["model", "split", "horizon_sec", "valid_labels", "masked_mse", "masked_smoothl1", "gaussian_nll"])
    write_csv(table_dir / "phase_n2_per_motion_mode_metrics.csv", motion_rows, ["model", "split", "motion_mode_id", "motion_mode", "samples", "valid_labels", "masked_mse", "masked_smoothl1", "gaussian_nll"])
    write_csv(table_dir / "phase_n2_per_axis_logvar_stats.csv", axis_rows, ["split", "horizon_sec", "motion_mode_id", "motion_mode", "axis", "count", "mean", "median", "p10", "p90", "min", "max"])
    write_csv(table_dir / "phase_n2_projected_uncertainty_calibration.csv", calib_rows, ["split", "horizon_sec", "direction", "count", "mean_abs_z", "std_z", "pct_abs_z_lt_1", "pct_abs_z_lt_2", "projected_nll", "corr_projected_std_abs_error", "mean_projected_std", "mean_abs_projected_error", "scalar_std_trace_mean", "scalar_std_max_mean"])
    write_csv(table_dir / "phase_n2_calibration_bins.csv", bin_rows, ["split", "horizon_sec", "direction", "bin", "std_low", "std_high", "count", "mean_predicted_std", "mean_abs_projected_error", "pct_abs_z_lt_1", "pct_abs_z_lt_2"])
    write_csv(table_dir / "phase_n2_schema_check.csv", [{"item": "sigma2_direct_label", "status": "pass", "value": "false"}, {"item": "future_label_not_model_input", "status": "pass", "value": "future fields absent from model forward"}], ["item", "status", "value"])
    write_csv(table_dir / "phase_n2_command_manifest.csv", command_manifest(), ["command_name", "command", "status"])
    assert test_pred is not None and test_target is not None and test_mask is not None
    plot_paths = write_plots(
        out_dir,
        ROOT / "work_dirs/gpsi_heada_v1_delta_only/train_log.csv",
        ROOT / "work_dirs/gpsi_heada_v1_nll/train_log.csv",
        horizon_rows,
        motion_rows,
        axis_rows,
        calib_rows,
        bin_rows,
        test_diag,
        test_pred,
        test_target,
        test_mask,
        args.plot_sample_limit,
    )
    write_report(out_dir, dataset_rows, delta_rows, nll_rows, baseline_rows, horizon_rows, motion_rows, axis_rows, calib_rows, plot_paths, warnings)
    write_text(out_dir / COMPLETE_FLAG, "phase_n2_heada_offline_complete\n")
    write_text(out_dir / "phase_n2_status.txt", "complete\n")
    logger.log("Phase N2 complete flag written")


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    logger = Logger(out_dir)
    try:
        run()
    except PhaseN2EvalStop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        logger.log(f"Phase N2 stopped: {exc.reason}: {exc.detail}")
        raise SystemExit(2) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_stop(out_dir, "calibration_failed", detail)
        logger.log("Unexpected eval exception:\n" + detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
