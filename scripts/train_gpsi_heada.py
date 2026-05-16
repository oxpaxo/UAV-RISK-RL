from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gpsi_head_a import GpsiHeadA


RESULT_DIR = ROOT / "results/env_v2_phase_n2_gpsi_heada_offline"
STOP_FLAGS = {
    "phase_n1_missing": "PHASE_N2_STOP_PHASE_N1_MISSING.flag",
    "dataset_read_failed": "PHASE_N2_STOP_DATASET_READ_FAILED.flag",
    "schema_mismatch": "PHASE_N2_STOP_SCHEMA_MISMATCH.flag",
    "delta_train_failed": "PHASE_N2_STOP_DELTA_TRAIN_FAILED.flag",
    "delta_not_learnable": "PHASE_N2_STOP_DELTA_NOT_LEARNABLE.flag",
    "nll_train_failed": "PHASE_N2_STOP_NLL_TRAIN_FAILED.flag",
    "logvar_collapse": "PHASE_N2_STOP_LOGVAR_COLLAPSE.flag",
}

INPUT_KEYS = ["ego_current", "obs_current", "history_rel_pos", "history_rel_vel", "history_valid_mask"]
LABEL_KEYS = ["delta_label_world", "future_valid_mask"]
FORBIDDEN_INPUT_KEYS = {"future_pos_world", "constant_velocity_pos_world", "delta_label_world", "future_valid_mask"}


class PhaseN2TrainStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, result_dir: Path, loss: str) -> None:
        self.path = result_dir / "logs" / (
            "phase_n2_train_nll.log" if loss == "gaussian_nll" else "phase_n2_train_delta_only.log"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class GpsiDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], norm: dict[str, torch.Tensor]) -> None:
        self.arrays = arrays
        self.norm = norm
        self.n = int(arrays["episode_id"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "ego_current": torch.from_numpy(self.arrays["ego_current"][index]).float(),
            "obs_current": torch.from_numpy(self.arrays["obs_current"][index]).float(),
            "history_rel_pos": torch.from_numpy(self.arrays["history_rel_pos"][index]).float(),
            "history_rel_vel": torch.from_numpy(self.arrays["history_rel_vel"][index]).float(),
            "history_valid_mask": torch.from_numpy(self.arrays["history_valid_mask"][index]).float(),
            "delta_label_world": torch.from_numpy(self.arrays["delta_label_world"][index]).float(),
            "future_valid_mask": torch.from_numpy(self.arrays["future_valid_mask"][index]).float(),
        }
        item["ego_current"] = normalize_tensor(item["ego_current"], self.norm["ego_current_mean"], self.norm["ego_current_std"])
        item["obs_current"] = normalize_tensor(item["obs_current"], self.norm["obs_current_mean"], self.norm["obs_current_std"])
        item["history_rel_pos"] = normalize_tensor(item["history_rel_pos"], self.norm["history_rel_pos_mean"], self.norm["history_rel_pos_std"])
        item["history_rel_vel"] = normalize_tensor(item["history_rel_vel"], self.norm["history_rel_vel_mean"], self.norm["history_rel_vel_std"])
        return item


def normalize_tensor(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / torch.clamp(std, min=1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gpsi-HeadA offline.")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", choices=["delta_mse", "delta_smoothl1", "gaussian_nll"], required=True)
    parser.add_argument("--init", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logvar-clamp", nargs=2, type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def ensure_dirs(out_dir: Path, result_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "logs").mkdir(parents=True, exist_ok=True)
    (result_dir / "tables").mkdir(parents=True, exist_ok=True)


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


def write_stop(reason: str, detail: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["delta_train_failed"])
    write_text(RESULT_DIR / flag, f"{reason}\n{detail}\n")
    write_text(RESULT_DIR / "phase_n2_status.txt", f"stopped:{flag}\n")
    write_text(
        RESULT_DIR / "PHASE_N2_HEADA_OFFLINE_REPORT.md",
        f"# Phase N2 HeadA Offline Report\n\n`terminal_decision = phase_n2_stopped_{reason}`\n\n```text\n{detail}\n```\n",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_json(ROOT / args.config)
    training = cfg.setdefault("training", {})
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["lr"] = args.lr
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers
    if args.logvar_clamp is not None:
        training["logvar_clamp"] = args.logvar_clamp
    training["loss"] = args.loss
    training["seed"] = args.seed
    training["init"] = args.init
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_npz(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"{split}.npz"
    if not path.exists():
        raise PhaseN2TrainStop("dataset_read_failed", f"missing split file: {path}")
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    for key in [*INPUT_KEYS, *LABEL_KEYS, "motion_mode_id"]:
        if key not in arrays:
            raise PhaseN2TrainStop("schema_mismatch", f"{split}.npz missing required key: {key}")
    return arrays


def compute_norm(train: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    norm: dict[str, torch.Tensor] = {}
    for key in ["ego_current", "obs_current"]:
        arr = train[key].astype(np.float32)
        norm[f"{key}_mean"] = torch.from_numpy(arr.mean(axis=0)).float()
        norm[f"{key}_std"] = torch.from_numpy(arr.std(axis=0) + 1.0e-6).float()
    for key in ["history_rel_pos", "history_rel_vel"]:
        arr = train[key].astype(np.float32)
        mask = train["history_valid_mask"].astype(bool)
        if mask.any():
            values = arr[mask]
        else:
            values = arr.reshape(-1, arr.shape[-1])
        norm[f"{key}_mean"] = torch.from_numpy(values.mean(axis=0)).float()
        norm[f"{key}_std"] = torch.from_numpy(values.std(axis=0) + 1.0e-6).float()
    return norm


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_name: str,
    logvar_clamp: tuple[float, float],
) -> dict[str, torch.Tensor]:
    target = batch["delta_label_world"]
    mask = batch["future_valid_mask"].unsqueeze(-1)
    denom = torch.clamp(mask.sum() * target.shape[-1], min=1.0)
    error = output["delta_hat"] - target
    mse = ((error.square() * mask).sum() / denom)
    smooth = (torch.nn.functional.smooth_l1_loss(output["delta_hat"], target, reduction="none") * mask).sum() / denom
    if loss_name == "delta_mse":
        loss = mse
        nll = torch.zeros_like(loss)
        logvar = torch.clamp(output["logvar_hat"].detach(), min=logvar_clamp[0], max=logvar_clamp[1])
    elif loss_name == "delta_smoothl1":
        loss = smooth
        nll = torch.zeros_like(loss)
        logvar = torch.clamp(output["logvar_hat"].detach(), min=logvar_clamp[0], max=logvar_clamp[1])
    else:
        logvar = torch.clamp(output["logvar_hat"], min=logvar_clamp[0], max=logvar_clamp[1])
        nll_terms = 0.5 * (torch.exp(-logvar) * error.square() + logvar)
        nll = (nll_terms * mask).sum() / denom
        loss = nll
    return {
        "loss": loss,
        "mse": mse.detach(),
        "smoothl1": smooth.detach(),
        "nll": nll.detach(),
        "mean_logvar": logvar.mean().detach(),
        "min_logvar": logvar.min().detach(),
        "max_logvar": logvar.max().detach(),
    }


def run_epoch(
    model: GpsiHeadA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_name: str,
    logvar_clamp: tuple[float, float],
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ["loss", "mse", "smoothl1", "nll", "mean_logvar", "min_logvar", "max_logvar"]}
    batches = 0
    with torch.set_grad_enabled(training):
        for batch_cpu in loader:
            batch = move_batch(batch_cpu, device)
            output = model(
                batch["ego_current"],
                batch["obs_current"],
                batch["history_rel_pos"],
                batch["history_rel_vel"],
                batch["history_valid_mask"],
            )
            losses = masked_losses(output, batch, loss_name, logvar_clamp)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def zero_baseline_mse(arrays: dict[str, np.ndarray]) -> float:
    target = arrays["delta_label_world"].astype(np.float64)
    mask = arrays["future_valid_mask"].astype(np.float64)[..., None]
    denom = max(float(mask.sum() * target.shape[-1]), 1.0)
    return float(((target ** 2) * mask).sum() / denom)


def save_checkpoint(
    path: Path,
    model: GpsiHeadA,
    cfg: dict[str, Any],
    norm: dict[str, torch.Tensor],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "config": cfg,
        "normalization": {key: value.detach().cpu() for key, value in norm.items()},
        "epoch": epoch,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    data_dir = ROOT / args.data_dir
    ensure_dirs(out_dir, RESULT_DIR)
    logger = Logger(RESULT_DIR, args.loss)
    logger.log("Phase N2 Gpsi HeadA training started")
    logger.log("Command: " + " ".join(["python", *sys.argv]))
    if not (ROOT / "results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag").exists():
        raise PhaseN2TrainStop("phase_n1_missing", "Phase N1 complete flag missing")
    cfg = resolve_config(args)
    save_json(out_dir / "config_resolved.yaml", cfg)
    set_seed(args.seed)
    device = device_from_arg(args.device)
    logger.log(f"Using device: {device}")

    train_arrays = load_npz(data_dir, "train")
    val_arrays = load_npz(data_dir, "val")
    _test_arrays = load_npz(data_dir, "test")
    schema_path = data_dir / "schema.json"
    if not schema_path.exists():
        raise PhaseN2TrainStop("schema_mismatch", "schema.json missing")
    schema = load_json(schema_path)
    if schema.get("target", {}).get("sigma2_direct_label") is not False:
        raise PhaseN2TrainStop("schema_mismatch", "schema must declare sigma2_direct_label=false")
    forbidden = FORBIDDEN_INPUT_KEYS & set(INPUT_KEYS)
    if forbidden:
        raise PhaseN2TrainStop("schema_mismatch", f"forbidden model input keys configured: {sorted(forbidden)}")

    norm_cpu = compute_norm(train_arrays)
    train_ds = GpsiDataset(train_arrays, norm_cpu)
    val_ds = GpsiDataset(val_arrays, norm_cpu)
    batch_size = int(cfg["training"].get("batch_size", 1024))
    num_workers = int(cfg["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model = GpsiHeadA(cfg["model"]).to(device)
    if args.init:
        checkpoint = torch.load(ROOT / args.init, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        logger.log(f"Loaded init checkpoint {args.init}; missing={missing}, unexpected={unexpected}")
        if "normalization" in checkpoint:
            norm_cpu = {key: value.detach().cpu() for key, value in checkpoint["normalization"].items()}
            train_ds.norm = norm_cpu
            val_ds.norm = norm_cpu

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-6)),
    )
    epochs = int(cfg["training"].get("epochs", 20))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    clamp_values = cfg["training"].get("logvar_clamp", [-5.0, 3.0])
    if args.logvar_clamp is not None:
        clamp_values = args.logvar_clamp
    logvar_clamp = (float(clamp_values[0]), float(clamp_values[1]))
    best_metric = math.inf
    train_log: list[dict[str, Any]] = []
    zero_val_mse = zero_baseline_mse(val_arrays)
    logger.log(f"Zero residual val MSE baseline: {zero_val_mse:.6f}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.loss, logvar_clamp, grad_clip)
        val_metrics = run_epoch(model, val_loader, None, device, args.loss, logvar_clamp, grad_clip)
        monitor = val_metrics["nll"] if args.loss == "gaussian_nll" else val_metrics["mse"]
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["zero_val_mse"] = zero_val_mse
        train_log.append(row)
        logger.log(
            f"epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f} "
            f"val_nll={val_metrics['nll']:.6f} mean_logvar={val_metrics['mean_logvar']:.4f}"
        )
        save_checkpoint(out_dir / "last.pth", model, cfg, norm_cpu, epoch, val_metrics)
        if monitor < best_metric:
            best_metric = monitor
            save_checkpoint(out_dir / "best.pth", model, cfg, norm_cpu, epoch, val_metrics)

    write_csv(
        out_dir / "train_log.csv",
        train_log,
        [
            "epoch",
            "train_loss",
            "train_mse",
            "train_smoothl1",
            "train_nll",
            "train_mean_logvar",
            "train_min_logvar",
            "train_max_logvar",
            "val_loss",
            "val_mse",
            "val_smoothl1",
            "val_nll",
            "val_mean_logvar",
            "val_min_logvar",
            "val_max_logvar",
            "zero_val_mse",
        ],
    )
    best = torch.load(out_dir / "best.pth", map_location="cpu", weights_only=False)
    best_mse = float(best["metrics"]["mse"])
    if args.loss != "gaussian_nll":
        if not np.isfinite(best_mse):
            raise PhaseN2TrainStop("delta_train_failed", "delta-only best val MSE is not finite")
        if best_mse >= zero_val_mse:
            raise PhaseN2TrainStop(
                "delta_not_learnable",
                f"delta-only best val MSE {best_mse:.6f} does not improve over zero baseline {zero_val_mse:.6f}",
            )
    else:
        metrics = best["metrics"]
        if not all(np.isfinite(float(metrics[key])) for key in ["nll", "mse", "mean_logvar", "min_logvar", "max_logvar"]):
            raise PhaseN2TrainStop("nll_train_failed", f"NLL metrics are not finite: {metrics}")
        span = float(metrics["max_logvar"]) - float(metrics["min_logvar"])
        if span < 0.05:
            raise PhaseN2TrainStop("logvar_collapse", f"logvar span too small in best checkpoint: {span:.6f}")
    logger.log("Training completed")


def main() -> None:
    try:
        train()
    except PhaseN2TrainStop as exc:
        write_stop(exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        reason = "nll_train_failed" if "--loss gaussian_nll" in " ".join(sys.argv) else "delta_train_failed"
        write_stop(reason, detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
