from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gpsi_head_a import GpsiHeadA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen-Gpsi z statistics on an N1/N2 split.")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--checkpoint", default="work_dirs/gpsi_heada_v1_nll/best.pth")
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--std-floor", type=float, default=1.0e-3)
    parser.add_argument("--degenerate-std-threshold", type=float, default=1.0e-5)
    parser.add_argument("--degenerate-std-floor", type=float, default=1.0)
    return parser.parse_args()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    write_csv(path, old + rows)


def effective_std(std: torch.Tensor, threshold: float, floor: float) -> torch.Tensor:
    std = std.float().clamp(min=1.0e-6)
    if threshold <= 0.0:
        return std
    return torch.where(std <= threshold, torch.full_like(std, max(float(floor), 1.0e-6)), std)


def normalize_np(
    value: np.ndarray,
    norm: dict[str, torch.Tensor],
    key: str,
    *,
    threshold: float,
    floor: float,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).float()
    mean = norm[f"{key}_mean"].detach().cpu().float()
    std = effective_std(norm[f"{key}_std"].detach().cpu(), threshold, floor)
    return ((tensor - mean) / std).numpy().astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GpsiHeadA, dict[str, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing Gpsi checkpoint: {rel(path)}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"invalid Gpsi checkpoint schema: {rel(path)}")
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model = GpsiHeadA(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw_norm = checkpoint.get("normalization", {})
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
    missing = [key for key in required if key not in raw_norm]
    if missing:
        raise ValueError(f"Gpsi checkpoint missing normalization stats: {missing}")
    norm = {key: torch.as_tensor(raw_norm[key], dtype=torch.float32) for key in required}
    return model, norm, checkpoint


def load_split(data_dir: Path, split: str, max_samples: int) -> dict[str, np.ndarray]:
    path = data_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing split: {rel(path)}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = ["ego_current", "obs_current", "history_rel_pos", "history_rel_vel", "history_valid_mask"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{split}.npz missing required keys: {missing}")
    n = int(arrays["ego_current"].shape[0])
    limit = n if max_samples <= 0 else min(n, int(max_samples))
    return {key: value[:limit] for key, value in arrays.items()}


def percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def main() -> int:
    args = parse_args()
    data_dir = ROOT / args.data_dir
    checkpoint_path = ROOT / args.checkpoint
    out_csv = ROOT / args.out
    out_npz = ROOT / args.out_npz
    device = device_from_arg(args.device)

    model, norm, checkpoint = load_checkpoint(checkpoint_path, device)
    arrays = load_split(data_dir, args.split, int(args.max_samples))
    n = int(arrays["ego_current"].shape[0])
    normalized = {
        "ego_current": normalize_np(
            arrays["ego_current"],
            norm,
            "ego_current",
            threshold=float(args.degenerate_std_threshold),
            floor=float(args.degenerate_std_floor),
        ),
        "obs_current": normalize_np(
            arrays["obs_current"],
            norm,
            "obs_current",
            threshold=float(args.degenerate_std_threshold),
            floor=float(args.degenerate_std_floor),
        ),
        "history_rel_pos": normalize_np(
            arrays["history_rel_pos"],
            norm,
            "history_rel_pos",
            threshold=float(args.degenerate_std_threshold),
            floor=float(args.degenerate_std_floor),
        ),
        "history_rel_vel": normalize_np(
            arrays["history_rel_vel"],
            norm,
            "history_rel_vel",
            threshold=float(args.degenerate_std_threshold),
            floor=float(args.degenerate_std_floor),
        ),
        "history_valid_mask": np.asarray(arrays["history_valid_mask"], dtype=np.float32),
    }

    z_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, int(args.batch_size)):
            end = min(start + int(args.batch_size), n)
            output = model(
                torch.from_numpy(normalized["ego_current"][start:end]).float().to(device),
                torch.from_numpy(normalized["obs_current"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_rel_pos"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_rel_vel"][start:end]).float().to(device),
                torch.from_numpy(normalized["history_valid_mask"][start:end]).float().to(device),
            )
            z_parts.append(output["z"].detach().cpu().numpy().astype(np.float32))
    z = np.concatenate(z_parts, axis=0)
    z_mean = z.mean(axis=0).astype(np.float32)
    z_std = z.std(axis=0).astype(np.float32)
    z_std_effective = np.maximum(z_std, max(float(args.std_floor), 1.0e-8)).astype(np.float32)
    z_norm = (z - z_mean[None, :]) / z_std_effective[None, :]
    l2_raw = np.linalg.norm(z, axis=1)
    l2_norm = np.linalg.norm(z_norm, axis=1)
    rows: list[dict[str, Any]] = [
        {
            "row_type": "summary",
            "split": args.split,
            "samples": int(n),
            "checkpoint": rel(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
            "z_dim": int(z.shape[1]),
            "std_floor": float(args.std_floor),
            "degenerate_z_std_dims": int((z_std < float(args.std_floor)).sum()),
            "z_l2_mean_raw": float(np.mean(l2_raw)),
            "z_l2_median_raw": float(np.median(l2_raw)),
            "z_l2_p95_raw": percentile(l2_raw, 95),
            "z_l2_max_raw": float(np.max(l2_raw)),
            "z_l2_mean_after_norm": float(np.mean(l2_norm)),
            "z_l2_median_after_norm": float(np.median(l2_norm)),
            "z_l2_p95_after_norm": percentile(l2_norm, 95),
            "z_l2_max_after_norm": float(np.max(l2_norm)),
            "z_raw_nan_count": int(np.isnan(z).sum()),
            "z_raw_inf_count": int(np.isinf(z).sum()),
            "command": " ".join(["python", *sys.argv]),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    for idx in range(z.shape[1]):
        rows.append(
            {
                "row_type": "per_dim",
                "split": args.split,
                "dim": int(idx),
                "z_mean": float(z_mean[idx]),
                "z_std": float(z_std[idx]),
                "z_std_effective": float(z_std_effective[idx]),
                "std_floored": int(z_std[idx] < float(args.std_floor)),
                "z_min": float(np.min(z[:, idx])),
                "z_max": float(np.max(z[:, idx])),
                "z_p05": percentile(z[:, idx], 5),
                "z_p50": percentile(z[:, idx], 50),
                "z_p95": percentile(z[:, idx], 95),
            }
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        z_mean=z_mean,
        z_std=z_std,
        z_std_effective=z_std_effective,
        z_std_floor=np.asarray(float(args.std_floor), dtype=np.float32),
        split=np.asarray(args.split),
        samples=np.asarray(n, dtype=np.int64),
        checkpoint=np.asarray(rel(checkpoint_path)),
    )
    write_csv(out_csv, rows)
    manifest_prefix = out_csv.stem.rsplit("_z_stats", 1)[0] if out_csv.stem.endswith("_z_stats") else "phase_n3r"
    append_csv(
        out_csv.parents[1] / f"tables/{manifest_prefix}_command_manifest.csv",
        [
            {
                "stage": "z_stats",
                "method_key": "z_norm",
                "method": "gpsi_full_z_norm_repaired",
                "command": " ".join(["python", *sys.argv]),
                "checkpoint": rel(checkpoint_path),
                "split": args.split,
                "samples": int(n),
                "out": rel(out_csv),
                "out_npz": rel(out_npz),
            }
        ],
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "split": args.split,
                "samples": n,
                "out": rel(out_csv),
                "out_npz": rel(out_npz),
                "z_l2_p95_raw": rows[0]["z_l2_p95_raw"],
                "z_l2_p95_after_norm": rows[0]["z_l2_p95_after_norm"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
