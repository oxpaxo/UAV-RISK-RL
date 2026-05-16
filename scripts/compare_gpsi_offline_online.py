from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase_n3_5_audit_common import (
    ROOT,
    PhaseN35Stop,
    check_prerequisites,
    command_manifest_row,
    ensure_dirs,
    forward_gpsi_np,
    load_gpsi_checkpoint,
    load_npz,
    make_wrapper,
    metric_diff_rows,
    rel,
    save_scatter,
    write_csv,
    write_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3.5 offline-online Gpsi wrapper equivalence audit.")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--checkpoint", default="work_dirs/gpsi_heada_v1_nll/best.pth")
    parser.add_argument("--wrapper", default="envs/wrappers/gpsi_obs_wrapper.py")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3_5_gpsi_wrapper_audit")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--repaired-std-threshold", type=float, default=1.0e-5)
    parser.add_argument("--repaired-std-floor", type=float, default=1.0)
    return parser.parse_args()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def select_indices(arrays: dict[str, np.ndarray], n: int, seed: int) -> np.ndarray:
    total = int(arrays["ego_current"].shape[0])
    if n >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    mode = arrays.get("motion_mode_id", np.zeros(total, dtype=np.int16))
    hist_len = arrays.get("history_valid_length", arrays["history_valid_mask"].sum(axis=1)).astype(np.float32)
    repl = arrays.get("replacement_boundary_nearby", np.zeros(total, dtype=np.int8))
    groups: dict[tuple[int, str, int], list[int]] = {}
    for idx in range(total):
        length_bin = "partial" if float(hist_len[idx]) < arrays["history_valid_mask"].shape[1] else "full"
        key = (int(mode[idx]), length_bin, int(repl[idx]))
        groups.setdefault(key, []).append(idx)
    selected: list[int] = []
    target_per_group = max(1, n // max(len(groups), 1))
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.int64)
        take = min(target_per_group, values.size)
        if take:
            selected.extend(rng.choice(values, size=take, replace=False).tolist())
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(total, dtype=np.int64), np.asarray(selected, dtype=np.int64), assume_unique=False)
        take = min(n - len(selected), remaining.size)
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    selected = selected[:n]
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def subset_inputs(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ego_current": arrays["ego_current"][indices].astype(np.float32),
        "obs_current": arrays["obs_current"][indices].astype(np.float32),
        "history_rel_pos": arrays["history_rel_pos"][indices].astype(np.float32),
        "history_rel_vel": arrays["history_rel_vel"][indices].astype(np.float32),
        "history_valid_mask": arrays["history_valid_mask"][indices].astype(np.float32),
    }


def rows_for_stage(
    *,
    stage: str,
    inputs: dict[str, np.ndarray],
    norm_payload: dict[str, torch.Tensor],
    model: Any,
    checkpoint: Path,
    device: torch.device,
    threshold: float,
    floor: float,
    tolerance: float,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    offline_out, offline_norm = forward_gpsi_np(
        model,
        inputs,
        norm_payload,
        device,
        threshold=threshold,
        floor=floor,
        batch_size=batch_size,
    )
    wrapper = make_wrapper(
        checkpoint,
        "eval_flow_id",
        str(device),
        threshold=threshold,
        floor=floor,
    )
    online_norm = {
        "ego_current": wrapper._normalize_numpy(inputs["ego_current"], "ego_current"),
        "obs_current": wrapper._normalize_numpy(inputs["obs_current"], "obs_current"),
        "history_rel_pos": wrapper._normalize_numpy(inputs["history_rel_pos"], "history_rel_pos"),
        "history_rel_vel": wrapper._normalize_numpy(inputs["history_rel_vel"], "history_rel_vel"),
        "history_valid_mask": inputs["history_valid_mask"].astype(np.float32),
    }
    online_out = wrapper._forward_gpsi(
        inputs["ego_current"],
        inputs["obs_current"],
        inputs["history_rel_pos"],
        inputs["history_rel_vel"],
        inputs["history_valid_mask"],
    )
    rows: list[dict[str, Any]] = []
    for key in ["ego_current", "obs_current", "history_rel_pos", "history_rel_vel", "history_valid_mask"]:
        row = metric_diff_rows(f"normalized_{key}", offline_norm[key], online_norm[key], tolerance=tolerance)
        row["repair_stage"] = stage
        rows.append(row)
    for key in ["z", "delta_hat", "logvar_hat"]:
        row = metric_diff_rows(key, offline_out[key], online_out[key], tolerance=tolerance)
        row["repair_stage"] = stage
        rows.append(row)
    return rows, offline_out, online_out


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    checkpoint = ROOT / args.checkpoint
    log_path = out_dir / "logs/phase_n3_5_offline_online_compare.log"
    ensure_dirs(out_dir)
    try:
        check_prerequisites(out_dir, checkpoint)
        data_path = ROOT / args.data_dir / f"{args.split}.npz"
        if not data_path.exists():
            raise PhaseN35Stop("offline_online_mismatch", f"missing split data: {rel(data_path)}")
        arrays = load_npz(data_path)
        indices = select_indices(arrays, int(args.num_samples), int(args.seed))
        inputs = subset_inputs(arrays, indices)
        device = device_from_arg(args.device)
        model, norm_payload, ckpt = load_gpsi_checkpoint(checkpoint, device)
        rows: list[dict[str, Any]] = []
        before_rows, before_offline, before_online = rows_for_stage(
            stage="before_fix",
            inputs=inputs,
            norm_payload=norm_payload,
            model=model,
            checkpoint=checkpoint,
            device=device,
            threshold=0.0,
            floor=1.0,
            tolerance=float(args.tolerance),
            batch_size=int(args.batch_size),
        )
        rows.extend(before_rows)
        after_rows, after_offline, after_online = rows_for_stage(
            stage="after_fix",
            inputs=inputs,
            norm_payload=norm_payload,
            model=model,
            checkpoint=checkpoint,
            device=device,
            threshold=float(args.repaired_std_threshold),
            floor=float(args.repaired_std_floor),
            tolerance=float(args.tolerance),
            batch_size=int(args.batch_size),
        )
        rows.extend(after_rows)
        for row in rows:
            row["split"] = args.split
            row["num_samples"] = int(indices.size)
            row["checkpoint"] = rel(checkpoint)
            row["device"] = str(device)
            row["model_version"] = str(ckpt.get("config", {}).get("version", "unknown"))
        fieldnames = [
            "repair_stage",
            "split",
            "num_samples",
            "component",
            "shape",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse_diff",
            "corr",
            "allclose_pass",
            "tolerance",
            "nan_count_offline",
            "nan_count_online",
            "inf_count_offline",
            "inf_count_online",
            "checkpoint",
            "device",
            "model_version",
        ]
        write_csv(out_dir / "tables/phase_n3_5_offline_online_equivalence.csv", rows, fieldnames)
        sample_rows = []
        for i, idx in enumerate(indices[:200]):
            sample_rows.append(
                {
                    "sample_order": i,
                    "source_index": int(idx),
                    "episode_id": int(arrays["episode_id"][idx]),
                    "step": int(arrays["step"][idx]),
                    "obstacle_id": int(arrays["obstacle_id"][idx]),
                    "obstacle_slot": int(arrays["obstacle_slot"][idx]),
                    "motion_mode_id": int(arrays["motion_mode_id"][idx]),
                    "threat_class_id": int(arrays["threat_class_id"][idx]),
                    "history_valid_length": int(arrays["history_valid_mask"][idx].sum()),
                    "replacement_boundary_nearby": int(arrays["replacement_boundary_nearby"][idx]),
                }
            )
        write_csv(out_dir / "tables/phase_n3_5_equivalence_sample_manifest.csv", sample_rows)
        save_scatter(
            out_dir / "plots/offline_vs_online_delta_scatter.png",
            after_offline["delta_hat"],
            after_online["delta_hat"],
            "Offline vs Online Delta Hat",
            "offline",
            "online",
        )
        save_scatter(
            out_dir / "plots/offline_vs_online_logvar_scatter.png",
            after_offline["logvar_hat"],
            after_online["logvar_hat"],
            "Offline vs Online Logvar Hat",
            "offline",
            "online",
        )
        manifest_path = out_dir / "tables/phase_n3_5_command_manifest.csv"
        manifest = read_existing_manifest(manifest_path)
        manifest.append(command_manifest_row("offline_online_equivalence", " ".join(sys.argv), "completed"))
        write_csv(manifest_path, manifest, ["command_name", "command", "status"])
        write_csv(out_dir / "tables/phase_n3_5_offline_online_status.csv", [{"status": "pass", "samples": int(indices.size)}])
        log_path.write_text(
            "offline-online equivalence completed\n"
            f"samples={indices.size}\n"
            f"max_after_delta_diff={next(row['max_abs_diff'] for row in rows if row['repair_stage']=='after_fix' and row['component']=='delta_hat')}\n",
            encoding="utf-8",
        )
        failed = [
            row
            for row in rows
            if row["repair_stage"] == "after_fix"
            and row["component"] in {"z", "delta_hat", "logvar_hat", "history_valid_mask"}
            and int(row["allclose_pass"]) != 1
        ]
        if failed:
            detail = "offline-online equivalence failed after repair:\n" + "\n".join(str(row) for row in failed)
            raise PhaseN35Stop("offline_online_mismatch", detail)
        return 0
    except PhaseN35Stop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        return 2


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
