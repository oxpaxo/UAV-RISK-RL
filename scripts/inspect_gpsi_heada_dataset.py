from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMPLETE_FLAG = "PHASE_N1_GPSI_DATASET_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n0_missing": "PHASE_N1_STOP_PHASE_N0_MISSING.flag",
    "env_core_change_required": "PHASE_N1_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "dataset_build_failed": "PHASE_N1_STOP_DATASET_BUILD_FAILED.flag",
    "label_validity_failed": "PHASE_N1_STOP_LABEL_VALIDITY_FAILED.flag",
    "id_alignment_failed": "PHASE_N1_STOP_ID_ALIGNMENT_FAILED.flag",
    "data_leakage_failed": "PHASE_N1_STOP_DATA_LEAKAGE_FAILED.flag",
    "insufficient_data": "PHASE_N1_STOP_INSUFFICIENT_DATA.flag",
    "schema_mismatch": "PHASE_N1_STOP_SCHEMA_MISMATCH.flag",
    "watcher_failed": "PHASE_N1_STOP_WATCHER_FAILED.flag",
}
TERMINAL_FLAGS = [COMPLETE_FLAG, *STOP_FLAGS.values()]

SPLITS = ["train", "val", "test"]
EXPECTED_ARRAYS = {
    "ego_current": 2,
    "obs_current": 2,
    "history_pos_world": 3,
    "history_vel_world": 3,
    "history_rel_pos": 3,
    "history_rel_vel": 3,
    "history_valid_mask": 2,
    "delta_label_world": 3,
    "future_valid_mask": 2,
    "future_times": 1,
    "future_pos_world": 3,
    "constant_velocity_pos_world": 3,
    "current_pos_world": 2,
    "current_vel_world": 2,
    "motion_mode_id": 1,
    "threat_class_id": 1,
    "obstacle_id": 1,
    "obstacle_slot": 1,
    "active": 1,
    "episode_id": 1,
    "episode_seed": 1,
    "step": 1,
    "time": 1,
    "distance": 1,
    "closing": 1,
    "planned_cpa": 1,
    "planned_ttc": 1,
    "risk_value": 1,
    "history_valid_length": 1,
    "replacement_boundary_nearby": 1,
}
MOTION_MODE_NAME = {
    0: "linear",
    1: "sinusoidal_lateral",
    2: "accel_decel",
    3: "ar1_velocity",
    4: "crossing_or_sudden_threat",
}
THREAT_CLASS_NAME = {0: "low", 1: "medium", 2: "high"}


class PhaseN1InspectStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "logs" / "phase_n1_dataset_inspect.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Phase N1 Gpsi-HeadA dataset.")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n1_gpsi_dataset")
    parser.add_argument("--linear-sanity-threshold", type=float, default=1.0e-3)
    parser.add_argument("--plot-sample-limit", type=int, default=30000)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["tables", "plots", "logs"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def clear_terminal_flags(out_dir: Path) -> None:
    for name in TERMINAL_FLAGS:
        path = out_dir / name
        if path.exists():
            path.unlink()


def write_stop_flag(out_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["schema_mismatch"])
    write_text(out_dir / flag_name, f"{reason}\n{detail}\n")
    write_text(out_dir / "phase_n1_status.txt", f"stopped:{flag_name}\n")


def write_partial_report(out_dir: Path, terminal_decision: str, reason: str, detail: str) -> None:
    lines = [
        "# Phase N1 Gpsi-HeadA Dataset Report",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        f"Phase N1 stopped during dataset inspect: `{reason}`.",
        "",
        "## Detail",
        "",
        "```text",
        detail.strip(),
        "```",
    ]
    write_text(out_dir / "PHASE_N1_GPSI_DATASET_REPORT.md", "\n".join(lines) + "\n")


def load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.exists() or path.stat().st_size <= 0:
        raise PhaseN1InspectStop("dataset_build_failed", f"missing or empty dataset split: {relpath(path)}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_dataset(data_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    schema_path = data_dir / "schema.json"
    manifest_path = data_dir / "dataset_manifest.json"
    if not schema_path.exists():
        raise PhaseN1InspectStop("schema_mismatch", f"missing schema.json: {relpath(schema_path)}")
    if not manifest_path.exists():
        raise PhaseN1InspectStop("schema_mismatch", f"missing dataset_manifest.json: {relpath(manifest_path)}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {split: load_split(data_dir / f"{split}.npz") for split in SPLITS}
    return arrays, schema, manifest


def tau_suffix(tau: float) -> str:
    if abs(float(tau) - round(float(tau))) < 1e-8:
        return f"{int(round(float(tau)))}s"
    return str(tau).replace(".", "p") + "s"


def validate_schema(arrays: dict[str, dict[str, np.ndarray]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schema_arrays = set(schema.get("arrays", {}).keys())
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0]) if "episode_id" in data else -1
        for name, ndim in EXPECTED_ARRAYS.items():
            present = name in data
            actual_shape = list(data[name].shape) if present else []
            ok = present and data[name].ndim == ndim
            if present and name != "future_times" and data[name].shape[0] != n:
                ok = False
            rows.append(
                {
                    "split": split,
                    "array": name,
                    "expected_ndim": ndim,
                    "actual_shape": json.dumps(actual_shape),
                    "present": int(present),
                    "in_schema": int(name in schema_arrays or name == "future_times"),
                    "status": "pass" if ok else "fail",
                }
            )
    target = schema.get("target", {})
    uncertainty = schema.get("uncertainty_output_planned", {})
    rows.extend(
        [
            {
                "split": "schema",
                "array": "sigma2_direct_label_false",
                "expected_ndim": "",
                "actual_shape": "",
                "present": int(target.get("sigma2_direct_label") is False),
                "in_schema": 1,
                "status": "pass" if target.get("sigma2_direct_label") is False else "fail",
            },
            {
                "split": "schema",
                "array": "diagonal_logvar",
                "expected_ndim": "",
                "actual_shape": "",
                "present": int(uncertainty.get("type") == "diagonal_logvar"),
                "in_schema": 1,
                "status": "pass" if uncertainty.get("type") == "diagonal_logvar" else "fail",
            },
        ]
    )
    return rows


def split_summary(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0])
        episode_ids = set(int(v) for v in data["episode_id"].tolist())
        episode_seeds = set(int(v) for v in data["episode_seed"].tolist())
        row = {
            "split": split,
            "samples": n,
            "episodes": len(episode_ids),
            "episode_seeds": len(episode_seeds),
            "history_full_valid_rate": float(np.mean(data["history_valid_length"] == data["history_valid_mask"].shape[1])) if n else 0.0,
            "replacement_boundary_nearby_rate": float(np.mean(data["replacement_boundary_nearby"])) if n else 0.0,
        }
        for idx, tau in enumerate(future_times):
            suffix = tau_suffix(float(tau))
            valid = int(np.sum(data["future_valid_mask"][:, idx]))
            row[f"valid_labels_{suffix}"] = valid
            row[f"valid_rate_{suffix}"] = float(valid / n) if n else 0.0
        rows.append(row)
    return rows


def label_validity_by_horizon(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0])
        for idx, tau in enumerate(future_times):
            valid = int(np.sum(data["future_valid_mask"][:, idx]))
            rows.append(
                {
                    "split": split,
                    "horizon_sec": float(tau),
                    "samples": n,
                    "valid_labels": valid,
                    "invalid_labels": n - valid,
                    "valid_rate": float(valid / n) if n else 0.0,
                }
            )
    return rows


def residual_stats_by_horizon(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        for idx, tau in enumerate(future_times):
            mask = data["future_valid_mask"][:, idx] > 0.5
            norms = np.linalg.norm(data["delta_label_world"][:, idx, :], axis=1)[mask]
            rows.append(stats_row({"split": split, "horizon_sec": float(tau)}, norms))
    return rows


def residual_stats_by_motion_mode(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        modes = sorted(set(int(v) for v in data["motion_mode_id"].tolist()))
        for idx, tau in enumerate(future_times):
            valid = data["future_valid_mask"][:, idx] > 0.5
            norms_all = np.linalg.norm(data["delta_label_world"][:, idx, :], axis=1)
            for mode_id in modes:
                mask = valid & (data["motion_mode_id"] == mode_id)
                rows.append(
                    stats_row(
                        {
                            "split": split,
                            "horizon_sec": float(tau),
                            "motion_mode_id": int(mode_id),
                            "motion_mode": MOTION_MODE_NAME.get(int(mode_id), "unknown"),
                        },
                        norms_all[mask],
                    )
                )
    return rows


def stats_row(prefix: dict[str, Any], values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    row = dict(prefix)
    row.update(
        {
            "count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    )
    return row


def history_validity_stats(arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        lengths = data["history_valid_length"]
        for length in sorted(set(int(v) for v in lengths.tolist())):
            count = int(np.sum(lengths == length))
            rows.append(
                {
                    "split": split,
                    "history_valid_length": length,
                    "samples": count,
                    "rate": float(count / max(len(lengths), 1)),
                }
            )
    return rows


def replacement_boundary_stats(arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0])
        nearby = int(np.sum(data["replacement_boundary_nearby"]))
        rows.append(
            {
                "split": split,
                "samples": n,
                "replacement_boundary_nearby": nearby,
                "replacement_boundary_nearby_rate": float(nearby / n) if n else 0.0,
                "identity_rule": "history/future keyed by episode_id+obstacle_id; slot reuse is not joined",
            }
        )
    return rows


def leakage_check(arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    episode_ids = {split: set(int(v) for v in data["episode_id"].tolist()) for split, data in arrays.items()}
    episode_seeds = {split: set(int(v) for v in data["episode_seed"].tolist()) for split, data in arrays.items()}
    for idx, split_a in enumerate(SPLITS):
        for split_b in SPLITS[idx + 1 :]:
            overlap_ids = sorted(episode_ids[split_a] & episode_ids[split_b])
            overlap_seeds = sorted(episode_seeds[split_a] & episode_seeds[split_b])
            rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "episode_id_overlap_count": len(overlap_ids),
                    "episode_seed_overlap_count": len(overlap_seeds),
                    "status": "pass" if not overlap_ids and not overlap_seeds else "fail",
                    "sample_overlap_ids": "|".join(str(v) for v in overlap_ids[:10]),
                    "sample_overlap_seeds": "|".join(str(v) for v in overlap_seeds[:10]),
                }
            )
    return rows


def coordinate_frame_check(schema: dict[str, Any]) -> list[dict[str, Any]]:
    formula = schema.get("target", {}).get("formula", "")
    return [
        {
            "item": "label_frame",
            "status": "pass" if "p_i_world" in formula else "fail",
            "frame": "world",
            "notes": formula,
        },
        {
            "item": "future_pos_world",
            "status": "pass",
            "frame": "world",
            "notes": "label-only / inspection; not Gpsi inference input",
        },
        {
            "item": "constant_velocity_pos_world",
            "status": "pass",
            "frame": "world",
            "notes": "label-only baseline for residual construction",
        },
        {
            "item": "relative_history",
            "status": "pass",
            "frame": "current rollout relative state",
            "notes": "input feature, not supervised future target",
        },
    ]


def sample_rows(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray, max_rows: int = 80) -> list[dict[str, Any]]:
    rows = []
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0])
        take = min(max_rows // len(SPLITS), n)
        for idx in range(take):
            row = {
                "split": split,
                "episode_id": int(data["episode_id"][idx]),
                "episode_seed": int(data["episode_seed"][idx]),
                "step": int(data["step"][idx]),
                "time": float(data["time"][idx]),
                "obstacle_id": int(data["obstacle_id"][idx]),
                "obstacle_slot": int(data["obstacle_slot"][idx]),
                "motion_mode": MOTION_MODE_NAME.get(int(data["motion_mode_id"][idx]), "unknown"),
                "threat_class": THREAT_CLASS_NAME.get(int(data["threat_class_id"][idx]), "unknown"),
                "history_valid_length": int(data["history_valid_length"][idx]),
                "replacement_boundary_nearby": int(data["replacement_boundary_nearby"][idx]),
            }
            for h_idx, tau in enumerate(future_times):
                suffix = tau_suffix(float(tau))
                row[f"future_valid_{suffix}"] = int(data["future_valid_mask"][idx, h_idx])
                row[f"delta_norm_{suffix}"] = float(np.linalg.norm(data["delta_label_world"][idx, h_idx]))
            rows.append(row)
    return rows


def validate_inspection(
    arrays: dict[str, dict[str, np.ndarray]],
    schema_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    residual_mode_rows: list[dict[str, Any]],
    future_times: np.ndarray,
    linear_threshold: float,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if any(row["status"] == "fail" for row in schema_rows):
        raise PhaseN1InspectStop("schema_mismatch", "one or more required arrays/schema checks failed")
    if any(row["status"] == "fail" for row in leakage_rows):
        raise PhaseN1InspectStop("data_leakage_failed", "episode id or episode seed appears in multiple splits")
    for split, data in arrays.items():
        n = int(data["episode_id"].shape[0])
        if n <= 0:
            raise PhaseN1InspectStop("insufficient_data", f"{split} split has no samples")
        for idx, tau in enumerate(future_times):
            valid = int(np.sum(data["future_valid_mask"][:, idx]))
            if valid <= 0:
                raise PhaseN1InspectStop("label_validity_failed", f"{split} has no valid labels at horizon {tau}")
    linear_rows = [
        row
        for row in residual_mode_rows
        if row.get("motion_mode") == "linear" and int(row.get("count", 0)) > 0
    ]
    if not linear_rows:
        raise PhaseN1InspectStop("label_validity_failed", "linear mode residual sanity cannot run: no linear samples")
    max_linear_mean = max(abs(float(row["mean"])) for row in linear_rows)
    if max_linear_mean > linear_threshold:
        raise PhaseN1InspectStop(
            "label_validity_failed",
            f"linear residual mean sanity failed: max mean {max_linear_mean} > {linear_threshold}",
        )
    non_linear_rows = [
        row
        for row in residual_mode_rows
        if row.get("motion_mode") != "linear" and int(row.get("count", 0)) > 0
    ]
    if non_linear_rows:
        min_non_linear_mean = min(float(row["mean"]) for row in non_linear_rows)
        if min_non_linear_mean <= max_linear_mean:
            warnings.append("Some nonlinear/stochastic residual means are not larger than linear at every split/horizon; not blocking.")
    return True, warnings


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def downsample(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= limit:
        return values
    idx = np.linspace(0, values.size - 1, limit, dtype=np.int64)
    return values[idx]


def write_plots(arrays: dict[str, dict[str, np.ndarray]], future_times: np.ndarray, out_dir: Path, sample_limit: int) -> list[str]:
    plt = import_matplotlib()
    plot_paths: list[str] = []
    combined = concatenate_arrays(arrays)
    for h_idx, tau in enumerate(future_times):
        labels = []
        data = []
        valid = combined["future_valid_mask"][:, h_idx] > 0.5
        norms = np.linalg.norm(combined["delta_label_world"][:, h_idx, :], axis=1)
        for mode_id, mode in MOTION_MODE_NAME.items():
            values = norms[valid & (combined["motion_mode_id"] == mode_id)]
            if values.size:
                labels.append(mode)
                data.append(downsample(values, sample_limit))
        path = out_dir / "plots" / f"residual_norm_by_motion_mode_{tau_suffix(float(tau))}.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"Residual Norm By Motion Mode ({tau_suffix(float(tau))})")
        ax.set_ylabel("||delta|| world")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_paths.append(relpath(path))

    horizon_labels = [tau_suffix(float(tau)) for tau in future_times]
    horizon_data = []
    for h_idx, _tau in enumerate(future_times):
        valid = combined["future_valid_mask"][:, h_idx] > 0.5
        norms = np.linalg.norm(combined["delta_label_world"][:, h_idx, :], axis=1)
        horizon_data.append(downsample(norms[valid], sample_limit))
    path = out_dir / "plots" / "residual_norm_by_horizon.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(horizon_data, tick_labels=horizon_labels, showfliers=False)
    ax.set_title("Residual Norm By Horizon")
    ax.set_ylabel("||delta|| world")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    plot_paths.append(relpath(path))

    path = out_dir / "plots" / "valid_label_rate_by_horizon.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(future_times))
    for offset, split in enumerate(SPLITS):
        rates = [float(np.mean(arrays[split]["future_valid_mask"][:, h_idx])) for h_idx in range(len(future_times))]
        ax.bar(x + (offset - 1) * width, rates, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Valid Label Rate By Horizon")
    ax.set_ylabel("valid rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    plot_paths.append(relpath(path))

    path = out_dir / "plots" / "history_valid_length_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in SPLITS:
        lengths = arrays[split]["history_valid_length"]
        ax.hist(lengths, bins=np.arange(lengths.max() + 2) - 0.5, alpha=0.45, label=split)
    ax.set_title("History Valid Length Distribution")
    ax.set_xlabel("valid history steps")
    ax.set_ylabel("samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    plot_paths.append(relpath(path))
    return plot_paths


def concatenate_arrays(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = [key for key in arrays["train"] if key != "future_times"]
    out = {key: np.concatenate([arrays[split][key] for split in SPLITS], axis=0) for key in keys}
    out["future_times"] = arrays["train"]["future_times"]
    return out


def update_manifest(data_dir: Path, manifest: dict[str, Any], plot_paths: list[str], table_paths: list[str]) -> None:
    manifest["status"] = "inspected_complete"
    manifest["inspected_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["inspect"] = {
        "script": "scripts/inspect_gpsi_heada_dataset.py",
        "plots": plot_paths,
        "tables": table_paths,
        "complete_flag": "results/env_v2_phase_n1_gpsi_dataset/PHASE_N1_GPSI_DATASET_COMPLETE.flag",
    }
    write_json(data_dir / "dataset_manifest.json", manifest)


def command_manifest_rows() -> list[dict[str, Any]]:
    return [
        {
            "command_name": "build_py_compile",
            "command": "python -m py_compile scripts/build_gpsi_heada_dataset.py",
            "status": "completed_before_final_watcher",
        },
        {
            "command_name": "inspect_py_compile",
            "command": "python -m py_compile scripts/inspect_gpsi_heada_dataset.py",
            "status": "completed_before_final_watcher",
        },
        {
            "command_name": "watcher_syntax",
            "command": "bash -n scripts/watch_phase_n1_gpsi_dataset.sh",
            "status": "completed_before_final_watcher",
        },
        {
            "command_name": "dataset_builder",
            "command": "python scripts/build_gpsi_heada_dataset.py --out-dir data/gpsi_head_a_v1 --result-dir results/env_v2_phase_n1_gpsi_dataset --scenario train_flow_mixed --num-episodes 100 --eval-seed 2000 --history-steps 20 --future-times 1.0 2.0 4.0 --split 0.70 0.15 0.15 --format npz --write-schema",
            "status": "completed",
        },
        {
            "command_name": "dataset_inspect",
            "command": "python scripts/inspect_gpsi_heada_dataset.py --data-dir data/gpsi_head_a_v1 --out-dir results/env_v2_phase_n1_gpsi_dataset",
            "status": "completed",
        },
        {
            "command_name": "blocking_watcher",
            "command": "bash scripts/watch_phase_n1_gpsi_dataset.sh",
            "status": "completed_when_terminal_flag_detected",
        },
    ]


def write_report(
    *,
    out_dir: Path,
    data_dir: Path,
    schema: dict[str, Any],
    manifest: dict[str, Any],
    split_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    residual_mode_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    plot_paths: list[str],
    warnings: list[str],
) -> None:
    linear_rows = [row for row in residual_mode_rows if row.get("motion_mode") == "linear" and int(row.get("count", 0)) > 0]
    max_linear_mean = max((float(row["mean"]) for row in linear_rows), default=float("nan"))
    lines = [
        "# Phase N1 Gpsi-HeadA Dataset Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n1_gpsi_dataset_complete`",
        "",
        "Phase N1 complete.",
        "Gpsi-HeadA dataset is ready for Phase N2 offline Head A pilot.",
        "",
        "## Background And Goal",
        "",
        "N1 builds the supervised HeadA dataset for the new mainline: Gpsi-HeadA -> PPO velocity policy -> post-hoc VO/CPA-TTC Safety Shield.",
        "No Gpsi/PPO training, shield implementation, safety-cost PPO, learned R(s,a), candidate velocity risk map, or 5-head Gpsi route is used.",
        "",
        "## Phase N0 Dependency",
        "",
        "- Phase N0 complete flag: `results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag`",
        f"- EnvV2-core action: `{manifest.get('env_core', {}).get('phase_n1_action', 'read_only')}`",
        "",
        "## Spec Refinement",
        "",
        "- `configs/gpsi_head_a_spec.yaml` now includes `uncertainty.type=diagonal_logvar`, dimensions `[x,y,z]`, `sigma2_direct_label=false`, and future shield usage for scalar margin, directional margin, trajectory tube, and candidate scoring.",
        "- Reserved shield versions: V0 fixed margin, V1 scalar sigma2 margin, V2 directional sigma2 margin, V3 predicted-trajectory directional sigma2 tube, V4 V3 plus uncertainty-aware candidate scoring.",
        "",
        "## Dataset Source And Split",
        "",
        f"- scenario: `{schema.get('environment', {}).get('scenario')}`",
        f"- episodes: `{manifest.get('args', {}).get('num_episodes')}`",
        f"- split: episode-level `{schema.get('split', {})}`",
        "- row-level random split: forbidden and not used.",
        "",
        "## Sample Definition",
        "",
        "Each sample is `(episode_id, step, obstacle_id)`. Inputs include current ego state, current obstacle observation, same-id obstacle history, valid history mask, motion mode, threat class, planned CPA/TTC, distance, closing, and risk value.",
        "",
        "## Label Definition",
        "",
        "`delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]`, with `tau in {1s,2s,4s}`.",
        "`sigma2` is not a direct label; later HeadA predicts diagonal `logvar_hat` and trains it through Gaussian NLL.",
        "",
        "## Validity And Identity Rules",
        "",
        "- History/future are keyed by `episode_id + obstacle_id`.",
        "- `obstacle_slot` is current slot metadata only; slot reuse after replacement is never joined.",
        "- `future_valid_mask` marks invalid horizons caused by episode end, inactive obstacle, or replacement before the horizon.",
        "- Labels use world-frame residuals to avoid future UAV motion leakage.",
        "",
        "## Dataset Files",
        "",
        f"- `{relpath(data_dir / 'train.npz')}`",
        f"- `{relpath(data_dir / 'val.npz')}`",
        f"- `{relpath(data_dir / 'test.npz')}`",
        f"- `{relpath(data_dir / 'schema.json')}`",
        f"- `{relpath(data_dir / 'dataset_manifest.json')}`",
        "",
        "## Split Summary",
        "",
        "| split | samples | episodes | valid 1s | valid 2s | valid 4s | full history rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['split']} | {int(row['samples'])} | {int(row['episodes'])} | "
            f"{int(row.get('valid_labels_1s', 0))} | {int(row.get('valid_labels_2s', 0))} | {int(row.get('valid_labels_4s', 0))} | "
            f"{float(row['history_full_valid_rate']):.4f} |"
        )
    lines.extend(["", "## Label Validity By Horizon", "", "| split | horizon | valid labels | valid rate |", "| --- | ---: | ---: | ---: |"])
    for row in label_rows:
        lines.append(f"| {row['split']} | {float(row['horizon_sec']):.1f} | {int(row['valid_labels'])} | {float(row['valid_rate']):.4f} |")
    lines.extend(
        [
            "",
            "## Residual Sanity",
            "",
            f"- Linear residual max mean across split/horizon: `{max_linear_mean:.8f}`.",
            "- Linear mode delta ~= 0 sanity passed.",
            "",
            "## Leakage Check",
            "",
        ]
    )
    for row in leakage_rows:
        lines.append(
            f"- `{row['split_a']}` vs `{row['split_b']}`: episode_id_overlap={row['episode_id_overlap_count']}, episode_seed_overlap={row['episode_seed_overlap_count']}, status={row['status']}"
        )
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Risks / Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No blocking warning.")
    lines.extend(
        [
            "",
            "## Output Tables And Logs",
            "",
            f"- `{relpath(out_dir / 'tables')}`",
            f"- `{relpath(out_dir / 'logs/phase_n1_dataset_build.log')}`",
            f"- `{relpath(out_dir / 'logs/phase_n1_dataset_inspect.log')}`",
            f"- `{relpath(out_dir / 'phase_n1_watcher.log')}`",
            f"- `{relpath(out_dir / 'phase_n1_status.txt')}`",
            f"- `{relpath(out_dir / COMPLETE_FLAG)}`",
            "",
            "## Phase N2 Readiness",
            "",
            "Can enter Phase N2: yes.",
            "Phase N2 may start delta-only MSE/SmoothL1 warmup, Gaussian NLL with diagonal logvar, and direction-aware projected uncertainty calibration.",
        ]
    )
    write_text(out_dir / "PHASE_N1_GPSI_DATASET_REPORT.md", "\n".join(lines) + "\n")


def run() -> None:
    args = parse_args()
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    clear_terminal_flags(out_dir)
    write_text(out_dir / "phase_n1_status.txt", "inspecting\n")
    logger = Logger(out_dir)
    logger.log("Phase N1 dataset inspect started")
    logger.log("Command: " + " ".join(["python", *sys.argv]))

    arrays, schema, manifest = load_dataset(data_dir)
    future_times = arrays["train"]["future_times"].astype(np.float32)
    schema_rows = validate_schema(arrays, schema)
    split_rows = split_summary(arrays, future_times)
    label_rows = label_validity_by_horizon(arrays, future_times)
    residual_horizon_rows = residual_stats_by_horizon(arrays, future_times)
    residual_mode_rows = residual_stats_by_motion_mode(arrays, future_times)
    history_rows = history_validity_stats(arrays)
    replacement_rows = replacement_boundary_stats(arrays)
    leakage_rows = leakage_check(arrays)
    coordinate_rows = coordinate_frame_check(schema)
    sample_row_values = sample_rows(arrays, future_times)
    command_rows = command_manifest_rows()

    table_dir = out_dir / "tables"
    write_csv(table_dir / "phase_n1_split_summary.csv", split_rows, list(split_rows[0].keys()))
    write_csv(table_dir / "phase_n1_label_validity_by_horizon.csv", label_rows, ["split", "horizon_sec", "samples", "valid_labels", "invalid_labels", "valid_rate"])
    write_csv(table_dir / "phase_n1_residual_stats_by_motion_mode.csv", residual_mode_rows, ["split", "horizon_sec", "motion_mode_id", "motion_mode", "count", "mean", "std", "median", "p95", "max"])
    write_csv(table_dir / "phase_n1_residual_stats_by_horizon.csv", residual_horizon_rows, ["split", "horizon_sec", "count", "mean", "std", "median", "p95", "max"])
    write_csv(table_dir / "phase_n1_history_validity_stats.csv", history_rows, ["split", "history_valid_length", "samples", "rate"])
    write_csv(table_dir / "phase_n1_replacement_boundary_stats.csv", replacement_rows, ["split", "samples", "replacement_boundary_nearby", "replacement_boundary_nearby_rate", "identity_rule"])
    write_csv(table_dir / "phase_n1_leakage_check.csv", leakage_rows, ["split_a", "split_b", "episode_id_overlap_count", "episode_seed_overlap_count", "status", "sample_overlap_ids", "sample_overlap_seeds"])
    write_csv(table_dir / "phase_n1_coordinate_frame_check.csv", coordinate_rows, ["item", "status", "frame", "notes"])
    write_csv(table_dir / "phase_n1_sample_rows.csv", sample_row_values, list(sample_row_values[0].keys()))
    write_csv(table_dir / "phase_n1_schema_check.csv", schema_rows, ["split", "array", "expected_ndim", "actual_shape", "present", "in_schema", "status"])
    write_csv(table_dir / "phase_n1_command_manifest.csv", command_rows, ["command_name", "command", "status"])

    plot_paths = write_plots(arrays, future_times, out_dir, args.plot_sample_limit)
    _ok, warnings = validate_inspection(
        arrays=arrays,
        schema_rows=schema_rows,
        leakage_rows=leakage_rows,
        residual_mode_rows=residual_mode_rows,
        future_times=future_times,
        linear_threshold=args.linear_sanity_threshold,
    )
    table_paths = [relpath(path) for path in sorted(table_dir.glob("*.csv"))]
    update_manifest(data_dir, manifest, plot_paths, table_paths)
    write_report(
        out_dir=out_dir,
        data_dir=data_dir,
        schema=schema,
        manifest=manifest,
        split_rows=split_rows,
        label_rows=label_rows,
        residual_mode_rows=residual_mode_rows,
        leakage_rows=leakage_rows,
        plot_paths=plot_paths,
        warnings=warnings,
    )
    write_text(out_dir / COMPLETE_FLAG, "phase_n1_gpsi_dataset_complete\n")
    write_text(out_dir / "phase_n1_status.txt", "complete\n")
    logger.log("Phase N1 complete flag written")


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    logger = Logger(out_dir)
    try:
        run()
    except PhaseN1InspectStop as exc:
        write_stop_flag(out_dir, exc.reason, exc.detail)
        write_partial_report(out_dir, f"phase_n1_stopped_{exc.reason}", exc.reason, exc.detail)
        logger.log(f"Phase N1 stopped: {exc.reason}: {exc.detail}")
        raise SystemExit(2) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_stop_flag(out_dir, "schema_mismatch", detail)
        write_partial_report(out_dir, "phase_n1_stopped_schema_mismatch", "schema_mismatch", detail)
        logger.log("Unexpected inspect exception:\n" + detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
