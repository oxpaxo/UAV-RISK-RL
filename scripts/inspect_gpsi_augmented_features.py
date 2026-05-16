from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audit_gpsi_online_wrapper import collect_rollouts
from phase_n3_5_audit_common import (
    ROOT,
    PhaseN35Stop,
    block_stats_row,
    check_prerequisites,
    command_manifest_row,
    ensure_dirs,
    finite_stats,
    save_bar,
    save_histogram,
    write_csv,
    write_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3.5 PPO augmented feature block-scale audit.")
    parser.add_argument("--checkpoint", default="work_dirs/gpsi_heada_v1_nll/best.pth")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3_5_gpsi_wrapper_audit")
    parser.add_argument("--scenarios", nargs="+", default=["eval_flow_id", "eval_flow_high_speed", "eval_flow_mixed_ood"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=160)
    parser.add_argument("--policy", default="random_or_straight_line", choices=["random_or_straight_line", "straight_line", "hold_position", "random"])
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--repaired-std-threshold", type=float, default=1.0e-5)
    parser.add_argument("--repaired-std-floor", type=float, default=1.0)
    return parser.parse_args()


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_rows(result: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in [
        "obs_i_12",
        "z_i_64",
        "delta_hat_9_before_scale",
        "delta_hat_9_after_scale",
        "logvar_hat_9_clamped",
        "full_aug_obs",
    ]:
        arr = np.asarray(result["aug_blocks"].get(block, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        if arr.size == 0:
            continue
        row = block_stats_row({"repair_stage": stage, "active_only": 1, "samples": int(arr.shape[0])}, block, arr)
        rows.append(row)
    return rows


def validate_feature_scale(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_block = {row["block"]: row for row in rows if row["repair_stage"] == "after_fix"}
    status_rows = []
    for block in ["obs_i_12", "z_i_64", "delta_hat_9_after_scale", "logvar_hat_9_clamped", "full_aug_obs"]:
        row = by_block.get(block)
        if row is None:
            status_rows.append({"metric": f"{block}_present", "value": 0, "status": "fail"})
            continue
        l2_p95 = float(row["l2_norm_p95"])
        max_abs = max(abs(float(row["min"])), abs(float(row["max"])))
        status = "pass"
        if not np.isfinite(l2_p95) or not np.isfinite(max_abs):
            status = "fail"
        elif block == "full_aug_obs" and max_abs > 100.0:
            status = "fail"
        elif block != "full_aug_obs" and max_abs > 50.0:
            status = "fail"
        status_rows.append({"metric": f"{block}_l2_p95", "value": l2_p95, "status": status})
        status_rows.append({"metric": f"{block}_max_abs", "value": max_abs, "status": status})
    z_row = by_block.get("z_i_64")
    obs_row = by_block.get("obs_i_12")
    if z_row and obs_row:
        ratio = float(z_row["l2_norm_p95"]) / max(float(obs_row["l2_norm_p95"]), 1.0e-9)
        status_rows.append({"metric": "z_p95_to_obs_p95_l2_ratio", "value": ratio, "status": "warn" if ratio > 10.0 else "pass"})
    write_csv(out_dir / "tables/phase_n3_5_feature_scale_status.csv", status_rows)
    failed = [row for row in status_rows if row["status"] == "fail"]
    if failed:
        raise PhaseN35Stop("feature_scale_invalid", f"after-fix augmented feature scale invalid: {failed}")


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    checkpoint = ROOT / args.checkpoint
    ensure_dirs(out_dir)
    try:
        check_prerequisites(out_dir, checkpoint)
        stages = [
            ("before_fix", 0.0, 1.0),
            ("after_fix", float(args.repaired_std_threshold), float(args.repaired_std_floor)),
        ]
        rows: list[dict[str, Any]] = []
        collected: dict[str, dict[str, Any]] = {}
        for stage, threshold, floor in stages:
            result = collect_rollouts(
                checkpoint=checkpoint,
                scenarios=list(args.scenarios),
                num_episodes=int(args.num_episodes),
                max_steps=int(args.max_steps_per_episode),
                policy=str(args.policy),
                seed=int(args.seed),
                device=str(args.device),
                threshold=threshold,
                floor=floor,
                stage=stage,
            )
            collected[stage] = result
            rows.extend(feature_rows(result, stage))
        fieldnames = [
            "repair_stage",
            "active_only",
            "samples",
            "block",
            "count",
            "nan_count",
            "inf_count",
            "mean",
            "std",
            "l2_norm_mean",
            "l2_norm_median",
            "l2_norm_p95",
            "min",
            "max",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
        ]
        write_csv(out_dir / "tables/phase_n3_5_aug_feature_block_stats.csv", rows, fieldnames)
        if args.write_plots:
            after_rows = [row for row in rows if row["repair_stage"] == "after_fix"]
            save_bar(
                out_dir / "plots/aug_feature_block_scale.png",
                after_rows,
                "block",
                "l2_norm_p95",
                "After-Fix Augmented Feature Block Scale",
                "p95 L2 norm",
            )
            dist_rows = []
            for block in ["obs_i_12", "z_i_64", "delta_hat_9_after_scale", "logvar_hat_9_clamped"]:
                arr = np.asarray(collected["after_fix"]["aug_blocks"].get(block, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
                if arr.size:
                    norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
                    dist_rows.append((block, norms))
            save_histogram(
                out_dir / "plots/input_distribution_shift.png",
                dist_rows,
                "After-Fix Augmented Block Norms",
                "block L2 norm",
            )
        recommendation = [
            {
                "decision_item": "n3_original_validity",
                "recommendation": "invalid_if_before_fix_online_scale_matches_n3_anomaly",
                "reason": "before_fix uses legacy near-zero ego velocity normalization and reproduces large Gpsi output scales.",
            },
            {
                "decision_item": "n3_rerun",
                "recommendation": "required",
                "reason": "N3 PPO checkpoints were trained/evaluated with legacy wrapper normalization.",
            },
            {
                "decision_item": "n4_entry",
                "recommendation": "blocked",
                "reason": "Must rerun N3 or N3-lite with repaired feature pipeline before shield comparison.",
            },
            {
                "decision_item": "feature_normalization",
                "recommendation": "rerun_with_repaired_gpsi_input_norm_and_consider_z_block_normalization_or_ablation",
                "reason": "z_i remains unnormalized relative to base obs; this is an engineering risk even after output-scale repair.",
            },
        ]
        write_csv(out_dir / "tables/phase_n3_5_rerun_recommendation.csv", recommendation)
        manifest_path = out_dir / "tables/phase_n3_5_command_manifest.csv"
        manifest = read_existing_manifest(manifest_path)
        manifest.append(command_manifest_row("augmented_feature_scale", " ".join(sys.argv), "completed"))
        write_csv(manifest_path, manifest, ["command_name", "command", "status"])
        (out_dir / "logs/phase_n3_5_feature_scale.log").write_text("augmented feature-scale audit completed\n", encoding="utf-8")
        validate_feature_scale(out_dir, rows)
        return 0
    except PhaseN35Stop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
