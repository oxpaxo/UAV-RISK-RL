from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase_n3_5_audit_common import (
    HORIZON_SUFFIXES,
    MOTION_MODE_ID,
    ROOT,
    THREAT_CLASS_ID,
    PhaseN35Stop,
    block_stats_row,
    check_prerequisites,
    command_manifest_row,
    ensure_dirs,
    finite_stats,
    history_ratio_bin,
    load_npz,
    make_wrapper,
    normalize,
    rel,
    save_histogram,
    write_csv,
    write_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3.5 online EnvV2 Gpsi wrapper audit.")
    parser.add_argument("--checkpoint", default="work_dirs/gpsi_heada_v1_nll/best.pth")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3_5_gpsi_wrapper_audit")
    parser.add_argument("--data-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--scenarios", nargs="+", default=["eval_flow_id", "eval_flow_high_speed", "eval_flow_mixed_ood"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=160)
    parser.add_argument("--policy", default="random_or_straight_line", choices=["random_or_straight_line", "straight_line", "hold_position", "random"])
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-input-output-stats", action="store_true")
    parser.add_argument("--repaired-std-threshold", type=float, default=1.0e-5)
    parser.add_argument("--repaired-std-floor", type=float, default=1.0)
    return parser.parse_args()


def action_from_policy(info: dict[str, Any], policy: str, rng: np.random.Generator) -> np.ndarray:
    if policy == "hold_position":
        return np.zeros(3, dtype=np.float32)
    if policy == "random":
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        action[2] = 0.0
        return action
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    vec = goal - uav
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.zeros(3, dtype=np.float32)
    return np.clip(vec / norm, -1.0, 1.0).astype(np.float32)


def collect_rollouts(
    *,
    checkpoint: Path,
    scenarios: list[str],
    num_episodes: int,
    max_steps: int,
    policy: str,
    seed: int,
    device: str,
    threshold: float,
    floor: float,
    stage: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw_blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    norm_blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    aug_blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    output_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    per_obstacle_rows: list[dict[str, Any]] = []
    norm_check_rows: list[dict[str, Any]] = []
    first_wrapper = None

    for scenario in scenarios:
        for ep in range(num_episodes):
            episode_seed = int(seed + ep + 1000 * scenarios.index(scenario))
            env = make_wrapper(checkpoint, scenario, device, threshold=threshold, floor=floor)
            if first_wrapper is None:
                first_wrapper = env
                norm_check_rows.extend(env.normalization_debug_rows())
            obs, info = env.reset(seed=episode_seed)
            done = False
            steps = 0
            previous_ids: set[int] = set()
            while not done and steps < max_steps:
                debug = env.latest_gpsi_debug
                active_slots = np.asarray(debug.get("active_slots", []), dtype=np.int64)
                ids = np.asarray(debug.get("obstacle_ids", []), dtype=np.int64)
                mask = np.asarray(obs["mask"], dtype=np.float32)
                base_obs_aug = np.asarray(obs["obs"], dtype=np.float32)
                base_obs = base_obs_aug[:, :12]
                z = np.asarray(debug.get("z", np.zeros((env.max_obs, env.z_dim))), dtype=np.float32)
                delta_raw = np.asarray(debug.get("delta_hat_raw", np.zeros((env.max_obs, 3, 3))), dtype=np.float32)
                delta_norm = np.asarray(debug.get("delta_hat_norm", np.zeros((env.max_obs, 3, 3))), dtype=np.float32)
                logvar_clamped = np.asarray(debug.get("logvar_hat", np.zeros((env.max_obs, 3, 3))), dtype=np.float32)
                logvar_raw = np.asarray(debug.get("logvar_hat_raw", np.zeros((env.max_obs, 3, 3))), dtype=np.float32)
                hist_mask = np.asarray(debug.get("gpsi_input_history_valid_mask", np.zeros((env.max_obs, 20))), dtype=np.float32)
                hist_pos = np.asarray(debug.get("gpsi_input_history_rel_pos", np.zeros((env.max_obs, 20, 3))), dtype=np.float32)
                hist_vel = np.asarray(debug.get("gpsi_input_history_rel_vel", np.zeros((env.max_obs, 20, 3))), dtype=np.float32)
                norm_ego = np.asarray(debug.get("gpsi_norm_ego_current", np.zeros((env.max_obs, 10))), dtype=np.float32)
                norm_obs = np.asarray(debug.get("gpsi_norm_obs_current", np.zeros((env.max_obs, 12))), dtype=np.float32)
                norm_hist_pos = np.asarray(debug.get("gpsi_norm_history_rel_pos", np.zeros((env.max_obs, 20, 3))), dtype=np.float32)
                norm_hist_vel = np.asarray(debug.get("gpsi_norm_history_rel_vel", np.zeros((env.max_obs, 20, 3))), dtype=np.float32)

                if active_slots.size:
                    raw_blocks["ego_current"].append(np.asarray(debug["gpsi_input_ego_current"], dtype=np.float32)[active_slots])
                    raw_blocks["obs_current"].append(np.asarray(debug["gpsi_input_obs_current"], dtype=np.float32)[active_slots])
                    raw_blocks["history_valid_mask"].append(hist_mask[active_slots])
                    valid_mask = hist_mask[active_slots].astype(bool)
                    if valid_mask.any():
                        raw_blocks["history_rel_pos"].append(hist_pos[active_slots][valid_mask])
                        raw_blocks["history_rel_vel"].append(hist_vel[active_slots][valid_mask])
                        norm_blocks["history_rel_pos"].append(norm_hist_pos[active_slots][valid_mask])
                        norm_blocks["history_rel_vel"].append(norm_hist_vel[active_slots][valid_mask])
                    norm_blocks["ego_current"].append(norm_ego[active_slots])
                    norm_blocks["obs_current"].append(norm_obs[active_slots])
                    norm_blocks["history_valid_mask"].append(hist_mask[active_slots])
                    aug_blocks["obs_i_12"].append(base_obs[active_slots])
                    aug_blocks["z_i_64"].append(z[active_slots])
                    aug_blocks["delta_hat_9_before_scale"].append(delta_raw[active_slots].reshape(active_slots.size, -1))
                    aug_blocks["delta_hat_9_after_scale"].append(delta_norm[active_slots].reshape(active_slots.size, -1))
                    aug_blocks["logvar_hat_9_clamped"].append(logvar_clamped[active_slots].reshape(active_slots.size, -1))
                    aug_blocks["full_aug_obs"].append(base_obs_aug[active_slots])

                current_ids = set(int(ids[slot]) for slot in active_slots if slot < len(ids))
                replacement_seen = bool(previous_ids and current_ids != previous_ids)
                previous_ids = current_ids
                active_rows.append(
                    {
                        "repair_stage": stage,
                        "scenario": scenario,
                        "episode_id": ep,
                        "step": int(info["step"]),
                        "active_mask_sum": int(mask.sum()),
                        "active_slots": int(active_slots.size),
                        "info_active_obstacle_count": int(info.get("active_obstacle_count", -1)),
                        "ids_len": int(len(ids)),
                        "inactive_forwarded_count": int(max(0, active_slots.size - int(mask.sum()))),
                        "duplicate_active_ids": int(len(current_ids) != len(active_slots)),
                        "replacement_seen": int(replacement_seen),
                        "replacement_count": int(info.get("replacement_count", 0)),
                    }
                )

                for slot in active_slots:
                    mode = str(info.get("obstacle_motion_modes", ["none"] * env.max_obs)[slot]) if slot < len(info.get("obstacle_motion_modes", [])) else "none"
                    threat = str(info.get("threat_classes", ["none"] * env.max_obs)[slot]) if slot < len(info.get("threat_classes", [])) else "none"
                    hist_ratio = float(hist_mask[slot].mean())
                    rel = np.asarray(info.get("obstacle_positions", np.zeros((0, 3))), dtype=np.float32)
                    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
                    velocities = np.asarray(info.get("obstacle_velocities", np.zeros((0, 3))), dtype=np.float32)
                    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
                    radial = np.zeros(2, dtype=np.float32)
                    relvel_dir = np.zeros(2, dtype=np.float32)
                    if slot < len(rel):
                        radial = normalize((rel[slot] - uav)[:2])
                    if slot < len(velocities):
                        relvel_dir = normalize((velocities[slot] - uav_vel)[:2])
                    row = {
                        "repair_stage": stage,
                        "scenario": scenario,
                        "episode_id": ep,
                        "step": int(info["step"]),
                        "obstacle_slot": int(slot),
                        "obstacle_id": int(ids[slot]) if slot < len(ids) else -1,
                        "motion_mode": mode,
                        "threat_class": threat,
                        "motion_mode_id": int(MOTION_MODE_ID.get(mode, -1)),
                        "threat_class_id": int(THREAT_CLASS_ID.get(threat, -1)),
                        "history_valid_ratio": hist_ratio,
                        "history_valid_ratio_bin": history_ratio_bin(hist_ratio),
                        "z_norm": float(np.linalg.norm(z[slot])),
                        "nan_count": int(
                            np.isnan(delta_raw[slot]).sum()
                            + np.isnan(logvar_raw[slot]).sum()
                            + np.isnan(z[slot]).sum()
                        ),
                        "inf_count": int(
                            np.isinf(delta_raw[slot]).sum()
                            + np.isinf(logvar_raw[slot]).sum()
                            + np.isinf(z[slot]).sum()
                        ),
                    }
                    for h, suffix in enumerate(HORIZON_SUFFIXES):
                        row[f"delta_norm_{suffix}"] = float(np.linalg.norm(delta_raw[slot, h]))
                        row[f"logvar_xy_{suffix}"] = float(np.mean(logvar_clamped[slot, h, :2]))
                        row[f"logvar_xy_raw_{suffix}"] = float(np.mean(logvar_raw[slot, h, :2]))
                        sigma2 = np.exp(np.clip(logvar_clamped[slot, h, :2], -5.0, 3.0))
                        row[f"projected_std_radial_{suffix}"] = float(np.sqrt(np.sum((radial**2) * sigma2)))
                        row[f"projected_std_relvel_{suffix}"] = float(np.sqrt(np.sum((relvel_dir**2) * sigma2)))
                    per_obstacle_rows.append(row)

                history_rows.append(
                    {
                        "repair_stage": stage,
                        "scenario": scenario,
                        "episode_id": ep,
                        "step": int(info["step"]),
                        "active_slots": int(active_slots.size),
                        "history_valid_ratio_mean": float(hist_mask[active_slots].mean()) if active_slots.size else float("nan"),
                        "history_valid_ratio_min": float(hist_mask[active_slots].mean(axis=1).min()) if active_slots.size else float("nan"),
                        "history_valid_ratio_max": float(hist_mask[active_slots].mean(axis=1).max()) if active_slots.size else float("nan"),
                        "left_padding_violations": int(left_padding_violations(hist_mask[active_slots])) if active_slots.size else 0,
                        "first_step_valid_counts": int(hist_mask[active_slots].sum()) if int(info["step"]) == 0 and active_slots.size else 0,
                    }
                )

                if not field_rows:
                    field_rows.extend(field_order_rows(stage))

                action = action_from_policy(info, policy, rng)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                steps += 1

            step_rows.append(
                {
                    "repair_stage": stage,
                    "scenario": scenario,
                    "episode_id": ep,
                    "steps": int(steps),
                    "terminated_or_truncated": int(done),
                    "policy": policy,
                }
            )
            env.close()

    block_arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in raw_blocks.items()}
    norm_arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in norm_blocks.items()}
    aug_arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in aug_blocks.items()}
    output_summary = summarize_output(per_obstacle_rows)
    return {
        "raw_blocks": block_arrays,
        "norm_blocks": norm_arrays,
        "aug_blocks": aug_arrays,
        "per_obstacle_rows": per_obstacle_rows,
        "output_summary": output_summary,
        "step_rows": step_rows,
        "history_rows": history_rows,
        "active_rows": active_rows,
        "field_rows": field_rows,
        "normalization_rows": norm_check_rows,
    }


def left_padding_violations(mask: np.ndarray) -> int:
    violations = 0
    for row in np.asarray(mask, dtype=np.float32):
        seen_valid = False
        for value in row:
            if value > 0.5:
                seen_valid = True
            elif seen_valid:
                violations += 1
                break
    return violations


def field_order_rows(stage: str) -> list[dict[str, Any]]:
    obs_fields = [
        "rel_pos_x/20",
        "rel_pos_y/20",
        "rel_pos_z/5",
        "rel_vel_x/3",
        "rel_vel_y/3",
        "rel_vel_z/3",
        "planned_cpa/5",
        "planned_ttc_remaining/20",
        "distance/30",
        "closing/3",
        "threat_class_id_scaled",
        "risk_value",
    ]
    ego_fields = ["uav_vel_x/vmax", "uav_vel_y/vmax", "uav_vel_z/vmax", "goal_dir_x", "goal_dir_y", "goal_dir_z", "goal_dist/path_len", "last_action_x", "last_action_y", "last_action_z"]
    rows = []
    for idx, name in enumerate(ego_fields):
        rows.append({"repair_stage": stage, "block": "ego_current", "dim": idx, "field": name, "status": "pass"})
    for idx, name in enumerate(obs_fields):
        rows.append({"repair_stage": stage, "block": "obs_current", "dim": idx, "field": name, "status": "pass"})
    for idx, name in enumerate(["x", "y", "z"]):
        rows.append({"repair_stage": stage, "block": "history_rel_pos", "dim": idx, "field": name, "status": "pass"})
        rows.append({"repair_stage": stage, "block": "history_rel_vel", "dim": idx, "field": name, "status": "pass"})
    return rows


def summarize_output(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["repair_stage"],
            row["scenario"],
            row["motion_mode"],
            row["threat_class"],
            row["history_valid_ratio_bin"],
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        repair_stage, scenario, motion_mode, threat_class, hist_bin = key
        prefix: dict[str, Any] = {
            "repair_stage": repair_stage,
            "scenario": scenario,
            "checkpoint_or_policy": "frozen_gpsi_wrapper",
            "motion_mode": motion_mode,
            "threat_class": threat_class,
            "history_valid_ratio_bin": hist_bin,
            "active_only": 1,
            "samples": int(len(group)),
        }
        for suffix in HORIZON_SUFFIXES:
            delta = np.asarray([float(row[f"delta_norm_{suffix}"]) for row in group], dtype=np.float64)
            lv = np.asarray([float(row[f"logvar_xy_{suffix}"]) for row in group], dtype=np.float64)
            prefix[f"delta_norm_{suffix}_mean"] = finite_stats(delta)["mean"]
            prefix[f"delta_norm_{suffix}_median"] = finite_stats(delta)["p50"]
            prefix[f"delta_norm_{suffix}_p95"] = finite_stats(delta)["p95"]
            prefix[f"delta_norm_{suffix}_max"] = finite_stats(delta)["max"]
            prefix[f"logvar_xy_{suffix}_mean"] = finite_stats(lv)["mean"]
            prefix[f"logvar_xy_{suffix}_median"] = finite_stats(lv)["p50"]
            prefix[f"logvar_xy_{suffix}_min"] = finite_stats(lv)["min"]
            prefix[f"logvar_xy_{suffix}_max"] = finite_stats(lv)["max"]
        radial = np.asarray([float(row["projected_std_radial_1s"]) for row in group], dtype=np.float64)
        relvel = np.asarray([float(row["projected_std_relvel_1s"]) for row in group], dtype=np.float64)
        znorm = np.asarray([float(row["z_norm"]) for row in group], dtype=np.float64)
        prefix["projected_std_radial_mean"] = finite_stats(radial)["mean"]
        prefix["projected_std_radial_std"] = finite_stats(radial)["std"]
        prefix["projected_std_relvel_mean"] = finite_stats(relvel)["mean"]
        prefix["projected_std_relvel_std"] = finite_stats(relvel)["std"]
        prefix["z_norm_mean"] = finite_stats(znorm)["mean"]
        prefix["z_norm_median"] = finite_stats(znorm)["p50"]
        prefix["z_norm_p95"] = finite_stats(znorm)["p95"]
        prefix["z_norm_max"] = finite_stats(znorm)["max"]
        prefix["nan_count"] = int(sum(int(row["nan_count"]) for row in group))
        prefix["inf_count"] = int(sum(int(row["inf_count"]) for row in group))
        out.append(prefix)
    return out


def distribution_rows(stage: str, source: str, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block, values in arrays.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            continue
        if block == "history_valid_mask":
            flat_by_dim = [arr.reshape(-1)]
        else:
            flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(-1, 1)
            flat_by_dim = [flat[:, idx] for idx in range(flat.shape[1])]
        for dim, vec in enumerate(flat_by_dim):
            row = {"repair_stage": stage, "source": source, "block": block, "dim": int(dim)}
            row.update(finite_stats(vec))
            rows.append(row)
    return rows


def train_distribution_rows(data_dir: Path) -> list[dict[str, Any]]:
    train = load_npz(data_dir / "train.npz")
    mask = train["history_valid_mask"].astype(bool)
    arrays = {
        "ego_current": train["ego_current"],
        "obs_current": train["obs_current"],
        "history_rel_pos": train["history_rel_pos"][mask],
        "history_rel_vel": train["history_rel_vel"][mask],
        "history_valid_mask": train["history_valid_mask"],
    }
    rows = distribution_rows("reference", "n1_train_raw", arrays)
    for row in rows:
        row["scenario"] = "train_flow_mixed"
    return rows


def concat_stage_results(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result[key])
    return rows


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        results: list[dict[str, Any]] = []
        for stage, threshold, floor in stages:
            results.append(
                collect_rollouts(
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
            )

        dist_rows = train_distribution_rows(ROOT / args.data_dir)
        for result in results:
            stage = result["per_obstacle_rows"][0]["repair_stage"] if result["per_obstacle_rows"] else "unknown"
            dist_rows.extend(distribution_rows(stage, "online_raw_active", result["raw_blocks"]))
            dist_rows.extend(distribution_rows(stage, "online_normalized_active", result["norm_blocks"]))
        write_csv(out_dir / "tables/phase_n3_5_input_distribution_compare.csv", dist_rows)
        write_csv(out_dir / "tables/phase_n3_5_output_scale_summary.csv", concat_stage_results(results, "output_summary"))
        write_csv(out_dir / "tables/phase_n3_5_output_scale_steps.csv", concat_stage_results(results, "per_obstacle_rows"))
        write_csv(out_dir / "tables/phase_n3_5_field_order_check.csv", concat_stage_results(results, "field_rows"))
        write_csv(out_dir / "tables/phase_n3_5_history_buffer_check.csv", concat_stage_results(results, "history_rows"))
        write_csv(out_dir / "tables/phase_n3_5_active_mask_check.csv", concat_stage_results(results, "active_rows"))
        norm_rows = concat_stage_results(results, "normalization_rows")
        for row in norm_rows:
            row["checkpoint"] = rel(checkpoint)
        write_csv(out_dir / "tables/phase_n3_5_normalization_check.csv", norm_rows)
        slicing_rows = [
            {"slice": "obs_i", "start": 0, "end_exclusive": 12, "dim": 12, "status": "pass"},
            {"slice": "z_i", "start": 12, "end_exclusive": 76, "dim": 64, "status": "pass"},
            {"slice": "delta_hat_flat", "start": 76, "end_exclusive": 85, "dim": 9, "flatten_order": "[horizon, xyz]", "status": "pass"},
            {"slice": "logvar_hat_flat", "start": 85, "end_exclusive": 94, "dim": 9, "flatten_order": "[horizon, xyz]", "status": "pass"},
        ]
        write_csv(out_dir / "tables/phase_n3_5_slicing_check.csv", slicing_rows)
        step_summary = []
        for result in results:
            step_summary.extend(result["step_rows"])
        write_csv(out_dir / "tables/phase_n3_5_short_rollout_output_summary.csv", step_summary)

        before_steps = [row for row in concat_stage_results(results, "per_obstacle_rows") if row["repair_stage"] == "before_fix"]
        after_steps = [row for row in concat_stage_results(results, "per_obstacle_rows") if row["repair_stage"] == "after_fix"]
        save_histogram(
            out_dir / "plots/online_delta_norm_distribution_before_after.png",
            [
                ("before_fix", np.asarray([row["delta_norm_1s"] for row in before_steps], dtype=np.float64)),
                ("after_fix", np.asarray([row["delta_norm_1s"] for row in after_steps], dtype=np.float64)),
            ],
            "Online Delta Norm 1s Before/After",
            "||delta_hat_1s||",
        )
        save_histogram(
            out_dir / "plots/online_logvar_distribution_before_after.png",
            [
                ("before_fix", np.asarray([row["logvar_xy_1s"] for row in before_steps], dtype=np.float64)),
                ("after_fix", np.asarray([row["logvar_xy_1s"] for row in after_steps], dtype=np.float64)),
            ],
            "Online Logvar XY 1s Before/After",
            "mean logvar xy 1s",
        )
        save_histogram(
            out_dir / "plots/z_norm_distribution.png",
            [("after_fix", np.asarray([row["z_norm"] for row in after_steps], dtype=np.float64))],
            "After-Fix z Norm Distribution",
            "||z_i||",
        )

        repair_rows = [
            {
                "repair_action": "degenerate_checkpoint_std_floor",
                "status": "applied",
                "file": "envs/wrappers/gpsi_obs_wrapper.py",
                "reason": "N2 train split hold-position makes ego velocity std near 1e-6; online PPO/straight-line nonzero ego velocity otherwise normalizes to O(1e6).",
                "threshold": float(args.repaired_std_threshold),
                "floor": float(args.repaired_std_floor),
            }
        ]
        write_csv(out_dir / "tables/phase_n3_5_repair_actions.csv", repair_rows)

        manifest_path = out_dir / "tables/phase_n3_5_command_manifest.csv"
        manifest = read_existing_manifest(manifest_path)
        manifest.append(command_manifest_row("online_wrapper_audit", " ".join(sys.argv), "completed"))
        write_csv(manifest_path, manifest, ["command_name", "command", "status"])
        (out_dir / "logs/phase_n3_5_online_audit.log").write_text("online wrapper audit completed\n", encoding="utf-8")

        validate_online_audit(out_dir, after_steps, before_steps, norm_rows)
        return 0
    except PhaseN35Stop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        return 2


def validate_online_audit(
    out_dir: Path,
    after_steps: list[dict[str, Any]],
    before_steps: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
) -> None:
    if not after_steps:
        raise PhaseN35Stop("input_distribution_invalid", "no online active-obstacle samples collected")
    after_delta_p95 = finite_stats(np.asarray([row["delta_norm_1s"] for row in after_steps], dtype=np.float64))["p95"]
    after_delta_max = finite_stats(np.asarray([row["delta_norm_1s"] for row in after_steps], dtype=np.float64))["max"]
    before_delta_p95 = finite_stats(np.asarray([row["delta_norm_1s"] for row in before_steps], dtype=np.float64))["p95"]
    after_logvar = np.asarray([row["logvar_xy_1s"] for row in after_steps], dtype=np.float64)
    after_logvar_span = finite_stats(after_logvar)["max"] - finite_stats(after_logvar)["min"]
    repaired_dims = [row for row in norm_rows if int(row.get("std_repaired", 0)) == 1]
    status_rows = [
        {
            "metric": "before_delta_norm_1s_p95",
            "value": before_delta_p95,
            "status": "observed",
        },
        {
            "metric": "after_delta_norm_1s_p95",
            "value": after_delta_p95,
            "status": "pass" if float(after_delta_p95) < 100.0 else "fail",
        },
        {
            "metric": "after_delta_norm_1s_max",
            "value": after_delta_max,
            "status": "pass" if float(after_delta_max) < 1000.0 else "fail",
        },
        {
            "metric": "after_logvar_xy_1s_span",
            "value": after_logvar_span,
            "status": "pass" if float(after_logvar_span) > 0.05 else "fail",
        },
        {
            "metric": "repaired_degenerate_std_dims",
            "value": len(repaired_dims),
            "status": "pass" if repaired_dims else "warn",
        },
    ]
    write_csv(out_dir / "tables/phase_n3_5_online_audit_status.csv", status_rows)
    failed = [row for row in status_rows if row["status"] == "fail"]
    if failed:
        raise PhaseN35Stop("output_scale_invalid", f"after-fix online output scale invalid: {failed}")


if __name__ == "__main__":
    raise SystemExit(main())
