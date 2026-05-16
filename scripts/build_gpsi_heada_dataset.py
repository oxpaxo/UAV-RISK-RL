from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv


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

MOTION_MODE_ID = {
    "linear": 0,
    "sinusoidal_lateral": 1,
    "accel_decel": 2,
    "ar1_velocity": 3,
    "crossing_or_sudden_threat": 4,
}
THREAT_CLASS_ID = {"low": 0, "medium": 1, "high": 2}

ARRAY_DTYPES = {
    "ego_current": np.float32,
    "obs_current": np.float32,
    "history_pos_world": np.float32,
    "history_vel_world": np.float32,
    "history_rel_pos": np.float32,
    "history_rel_vel": np.float32,
    "history_valid_mask": np.float32,
    "delta_label_world": np.float32,
    "future_valid_mask": np.float32,
    "future_pos_world": np.float32,
    "constant_velocity_pos_world": np.float32,
    "current_pos_world": np.float32,
    "current_vel_world": np.float32,
    "motion_mode_id": np.int16,
    "threat_class_id": np.int16,
    "obstacle_id": np.int32,
    "obstacle_slot": np.int16,
    "active": np.int8,
    "episode_id": np.int32,
    "episode_seed": np.int32,
    "step": np.int32,
    "time": np.float32,
    "distance": np.float32,
    "closing": np.float32,
    "planned_cpa": np.float32,
    "planned_ttc": np.float32,
    "risk_value": np.float32,
    "history_valid_length": np.int16,
    "replacement_boundary_nearby": np.int8,
}


class PhaseN1Stop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, result_dir: Path) -> None:
        self.path = result_dir / "logs" / "phase_n1_dataset_build.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase N1 Gpsi-HeadA supervised dataset.")
    parser.add_argument("--out-dir", default="data/gpsi_head_a_v1")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n1_gpsi_dataset")
    parser.add_argument("--scenario", default="train_flow_mixed")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--future-times", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    parser.add_argument("--format", choices=["npz"], default="npz")
    parser.add_argument("--write-schema", action="store_true")
    parser.add_argument("--rollout-policy", choices=["hold_position", "straight_line"], default="hold_position")
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--spec", default="configs/gpsi_head_a_spec.yaml")
    return parser.parse_args()


def ensure_dirs(out_dir: Path, result_dir: Path) -> None:
    for path in [
        out_dir,
        out_dir / "stats",
        result_dir,
        result_dir / "tables",
        result_dir / "plots",
        result_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_terminal_flags(result_dir: Path) -> None:
    for name in TERMINAL_FLAGS:
        path = result_dir / name
        if path.exists():
            path.unlink()


def write_stop_flag(result_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["dataset_build_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail}\n")
    write_text(result_dir / "phase_n1_status.txt", f"stopped:{flag_name}\n")


def write_partial_report(result_dir: Path, terminal_decision: str, reason: str, detail: str) -> None:
    lines = [
        "# Phase N1 Gpsi-HeadA Dataset Report",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        f"Phase N1 stopped during dataset build: `{reason}`.",
        "",
        "## Detail",
        "",
        "```text",
        detail.strip(),
        "```",
    ]
    write_text(result_dir / "PHASE_N1_GPSI_DATASET_REPORT.md", "\n".join(lines) + "\n")


def check_prerequisites(args: argparse.Namespace) -> None:
    n0_flag = ROOT / "results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag"
    if not n0_flag.exists():
        raise PhaseN1Stop("phase_n0_missing", f"missing Phase N0 complete flag: {relpath(n0_flag)}")
    if args.num_episodes < args.min_episodes:
        raise PhaseN1Stop(
            "insufficient_data",
            f"num_episodes={args.num_episodes} is below minimum {args.min_episodes}",
        )
    if abs(sum(args.split) - 1.0) > 1e-6:
        raise PhaseN1Stop("schema_mismatch", f"split fractions must sum to 1.0, got {args.split}")
    spec_path = ROOT / args.spec
    if not spec_path.exists():
        raise PhaseN1Stop("schema_mismatch", f"missing spec file: {relpath(spec_path)}")
    spec_text = spec_path.read_text(encoding="utf-8").lower()
    required = [
        "type: diagonal_logvar",
        "sigma2_direct_label: false",
        "scalar_margin",
        "directional_margin",
        "trajectory_tube",
        "candidate_scoring",
        "v0: fixed_margin_shield",
        "v4: v3_plus_uncertainty_aware_candidate_scoring",
    ]
    missing = [item for item in required if item not in spec_text]
    if missing:
        raise PhaseN1Stop("schema_mismatch", f"spec missing N1 uncertainty/shield support: {missing}")


def action_from_policy(info: dict[str, Any], policy: str) -> np.ndarray:
    if policy == "hold_position":
        return np.zeros(3, dtype=np.float32)
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    vec = goal - uav
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return np.clip(vec / norm, -1.0, 1.0).astype(np.float32)


def make_record(obs: dict[str, np.ndarray], info: dict[str, Any], done: bool) -> dict[str, Any]:
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    modes = list(info.get("obstacle_motion_modes", []))
    classes = list(info.get("threat_classes", []))
    cpas = np.asarray(info.get("planned_cpa_values", []), dtype=np.float32)
    ttcs = np.asarray(info.get("planned_ttc_values", []), dtype=np.float32)
    obs_array = np.asarray(obs.get("obs", np.zeros((10, 12))), dtype=np.float32)
    mask = np.asarray(obs.get("mask", np.zeros(10)), dtype=np.float32)
    obstacles_by_id: dict[int, dict[str, Any]] = {}
    slot_to_id: dict[int, int] = {}
    for slot, pos in enumerate(positions):
        vel = velocities[slot] if slot < len(velocities) else np.zeros(3, dtype=np.float32)
        rel_pos = pos - uav
        rel_vel = vel - uav_vel
        distance = float(np.linalg.norm(rel_pos))
        closing = -float(np.dot(rel_pos / (distance + 1e-8), rel_vel))
        obstacle_id = int(ids[slot]) if slot < len(ids) else int(slot)
        row = {
            "slot": int(slot),
            "id": obstacle_id,
            "pos": np.asarray(pos, dtype=np.float32),
            "vel": np.asarray(vel, dtype=np.float32),
            "rel_pos": np.asarray(rel_pos, dtype=np.float32),
            "rel_vel": np.asarray(rel_vel, dtype=np.float32),
            "distance": np.float32(distance),
            "closing": np.float32(closing),
            "planned_cpa": np.float32(cpas[slot]) if slot < len(cpas) else np.float32(np.nan),
            "planned_ttc": np.float32(ttcs[slot]) if slot < len(ttcs) else np.float32(np.nan),
            "threat_class": str(classes[slot]) if slot < len(classes) else "none",
            "motion_mode": str(modes[slot]) if slot < len(modes) else "none",
            "risk_value": np.float32(np.clip((3.0 - distance) / 3.0, 0.0, 1.0)),
            "obs_current": np.asarray(obs_array[slot], dtype=np.float32),
            "active": int(slot < len(mask) and mask[slot] > 0.5),
        }
        obstacles_by_id[obstacle_id] = row
        slot_to_id[int(slot)] = obstacle_id
    return {
        "step": int(info["step"]),
        "time": np.float32(info["time"]),
        "done": int(done),
        "ego_current": np.asarray(obs.get("ego", np.zeros(10)), dtype=np.float32),
        "uav_pos": uav,
        "uav_vel": uav_vel,
        "obstacles_by_id": obstacles_by_id,
        "slot_to_id": slot_to_id,
    }


def collect_episode(args: argparse.Namespace, episode_id: int, episode_seed: int) -> list[dict[str, Any]]:
    env = DynamicObstacleFlowEnv(scenario=args.scenario)
    obs, info = env.reset(seed=episode_seed)
    records: list[dict[str, Any]] = []
    done = False
    while True:
        records.append(make_record(obs, info, done))
        if done or int(info["step"]) >= args.max_steps_per_episode:
            break
        action = action_from_policy(info, args.rollout_policy)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
    return records


def episode_split_map(num_episodes: int, split: list[float], seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    episode_ids = np.arange(num_episodes, dtype=np.int32)
    rng.shuffle(episode_ids)
    n_train = int(np.floor(split[0] * num_episodes))
    n_val = int(np.floor(split[1] * num_episodes))
    split_map: dict[int, str] = {}
    for eid in episode_ids[:n_train]:
        split_map[int(eid)] = "train"
    for eid in episode_ids[n_train : n_train + n_val]:
        split_map[int(eid)] = "val"
    for eid in episode_ids[n_train + n_val :]:
        split_map[int(eid)] = "test"
    return split_map


def build_episode_arrays(
    *,
    records: list[dict[str, Any]],
    episode_id: int,
    episode_seed: int,
    history_steps: int,
    future_times: list[float],
    dt: float,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[Any]] = {name: [] for name in ARRAY_DTYPES}
    max_offset = max(int(round(tau / dt)) for tau in future_times)

    for record_index, record in enumerate(records):
        step = int(record["step"])
        for obstacle_id, current in record["obstacles_by_id"].items():
            hist_pos = np.zeros((history_steps, 3), dtype=np.float32)
            hist_vel = np.zeros((history_steps, 3), dtype=np.float32)
            hist_rel_pos = np.zeros((history_steps, 3), dtype=np.float32)
            hist_rel_vel = np.zeros((history_steps, 3), dtype=np.float32)
            hist_mask = np.zeros(history_steps, dtype=np.float32)
            for h_idx in range(history_steps):
                source_index = record_index - history_steps + 1 + h_idx
                if source_index < 0 or source_index >= len(records):
                    continue
                source = records[source_index]["obstacles_by_id"].get(obstacle_id)
                if source is None:
                    continue
                hist_pos[h_idx] = source["pos"]
                hist_vel[h_idx] = source["vel"]
                hist_rel_pos[h_idx] = source["rel_pos"]
                hist_rel_vel[h_idx] = source["rel_vel"]
                hist_mask[h_idx] = 1.0

            delta = np.zeros((len(future_times), 3), dtype=np.float32)
            future_mask = np.zeros(len(future_times), dtype=np.float32)
            future_pos = np.zeros((len(future_times), 3), dtype=np.float32)
            cv_pos = np.zeros((len(future_times), 3), dtype=np.float32)
            p_now = np.asarray(current["pos"], dtype=np.float32)
            v_now = np.asarray(current["vel"], dtype=np.float32)
            for tau_idx, tau in enumerate(future_times):
                offset = int(round(tau / dt))
                target_index = record_index + offset
                cv = p_now + np.float32(tau) * v_now
                cv_pos[tau_idx] = cv
                if target_index >= len(records):
                    continue
                target = records[target_index]["obstacles_by_id"].get(obstacle_id)
                if target is None:
                    continue
                future = np.asarray(target["pos"], dtype=np.float32)
                future_pos[tau_idx] = future
                delta[tau_idx] = future - cv
                future_mask[tau_idx] = 1.0

            replacement_boundary_nearby = 0
            slot = int(current["slot"])
            for source_index in range(max(0, record_index - history_steps + 1), min(len(records), record_index + max_offset + 1)):
                candidate_id = records[source_index]["slot_to_id"].get(slot)
                if candidate_id is not None and int(candidate_id) != int(obstacle_id):
                    replacement_boundary_nearby = 1
                    break

            fields["ego_current"].append(record["ego_current"])
            fields["obs_current"].append(current["obs_current"])
            fields["history_pos_world"].append(hist_pos)
            fields["history_vel_world"].append(hist_vel)
            fields["history_rel_pos"].append(hist_rel_pos)
            fields["history_rel_vel"].append(hist_rel_vel)
            fields["history_valid_mask"].append(hist_mask)
            fields["delta_label_world"].append(delta)
            fields["future_valid_mask"].append(future_mask)
            fields["future_pos_world"].append(future_pos)
            fields["constant_velocity_pos_world"].append(cv_pos)
            fields["current_pos_world"].append(p_now)
            fields["current_vel_world"].append(v_now)
            fields["motion_mode_id"].append(MOTION_MODE_ID.get(str(current["motion_mode"]), -1))
            fields["threat_class_id"].append(THREAT_CLASS_ID.get(str(current["threat_class"]), -1))
            fields["obstacle_id"].append(int(obstacle_id))
            fields["obstacle_slot"].append(slot)
            fields["active"].append(int(current["active"]))
            fields["episode_id"].append(int(episode_id))
            fields["episode_seed"].append(int(episode_seed))
            fields["step"].append(step)
            fields["time"].append(record["time"])
            fields["distance"].append(current["distance"])
            fields["closing"].append(current["closing"])
            fields["planned_cpa"].append(current["planned_cpa"])
            fields["planned_ttc"].append(current["planned_ttc"])
            fields["risk_value"].append(current["risk_value"])
            fields["history_valid_length"].append(int(hist_mask.sum()))
            fields["replacement_boundary_nearby"].append(replacement_boundary_nearby)

    arrays: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        dtype = ARRAY_DTYPES[name]
        if not values:
            arrays[name] = np.empty((0,), dtype=dtype)
        else:
            arrays[name] = np.asarray(values, dtype=dtype)
    return arrays


def append_chunks(target: dict[str, list[np.ndarray]], arrays: dict[str, np.ndarray]) -> None:
    if arrays["episode_id"].shape[0] == 0:
        return
    for name, arr in arrays.items():
        target[name].append(arr)


def concatenate_split(chunks: dict[str, list[np.ndarray]], future_times: list[float]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, dtype in ARRAY_DTYPES.items():
        if chunks[name]:
            out[name] = np.concatenate(chunks[name], axis=0).astype(dtype, copy=False)
        else:
            out[name] = np.asarray([], dtype=dtype)
    out["future_times"] = np.asarray(future_times, dtype=np.float32)
    return out


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def schema_payload(args: argparse.Namespace, dt: float) -> dict[str, Any]:
    return {
        "version": "gpsi_head_a_v1",
        "phase": "phase_n1_dataset_construction",
        "environment": {
            "name": "DynamicObstacleFlowEnv",
            "scenario": args.scenario,
            "dt_sec": dt,
            "env_core_modified_by_phase_n1": False,
        },
        "sample_unit": ["episode_id", "step", "obstacle_id"],
        "target": {
            "type": "world_frame_residual_to_constant_velocity",
            "formula": "delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]",
            "future_times_sec": args.future_times,
            "sigma2_direct_label": False,
        },
        "uncertainty_output_planned": {
            "type": "diagonal_logvar",
            "dimensions": ["x", "y", "z"],
            "shape": ["num_horizons", 3],
            "training": "Gaussian NLL from residual prediction error",
            "used_by_future_shield": [
                "scalar_margin",
                "directional_margin",
                "trajectory_tube",
                "candidate_scoring",
            ],
        },
        "future_shield_versions_reserved": {
            "V0": "fixed-margin shield",
            "V1": "scalar sigma2-margin shield",
            "V2": "directional sigma2-margin shield",
            "V3": "predicted-trajectory directional sigma2-tube shield",
            "V4": "V3 + uncertainty-aware candidate scoring",
        },
        "history": {
            "history_steps": args.history_steps,
            "padding": "left_pad_invalid",
            "valid_mask": True,
            "identity_key": ["episode_id", "obstacle_id"],
        },
        "split": {
            "type": "episode_level",
            "train": args.split[0],
            "val": args.split[1],
            "test": args.split[2],
            "row_level_random_split": False,
        },
        "arrays": {
            "ego_current": {"shape": ["N", 10], "role": "input"},
            "obs_current": {"shape": ["N", 12], "role": "input"},
            "history_pos_world": {"shape": ["N", args.history_steps, 3], "role": "input"},
            "history_vel_world": {"shape": ["N", args.history_steps, 3], "role": "input"},
            "history_rel_pos": {"shape": ["N", args.history_steps, 3], "role": "input"},
            "history_rel_vel": {"shape": ["N", args.history_steps, 3], "role": "input"},
            "history_valid_mask": {"shape": ["N", args.history_steps], "role": "input_mask"},
            "delta_label_world": {"shape": ["N", len(args.future_times), 3], "role": "supervised_label"},
            "future_valid_mask": {"shape": ["N", len(args.future_times)], "role": "label_mask"},
            "future_pos_world": {"shape": ["N", len(args.future_times), 3], "role": "inspection_label_only_not_inference_input"},
            "constant_velocity_pos_world": {
                "shape": ["N", len(args.future_times), 3],
                "role": "inspection_label_only_not_inference_input",
            },
            "current_pos_world": {"shape": ["N", 3], "role": "input_debug_current_state"},
            "current_vel_world": {"shape": ["N", 3], "role": "input_debug_current_state"},
            "motion_mode_id": {"shape": ["N"], "role": "metadata_input"},
            "threat_class_id": {"shape": ["N"], "role": "metadata_input"},
            "obstacle_id": {"shape": ["N"], "role": "identity_metadata"},
            "obstacle_slot": {"shape": ["N"], "role": "current_slot_metadata_not_identity"},
            "episode_id": {"shape": ["N"], "role": "split_identity_metadata"},
            "episode_seed": {"shape": ["N"], "role": "split_identity_metadata"},
            "step": {"shape": ["N"], "role": "metadata"},
            "time": {"shape": ["N"], "role": "metadata"},
            "distance": {"shape": ["N"], "role": "metadata_input"},
            "closing": {"shape": ["N"], "role": "metadata_input"},
            "planned_cpa": {"shape": ["N"], "role": "metadata_input"},
            "planned_ttc": {"shape": ["N"], "role": "metadata_input"},
            "risk_value": {"shape": ["N"], "role": "metadata_input"},
            "replacement_boundary_nearby": {"shape": ["N"], "role": "inspection_metadata"},
        },
        "categorical_maps": {
            "motion_mode_id": MOTION_MODE_ID,
            "threat_class_id": THREAT_CLASS_ID,
        },
        "forbidden": {
            "sigma2_label": "not present",
            "learned_R_s_a": "not used",
            "candidate_velocity_risk_map_as_ppo_input": "not used",
            "full_5_head_gpsi": "not used",
        },
    }


def split_summary_rows(split_arrays: dict[str, dict[str, np.ndarray]], future_times: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, arrays in split_arrays.items():
        n = int(arrays["episode_id"].shape[0])
        episodes = sorted(set(int(v) for v in arrays["episode_id"].tolist())) if n else []
        row = {
            "split": split_name,
            "samples": n,
            "episodes": len(episodes),
            "first_episode_id": min(episodes) if episodes else "",
            "last_episode_id": max(episodes) if episodes else "",
            "full_history_rate": float(np.mean(arrays["history_valid_length"] == arrays["history_valid_mask"].shape[1])) if n else 0.0,
            "replacement_boundary_nearby_rate": float(np.mean(arrays["replacement_boundary_nearby"])) if n else 0.0,
        }
        for idx, tau in enumerate(future_times):
            suffix = tau_suffix(tau)
            row[f"valid_rate_{suffix}"] = float(np.mean(arrays["future_valid_mask"][:, idx])) if n else 0.0
            row[f"valid_count_{suffix}"] = int(np.sum(arrays["future_valid_mask"][:, idx])) if n else 0
        rows.append(row)
    return rows


def label_validity_rows(split_arrays: dict[str, dict[str, np.ndarray]], future_times: list[float]) -> list[dict[str, Any]]:
    rows = []
    for split_name, arrays in split_arrays.items():
        n = int(arrays["episode_id"].shape[0])
        for idx, tau in enumerate(future_times):
            valid = int(np.sum(arrays["future_valid_mask"][:, idx])) if n else 0
            rows.append(
                {
                    "split": split_name,
                    "horizon_sec": tau,
                    "samples": n,
                    "valid_labels": valid,
                    "valid_rate": float(valid / n) if n else 0.0,
                }
            )
    return rows


def motion_mode_count_rows(split_arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    inv_motion = {value: key for key, value in MOTION_MODE_ID.items()}
    rows = []
    for split_name, arrays in split_arrays.items():
        counts = Counter(int(v) for v in arrays["motion_mode_id"].tolist())
        for mode_id, count in sorted(counts.items()):
            rows.append({"split": split_name, "motion_mode_id": mode_id, "motion_mode": inv_motion.get(mode_id, "unknown"), "samples": count})
    return rows


def tau_suffix(tau: float) -> str:
    if abs(tau - round(tau)) < 1e-8:
        return f"{int(round(tau))}s"
    return str(tau).replace(".", "p") + "s"


def write_dataset_stats(
    out_dir: Path,
    result_dir: Path,
    split_arrays: dict[str, dict[str, np.ndarray]],
    future_times: list[float],
) -> None:
    split_fields = [
        "split",
        "samples",
        "episodes",
        "first_episode_id",
        "last_episode_id",
        "full_history_rate",
        "replacement_boundary_nearby_rate",
    ]
    for tau in future_times:
        suffix = tau_suffix(tau)
        split_fields.extend([f"valid_rate_{suffix}", f"valid_count_{suffix}"])
    split_rows = split_summary_rows(split_arrays, future_times)
    validity_rows = label_validity_rows(split_arrays, future_times)
    mode_rows = motion_mode_count_rows(split_arrays)
    for root in [out_dir / "stats", result_dir / "tables"]:
        write_csv(root / "phase_n1_build_split_summary.csv", split_rows, split_fields)
        write_csv(root / "phase_n1_build_label_validity_by_horizon.csv", validity_rows, ["split", "horizon_sec", "samples", "valid_labels", "valid_rate"])
        write_csv(root / "phase_n1_build_motion_mode_counts.csv", mode_rows, ["split", "motion_mode_id", "motion_mode", "samples"])


def validate_built_arrays(split_arrays: dict[str, dict[str, np.ndarray]], future_times: list[float]) -> None:
    for split_name, arrays in split_arrays.items():
        n = int(arrays["episode_id"].shape[0])
        if n <= 0:
            raise PhaseN1Stop("insufficient_data", f"{split_name} split is empty")
        if arrays["history_valid_mask"].shape[0] != n or arrays["future_valid_mask"].shape[0] != n:
            raise PhaseN1Stop("schema_mismatch", f"{split_name} mask shapes do not match samples")
        for idx, tau in enumerate(future_times):
            valid = int(np.sum(arrays["future_valid_mask"][:, idx]))
            if valid <= 0:
                raise PhaseN1Stop("label_validity_failed", f"{split_name} has no valid labels for horizon {tau}")
    split_episode_sets = {
        split_name: set(int(v) for v in arrays["episode_seed"].tolist())
        for split_name, arrays in split_arrays.items()
    }
    for a in split_episode_sets:
        for b in split_episode_sets:
            if a >= b:
                continue
            overlap = split_episode_sets[a] & split_episode_sets[b]
            if overlap:
                raise PhaseN1Stop("data_leakage_failed", f"episode_seed leakage between {a} and {b}: {sorted(overlap)[:10]}")


def manifest_payload(
    args: argparse.Namespace,
    out_dir: Path,
    result_dir: Path,
    split_map: dict[int, str],
    split_arrays: dict[str, dict[str, np.ndarray]],
    dt: float,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        path = out_dir / f"{split_name}.npz"
        files[f"{split_name}.npz"] = {
            "path": relpath(path),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "samples": int(split_arrays[split_name]["episode_id"].shape[0]),
            "episodes": len(set(int(v) for v in split_arrays[split_name]["episode_id"].tolist())),
        }
    for name in ["schema.json", "dataset_manifest.json"]:
        path = out_dir / name
        files[name] = {
            "path": relpath(path),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    episode_rows = [
        {
            "episode_id": int(episode_id),
            "episode_seed": int(args.eval_seed + episode_id),
            "split": split,
        }
        for episode_id, split in sorted(split_map.items())
    ]
    return {
        "version": "gpsi_head_a_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "scripts/build_gpsi_heada_dataset.py",
        "args": vars(args),
        "dt_sec": dt,
        "phase_n0_complete_flag": relpath(ROOT / "results/env_v2_phase_n0_design_freeze/PHASE_N0_DESIGN_FREEZE_COMPLETE.flag"),
        "env_core": {
            "file": "envs/dynamic_obstacle_flow_env.py",
            "sha256": sha256_file(ROOT / "envs/dynamic_obstacle_flow_env.py"),
            "phase_n1_action": "read_only",
        },
        "spec": {
            "file": args.spec,
            "sha256": sha256_file(ROOT / args.spec),
            "diagonal_logvar_supported": True,
            "sigma2_direct_label": False,
        },
        "files": files,
        "episode_split": episode_rows,
        "terminal_flag_written_by": "inspect_gpsi_heada_dataset.py",
        "status": "built_pending_inspect",
        "result_dir": relpath(result_dir),
    }


def run() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    result_dir = ROOT / args.result_dir
    ensure_dirs(out_dir, result_dir)
    clear_terminal_flags(result_dir)
    write_text(result_dir / "phase_n1_status.txt", "building\n")
    logger = Logger(result_dir)
    logger.log("Phase N1 dataset builder started")
    logger.log("Command: " + " ".join(["python", *sys.argv]))
    check_prerequisites(args)

    probe_env = DynamicObstacleFlowEnv(scenario=args.scenario)
    _obs, info = probe_env.reset(seed=args.eval_seed)
    dt = float(info.get("dt", 0.2))
    split_map = episode_split_map(args.num_episodes, args.split, args.split_seed)
    episode_split_rows = [
        {"episode_id": eid, "episode_seed": args.eval_seed + eid, "split": split_map[eid]}
        for eid in sorted(split_map)
    ]
    write_csv(result_dir / "tables" / "phase_n1_episode_split_plan.csv", episode_split_rows, ["episode_id", "episode_seed", "split"])
    write_csv(out_dir / "stats" / "phase_n1_episode_split_plan.csv", episode_split_rows, ["episode_id", "episode_seed", "split"])

    split_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: {field: [] for field in ARRAY_DTYPES}
        for name in ["train", "val", "test"]
    }

    for episode_id in range(args.num_episodes):
        episode_seed = int(args.eval_seed + episode_id)
        records = collect_episode(args, episode_id, episode_seed)
        if not records:
            raise PhaseN1Stop("dataset_build_failed", f"episode {episode_id} produced no records")
        arrays = build_episode_arrays(
            records=records,
            episode_id=episode_id,
            episode_seed=episode_seed,
            history_steps=args.history_steps,
            future_times=args.future_times,
            dt=dt,
        )
        append_chunks(split_chunks[split_map[episode_id]], arrays)
        if (episode_id + 1) % 10 == 0 or episode_id + 1 == args.num_episodes:
            logger.log(f"Collected episode {episode_id + 1}/{args.num_episodes}")

    split_arrays = {
        split_name: concatenate_split(chunks, args.future_times)
        for split_name, chunks in split_chunks.items()
    }
    validate_built_arrays(split_arrays, args.future_times)

    if args.write_schema:
        write_json(out_dir / "schema.json", schema_payload(args, dt))
    else:
        write_json(out_dir / "schema.json", schema_payload(args, dt))

    for split_name, arrays in split_arrays.items():
        save_npz(out_dir / f"{split_name}.npz", arrays)
        logger.log(f"Wrote {split_name}.npz with {arrays['episode_id'].shape[0]} samples")

    write_dataset_stats(out_dir, result_dir, split_arrays, args.future_times)
    manifest = manifest_payload(args, out_dir, result_dir, split_map, split_arrays, dt)
    write_json(out_dir / "dataset_manifest.json", manifest)
    write_text(result_dir / "phase_n1_status.txt", "built_pending_inspect\n")
    logger.log("Phase N1 dataset build completed; inspect still required")


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    ensure_dirs(ROOT / args.out_dir, result_dir)
    logger = Logger(result_dir)
    try:
        run()
    except PhaseN1Stop as exc:
        write_stop_flag(result_dir, exc.reason, exc.detail)
        write_partial_report(result_dir, f"phase_n1_stopped_{exc.reason}", exc.reason, exc.detail)
        logger.log(f"Phase N1 stopped: {exc.reason}: {exc.detail}")
        raise SystemExit(2) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_stop_flag(result_dir, "dataset_build_failed", detail)
        write_partial_report(result_dir, "phase_n1_stopped_dataset_build_failed", "dataset_build_failed", detail)
        logger.log("Unexpected dataset build exception:\n" + detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
